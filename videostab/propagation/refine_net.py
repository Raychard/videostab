"""传播残差细化网络 (DUT 点云式关键点注意力).

DUT 把关键点与网格顶点视作两个点云, 对每个顶点在其邻域关键点上做注意力
聚合: 输入 [距离向量, 关键点残差运动] -> 双编码器 -> 1D 卷积 -> 注意力
权重 -> 加权聚合 -> 解码残差运动; 最终 n = n̂(多单应初始化) + Δn.

与早期实现的关键差别: 早期把关键点 splat 到网格再用 2D 卷积 + ECA
**通道**注意力. ECA 在原理上无法在关键点之间做选择, 而"选择信任哪些
关键点"正是 DUT 对抗离群点/误跟踪的核心鲁棒机制; splat 还会在进网络前
就把同格关键点平均掉, 并丢失距离向量信息.
"""
import numpy as np
import torch
import torch.nn as nn

MOTION_NORM = 16.0   # 运动归一化尺度(px), 训练/推理必须一致
DIST_NORM = 100.0    # 距离向量归一化尺度(px)
K_NEIGHBORS = 32     # 每个顶点聚合的最近关键点数


def _mlp1d(cin, cout, hidden):
    return nn.Sequential(nn.Conv1d(cin, hidden, 1), nn.GELU(),
                         nn.Conv1d(hidden, cout, 1))


class ResidualRefineNet(nn.Module):
    def __init__(self, hidden: int = 32, k_neighbors: int = K_NEIGHBORS):
        super().__init__()
        self.k = k_neighbors
        self.dist_enc = _mlp1d(2, hidden, hidden)    # 距离向量编码器
        # 运动编码器输入 3 通道: 残差运动(2) + 追踪置信度(1).
        # 让网络能显式看到"这个观测有多可靠", 而非对所有关键点一视同仁.
        self.mot_enc = _mlp1d(3, hidden, hidden)
        self.fuse = nn.Sequential(
            nn.Conv1d(2 * hidden, hidden, 1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 1), nn.GELU())
        self.attn = nn.Conv1d(hidden, 1, 1)          # 关键点注意力
        self.dec = nn.Sequential(nn.Conv1d(hidden, hidden, 1), nn.GELU(),
                                 nn.Conv1d(hidden, 2, 1))
        nn.init.zeros_(self.dec[-1].weight)  # 零初始化 => 退化到多单应初始化
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, verts, kp, kp_resid, mask, conf=None):
        """verts (B,V,2) 顶点坐标; kp (B,N,2); kp_resid (B,N,2) 关键点残差
        运动(观测-初始化); mask (B,N); conf (B,N) 追踪置信度(缺省=1).
        返回 (B,V,2) 顶点残差(归一化)."""
        B, V, _ = verts.shape
        N = kp.shape[1]
        if conf is None:
            conf = torch.ones(B, N, device=kp.device, dtype=kp.dtype)
        k = min(self.k, N)
        # 邻域选择: 每个顶点取最近 k 个有效关键点
        d2 = torch.cdist(verts, kp).pow(2)                     # (B,V,N)
        d2 = d2.masked_fill(~mask.unsqueeze(1), float("inf"))
        idx = d2.topk(k, dim=-1, largest=False).indices        # (B,V,k)
        bi = torch.arange(B, device=verts.device).view(B, 1, 1)
        nb_kp = kp[bi, idx]                                    # (B,V,k,2)
        nb_mot = kp_resid[bi, idx]
        nb_ok = mask[bi, idx]                                  # (B,V,k)
        nb_conf = conf[bi, idx].unsqueeze(-1)                  # (B,V,k,1)

        dist = (verts.unsqueeze(2) - nb_kp) / DIST_NORM
        feat_d = self.dist_enc(dist.reshape(B * V, k, 2).transpose(1, 2))
        mot_in = torch.cat([nb_mot / MOTION_NORM, nb_conf], dim=-1)
        feat_m = self.mot_enc(mot_in.reshape(B * V, k, 3).transpose(1, 2))
        f = self.fuse(torch.cat([feat_d, feat_m], dim=1))      # (B*V,H,k)

        logit = self.attn(f)                                   # (B*V,1,k)
        logit = logit.masked_fill(
            ~nb_ok.reshape(B * V, 1, k), float("-inf"))
        # 全部无效的顶点: 退化为均匀权重, 后续残差仍近似 0
        allbad = (~nb_ok).all(dim=-1).reshape(B * V, 1, 1)
        logit = torch.where(allbad, torch.zeros_like(logit), logit)
        w = torch.softmax(logit, dim=-1)

        attended = (f * w).sum(dim=-1, keepdim=True)           # (B*V,H,1)
        out = self.dec(attended).squeeze(-1).reshape(B, V, 2)
        return out


def grid_vertex_tensor(shape_hw, grid_size, device):
    """(1,V,2) 网格顶点坐标, 与 propagation.grid_vertices 一致."""
    h, w = float(shape_hw[0]), float(shape_hw[1])
    gh, gw = grid_size
    ys = torch.linspace(0, h - 1, gh, device=device)
    xs = torch.linspace(0, w - 1, gw, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(1, gh * gw, 2)


def grid_vertex_batch(shape_hw: torch.Tensor, grid_size) -> torch.Tensor:
    """(B,V,2) 逐样本网格顶点坐标, 支持混合分辨率 batch.
    shape_hw: (B,2) 张量 [h,w]."""
    gh, gw = grid_size
    dev = shape_hw.device
    h = shape_hw[:, 0].float().view(-1, 1, 1) - 1
    w = shape_hw[:, 1].float().view(-1, 1, 1) - 1
    uy = torch.linspace(0, 1, gh, device=dev).view(1, gh, 1)
    ux = torch.linspace(0, 1, gw, device=dev).view(1, 1, gw)
    gy = (uy * h).expand(-1, gh, gw)
    gx = (ux * w).expand(-1, gh, gw)
    return torch.stack([gx, gy], dim=-1).reshape(shape_hw.shape[0], gh * gw, 2)


@torch.no_grad()
def refine_grid(model: ResidualRefineNet, grid_init, pts, motions, kp_init,
                shape_hw, device: str = "cpu", conf=None):
    """推理: 多单应初始化 + 网络残差 -> 细化网格运动场 (GH,GW,2)."""
    gh, gw = grid_init.shape[:2]
    if len(pts) == 0:
        return grid_init
    verts = grid_vertex_tensor(shape_hw, (gh, gw), device)
    kp = torch.from_numpy(np.ascontiguousarray(pts)).float()[None].to(device)
    resid = torch.from_numpy(
        np.ascontiguousarray(motions - kp_init)).float()[None].to(device)
    mask = torch.ones(1, kp.shape[1], dtype=torch.bool, device=device)
    c = (None if conf is None else
         torch.from_numpy(np.ascontiguousarray(conf)).float()[None].to(device))
    out = model(verts, kp, resid, mask, c)[0].cpu().numpy() * MOTION_NORM
    return grid_init + out.reshape(gh, gw, 2)

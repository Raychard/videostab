"""传播残差细化网络 (NLNL EfficientMotionPro 的精简复刻).

输入 5 通道网格特征: 多单应初始化运动(2) + 关键点残差 splat(2) + splat 权重(1);
输出逐顶点残差运动(2). 参数量 ~20K, 深度可分离卷积 + ECA 通道注意力.
"""
import numpy as np
import torch
import torch.nn as nn

MOTION_NORM = 16.0  # 运动归一化尺度(px), 训练/推理必须一致


class ECA(nn.Module):
    """Efficient Channel Attention: GAP + 1D conv 门控."""

    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        w = x.mean((-2, -1))                       # (B,C)
        w = self.conv(w.unsqueeze(1)).squeeze(1)   # (B,C)
        return x * torch.sigmoid(w)[..., None, None]


class _Block(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)
        self.eca = ECA(ch)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.eca(self.act(self.pw(self.dw(x))))


class ResidualRefineNet(nn.Module):
    def __init__(self, in_ch: int = 5, hidden: int = 48, blocks: int = 2):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.body = nn.Sequential(*[_Block(hidden) for _ in range(blocks)])
        self.head = nn.Conv2d(hidden, 2, 3, padding=1)
        nn.init.zeros_(self.head.weight)  # 初始为零残差, 训练前不破坏初始化场
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """x: (B,5,GH,GW) 归一化特征 -> (B,2,GH,GW) 归一化残差."""
        return self.head(self.body(nn.functional.gelu(self.stem(x))))


def _splat_to_grid(pts, values, shape_hw, grid_size):
    """双线性 splat 稀疏值到顶点网格. 返回 (GH,GW,C), (GH,GW) 权重."""
    gh, gw = grid_size
    h, w = shape_hw
    acc = np.zeros((gh, gw, values.shape[1]), np.float32)
    wacc = np.zeros((gh, gw), np.float32)
    if len(pts) == 0:
        return acc, wacc
    u = pts[:, 0] / (w - 1) * (gw - 1)
    v = pts[:, 1] / (h - 1) * (gh - 1)
    i0 = np.clip(np.floor(u).astype(int), 0, gw - 2)
    j0 = np.clip(np.floor(v).astype(int), 0, gh - 2)
    fu, fv = u - i0, v - j0
    for dj, di, wgt in ((0, 0, (1 - fu) * (1 - fv)), (0, 1, fu * (1 - fv)),
                        (1, 0, (1 - fu) * fv), (1, 1, fu * fv)):
        np.add.at(acc, (j0 + dj, i0 + di), wgt[:, None] * values)
        np.add.at(wacc, (j0 + dj, i0 + di), wgt)
    ok = wacc > 0
    acc[ok] /= wacc[ok, None]
    return acc, wacc


def build_refine_input(grid_init, pts, motions, kp_init, shape_hw):
    """组装网络输入 (5,GH,GW) float32 (已归一化)."""
    gh, gw = grid_init.shape[:2]
    resid, wgt = _splat_to_grid(pts, (motions - kp_init).astype(np.float32),
                                shape_hw, (gh, gw))
    feat = np.concatenate([
        grid_init / MOTION_NORM,
        resid / MOTION_NORM,
        np.minimum(wgt, 4.0)[..., None] / 4.0,
    ], axis=-1).astype(np.float32)
    return feat.transpose(2, 0, 1)  # (5,GH,GW)


@torch.no_grad()
def refine_grid(model: ResidualRefineNet, grid_init, pts, motions, kp_init,
                shape_hw, device: str = "cpu"):
    """推理: 初始化场 + 网络残差 -> 细化网格运动场 (GH,GW,2)."""
    x = torch.from_numpy(
        build_refine_input(grid_init, pts, motions, kp_init, shape_hw)
    )[None].to(device)
    res = model(x)[0].cpu().numpy().transpose(1, 2, 0) * MOTION_NORM
    return grid_init + res

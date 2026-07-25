"""轨迹平滑网络: DUT 的 Jacobi 数据锚求解器 + 预算感知 λ.

DUT 更新式 (每次迭代都锚定原始轨迹 T, 网络预测核权重 w 与平衡系数 λ):
    T̂ᵗ = [ T + λ·Σⱼ wⱼ·T̂ᵗ⁻¹ ] / [ 1 + λ·Σⱼ wⱼ ]

为何必须有数据锚 (实测): 纯卷积迭代 P←K*P 没有不动点, 平滑量随迭代
无界增长 (rough 0.043→0.0003), "核宽×迭代次数"高度简并, 网络学不到
稳定策略; Jacobi 形式收敛到唯一不动点 (~15 次迭代后稳定, 与 DUT 报告
的 15 次吻合), 且 λ 是单调良态的平滑/保真旋钮 (λ 0.1→100 对应
rough 0.208→0.0006).

超越 DUT 的扩展 —— 预算感知 λ:
DUT 没有裁剪预算概念 (其 cropping ratio 仅 0.704, 丢掉近 30% 画面).
本实现把"参考平滑相对预算的余量"作为网络输入, 使 λ 能在余量充足处
放开平滑、逼近边界时自动收敛, 把裁剪预算从事后钳位变成优化的一部分.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import SmoothingConfig

VEL_NORM = 8.0        # 速度归一化尺度(px/帧)
JACOBI_ITERS = 15     # DUT 报告的收敛迭代数
TAP_RADIUS = 21       # 时间抽头触达上限(配合膨胀抽头, 见 DILATED_TAPS)


# 膨胀(对数间隔)时间抽头: 以少量抽头换长触达.
# DUT 用连续抽头 (±6), 15 次迭代后等效 σ 仅约 13 帧, 够不到强平滑区;
# 这里用 Fibonacci 式间隔, 14 个抽头即可触达 ±21, 等效 σ 约 39 帧,
# 使网络能覆盖经典高斯 (σ 可到 12x√3≈21 帧) 的整个工作范围.
DILATED_TAPS = (1, 2, 3, 5, 8, 13, 21)


def _tap_offsets(r: int):
    """r 作为触达上限, 返回对称的膨胀抽头偏移(排除中心)."""
    taps = [t for t in DILATED_TAPS if t <= max(r, 1)] or [1]
    return [-t for t in reversed(taps)] + list(taps)


def gaussian_base_torch(C: torch.Tensor, cfg: SmoothingConfig) -> torch.Tensor:
    """经典逐顶点自适应高斯平滑的可微实现(用作预算余量参考 + 推理兜底)."""
    B, _, T, GH, GW = C.shape
    r = min(cfg.radius, T - 1)
    offsets = torch.arange(-r, r + 1, device=C.device)
    raw = torch.arange(T, device=C.device)[:, None] + offsets[None, :]
    idx = raw.abs()
    over = idx > T - 1
    idx = torch.where(over, (2 * (T - 1) - idx).clamp(min=0), idx).clamp(0, T - 1)
    lm, rm = raw < 0, raw > T - 1
    v = torch.diff(C, dim=2)
    speed = v.norm(dim=1)
    speed = torch.cat([speed[:, :1], speed], dim=1)
    sigma = (cfg.base_sigma * torch.exp(-speed / cfg.adapt_v0)).clamp(min=1.0)
    w = torch.exp(-offsets.view(1, -1, 1, 1, 1).float() ** 2
                  / (2 * sigma.unsqueeze(1) ** 2))
    w = w / w.sum(dim=1, keepdim=True)
    K = 2 * r + 1
    P = C
    for _ in range(cfg.iterations):
        vals = P[:, :, idx.reshape(-1)].reshape(B, 2, T, K, GH, GW)
        vals = vals.permute(0, 1, 3, 2, 4, 5)
        lmask = lm.T.view(1, 1, K, T, 1, 1)
        rmask = rm.T.view(1, 1, K, T, 1, 1)
        vals = torch.where(lmask, 2 * P[:, :, :1].unsqueeze(2) - vals, vals)
        vals = torch.where(rmask, 2 * P[:, :, -1:].unsqueeze(2) - vals, vals)
        P = (w.unsqueeze(1) * vals).sum(dim=2)
    return P


def jacobi_smooth(C: torch.Tensor, w: torch.Tensor, lam: torch.Tensor,
                  offsets, iters: int = JACOBI_ITERS) -> torch.Tensor:
    """DUT Jacobi 数据锚迭代. C (B,2,T,GH,GW); w (B,K,T,GH,GW)>=0;
    lam (B,1,T,GH,GW)>0. 返回平滑路径 P, 同形状."""
    r = max(abs(o) for o in offsets)
    T = C.shape[2]
    wsum = w.sum(dim=1, keepdim=True)                    # (B,1,T,GH,GW)
    denom = 1.0 + lam * wsum
    P = C
    for _ in range(iters):
        pad = F.pad(P, (0, 0, 0, 0, r, r), mode="replicate")
        taps = torch.stack([pad[:, :, r + o: r + o + T] for o in offsets],
                           dim=1)                        # (B,K,2,T,GH,GW)
        agg = (w.unsqueeze(2) * taps).sum(dim=1)         # (B,2,T,GH,GW)
        P = (C + lam * agg) / denom
    return P


class DynamicKernelNet(nn.Module):
    """预测 Jacobi 求解器的核权重 w 与平衡系数 λ.

    输入 4 通道: 归一化速度场(2) + 参考平滑相对裁剪预算的余量(2).
    后者让 λ 具备预算感知 —— 这是 DUT 所没有的.
    """

    def __init__(self, radius: int = 30, hidden: int = 16,
                 tap_radius: int = TAP_RADIUS):
        super().__init__()
        self.radius = radius                  # 参考高斯基底的窗口半径
        self.tap_radius = tap_radius
        self.offsets = _tap_offsets(tap_radius)
        K = len(self.offsets)
        self.trunk = nn.Sequential(
            nn.Conv3d(4, hidden, (5, 3, 3), padding=(2, 1, 1)), nn.GELU(),
            nn.Conv3d(hidden, hidden, (5, 3, 3), padding=(2, 1, 1)), nn.GELU())
        self.w_head = nn.Conv3d(hidden, K, 1)
        self.lam_head = nn.Conv3d(hidden, 1, 1)
        # 零初始化 => w 全 1(经 softplus 后为常数), λ 取先验值 => 各向同性
        # 均匀 Jacobi 平滑, 是一个良态且已经有效的起点.
        for h in (self.w_head, self.lam_head):
            nn.init.zeros_(h.weight)
        nn.init.constant_(self.w_head.bias, 0.5413)   # softplus(0.5413)≈1.0
        nn.init.constant_(self.lam_head.bias, 1.2564)  # softplus(1.2564)≈1.5

    def forward(self, vel, headroom):
        """vel/headroom: (B,2,T,GH,GW) -> (w, lam)."""
        f = self.trunk(torch.cat([vel, headroom], dim=1))
        w = F.softplus(self.w_head(f))
        lam = F.softplus(self.lam_head(f))
        return w, lam


def _headroom(C, cfg: SmoothingConfig):
    """参考高斯平滑的校正量相对裁剪预算的占比(带符号), 供 λ 感知余量."""
    base = gaussian_base_torch(C, cfg)
    h, w = cfg.proxy_hw
    lim = C.new_tensor([cfg.crop_ratio / 2 * w, cfg.crop_ratio / 2 * h])
    return ((base - C) / lim.view(1, 2, 1, 1, 1)).clamp(-3, 3)


def smooth_path_torch(model: DynamicKernelNet, C: torch.Tensor,
                      iterations: int = JACOBI_ITERS,
                      cfg: SmoothingConfig = None) -> torch.Tensor:
    """可微版本(训练用). C: (B,2,T,GH,GW) -> P 同形状."""
    cfg = cfg or SmoothingConfig()
    vel = torch.diff(C, dim=2, prepend=C[:, :, :1]) / VEL_NORM
    w, lam = model(vel, _headroom(C, cfg))
    return jacobi_smooth(C, w, lam, model.offsets, iterations)


@torch.no_grad()
def smooth_path_nn(model: DynamicKernelNet, C: np.ndarray,
                   iterations: int = JACOBI_ITERS, device: str = "cpu",
                   cfg: SmoothingConfig = None) -> np.ndarray:
    """推理入口. C (T,GH,GW,2) numpy -> P 同形状."""
    t = torch.from_numpy(C.transpose(3, 0, 1, 2))[None].to(device)
    P = smooth_path_torch(model.to(device), t, iterations, cfg)
    return P[0].cpu().numpy().transpose(1, 2, 3, 0).astype(np.float32)

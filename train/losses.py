"""无监督训练损失 (DUT 配方 + NLNL 自适应/频域强化). 全部可微."""
import torch
import torch.nn.functional as F


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def sample_field(field: torch.Tensor, pts: torch.Tensor,
                 shape_hw) -> torch.Tensor:
    """在关键点处双线性采样网格场. field (B,2,GH,GW), pts (B,N,2) 像素坐标.

    shape_hw: (h,w) 元组, 或逐样本 (B,2) 张量[h,w] —— 混合宽高比 batch
    必须逐样本归一化, 否则采样坐标错位.
    """
    if torch.is_tensor(shape_hw) and shape_hw.dim() == 2:
        h = shape_hw[:, 0:1].to(pts)          # (B,1)
        w = shape_hw[:, 1:2].to(pts)
    else:
        h, w = float(shape_hw[0]), float(shape_hw[1])
    gx = pts[..., 0] / (w - 1) * 2 - 1
    gy = pts[..., 1] / (h - 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(1)      # (B,1,N,2)
    out = F.grid_sample(field, grid, align_corners=True)   # (B,2,1,N)
    return out[:, :, 0].permute(0, 2, 1)                   # (B,N,2)


def grid_laplacian(field: torch.Tensor) -> torch.Tensor:
    """网格场空间拉普拉斯幅值(保形/空间一致性项). field (B,2,GH,GW)."""
    k = field.new_tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
    k = k.view(1, 1, 3, 3).repeat(field.shape[1], 1, 1, 1)
    lap = F.conv2d(field, k, padding=1, groups=field.shape[1])
    return charbonnier(lap).mean()


def propagation_loss(pred_grid, pts, motions, mask, shape_hw,
                     lam_shape: float = 0.5):
    """关键点一致性 + 保形. pred_grid (B,2,GH,GW); pts/motions (B,N,2)."""
    sampled = sample_field(pred_grid, pts, shape_hw)
    m = mask.unsqueeze(-1).float()
    data = (charbonnier(sampled - motions) * m).sum() / m.sum().clamp(min=1)
    return data + lam_shape * grid_laplacian(pred_grid), data


def smoother_loss(P, C, shape_hw, crop_ratio: float = 0.12,
                  adapt_v0: float = 6.0, freq_cut_div: int = 6,
                  lam=(1.0, 0.5, 0.05, 0.2)):
    """平滑网络总损失. P/C: (B,2,T,GH,GW).

    lam: (自适应二阶, 频域, 预算距离, 空间一致性) 权重.
    """
    lam_t, lam_f, lam_d, lam_s = lam
    # 1) 运动自适应二阶惩罚: 快速运镜段(速度大)降低平滑压力
    vel = torch.diff(C, dim=2)
    speed = vel.norm(dim=1, keepdim=True).mean((3, 4), keepdim=True)
    wgt = torch.exp(-speed / adapt_v0)                     # (B,1,T-1,1,1)
    acc = P[:, :, 2:] - 2 * P[:, :, 1:-1] + P[:, :, :-2]
    l_t = (wgt[:, :, 1:] * acc.pow(2)).mean()
    # 2) 频域高频抑制 (NLNL): DC 与低频段之外的能量
    T = P.shape[2]
    spec = torch.fft.rfft(P - P.mean(dim=2, keepdim=True), dim=2)
    cut = max(2, T // freq_cut_div)
    l_f = spec[:, :, cut:].abs().pow(2).mean() / T
    # 3) 预算距离: 偏离超出裁剪预算的部分重罚(软化的硬约束)
    h, w = shape_hw
    lim = P.new_tensor([crop_ratio / 2 * w, crop_ratio / 2 * h])
    excess = F.relu((P - C).abs() - lim.view(1, 2, 1, 1, 1))
    l_d = (P - C).pow(2).mean() * 0.01 + excess.pow(2).mean()
    # 4) 校正场空间一致性: 相邻顶点校正相近, 防网格扭曲
    B_field = (P - C).permute(0, 2, 1, 3, 4).flatten(0, 1)  # (B*T,2,GH,GW)
    l_s = grid_laplacian(B_field)
    total = lam_t * l_t + lam_f * l_f + lam_d * l_d + lam_s * l_s
    return total, {"temporal": l_t.item(), "freq": l_f.item(),
                   "dist": l_d.item(), "spatial": l_s.item()}

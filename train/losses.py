"""无监督训练损失 (对齐 DUT 公式 + 本项目的预算感知扩展). 全部可微.

对齐 DUT 的关键点:
- 保形用 R90 矩形保持 (DUT L_sp), 而非运动场拉普拉斯. 后者对纯剪切
  (真畸变) 与全局平移 (零畸变) 几乎给出相同惩罚, 且零填充会把无畸变的
  平移在边界重罚 (实测常数场 0.876 vs 修正后 0.001, 影响 27% 顶点).
- 传播损失含两个数据项: 顶点-邻域关键点运动一致 (L_vm) 与关键点投影
  一致 (L_kp), 权重按 DUT: λ_m=10, λ_v=40, λ_s=40.
- 平滑损失恢复数据保真项 ||P-C||² (DUT L_ts 的组成). 缺了它, freq+temporal
  的全局最优是一条完全平坦的路径 (退化解).
"""
import torch
import torch.nn.functional as F


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def sample_field(field: torch.Tensor, pts: torch.Tensor,
                 shape_hw) -> torch.Tensor:
    """在关键点处双线性采样网格场. field (B,2,GH,GW), pts (B,N,2) 像素坐标.

    与渲染器一致: warp_frame 也是对网格场做双线性上采样.
    shape_hw: (h,w) 或逐样本 (B,2) 张量 (混合宽高比 batch 必须逐样本).
    """
    if torch.is_tensor(shape_hw) and shape_hw.dim() == 2:
        h = shape_hw[:, 0:1].to(pts)
        w = shape_hw[:, 1:2].to(pts)
    else:
        h, w = float(shape_hw[0]), float(shape_hw[1])
    gx = pts[..., 0] / (w - 1) * 2 - 1
    gy = pts[..., 1] / (h - 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(1)      # (B,1,N,2)
    out = F.grid_sample(field, grid, align_corners=True)   # (B,2,1,N)
    return out[:, :, 0].permute(0, 2, 1)                   # (B,N,2)


def _cell_size(shape_hw, GH: int, GW: int, device):
    """网格单元像素尺寸 (ch, cw), 支持逐样本 shape."""
    if torch.is_tensor(shape_hw) and shape_hw.dim() == 2:
        h = shape_hw[:, 0].float().to(device).view(-1, 1, 1)
        w = shape_hw[:, 1].float().to(device).view(-1, 1, 1)
    else:
        h = torch.tensor(float(shape_hw[0]), device=device).view(1, 1, 1)
        w = torch.tensor(float(shape_hw[1]), device=device).view(1, 1, 1)
    return (h - 1) / (GH - 1), (w - 1) / (GW - 1)


def shape_preservation(motion: torch.Tensor, shape_hw) -> torch.Tensor:
    """DUT L_sp: R90 矩形保持. motion (B,2,GH,GW) 像素运动场.

    形变后顶点 v̂³ 应满足 v̂³ = v̂² + R₉₀(v̂¹ - v̂²). 在"单位格"坐标下
    (顶点 (i,j) 位于 (j,i), 运动按格尺寸归一) 未形变网格精确满足该式,
    故残差直接度量网格单元的非矩形程度 = 可见畸变.
    """
    B, _, GH, GW = motion.shape
    ch, cw = _cell_size(shape_hw, GH, GW, motion.device)
    ii = torch.arange(GH, device=motion.device).view(1, GH, 1)
    jj = torch.arange(GW, device=motion.device).view(1, 1, GW)
    px = jj + motion[:, 0] / cw          # (B,GH,GW) 单位格坐标
    py = ii + motion[:, 1] / ch

    def vert(rs, cs):
        return torch.stack([px[:, rs, cs], py[:, rs, cs]], dim=-1)

    v1 = vert(slice(None, -1), slice(None, -1))   # (i,   j)
    v2 = vert(slice(1, None), slice(None, -1))    # (i+1, j)
    v3 = vert(slice(1, None), slice(1, None))     # (i+1, j+1)
    d = v1 - v2
    r90 = torch.stack([-d[..., 1], d[..., 0]], dim=-1)
    return charbonnier(v3 - (v2 + r90)).mean()


def propagation_loss(pred_grid, pts, motions, mask, shape_hw,
                     kp_field=None, lam=(10.0, 40.0, 40.0)):
    """DUT L_MR = λ_m·L_vm + λ_v·L_kp + λ_s·L_sp.

    pred_grid (B,2,GH,GW); pts/motions (B,N,2); mask (B,N).
    kp_field: 预先算好的 pred_grid 在关键点处的采样(可选, 省一次采样).
    返回 (total, data_err) —— data_err 用于日志(关键点残差, px).
    """
    lam_m, lam_v, lam_s = lam
    m = mask.unsqueeze(-1).float()
    denom = m.sum().clamp(min=1)
    sampled = kp_field if kp_field is not None else sample_field(
        pred_grid, pts, shape_hw)

    # L_kp: 关键点投影一致(场在关键点处应等于观测运动), L2
    l_kp = (charbonnier(sampled - motions).pow(2) * m).sum() / denom
    # L_vm: 顶点-邻域关键点运动一致, L1 (对离群关键点更鲁棒)
    l_vm = (charbonnier(sampled - motions) * m).sum() / denom
    # L_sp: R90 保形
    l_sp = shape_preservation(pred_grid, shape_hw)

    total = lam_m * l_vm + lam_v * l_kp + lam_s * l_sp
    data_err = (charbonnier(sampled - motions) * m).sum() / denom
    return total, data_err


def smoother_loss(P, C, shape_hw, crop_ratio: float = 0.12,
                  adapt_v0: float = 6.0, freq_cut_div: int = 16,
                  lam=(0.01, 100.0, 0.0, 50.0, 1.0)):
    """平滑损失. P/C: (B,2,T,GH,GW).

    lam: (数据保真, 自适应二阶, 频域, 预算越界, 保形).

    权重经"损失-渲染对齐"扫描标定 (scratchpad/align.py), 而非拍脑袋:
    - 数据保真项量级可达 45, 权重必须很小, 否则完全主导损失并把网络推向
      "不平滑"的退化解 (前三次训练崩溃的真凶). 这里它只作为让 λ 保持有限
      的弱正则 —— 真正的锚是裁剪预算项 (DUT 无预算概念故必须重用数据项).
    - 二阶项是唯一与渲染稳定度单调同向的项 (它本身就是路径二阶差分),
      且对线性漂移的响应恰为 0 —— 正是"保留匀速运镜、只压抖动"所需,
      故给最大权重.
    - 频域项权重置 0 (保留代码供实验): 实测它测的不是抖动. 完美平滑的
      线性漂移因端点不连续产生频谱泄漏, 高频能量达 64.6, 而真实抖动
      噪声仅 1.29 —— 对合法运镜的惩罚是对抖动的 50 倍, 方向完全反了.
      它还曾占总损失 92% 却几乎不随 λ 变化, 严重恶化梯度条件数.
    """
    lam_d, lam_t, lam_f, lam_b, lam_s = lam
    B_field = P - C

    # 1) 数据保真 (DUT L_ts 的数据项): 平滑路径不应远离原始路径
    l_data = charbonnier(B_field).pow(2).mean()
    # 2) 运动自适应二阶惩罚: 快速运镜段降低平滑压力
    vel = torch.diff(C, dim=2)
    speed = vel.norm(dim=1, keepdim=True).mean((3, 4), keepdim=True)
    wgt = torch.exp(-speed / adapt_v0)                     # (B,1,T-1,1,1)
    acc = P[:, :, 2:] - 2 * P[:, :, 1:-1] + P[:, :, :-2]
    l_t = (wgt[:, :, 1:] * acc.pow(2)).mean()
    # 3) 频域高频抑制
    T = P.shape[2]
    spec = torch.fft.rfft(P - P.mean(dim=2, keepdim=True), dim=2)
    cut = max(2, T // freq_cut_div)
    l_f = spec[:, :, cut:].abs().pow(2).mean() / T
    # 4) 裁剪预算越界 (本项目扩展, DUT 无此概念)
    h, w = float(shape_hw[0]), float(shape_hw[1])
    lim = P.new_tensor([crop_ratio / 2 * w, crop_ratio / 2 * h])
    excess = F.relu(B_field.abs() - lim.view(1, 2, 1, 1, 1))
    l_b = excess.pow(2).mean()
    # 5) 保形: 校正场逐帧的 R90 残差
    Bf = B_field.permute(0, 2, 1, 3, 4).flatten(0, 1)      # (B*T,2,GH,GW)
    l_s = shape_preservation(Bf, (h, w))

    total = (lam_d * l_data + lam_t * l_t + lam_f * l_f
             + lam_b * l_b + lam_s * l_s)
    return total, {"data": l_data.item(), "temporal": l_t.item(),
                   "freq": l_f.item(), "budget": l_b.item(),
                   "shape": l_s.item()}

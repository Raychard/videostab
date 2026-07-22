"""轨迹平滑求解器: 经典 fallback(运动自适应高斯) + 裁剪预算硬约束.

裁剪预算是产品承诺: |B| 分量级钳位在预算框内, 保证裁剪后无黑边.
"""
import numpy as np

from ..config import SmoothingConfig


def accumulate_path(motions) -> np.ndarray:
    """[(GH,GW,2)] * (T-1) -> 相机路径 C (T,GH,GW,2), C[0]=0."""
    m = np.stack(motions).astype(np.float32)
    C = np.zeros((len(m) + 1,) + m.shape[1:], np.float32)
    np.cumsum(m, axis=0, out=C[1:])
    return C


def _adaptive_sigmas(C: np.ndarray, cfg: SmoothingConfig) -> np.ndarray:
    """逐顶点运动自适应 sigma (T,GH,GW): 快速运镜的区域降低平滑强度,
    保跟拍意图 (NLNL 先验); 逐顶点而非全局标量, 运镜+局部视差不再一刀切."""
    v = np.diff(C, axis=0)                            # (T-1,GH,GW,2)
    speed = np.linalg.norm(v, axis=-1)                # (T-1,GH,GW)
    speed = np.concatenate([speed[:1], speed])        # 对齐到 T
    return np.maximum(
        cfg.base_sigma * np.exp(-speed / cfg.adapt_v0), 1.0
    ).astype(np.float32)


def _reflect_idx(T: int, offsets: np.ndarray) -> np.ndarray:
    """反射式时间索引 (T,K), 与越界掩码配合实现奇反射."""
    idx = np.arange(T)[:, None] + offsets[None, :]
    idx = np.abs(idx)                       # 左边界反射
    over = idx > T - 1
    idx[over] = np.maximum(2 * (T - 1) - idx[over], 0)  # 右边界反射
    return np.clip(idx, 0, T - 1)


def gaussian_smooth_path(C: np.ndarray, cfg: SmoothingConfig = None,
                         chunk: int = 256) -> np.ndarray:
    """双向逐顶点自适应高斯平滑. C/返回值 (T,GH,GW,2). 按时间分块控制内存.

    边界用奇反射(点对称)外推 P_ext(-k) = 2*P[0] - P[k]: 线性运镜路径
    被精确保持(clamp/偶反射会把匀速段拉弯), 抖动仍被正常平均.
    """
    cfg = cfg or SmoothingConfig()
    T = len(C)
    r = min(cfg.radius, T - 1)
    sigmas = _adaptive_sigmas(C, cfg)                 # (T,GH,GW)
    offsets = np.arange(-r, r + 1)
    raw = np.arange(T)[:, None] + offsets[None, :]    # (T,K) 可越界
    idx = _reflect_idx(T, offsets)
    left, right = raw < 0, raw > T - 1
    P = C.astype(np.float32)
    for _ in range(cfg.iterations):
        out = np.empty_like(P)
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            vals = P[idx[s:e]]                        # (c,K,GH,GW,2)
            lm, rm = left[s:e], right[s:e]
            if lm.any():
                vals[lm] = 2 * P[0] - vals[lm]        # 奇反射外推
            if rm.any():
                vals[rm] = 2 * P[-1] - vals[rm]
            w = np.exp(-offsets[None, :, None, None] ** 2
                       / (2 * sigmas[s:e, None] ** 2))
            w /= w.sum(1, keepdims=True)
            out[s:e] = np.einsum("tkhw,tkhwc->thwc", w, vals)
        P = out
    return P


def crop_budget_project(C: np.ndarray, P: np.ndarray, shape_hw: tuple,
                        crop_ratio: float, iters: int = 5) -> np.ndarray:
    """裁剪预算硬约束: 把平滑路径 P 投影到以 C 为中心的预算管道内.

    投影在 P 空间交替进行 "3 抽头平滑 -> 钳位到 [C-lim, C+lim]",
    保持 P 平滑的同时满足约束; 绝不能平滑 B=P-C 本身 —— B 必须精确
    携带 -C 的高频分量才能抵消抖动.
    返回 B (T,GH,GW,2), 保证 |Bx|<=margin_x, |By|<=margin_y.
    """
    h, w = shape_hw
    lim = np.array([crop_ratio / 2 * w, crop_ratio / 2 * h], np.float32)
    Pp = np.clip(P, C - lim, C + lim)
    for _ in range(iters if len(Pp) >= 3 else 0):
        Ps = Pp.copy()
        Ps[1:-1] = 0.25 * Pp[:-2] + 0.5 * Pp[1:-1] + 0.25 * Pp[2:]
        Pp = np.clip(Ps, C - lim, C + lim)
    return (Pp - C).astype(np.float32)

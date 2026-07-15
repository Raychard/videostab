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
    """运动自适应 sigma: 快速运镜段降低平滑强度, 保跟拍意图 (NLNL 先验)."""
    v = np.diff(C, axis=0)                      # (T-1,GH,GW,2)
    speed = np.linalg.norm(v, axis=-1).mean((1, 2))  # (T-1,)
    speed = np.concatenate([speed[:1], speed])       # 对齐到 T
    return np.maximum(cfg.base_sigma * np.exp(-speed / cfg.adapt_v0), 1.0)


def gaussian_smooth_path(C: np.ndarray, cfg: SmoothingConfig = None) -> np.ndarray:
    """双向运动自适应高斯平滑. C/返回值 (T,GH,GW,2)."""
    cfg = cfg or SmoothingConfig()
    T = len(C)
    r = min(cfg.radius, T - 1)
    sigmas = _adaptive_sigmas(C, cfg)
    offsets = np.arange(-r, r + 1)
    P = C.copy()
    for _ in range(cfg.iterations):
        idx = np.clip(np.arange(T)[:, None] + offsets[None, :], 0, T - 1)
        w = np.exp(-offsets[None, :] ** 2 / (2 * sigmas[:, None] ** 2))
        w /= w.sum(1, keepdims=True)
        P = np.einsum("tk,tk...->t...", w, P[idx])
    return P.astype(np.float32)


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

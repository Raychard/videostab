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


def similarity_split(B: np.ndarray, shape_hw: tuple):
    """把校正场 B (T,GH,GW,2) 最小二乘分解为 (相似变换拟合场, 残差).

    相似变换 = 平移 + 旋转 + 均匀缩放 (4 自由度), 在中心化顶点坐标下
    有闭式解; 残差即 B 中无法用任何相似变换解释的各向异性(剪切/拉伸)分量.
    """
    h, w = shape_hw
    yy, xx = np.meshgrid(np.linspace(0, h, B.shape[1]),
                         np.linspace(0, w, B.shape[2]), indexing="ij")
    x, y = (xx - xx.mean()).ravel(), (yy - yy.mean()).ravel()
    denom = (x * x + y * y).sum()
    d = B.reshape(len(B), -1, 2)
    bx, by = d[..., 0].mean(1), d[..., 1].mean(1)      # 平移分量
    cx, cy = d[..., 0] - bx[:, None], d[..., 1] - by[:, None]
    a = (x * cx + y * cy).sum(1) / denom               # 缩放 - 1
    c = (x * cy - y * cx).sum(1) / denom               # 旋转
    fit = np.stack([a[:, None] * x - c[:, None] * y + bx[:, None],
                    c[:, None] * x + a[:, None] * y + by[:, None]],
                   -1).reshape(B.shape).astype(np.float32)
    return fit, B - fit


def limit_anisotropy(B: np.ndarray, shape_hw: tuple, cap_ratio: float,
                     crop_ratio: float, iters: int = 1) -> np.ndarray:
    """封顶 B 的非相似(各向异性)分量幅值.

    为何必要: 相机路径按逐顶点平移累积、并逐顶点独立平滑. 平移主导的
    场景下各顶点运动几乎相同, 独立平滑无害; 但缩放/前进类运镜下顶点沿
    半径以不同速率运动, 独立平滑后各顶点相位不再协调, 合成的校正场不再
    是一个合法的相似变换, 网格被剪切拉伸 —— 这正是 Zooming 类 distortion
    崩坏的成因(实测 B 的非相似占比 Zooming 23~32% vs Regular 6%).

    为何用绝对幅值封顶而非整体缩放: 整体缩放会连同真实视差信号一起削弱
    (实测 Parallax rough 0.26->0.80); 封顶只削掉超出阈值的部分, 平移/
    视差场景的残差本就在阈值下, 几乎不受影响.

    为何只能限幅值, 不能靠空间滤波: 有害的各向异性是**整块栅格的大尺度
    低频剪切**, 空间上本就是平滑的 —— 实测对残差做空间低通(σ=0.8/1.5)
    只换来 +1.0~1.7% distortion, 而同样条件下限幅换来 +7.2%.

    效果(NUS 144 段, 默认 5px): distortion 0.8679->0.9033, 117/144 段
    改善(符号检验 z=+7.50); rough 与 stability 均无显著变化. Zooming 类
    是双赢 —— distortion 0.796->0.870 的同时 rough 也从 0.628 降到 0.551.

    关于 iters(默认 1, 即"削一次再钳一次"): 预算钳位是逐顶点分量级的非线性
    操作, 会把刚压下去的各向异性重新引入 —— 实测封顶设 5px 而残差最差帧
    达 13.85px, 且**预算越紧越严重**(crop 0.30 时反降到 7.65px), 因为校正
    需求远超预算盒时, 逐顶点 clip 把不同顶点削去不同的量, 这本身就是剪切.
    iters>1 走交替投影(两约束集均为凸, 收敛), 残差确有改善(Zooming/16
    13.85->11.61px), 但**端到端 A/B(36 段)三项指标全落在噪声内**
    (distortion z=+0.33, rough z=0.00, stability z=+0.67), 故默认保持 1.
    这是预算硬约束与相似性保持之间的固有张力, 补救式迭代解决不了; 真正的
    方向是让预算投影本身保持相似结构(对相似分量整体缩放而非逐顶点钳位).
    """
    cap = cap_ratio * shape_hw[0]
    lim = np.array([crop_ratio / 2 * shape_hw[1],
                    crop_ratio / 2 * shape_hw[0]], np.float32)
    out = B
    for _ in range(iters):
        fit, res = similarity_split(out, shape_hw)
        n = np.linalg.norm(res, axis=-1, keepdims=True)
        res = res * np.minimum(1.0, cap / np.maximum(n, 1e-6))  # 只削幅值
        out = np.clip(fit + res, -lim, lim)          # 预算是产品硬承诺
    return out.astype(np.float32)


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

"""稀疏→稠密运动传播: 自适应 K 多单应初始化 + 软融合 (DUT 硬归属的升级).

- K 自适应: 从 1 开始, 中位重投影误差超阈值且簇够大时增大 K (替代 DUT
  固定两平面 + 手调阈值).
- 软融合: 顶点对各平面单应按邻域关键点隶属度 + 距离衰减加权混合,
  消除平面边界撕裂 (NLNL).
"""
import cv2
import numpy as np

from ..config import PropagationConfig


def grid_vertices(shape_hw: tuple, grid_size: tuple) -> np.ndarray:
    """(GH,GW,2) 顶点坐标, 覆盖 [0,w-1]x[0,h-1]."""
    h, w = shape_hw
    gh, gw = grid_size
    xs = np.linspace(0, w - 1, gw, dtype=np.float32)
    ys = np.linspace(0, h - 1, gh, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy], axis=-1)


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


def _perspective_px(H: np.ndarray, shape_hw: tuple) -> float:
    """单应透视分量在四角产生的最大位移(px).

    真实帧间相机运动的透视分量极小(<1px); 双平面运动被单个 8 自由度
    单应"弯曲拟合"时, 残差可以很小但透视分量异常大 —— 这是病态拟合
    的可靠指纹, 残差分位数无法发现它.
    """
    h, w = shape_hw
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]],
                       np.float32)
    full = _apply_h(H, corners)
    affine = corners @ H[:2, :2].T + H[:2, 2]
    return float(np.abs(full - affine).max())


def _fit_cluster_homographies(pts, motions, labels, K, shape_hw,
                              cfg: PropagationConfig):
    """逐簇 RANSAC 单应. 返回 (Hs, errs): 失败/病态簇 err=inf."""
    Hs, errs = [], []
    for k in range(K):
        idx = labels == k
        if idx.sum() < 8:
            Hs.append(None)
            errs.append(np.inf)
            continue
        H, _ = cv2.findHomography(pts[idx], pts[idx] + motions[idx],
                                  cv2.RANSAC, 3.0)
        if H is None:
            Hs.append(None)
            errs.append(np.inf)
            continue
        Hs.append(H)
        if _perspective_px(H, shape_hw) > cfg.max_perspective_px:
            errs.append(np.inf)  # 病态拟合: 推动 K 增大
            continue
        err = np.linalg.norm(
            _apply_h(H, pts[idx]) - (pts[idx] + motions[idx]), axis=1)
        # 75 分位而非中位数: RANSAC 拟合半数点(如双平面)时中位数会假性为 0
        errs.append(float(np.percentile(err, 75)))
    return Hs, errs


def _adaptive_cluster(pts, motions, shape_hw, cfg: PropagationConfig):
    """自适应选 K: 误差达标即停, 簇过小不再分裂. 返回 (labels, Hs)."""
    n = len(pts)
    best = None
    for K in range(1, cfg.max_planes + 1):
        if K == 1:
            labels = np.zeros(n, np.int32)
        else:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
            _, labels, _ = cv2.kmeans(motions.astype(np.float32), K, None,
                                      crit, 3, cv2.KMEANS_PP_CENTERS)
            labels = labels.ravel()
            counts = np.bincount(labels, minlength=K)
            if counts.min() < max(8, cfg.min_cluster_frac * n):
                break  # 分裂出碎簇, 停止加 K
        Hs, errs = _fit_cluster_homographies(pts, motions, labels, K,
                                             shape_hw, cfg)
        med = max(errs) if errs else np.inf  # 最差簇决定质量(病态=inf)
        if best is None or med < best[2]:
            best = (labels, Hs, med)
        if med < cfg.kmeans_err_thresh:
            break
    return best[0], best[1], best[2]


def _soft_fusion(query: np.ndarray, pts, labels, Hs, sigma: float):
    """对 query 点集按软权重混合各簇单应的位移. 返回 (M,2)."""
    q = query.reshape(-1, 2)
    disp = np.zeros((len(q), 2), np.float32)
    wsum = np.zeros(len(q), np.float32)
    for k, H in enumerate(Hs):
        if H is None:
            continue
        src = pts[labels == k]
        if len(src) == 0:
            continue
        d2 = ((q[:, None, :] - src[None, :, :]) ** 2).sum(-1)
        w = np.exp(-d2 / (2 * sigma ** 2)).sum(1) + 1e-6
        disp += w[:, None] * (_apply_h(H, q) - q)
        wsum += w
    ok = wsum > 0
    disp[ok] /= wsum[ok, None]
    return disp


def propagate_homography(pts: np.ndarray, motions: np.ndarray,
                         shape_hw: tuple, cfg: PropagationConfig = None):
    """返回 (grid_motion (GH,GW,2), kp_init (N,2), info dict).

    kp_init 为初始化场在关键点处的取值, 供残差网络输入/训练用.
    info['grid_err'] 为 QG-2 守门信号.
    """
    cfg = cfg or PropagationConfig()
    verts = grid_vertices(shape_hw, cfg.grid_size)
    gh, gw = cfg.grid_size

    if len(pts) < 8:  # 退化: 全局平移(中位数), 无点则零运动
        t = np.median(motions, axis=0) if len(pts) else np.zeros(2, np.float32)
        grid = np.broadcast_to(t.astype(np.float32), (gh, gw, 2)).copy()
        kp_init = np.broadcast_to(t.astype(np.float32), (len(pts), 2)).copy()
        return grid, kp_init, {"K": 0, "grid_err": np.inf}

    labels, Hs, med_err = _adaptive_cluster(pts, motions, shape_hw, cfg)
    sigma = cfg.soft_sigma_frac * min(shape_hw)
    grid = _soft_fusion(verts, pts, labels, Hs, sigma).reshape(gh, gw, 2)
    kp_init = _soft_fusion(pts, pts, labels, Hs, sigma)
    grid_err = float(np.median(np.linalg.norm(kp_init - motions, axis=1)))
    K_used = sum(1 for H in Hs if H is not None)
    return grid, kp_init, {"K": K_used, "grid_err": grid_err,
                           "fit_err": med_err}

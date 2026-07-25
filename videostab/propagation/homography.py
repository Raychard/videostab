"""稀疏→稠密运动传播: 自适应 K 多单应初始化 + 软融合 (DUT 硬归属的升级).

- K 自适应: 相对模型选择, 仅当 K+1 显著降低拟合误差时才分裂 (替代 DUT
  固定两平面 + 手调阈值).
- 软融合: 顶点对各平面单应按邻域关键点隶属度 + 距离衰减加权混合,
  消除平面边界撕裂 (NLNL).

实测提醒: 真实"视差"场景是深度**连续**分布, 而非两个离散平面.
NUS Parallax 上 err(K=2)/err(K=1) 的中位数为 0.993 —— 一半帧对上分平面
对拟合毫无改善, 25% 以上情况反而更差. 因此多单应机制的适用面比
"分平面处理视差"的直觉要窄得多, 连续深度变化主要由后续残差网络吸收
(这也是传播网改善 stability 却损失 distortion 的来源).
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


def _fit_cluster_homographies(pts, motions, labels, K, shape_hw=None,
                              cfg: PropagationConfig = None):
    """逐簇 RANSAC 单应. 返回 (Hs, errs): 拟合失败的簇 err=inf.

    误差用 75 分位而非中位数: RANSAC 只需拟合半数点(如双平面场景)时,
    中位数会假性为 0.
    """
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
        err = np.linalg.norm(
            _apply_h(H, pts[idx]) - (pts[idx] + motions[idx]), axis=1)
        errs.append(float(np.percentile(err, 75)))
    return Hs, errs


def _adaptive_cluster(pts, motions, shape_hw, cfg: PropagationConfig):
    """自适应选 K —— 相对模型选择: 只有当 K+1 把误差降到 K 的
    split_gain 倍以下时才接受分裂. 返回 (labels, Hs, err).

    为何用相对判据而非绝对像素阈值: 单应拟合误差的绝对量级随场景/运动
    幅度剧烈变化, 任何固定阈值都会在一部分数据上失效. 早先版本用
    "误差<3px 即停 + 透视分量>2px 判为病态", 结果在真实视差数据上
    单应的合法透视分量中位数就有 75px(视差本身就是这么来的), 守门逢帧
    必触发, 把所有 K 的误差都置为 inf, 自适应逻辑完全瘫痪(实测 Parallax
    上 K>=2 占比仅 1%). 相对判据不依赖绝对尺度, 且天然能识别"单个单应
    弯曲拟合双平面"——那种情况下 K=2 会显著降低误差.
    """
    n = len(pts)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    labels = np.zeros(n, np.int32)
    Hs, errs = _fit_cluster_homographies(pts, motions, labels, 1)
    best = (labels, Hs, max(errs))

    for K in range(2, cfg.max_planes + 1):
        _, lab, _ = cv2.kmeans(motions.astype(np.float32), K, None,
                               crit, 3, cv2.KMEANS_PP_CENTERS)
        lab = lab.ravel()
        counts = np.bincount(lab, minlength=K)
        if counts.min() < max(8, cfg.min_cluster_frac * n):
            break                      # 分裂出碎簇, 停止加 K
        Hs_k, errs_k = _fit_cluster_homographies(pts, motions, lab, K)
        err_k = max(errs_k)
        if not np.isfinite(err_k) or err_k > best[2] * cfg.split_gain:
            break                      # 增益不足, 不值得多一个平面
        best = (lab, Hs_k, err_k)
    return best


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

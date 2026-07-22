"""稀疏运动估计: 检测 + 跟踪 + 几何法前景剔除, 并输出质量守门信号 QG-1."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import MotionConfig
from .detectors import detect_keypoints
from .flow import track_lk


@dataclass
class SparseMotion:
    """一对相邻帧的稀疏背景运动与质量信号."""
    pts: np.ndarray        # (N,2) 背景关键点(帧 t 代理坐标)
    motions: np.ndarray    # (N,2) t->t+1 运动
    n_detected: int = 0    # 检测到的关键点数
    n_tracked: int = 0     # 跟踪成功(过前后向校验)数
    inlier_ratio: float = 0.0  # RANSAC 背景内点率
    signals: dict = field(default_factory=dict)


def _reject_foreground(pts, motions, thresh: float, max_models: int = 3):
    """顺序多模型 RANSAC 剔除动态前景/误跟踪.

    依次拟合最多 max_models 个单应平面(拟合->移除内点->对剩余点再拟合),
    保留所有主平面内点的并集; 被剔除的是不属于任何主平面的点.
    单模型版本会把视差场景的第二平面整体误判为前景, 使下游多单应失效.
    """
    n = len(pts)
    if n < 8:
        return np.ones(n, bool), 0.0
    keep = np.zeros(n, bool)
    remaining = np.arange(n)
    min_plane = max(8, int(0.05 * n))  # 平面内点数下限, 太小不算主平面
    for _ in range(max_models):
        if len(remaining) < min_plane:
            break
        H, mask = cv2.findHomography(pts[remaining],
                                     pts[remaining] + motions[remaining],
                                     cv2.RANSAC, thresh)
        if H is None:
            break
        inl = mask.ravel().astype(bool)
        if inl.sum() < min_plane:
            break
        keep[remaining[inl]] = True
        remaining = remaining[~inl]
    if not keep.any():
        return np.ones(n, bool), 0.0
    return keep, float(keep.mean())


def estimate_sparse_motion(gray0: np.ndarray, gray1: np.ndarray,
                           cfg: MotionConfig = None,
                           tracker=None) -> SparseMotion:
    """tracker: 可注入 RaftFlow().track, 默认 LK."""
    cfg = cfg or MotionConfig()
    pts = detect_keypoints(gray0, cfg)
    if tracker is None:
        motions, valid = track_lk(gray0, gray1, pts, cfg.lk_win, cfg.fb_thresh)
    else:
        motions, valid = tracker(gray0, gray1, pts, cfg.fb_thresh)
    n_det = len(pts)
    pts, motions = pts[valid], motions[valid]
    inl, ratio = _reject_foreground(pts, motions, cfg.ransac_thresh)
    sm = SparseMotion(pts=pts[inl], motions=motions[inl],
                      n_detected=n_det, n_tracked=int(valid.sum()),
                      inlier_ratio=ratio)
    sm.signals = {"n_kp": len(sm.pts), "inlier_ratio": ratio,
                  "track_ratio": sm.n_tracked / max(n_det, 1)}
    return sm

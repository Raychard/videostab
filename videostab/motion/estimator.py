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


def _reject_foreground(pts, motions, thresh: float):
    """RANSAC 全局单应内点判定, 剔除运动异常点(动态前景/误跟踪)."""
    if len(pts) < 8:
        return np.ones(len(pts), bool), 0.0
    H, mask = cv2.findHomography(pts, pts + motions, cv2.RANSAC, thresh)
    if H is None:
        return np.ones(len(pts), bool), 0.0
    inl = mask.ravel().astype(bool)
    return inl, float(inl.mean())


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

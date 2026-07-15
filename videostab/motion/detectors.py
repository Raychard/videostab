"""关键点检测: 双检测器协同 (ORB + Shi-Tomasi) + 空间均匀化 (NLNL).

接口只依赖灰度图, 返回 (N,2) float32 坐标; 换用 ALIKE/SuperPoint 时
只需新增一个 _detect_xxx 并加入 DETECTORS.
"""
import cv2
import numpy as np

from ..config import MotionConfig


def _detect_orb(gray: np.ndarray, n: int) -> np.ndarray:
    orb = cv2.ORB_create(nfeatures=n)
    kps = orb.detect(gray, None)
    if not kps:
        return np.empty((0, 3), np.float32)
    return np.array([[k.pt[0], k.pt[1], k.response] for k in kps], np.float32)


def _detect_gftt(gray: np.ndarray, n: int) -> np.ndarray:
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=n, qualityLevel=0.01,
                                  minDistance=8)
    if pts is None:
        return np.empty((0, 3), np.float32)
    pts = pts.reshape(-1, 2)
    # GFTT 无响应值, 统一给中等响应, 排序时让位给 ORB 强角点
    resp = np.full((len(pts), 1), 0.5, np.float32)
    return np.hstack([pts.astype(np.float32), resp])


DETECTORS = (_detect_orb, _detect_gftt)


def _uniformize(pts: np.ndarray, shape_hw: tuple, cfg: MotionConfig) -> np.ndarray:
    """网格化空间均匀采样: 每格按响应保留 top-k, 防纹理聚集偏差."""
    if len(pts) == 0:
        return pts[:, :2] if pts.size else np.empty((0, 2), np.float32)
    h, w = shape_hw
    rows, cols = cfg.grid_cells
    cell_r = np.clip((pts[:, 1] / h * rows).astype(int), 0, rows - 1)
    cell_c = np.clip((pts[:, 0] / w * cols).astype(int), 0, cols - 1)
    cell_id = cell_r * cols + cell_c
    keep = []
    for cid in np.unique(cell_id):
        idx = np.where(cell_id == cid)[0]
        order = idx[np.argsort(-pts[idx, 2])]
        keep.extend(order[: cfg.cap_per_cell])
    keep = np.array(keep)
    if len(keep) > cfg.max_keypoints:
        keep = keep[np.argsort(-pts[keep, 2])[: cfg.max_keypoints]]
    return pts[keep, :2].astype(np.float32)


def detect_keypoints(gray: np.ndarray, cfg: MotionConfig = None) -> np.ndarray:
    """双检测器协同 + 去重 + 空间均匀化. 返回 (N,2) float32."""
    cfg = cfg or MotionConfig()
    all_pts = [d(gray, cfg.max_keypoints) for d in DETECTORS]
    pts = np.vstack([p for p in all_pts if len(p)]) if any(
        len(p) for p in all_pts) else np.empty((0, 3), np.float32)
    if len(pts) == 0:
        return np.empty((0, 2), np.float32)
    # 简易 NMS 去重: 量化到 4px 格, 每格留最高响应(先按响应降序再取首次出现)
    pts = pts[np.argsort(-pts[:, 2])]
    key = (pts[:, :2] / 4).astype(int)
    _, first = np.unique(key, axis=0, return_index=True)
    pts = pts[np.sort(first)]
    return _uniformize(pts, gray.shape, cfg)

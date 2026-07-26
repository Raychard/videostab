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
    # GFTT 返回序即质量降序, 按序编码递减响应(供秩归一化保序)
    resp = np.linspace(1.0, 0.01, len(pts), dtype=np.float32)[:, None]
    return np.hstack([pts.astype(np.float32), resp])


class _AlikedDetector:
    """ALIKED (kornia) 学习式检测器. 懒加载, 首次调用才建模型.

    NLNL 用 SIFT+ORB+SuperPoint+ALIKE 四检测器协同; 这里提供 ALIKED
    作为学习式一员, 与 ORB 组成"学习式 + 传统兜底"的组合.
    """

    def __init__(self, device: str = "cuda", top_k: int = 512):
        self.device = device
        self.top_k = top_k
        self._model = None

    def _lazy(self):
        if self._model is None:
            import torch
            from kornia.feature import ALIKED
            dev = self.device if __import__("torch").cuda.is_available() \
                else "cpu"
            self._model = ALIKED(detection_threshold=0.0,
                                 max_num_keypoints=self.top_k).to(dev).eval()
            self._dev = dev
            self._torch = torch
        return self._model

    def __call__(self, gray: np.ndarray, n: int) -> np.ndarray:
        model = self._lazy()
        torch = self._torch
        t = torch.from_numpy(gray).float()[None, None] / 255.0
        t = t.repeat(1, 3, 1, 1).to(self._dev)
        with torch.no_grad():
            feats = model(t)[0]          # ALIKEDFeatures
        kp = feats.keypoints.cpu().numpy()
        sc = feats.keypoint_scores.cpu().numpy()
        if len(kp) == 0:
            return np.empty((0, 3), np.float32)
        return np.hstack([kp.astype(np.float32),
                          sc.reshape(-1, 1)]).astype(np.float32)


_ALIKED = _AlikedDetector()

# 检测器组合. 通过 set_detectors() 切换, 便于做消融对比.
DETECTORS = (_detect_orb, _detect_gftt)


_COMBOS = {
    "orb_gftt": (_detect_orb, _detect_gftt),
    "orb_aliked": (_detect_orb, _ALIKED),
    # 单检测器组合: 用于隔离对比(混合会因秩归一化而均等代表, 掩盖差异)
    "gftt": (_detect_gftt,),
    "aliked": (_ALIKED,),
    "orb": (_detect_orb,),
    "gftt_aliked": (_detect_gftt, _ALIKED),
    "orb_gftt_aliked": (_detect_orb, _detect_gftt, _ALIKED),
}


def set_detectors(name: str):
    """切换检测器组合. 可选: orb_gftt(默认) | orb_aliked | gftt | aliked | orb."""
    global DETECTORS
    if name not in _COMBOS:
        raise ValueError(f"未知检测器组合: {name}")
    DETECTORS = _COMBOS[name]
    return DETECTORS


def _rank_normalize(pts: np.ndarray) -> np.ndarray:
    """检测器内部响应秩归一化到 (0,1]: 不同检测器响应量级不可比
    (ORB Harris ~1e-3, GFTT 恒 0.5), 直接比较会让单一检测器垄断排序."""
    if len(pts) == 0:
        return pts
    rank = np.argsort(np.argsort(pts[:, 2]))
    pts[:, 2] = (rank + 1) / len(pts)
    return pts


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
    dets = set_detectors(cfg.detectors) if cfg.detectors else DETECTORS
    all_pts = [_rank_normalize(d(gray, cfg.max_keypoints)) for d in dets]
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

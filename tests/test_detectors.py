import numpy as np

from videostab.config import MotionConfig
from videostab.motion import detect_keypoints


def test_detects_enough_keypoints(texture):
    pts = detect_keypoints(texture)
    assert len(pts) >= 50
    h, w = texture.shape
    assert (pts[:, 0] >= 0).all() and (pts[:, 0] < w).all()
    assert (pts[:, 1] >= 0).all() and (pts[:, 1] < h).all()


def test_uniformization_caps_per_cell(texture):
    cfg = MotionConfig()
    pts = detect_keypoints(texture, cfg)
    h, w = texture.shape
    rows, cols = cfg.grid_cells
    cell = (np.clip((pts[:, 1] / h * rows).astype(int), 0, rows - 1) * cols
            + np.clip((pts[:, 0] / w * cols).astype(int), 0, cols - 1))
    counts = np.bincount(cell)
    assert counts.max() <= cfg.cap_per_cell
    assert len(pts) <= cfg.max_keypoints


def test_blank_image_returns_empty():
    blank = np.full((240, 320), 128, np.uint8)
    pts = detect_keypoints(blank)
    assert len(pts) < 8  # 无纹理 -> 几乎无关键点(守门降级依据)

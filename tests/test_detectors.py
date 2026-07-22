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


def test_rank_normalization_makes_detectors_comparable(texture):
    """ORB 响应 ~1e-3 vs GFTT 编码响应: 秩归一化后两者同尺度,
    避免单一检测器垄断 NMS/均匀化排序 (回归防护)."""
    from videostab.motion.detectors import (_detect_gftt, _detect_orb,
                                            _rank_normalize)
    orb = _rank_normalize(_detect_orb(texture, 512))
    gftt = _rank_normalize(_detect_gftt(texture, 512))
    for pts in (orb, gftt):
        assert pts[:, 2].max() <= 1.0 + 1e-6
        assert pts[:, 2].max() > 0.99      # 各自最强点都能到 1.0 量级
        assert pts[:, 2].min() > 0.0


def test_both_detectors_contribute(texture):
    """最终关键点中应同时存在两个来源的点(4px 格内归属判断)."""
    from videostab.motion.detectors import _detect_gftt, _detect_orb
    kps = detect_keypoints(texture)
    orb = _detect_orb(texture, 512)[:, :2]
    gftt = _detect_gftt(texture, 512)[:, :2]

    def near(a, b, tol=2.0):
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).min(1)
        return d < tol

    from_orb = near(kps, orb) & ~near(kps, gftt)
    from_gftt = near(kps, gftt) & ~near(kps, orb)
    assert from_orb.sum() > 10, "ORB 独有点未进入最终集合"
    assert from_gftt.sum() > 10, "GFTT 独有点未进入最终集合"


def test_blank_image_returns_empty():
    blank = np.full((240, 320), 128, np.uint8)
    pts = detect_keypoints(blank)
    assert len(pts) < 8  # 无纹理 -> 几乎无关键点(守门降级依据)

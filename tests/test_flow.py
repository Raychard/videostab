import numpy as np

from conftest import make_texture
from videostab.motion import detect_keypoints, estimate_sparse_motion
from videostab.motion.flow import track_lk


def _shifted_pair(dx=3, dy=-2, seed=0):
    big = make_texture(300, 400, seed)
    g0 = big[30:270, 40:360]
    g1 = big[30 + dy : 270 + dy, 40 + dx : 360 + dx]
    return g0, g1


def test_lk_recovers_translation():
    g0, g1 = _shifted_pair(3, -2)
    pts = detect_keypoints(g0)
    motions, valid = track_lk(g0, g1, pts)
    assert valid.sum() > 30
    med = np.median(motions[valid], axis=0)
    # 相机右下移 -> 内容左上移: 特征运动 = -位移
    assert abs(med[0] - (-3)) < 0.3
    assert abs(med[1] - 2) < 0.3


def test_estimator_signals():
    g0, g1 = _shifted_pair(2, 1)
    sm = estimate_sparse_motion(g0, g1)
    assert sm.signals["n_kp"] > 30
    assert sm.signals["inlier_ratio"] > 0.8  # 纯平移场景内点率应很高
    med = np.median(sm.motions, axis=0)
    assert np.allclose(med, [-2, -1], atol=0.3)


def test_estimator_handles_blank():
    blank = np.full((240, 320), 128, np.uint8)
    sm = estimate_sparse_motion(blank, blank)
    assert sm.signals["n_kp"] < 8  # 触发 L2 守门

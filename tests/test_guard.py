import numpy as np

from videostab.config import GuardConfig
from videostab.guard import GuardLevel, decide_level, strength_curve


def test_levels():
    good = {"n_kp": 300, "inlier_ratio": 0.9, "grid_err": 1.0}
    assert decide_level(good) == GuardLevel.L0_FULL
    assert decide_level({"n_kp": 60, "inlier_ratio": 0.9, "grid_err": 1.0}) \
        == GuardLevel.L1_CONSERVATIVE
    assert decide_level({"n_kp": 300, "inlier_ratio": 0.3, "grid_err": 1.0}) \
        == GuardLevel.L1_CONSERVATIVE
    assert decide_level({"n_kp": 300, "inlier_ratio": 0.9, "grid_err": 20.0}) \
        == GuardLevel.L1_CONSERVATIVE
    assert decide_level({"n_kp": 5}) == GuardLevel.L2_PASSTHROUGH


def test_strength_ramp_no_jump():
    cfg = GuardConfig(ramp_frames=10)
    levels = ([GuardLevel.L0_FULL] * 30 + [GuardLevel.L2_PASSTHROUGH] * 20
              + [GuardLevel.L0_FULL] * 30)
    s = strength_curve(levels, cfg)
    assert np.all(np.abs(np.diff(s)) <= 0.1 + 1e-6)  # 每帧变化 <= 1/ramp
    assert np.all(s[35:45] == 0.0)                    # L2 段中心强度为 0
    assert s[0] == 1.0 and s[-1] == 1.0
    # L2 段之前就开始渐降(反向扫描的效果)
    assert s[29] < 1.0


def test_all_good_full_strength():
    s = strength_curve([GuardLevel.L0_FULL] * 50)
    assert np.all(s == 1.0)

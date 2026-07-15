"""三级失效降级状态机 + 防抖强度渐变.

L0 完整管线 / L1 保守(全局平移+强平滑约束) / L2 直通.
逐帧判级取滚动最差, 强度切换用线性渐变(约 1s), 绝不跳变.
"""
from enum import IntEnum

import numpy as np

from ..config import GuardConfig


class GuardLevel(IntEnum):
    L0_FULL = 0
    L1_CONSERVATIVE = 1
    L2_PASSTHROUGH = 2


def decide_level(signals: dict, cfg: GuardConfig = None) -> GuardLevel:
    """signals: {n_kp, inlier_ratio, grid_err} -> 降级等级."""
    cfg = cfg or GuardConfig()
    n_kp = signals.get("n_kp", 0)
    if n_kp < cfg.min_kp_l2:
        return GuardLevel.L2_PASSTHROUGH
    if (n_kp < cfg.min_kp_l1
            or signals.get("inlier_ratio", 1.0) < cfg.min_inlier_ratio
            or signals.get("grid_err", 0.0) > cfg.max_grid_err):
        return GuardLevel.L1_CONSERVATIVE
    return GuardLevel.L0_FULL


def strength_curve(levels, cfg: GuardConfig = None) -> np.ndarray:
    """逐帧等级序列 -> 逐帧防抖强度 s∈[0,1], 线性渐变防跳变.

    L0/L1 目标强度 1.0(L1 已在上游换保守算法), L2 目标 0.0.
    每帧强度变化量不超过 1/ramp_frames.
    """
    cfg = cfg or GuardConfig()
    target = np.array([0.0 if l == GuardLevel.L2_PASSTHROUGH else 1.0
                       for l in levels], np.float32)
    step = 1.0 / max(cfg.ramp_frames, 1)
    # 下包络 s[i] = min_j (target[j] + step*|i-j|): 两遍扫描
    s = target.copy()
    for i in range(1, len(s)):            # 前向
        s[i] = min(s[i], s[i - 1] + step)
    for i in range(len(s) - 2, -1, -1):   # 反向
        s[i] = min(s[i], s[i + 1] + step)
    return np.clip(s, 0.0, 1.0)

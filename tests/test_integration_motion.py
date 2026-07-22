"""estimator -> propagation 集成测试: 曾因单模型 RANSAC 预过滤
把视差第二平面整体误杀, 导致流水线中自适应 K 恒为 1 (回归防护)."""
import numpy as np

from conftest import make_texture
from videostab.motion import estimate_sparse_motion
from videostab.propagation import propagate_homography


def _two_plane_pair(bg_dx=2, fg_dx=8, h=240, w=320):
    """背景移动 bg_dx, 左侧 1/3 区域为另一平面移动 fg_dx."""
    big = make_texture(400, 500, seed=7)
    g0 = big[80 : 80 + h, 90 : 90 + w].copy()
    fg = make_texture(h, w // 3, seed=11)
    g0[:, 20 : 20 + w // 3] = fg
    g1 = np.roll(g0, bg_dx, axis=1).copy()
    g1[:, 20 + fg_dx : 20 + fg_dx + w // 3] = fg
    return g0, g1


def test_second_plane_survives_rejection():
    g0, g1 = _two_plane_pair()
    sm = estimate_sparse_motion(g0, g1)
    dx = np.round(sm.motions[:, 0])
    # 两个平面的运动都应存在于剔除后的点集中
    assert (dx == 2).sum() > 20, "背景平面点丢失"
    assert (dx == 8).sum() > 20, "第二平面被前景剔除误杀"


def test_pipeline_path_adaptive_k_reaches_two():
    g0, g1 = _two_plane_pair()
    sm = estimate_sparse_motion(g0, g1)
    _, _, info = propagate_homography(sm.pts, sm.motions, g0.shape)
    assert info["K"] >= 2, "流水线路径下多单应未生效"


def test_true_foreground_still_rejected():
    """散乱随机运动(真前景/误跟踪)仍应被剔除, 不因多模型而放宽."""
    rng = np.random.default_rng(0)
    n_bg, n_fg = 300, 30
    pts = rng.uniform([10, 10], [310, 230], (n_bg + n_fg, 2)).astype(np.float32)
    motions = np.tile(np.array([3.0, 0.0], np.float32), (n_bg + n_fg, 1))
    # 30 个点随机运动且不成平面
    motions[n_bg:] = rng.uniform(-15, 15, (n_fg, 2)).astype(np.float32)
    from videostab.motion.estimator import _reject_foreground
    keep, ratio = _reject_foreground(pts, motions, thresh=2.0)
    assert keep[:n_bg].mean() > 0.95      # 背景保留
    assert keep[n_bg:].mean() < 0.4       # 散乱运动大部分被剔除

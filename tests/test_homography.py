import numpy as np

from videostab.config import PropagationConfig
from videostab.propagation import grid_vertices, propagate_homography

SHAPE = (240, 320)


def _grid_pts(seed=0, n=200):
    rng = np.random.default_rng(seed)
    return rng.uniform([10, 10], [310, 230], (n, 2)).astype(np.float32)


def test_pure_translation():
    pts = _grid_pts()
    motions = np.tile(np.array([4.0, -3.0], np.float32), (len(pts), 1))
    grid, kp_init, info = propagate_homography(pts, motions, SHAPE)
    assert np.allclose(grid[..., 0], 4.0, atol=0.2)
    assert np.allclose(grid[..., 1], -3.0, atol=0.2)
    assert info["grid_err"] < 0.5
    assert np.allclose(kp_init, motions, atol=0.2)


def test_two_plane_adaptive_k():
    pts = _grid_pts(n=300)
    motions = np.where(pts[:, :1] < 160,
                       np.array([6.0, 0.0], np.float32),
                       np.array([-6.0, 0.0], np.float32))
    cfg = PropagationConfig()
    grid, _, info = propagate_homography(pts, motions, SHAPE, cfg)
    assert info["K"] >= 2  # 单平面无法拟合, 应自适应分裂
    verts = grid_vertices(SHAPE, cfg.grid_size)
    left = grid[verts[..., 0] < 60]
    right = grid[verts[..., 0] > 260]
    assert left[:, 0].mean() > 3.0    # 左区跟随左平面运动
    assert right[:, 0].mean() < -3.0  # 右区跟随右平面运动


def test_degenerate_few_points():
    pts = np.array([[50, 50], [100, 100]], np.float32)
    motions = np.array([[2, 1], [2, 1]], np.float32)
    grid, _, info = propagate_homography(pts, motions, SHAPE)
    assert info["K"] == 0
    assert np.allclose(grid, [2, 1])  # 中位平移退化


def test_grid_vertices_cover_frame():
    v = grid_vertices(SHAPE, (12, 16))
    assert v.shape == (12, 16, 2)
    assert v[0, 0].tolist() == [0, 0]
    assert v[-1, -1].tolist() == [319, 239]

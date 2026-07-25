import numpy as np
import torch

from videostab.propagation import ResidualRefineNet, grid_vertex_tensor
from videostab.propagation.refine_net import (MOTION_NORM, grid_vertex_batch,
                                              refine_grid)

SHAPE = (240, 320)
GRID = (12, 16)


def _sample(n=100, seed=0):
    rng = np.random.default_rng(seed)
    grid_init = rng.normal(0, 2, GRID + (2,)).astype(np.float32)
    pts = rng.uniform([0, 0], [319, 239], (n, 2)).astype(np.float32)
    kp_init = rng.normal(0, 2, (n, 2)).astype(np.float32)
    motions = kp_init + rng.normal(0, 1, (n, 2)).astype(np.float32)
    return grid_init, pts, motions, kp_init


def test_param_budget_and_shape():
    net = ResidualRefineNet()
    assert sum(p.numel() for p in net.parameters()) < 100_000
    B, V, N = 2, GRID[0] * GRID[1], 64
    verts = torch.rand(B, V, 2) * 200
    kp = torch.rand(B, N, 2) * 200
    resid = torch.randn(B, N, 2)
    mask = torch.ones(B, N, dtype=torch.bool)
    assert net(verts, kp, resid, mask).shape == (B, V, 2)


def test_zero_init_preserves_initialization():
    """零初始化: 残差为 0, 精确保留多单应初始化场."""
    net = ResidualRefineNet().eval()
    grid_init, pts, motions, kp_init = _sample()
    refined = refine_grid(net, grid_init, pts, motions, kp_init, SHAPE)
    assert np.allclose(refined, grid_init, atol=1e-5)


def test_handles_masked_keypoints():
    """padding 的无效关键点不应污染结果或产生 NaN."""
    net = ResidualRefineNet().eval()
    B, V, N = 1, GRID[0] * GRID[1], 64
    verts = torch.rand(B, V, 2) * 200
    kp = torch.rand(B, N, 2) * 200
    resid = torch.randn(B, N, 2)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :10] = True                      # 只有 10 个有效
    out = net(verts, kp, resid, mask)
    assert torch.isfinite(out).all()
    mask_all_bad = torch.zeros(B, N, dtype=torch.bool)
    assert torch.isfinite(net(verts, kp, resid, mask_all_bad)).all()


def test_grid_vertex_batch_matches_single():
    shp = torch.tensor([[240, 320], [480, 640]])
    vb = grid_vertex_batch(shp, GRID)
    v0 = grid_vertex_tensor((240, 320), GRID, "cpu")
    assert vb.shape == (2, GRID[0] * GRID[1], 2)
    assert torch.allclose(vb[0], v0[0], atol=1e-4)


def test_attention_downweights_outlier_keypoints():
    """关键点注意力应能在关键点之间做选择(ECA 通道注意力做不到).

    验证: 网络输出对某个关键点的残差值敏感 => 存在可学习的选择通路.
    """
    torch.manual_seed(0)
    net = ResidualRefineNet()
    for p in net.dec[-1].parameters():       # 解除零初始化以观察通路
        torch.nn.init.normal_(p, std=0.1)
    B, V, N = 1, 16, 32
    verts = torch.rand(B, V, 2) * 200
    kp = torch.rand(B, N, 2) * 200
    resid = torch.zeros(B, N, 2)
    out0 = net(verts, kp, resid, torch.ones(B, N, dtype=torch.bool))
    resid[0, 0] = 50.0                       # 制造一个离群关键点
    out1 = net(verts, kp, resid, torch.ones(B, N, dtype=torch.bool))
    assert not torch.allclose(out0, out1)    # 单个关键点可影响输出


def test_overfit_single_sample():
    """过拟合单样本: 关键点一致性误差应显著下降."""
    from train.losses import propagation_loss
    torch.manual_seed(0)
    net = ResidualRefineNet()
    grid_init, pts, motions, kp_init = _sample()
    gi = torch.from_numpy(grid_init.transpose(2, 0, 1))[None]
    verts = grid_vertex_tensor(SHAPE, GRID, "cpu")
    kp = torch.from_numpy(pts)[None]
    tm = torch.from_numpy(motions)[None]
    tr = torch.from_numpy(motions - kp_init)[None]
    mask = torch.ones(1, len(pts), dtype=torch.bool)
    opt = torch.optim.Adam(net.parameters(), lr=2e-2)
    losses = []
    for _ in range(200):
        delta = net(verts, kp, tr, mask)
        pred = gi + delta.transpose(1, 2).reshape(1, 2, *GRID) * MOTION_NORM
        loss, err = propagation_loss(pred, verts, kp, tm, mask, SHAPE)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(err.item())
    assert losses[-1] < losses[0] * 0.8

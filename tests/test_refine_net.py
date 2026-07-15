import numpy as np
import torch

from videostab.propagation import ResidualRefineNet, build_refine_input
from videostab.propagation.refine_net import MOTION_NORM, refine_grid


def _sample(gh=12, gw=16, n=100, seed=0):
    rng = np.random.default_rng(seed)
    shape_hw = (240, 320)
    grid_init = rng.normal(0, 2, (gh, gw, 2)).astype(np.float32)
    pts = rng.uniform([0, 0], [319, 239], (n, 2)).astype(np.float32)
    kp_init = rng.normal(0, 2, (n, 2)).astype(np.float32)
    motions = kp_init + rng.normal(0, 1, (n, 2)).astype(np.float32)
    return grid_init, pts, motions, kp_init, shape_hw


def test_param_budget_and_shape():
    net = ResidualRefineNet()
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params < 100_000  # 轻量承诺
    x = torch.randn(4, 5, 12, 16)
    assert net(x).shape == (4, 2, 12, 16)


def test_zero_init_preserves_initialization():
    """head 零初始化: 未训练时残差为 0, 不破坏多单应初始化场."""
    net = ResidualRefineNet().eval()
    grid_init, pts, motions, kp_init, shape_hw = _sample()
    refined = refine_grid(net, grid_init, pts, motions, kp_init, shape_hw)
    assert np.allclose(refined, grid_init, atol=1e-5)


def test_build_input_shape_and_norm():
    grid_init, pts, motions, kp_init, shape_hw = _sample()
    feat = build_refine_input(grid_init, pts, motions, kp_init, shape_hw)
    assert feat.shape == (5, 12, 16)
    assert np.abs(feat).max() < 10  # 归一化后应在合理范围


def test_overfit_single_sample():
    """30 步过拟合单样本: 关键点一致性损失应显著下降."""
    from train.losses import propagation_loss
    torch.manual_seed(0)
    net = ResidualRefineNet()
    grid_init, pts, motions, kp_init, shape_hw = _sample()
    feat = torch.from_numpy(
        build_refine_input(grid_init, pts, motions, kp_init, shape_hw))[None]
    gi = torch.from_numpy(grid_init.transpose(2, 0, 1))[None]
    tp = torch.from_numpy(pts)[None]
    tm = torch.from_numpy(motions)[None]
    mask = torch.ones(1, len(pts), dtype=torch.bool)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    losses = []
    for _ in range(30):
        pred = gi + net(feat) * MOTION_NORM
        loss, data = propagation_loss(pred, tp, tm, mask, shape_hw)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(data.item())
    assert losses[-1] < losses[0] * 0.8

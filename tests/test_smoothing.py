import numpy as np
import torch

from videostab.config import SmoothingConfig
from videostab.smoothing import (DynamicKernelNet, accumulate_path,
                                 crop_budget_project, gaussian_smooth_path,
                                 smooth_path_nn)


def _jitter_path(T=80, gh=4, gw=5, amp=6.0, seed=0):
    rng = np.random.default_rng(seed)
    jit = np.zeros((T, 2), np.float32)
    for t in range(1, T):
        jit[t] = 0.6 * jit[t - 1] + rng.normal(0, amp * 0.5, 2)
    C = np.zeros((T, gh, gw, 2), np.float32)
    C += jit[:, None, None, :]
    return C


def _roughness(P):
    """二阶差分能量: 抖动强度代理."""
    return float(np.mean((P[2:] - 2 * P[1:-1] + P[:-2]) ** 2))


def test_accumulate_path():
    motions = [np.full((2, 3, 2), i + 1, np.float32) for i in range(3)]
    C = accumulate_path(motions)
    assert C.shape == (4, 2, 3, 2)
    assert np.allclose(C[0], 0) and np.allclose(C[3], 6)


def test_gaussian_smoothing_reduces_jitter():
    C = _jitter_path()
    P = gaussian_smooth_path(C, SmoothingConfig())
    assert _roughness(P) < _roughness(C) * 0.1
    assert P.shape == C.shape


def test_adaptive_sigma_preserves_intentional_pan():
    """快速匀速运镜段: 自适应平滑不应把线性路径拉平."""
    T = 60
    C = np.zeros((T, 2, 2, 2), np.float32)
    C[:, ..., 0] = (np.arange(T) * 8.0)[:, None, None]  # 快速横移
    P = gaussian_smooth_path(C, SmoothingConfig())
    # 匀速段本身平滑, 输出应基本不变(端点外)
    assert np.abs(P[10:-10] - C[10:-10]).max() < 1.0


def test_crop_budget_hard_constraint():
    C = _jitter_path(amp=30.0)  # 大抖动, 必然超预算
    P = gaussian_smooth_path(C, SmoothingConfig())
    shape_hw = (240, 320)
    ratio = 0.1
    B = crop_budget_project(C, P, shape_hw, ratio)
    assert np.abs(B[..., 0]).max() <= ratio / 2 * 320 + 1e-4
    assert np.abs(B[..., 1]).max() <= ratio / 2 * 240 + 1e-4


def test_kernel_net_zero_init_is_box_filter():
    """Jacobi 数据锚求解器: 零初始化 = 均匀 w + 固定 λ, 已是良态平滑器."""
    torch.manual_seed(0)
    cfg = SmoothingConfig(radius=8, proxy_hw=(240, 320))
    net = DynamicKernelNet(radius=8).eval()
    C = _jitter_path(T=60)
    P = smooth_path_nn(net, C, cfg=cfg)
    assert P.shape == C.shape
    assert _roughness(P) < _roughness(C) * 0.2


def test_jacobi_has_fixed_point():
    """DUT 数据锚形式的核心性质: 迭代收敛到不动点, 对迭代次数不敏感.

    纯卷积形式没有这个性质(平滑量随迭代无界增长), 是早期实现失败的根因.
    """
    from videostab.smoothing.kernel_net import jacobi_smooth
    C = _jitter_path(T=80)
    t = torch.from_numpy(C.transpose(3, 0, 1, 2))[None]
    offsets = [-3, -2, -1, 1, 2, 3]
    w = torch.ones(1, len(offsets), *t.shape[2:])
    lam = torch.full((1, 1) + t.shape[2:], 1.5)
    P15 = jacobi_smooth(t, w, lam, offsets, iters=15)
    P200 = jacobi_smooth(t, w, lam, offsets, iters=200)
    # 15 次与 200 次结果几乎相同 => 已收敛到不动点
    assert (P15 - P200).abs().max() < 0.05 * C.std()


def test_lambda_monotonically_controls_smoothing():
    """λ 是良态旋钮: 单调控制平滑/保真权衡."""
    from videostab.smoothing.kernel_net import jacobi_smooth
    C = _jitter_path(T=80)
    t = torch.from_numpy(C.transpose(3, 0, 1, 2))[None]
    offsets = [-3, -2, -1, 1, 2, 3]
    w = torch.ones(1, len(offsets), *t.shape[2:])
    roughs = []
    for lv in (0.1, 1.0, 10.0):
        lam = torch.full((1, 1) + t.shape[2:], lv)
        P = jacobi_smooth(t, w, lam, offsets, iters=30)
        roughs.append(_roughness(P[0].numpy().transpose(1, 2, 3, 0)))
    assert roughs[0] > roughs[1] > roughs[2]   # λ 越大越平滑


def test_kernel_net_param_budget():
    net = DynamicKernelNet(radius=30)
    assert sum(p.numel() for p in net.parameters()) < 100_000

"""角点空间求解 (M2): 果冻免疫是结构保证, 必须用测试锁死."""
import numpy as np

from videostab.config import SmoothingConfig
from videostab.eval.metrics import grid_bend_px
from videostab.smoothing.corner import (corner_motions, corner_solve,
                                        field_from_corner_disp)

HW = (360.0, 640.0)
GRID = (12, 16)


def _noisy_motions(T=40, seed=0):
    """带旋转/缩放/平移抖动 + 逐顶点噪声的运动序列 —— 对抗性输入:
    逐顶点噪声在顶点空间会被平滑器揉成果冻, 角点空间必须把它投影掉."""
    rng = np.random.default_rng(seed)
    gh, gw = GRID
    yy, xx = np.meshgrid(np.linspace(0, HW[0], gh),
                         np.linspace(0, HW[1], gw), indexing="ij")
    cx, cy = xx - xx.mean(), yy - yy.mean()
    out = []
    for _ in range(T):
        a, c = rng.normal(0, 0.004), rng.normal(0, 0.004)
        bx, by = rng.normal(0, 3, 2)
        m = np.stack([a * cx - c * cy + bx, c * cx + a * cy + by], -1)
        m += rng.normal(0, 1.5, m.shape)          # 逐顶点非刚性噪声
        out.append(m.astype(np.float32))
    return out


def test_corner_solve_bend_is_structurally_zero():
    """核心恒等式: 不管输入多脏, 输出场逐帧都是单应位移 => 弯曲 ~ 0.

    对照: 同样输入走顶点空间经典链路, 弯曲显著非零. 这两个断言合起来
    就是整个 M2 改造的存在理由.
    """
    from videostab.smoothing.solver import (accumulate_path,
                                            crop_budget_project,
                                            gaussian_smooth_path)
    motions = _noisy_motions()
    cfg = SmoothingConfig()
    B = corner_solve(motions, HW, GRID, cfg)
    assert B.shape == (41, 12, 16, 2)
    assert grid_bend_px(B, HW).max() < 0.05          # 仅剩透视采样残留

    C = accumulate_path(motions)                     # 顶点空间对照
    P = gaussian_smooth_path(C, cfg)
    Bv = crop_budget_project(C, P, HW, cfg.crop_ratio)
    assert grid_bend_px(Bv, HW).max() > 1.0          # 同输入, 顶点空间弯曲显著


def test_corner_solve_respects_crop_budget():
    """预算是产品硬承诺: 角点上钳住 + 单应插值后, 全场不得越界.

    仿射位移场在矩形域的极值必在角点取到; 透视分量留 5% 余量核验.
    """
    cfg = SmoothingConfig()
    motions = [m * 6 for m in _noisy_motions(T=30, seed=1)]  # 大运动逼饱和
    B = corner_solve(motions, HW, GRID, cfg)
    lim_x = cfg.crop_ratio / 2 * HW[1]
    lim_y = cfg.crop_ratio / 2 * HW[0]
    assert np.abs(B[..., 0]).max() <= lim_x * 1.05
    assert np.abs(B[..., 1]).max() <= lim_y * 1.05


def test_corner_motions_project_out_local_noise():
    """投影丢弃非刚性分量: 纯相似运动 + 逐顶点噪声, 角点位移应与
    无噪版本几乎一致(噪声零均值, 192 点最小二乘把它平均掉)."""
    clean = _noisy_motions(T=10, seed=2)
    rng = np.random.default_rng(3)
    noisy = [m + rng.normal(0, 2, m.shape).astype(np.float32) for m in clean]
    cm_clean = corner_motions(clean, HW)
    cm_noisy = corner_motions(noisy, HW)
    diff = max(np.abs(a - b).max() for a, b in zip(cm_clean, cm_noisy))
    assert diff < 2.0                                # << 噪声幅度 2px * 网格数


def test_field_from_corner_disp_exact_at_corners():
    """反解场在 4 角点上必须精确还原给定位移(恰定单应, 无残差)."""
    rng = np.random.default_rng(4)
    Bc = rng.normal(0, 8, (5, 2, 2, 2)).astype(np.float32)
    F = field_from_corner_disp(Bc, HW, GRID)
    # 网格 (0,0)/(0,-1)/(-1,0)/(-1,-1) 顶点恰为 4 角点
    got = np.stack([F[:, 0, 0], F[:, 0, -1], F[:, -1, 0], F[:, -1, -1]], 1)
    want = Bc.reshape(5, 4, 2)
    assert np.abs(got - want).max() < 1e-2

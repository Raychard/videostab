"""训练链路冒烟: 合成视频 -> 特征缓存 -> 两个数据集 -> 一步训练前向/反向."""
import subprocess
import sys
from pathlib import Path

import torch

from conftest import make_shaky_clip, write_video

ROOT = Path(__file__).resolve().parent.parent


def _make_cache(tmp_path):
    vid_dir = tmp_path / "videos"
    vid_dir.mkdir()
    frames, _ = make_shaky_clip(T=40, amp=4.0)
    write_video(frames, str(vid_dir / "clip.mp4"))
    cache_dir = tmp_path / "cache"
    r = subprocess.run(
        [sys.executable, str(ROOT / "train" / "extract_cache.py"),
         "--videos", str(vid_dir), "--out", str(cache_dir)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert list(cache_dir.glob("*.npz"))
    return cache_dir


def test_cache_and_datasets_and_one_step(tmp_path):
    cache_dir = _make_cache(tmp_path)

    from train.dataset import PathWindowDataset, PropagationDataset
    from train.losses import propagation_loss, smoother_loss
    from videostab.propagation.refine_net import (MOTION_NORM,
                                                  ResidualRefineNet)
    from videostab.smoothing.kernel_net import (DynamicKernelNet,
                                                smooth_path_torch)

    # 传播网络: 一步反向传播
    ds = PropagationDataset(str(cache_dir))
    assert len(ds) > 10
    b = ds[0]
    net = ResidualRefineNet()
    pred = (b["grid_init"][None]
            + net(b["feat"][None]) * MOTION_NORM)
    loss, _ = propagation_loss(pred, b["kp"][None], b["motion"][None],
                               b["mask"][None],
                               tuple(b["shape_hw"].tolist()))
    loss.backward()
    assert torch.isfinite(loss)

    # 平滑网络: 一步反向传播
    ds2 = PathWindowDataset(str(cache_dir), window=32)
    C = ds2[0][None]
    net2 = DynamicKernelNet(radius=8)
    P = smooth_path_torch(net2, C, iterations=1)
    loss2, parts = smoother_loss(P, C, (240, 320))
    loss2.backward()
    assert torch.isfinite(loss2)
    assert set(parts) == {"temporal", "freq", "dist", "spatial"}


def test_path_windows_never_cross_segments(tmp_path):
    """人造双段缓存: 窗口采样不得跨 segment 断点."""
    import numpy as np
    from train.dataset import PathWindowDataset
    cache = tmp_path / "cache2"
    cache.mkdir()
    n1, n2, gh, gw = 40, 50, 12, 16
    grid = np.random.default_rng(0).normal(
        0, 2, (n1 + n2, gh, gw, 2)).astype(np.float32)
    seg = np.array([0] * n1 + [1] * n2, np.int64)
    np.savez_compressed(cache / "fake.npz", grid_init=grid, segment=seg,
                        shape_hw=np.array([240, 320]))
    window = 32
    ds = PathWindowDataset(str(cache), window=window)
    # 段1 路径长 41 -> 10 个窗口; 段2 路径长 51 -> 20 个窗口
    assert len(ds) == (n1 + 1 - window + 1) + (n2 + 1 - window + 1)
    for i in range(len(ds)):
        assert ds[i].shape == (2, window, gh, gw)


def test_per_sample_shape_normalization():
    """混合宽高比 batch: sample_field 逐样本归一化 (回归防护)."""
    import numpy as np
    from train.losses import sample_field
    field = torch.zeros(2, 2, 12, 16)
    # 场值 = x 坐标归一化值, 便于校验采样位置
    field[:, 0] = torch.linspace(0, 1, 16).view(1, 1, 16)
    # 两个样本分辨率不同, 同一像素点应映射到不同网格位置
    pts = torch.full((2, 1, 2), 160.0)      # x=160
    shapes = torch.tensor([[240.0, 320.0], [240.0, 640.0]])
    out = sample_field(field, pts, shapes)
    x_narrow = out[0, 0, 0].item()   # 160/319 ≈ 0.50
    x_wide = out[1, 0, 0].item()     # 160/639 ≈ 0.25
    assert abs(x_narrow - 160 / 319) < 0.01
    assert abs(x_wide - 160 / 639) < 0.01

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

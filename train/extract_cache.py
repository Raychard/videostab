#!/usr/bin/env python
"""Pass-0 特征缓存: 视频 -> 逐帧对稀疏运动 + 多单应初始化场, 存 npz.

训练只依赖缓存, 与视频解码解耦; 重训代价 = 只跑网络, 分钟级.
用法: python train/extract_cache.py --videos data/train --out data/cache
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from videostab.config import PipelineConfig  # noqa: E402
from videostab.motion import estimate_sparse_motion  # noqa: E402
from videostab.propagation import propagate_homography  # noqa: E402
from videostab.utils.video_io import VideoReader, to_proxy  # noqa: E402

MAX_KP = 512


def _pad(a: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n,) + a.shape[1:], a.dtype)
    out[: min(len(a), n)] = a[:n]
    return out


def extract(video: Path, out_dir: Path, cfg: PipelineConfig):
    reader = VideoReader(str(video))
    kps, mots, masks, inits, grids = [], [], [], [], []
    prev, shape_hw = None, None
    for frame in reader:
        gray, _ = to_proxy(frame, cfg.proxy_height)
        if prev is not None:
            sm = estimate_sparse_motion(prev, gray, cfg.motion)
            grid, kp_init, info = propagate_homography(
                sm.pts, sm.motions, prev.shape, cfg.propagation)
            n = len(sm.pts)
            if n >= 8 and np.isfinite(info["grid_err"]):
                kps.append(_pad(sm.pts, MAX_KP))
                mots.append(_pad(sm.motions, MAX_KP))
                inits.append(_pad(kp_init, MAX_KP))
                mask = np.zeros(MAX_KP, bool)
                mask[: min(n, MAX_KP)] = True
                masks.append(mask)
                grids.append(grid.astype(np.float32))
        prev, shape_hw = gray, gray.shape
    if not grids:
        print(f"  !! {video.name}: 无有效帧对, 跳过")
        return
    np.savez_compressed(
        out_dir / f"{video.stem}.npz",
        kp=np.stack(kps), motion=np.stack(mots), kp_init=np.stack(inits),
        mask=np.stack(masks), grid_init=np.stack(grids),
        shape_hw=np.array(shape_hw))
    print(f"  -> {video.stem}.npz ({len(grids)} 对)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True)
    p.add_argument("--out", default="data/cache")
    p.add_argument("--proxy-height", type=int, default=480)
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig(proxy_height=args.proxy_height)
    videos = sorted(sum([list(Path(args.videos).glob(f"*.{e}"))
                         for e in ("mp4", "avi", "mov", "MP4")], []))
    if not videos:
        sys.exit(f"未在 {args.videos} 找到视频")
    for v in videos:
        print(f"提取 {v.name}")
        extract(v, out_dir, cfg)


if __name__ == "__main__":
    main()

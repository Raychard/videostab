#!/usr/bin/env python
"""合成抖动数据生成: 对稳定视频施加随机游走单应抖动, 产出不稳定训练视频.

无监督训练只需不稳定视频本身, 合成数据用于: (1) 公开数据下载失败的兜底;
(2) 按场景配比扩充难例. 抖动模型: 平移/旋转/透视的 AR(1) 随机游走.

用法: python scripts/make_synthetic.py --src <稳定视频目录> --out data/train \
        [--per-video 2] [--amp 8.0]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from videostab.utils.video_io import VideoReader, VideoWriter  # noqa: E402


def shake_homography(state: np.ndarray, rng, amp: float, rho=0.85):
    """AR(1) 随机游走状态 -> 3x3 单应. state: [tx,ty,rot,persp_x,persp_y]."""
    noise = rng.normal(0, 1, 5) * np.array(
        [amp, amp, amp * 0.002, amp * 2e-5, amp * 2e-5])
    state[:] = rho * state + noise
    tx, ty, rot, px, py = state
    c, s = np.cos(rot), np.sin(rot)
    H = np.array([[c, -s, tx], [s, c, ty], [px, py, 1.0]], np.float64)
    return H


def synthesize(src: Path, out_dir: Path, per_video: int, amp: float, seed: int):
    reader = VideoReader(str(src))
    w, h = reader.width, reader.height
    for i in range(per_video):
        rng = np.random.default_rng(seed + i)
        state = np.zeros(5)
        out_path = out_dir / f"{src.stem}_shake{i}.mp4"
        with VideoWriter(str(out_path), reader.fps, (w, h)) as writer:
            for frame in reader:
                H = shake_homography(state, rng, amp)
                warped = cv2.warpPerspective(
                    frame, H, (w, h), borderMode=cv2.BORDER_REFLECT)
                writer.write(warped)
        print(f"  -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="稳定视频目录")
    p.add_argument("--out", default="data/train")
    p.add_argument("--per-video", type=int, default=2)
    p.add_argument("--amp", type=float, default=8.0, help="抖动幅度(px)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(sum([list(Path(args.src).glob(f"*.{e}"))
                         for e in ("mp4", "avi", "mov", "MP4")], []))
    if not videos:
        sys.exit(f"未在 {args.src} 找到视频")
    for v in videos:
        print(f"处理 {v.name}")
        synthesize(v, out_dir, args.per_video, args.amp, args.seed)


if __name__ == "__main__":
    main()

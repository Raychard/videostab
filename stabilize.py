#!/usr/bin/env python
"""离线防抖推理入口.

用法:
  python stabilize.py input.mp4 output.mp4
  python stabilize.py input.mp4 output.mp4 --crop 0.12 \
      --refine-weights weights/refine.pt --smoother-weights weights/smoother.pt \
      --metrics
"""
import argparse
import json
import sys

from videostab.config import PipelineConfig
from videostab.pipeline import Stabilizer


def main(argv=None):
    p = argparse.ArgumentParser(description="离线视频防抖 (DUT x NLNL)")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--crop", type=float, default=0.12,
                   help="裁剪预算 c_max (总比例, 硬约束)")
    p.add_argument("--proxy-height", type=int, default=480)
    p.add_argument("--refine-weights", default="")
    p.add_argument("--smoother-weights", default="")
    p.add_argument("--flow", choices=["lk", "raft"], default="lk",
                   help="关键点跟踪: lk=金字塔LK(CPU实时) raft=RAFT质量档(需GPU)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--metrics", action="store_true",
                   help="输出后计算 C/D/S 指标(采样帧)")
    args = p.parse_args(argv)

    cfg = PipelineConfig(proxy_height=args.proxy_height,
                         refine_weights=args.refine_weights,
                         smoother_weights=args.smoother_weights,
                         flow=args.flow, device=args.device)
    cfg.smoothing.crop_ratio = args.crop

    report = Stabilizer(cfg).stabilize(args.input, args.output)

    if args.metrics:
        from videostab.eval.metrics import evaluate
        from videostab.utils.video_io import VideoReader

        def sample(path, stride=5, limit=120):
            frames = []
            for i, f in enumerate(VideoReader(path)):
                if i % stride == 0:
                    frames.append(f)
                if len(frames) >= limit:
                    break
            return frames

        report["metrics"] = {
            k: round(v, 4)
            for k, v in evaluate(sample(args.input),
                                 sample(args.output)).items()
        }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

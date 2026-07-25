#!/usr/bin/env python
"""分场景评测报表: 按类别输出 C/D/S + rough + 降级率 + 多单应平面数.

设计要点(源自本项目踩过的坑):
- **按场景分档汇总**: 均值会掩盖"高抖动大赢、低抖动小输"这类关键信息,
  而这恰恰是判断网络价值的依据.
- **以 rough 为主指标**: NUS stability 测的是低频能量*占比*, 强平滑下
  输出越接近静止总能量越小、占比越被估计噪声主导, 实测曾与 rough 方向
  相反. stability 仅作参考.
- **统计 fallback 率**: 三级降级状态机是产品鲁棒性的真正来源, 但在
  DeepStab Regular 上触发率恒为 0, 等于从未被验证. 低光/弱纹理素材
  才能测到它.
- **统计多单应平面数 K**: 自适应 K + 软融合是相对 DUT 的改进点, 但
  常规手持素材上 K 恒为 1, 该机制未被真正考验. Parallax 类是关键.

用法:
  python scripts/eval_bench.py --data data/eval --out bench.json
  python scripts/eval_bench.py --data data/eval --configs classic full \
      --limit 5 --device cuda
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from videostab.config import PipelineConfig  # noqa: E402
from videostab.eval.metrics import evaluate  # noqa: E402
from videostab.pipeline import Stabilizer  # noqa: E402
from videostab.utils.video_io import VideoReader  # noqa: E402

VIDEO_EXT = (".avi", ".mp4", ".mov", ".mpg", ".m4v")
# 配置名 -> (启用传播网, 启用平滑网)
CONFIGS = {"classic": (0, 0), "refine": (1, 0), "smooth": (0, 1),
           "full": (1, 1)}


def path_roughness(frames) -> float:
    """输出视频的帧间平移路径二阶差分(绝对残余抖动). 主指标."""
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
         for f in frames]
    p = [np.zeros(2)]
    for a, b in zip(g[:-1], g[1:]):
        (dx, dy), _ = cv2.phaseCorrelate(a, b)
        p.append(p[-1] + [dx, dy])
    p = np.array(p)
    return float(np.abs(p[2:] - 2 * p[1:-1] + p[:-2]).mean())


def make_cfg(name, args):
    use_r, use_s = CONFIGS[name]
    c = PipelineConfig(device=args.device, proxy_height=args.proxy_height)
    c.smoothing.crop_ratio = args.crop
    if use_r:
        c.refine_weights = args.refine_weights
    if use_s:
        c.smoother_weights = args.smoother_weights
    return c


def run_one(cfg, src, dst):
    t0 = time.time()
    report = Stabilizer(cfg).stabilize(src, dst)
    report["sec_per_frame"] = (time.time() - t0) / max(report["frames"], 1)
    return report


def fmt(v, nd=3, dash="-"):
    return dash if v is None else f"{v:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/eval",
                   help="根目录, 其下每个子目录为一个场景类别")
    p.add_argument("--out", default="bench.json")
    p.add_argument("--configs", nargs="*", default=["classic", "smooth",
                                                    "full"],
                   choices=list(CONFIGS))
    p.add_argument("--limit", type=int, default=0,
                   help="每类别最多评测多少段(0=全部)")
    p.add_argument("--crop", type=float, default=0.12)
    p.add_argument("--proxy-height", type=int, default=480)
    p.add_argument("--refine-weights", default="weights/refine.pt")
    p.add_argument("--smoother-weights", default="weights/smoother.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--work", default="", help="中间输出目录(默认临时)")
    args = p.parse_args()

    root = Path(args.data)
    cats = sorted([d for d in root.iterdir() if d.is_dir()]) \
        if root.is_dir() else []
    if not cats:
        sys.exit(f"未在 {root} 找到类别子目录; 先跑 scripts/prepare_nus.py")
    work = Path(args.work) if args.work else root / "_stab_out"
    work.mkdir(parents=True, exist_ok=True)

    # 断点续跑: 已有结果直接复用
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for cat in cats:
        vids = sorted([v for v in cat.iterdir()
                       if v.suffix.lower() in VIDEO_EXT])
        if args.limit:
            vids = vids[: args.limit]
        if not vids:
            continue
        print(f"\n### {cat.name}  ({len(vids)} 段)")
        for vid in vids:
            key0 = f"{cat.name}/{vid.name}"
            try:
                orig = list(VideoReader(str(vid)))
            except Exception as e:
                print(f"  跳过 {vid.name}: {e}")
                continue
            r_in = path_roughness(orig)
            results.setdefault(key0, {})["input_rough"] = r_in
            for name in args.configs:
                if name in results[key0]:
                    continue                       # 已算过
                dst = work / f"{cat.name}_{vid.stem}_{name}.avi"
                try:
                    rep = run_one(make_cfg(name, args), str(vid), str(dst))
                    out = list(VideoReader(str(dst)))
                    m = evaluate(orig, out)
                    rep.update(m)
                    rep["rough"] = path_roughness(out)
                    results[key0][name] = rep
                except Exception as e:
                    print(f"  {vid.name} [{name}] 失败: {e}")
                finally:
                    dst.unlink(missing_ok=True)
            out_path.write_text(json.dumps(results, indent=1,
                                           ensure_ascii=False))
            got = results[key0].get(args.configs[0], {})
            print(f"  {vid.name:28} in={r_in:.3f} "
                  f"rough={fmt(got.get('rough'))} "
                  f"L1/L2={fmt(got.get('l1_ratio'),2)}/"
                  f"{fmt(got.get('l2_ratio'),2)} "
                  f"K={fmt(got.get('k_mean'),2)}")

    # ---------- 汇总 ----------
    def agg(keys, name, field):
        vals = [results[k][name][field] for k in keys
                if name in results.get(k, {}) and field in results[k][name]]
        return float(np.mean(vals)) if vals else None

    print("\n" + "=" * 76)
    print("分场景报表 (rough=绝对残余抖动, 主指标; 越小越好)")
    hdr = f"{'类别':14}{'配置':9}{'crop':>7}{'distort':>8}{'stab':>7}" \
          f"{'rough':>8}{'L1%':>6}{'L2%':>6}{'K均值':>7}{'K≥2%':>7}"
    print(hdr)
    print("-" * 76)
    for cat in cats:
        keys = [k for k in results if k.startswith(cat.name + "/")]
        if not keys:
            continue
        rin = agg(keys, args.configs[0], "input_rough")
        ins = [results[k]["input_rough"] for k in keys
               if "input_rough" in results[k]]
        print(f"{cat.name:14}{'(输入)':9}{'':>7}{'':>8}{'':>7}"
              f"{np.mean(ins):8.4f}")
        for name in args.configs:
            print(f"{'':14}{name:9}"
                  f"{fmt(agg(keys,name,'cropping')):>7}"
                  f"{fmt(agg(keys,name,'distortion')):>8}"
                  f"{fmt(agg(keys,name,'stability')):>7}"
                  f"{fmt(agg(keys,name,'rough'),4):>8}"
                  f"{fmt(v*100 if (v:=agg(keys,name,'l1_ratio')) is not None else None,1):>6}"
                  f"{fmt(v*100 if (v:=agg(keys,name,'l2_ratio')) is not None else None,1):>6}"
                  f"{fmt(agg(keys,name,'k_mean'),2):>7}"
                  f"{fmt(v*100 if (v:=agg(keys,name,'k_ge2_ratio')) is not None else None,1):>7}")
        print("-" * 76)

    allk = list(results)
    print(f"{'全部':14}{'':9}")
    for name in args.configs:
        print(f"{'':14}{name:9}"
              f"{fmt(agg(allk,name,'cropping')):>7}"
              f"{fmt(agg(allk,name,'distortion')):>8}"
              f"{fmt(agg(allk,name,'stability')):>7}"
              f"{fmt(agg(allk,name,'rough'),4):>8}"
              f"{fmt(v*100 if (v:=agg(allk,name,'l1_ratio')) is not None else None,1):>6}"
              f"{fmt(v*100 if (v:=agg(allk,name,'l2_ratio')) is not None else None,1):>6}"
              f"{fmt(agg(allk,name,'k_mean'),2):>7}"
              f"{fmt(v*100 if (v:=agg(allk,name,'k_ge2_ratio')) is not None else None,1):>7}")
    spf = agg(allk, args.configs[-1], "sec_per_frame")
    if spf:
        print(f"\n吞吐({args.configs[-1]}): {spf*1000:.0f} ms/帧 @{args.device}")
    print(f"明细已存 {out_path} (重跑本脚本会跳过已算项)")


if __name__ == "__main__":
    main()

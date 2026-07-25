#!/usr/bin/env python
"""解压 NUS 数据集并按类别整理, 供分场景评测使用.

NUS 6 类正对应本项目已知的薄弱环节:
  Parallax      -> 自适应 K 多单应 / 软融合 (DeepStab Regular 上 K 恒为 1,
                   这套机制几乎没被真正考验过)
  Crowd/Running -> 顺序多模型 RANSAC 前景剔除; 传播网 R90 退化的可能来源
  QuickRotation -> 运动自适应 sigma / "保留跟拍意图"
  Zooming       -> 尺度变化下的单应估计
  Regular       -> 常规手持基线

用法: python scripts/prepare_nus.py --raw data/nus_raw --out data/eval
"""
import argparse
import zipfile
from pathlib import Path

VIDEO_EXT = (".avi", ".mp4", ".mov", ".mpg", ".m4v")
CATEGORIES = ("Regular", "QuickRotation", "Zooming", "Parallax",
              "Crowd", "Running")


def extract(zip_path: Path, out_dir: Path) -> int:
    """把 zip 内所有视频平铺解压到 out_dir, 返回数量."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(VIDEO_EXT):
                continue
            if name.startswith("__MACOSX"):
                continue
            dst = out_dir / Path(name).name
            with z.open(name) as src, open(dst, "wb") as f:
                f.write(src.read())
            n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/nus_raw", help="zip 所在目录")
    p.add_argument("--out", default="data/eval", help="按类别输出根目录")
    args = p.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    total = 0
    print(f"{'类别':16} {'状态':>10} {'视频数':>7}")
    for cat in CATEGORIES:
        zp = raw / f"{cat}.zip"
        if not zp.exists():
            print(f"{cat:16} {'缺失':>10} {'-':>7}")
            continue
        try:
            n = extract(zp, out / cat)
            total += n
            print(f"{cat:16} {'OK':>10} {n:7d}")
        except zipfile.BadZipFile:
            print(f"{cat:16} {'损坏':>10} {'-':>7}  (删除后重下)")
    print(f"\n合计 {total} 段视频 -> {out}/<类别>/")
    if total:
        print("分场景评测: python scripts/eval_bench.py --data", out)


if __name__ == "__main__":
    main()

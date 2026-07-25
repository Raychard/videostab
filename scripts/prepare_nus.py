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


def extract(zip_path: Path, out_dir: Path) -> tuple:
    """解压 zip 内的**不稳定输入**视频, 返回 (输入数, 跳过的stb数).

    NUS 每类含两套同名文件: `N.avi` 是不稳定输入, `Nstb.avi` 是原论文
    方法的稳定化输出(参考结果). 必须只取前者 —— 把 stb 当输入等于对
    已稳定视频再做一次防抖, 评测结果毫无意义.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = skipped = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(VIDEO_EXT):
                continue
            if name.startswith("__MACOSX"):
                continue
            stem = Path(name).stem.lower()
            if stem.endswith("stb"):
                skipped += 1
                continue
            dst = out_dir / Path(name).name
            with z.open(name) as src, open(dst, "wb") as f:
                f.write(src.read())
            n += 1
    return n, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/nus_raw", help="zip 所在目录")
    p.add_argument("--out", default="data/eval", help="按类别输出根目录")
    args = p.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    total = 0
    print(f"{'类别':16} {'状态':>10} {'输入':>6} {'跳过stb':>8}")
    for cat in CATEGORIES:
        zp = raw / f"{cat}.zip"
        if not zp.exists() or zp.stat().st_size == 0:
            print(f"{cat:16} {'缺失/未完成':>10} {'-':>6} {'-':>8}")
            continue
        try:
            n, skipped = extract(zp, out / cat)
            total += n
            print(f"{cat:16} {'OK':>10} {n:6d} {skipped:8d}")
        except zipfile.BadZipFile:
            print(f"{cat:16} {'未下完':>10} {'-':>6} {'-':>8}")
    print(f"\n合计 {total} 段不稳定输入 -> {out}/<类别>/")
    if total:
        print("分场景评测: python scripts/eval_bench.py --data", out)


if __name__ == "__main__":
    main()

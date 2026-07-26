#!/usr/bin/env python
"""NUS 数据集分块下载器.

官方服务器 (liushuaicheng.org) 会掐断长连接: 整包 wget/curl 必失败,
但 Range 分块请求正常. 本脚本按块下载 + 断点续传 + 重试.

用法: python scripts/fetch_nus.py [--out data/nus_raw] [--chunk-mb 4]
      python scripts/fetch_nus.py --only Parallax Crowd
"""
import argparse
import http.client
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = "http://liushuaicheng.org/SIGGRAPH2013/data"
REFERER = "http://liushuaicheng.org/SIGGRAPH2013/database.html"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
CATEGORIES = ("Regular", "QuickRotation", "Zooming", "Parallax",
              "Crowd", "Running")


def _req(url, start=None, end=None):
    r = urllib.request.Request(url, headers={"User-Agent": UA,
                                             "Referer": REFERER})
    if start is not None:
        r.add_header("Range", f"bytes={start}-{end}")
    return r


def total_size(url, retries=5):
    """服务器 HEAD 不可靠, 用 Range 探测 Content-Range 拿总长."""
    for i in range(retries):
        try:
            with urllib.request.urlopen(_req(url, 0, 0), timeout=30) as r:
                cr = r.headers.get("Content-Range", "")
                if "/" in cr:
                    return int(cr.rsplit("/", 1)[1])
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def fetch(cat, out_dir: Path, chunk: int, retries: int = 8) -> bool:
    url = f"{BASE}/{cat}.zip"
    dst = out_dir / f"{cat}.zip"
    size = total_size(url)
    if size is None:
        print(f"  {cat}: 无法获取大小, 跳过")
        return False
    done = dst.stat().st_size if dst.exists() else 0
    if done >= size:
        print(f"  {cat}: 已完整 ({size/2**20:.0f} MB)")
        return True
    print(f"  {cat}: {size/2**20:.0f} MB, 从 {done/2**20:.0f} MB 续传")
    with open(dst, "ab") as f:
        stall = 0
        while done < size:
            end = min(done + chunk - 1, size - 1)
            data = b""
            for i in range(retries):
                try:
                    with urllib.request.urlopen(_req(url, done, end),
                                                timeout=60) as r:
                        data = r.read()
                    break
                except http.client.IncompleteRead as e:
                    # 服务器常在分块中途断开; 已读到的部分是有效数据,
                    # 收下并从新偏移继续, 不必整块重来.
                    data = e.partial
                    break
                except Exception as e:
                    if i == retries - 1:
                        print(f"\n  {cat}: 块 {done} 失败 ({e})")
                        return False
                    time.sleep(min(2 ** i, 20))
            if not data:
                stall += 1
                if stall > retries:
                    print(f"\n  {cat}: 在 {done} 处持续无数据, 稍后重跑续传")
                    return False
                time.sleep(min(2 ** stall, 20))
                continue
            stall = 0
            f.write(data)
            f.flush()
            done += len(data)
            print(f"\r    {done/size*100:5.1f}%  {done/2**20:6.0f}/"
                  f"{size/2**20:.0f} MB", end="", flush=True)
    print()
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/nus_raw")
    p.add_argument("--chunk-mb", type=float, default=4.0)
    p.add_argument("--only", nargs="*", default=None,
                   help="只下指定类别, 默认全部")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cats = args.only or list(CATEGORIES)
    ok = []
    for cat in cats:
        if fetch(cat, out, int(args.chunk_mb * 2 ** 20)):
            zp = out / f"{cat}.zip"
            try:
                with zipfile.ZipFile(zp) as z:
                    n = sum(1 for x in z.namelist()
                            if x.lower().endswith((".avi", ".mp4", ".mov")))
                print(f"  {cat}: zip 校验通过, {n} 段视频")
                ok.append(cat)
            except zipfile.BadZipFile:
                print(f"  {cat}: zip 损坏, 删除后重跑本脚本")
                zp.unlink(missing_ok=True)
    print(f"\n完成 {len(ok)}/{len(cats)}: {', '.join(ok) if ok else '无'}")
    print("下一步: python scripts/prepare_nus.py --raw "
          f"{args.out} --out data/eval")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

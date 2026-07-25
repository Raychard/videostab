#!/usr/bin/env bash
# 训练/评测数据下载. 公开数据集链接随年代可能失效, 每项失败不中断;
# 全部失败时用 make_synthetic.py 从任意稳定视频生成训练数据(推荐兜底).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

echo "== [1/2] DeepStab (StabNet 配套, ~7.9GB, 含成对稳定/不稳定视频) =="
echo "   本方案无监督训练只使用其 unstable 一侧"
wget -c -q --show-progress -P data/raw \
    "http://cg.cs.tsinghua.edu.cn/people/~miao/stabnet/data.zip" \
    && unzip -qo data/raw/data.zip -d data/raw/deepstab \
    || echo "!! DeepStab 下载失败(链接可能已失效), 跳过"

echo "== [2/2] NUS 数据集 (Bundled Camera Paths, SIGGRAPH 2013, 评测基准) =="
echo "   官方页面: http://liushuaicheng.org/SIGGRAPH2013/database.html"
echo "   6 类共 ~140 段不稳定视频; 无监督训练与 C/D/S 评测均只需不稳定视频"
mkdir -p data/nus_raw
for cat in Regular QuickRotation Zooming Parallax Crowd Running; do
    # 注意路径是 data/ 而非 database/; Parallax/Crowd 服务端偶发 502,
    # 用 -c 断点续传 + 多次重试
    wget -c -q --show-progress --tries=5 --timeout=120 --waitretry=10 \
        "http://liushuaicheng.org/SIGGRAPH2013/data/${cat}.zip" \
        -O "data/nus_raw/${cat}.zip" \
        || echo "!! NUS/${cat} 下载失败, 稍后重跑本脚本可断点续传"
done
echo "解压: python scripts/prepare_nus.py --raw data/nus_raw --out data/eval"

echo
echo "下载结束. 若全部失败, 用合成数据兜底:"
echo "  python scripts/make_synthetic.py --src <任意稳定视频目录> --out data/train"

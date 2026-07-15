#!/usr/bin/env bash
# 创建虚拟环境并安装依赖。
# 用法:  bash setup_env.sh          # CPU 版 torch(体积小, 适合开发/测试)
#        bash setup_env.sh --cuda   # CUDA 版 torch(训练/GPU 推理)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
VENV=.venv

if [ ! -d "$VENV" ]; then
    # 无 python3-venv 包(ensurepip)的系统上退化为 --without-pip + get-pip 自举
    "$PY" -m venv "$VENV" 2>/dev/null || {
        "$PY" -m venv --without-pip "$VENV"
        curl -sSL https://bootstrap.pypa.io/get-pip.py | "$VENV/bin/python"
    }
fi
source "$VENV/bin/activate"
pip install --upgrade pip

if [ "${1:-}" = "--cuda" ]; then
    pip install torch torchvision
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
pip install numpy opencv-python tqdm pytest

echo "环境就绪: source $VENV/bin/activate"
python -c "import torch, cv2; print('torch', torch.__version__, '| cv2', cv2.__version__, '| cuda', torch.cuda.is_available())"

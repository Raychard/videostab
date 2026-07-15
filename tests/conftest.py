"""测试公共设施: 合成纹理帧与已知抖动路径的合成视频, 全部离线可跑."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_texture(h: int, w: int, seed: int = 0) -> np.ndarray:
    """带角点结构的随机纹理 (uint8 灰度)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w), np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 1.2)
    for _ in range(60):  # 叠加随机矩形制造强角点
        x, y = rng.integers(0, w - 20), rng.integers(0, h - 20)
        ww, hh = rng.integers(8, 20, 2)
        val = int(rng.integers(0, 255))
        cv2.rectangle(img, (x, y), (x + ww, y + hh), val, -1)
    return img


def make_shaky_clip(T: int = 40, size=(240, 320), amp: float = 5.0,
                    seed: int = 0):
    """已知抖动的合成视频. 返回 (frames [BGR], offsets (T,2) 真值路径)."""
    h, w = size
    margin = int(amp * 4) + 8
    canvas = make_texture(h + 2 * margin, w + 2 * margin, seed)
    rng = np.random.default_rng(seed)
    # AR(1) 零均值抖动
    offs = np.zeros((T, 2), np.float32)
    for t in range(1, T):
        offs[t] = 0.7 * offs[t - 1] + rng.normal(0, amp * 0.5, 2)
    offs = np.clip(offs, -margin + 1, margin - 1)
    frames = []
    for t in range(T):
        ox, oy = int(round(offs[t, 0])), int(round(offs[t, 1]))
        crop = canvas[margin + oy : margin + oy + h,
                      margin + ox : margin + ox + w]
        frames.append(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
    return frames, offs


def write_video(frames, path: str, fps: float = 30.0):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened()
    for f in frames:
        vw.write(f)
    vw.release()


@pytest.fixture
def texture():
    return make_texture(240, 320)


@pytest.fixture
def shaky_clip():
    return make_shaky_clip()

import cv2
import numpy as np

from conftest import make_texture
from videostab.preprocess import detect_shots


def _bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_detects_cut_between_different_scenes():
    a = make_texture(120, 160, seed=1)
    # 场景 B: 亮度分布明显不同的另一场景
    b = (make_texture(120, 160, seed=99) * 0.35 + 10).astype(np.uint8)
    frames = [_bgr(a)] * 15 + [_bgr(b)] * 15
    shots = detect_shots(iter(frames))
    assert shots == [(0, 15), (15, 30)]


def test_single_shot_no_false_cut():
    a = make_texture(120, 160, seed=2)
    frames = []
    for t in range(20):  # 同场景轻微亮度变化不应切分
        frames.append(_bgr(np.clip(a.astype(int) + t, 0, 255).astype(np.uint8)))
    shots = detect_shots(iter(frames))
    assert shots == [(0, 20)]


def test_empty_input():
    assert detect_shots(iter([])) == []

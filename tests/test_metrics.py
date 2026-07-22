import cv2
import numpy as np

from conftest import make_shaky_clip, make_texture
from videostab.eval import cropping_ratio, distortion_value, stability_score
from videostab.eval.metrics import _lowfreq_ratio


def test_lowfreq_ratio():
    t = np.arange(200)
    smooth = np.sin(2 * np.pi * 2 * t / 200)          # 纯低频
    rng = np.random.default_rng(0)
    assert _lowfreq_ratio(smooth) > 0.95
    assert _lowfreq_ratio(rng.normal(size=200)) < 0.5  # 白噪声


def test_identity_metrics():
    frames = [cv2.cvtColor(make_texture(240, 320, s), cv2.COLOR_GRAY2BGR)
              for s in range(5)]
    assert cropping_ratio(frames, frames) > 0.98
    assert distortion_value(frames, frames) > 0.95


def test_cropping_detects_zoom():
    frames = [cv2.cvtColor(make_texture(240, 320, s), cv2.COLOR_GRAY2BGR)
              for s in range(5)]
    zoomed = [cv2.resize(f[12:228, 16:304], (320, 240)) for f in frames]
    c = cropping_ratio(frames, zoomed)
    assert 0.85 < c < 0.95  # 10% 裁剪 -> 比值约 0.9


def test_evaluate_matches_individual_metrics():
    """共享单应的 evaluate 应与逐个调用完全一致."""
    from videostab.eval.metrics import evaluate
    frames = [cv2.cvtColor(make_texture(240, 320, s), cv2.COLOR_GRAY2BGR)
              for s in range(6)]
    zoomed = [cv2.resize(f[12:228, 16:304], (320, 240)) for f in frames]
    ev = evaluate(frames, zoomed)
    assert abs(ev["cropping"] - cropping_ratio(frames, zoomed)) < 1e-9
    assert abs(ev["distortion"] - distortion_value(frames, zoomed)) < 1e-9
    assert abs(ev["stability"] - stability_score(zoomed)) < 1e-9


def test_stability_ranks_shaky_below_static():
    shaky, _ = make_shaky_clip(T=60, amp=6.0)
    static, _ = make_shaky_clip(T=60, amp=0.0)
    assert stability_score(static) > stability_score(shaky) + 0.1

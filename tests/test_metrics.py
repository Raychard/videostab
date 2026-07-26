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


def test_camera_path_recovers_known_motion():
    """camera_path 基本契约: 匀速平移下路径应线性、二阶差分近零.

    注: 本项目曾用 cv2.phaseCorrelate 估路径, 在真实大位移素材上会锁错
    相关峰(实测某帧给出 (0,-166)px 与 (-68,+690)px, 相关峰值同时塌到
    0.00~0.02), 把 NUS QuickRotation 整类误判为算法失效. 该失效依赖真实
    视频的多尺度结构与内容变化, **合成纹理无法可靠复现**, 故此处只锁
    基本契约; 真实数据上的三方比对结论记录在 camera_path 的文档字符串.
    """
    from videostab.eval import camera_path, path_roughness
    big = make_texture(300, 900, seed=5)
    frames = [cv2.cvtColor(big[40:280, 20 + 8 * t: 340 + 8 * t],
                           cv2.COLOR_GRAY2BGR) for t in range(20)]
    p = camera_path(frames)
    dx = np.diff(p[:, 0])
    assert np.allclose(dx, -8.0, atol=1.0), f"未还原匀速平移: {dx[:5]}"
    assert path_roughness(frames) < 1.0    # 匀速 => 二阶差分近零


def test_stability_ranks_shaky_below_static():
    shaky, _ = make_shaky_clip(T=60, amp=6.0)
    static, _ = make_shaky_clip(T=60, amp=0.0)
    assert stability_score(static) > stability_score(shaky) + 0.1

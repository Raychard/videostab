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


# ---------------------------------------------------------------- 果冻/弯曲

def _similarity_field(T=6, GH=12, GW=16, h=360.0, w=640.0):
    """纯相似变换校正场: 平移+旋转+等比缩放, 弯曲必须恒为 0."""
    yy, xx = np.meshgrid(np.linspace(0, h, GH), np.linspace(0, w, GW),
                         indexing="ij")
    cx, cy = xx - xx.mean(), yy - yy.mean()
    B = np.zeros((T, GH, GW, 2), np.float32)
    rng = np.random.default_rng(7)
    for t in range(T):
        a, c = rng.normal(0, 0.02), rng.normal(0, 0.02)   # 缩放-1, 旋转
        bx, by = rng.normal(0, 5, 2)
        B[t, ..., 0] = a * cx - c * cy + bx
        B[t, ..., 1] = c * cx + a * cy + by
    return B, (h, w)


def test_grid_bend_zero_on_similarity():
    """相似变换保持直线 -> bend 恒为 0. 该恒等式是角点空间改造的根基:
    只要校正场来自全局相似/仿射, 果冻(直线弯曲)在数学上不可能发生."""
    from videostab.eval.metrics import grid_bend_px, grid_jello
    B, hw = _similarity_field()
    assert grid_bend_px(B, hw).max() < 1e-3
    j = grid_jello(B, hw)
    assert j["bend_p95"] < 1e-3
    assert j["persist_px"] < 1e-3          # 相似场无非相似残差


def test_grid_bend_detects_shear():
    """剪切场必须被检出, 且弯曲量与剪切幅度同量级."""
    from videostab.eval.metrics import grid_bend_px
    B, hw = _similarity_field()
    T, GH, GW, _ = B.shape
    bend0 = grid_bend_px(B, hw).max()
    # 沿行方向加正弦弯曲: 一根横线被弯成 S 形, 峰值 3px
    xs = np.linspace(0, np.pi * 2, GW)
    B2 = B.copy()
    B2[..., 1] += 3.0 * np.sin(xs)[None, None, :]
    bend = grid_bend_px(B2, hw)
    assert bend.min() > 1.0                # 显著非零
    assert bend.max() < 3.0 * 1.5          # 与注入幅度同量级
    assert bend.max() > bend0 * 100


def test_grid_jello_persist_vs_fluct():
    """恒定剪切进 persist, 交替翻转的剪切进 fluct —— 两个分量必须分得开.
    persist 对应人工观察到的"一根直杆一直是弯的"."""
    from videostab.eval.metrics import grid_jello
    B, hw = _similarity_field(T=8)
    xs = np.sin(np.linspace(0, np.pi * 2, B.shape[2]))
    const, alt = B.copy(), B.copy()
    const[..., 1] += 2.0 * xs[None, None, :]              # 时间恒定
    signs = np.array([1, -1] * 4, np.float32)[:, None, None]
    alt[..., 1] += 2.0 * xs[None, None, :] * signs        # 逐帧翻转
    jc, ja = grid_jello(const, hw), grid_jello(alt, hw)
    assert jc["persist_px"] > 3 * jc["fluct_px"]
    assert ja["fluct_px"] > 3 * ja["persist_px"]
    assert jc["persist_px"] > 3 * ja["persist_px"]

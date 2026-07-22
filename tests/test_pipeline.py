"""端到端冒烟测试: 合成抖动视频 -> 防抖 -> 残余抖动应大幅下降.

注: NUS stability 分数对近似静止的输出会被估计噪声主导(总能量极小时
低频占比无意义), 端到端断言用路径二阶差分粗糙度这一直接度量.
"""
import cv2
import numpy as np

from conftest import make_shaky_clip, write_video
from videostab.config import PipelineConfig
from videostab.motion import estimate_sparse_motion
from videostab.pipeline import Stabilizer
from videostab.utils.video_io import VideoReader


def _path_roughness(frames):
    """估计帧间平移路径的二阶差分能量(抖动强度)."""
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    path = [np.zeros(2)]
    for g0, g1 in zip(grays[:-1], grays[1:]):
        sm = estimate_sparse_motion(g0, g1)
        step = (np.median(sm.motions, axis=0) if len(sm.motions)
                else np.zeros(2))
        path.append(path[-1] + step)
    p = np.array(path)
    return float(np.abs(p[2:] - 2 * p[1:-1] + p[:-2]).mean())


def test_end_to_end_reduces_jitter(tmp_path):
    frames, _ = make_shaky_clip(T=50, size=(240, 320), amp=5.0)
    src = str(tmp_path / "shaky.avi")   # MJPG 近无损, 排除编码噪声干扰
    dst = str(tmp_path / "stab.avi")
    write_video(frames, src)

    cfg = PipelineConfig()          # 纯经典模式(无权重), 开箱即用路径
    cfg.smoothing.crop_ratio = 0.15
    report = Stabilizer(cfg).stabilize(src, dst)

    assert report["frames"] == 50
    assert report["l2_ratio"] < 0.1          # 合成纹理不应触发直通
    out_frames = list(VideoReader(dst))
    assert len(out_frames) == 50
    assert out_frames[0].shape == frames[0].shape

    r_in = _path_roughness(frames)
    r_out = _path_roughness(out_frames)
    assert r_out < r_in * 0.5  # 残余抖动至少减半(实际应远低于此)

    # 裁剪预算承诺: 校正量不超预算框
    assert report["max_correction_px"] <= 0.15 / 2 * 320 + 1e-3


def test_report_fields(tmp_path):
    frames, _ = make_shaky_clip(T=30, amp=3.0)
    src = str(tmp_path / "a.mp4")
    dst = str(tmp_path / "b.mp4")
    write_video(frames, src)
    report = Stabilizer(PipelineConfig()).stabilize(src, dst)
    for key in ("frames", "shots", "l1_ratio", "l2_ratio",
                "max_correction_px", "audio"):
        assert key in report
    assert report["shots"] >= 1


def test_multi_shot_video(tmp_path):
    """双镜头视频: 转场应切分, 两段独立防抖, 输出帧数不变."""
    a, _ = make_shaky_clip(T=20, amp=4.0, seed=1)
    b, _ = make_shaky_clip(T=20, amp=4.0, seed=99)
    dark = [(f * 0.35 + 10).astype(np.uint8) for f in b]  # 亮度差异触发切点
    src = str(tmp_path / "cuts.avi")
    dst = str(tmp_path / "cuts_out.avi")
    write_video(a + dark, src)
    report = Stabilizer(PipelineConfig()).stabilize(src, dst)
    assert report["frames"] == 40
    assert report["shots"] == 2
    assert len(list(VideoReader(dst))) == 40


def test_very_short_video(tmp_path):
    """2 帧视频: 不崩溃, 输出帧数一致."""
    frames, _ = make_shaky_clip(T=2, amp=3.0)
    src = str(tmp_path / "short.avi")
    dst = str(tmp_path / "short_out.avi")
    write_video(frames, src)
    report = Stabilizer(PipelineConfig()).stabilize(src, dst)
    assert report["frames"] == 2
    assert len(list(VideoReader(dst))) == 2


def test_portrait_video(tmp_path):
    """竖屏 (320x240 -> 240x320): 网格/预算按宽高各自计算, 不崩溃."""
    frames, _ = make_shaky_clip(T=20, size=(320, 240), amp=4.0)
    src = str(tmp_path / "portrait.avi")
    dst = str(tmp_path / "portrait_out.avi")
    write_video(frames, src)
    report = Stabilizer(PipelineConfig()).stabilize(src, dst)
    out = list(VideoReader(dst))
    assert report["frames"] == 20
    assert out[0].shape == frames[0].shape


def test_smoother_ckpt_roundtrip(tmp_path):
    """新格式 checkpoint: radius 显式存取; 旧格式(裸 state_dict)兼容."""
    import torch
    from videostab.smoothing import DynamicKernelNet
    net = DynamicKernelNet(radius=8)
    new_p = str(tmp_path / "new.pt")
    old_p = str(tmp_path / "old.pt")
    torch.save({"radius": 8, "state_dict": net.state_dict()}, new_p)
    torch.save(net.state_dict(), old_p)
    for p in (new_p, old_p):
        cfg = PipelineConfig(smoother_weights=p)
        assert Stabilizer(cfg).kernel_net.radius == 8

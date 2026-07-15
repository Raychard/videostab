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
                "max_correction_px"):
        assert key in report
    assert report["shots"] >= 1

"""推理可视化诊断测试."""
import os

import numpy as np

from conftest import make_shaky_clip, write_video
from videostab.config import PipelineConfig
from videostab.debug import DebugOptions
from videostab.debug.visualize import (draw_correction_field,
                                       draw_grid_motion, draw_keypoints_flow,
                                       plot_guard_timeline, plot_paths,
                                       stack_grid)
from videostab.guard import GuardLevel
from videostab.pipeline import Stabilizer


# ---------- 可视化函数单测(纯函数, 形状/不崩溃) ----------

def test_draw_keypoints_flow_shape():
    gray = np.full((240, 320), 128, np.uint8)
    pts = np.array([[50, 60], [100, 120]], np.float32)
    mot = np.array([[2, -1], [3, 0]], np.float32)
    rej = np.array([[200, 200]], np.float32)
    rmot = np.array([[9, 9]], np.float32)
    img = draw_keypoints_flow(gray, pts, mot, rej, rmot, "hdr")
    assert img.shape == (240, 320, 3)
    assert img.dtype == np.uint8


def test_draw_grid_motion_and_correction():
    grid = np.random.default_rng(0).normal(0, 2, (12, 16, 2)).astype(np.float32)
    g = draw_grid_motion((240, 320), grid, K=2, grid_err=1.5, level=0)
    assert g.shape == (240, 320, 3)
    c = draw_correction_field((240, 320), grid, 0.12)
    assert c.shape == (240, 320, 3)


def test_plot_paths_and_timeline():
    T = 40
    C = np.cumsum(np.random.default_rng(1).normal(0, 1, (T, 4, 5, 2)),
                  axis=0).astype(np.float32)
    P = C * 0.1
    assert plot_paths(C, P).shape[2] == 3
    levels = [GuardLevel.L0_FULL] * 30 + [GuardLevel.L2_PASSTHROUGH] * 10
    strength = np.linspace(1, 0, T)
    assert plot_guard_timeline(levels, strength).shape[2] == 3


def test_stack_grid_tiles():
    imgs = [np.zeros((50, 60, 3), np.uint8) for _ in range(3)]
    panel = stack_grid(imgs, cols=3)
    assert panel.shape[0] >= 50 and panel.shape[1] >= 3 * 60


# ---------- 端到端: pipeline 产出诊断文件 ----------

def test_pipeline_writes_debug_artifacts(tmp_path):
    frames, _ = make_shaky_clip(T=40, size=(240, 320), amp=5.0)
    src = str(tmp_path / "in.avi")
    dst = str(tmp_path / "out.avi")
    write_video(frames, src)
    dbg_dir = str(tmp_path / "debug")
    opts = DebugOptions(out_dir=dbg_dir, stride=10, max_frames=60)

    report = Stabilizer(PipelineConfig()).stabilize(src, dst, debug=opts)

    assert report["debug_dir"] == dbg_dir
    assert os.path.isfile(os.path.join(dbg_dir, "summary.txt"))
    assert os.path.isfile(os.path.join(dbg_dir, "guard_timeline.png"))
    assert os.path.isfile(os.path.join(dbg_dir, "paths_shot0.png"))
    frame_imgs = os.listdir(os.path.join(dbg_dir, "frames"))
    assert len(frame_imgs) == 4          # T=40, stride=10 -> idx 0,10,20,30
    assert all(f.endswith(".jpg") for f in frame_imgs)


def test_debug_stride_and_max_frames(tmp_path):
    frames, _ = make_shaky_clip(T=60, amp=4.0)
    src = str(tmp_path / "in.avi")
    dst = str(tmp_path / "out.avi")
    write_video(frames, src)
    dbg_dir = str(tmp_path / "d")
    opts = DebugOptions(out_dir=dbg_dir, stride=2, max_frames=5)
    Stabilizer(PipelineConfig()).stabilize(src, dst, debug=opts)
    assert len(os.listdir(os.path.join(dbg_dir, "frames"))) == 5  # 上限生效


def test_no_debug_no_overhead(tmp_path):
    """不传 debug 时不产生任何诊断文件, report 无 debug_dir."""
    frames, _ = make_shaky_clip(T=20, amp=3.0)
    src = str(tmp_path / "in.avi")
    dst = str(tmp_path / "out.avi")
    write_video(frames, src)
    report = Stabilizer(PipelineConfig()).stabilize(src, dst)
    assert "debug_dir" not in report

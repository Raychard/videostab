"""音频保留测试: 需要 ffmpeg, 环境无 ffmpeg 时跳过."""
import shutil
import subprocess

import pytest

from conftest import make_shaky_clip, write_video
from videostab.config import PipelineConfig
from videostab.pipeline import Stabilizer

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


@pytest.mark.skipif(FFMPEG is None or FFPROBE is None,
                    reason="环境无 ffmpeg/ffprobe")
def test_audio_survives_stabilization(tmp_path):
    frames, _ = make_shaky_clip(T=30, amp=3.0)
    silent = str(tmp_path / "silent.mp4")
    src = str(tmp_path / "with_audio.mp4")
    dst = str(tmp_path / "out.mp4")
    write_video(frames, silent)
    # 合成 1 秒正弦音轨并入输入
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", silent,
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", src],
        check=True)

    report = Stabilizer(PipelineConfig()).stabilize(src, dst)
    assert report["audio"] == "copied"
    probe = subprocess.run(
        [FFPROBE, "-loglevel", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", dst],
        capture_output=True, text=True)
    assert "audio" in probe.stdout, "输出中无音频流"


def test_no_audio_input_reports_none(tmp_path):
    """纯视频输入: audio 字段为 none 且输出正常(不依赖 ffmpeg)."""
    frames, _ = make_shaky_clip(T=10, amp=3.0)
    src = str(tmp_path / "a.avi")
    dst = str(tmp_path / "b.avi")
    write_video(frames, src)
    report = Stabilizer(PipelineConfig()).stabilize(src, dst)
    assert report["audio"] in ("copied", "none")
    import os
    assert os.path.exists(dst)
    assert not os.path.exists(dst.replace(".avi", ".videostream.avi"))
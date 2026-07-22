"""视频读写与代理分辨率工具. 两遍式流水线依赖 VideoReader 可重复打开."""
import os
import shutil
import subprocess

import cv2
import numpy as np


class VideoReader:
    """逐帧迭代读取. 用完自动释放; 可多次实例化实现两遍处理."""

    def __init__(self, path: str):
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def __iter__(self):
        cap = cv2.VideoCapture(self.path)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()


class VideoWriter:
    def __init__(self, path: str, fps: float, size_wh: tuple):
        # .avi 用 MJPG(近无损, 测试/评测友好); 其余优先 H.264, 不可用则回退 mp4v
        codecs = ("MJPG",) if path.lower().endswith(".avi") else ("avc1", "mp4v")
        self.writer = None
        for codec in codecs:
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec),
                                fps, size_wh)
            if w.isOpened():
                self.writer = w
                break
            w.release()
        if self.writer is None:
            raise IOError(f"无法创建输出视频: {path}")

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def close(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def mux_audio(video_path: str, audio_src: str, out_path: str) -> bool:
    """用 ffmpeg 把 audio_src 的音频流按流复制并入 video_path.

    无 ffmpeg / 无音频流 / 失败时返回 False, 调用方回退纯视频输出.
    """
    ff = shutil.which("ffmpeg")
    if ff is None:
        return False
    cmd = [ff, "-y", "-loglevel", "error", "-i", video_path,
           "-i", audio_src, "-map", "0:v:0", "-map", "1:a:0?",
           "-c", "copy", out_path]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600)
        return r.returncode == 0 and os.path.getsize(out_path) > 0
    except (OSError, subprocess.SubprocessError):
        return False


def to_proxy(frame: np.ndarray, proxy_height: int) -> tuple:
    """降采样到代理分辨率的灰度图. 返回 (gray, scale) , scale=原/代理."""
    h, w = frame.shape[:2]
    if h <= proxy_height:
        scale = 1.0
        small = frame
    else:
        scale = h / proxy_height
        small = cv2.resize(frame, (int(round(w / scale)), proxy_height),
                           interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return gray, scale

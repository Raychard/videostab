"""视频读写与代理分辨率工具. 两遍式流水线依赖 VideoReader 可重复打开."""
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
        # .avi 用 MJPG(近无损, 测试/评测友好), 其余用 mp4v
        codec = "MJPG" if path.lower().endswith(".avi") else "mp4v"
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(path, fourcc, fps, size_wh)
        if not self.writer.isOpened():
            raise IOError(f"无法创建输出视频: {path}")

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def close(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


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

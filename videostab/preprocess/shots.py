"""转场切分: HSV 直方图帧间距离的简易镜头边界检测.

跨转场平滑轨迹是经典失效源, 各镜头段必须独立防抖.
"""
import cv2
import numpy as np


def _hsv_hist(frame: np.ndarray) -> np.ndarray:
    # H+V 双通道: 灰度内容下 H/S 退化, V 通道保证仍可判别
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 2], None, [16, 16],
                        [0, 180, 0, 256])
    return cv2.normalize(hist, None, norm_type=cv2.NORM_L1).flatten()


def detect_shots(frames, thresh: float = 0.5, min_len: int = 10) -> list:
    """输入帧迭代器(BGR), 返回镜头段 [(start, end), ...), end 为开区间.

    相邻帧直方图巴氏距离超过 thresh 判为切点; 短于 min_len 的段并入前段.
    """
    cuts = [0]
    prev = None
    n = 0
    for n, frame in enumerate(frames):
        h = _hsv_hist(frame)
        if prev is not None:
            d = cv2.compareHist(
                prev.astype(np.float32), h.astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA)
            if d > thresh and n - cuts[-1] >= min_len:
                cuts.append(n)
        prev = h
    total = n + 1 if prev is not None else 0
    if total == 0:
        return []
    cuts.append(total)
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

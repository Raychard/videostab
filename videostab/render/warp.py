"""渲染: 顶点校正场 -> 全分辨率稠密位移 -> remap 反向映射 -> 预算裁剪.

约定: B 为对内容施加的平移场(代理坐标系), 渲染前按 scale 放大到原分辨率.
out(p) = in(p - B(p)) —— 平滑网格场下的标准反向映射近似 (MeshFlow 同款).
"""
import cv2
import numpy as np


def warp_frame(frame: np.ndarray, grid_B: np.ndarray,
               scale: float = 1.0) -> np.ndarray:
    """frame (H,W,3), grid_B (GH,GW,2) 代理坐标校正场, scale=原/代理."""
    h, w = frame.shape[:2]
    dense = cv2.resize(grid_B * scale, (w, h), interpolation=cv2.INTER_LINEAR)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    map_x = xs - dense[..., 0]
    map_y = ys - dense[..., 1]
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def crop_and_resize(frame: np.ndarray, crop_ratio: float) -> np.ndarray:
    """按预算 crop_ratio 对称裁剪并放大回原分辨率(确定性承诺)."""
    h, w = frame.shape[:2]
    my, mx = int(round(h * crop_ratio / 2)), int(round(w * crop_ratio / 2))
    if mx == 0 and my == 0:
        return frame
    cropped = frame[my : h - my, mx : w - mx]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

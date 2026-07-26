"""关键点运动估计: 金字塔 LK(默认, CPU 实时) / RAFT(可选, 质量档).

两条路径统一输出 (pts, motions, valid): 前后向一致性校验产出 valid 掩码.
"""
import cv2
import numpy as np


def track_lk(gray0: np.ndarray, gray1: np.ndarray, pts: np.ndarray,
             win: int = 21, fb_thresh: float = 1.0):
    """LK 前后向跟踪. 返回 (motions (N,2), valid (N,) bool)."""
    if len(pts) == 0:
        return (np.empty((0, 2), np.float32), np.empty((0,), bool),
                np.empty((0,), np.float32))
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    lk = dict(winSize=(win, win), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **lk)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, p1, None, **lk)
    fb_err = np.linalg.norm((p0b - p0).reshape(-1, 2), axis=1)
    valid = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb_err < fb_thresh)
    motions = (p1 - p0).reshape(-1, 2)
    h, w = gray1.shape[:2]
    inside = ((p1[:, 0, 0] >= 0) & (p1[:, 0, 0] < w) &
              (p1[:, 0, 1] >= 0) & (p1[:, 0, 1] < h))
    return (motions.astype(np.float32), valid & inside,
            fb_err.astype(np.float32))


class RaftFlow:
    """torchvision RAFT-Small 封装(懒加载). 双向流 + 一致性校验后在关键点采样."""

    def __init__(self, device: str = "cpu"):
        import torch
        from torchvision.models.optical_flow import (
            raft_small, Raft_Small_Weights)
        self.torch = torch
        self.device = device
        self.model = raft_small(
            weights=Raft_Small_Weights.DEFAULT).to(device).eval()

    def _dense(self, g0, g1):
        torch = self.torch
        def prep(g):
            t = torch.from_numpy(g).float()[None, None] / 127.5 - 1.0
            t = t.repeat(1, 3, 1, 1)
            # RAFT 要求边长为 8 的倍数
            h, w = t.shape[-2:]
            ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
            return torch.nn.functional.pad(t, (0, pw, 0, ph)), (h, w)
        t0, (h, w) = prep(g0)
        t1, _ = prep(g1)
        with torch.no_grad():
            fw = self.model(t0.to(self.device), t1.to(self.device))[-1]
            bw = self.model(t1.to(self.device), t0.to(self.device))[-1]
        return (fw[0, :, :h, :w].cpu().numpy().transpose(1, 2, 0),
                bw[0, :, :h, :w].cpu().numpy().transpose(1, 2, 0))

    @staticmethod
    def _sample(field: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """在任意亚像素位置双线性采样光流场. field (H,W,2), xy (N,2).

        必须用双线性而非整数取整: 取整会给每个关键点引入最多 0.5px 误差,
        而防抖路径的精度正是亚像素级的 —— 实测取整版本会让 RAFT 的
        端到端 rough 劣于 LK, 得出"学习式光流更差"的错误结论.
        """
        h, w = field.shape[:2]
        x = np.clip(xy[:, 0], 0, w - 1.001)
        y = np.clip(xy[:, 1], 0, h - 1.001)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        return (field[y0, x0] * (1 - fx) * (1 - fy)
                + field[y0, x0 + 1] * fx * (1 - fy)
                + field[y0 + 1, x0] * (1 - fx) * fy
                + field[y0 + 1, x0 + 1] * fx * fy)

    def track(self, gray0, gray1, pts, fb_thresh: float = 1.0):
        if len(pts) == 0:
            return (np.empty((0, 2), np.float32), np.empty((0,), bool),
                    np.empty((0,), np.float32))
        fw, bw = self._dense(gray0, gray1)
        motions = self._sample(fw, pts)
        # 前后向一致性: fw(p) + bw(p + fw(p)) 应接近 0
        fb_err = np.linalg.norm(motions + self._sample(bw, pts + motions),
                                axis=1)
        return (motions.astype(np.float32), fb_err < fb_thresh,
                fb_err.astype(np.float32))

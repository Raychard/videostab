"""关键点运动估计: 金字塔 LK(默认, CPU 实时) / RAFT(可选, 质量档).

两条路径统一输出 (pts, motions, valid): 前后向一致性校验产出 valid 掩码.
"""
import cv2
import numpy as np


def track_lk(gray0: np.ndarray, gray1: np.ndarray, pts: np.ndarray,
             win: int = 21, fb_thresh: float = 1.0):
    """LK 前后向跟踪. 返回 (motions (N,2), valid (N,) bool)."""
    if len(pts) == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), bool)
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
    return motions.astype(np.float32), valid & inside


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

    def track(self, gray0, gray1, pts, fb_thresh: float = 1.0):
        if len(pts) == 0:
            return np.empty((0, 2), np.float32), np.empty((0,), bool)
        fw, bw = self._dense(gray0, gray1)
        h, w = gray0.shape[:2]
        xi = np.clip(pts[:, 0].round().astype(int), 0, w - 1)
        yi = np.clip(pts[:, 1].round().astype(int), 0, h - 1)
        motions = fw[yi, xi]
        # 前后向一致性: fw(p) + bw(p + fw(p)) 应接近 0
        x1 = np.clip((pts[:, 0] + motions[:, 0]).round().astype(int), 0, w - 1)
        y1 = np.clip((pts[:, 1] + motions[:, 1]).round().astype(int), 0, h - 1)
        fb_err = np.linalg.norm(motions + bw[y1, x1], axis=1)
        return motions.astype(np.float32), fb_err < fb_thresh

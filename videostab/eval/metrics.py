"""NUS 三指标: Cropping ratio / Distortion / Stability.

注意: 跨论文的 C/D/S 数字不可直接比较, 本实现只用于本仓库内部
版本间与基线间的一致比较.
"""
import cv2
import numpy as np

from ..motion.flow import track_lk


def _pair_homography(gray0: np.ndarray, gray1: np.ndarray):
    """两帧间全局单应 (LK 跟踪 + RANSAC). 失败返回 None."""
    pts = cv2.goodFeaturesToTrack(gray0, 400, 0.01, 8)
    if pts is None or len(pts) < 8:
        return None
    pts = pts.reshape(-1, 2).astype(np.float32)
    motions, valid = track_lk(gray0, gray1, pts, fb_thresh=3.0)
    pts, motions = pts[valid], motions[valid]
    if len(pts) < 8:
        return None
    H, _ = cv2.findHomography(pts, pts + motions, cv2.RANSAC, 3.0)
    return H


def _to_grays(frames):
    return [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            for f in frames]


def cropping_ratio(orig_frames, stab_frames) -> float:
    """对应帧 orig->stab 单应的尺度分量: 放大越多裁剪越大, 1 为无裁剪."""
    ratios = []
    for g0, g1 in zip(_to_grays(orig_frames), _to_grays(stab_frames)):
        H = _pair_homography(g0, g1)
        if H is None:
            continue
        s = np.sqrt(abs(np.linalg.det(H[:2, :2])))
        if s > 0:
            ratios.append(min(1.0, 1.0 / s))
    return float(np.mean(ratios)) if ratios else 0.0


def distortion_value(orig_frames, stab_frames) -> float:
    """orig->stab 单应仿射部分的各向异性 sqrt(λmin/λmax), 取最差帧."""
    worst = 1.0
    got = False
    for g0, g1 in zip(_to_grays(orig_frames), _to_grays(stab_frames)):
        H = _pair_homography(g0, g1)
        if H is None:
            continue
        got = True
        w = np.linalg.eigvalsh(H[:2, :2].T @ H[:2, :2])
        if w[1] > 1e-9:
            worst = min(worst, float(np.sqrt(max(w[0], 0.0) / w[1])))
    return worst if got else 0.0


def _lowfreq_ratio(sig: np.ndarray) -> float:
    """1D 信号 2~6 号频点能量占比(去 DC), NUS 稳定度定义."""
    spec = np.abs(np.fft.rfft(sig - sig.mean())) ** 2
    total = spec[1:].sum()
    if total < 1e-9:
        return 1.0
    return float(spec[1:7].sum() / total)


def stability_score(stab_frames) -> float:
    """稳定视频自身帧间路径(平移 x/y + 旋转)的低频能量占比."""
    grays = _to_grays(stab_frames)
    tx, ty, rot = [0.0], [0.0], [0.0]
    for g0, g1 in zip(grays[:-1], grays[1:]):
        H = _pair_homography(g0, g1)
        if H is None:
            H = np.eye(3)
        tx.append(tx[-1] + H[0, 2])
        ty.append(ty[-1] + H[1, 2])
        rot.append(rot[-1] + np.arctan2(H[1, 0], H[0, 0]))
    return float(np.mean([_lowfreq_ratio(np.array(p)) for p in (tx, ty, rot)]))

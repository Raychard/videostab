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
    motions, valid, _ = track_lk(gray0, gray1, pts, fb_thresh=3.0)
    pts, motions = pts[valid], motions[valid]
    if len(pts) < 8:
        return None
    H, _ = cv2.findHomography(pts, pts + motions, cv2.RANSAC, 3.0)
    return H


def _to_grays(frames):
    return [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            for f in frames]


def _cross_homographies(orig_frames, stab_frames) -> list:
    """对应帧 orig->stab 的逐帧单应列表(失败为 None). C/D 共享此计算."""
    return [_pair_homography(g0, g1) for g0, g1 in
            zip(_to_grays(orig_frames), _to_grays(stab_frames))]


def cropping_ratio(orig_frames, stab_frames, _hs: list = None) -> float:
    """对应帧 orig->stab 单应的尺度分量: 放大越多裁剪越大, 1 为无裁剪."""
    hs = _hs if _hs is not None else _cross_homographies(
        orig_frames, stab_frames)
    ratios = []
    for H in hs:
        if H is None:
            continue
        s = np.sqrt(abs(np.linalg.det(H[:2, :2])))
        if s > 0:
            ratios.append(min(1.0, 1.0 / s))
    return float(np.mean(ratios)) if ratios else 0.0


def distortion_value(orig_frames, stab_frames, _hs: list = None) -> float:
    """orig->stab 单应仿射部分的各向异性 sqrt(λmin/λmax), 取最差帧."""
    hs = _hs if _hs is not None else _cross_homographies(
        orig_frames, stab_frames)
    worst = 1.0
    got = False
    for H in hs:
        if H is None:
            continue
        got = True
        w = np.linalg.eigvalsh(H[:2, :2].T @ H[:2, :2])
        if w[1] > 1e-9:
            worst = min(worst, float(np.sqrt(max(w[0], 0.0) / w[1])))
    return worst if got else 0.0


def evaluate(orig_frames, stab_frames) -> dict:
    """一次性计算 C/D/S + rough, 单应与路径各只算一遍.

    rough(绝对残余抖动)是主指标; stability 测的是低频能量*占比*, 强平滑下
    输出越接近静止越被估计噪声主导, 仅作参考.
    """
    orig = list(orig_frames)
    stab = list(stab_frames)
    hs = _cross_homographies(orig, stab)
    p_in, p_out = camera_path(orig), camera_path(stab)
    return {
        "cropping": cropping_ratio(orig, stab, _hs=hs),
        "distortion": distortion_value(orig, stab, _hs=hs),
        "stability": stability_score(stab, _path=p_out),
        "stability_input": stability_score(orig, _path=p_in),
        "rough": path_roughness(stab, _path=p_out),
        "input_rough": path_roughness(orig, _path=p_in),
    }


def _lowfreq_ratio(sig: np.ndarray) -> float:
    """1D 信号 2~6 号频点能量占比(去 DC), NUS 稳定度定义."""
    spec = np.abs(np.fft.rfft(sig - sig.mean())) ** 2
    total = spec[1:].sum()
    if total < 1e-9:
        return 1.0
    return float(spec[1:7].sum() / total)


def camera_path(frames) -> np.ndarray:
    """累积帧间路径 (T,3) = [tx, ty, rot]. 基于特征跟踪 + RANSAC 单应.

    **不要用 cv2.phaseCorrelate 做这件事**: 它假设纯平移, 在大位移
    (>40px/帧) 叠加裁剪缩放(1/(1-crop)≈1.14x) 时会锁错相关峰, 给出
    非物理的路径跳变(实测某帧二阶差分达 158px, 而实际校正仅 5px).
    该失效只在大运动素材上出现, 曾导致 NUS QuickRotation 整类被误判为
    "算法失效"(相位相关测得恶化 5.2x, 特征跟踪测得实为改善).
    """
    grays = _to_grays(frames)
    path = [np.zeros(3)]
    for g0, g1 in zip(grays[:-1], grays[1:]):
        H = _pair_homography(g0, g1)
        if H is None:
            H = np.eye(3)
        path.append(path[-1] + np.array(
            [H[0, 2], H[1, 2], np.arctan2(H[1, 0], H[0, 0])]))
    return np.array(path)


def path_roughness(frames, _path=None) -> float:
    """路径平移分量的二阶差分均值(px) —— 绝对残余抖动, 越小越好."""
    p = _path if _path is not None else camera_path(frames)
    t = p[:, :2]
    if len(t) < 3:
        return 0.0
    return float(np.abs(t[2:] - 2 * t[1:-1] + t[:-2]).mean())


def stability_score(stab_frames, _path=None) -> float:
    """稳定视频自身帧间路径(平移 x/y + 旋转)的低频能量占比."""
    p = _path if _path is not None else camera_path(stab_frames)
    return float(np.mean([_lowfreq_ratio(p[:, i]) for i in range(3)]))

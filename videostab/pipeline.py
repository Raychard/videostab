"""离线防抖流水线编排 (两遍式).

Pass 1 (流式): 代理灰度 -> 转场检测 + 稀疏运动 + 多单应传播(+残差网络)
               + 守门信号, 只缓存小尺寸网格运动, 不缓存帧.
按镜头段:      路径累积 -> 平滑(动态核网络或经典 fallback) -> 强度调制
               -> 裁剪预算投影.
Pass 2 (流式): 原分辨率 warp + 预算裁剪 + 写出.
"""
import os

import numpy as np
import torch

from .config import PipelineConfig
from .guard import GuardLevel, decide_level, strength_curve
from .motion import estimate_sparse_motion
from .preprocess.shots import _hsv_hist
from .propagation import (ResidualRefineNet, propagate_homography)
from .propagation.refine_net import refine_grid
from .render import crop_and_resize, warp_frame
from .smoothing import (DynamicKernelNet, accumulate_path,
                        crop_budget_project, gaussian_smooth_path,
                        smooth_path_nn)
from .utils.video_io import VideoReader, VideoWriter, mux_audio, to_proxy

import cv2


class Stabilizer:
    def __init__(self, cfg: PipelineConfig = None):
        self.cfg = cfg or PipelineConfig()
        self.refine_net = self._load(ResidualRefineNet, self.cfg.refine_weights)
        self.kernel_net = self._load_smoother(self.cfg.smoother_weights)
        self.tracker = None
        if self.cfg.flow == "raft":
            from .motion.flow import RaftFlow
            self.tracker = RaftFlow(self.cfg.device).track
        self._dbg = None            # 诊断状态; stabilize() 内按需赋值
        self._dbg_records = []
        self._dbg_shots = []

    def _load(self, ctor, path):
        if not path:
            return None
        model = ctor()
        model.load_state_dict(
            torch.load(path, map_location=self.cfg.device, weights_only=True))
        return model.to(self.cfg.device).eval()

    def _load_smoother(self, path):
        if not path:
            return None
        ckpt = torch.load(path, map_location=self.cfg.device,
                          weights_only=True)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:  # 新格式
            radius, sd = ckpt["radius"], ckpt["state_dict"]
        else:  # 旧格式: 从输出核宽 K=2r+1 推断
            sd = ckpt
            radius = (sd["net.4.weight"].shape[0] - 1) // 2
        model = DynamicKernelNet(radius=radius)
        model.load_state_dict(sd)
        return model.to(self.cfg.device).eval()

    # ---------- Pass 1: 分析 ----------
    def _analyze(self, reader: VideoReader):
        cfg = self.cfg
        gh, gw = cfg.propagation.grid_size
        motions, levels, cuts = [], [GuardLevel.L0_FULL], [0]
        prev_gray, prev_hist, shape_hw = None, None, None
        n = 0
        for n, frame in enumerate(reader):
            gray, _ = to_proxy(frame, cfg.proxy_height)
            small = cv2.resize(frame, gray.shape[::-1])
            hist = _hsv_hist(small)
            is_cut = False
            if prev_hist is not None:
                d = cv2.compareHist(prev_hist.astype(np.float32),
                                    hist.astype(np.float32),
                                    cv2.HISTCMP_BHATTACHARYYA)
                is_cut = d > cfg.shot_hist_thresh and n - cuts[-1] >= 10
            if prev_gray is not None:
                if is_cut:
                    cuts.append(n)
                    motions.append(np.zeros((gh, gw, 2), np.float32))
                    levels.append(GuardLevel.L0_FULL)
                else:
                    want = self._dbg_wants(n - 1)
                    m, lvl, dbg = self._pair_motion(prev_gray, gray, want)
                    motions.append(m)
                    levels.append(lvl)
                    if dbg is not None:
                        dbg.update(idx=n - 1, gray0=prev_gray)
                        self._dbg_records.append(dbg)
            prev_gray, prev_hist, shape_hw = gray, hist, gray.shape
        total = n + 1 if prev_gray is not None else 0
        cuts.append(total)
        shots = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
        return motions, levels, shots, shape_hw

    def _dbg_wants(self, pair_idx: int) -> bool:
        """是否为需收集诊断的采样帧对(按 stride 且不超过上限)."""
        d = self._dbg
        return (d is not None and d.save_per_frame
                and pair_idx % d.stride == 0
                and len(self._dbg_records) < d.max_frames)

    def _pair_motion(self, gray0, gray1, want_debug=False):
        """单对帧: 稀疏运动 -> 先判级 -> 按级选传播算法.
        返回 (m, level, dbg): dbg 为中间产物字典或 None.

        判级前置: L1/L2 时跳过多单应传播(白算), L0 传播后仍可被
        grid_err 二次降级到 L1.
        """
        cfg = self.cfg
        gh, gw = cfg.propagation.grid_size
        sm = estimate_sparse_motion(gray0, gray1, cfg.motion,
                                    tracker=self.tracker,
                                    keep_debug=want_debug)
        level = decide_level(sm.signals, cfg.guard)

        def pack(grid, K=None, grid_err=None):
            if not want_debug:
                return None
            return {"sm": sm, "grid": np.asarray(grid), "K": K,
                    "grid_err": grid_err, "level": int(level)}

        if level == GuardLevel.L2_PASSTHROUGH:
            g = np.zeros((gh, gw, 2), np.float32)
            return g, level, pack(g)

        def conservative():
            # 保守模式: 全局中位平移场, 不会产生畸变
            t = (np.median(sm.motions, axis=0) if len(sm.motions)
                 else np.zeros(2, np.float32))
            return np.broadcast_to(t.astype(np.float32), (gh, gw, 2)).copy()

        if level == GuardLevel.L1_CONSERVATIVE:
            g = conservative()
            return g, level, pack(g)
        grid, kp_init, info = propagate_homography(
            sm.pts, sm.motions, gray0.shape, cfg.propagation)
        if info["grid_err"] > cfg.guard.max_grid_err:  # 传播质量二次守门
            g = conservative()
            dbg = pack(g, info["K"], info["grid_err"])
            if dbg is not None:
                dbg["level"] = int(GuardLevel.L1_CONSERVATIVE)
            return g, GuardLevel.L1_CONSERVATIVE, dbg
        if self.refine_net is not None:
            grid = refine_grid(self.refine_net, grid, sm.pts, sm.motions,
                               kp_init, gray0.shape, cfg.device)
        return grid.astype(np.float32), level, pack(
            grid, info["K"], info["grid_err"])

    # ---------- 逐镜头平滑 ----------
    def _solve_shot(self, motions, levels, shape_hw, shot_range=None):
        cfg = self.cfg.smoothing
        C = accumulate_path(motions)
        if self.kernel_net is not None and len(C) > 4:
            P = smooth_path_nn(self.kernel_net, C, cfg.iterations,
                               self.cfg.device)
        else:
            P = gaussian_smooth_path(C, cfg)
        s = strength_curve(levels, self.cfg.guard)
        B = crop_budget_project(C, P, shape_hw, cfg.crop_ratio)
        if self._dbg is not None and self._dbg.save_summary:
            self._dbg_shots.append(
                {"range": shot_range, "C": C, "P": P, "strength": s})
        return B * s[:, None, None, None]

    # ---------- 主入口 ----------
    def stabilize(self, in_path: str, out_path: str, progress=None,
                  debug=None) -> dict:
        """debug: DebugOptions, 非 None 时输出各阶段可视化到 debug.out_dir."""
        self._dbg = debug
        self._dbg_records = []
        self._dbg_shots = []
        reader = VideoReader(in_path)
        motions, levels, shots, shape_hw = self._analyze(reader)
        if shape_hw is None:
            raise ValueError(f"空视频: {in_path}")

        B_all = []
        for s0, s1 in shots:
            shot_motions = motions[s0 : s1 - 1] if s1 - s0 > 1 else []
            shot_levels = levels[s0:s1]
            if shot_motions:
                B = self._solve_shot(shot_motions, shot_levels, shape_hw,
                                     shot_range=(s0, s1))
            else:
                gh, gw = self.cfg.propagation.grid_size
                B = np.zeros((s1 - s0, gh, gw, 2), np.float32)
            B_all.append(B)
        B_all = np.concatenate(B_all, axis=0)

        scale = reader.height / shape_hw[0]
        crop = self.cfg.smoothing.crop_ratio
        root, ext = os.path.splitext(out_path)
        tmp_path = f"{root}.videostream{ext}"  # 先写纯视频流, 再并入音频
        with VideoWriter(tmp_path, reader.fps,
                         (reader.width, reader.height)) as writer:
            for t, frame in enumerate(reader):
                out = warp_frame(frame, B_all[t], scale)
                writer.write(crop_and_resize(out, crop))
                if progress:
                    progress(t)
        audio_copied = mux_audio(tmp_path, in_path, out_path)
        if audio_copied:
            os.remove(tmp_path)
        else:  # 无 ffmpeg 或无音频流: 回退纯视频
            os.replace(tmp_path, out_path)

        lv = np.array([int(l) for l in levels])
        report = {
            "frames": len(lv), "shots": len(shots),
            "l1_ratio": float((lv == 1).mean()),
            "l2_ratio": float((lv == 2).mean()),
            "max_correction_px": float(np.abs(B_all).max()),
            "audio": "copied" if audio_copied else "none",
        }
        if debug is not None:
            from .debug.render import render_debug
            report["debug_dir"] = render_debug(
                debug, self._dbg_records, self._dbg_shots, B_all, levels,
                shape_hw, crop)
        return report

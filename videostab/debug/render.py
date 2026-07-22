"""把 pipeline 收集的各阶段中间产物渲染成诊断图并落盘.

输出目录结构:
  <out_dir>/
    frames/frame_XXXX.jpg   逐帧面板(关键点+光流 | 网格运动 | 校正场)
    paths_shotK.png         每镜头相机路径 C vs P 曲线
    guard_timeline.png      守门等级色带 + 强度曲线
    summary.txt             文字汇总
"""
import os

import cv2
import numpy as np

from .visualize import (draw_correction_field, draw_grid_motion,
                        draw_keypoints_flow, plot_guard_timeline, plot_paths,
                        stack_grid)

LEVEL_NAMES = {0: "L0-full", 1: "L1-conservative", 2: "L2-passthrough"}


def _per_frame_panel(rec, B_frame, shape_hw, crop_ratio):
    sm = rec["sm"]
    header = (f"frame {rec['idx']:04d}  {LEVEL_NAMES.get(rec['level'], '?')}"
              f"  inlier={sm.signals.get('inlier_ratio', 0):.2f}"
              f"  track={sm.signals.get('track_ratio', 0):.2f}")
    a = draw_keypoints_flow(rec["gray0"], sm.pts, sm.motions,
                            sm.rejected_pts, sm.rejected_motions, header)
    b = draw_grid_motion(shape_hw, rec["grid"], rec.get("K"),
                         rec.get("grid_err"), rec.get("level"))
    c = draw_correction_field(shape_hw, B_frame, crop_ratio)
    return stack_grid([a, b, c], cols=3)


def render_debug(opts, records, shots, B_all, levels, shape_hw,
                 crop_ratio) -> str:
    os.makedirs(opts.out_dir, exist_ok=True)

    if opts.save_per_frame and records:
        fdir = os.path.join(opts.out_dir, "frames")
        os.makedirs(fdir, exist_ok=True)
        for rec in records:
            panel = _per_frame_panel(rec, B_all[rec["idx"]], shape_hw,
                                     crop_ratio)
            cv2.imwrite(os.path.join(fdir, f"frame_{rec['idx']:04d}.jpg"),
                        panel)

    if opts.save_summary:
        for k, sh in enumerate(shots):
            if len(sh["C"]) < 2:
                continue
            img = plot_paths(sh["C"], sh["P"])
            cv2.imwrite(os.path.join(opts.out_dir, f"paths_shot{k}.png"), img)
        strength = np.concatenate([sh["strength"] for sh in shots]) \
            if shots else None
        cv2.imwrite(os.path.join(opts.out_dir, "guard_timeline.png"),
                    plot_guard_timeline(levels, strength))
        _write_summary(opts.out_dir, records, shots, B_all, levels, shape_hw)
    return opts.out_dir


def _write_summary(out_dir, records, shots, B_all, levels, shape_hw):
    lv = np.array([int(l) for l in levels])
    lines = [
        "== 防抖诊断汇总 ==",
        f"总帧数: {len(lv)}",
        f"镜头段数: {len(shots)}",
        f"守门分布: L0={int((lv==0).mean()*100)}%  "
        f"L1={int((lv==1).mean()*100)}%  L2={int((lv==2).mean()*100)}%",
        f"最大校正量: {float(np.abs(B_all).max()):.2f}px",
        f"采样诊断帧: {len(records)} 张 (frames/)",
        "",
        "各镜头拟合平面数 K (采样帧, 反映视差复杂度):",
    ]
    by_shot = {}
    for rec in records:
        k = rec.get("K")
        if k is not None:
            by_shot.setdefault("all", []).append(k)
    if by_shot.get("all"):
        ks = np.array(by_shot["all"])
        lines.append(f"  K 分布: 均值={ks.mean():.2f}  "
                     f"max={ks.max()}  (K>=2 占比 {int((ks>=2).mean()*100)}%)")
    lines += [
        "",
        "排查提示:",
        "  - L1/L2 偏多 -> 检查 frames/ 中该帧关键点是否稀疏/被大量剔除",
        "  - K 恒为 1 但画面有明显视差 -> 前景剔除或分裂阈值需调",
        "  - paths 曲线 P 贴合 C -> 平滑过弱; P 远离 C 撞裁剪框 -> 预算不足",
    ]
    with open(os.path.join(out_dir, "summary.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

"""推理可视化选项."""
from dataclasses import dataclass


@dataclass
class DebugOptions:
    out_dir: str               # 可视化结果输出目录
    stride: int = 5            # 每 N 帧渲染一张逐帧诊断图
    max_frames: int = 60       # 逐帧诊断图数量上限(控制内存/磁盘)
    save_per_frame: bool = True  # 是否输出逐帧诊断面板
    save_summary: bool = True    # 是否输出全局路径/守门/强度图

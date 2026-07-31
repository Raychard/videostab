"""角点空间求解: 把平滑从 384 维顶点空间搬进 8 维角点空间.

果冻(直线被弯)的结构根源: 12x16 顶点轨迹**各自独立**平滑, 旋转/缩放下
顶点速率不同, 平滑后相位失配, 合成场不再是合法几何变换 —— 实测传播网
+1.46px 弯曲、平滑网 +76% 持续分量, 且 τ 封顶只能压制不能根除.

本模块的解法(借鉴工程界四角点方案): 每帧运动先投影到全局单应子空间,
只取其对 4 个图像角点的位移(2x2 网格, 8 标量), 平滑与预算投影全部在
这 8 维里进行, 最后逐帧用**恰定**的 4 点单应反解回稠密网格场.

三条结构保证:
- 输出场恒为单应位移场 => 直线弯曲在数学上为零(bend 指标只剩透视采样
  的微小残留), 不依赖任何损失函数或调参;
- 4 角点复用 accumulate_path / gaussian_smooth_path / crop_budget_project
  原实现(它们对网格形状无假设), 语义与顶点空间逐位对齐, A/B 干净;
- 预算在角点上钳住即全帧成立: 仿射位移场在矩形域的极值必在角点取到,
  轻微透视下近似成立 —— 逐顶点钳位(P8 证明它本身制造各向异性)整体消失.

代价: 放弃逐顶点视差补偿. **"受限视差残差层"已实测否决(M4)**: 从顶点解
提取非单应分量、时间高斯低通(σ=4帧)保慢变压高频、τ=2px 硬帽后叠加 ——
Parallax+Crowd 41 段 rough 仅 0.7506->0.7392 (z=+1.09 不显著), 弯曲却从
0.154 弹回 3.295px, distortion 显著变差 (z=-5.15). 结论: 顶点网络在视差
类的优势不住在"慢变非单应残差"里(事后分离拿不回), 更可能来自逐区域平滑
过程本身的自由度. 果冻厌恶前提下该取舍不成立, 方向关闭.
"""
import cv2
import numpy as np

from ..config import SmoothingConfig
from .solver import accumulate_path, crop_budget_project, gaussian_smooth_path


def _corners(shape_hw: tuple) -> np.ndarray:
    """4 角点排成 2x2 网格 (行序: 左上 右上 / 左下 右下), float32 (2,2,2)."""
    h, w = float(shape_hw[0]), float(shape_hw[1])
    return np.array([[[0, 0], [w, 0]], [[0, h], [w, h]]], np.float32)


def corner_motions(motions, shape_hw: tuple) -> list:
    """逐对顶点运动场 -> 全局单应在 4 角点上的位移 [(2,2,2)] * (T-1).

    最小二乘拟合(全部 GHxGW 顶点), 等价于把运动场投影到单应子空间;
    多单应软融合携带的局部视差结构在此被有意丢弃.
    """
    corners = _corners(shape_hw).reshape(4, 2)
    out = []
    for m in motions:
        gh, gw = m.shape[:2]
        yy, xx = np.meshgrid(np.linspace(0, shape_hw[0], gh),
                             np.linspace(0, shape_hw[1], gw), indexing="ij")
        pts = np.stack([xx, yy], -1).reshape(-1, 2).astype(np.float32)
        H, _ = cv2.findHomography(pts, pts + m.reshape(-1, 2), 0)
        if H is None:                     # 退化: 回退为均值平移
            d = np.tile(m.reshape(-1, 2).mean(0), (4, 1))
        else:
            d = cv2.perspectiveTransform(
                corners.reshape(1, 4, 2), H.astype(np.float32)
            )[0] - corners
        out.append(d.reshape(2, 2, 2).astype(np.float32))
    return out


def field_from_corner_disp(B_corner: np.ndarray, shape_hw: tuple,
                           grid_size: tuple) -> np.ndarray:
    """角点位移 (T,2,2,2) -> 稠密网格位移场 (T,GH,GW,2).

    逐帧 4 点恰定单应(getPerspectiveTransform), 再在网格顶点上取值.
    输出场因此逐帧都是单应位移 —— 这就是"弯曲为零"的落点.
    """
    gh, gw = grid_size
    h, w = shape_hw
    corners = _corners(shape_hw).reshape(4, 2)
    yy, xx = np.meshgrid(np.linspace(0, h, gh), np.linspace(0, w, gw),
                         indexing="ij")
    grid = np.stack([xx, yy], -1).reshape(1, -1, 2).astype(np.float32)
    out = np.empty((len(B_corner), gh, gw, 2), np.float32)
    for t, d in enumerate(B_corner):
        G = cv2.getPerspectiveTransform(corners,
                                        corners + d.reshape(4, 2))
        out[t] = (cv2.perspectiveTransform(grid, G.astype(np.float32))[0]
                  - grid[0]).reshape(gh, gw, 2)
    return out


def corner_solve(motions, shape_hw: tuple, grid_size: tuple,
                 cfg: SmoothingConfig, net=None, device: str = "cpu"
                 ) -> np.ndarray:
    """角点空间求解整链: 投影 -> 累积 -> 平滑 -> 预算 -> 反解场.

    与顶点空间求解器共享 accumulate/gaussian/budget 三件套(对网格形状
    无假设), 唯一差异是形状为 (T,2,2,2). 返回 (T,GH,GW,2).

    net: 动态核网络(DynamicKernelNet). 网络栈对网格形状同样无假设
    (Conv3d 3x3 pad1 / 逐点头 / Jacobi 抽头全在时间维), 角点即 2x2
    网格. 预算感知 λ 的 headroom 在角点上恰是最紧的(位移极值点).

    **直接用顶点空间训练的 smoother.pt, 不要为角点重训** —— 已实测否决:
    NUS 144 段, 零样本 rough 0.6072, 角点数据从零重训 0.6521 (z=-5.50
    显著更差), 顶点权重微调 8 轮亦更差 (36 段 z=-2.00). 原因: 角点窗口
    每个只含 4 条轨迹, 顶点窗口 192 条, 网络在丰富的速度统计上学到的
    平滑策略泛化到角点, 反向则不成立.
    """
    cm = corner_motions(motions, shape_hw)
    C = accumulate_path(cm)                       # (T,2,2,2)
    if net is not None and len(C) > 4:
        from dataclasses import replace
        from .kernel_net import smooth_path_nn
        P = smooth_path_nn(net, C, device=device,
                           cfg=replace(cfg, proxy_hw=tuple(shape_hw)))
    else:
        P = gaussian_smooth_path(C, cfg)
    B_corner = crop_budget_project(C, P, shape_hw, cfg.crop_ratio)
    return field_from_corner_disp(B_corner, shape_hw, grid_size)

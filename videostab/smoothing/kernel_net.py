"""动态核平滑网络 (DUT 双向动态核的精简复刻, 离线 ±radius 全窗).

网络输入速度场(路径一阶差分, 归一化), 输出逐帧逐顶点在 ±radius 窗口上的
softmax 核权重; 平滑 = 核权重对路径的加权和, 可迭代多次.
零初始化 head => 初始即均匀盒式滤波, 未训练也能工作(优于随机初始化).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .solver import _adaptive_sigmas  # noqa: F401  (供训练脚本对比用)

VEL_NORM = 8.0  # 速度归一化尺度(px/帧)


class DynamicKernelNet(nn.Module):
    def __init__(self, radius: int = 30, hidden: int = 16):
        super().__init__()
        self.radius = radius
        K = 2 * radius + 1
        self.net = nn.Sequential(
            nn.Conv3d(2, hidden, (5, 3, 3), padding=(2, 1, 1)),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, (5, 3, 3), padding=(2, 1, 1)),
            nn.GELU(),
            nn.Conv3d(hidden, K, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, vel: torch.Tensor) -> torch.Tensor:
        """vel: (B,2,T,GH,GW) 归一化速度 -> 核权重 (B,K,T,GH,GW), K 维已 softmax."""
        return torch.softmax(self.net(vel), dim=1)


def apply_kernels(path: torch.Tensor, weights: torch.Tensor,
                  radius: int) -> torch.Tensor:
    """path (B,2,T,GH,GW) x weights (B,K,T,GH,GW) -> 平滑路径同形状."""
    B, C, T, GH, GW = path.shape
    padded = F.pad(path, (0, 0, 0, 0, radius, radius), mode="replicate")
    out = torch.zeros_like(path)
    for k in range(2 * radius + 1):
        out = out + weights[:, k : k + 1] * padded[:, :, k : k + T]
    return out


def smooth_path_torch(model: DynamicKernelNet, C: torch.Tensor,
                      iterations: int = 3) -> torch.Tensor:
    """可微版本(训练用). C: (B,2,T,GH,GW) -> P 同形状."""
    P = C
    for _ in range(iterations):
        vel = torch.diff(P, dim=2, prepend=P[:, :, :1]) / VEL_NORM
        w = model(vel)
        P = apply_kernels(P, w, model.radius)
    return P


@torch.no_grad()
def smooth_path_nn(model: DynamicKernelNet, C: np.ndarray,
                   iterations: int = 3, device: str = "cpu") -> np.ndarray:
    """推理入口. C (T,GH,GW,2) numpy -> P 同形状."""
    t = torch.from_numpy(C.transpose(3, 0, 1, 2))[None].to(device)  # (1,2,T,GH,GW)
    P = smooth_path_torch(model.to(device), t, iterations)
    return P[0].cpu().numpy().transpose(1, 2, 3, 0).astype(np.float32)

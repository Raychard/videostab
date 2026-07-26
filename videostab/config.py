"""全局配置. 所有模块只从这里读默认参数, 便于产品档位化."""
from dataclasses import dataclass, field


@dataclass
class MotionConfig:
    max_keypoints: int = 512          # 均匀化后保留的关键点上限
    grid_cells: tuple = (8, 12)       # 均匀化用的 (rows, cols) 采样格
    cap_per_cell: int = 8             # 每格保留的最大关键点数
    lk_win: int = 21                  # LK 光流窗口
    fb_thresh: float = 1.0            # 前后向一致性阈值(px)
    conf_scale: float = 0.5           # 置信度换算 exp(-fb_err/scale) 的尺度
    detectors: str = "orb_gftt"       # 关键点组合: orb_gftt | orb_aliked
    ransac_thresh: float = 2.0        # 前景剔除 RANSAC 重投影阈值(px)


@dataclass
class PropagationConfig:
    grid_size: tuple = (12, 16)       # 顶点网格 (GH, GW)
    max_planes: int = 3               # 自适应 K 上限
    split_gain: float = 0.9           # 相对模型选择: K+1 的误差需降到 K 的
                                      # 该倍数以下才接受分裂(不依赖绝对尺度).
                                      # 0.7/0.8/0.9 实测: Parallax 上
                                      # distortion 与 stability 随该值单调
                                      # 改善(0.926->0.936, 0.938->0.944),
                                      # DeepStab 上分裂率虽升到 19% 但指标
                                      # 无变化(误分裂无害)
    min_cluster_frac: float = 0.15    # 簇占比低于此值不再分裂
    soft_sigma_frac: float = 0.15     # 软融合距离核 sigma (相对短边)


@dataclass
class SmoothingConfig:
    radius: int = 30                  # 双向平滑半径(帧), 离线红利
    iterations: int = 3               # 迭代平滑次数
    base_sigma: float = 12.0          # fallback 高斯核基准 sigma
    adapt_v0: float = 6.0             # 运动自适应: 速度衰减常数(px/帧)
    crop_ratio: float = 0.12          # 裁剪预算 c_max (总裁剪比例, 硬约束)
    proxy_hw: tuple = (480, 854)      # 预算感知 λ 用的代理分辨率 (h,w)


@dataclass
class GuardConfig:
    min_kp_l1: int = 128              # 低于则降级 L1
    min_kp_l2: int = 32               # 低于则降级 L2
    min_inlier_ratio: float = 0.5
    max_grid_err: float = 8.0         # 网格重投影残差上限(px)
    ramp_frames: int = 30             # 强度渐变帧数(≈1s)


@dataclass
class PipelineConfig:
    proxy_height: int = 480           # 运动估计代理分辨率
    shot_hist_thresh: float = 0.5     # 转场检测直方图距离阈值
    motion: MotionConfig = field(default_factory=MotionConfig)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    refine_weights: str = ""          # 传播残差网络权重路径, 空=不启用
    smoother_weights: str = ""        # 动态核网络权重路径, 空=用经典 fallback
    flow: str = "lk"                  # 关键点跟踪: "lk" | "raft"(质量档, 需GPU)
    device: str = "cpu"

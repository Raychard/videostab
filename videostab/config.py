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
    # 校正场非相似分量的幅值上限(相对帧高; 5/360 => 360p 下约 5px).
    # 0 = 关闭. 详见 solver.limit_anisotropy.
    #
    # 取值来自 NUS 全量 144 段的 τ 扫描(distortion / rough / 收益代价比):
    #   不处理  0.8679 / 0.5795      τ=2px  0.9432 / 0.6282   1.03
    #   τ=5px   0.9033 / 0.5812  14.4 τ=1px  0.9641 / 0.6856   0.60
    #   τ=3px   0.9245 / 0.6020  1.67
    # 越紧的 τ 换来越多 distortion, 但 rough 代价涨得更快. 选 5px 是因为
    # 它是唯一**统计上免费**的点: 配对符号检验 distortion 117/144 段改善
    # (z=+7.50, 显著), rough 65/79 (z=-1.17, 与噪声无异), stability 持平.
    # 需要更高画质且能接受 stability 代价时可收紧到 3px.
    #
    # 注意两个已被实验推翻的直觉, 勿再重走:
    #  1) 缩样会严重误导 —— 9 段子集上曾显示"三指标全赢", 全量未复现;
    #     36 段子集又低估收益(其 Zooming 基线 0.906 vs 全量 0.796).
    #  2) 有害各向异性是**大尺度低频剪切**, 不是逐顶点噪声 —— 对残差做
    #     空间低通(σ=0.8/1.5)几乎无效(仅 +1.0~1.7% distortion), 只有
    #     限制幅值有效.
    anisotropy_cap_ratio: float = 5.0 / 360
    # 角点空间求解 (M2): 平滑发生在 4 角点(8 维)而非 12x16 顶点(384 维),
    # 输出场逐帧都是全局单应位移 => 直线弯曲在数学上为零, 果冻免疫.
    # 代价是放弃逐顶点视差补偿. False = 现行顶点空间路径.
    corner_space: bool = False


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


# ---------------------------------------------------------------- 产品档位

#: 三档工作点. 三者本质是同一个权衡——**裁掉多少画幅 ↔ 换来多稳**——上的
#: 三个取值, 故以 crop_ratio 为主轴, 大预算档辅以更紧的 τ 补偿画质.
#:
#: 参数全部来自 NUS 144 段全量实测 (full 配置, proxy 360). crop 主轴扫描
#: 0.06/0.08/0.12/0.16/0.20/0.30, 归一化后的关键读数:
#:
#:   crop   cropping  distortion(归一化)  rough
#:   0.06    0.9400        0.9143         0.8109
#:   0.08    0.9211        0.9118         0.7023
#:   0.12    0.8810        0.9083         0.5812
#:   0.16    0.8417        0.8978         0.5474
#:   0.20    0.8025        0.8555         0.5491
#:   0.30    0.7049        0.5611         0.6073
#:
#: 相邻档位配对符号检验定位出**拐点恰在 0.12**: 0.06→0.08 (rough z=+10.67)
#: 与 0.08→0.12 (z=+8.83) 都是 rough 大赢且 distortion 无显著代价; 越过
#: 0.12 后 0.12→0.16 的 rough 收益立刻掉进噪声 (z=+0.83) 而 distortion
#: 开始显著恶化 (z=-2.33). 故 recommended 取 0.12.
#:
#: 注意 rough 对 crop **非单调**: 0.20 起反而变差, 0.30 全面崩坏
#: (rough z=-9.50). "裁得越多越稳"是错的 —— 预算越紧, 逐顶点钳位削去的
#: 量差异越大, 而那本身就是剪切 (详见 solver.limit_anisotropy).
PRESETS = {
    # 保画幅优先. 不取 0.06: 0.06→0.08 是整条曲线性价比最高的一步
    # (rough 136/144 段改善 z=+10.67, distortion 零代价), 只多裁 1.9 个
    # 百分点画幅却换来 rough 改善 13%.
    "minimal_crop": dict(crop_ratio=0.08, anisotropy_cap_ratio=5.0 / 360),
    # 日常默认. 拐点值, 亦是历史出厂值.
    "recommended": dict(crop_ratio=0.12, anisotropy_cap_ratio=5.0 / 360),
    # 高动态素材专用. τ 由 5px 收紧到 4px 是必需的: crop=0.16 在 τ=5px 下
    # 归一化 distortion 只有 0.8978, 显著劣于推荐档的 0.9083; 收紧到 4px
    # 后回到 0.9071 (追平, 差 0.13%) 而 rough 仍优于推荐档 (0.5543 vs
    # 0.5812). τ=3px 则收得过头, rough 退回 0.5717.
    #
    # ⚠ 该档**并非全局更稳**: 相对推荐档的 rough 配对符号检验 z=+0.33,
    # 不显著. 收益高度集中 —— Running -14.5%、Zooming -8.6%, 而 Crowd
    # +4.3%、Regular +1.2% 反而略差. 分布上 74 段改善(均值 -0.0795px)
    # 对 70 段变差(均值 +0.0287px): 赢的幅度是输的 2.8 倍, 净 -0.0269px.
    # 因此它是**高动态素材的选项**, 不是"更好的推荐档".
    "most_stable": dict(crop_ratio=0.16, anisotropy_cap_ratio=4.0 / 360),
}

DEFAULT_PRESET = "recommended"


def preset(name: str = DEFAULT_PRESET, **overrides) -> PipelineConfig:
    """按档位名构造配置. overrides 直接覆盖 PipelineConfig 顶层字段.

    >>> preset().smoothing.crop_ratio
    0.12
    >>> preset("most_stable", device="cuda").smoothing.crop_ratio
    0.16
    """
    if name not in PRESETS:
        raise ValueError(
            f"未知档位 {name!r}; 可选: {', '.join(PRESETS)}")
    cfg = PipelineConfig(**overrides)
    for k, v in PRESETS[name].items():
        setattr(cfg.smoothing, k, v)
    return cfg

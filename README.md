# VideoStab — 离线视频防抖 (DUT 框架 × NLNL 改进)

以 DUT 的无监督网格轨迹框架为骨架、融合 NLNL 鲁棒性改进的**离线、纯视觉、
产品导向**防抖实现。设计方案见 [docs/离线防抖算法落地方案.md](docs/离线防抖算法落地方案.md)，
技术背景见 [docs/视频防抖技术调研报告.md](docs/视频防抖技术调研报告.md) 与
[docs/DUT算法详解与NoLabels-NoLookAhead对比.md](docs/DUT算法详解与NoLabels-NoLookAhead对比.md)。

**开箱即用**：不依赖任何训练权重即可运行（多单应传播 + 自适应高斯平滑的
纯经典路径）；训练两个 <15K 参数的小网络后自动升级为学习增强路径。

## 快速开始

```bash
bash setup_env.sh            # CPU 环境 (开发/测试);  --cuda 装 GPU 版
source .venv/bin/activate

# 推理 (纯经典模式, 无需权重)
python stabilize.py input.mp4 output.mp4 --crop 0.12 --metrics

# 推理 (学习增强模式)
python stabilize.py input.mp4 output.mp4 \
    --refine-weights weights/refine.pt --smoother-weights weights/smoother.pt
```

`--crop` 为裁剪预算硬约束（总裁剪比例），输出保证不出现黑边；`--metrics`
附带输出 NUS 三指标（C/D/S，仅供仓库内部一致比较）。

## 训练 (完全无监督, 只需不稳定视频)

```bash
# 1) 获取数据: 尝试下载 DeepStab / NUS (链接可能失效, 失败不中断)
bash scripts/download_data.sh
#    兜底/扩充: 对任意稳定视频合成抖动
python scripts/make_synthetic.py --src <稳定视频目录> --out data/train

# 2) 提取特征缓存 (之后重训只跑网络, 分钟级)
python train/extract_cache.py --videos data/train --out data/cache

# 3) 训练两个网络
python train/train_propagation.py --cache data/cache --out weights/refine.pt
python train/train_smoother.py    --cache data/cache --out weights/smoother.pt
```

## 测试

```bash
python -m pytest tests/ -q     # 36 个测试, 全部离线合成数据, CPU 可跑
```

含各模块单元测试与端到端冒烟测试（合成抖动视频 → 防抖 → 残余抖动路径
粗糙度需至少减半；实测合成平移抖动下降 ~100%）。

## 架构

```
输入 ─► 预处理(转场切分/代理流) 
     ─► Stage1 运动估计   ORB+GFTT 协同 + 空间均匀化 + LK 前后向校验
                          + RANSAC 前景剔除            [motion/]
     ─► Stage2 运动传播   自适应K多单应 + 软融合
                          + 残差细化网络(可选, ~9K 参数) [propagation/]
     ─► Stage3 轨迹平滑   双向运动自适应平滑(经典 or 动态核网络 ~13K 参数)
                          + 裁剪预算硬约束投影           [smoothing/]
     ─► Stage4 渲染       网格 warp + 预算裁剪           [render/]
     全程: 三级失效降级状态机 (L0 完整/L1 保守/L2 直通)   [guard/]
```

| 目录 | 职责 |
|---|---|
| `videostab/preprocess` | 转场切分 (HSV 直方图) |
| `videostab/motion` | 关键点协同检测、LK/RAFT 跟踪、前景剔除、QG-1 信号 |
| `videostab/propagation` | 多单应软融合初始化、残差细化网络、QG-2 信号 |
| `videostab/smoothing` | 路径累积、自适应高斯/动态核平滑、预算投影 |
| `videostab/render` | 稠密 remap 渲染、预算裁剪 |
| `videostab/guard` | 降级判级与强度渐变曲线 |
| `videostab/eval` | C/D/S 指标 |
| `videostab/pipeline.py` | 两遍式流水线编排 |
| `train/` | 特征缓存、无监督损失、两个训练脚本 |
| `scripts/` | 数据下载、合成抖动数据生成 |

## 与论文实现的关键差异

- 离线设定: 保留 DUT 双向平滑（NLNL 的因果化改动不采纳），采纳其全部
  鲁棒化改动（多检测器协同、软融合多单应、自适应/频域损失）；
- 产品层为论文不覆盖的新增: 三级降级状态机、裁剪预算硬约束、转场切分、
  代理分辨率两遍式流水线；
- 关键点/光流用 ORB+GFTT+LK 精简替代 ALIKE/RAFT（接口已预留,
  `motion/detectors.py::DETECTORS` 与 `estimator` 的 `tracker` 注入点）。

## 已知限制

- 单应+网格先验: 极端视差、大面积动态前景下退化（由 L1/L2 降级兜底）；
- 非全帧: warp 后按预算裁剪（全帧补全为二期高质档）；
- 果冻效应未显式校正（网格 warp 吸收部分）。

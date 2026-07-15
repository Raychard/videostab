"""训练数据集: 读取 extract_cache.py 产出的 npz 缓存."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from videostab.propagation.refine_net import build_refine_input  # noqa: E402


class PropagationDataset(Dataset):
    """样本 = 一个帧对: (5,GH,GW) 输入特征 + 关键点监督信号."""

    def __init__(self, cache_dir: str):
        self.index = []  # (file, pair_idx)
        for f in sorted(Path(cache_dir).glob("*.npz")):
            n = np.load(f)["grid_init"].shape[0]
            self.index += [(f, i) for i in range(n)]
        if not self.index:
            raise RuntimeError(f"缓存目录为空: {cache_dir}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        f, t = self.index[i]
        d = np.load(f)
        mask = d["mask"][t]
        shape_hw = tuple(d["shape_hw"])
        feat = build_refine_input(
            d["grid_init"][t], d["kp"][t][mask], d["motion"][t][mask],
            d["kp_init"][t][mask], shape_hw)
        return {
            "feat": torch.from_numpy(feat),
            "grid_init": torch.from_numpy(
                d["grid_init"][t].transpose(2, 0, 1)),
            "kp": torch.from_numpy(d["kp"][t]),
            "motion": torch.from_numpy(d["motion"][t]),
            "mask": torch.from_numpy(mask),
            "shape_hw": torch.tensor(shape_hw),
        }


class PathWindowDataset(Dataset):
    """样本 = 相机路径的随机时间窗 (2,T,GH,GW), 供平滑网络训练."""

    def __init__(self, cache_dir: str, window: int = 64):
        self.window = window
        self.paths = []
        for f in sorted(Path(cache_dir).glob("*.npz")):
            g = np.load(f)["grid_init"]           # (T-1,GH,GW,2)
            if len(g) + 1 >= window:
                C = np.concatenate(
                    [np.zeros((1,) + g.shape[1:], np.float32),
                     np.cumsum(g, axis=0)]).astype(np.float32)
                self.paths.append(C)
        if not self.paths:
            raise RuntimeError(f"无足够长视频(>= {window} 帧)的缓存: {cache_dir}")

    def __len__(self):
        return sum(len(C) - self.window + 1 for C in self.paths)

    def __getitem__(self, i):
        rng = np.random.default_rng(i)
        C = self.paths[rng.integers(len(self.paths))]
        s = rng.integers(len(C) - self.window + 1)
        win = C[s : s + self.window] - C[s]        # 平移归零
        return torch.from_numpy(win.transpose(3, 0, 1, 2))  # (2,T,GH,GW)

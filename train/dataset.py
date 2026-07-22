"""训练数据集: 读取 extract_cache.py 产出的 npz 缓存.

- 文件级 LRU 缓存: 压缩 npz 逐样本重复解压是训练 IO 瓶颈.
- 路径窗口按 segment(转场/无效对断点)切分, 窗口绝不跨断点,
  避免把镜头跳变当作待平滑的抖动喂给平滑网络.
"""
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from videostab.propagation.refine_net import build_refine_input  # noqa: E402

_CACHE_MAX_FILES = 8


class _NpzCache:
    """文件级 LRU: 每个 npz 只解压一次, 按文件粒度淘汰."""

    def __init__(self, maxsize: int = _CACHE_MAX_FILES):
        self.maxsize = maxsize
        self._data = OrderedDict()

    def get(self, path: Path) -> dict:
        key = str(path)
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}
        self._data[key] = d
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)
        return d


def _segments(d: dict) -> np.ndarray:
    """(P,) 每对帧的 segment id; 兼容无 segment 字段的旧缓存(视为单段)."""
    n = d["grid_init"].shape[0]
    return d.get("segment", np.zeros(n, np.int64))


class PropagationDataset(Dataset):
    """样本 = 一个帧对: (5,GH,GW) 输入特征 + 关键点监督信号."""

    def __init__(self, cache_dir: str):
        self.cache = _NpzCache()
        self.index = []  # (file, pair_idx)
        for f in sorted(Path(cache_dir).glob("*.npz")):
            n = self.cache.get(f)["grid_init"].shape[0]
            self.index += [(f, i) for i in range(n)]
        if not self.index:
            raise RuntimeError(f"缓存目录为空: {cache_dir}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        f, t = self.index[i]
        d = self.cache.get(f)
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
    """样本 = 相机路径的时间窗 (2,T,GH,GW). 窗口不跨 segment 断点,
    索引到窗口的映射确定且均匀."""

    def __init__(self, cache_dir: str, window: int = 64):
        self.window = window
        self.paths = []    # 每个连续段的路径 C (L+1,GH,GW,2)
        self.samples = []  # (path_idx, start)
        for f in sorted(Path(cache_dir).glob("*.npz")):
            with np.load(f) as z:
                d = {k: z[k] for k in z.files}
            g, seg = d["grid_init"], _segments(d)
            for sid in np.unique(seg):
                run = g[seg == sid]                    # 段内连续帧对
                if len(run) + 1 < window:
                    continue
                C = np.concatenate(
                    [np.zeros((1,) + run.shape[1:], np.float32),
                     np.cumsum(run, axis=0)]).astype(np.float32)
                pi = len(self.paths)
                self.paths.append(C)
                self.samples += [(pi, s)
                                 for s in range(len(C) - window + 1)]
        if not self.samples:
            raise RuntimeError(
                f"无长度 >= {window} 的连续段缓存: {cache_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        pi, s = self.samples[i]
        win = self.paths[pi][s : s + self.window]
        win = win - win[0]                             # 平移归零
        return torch.from_numpy(win.transpose(3, 0, 1, 2))  # (2,T,GH,GW)

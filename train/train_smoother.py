#!/usr/bin/env python
"""动态核平滑网络无监督训练.

用法: python train/train_smoother.py --cache data/cache \
        --out weights/smoother.pt [--window 64] [--radius 30]
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.dataset import PathWindowDataset  # noqa: E402
from train.losses import smoother_loss  # noqa: E402
from videostab.smoothing.kernel_net import (  # noqa: E402
    DynamicKernelNet, smooth_path_torch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", default="weights/smoother.pt")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--radius", type=int, default=30)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--crop", type=float, default=0.12)
    p.add_argument("--proxy-height", type=int, default=480)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    args = p.parse_args()

    ds = PathWindowDataset(args.cache, args.window)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True)
    model = DynamicKernelNet(radius=args.radius).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(len(dl), 1))
    shape_hw = (args.proxy_height, int(args.proxy_height * 16 / 9))

    print(f"窗口样本 {len(ds)} | 参数 "
          f"{sum(x.numel() for x in model.parameters())} | {args.device}")
    for ep in range(args.epochs):
        tot, nb, parts_acc = 0.0, 0, {}
        for C in dl:
            C = C.to(args.device)
            P = smooth_path_torch(model, C)
            loss, parts = smoother_loss(P, C, shape_hw, args.crop)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
            for k, v in parts.items():
                parts_acc[k] = parts_acc.get(k, 0.0) + v
        detail = "  ".join(f"{k}={v / nb:.4f}" for k, v in parts_acc.items())
        print(f"epoch {ep + 1}/{args.epochs}  loss={tot / nb:.4f}  {detail}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"已保存 {args.out}")


if __name__ == "__main__":
    main()

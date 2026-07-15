#!/usr/bin/env python
"""传播残差网络无监督训练.

用法: python train/train_propagation.py --cache data/cache \
        --out weights/refine.pt [--epochs 50] [--bs 64] [--lr 2e-4]
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.dataset import PropagationDataset  # noqa: E402
from train.losses import propagation_loss  # noqa: E402
from videostab.propagation.refine_net import (  # noqa: E402
    MOTION_NORM, ResidualRefineNet)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", default="weights/refine.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    args = p.parse_args()

    ds = PropagationDataset(args.cache)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=2,
                    drop_last=True)
    model = ResidualRefineNet().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(len(dl), 1))

    print(f"样本 {len(ds)} 对 | 参数 "
          f"{sum(x.numel() for x in model.parameters())} | {args.device}")
    for ep in range(args.epochs):
        tot, data_l, nb = 0.0, 0.0, 0
        for batch in dl:
            feat = batch["feat"].to(args.device)
            grid_init = batch["grid_init"].to(args.device)
            pred = grid_init + model(feat) * MOTION_NORM
            shape_hw = tuple(batch["shape_hw"][0].tolist())
            loss, dl_ = propagation_loss(
                pred, batch["kp"].to(args.device),
                batch["motion"].to(args.device),
                batch["mask"].to(args.device), shape_hw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            data_l += dl_.item()
            nb += 1
        print(f"epoch {ep + 1}/{args.epochs}  loss={tot / nb:.4f}  "
              f"kp_err={data_l / nb:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"已保存 {args.out}")


if __name__ == "__main__":
    main()

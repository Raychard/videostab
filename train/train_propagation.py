#!/usr/bin/env python
"""传播残差网络无监督训练 (DUT 点云式关键点注意力 + DUT L_MR 损失).

用法: python train/train_propagation.py --cache data/cache \
        --out weights/refine.pt [--epochs 60] [--bs 32] [--lr 2e-4]
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
    MOTION_NORM, ResidualRefineNet, grid_vertex_batch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", default="weights/refine.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--k-neighbors", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    args = p.parse_args()

    ds = PropagationDataset(args.cache)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=2,
                    drop_last=True)
    model = ResidualRefineNet(k_neighbors=args.k_neighbors).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(len(dl), 1))

    print(f"样本 {len(ds)} 对 | 参数 "
          f"{sum(x.numel() for x in model.parameters())} | {args.device}")
    for ep in range(args.epochs):
        tot, err_acc, nb = 0.0, 0.0, 0
        for batch in dl:
            dev = args.device
            grid_init = batch["grid_init"].to(dev)          # (B,2,GH,GW)
            kp = batch["kp"].to(dev)
            motion = batch["motion"].to(dev)
            kp_init = batch["kp_init"].to(dev)
            mask = batch["mask"].to(dev)
            shape_hw = batch["shape_hw"].to(dev)
            B, _, GH, GW = grid_init.shape

            conf = batch["conf"].to(dev)
            verts = grid_vertex_batch(shape_hw, (GH, GW))
            delta = model(verts, kp, motion - kp_init, mask, conf)  # (B,V,2)
            pred = grid_init + delta.transpose(1, 2).reshape(
                B, 2, GH, GW) * MOTION_NORM

            loss, err = propagation_loss(pred, verts, kp, motion, mask,
                                         shape_hw, conf=conf,
                                         k_neighbors=args.k_neighbors)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            err_acc += err.item()
            nb += 1
        print(f"epoch {ep + 1}/{args.epochs}  loss={tot / nb:.4f}  "
              f"kp_err={err_acc / nb:.4f}px")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"k_neighbors": args.k_neighbors,
                "state_dict": model.state_dict()}, args.out)
    print(f"已保存 {args.out}")


if __name__ == "__main__":
    main()

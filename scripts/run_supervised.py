"""
Supervised sample-efficiency experiment: HW-Net vs BaselineCNN at multiple
labeled-data sizes.

Replicates the multi-seed result from our synthetic experiments. With real
data, the same script tests whether biological priors give sample efficiency
advantages on natural images.

Usage:
  python scripts/run_supervised.py --source synthetic_10class
  python scripts/run_supervised.py --source cifar10 --image_size 32
  python scripts/run_supervised.py --source stl10 --image_size 64
  python scripts/run_supervised.py --source imagefolder --data_root /path/to/images
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from hwnet_jepa.data import load_dataset, make_balanced_subset
from hwnet_jepa.networks import HWNet, BaselineCNN, count_params
from hwnet_jepa.train import supervised_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="synthetic_10class",
                        choices=["synthetic_10class", "cifar10", "cifar100",
                                 "stl10", "imagefolder"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--n_per_class_values", type=int, nargs="+",
                        default=[5, 10, 20, 50, 100])
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--n_per_class_val", type=int, default=200)
    parser.add_argument("--n_per_class_train_pool", type=int, default=None,
                        help="Cap training pool per class (None = use all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--configs", type=str, nargs="+",
                        default=["HWNet_default", "HWNet_fast_frontend", "BaselineCNN"])
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Device: {args.device}")
    print(f"Source: {args.source}")
    print(f"Image size: {args.image_size}")

    # Load dataset
    pretrain_images, train_x, train_y, val_x, val_y, n_classes = load_dataset(
        args.source,
        data_root=args.data_root, image_size=args.image_size,
        n_per_class_train=args.n_per_class_train_pool,
        n_per_class_val=args.n_per_class_val)
    print(f"Train pool: {train_x.shape}, Val: {val_x.shape}, n_classes: {n_classes}")

    seeds = list(range(args.seeds))
    out_path = Path(args.out_dir) / f"supervised_{args.source}.json"
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
    else:
        results = {"source": args.source, "image_size": args.image_size,
                   "n_classes": n_classes,
                   "n_per_class_values": args.n_per_class_values,
                   "seeds": seeds, "models": {}}

    config_specs = {
        "HWNet_default": {"cls": HWNet, "frontend_lr_mult": 1.0},
        "HWNet_fast_frontend": {"cls": HWNet, "frontend_lr_mult": 5.0},
        "HWNet_endstopped": {"cls": HWNet, "frontend_lr_mult": 1.0,
                             "kwargs": {"use_end_stopped": True}},
        "BaselineCNN": {"cls": BaselineCNN, "frontend_lr_mult": 1.0},
    }

    for cfg_name in args.configs:
        if cfg_name not in config_specs:
            print(f"Unknown config {cfg_name}, skipping")
            continue
        spec = config_specs[cfg_name]
        ModelCls = spec["cls"]
        kwargs = spec.get("kwargs", {})
        print(f"\n--- {cfg_name} ---")
        if cfg_name not in results["models"]:
            results["models"][cfg_name] = []
        done_npcs = {r["n_per_class"]: r for r in results["models"][cfg_name]}
        cfg_results = list(results["models"][cfg_name])
        for npc in args.n_per_class_values:
            if npc in done_npcs:
                print(f"  npc={npc} already done -> {done_npcs[npc]['mean_acc']:.3f}")
                continue
            seed_accs = []; t0 = time.time()
            for seed in seeds:
                torch.manual_seed(seed * 1000 + hash(cfg_name) % 999)
                np.random.seed(seed * 1000 + hash(cfg_name) % 999)
                tx, ty = make_balanced_subset(train_x, train_y, npc, n_classes,
                                              seed=seed)
                model = ModelCls(num_classes=n_classes, **kwargs)
                # Adaptive epochs: more for smaller datasets
                n_ep = 25 if npc <= 10 else (15 if npc <= 50 else 10)
                acc = supervised_train(
                    model, tx, ty, val_x, val_y,
                    n_epochs=n_ep, batch_size=64, lr=3e-3,
                    frontend_lr_mult=spec["frontend_lr_mult"],
                    device=args.device)
                seed_accs.append(acc)
            ma = float(np.mean(seed_accs)); sa = float(np.std(seed_accs))
            print(f"  npc={npc:3d}  acc={ma:.3f} +/- {sa:.3f}  ({time.time()-t0:.0f}s)")
            cfg_results.append({"n_per_class": npc, "n_train": npc * n_classes,
                                "mean_acc": ma, "std_acc": sa,
                                "seed_accs": seed_accs})
            results["models"][cfg_name] = cfg_results
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    # Summary table
    print(f"\n{'='*78}\nSupervised summary ({args.source}, n_classes={n_classes})")
    print('='*78)
    header = f"{'n/cls':>6} {'n_total':>8}"
    for cfg in args.configs:
        if cfg in results["models"]:
            header += f"  {cfg:>22}"
    print(header)
    print("-" * len(header))
    for npc in args.n_per_class_values:
        row = f"{npc:>6d} {npc*n_classes:>8d}"
        for cfg in args.configs:
            if cfg not in results["models"]:
                continue
            r = next((x for x in results["models"][cfg]
                      if x["n_per_class"] == npc), None)
            if r:
                row += f"  {r['mean_acc']:.3f} +/- {r['std_acc']:.3f}    "
            else:
                row += f"  {'--':>22}"
        print(row)


if __name__ == "__main__":
    main()

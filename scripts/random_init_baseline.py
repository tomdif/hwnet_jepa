"""
Random-init probe baseline.

Runs the same linear / attn-pool / kNN probes that run_jepa.py runs in eval,
but on a fresh, *un-pretrained* encoder of the same architecture. This isolates
the contribution of pretraining: lift = pretrained - random_init.

Usage (mirror the architecture flags from your run_jepa.py invocation):
  python scripts/random_init_baseline.py \\
      --source hf --hf_dataset jxie/stl10 --image_size 64 --patch_size 8 \\
      --n_orientations 16 --n_scales 4 \\
      --embed_dim 384 --encoder_layers 12 --encoder_n_heads 6 \\
      --predictor_layers 4 --predictor_n_heads 6 \\
      --n_per_class_values 10 50 200 500 \\
      --readouts linear attn_pool knn \\
      --seeds 4 --tag _v5_random
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from hwnet_jepa.data import load_dataset, make_balanced_subset
from hwnet_jepa.jepa import IJEPA
from hwnet_jepa.train import linear_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="synthetic_10class")
    parser.add_argument("--hf_dataset", type=str, default="jxie/stl10")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="./results")

    # Architecture (must match the main run for apples-to-apples)
    parser.add_argument("--n_orientations", type=int, default=8)
    parser.add_argument("--n_scales", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--encoder_layers", type=int, default=3)
    parser.add_argument("--predictor_layers", type=int, default=2)
    parser.add_argument("--encoder_n_heads", type=int, default=4)
    parser.add_argument("--predictor_n_heads", type=int, default=3)
    parser.add_argument("--use_end_stopped", action="store_true")

    # Eval
    parser.add_argument("--n_per_class_values", type=int, nargs="+",
                        default=[5, 10, 20, 50])
    parser.add_argument("--readouts", type=str, nargs="+",
                        default=["linear", "attn_pool"])
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--n_per_class_val", type=int, default=200)

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", type=str, default="_random_init")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Device: {args.device}")
    print(f"Source: {args.source}, image size: {args.image_size}")

    extra_kwargs = {"hf_dataset": args.hf_dataset} if args.source == "hf" else {}
    _, train_x, train_y, val_x, val_y, n_classes = load_dataset(
        args.source, data_root=args.data_root, image_size=args.image_size,
        n_per_class_train=None, n_per_class_val=args.n_per_class_val,
        n_pretrain_max=1,  # we don't need a pretrain pool here
        **extra_kwargs)
    print(f"Train pool: {train_x.shape}, Val: {val_x.shape}, n_classes: {n_classes}")

    grid_size = args.image_size // args.patch_size
    print(f"Grid size: {grid_size}x{grid_size}")

    # Build a fresh, *un-pretrained* IJEPA. We only use its context_encoder.
    torch.manual_seed(0); np.random.seed(0)
    model = IJEPA(n_orientations=args.n_orientations, n_scales=args.n_scales,
                  patch_size=args.patch_size, embed_dim=args.embed_dim,
                  encoder_layers=args.encoder_layers,
                  predictor_layers=args.predictor_layers,
                  encoder_n_heads=args.encoder_n_heads,
                  predictor_n_heads=args.predictor_n_heads,
                  grid_size=grid_size,
                  use_end_stopped=args.use_end_stopped)
    n_params = sum(p.numel() for p in model.context_encoder.parameters())
    print(f"  encoder params: {n_params}")

    seeds = list(range(args.seeds))
    out_path = Path(args.out_dir) / f"random_init_results_{args.source}{args.tag}.json"
    results = {
        "source": args.source, "image_size": args.image_size,
        "n_classes": n_classes,
        "n_per_class_values": args.n_per_class_values, "seeds": seeds,
        "args": vars(args),
        "configs": {},
    }

    for readout in args.readouts:
        key = f"{readout}_random_init"
        print(f"\n--- {key} ---")
        cfg_results = []
        for npc in args.n_per_class_values:
            seed_accs = []; t0 = time.time()
            for seed in seeds:
                # Per-seed fresh encoder so init noise is part of the variance
                torch.manual_seed(seed * 7919 + hash(readout) % 999)
                np.random.seed(seed * 7919 + hash(readout) % 999)
                fresh = IJEPA(n_orientations=args.n_orientations,
                              n_scales=args.n_scales,
                              patch_size=args.patch_size,
                              embed_dim=args.embed_dim,
                              encoder_layers=args.encoder_layers,
                              predictor_layers=args.predictor_layers,
                              encoder_n_heads=args.encoder_n_heads,
                              predictor_n_heads=args.predictor_n_heads,
                              grid_size=grid_size,
                              use_end_stopped=args.use_end_stopped)
                enc = copy.deepcopy(fresh.context_encoder)
                tx, ty = make_balanced_subset(train_x, train_y, npc,
                                              n_classes, seed=seed)
                n_ep = 30 if npc <= 10 else 25
                acc = linear_probe(enc, tx, ty, val_x, val_y,
                                   num_classes=n_classes, readout=readout,
                                   n_epochs=n_ep, lr=3e-3,
                                   freeze_encoder=True, device=args.device)
                seed_accs.append(acc)
            ma = float(np.mean(seed_accs)); sa = float(np.std(seed_accs))
            print(f"  npc={npc:3d}  acc={ma:.3f} +/- {sa:.3f}  ({time.time()-t0:.0f}s)")
            cfg_results.append({"n_per_class": npc, "n_train": npc * n_classes,
                                "mean_acc": ma, "std_acc": sa,
                                "seed_accs": seed_accs})
            results["configs"][key] = cfg_results
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*78}\nRandom-init baseline summary ({args.source})\n{'='*78}")
    keys = list(results["configs"].keys())
    header = f"{'n/cls':>6} {'n_total':>8}"
    for k in keys:
        header += f"  {k:>22}"
    print(header)
    for npc in args.n_per_class_values:
        row = f"{npc:>6d} {npc*n_classes:>8d}"
        for k in keys:
            r = next((x for x in results["configs"][k]
                      if x["n_per_class"] == npc), None)
            if r:
                row += f"  {r['mean_acc']:.3f} +/- {r['std_acc']:.3f}    "
            else:
                row += f"  {'--':>22}"
        print(row)


if __name__ == "__main__":
    main()

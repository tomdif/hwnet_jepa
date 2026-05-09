"""
JEPA pretraining + linear-probe sample efficiency experiment.

Workflow:
  1. Load dataset (synthetic or real).
  2. JEPA-pretrain on the unlabeled pretraining pool.
  3. Evaluate frozen encoder via linear/attn-pool/kNN readouts at multiple
     low-data sample sizes.
  4. Optionally fine-tune the encoder at higher sample sizes.

Usage:
  # Synthetic (CPU-friendly, the original experiment)
  python scripts/run_jepa.py --source synthetic_10class

  # CIFAR-10 (the real-data version)
  python scripts/run_jepa.py --source cifar10 --image_size 32 \\
      --pretrain_epochs 30 --use_augmentation

  # STL-10 (designed for SSL, has 100K unlabeled images)
  python scripts/run_jepa.py --source stl10 --image_size 64 \\
      --pretrain_epochs 50 --use_augmentation --use_block_masks
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
from hwnet_jepa.networks import count_params
from hwnet_jepa.train import jepa_pretrain, linear_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="synthetic_10class")
    parser.add_argument("--hf_dataset", type=str, default="jxie/stl10",
                        help="HF Hub repo id when --source hf")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="./results")

    # Pretraining
    parser.add_argument("--n_pretrain_max", type=int, default=10000)
    parser.add_argument("--pretrain_epochs", type=int, default=15)
    parser.add_argument("--pretrain_batch_size", type=int, default=64)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--use_block_masks", action="store_true")
    parser.add_argument("--ema_init", type=float, default=0.95)
    parser.add_argument("--ema_final", type=float, default=0.999)
    parser.add_argument("--var_loss_weight", type=float, default=1.0,
                        help="Weight on VICReg-style variance loss (collapse prevention)")
    parser.add_argument("--checkpoint_every", type=int, default=0,
                        help="Save encoder every N epochs during pretraining (0 = off)")

    # Architecture
    parser.add_argument("--n_orientations", type=int, default=8)
    parser.add_argument("--n_scales", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--encoder_layers", type=int, default=3)
    parser.add_argument("--predictor_layers", type=int, default=2)
    parser.add_argument("--encoder_n_heads", type=int, default=4)
    parser.add_argument("--predictor_n_heads", type=int, default=3)
    parser.add_argument("--use_end_stopped", action="store_true")

    # Evaluation
    parser.add_argument("--n_per_class_values", type=int, nargs="+",
                        default=[5, 10, 20, 50])
    parser.add_argument("--readouts", type=str, nargs="+",
                        default=["linear", "attn_pool"])
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--n_per_class_val", type=int, default=200)
    parser.add_argument("--n_per_class_train_pool", type=int, default=None)
    parser.add_argument("--fine_tune_at", type=int, nargs="*", default=[],
                        help="Sample sizes at which to also evaluate fine-tuning")

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix on the output filename")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Device: {args.device}")
    print(f"Source: {args.source}, image size: {args.image_size}")

    # Load dataset
    extra_kwargs = {"hf_dataset": args.hf_dataset} if args.source == "hf" else {}
    pretrain_images, train_x, train_y, val_x, val_y, n_classes = load_dataset(
        args.source, data_root=args.data_root, image_size=args.image_size,
        n_per_class_train=args.n_per_class_train_pool,
        n_per_class_val=args.n_per_class_val,
        n_pretrain_max=args.n_pretrain_max,
        **extra_kwargs)
    print(f"Pretrain pool: {pretrain_images.shape}")
    print(f"Train pool: {train_x.shape}, Val: {val_x.shape}, n_classes: {n_classes}")

    # Compute grid_size based on image size and patch size
    grid_size = args.image_size // args.patch_size
    print(f"Grid size: {grid_size}x{grid_size} = {grid_size*grid_size} patches")

    # Build and pretrain JEPA
    print("\n[1] JEPA pretraining...")
    torch.manual_seed(0); np.random.seed(0)
    model = IJEPA(n_orientations=args.n_orientations, n_scales=args.n_scales,
                  patch_size=args.patch_size, embed_dim=args.embed_dim,
                  encoder_layers=args.encoder_layers,
                  predictor_layers=args.predictor_layers,
                  encoder_n_heads=args.encoder_n_heads,
                  predictor_n_heads=args.predictor_n_heads,
                  grid_size=grid_size, ema_decay=args.ema_init,
                  use_end_stopped=args.use_end_stopped)
    n_params = count_params(model)
    print(f"  trainable params: {n_params}")

    encoder_filename = f"jepa_encoder_{args.source}{args.tag}.pt"
    encoder_path = Path(args.out_dir) / encoder_filename
    history_path = Path(args.out_dir) / f"jepa_pretrain_history_{args.source}{args.tag}.json"

    if encoder_path.exists():
        print(f"  loading pretrained encoder from {encoder_path}")
        model.context_encoder.load_state_dict(
            torch.load(encoder_path, map_location=args.device))
        with open(history_path) as f:
            history = json.load(f)
    else:
        t0 = time.time()
        model, history = jepa_pretrain(
            model, pretrain_images,
            n_epochs=args.pretrain_epochs,
            batch_size=args.pretrain_batch_size,
            lr=args.pretrain_lr,
            use_augmentation=args.use_augmentation,
            use_block_masks=args.use_block_masks,
            ema_init=args.ema_init, ema_final=args.ema_final,
            var_loss_weight=args.var_loss_weight,
            checkpoint_path=str(encoder_path),
            checkpoint_every=args.checkpoint_every,
            device=args.device)
        print(f"  pretraining took {time.time()-t0:.0f}s")
        torch.save(model.context_encoder.state_dict(), encoder_path)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    # Linear probe at each sample size
    print("\n[2] Linear probe evaluation...")
    seeds = list(range(args.seeds))
    out_path = Path(args.out_dir) / f"jepa_results_{args.source}{args.tag}.json"
    results = {
        "source": args.source, "image_size": args.image_size,
        "n_classes": n_classes,
        "n_per_class_values": args.n_per_class_values, "seeds": seeds,
        "args": vars(args),
        "pretrain_history": history,
        "configs": {},
    }

    for readout in args.readouts:
        for freeze in [True, False]:
            if not freeze and not args.fine_tune_at:
                continue
            key = f"{readout}_{'frozen' if freeze else 'finetuned'}"
            print(f"\n--- {key} ---")
            cfg_results = []
            for npc in args.n_per_class_values:
                if not freeze and npc not in args.fine_tune_at:
                    continue
                seed_accs = []; t0 = time.time()
                for seed in seeds:
                    enc = copy.deepcopy(model.context_encoder)
                    torch.manual_seed(seed * 1000 + hash(key) % 999)
                    np.random.seed(seed * 1000 + hash(key) % 999)
                    tx, ty = make_balanced_subset(train_x, train_y, npc,
                                                  n_classes, seed=seed)
                    n_ep = 30 if npc <= 10 else 25
                    lr = 3e-3 if freeze else 5e-4
                    acc = linear_probe(enc, tx, ty, val_x, val_y,
                                        num_classes=n_classes, readout=readout,
                                        n_epochs=n_ep, lr=lr,
                                        freeze_encoder=freeze, device=args.device)
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
    print(f"\n{'='*78}\nJEPA summary ({args.source})\n{'='*78}")
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

"""
Transfer experiment.

Question: do JEPA-pretrained features generalize to a *different* downstream task,
or did the encoder overfit to the pretraining/eval distribution?

Workflow:
  1. Load a JEPA encoder pretrained on dataset A (e.g., synthetic_10class or cifar10).
  2. Evaluate it via linear probe on dataset B (e.g., synthetic_transfer or cifar100).
  3. Compare against:
     - Random encoder (same architecture, no pretraining): controls for arch
     - Supervised encoder pretrained on A, frontend frozen, new head: tests
       whether supervised features transfer (typically: poorly at low data)
     - End-to-end training on B (no pretraining): the from-scratch baseline

Usage:
  # The synthetic baseline experiment we ran during development
  python scripts/run_transfer.py \\
      --pretrain_source synthetic_10class \\
      --transfer_source synthetic_transfer

  # CIFAR-10 -> CIFAR-100 transfer (real data)
  python scripts/run_transfer.py \\
      --pretrain_source cifar10 --transfer_source cifar100

  # CIFAR-10 -> custom dataset
  python scripts/run_transfer.py \\
      --pretrain_source cifar10 \\
      --transfer_source imagefolder \\
      --transfer_data_root /path/to/your/dataset
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from hwnet_jepa.data import load_dataset, make_balanced_subset
from hwnet_jepa.networks import HWNet
from hwnet_jepa.jepa import IJEPA
from hwnet_jepa.train import supervised_train, linear_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_source", type=str, required=True)
    parser.add_argument("--transfer_source", type=str, required=True)
    parser.add_argument("--pretrain_data_root", type=str, default="./data")
    parser.add_argument("--transfer_data_root", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--encoder_path", type=str, default=None,
                        help="Pretrained JEPA encoder. If absent, runs JEPA pretraining first.")
    parser.add_argument("--sup_frontend_path", type=str, default=None,
                        help="Pretrained supervised HWNet frontend.")

    # Architecture (must match pretrained encoder)
    parser.add_argument("--n_orientations", type=int, default=8)
    parser.add_argument("--n_scales", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--encoder_layers", type=int, default=3)

    # Eval
    parser.add_argument("--n_per_class_values", type=int, nargs="+",
                        default=[5, 10, 25])
    parser.add_argument("--readout", type=str, default="attn_pool",
                        choices=["linear", "attn_pool", "knn"])
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--n_per_class_val", type=int, default=200)
    parser.add_argument("--n_per_class_train_pool", type=int, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    grid_size = args.image_size // args.patch_size

    # Need to (1) train/load supervised HWNet on pretrain source and (2) load JEPA
    # pretrained on pretrain source. Both go on the transfer source for eval.

    # 1. Pretraining-source supervised HWNet (for the "supervised features transfer"
    #    condition)
    print("Setting up supervised HWNet on pretraining source...")
    sup_path = args.sup_frontend_path or str(
        Path(args.out_dir) / f"sup_hwnet_frontend_{args.pretrain_source}.pt")
    if not Path(sup_path).exists():
        print(f"  training supervised HWNet on {args.pretrain_source}...")
        _, ptx, pty, pvx, pvy, pn_classes = load_dataset(
            args.pretrain_source, data_root=args.pretrain_data_root,
            image_size=args.image_size,
            n_per_class_train=args.n_per_class_train_pool,
            n_per_class_val=args.n_per_class_val)
        torch.manual_seed(0); np.random.seed(0)
        sup = HWNet(num_classes=pn_classes)
        a = supervised_train(sup, ptx, pty, pvx, pvy,
                              n_epochs=10, lr=3e-3, device=args.device)
        print(f"  acc on pretraining task: {a:.3f}")
        torch.save(sup.frontend.state_dict(), sup_path)
    else:
        print(f"  using existing {sup_path}")

    # 2. JEPA encoder on pretraining source
    print("\nSetting up JEPA encoder...")
    if args.encoder_path is None:
        encoder_path = str(Path(args.out_dir) /
                            f"jepa_encoder_{args.pretrain_source}.pt")
    else:
        encoder_path = args.encoder_path
    if not Path(encoder_path).exists():
        raise FileNotFoundError(
            f"JEPA encoder not found at {encoder_path}.\n"
            f"Run: python scripts/run_jepa.py --source {args.pretrain_source} first.")

    torch.manual_seed(0)
    jepa = IJEPA(n_orientations=args.n_orientations, n_scales=args.n_scales,
                  patch_size=args.patch_size, embed_dim=args.embed_dim,
                  encoder_layers=args.encoder_layers, grid_size=grid_size)
    jepa.context_encoder.load_state_dict(
        torch.load(encoder_path, map_location=args.device))
    print(f"  loaded {encoder_path}")

    # Random encoder control
    torch.manual_seed(999)
    rand = IJEPA(n_orientations=args.n_orientations, n_scales=args.n_scales,
                  patch_size=args.patch_size, embed_dim=args.embed_dim,
                  encoder_layers=args.encoder_layers, grid_size=grid_size)

    # 3. Load transfer-source data
    print(f"\nLoading transfer source: {args.transfer_source}")
    _, tx_full, ty_full, vx, vy, n_classes_t = load_dataset(
        args.transfer_source, data_root=args.transfer_data_root,
        image_size=args.image_size,
        n_per_class_train=args.n_per_class_train_pool,
        n_per_class_val=args.n_per_class_val)
    print(f"  Transfer train pool: {tx_full.shape}, val: {vx.shape}, n_classes: {n_classes_t}")

    seeds = list(range(args.seeds))
    out_path = Path(args.out_dir) / f"transfer_{args.pretrain_source}_to_{args.transfer_source}.json"
    results = {
        "pretrain_source": args.pretrain_source,
        "transfer_source": args.transfer_source,
        "image_size": args.image_size,
        "n_classes": n_classes_t,
        "n_per_class_values": args.n_per_class_values,
        "seeds": seeds,
        "args": vars(args),
        "conditions": {},
    }

    def run_cfg(name, runner):
        print(f"\n--- {name} ---")
        cfg = []
        for npc in args.n_per_class_values:
            sa_list = []; t0 = time.time()
            for seed in seeds:
                torch.manual_seed(seed * 1000 + hash(name) % 999)
                np.random.seed(seed * 1000 + hash(name) % 999)
                tx, ty = make_balanced_subset(tx_full, ty_full, npc,
                                              n_classes_t, seed=seed)
                acc = runner(tx, ty, vx, vy, npc)
                sa_list.append(acc)
            ma = float(np.mean(sa_list)); sd = float(np.std(sa_list))
            print(f"  npc={npc:3d}  acc={ma:.3f} +/- {sd:.3f}  ({time.time()-t0:.0f}s)")
            cfg.append({"n_per_class": npc, "n_train": npc * n_classes_t,
                        "mean_acc": ma, "std_acc": sd, "seed_accs": sa_list})
            results["conditions"][name] = cfg
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    # JEPA-pretrained, frozen, with chosen readout
    def run_jepa(tx, ty, vx, vy, npc):
        enc = copy.deepcopy(jepa.context_encoder)
        return linear_probe(enc, tx, ty, vx, vy, num_classes=n_classes_t,
                            readout=args.readout,
                            n_epochs=30 if npc <= 10 else 25,
                            freeze_encoder=True, device=args.device)
    run_cfg(f"jepa_{args.readout}", run_jepa)

    # Random encoder (same arch, no pretraining)
    def run_rand(tx, ty, vx, vy, npc):
        enc = copy.deepcopy(rand.context_encoder)
        return linear_probe(enc, tx, ty, vx, vy, num_classes=n_classes_t,
                            readout=args.readout,
                            n_epochs=30 if npc <= 10 else 25,
                            freeze_encoder=True, device=args.device)
    run_cfg(f"random_{args.readout}", run_rand)

    # Supervised HWNet frontend frozen + new head
    def run_sup(tx, ty, vx, vy, npc):
        m = HWNet(num_classes=n_classes_t)
        m.frontend.load_state_dict(torch.load(sup_path, map_location=args.device))
        for p in m.frontend.parameters():
            p.requires_grad = False
        return supervised_train(m, tx, ty, vx, vy,
                                 n_epochs=20 if npc <= 10 else 15,
                                 lr=3e-3, device=args.device)
    run_cfg("sup_hwnet_frozen", run_sup)

    # End-to-end HWNet on transfer (from scratch)
    def run_e2e(tx, ty, vx, vy, npc):
        m = HWNet(num_classes=n_classes_t)
        return supervised_train(m, tx, ty, vx, vy,
                                 n_epochs=20 if npc <= 10 else 15,
                                 lr=3e-3, device=args.device)
    run_cfg("hwnet_e2e", run_e2e)

    # Summary
    print(f"\n{'='*78}\nTransfer summary: {args.pretrain_source} -> {args.transfer_source}")
    print('='*78)
    keys = list(results["conditions"].keys())
    header = f"{'n/cls':>6} {'n_total':>8}"
    for k in keys:
        header += f"  {k:>22}"
    print(header)
    for npc in args.n_per_class_values:
        row = f"{npc:>6d} {npc*n_classes_t:>8d}"
        for k in keys:
            r = next((x for x in results["conditions"][k]
                      if x["n_per_class"] == npc), None)
            if r:
                row += f"  {r['mean_acc']:.3f} +/- {r['std_acc']:.3f}    "
            else:
                row += f"  {'--':>22}"
        print(row)


if __name__ == "__main__":
    main()

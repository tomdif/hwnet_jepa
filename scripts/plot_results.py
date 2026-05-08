"""
Plot sample-efficiency curves comparing supervised baselines vs JEPA.

Usage:
  python scripts/plot_results.py \\
      --supervised_results results/supervised_synthetic_10class.json \\
      --jepa_results results/jepa_results_synthetic_10class.json \\
      --out results/sample_efficiency_curves.png
"""
import argparse
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervised_results", type=str, default=None)
    parser.add_argument("--jepa_results", type=str, default=None)
    parser.add_argument("--transfer_results", type=str, default=None)
    parser.add_argument("--out", type=str, default="./results/curves.png")
    parser.add_argument("--title", type=str, default="Sample efficiency")
    args = parser.parse_args()

    if not HAS_MPL:
        print("matplotlib not available; skipping plotting")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    if args.supervised_results and Path(args.supervised_results).exists():
        with open(args.supervised_results) as f:
            sup = json.load(f)
        for cfg, runs in sup.get("models", {}).items():
            if not runs:
                continue
            ns = [r["n_train"] for r in runs]
            means = [r["mean_acc"] for r in runs]
            stds = [r["std_acc"] for r in runs]
            ax.errorbar(ns, means, yerr=stds, marker="o", label=f"sup: {cfg}",
                        capsize=3, alpha=0.7)

    if args.jepa_results and Path(args.jepa_results).exists():
        with open(args.jepa_results) as f:
            jp = json.load(f)
        for cfg, runs in jp.get("configs", {}).items():
            if not runs:
                continue
            ns = [r["n_train"] for r in runs]
            means = [r["mean_acc"] for r in runs]
            stds = [r["std_acc"] for r in runs]
            ax.errorbar(ns, means, yerr=stds, marker="s", label=f"jepa: {cfg}",
                        capsize=3, alpha=0.7, linewidth=2)

    if args.transfer_results and Path(args.transfer_results).exists():
        with open(args.transfer_results) as f:
            tr = json.load(f)
        for cfg, runs in tr.get("conditions", {}).items():
            if not runs:
                continue
            ns = [r["n_train"] for r in runs]
            means = [r["mean_acc"] for r in runs]
            stds = [r["std_acc"] for r in runs]
            ax.errorbar(ns, means, yerr=stds, marker="^", label=f"transfer: {cfg}",
                        capsize=3, alpha=0.7, linestyle="--")

    ax.set_xscale("log")
    ax.set_xlabel("Training examples (log scale)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(args.out, dpi=100)
    plt.close()
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()

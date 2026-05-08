"""
Olshausen-Field probe: train a sparse autoencoder on whitened natural-image
patches and visualize the learned filters. Tests whether natural-image
statistics produce Gabor-like receptive fields, providing empirical support
for the V1 frontend design.

Usage:
  python scripts/sparse_coding_probe.py --source synthetic_10class
  python scripts/sparse_coding_probe.py --source cifar10 --image_size 32
  python scripts/sparse_coding_probe.py --source stl10 --image_size 64
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from hwnet_jepa.data import load_dataset
from hwnet_jepa.networks import SparseAutoencoder


def extract_whitened_patches(images: torch.Tensor, patch_size: int = 12,
                              n_patches_per_image: int = 20):
    """Olshausen-Field-style: grayscale, DC-remove, PCA-whiten, normalize."""
    if images.shape[1] == 3:
        lum = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
    else:
        lum = images[:, 0]
    lum = lum.unsqueeze(1)
    N, _, H, W = lum.shape
    all_patches = []
    for _ in range(n_patches_per_image):
        ys = torch.randint(0, H - patch_size + 1, (N,))
        xs = torch.randint(0, W - patch_size + 1, (N,))
        for i in range(N):
            p = lum[i, 0, ys[i]:ys[i] + patch_size, xs[i]:xs[i] + patch_size]
            all_patches.append(p.flatten())
    patches = torch.stack(all_patches, dim=0)
    patches = patches - patches.mean(dim=1, keepdim=True)
    var = patches.var(dim=1)
    keep = var > var.median() * 0.3
    patches = patches[keep]
    cov = (patches.t() @ patches) / patches.shape[0]
    cov = cov + 1e-3 * torch.eye(cov.shape[0])
    eigvals, eigvecs = torch.linalg.eigh(cov)
    whitening = (eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals.clamp_min(1e-6)))
                 @ eigvecs.t())
    whitened = patches @ whitening
    whitened = whitened / (whitened.std(dim=0, keepdim=True) + 1e-6)
    return whitened


def train_sae(patches, n_features=64, patch_size=12, n_epochs=50,
              batch_size=256, lr=1e-2, l1_weight=0.1, device="cpu"):
    sae = SparseAutoencoder(patch_size=patch_size, n_features=n_features,
                             in_channels=1).to(device)
    # Init from data (jump-start)
    with torch.no_grad():
        idx = torch.randperm(patches.shape[0])[:n_features]
        seed = patches[idx]
        seed = seed / seed.norm(dim=1, keepdim=True).clamp_min(1e-6)
        sae.decoder.weight.data = seed.t().clone()
    opt = optim.Adam(sae.parameters(), lr=lr)
    n = patches.shape[0]
    n_batches = max(1, n // batch_size)
    for ep in range(n_epochs):
        perm = torch.randperm(n)
        warmup = min(1.0, (ep + 1) / max(1, int(0.3 * n_epochs)))
        cur_l1 = l1_weight * warmup
        ep_recon = 0.0
        for i in range(n_batches):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            x = patches[idx].to(device)
            x_hat, z = sae(x)
            recon = F.mse_loss(x_hat, x)
            l1 = z.abs().mean()
            loss = recon + cur_l1 * l1
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                W = sae.decoder.weight
                W /= W.norm(dim=0, keepdim=True).clamp_min(1e-6)
            ep_recon += recon.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1:3d}/{n_epochs}  recon={ep_recon/n_batches:.4f}  "
                  f"l1_w={cur_l1:.3f}")
    return sae


def fit_gabor_to_filter(filt: torch.Tensor):
    """Grid-search best Gabor fit to a learned filter. Returns (params, score)."""
    if filt.dim() == 3:
        signed = filt[0]
    else:
        signed = filt
    k = signed.shape[0]
    coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    sf = signed.flatten()
    sf = sf - sf.mean()
    sf = sf / sf.norm().clamp_min(1e-6)
    best_score = -float("inf"); best = None
    thetas = torch.linspace(0, math.pi, 16, dtype=torch.float32)[:-1]
    lambdas = torch.linspace(2.5, 8.0, 10, dtype=torch.float32)
    sigmas = torch.linspace(1.2, 4.0, 6, dtype=torch.float32)
    phases = torch.linspace(0, 2 * math.pi, 8, dtype=torch.float32)[:-1]
    for theta in thetas:
        cos_t = torch.cos(theta); sin_t = torch.sin(theta)
        x_t = xx * cos_t + yy * sin_t
        for sigma in sigmas:
            env = torch.exp(-(x_t ** 2 + ((-xx * sin_t + yy * cos_t)) ** 2)
                             / (2 * sigma ** 2))
            for lam in lambdas:
                for phase in phases:
                    g = env * torch.cos(2 * math.pi * x_t / lam + phase)
                    g = g - g.mean()
                    g = g / g.norm().clamp_min(1e-6)
                    score = abs((g.flatten() * sf).sum().item())
                    if score > best_score:
                        best_score = score
                        best = (float(theta), float(sigma), float(lam), float(phase))
    return best, best_score


def visualize_filters(sae: SparseAutoencoder, save_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping visualization")
        return
    filters = sae.get_filters()  # (n_features, 1, k, k)
    n = filters.shape[0]
    rows = int(math.ceil(math.sqrt(n)))
    fig, axes = plt.subplots(rows, rows, figsize=(rows * 1.2, rows * 1.2))
    for i, ax in enumerate(axes.flat):
        if i < n:
            f = filters[i, 0].detach().cpu().numpy()
            vmax = max(abs(f.min()), abs(f.max()))
            ax.imshow(f, cmap="gray", vmin=-vmax, vmax=vmax)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="synthetic_10class")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=12)
    parser.add_argument("--n_features", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--n_pretrain_max", type=int, default=4000)
    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.source}...")
    images, _, _, _, _, _ = load_dataset(args.source,
                                          data_root=args.data_root,
                                          image_size=args.image_size,
                                          n_pretrain_max=args.n_pretrain_max)
    print(f"  using {images.shape[0]} unlabeled images")

    print("Extracting whitened patches...")
    patches = extract_whitened_patches(images, patch_size=args.patch_size,
                                        n_patches_per_image=20)
    print(f"  {patches.shape[0]} patches of dim {patches.shape[1]}")

    print("Training sparse autoencoder...")
    sae = train_sae(patches, n_features=args.n_features,
                    patch_size=args.patch_size,
                    n_epochs=args.n_epochs, device=args.device)

    sae_path = Path(args.out_dir) / f"sae_{args.source}.pt"
    torch.save(sae.state_dict(), sae_path)
    print(f"Saved SAE to {sae_path}")

    fig_path = Path(args.out_dir) / f"sae_filters_{args.source}.png"
    visualize_filters(sae, str(fig_path))
    print(f"Saved filter visualization to {fig_path}")

    print("Computing Gabor fits...")
    filters = sae.get_filters()
    scores = []
    for i in range(filters.shape[0]):
        _, score = fit_gabor_to_filter(filters[i])
        scores.append(score)
    scores = np.array(scores)
    print(f"  Gabor fit scores: mean={scores.mean():.3f}, "
          f"median={np.median(scores):.3f}")
    print(f"  fraction > 0.5: {(scores > 0.5).mean():.2%}")
    print(f"  fraction > 0.7: {(scores > 0.7).mean():.2%}")


if __name__ == "__main__":
    main()

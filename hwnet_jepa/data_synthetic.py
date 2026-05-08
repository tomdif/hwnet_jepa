"""
Synthetic data with natural-image statistics (1/f spectrum) plus class-defining
spatial patterns. Used in the original synthetic experiments and as a baseline
for real-data comparisons.

The 10-class structure:
  0: horizontal-dominant edges
  1: vertical-dominant edges
  2: diag-up edges (45 deg)
  3: diag-down edges (135 deg)
  4: low-frequency blobs
  5: high-frequency texture
  6: top-half emphasis (blobs in top half)
  7: bottom-half emphasis
  8: concentric ripples
  9: radial spokes
"""
import math
import numpy as np
import torch


def pink_noise_batch(n: int, size: int = 32, n_channels: int = 3,
                     alpha: float = 1.0, seed: int = None) -> torch.Tensor:
    """1/f^alpha pink noise images via FFT. alpha=1.0 matches natural images."""
    if seed is not None:
        np.random.seed(seed)
    fy = np.fft.fftfreq(size).reshape(-1, 1)
    fx = np.fft.fftfreq(size).reshape(1, -1)
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1.0
    spectrum = 1.0 / (f ** alpha)
    spectrum[0, 0] = 0.0
    phases = np.exp(2j * np.pi * np.random.rand(n, n_channels, size, size))
    coeff = spectrum[None, None, :, :] * phases
    imgs = np.fft.ifft2(coeff, axes=(-2, -1)).real
    imgs = imgs - imgs.mean(axis=(-2, -1), keepdims=True)
    imgs_std = imgs.std(axis=(-2, -1), keepdims=True) + 1e-8
    imgs = imgs / imgs_std * 0.2 + 0.5
    return torch.from_numpy(np.clip(imgs, 0, 1).astype(np.float32))


def add_oriented_lines_batch(images, n_per_image=6, theta_mean=None,
                              theta_std=0.15, strength=0.3):
    images = images.clone()
    N, C, H, W = images.shape
    coords = torch.arange(H).float() - H / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    yy_e = yy.unsqueeze(0).expand(N, H, W)
    xx_e = xx.unsqueeze(0).expand(N, H, W)
    for _ in range(n_per_image):
        if theta_mean is not None:
            thetas = theta_mean + torch.randn(N) * theta_std
        else:
            thetas = torch.rand(N) * math.pi
        cy = torch.randint(H // 4, 3 * H // 4, (N,)).float() - H / 2
        cx = torch.randint(W // 4, 3 * W // 4, (N,)).float() - W / 2
        lengths = torch.randint(8, 14, (N,)).float()
        signs = torch.randint(0, 2, (N,)).float() * 2 - 1
        cos_e = torch.cos(thetas).view(N, 1, 1)
        sin_e = torch.sin(thetas).view(N, 1, 1)
        dy = yy_e - cy.view(N, 1, 1)
        dx = xx_e - cx.view(N, 1, 1)
        perp = (dy * cos_e - dx * sin_e).abs()
        along = (dy * sin_e + dx * cos_e).abs()
        mask = (perp < 1.5) & (along < lengths.view(N, 1, 1) / 2)
        delta = mask.float() * signs.view(N, 1, 1) * strength
        for c in range(C):
            images[:, c] = (images[:, c] + delta).clamp(0, 1)
    return images


def add_blobs_batch(images, n_per_image=4, radius_range=(3, 6),
                    location="any", strength=0.4):
    images = images.clone()
    N, C, H, W = images.shape
    coords = torch.arange(H).float()
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    yy_e = yy.unsqueeze(0).expand(N, H, W)
    xx_e = xx.unsqueeze(0).expand(N, H, W)
    for _ in range(n_per_image):
        radii = torch.rand(N) * (radius_range[1] - radius_range[0]) + radius_range[0]
        if location == "top":
            cy = torch.rand(N) * H * 0.45
        elif location == "bottom":
            cy = torch.rand(N) * H * 0.45 + H * 0.55
        else:
            cy = torch.rand(N) * H
        cx = torch.rand(N) * W
        signs = torch.randint(0, 2, (N,)).float() * 2 - 1
        d = torch.sqrt((yy_e - cy.view(N, 1, 1)) ** 2 +
                       (xx_e - cx.view(N, 1, 1)) ** 2)
        falloff = (1 - d / radii.view(N, 1, 1)).clamp(0, 1)
        delta = falloff * signs.view(N, 1, 1) * strength
        for c in range(C):
            images[:, c] = (images[:, c] + delta).clamp(0, 1)
    return images


def add_concentric_batch(images, strength=0.4):
    images = images.clone()
    N, C, H, W = images.shape
    coords = torch.arange(H).float()
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    yy_e = yy.unsqueeze(0).expand(N, H, W)
    xx_e = xx.unsqueeze(0).expand(N, H, W)
    cy = (H / 2 + torch.randint(-3, 4, (N,)).float()).view(N, 1, 1)
    cx = (W / 2 + torch.randint(-3, 4, (N,)).float()).view(N, 1, 1)
    d = torch.sqrt((yy_e - cy) ** 2 + (xx_e - cx) ** 2)
    delta = strength * torch.sin(d * 1.2)
    for c in range(C):
        images[:, c] = (images[:, c] + delta).clamp(0, 1)
    return images


def add_radial_batch(images, strength=0.4):
    images = images.clone()
    N, C, H, W = images.shape
    coords = torch.arange(H).float()
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    yy_e = yy.unsqueeze(0).expand(N, H, W)
    xx_e = xx.unsqueeze(0).expand(N, H, W)
    cy = H / 2
    cx = W / 2
    angles = torch.atan2(yy_e - cy, xx_e - cx)
    n_spokes = (torch.randint(0, 3, (N,)).float() * 2 + 4).view(N, 1, 1)
    delta = strength * torch.cos(angles * n_spokes)
    for c in range(C):
        images[:, c] = (images[:, c] + delta).clamp(0, 1)
    return images


def add_high_freq_texture_batch(images, strength=0.25):
    return (images + torch.randn_like(images) * strength).clamp(0, 1)


def generate_pretrain_dataset(n_images=2000, size=32, seed=0):
    """Pink noise + random oriented edges. For SSL pretraining."""
    np.random.seed(seed); torch.manual_seed(seed)
    images = pink_noise_batch(n_images, size=size, alpha=1.0)
    mask = torch.rand(n_images) < 0.7
    if mask.any():
        images[mask] = add_oriented_lines_batch(images[mask], n_per_image=5,
                                                strength=0.3)
    return images


def generate_classification_dataset(n_per_class=200, n_classes=10,
                                    size=32, seed=0):
    """10-class single-pattern dataset (the original task)."""
    np.random.seed(seed); torch.manual_seed(seed)
    all_images, all_labels = [], []
    for cls in range(n_classes):
        base = pink_noise_batch(n_per_class, size=size, alpha=1.0) * 0.5 + 0.25
        if cls == 0:
            imgs = add_oriented_lines_batch(base, n_per_image=8, theta_mean=0.0,
                                            theta_std=0.15)
        elif cls == 1:
            imgs = add_oriented_lines_batch(base, n_per_image=8,
                                            theta_mean=math.pi / 2, theta_std=0.15)
        elif cls == 2:
            imgs = add_oriented_lines_batch(base, n_per_image=8,
                                            theta_mean=math.pi / 4, theta_std=0.15)
        elif cls == 3:
            imgs = add_oriented_lines_batch(base, n_per_image=8,
                                            theta_mean=3 * math.pi / 4, theta_std=0.15)
        elif cls == 4:
            imgs = add_blobs_batch(base, n_per_image=4, radius_range=(5, 10))
        elif cls == 5:
            imgs = add_high_freq_texture_batch(base)
        elif cls == 6:
            imgs = add_blobs_batch(base, n_per_image=4, radius_range=(3, 6),
                                   location="top")
        elif cls == 7:
            imgs = add_blobs_batch(base, n_per_image=4, radius_range=(3, 6),
                                   location="bottom")
        elif cls == 8:
            imgs = add_concentric_batch(base)
        elif cls == 9:
            imgs = add_radial_batch(base)
        all_images.append(imgs)
        all_labels.append(torch.full((n_per_class,), cls, dtype=torch.long))
    images = torch.cat(all_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    perm = torch.randperm(images.shape[0])
    return images[perm], labels[perm]


def generate_transfer_dataset(n_per_class=200, n_classes=8, size=32, seed=123):
    """8-class compositional dataset for the transfer task."""
    np.random.seed(seed); torch.manual_seed(seed)
    all_images, all_labels = [], []
    class_recipes = [
        (0.0,             "concentric", {}),
        (0.0,             "radial",     {}),
        (math.pi / 2,     "concentric", {}),
        (math.pi / 2,     "radial",     {}),
        (math.pi / 4,     "blobs",      {"radius_range": (5, 10)}),
        (math.pi / 4,     "texture",    {}),
        (3 * math.pi / 4, "blobs",      {"radius_range": (5, 10)}),
        (3 * math.pi / 4, "texture",    {}),
    ]
    for cls in range(n_classes):
        theta, second_op, kw = class_recipes[cls]
        base = pink_noise_batch(n_per_class, size=size, alpha=1.0) * 0.5 + 0.25
        imgs = add_oriented_lines_batch(base, n_per_image=6, theta_mean=theta,
                                        theta_std=0.15, strength=0.22)
        if second_op == "concentric":
            imgs = add_concentric_batch(imgs, strength=0.3)
        elif second_op == "radial":
            imgs = add_radial_batch(imgs, strength=0.3)
        elif second_op == "blobs":
            imgs = add_blobs_batch(imgs, n_per_image=3, strength=0.3, **kw)
        elif second_op == "texture":
            imgs = add_high_freq_texture_batch(imgs, strength=0.18)
        all_images.append(imgs)
        all_labels.append(torch.full((n_per_class,), cls, dtype=torch.long))
    images = torch.cat(all_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    perm = torch.randperm(images.shape[0])
    return images[perm], labels[perm]

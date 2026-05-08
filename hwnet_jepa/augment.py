"""
Augmentation pipeline and readout heads for downstream evaluation.

Augmentation: standard SSL recipe (random crop+resize, horizontal flip,
brightness/contrast jitter). Match these to your data domain - on our
synthetic data flip and crop hurt because they destroyed class-defining
features. On natural images they should help.

Readouts:
- JEPAClassifier (in jepa.py): mean-pool linear probe (cheap, simple)
- AttnPoolClassifier: learned attention pooling + linear head (best in our
  experiments, +5pt over linear probe at low data)
- knn_eval: parameter-free kNN classifier on mean-pooled embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def augment_batch(images: torch.Tensor,
                  crop_min: float = 0.6, crop_max: float = 1.0,
                  do_flip: bool = True,
                  brightness_range: tuple = (0.8, 1.2),
                  contrast_range: tuple = (0.8, 1.2)) -> torch.Tensor:
    """Random crop+resize, hflip, brightness/contrast jitter."""
    B, C, H, W = images.shape
    out = images.clone()

    if do_flip:
        flip_mask = torch.rand(B) < 0.5
        if flip_mask.any():
            out[flip_mask] = torch.flip(out[flip_mask], dims=[-1])

    crop_scales = torch.empty(B).uniform_(crop_min, crop_max)
    crop_sizes = (crop_scales * H).round().clamp(min=8).int()
    cropped = []
    for i in range(B):
        cs = int(crop_sizes[i].item())
        y0 = torch.randint(0, H - cs + 1, (1,)).item()
        x0 = torch.randint(0, W - cs + 1, (1,)).item()
        patch = out[i:i + 1, :, y0:y0 + cs, x0:x0 + cs]
        patch = F.interpolate(patch, size=(H, W), mode="bilinear", align_corners=False)
        cropped.append(patch)
    out = torch.cat(cropped, dim=0)

    brightness = torch.empty(B, 1, 1, 1).uniform_(*brightness_range)
    contrast = torch.empty(B, 1, 1, 1).uniform_(*contrast_range)
    mean = out.mean(dim=(-2, -1), keepdim=True)
    out = (out - mean) * contrast + mean * brightness
    return out.clamp(0, 1)


class AttentionPool(nn.Module):
    """Single-query attention pool over patch tokens."""
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.norm(out.squeeze(1))


class AttnPoolClassifier(nn.Module):
    """Encoder + attention-pool readout. Beat linear probe by ~5 points in our runs."""
    def __init__(self, encoder, num_classes: int = 10, freeze_encoder: bool = True,
                 n_heads: int = 4):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.pool = AttentionPool(encoder.embed_dim, n_heads=n_heads)
        self.head = nn.Linear(encoder.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                tokens = self.encoder.forward_full(x)
        else:
            tokens = self.encoder.forward_full(x)
        pooled = self.pool(tokens)
        return self.head(pooled)


def knn_eval(encoder, train_x: torch.Tensor, train_y: torch.Tensor,
             val_x: torch.Tensor, val_y: torch.Tensor, k: int = 5,
             num_classes: int = 10, batch_size: int = 128,
             device: str = "cpu") -> float:
    """kNN classifier on mean-pooled embeddings."""
    encoder.eval()
    encoder.to(device)
    train_x = train_x.to(device); train_y = train_y.to(device)
    val_x = val_x.to(device); val_y = val_y.to(device)
    with torch.no_grad():
        def embed(x):
            embs = []
            for i in range(0, x.shape[0], batch_size):
                tokens = encoder.forward_full(x[i:i + batch_size])
                embs.append(tokens.mean(dim=1))
            return torch.cat(embs, dim=0)
        train_emb = F.normalize(embed(train_x), dim=1)
        val_emb = F.normalize(embed(val_x), dim=1)
        sims = val_emb @ train_emb.t()
        topk_vals, topk_idx = sims.topk(k, dim=1)
        topk_labels = train_y[topk_idx]
        votes = torch.zeros(val_x.shape[0], num_classes, device=device)
        for kk in range(k):
            for c in range(num_classes):
                mask = (topk_labels[:, kk] == c).float()
                votes[:, c] += topk_vals[:, kk] * mask
        preds = votes.argmax(dim=1)
        return (preds == val_y).float().mean().item()

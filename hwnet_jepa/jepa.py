"""
I-JEPA on top of the H&W biological frontend.

Architecture:
  HWFrontEnd: image -> (B, 16, H, W) bio features
  PatchProjector: -> (B, n_patches, embed_dim) patch tokens
  Context encoder: small transformer over visible (subset) tokens
  Target encoder: EMA of context encoder, no gradients
  Predictor: predicts target latents at masked positions from visible context

Loss: smooth-L1 between predicted and target latents at masked positions, with
a VICReg-style variance regularizer to prevent representation collapse.
"""
import copy
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bio_layers import HWFrontEnd, HWFrontEndPlus


# ----------------------------- Building blocks ----------------------------- #


class PatchProjector(nn.Module):
    """Aggregate the V1 feature map into patch tokens via strided conv."""
    def __init__(self, in_channels: int = 16, patch_size: int = 4,
                 embed_dim: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        z = self.proj(x)
        B, D, Hp, Wp = z.shape
        z = z.flatten(2).transpose(1, 2)
        return z, (Hp, Wp)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""
    def __init__(self, dim: int, n_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=0.0,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, attn_mask=None):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ----------------------------- Encoders / Predictor ----------------------------- #


class JEPAEncoder(nn.Module):
    """HW frontend -> patch tokens -> transformer.

    Uses HWFrontEnd by default. Set use_end_stopped=True to use HWFrontEndPlus.
    """
    def __init__(self, n_orientations: int = 8, n_scales: int = 2,
                 patch_size: int = 4, embed_dim: int = 64,
                 n_layers: int = 3, n_heads: int = 4,
                 grid_size: int = 8, use_end_stopped: bool = False,
                 input_channels: int = 3):
        super().__init__()
        if use_end_stopped:
            self.frontend = HWFrontEndPlus(n_orientations=n_orientations,
                                           n_scales=n_scales,
                                           input_channels=input_channels)
        else:
            self.frontend = HWFrontEnd(n_orientations=n_orientations,
                                       n_scales=n_scales,
                                       input_channels=input_channels)
        self.patcher = PatchProjector(in_channels=self.frontend.out_channels,
                                      patch_size=patch_size, embed_dim=embed_dim)
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.n_patches = grid_size * grid_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads=n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.frontend(x)
        tokens, _ = self.patcher(feat)
        tokens = tokens + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)

    def forward_subset(self, x: torch.Tensor,
                       visible_idx: torch.Tensor) -> torch.Tensor:
        feat = self.frontend(x)
        tokens, _ = self.patcher(feat)
        tokens = tokens + self.pos_embed
        B, _, D = tokens.shape
        idx_exp = visible_idx.unsqueeze(-1).expand(-1, -1, D)
        visible = torch.gather(tokens, 1, idx_exp)
        for blk in self.blocks:
            visible = blk(visible)
        return self.norm(visible)


class JEPAPredictor(nn.Module):
    """Predict target latents at masked positions from visible context tokens."""
    def __init__(self, embed_dim: int = 64, predictor_dim: int = 48,
                 n_layers: int = 2, n_heads: int = 3, n_patches: int = 64):
        super().__init__()
        self.context_proj = nn.Linear(embed_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, predictor_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, n_heads=n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.out_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(self, context_tokens: torch.Tensor,
                visible_idx: torch.Tensor,
                target_idx: torch.Tensor) -> torch.Tensor:
        B, n_vis, _ = context_tokens.shape
        n_tgt = target_idx.shape[1]
        ctx = self.context_proj(context_tokens)
        ctx_pos = torch.gather(self.pos_embed.expand(B, -1, -1), 1,
                               visible_idx.unsqueeze(-1).expand(-1, -1, ctx.shape[-1]))
        ctx = ctx + ctx_pos
        tgt_pos = torch.gather(self.pos_embed.expand(B, -1, -1), 1,
                               target_idx.unsqueeze(-1).expand(-1, -1,
                               self.mask_token.shape[-1]))
        tgt = self.mask_token.expand(B, n_tgt, -1) + tgt_pos
        seq = torch.cat([ctx, tgt], dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        return self.out_proj(seq[:, n_vis:])


# ----------------------------- I-JEPA model ----------------------------- #


class IJEPA(nn.Module):
    """I-JEPA: context encoder + EMA target encoder + predictor."""
    def __init__(self, n_orientations: int = 8, n_scales: int = 2,
                 patch_size: int = 4, embed_dim: int = 64,
                 encoder_layers: int = 3, predictor_layers: int = 2,
                 grid_size: int = 8, ema_decay: float = 0.996,
                 use_end_stopped: bool = False, input_channels: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.n_patches = grid_size * grid_size
        self.ema_decay = ema_decay

        self.context_encoder = JEPAEncoder(
            n_orientations=n_orientations, n_scales=n_scales,
            patch_size=patch_size, embed_dim=embed_dim,
            n_layers=encoder_layers, grid_size=grid_size,
            use_end_stopped=use_end_stopped, input_channels=input_channels)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = JEPAPredictor(
            embed_dim=embed_dim, predictor_dim=embed_dim * 3 // 4,
            n_layers=predictor_layers, n_patches=self.n_patches)

    @torch.no_grad()
    def update_target(self):
        for p_t, p_c in zip(self.target_encoder.parameters(),
                            self.context_encoder.parameters()):
            p_t.data.mul_(self.ema_decay).add_(p_c.data, alpha=1 - self.ema_decay)
        for b_t, b_c in zip(self.target_encoder.buffers(),
                            self.context_encoder.buffers()):
            b_t.data.copy_(b_c.data)

    def forward(self, x: torch.Tensor,
                visible_idx: torch.Tensor,
                target_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx = self.context_encoder.forward_subset(x, visible_idx)
        preds = self.predictor(ctx, visible_idx, target_idx)
        with torch.no_grad():
            full_targets = self.target_encoder.forward_full(x)
            B = x.shape[0]
            D = full_targets.shape[-1]
            tgt_exp = target_idx.unsqueeze(-1).expand(-1, -1, D)
            targets = torch.gather(full_targets, 1, tgt_exp)
        return preds, targets


# ----------------------------- Masking ----------------------------- #


def sample_jepa_masks(batch_size: int, grid_size: int = 8,
                      visible_frac: float = 0.5,
                      target_frac: float = 0.25,
                      device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Random non-overlapping visible and target patches, fixed counts per sample.

    Simpler than canonical I-JEPA's block masking but easier to debug. Replace
    with sample_jepa_block_masks for the full I-JEPA recipe (TODO).
    """
    n_patches = grid_size * grid_size
    n_visible = int(n_patches * visible_frac)
    n_target = int(n_patches * target_frac)
    visible_list = []
    target_list = []
    for _ in range(batch_size):
        perm = torch.randperm(n_patches, device=device)
        visible_list.append(perm[:n_visible])
        target_list.append(perm[n_visible:n_visible + n_target])
    visible_idx = torch.stack(visible_list, dim=0)
    target_idx = torch.stack(target_list, dim=0)
    return visible_idx, target_idx


def sample_jepa_block_masks(batch_size: int, grid_size: int = 8,
                            n_target_blocks: int = 4,
                            target_block_size: Tuple[int, int] = (3, 3),
                            visible_block_scale: float = 0.85,
                            device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Canonical I-JEPA block masking.

    Each sample gets:
    - Several rectangular target blocks (default 4 of size 3x3)
    - One large visible block (default 85% scale of grid) MINUS any target blocks

    Returns visible_idx (B, n_visible_max) and target_idx (B, n_target_max),
    padded with -1 where samples have fewer indices than the max.
    """
    n_patches = grid_size * grid_size
    visible_lists = []
    target_lists = []
    max_vis = 0
    max_tgt = 0
    for _ in range(batch_size):
        # Sample target blocks
        target_set = set()
        for _b in range(n_target_blocks):
            h, w = target_block_size
            y0 = torch.randint(0, grid_size - h + 1, (1,)).item()
            x0 = torch.randint(0, grid_size - w + 1, (1,)).item()
            for dy in range(h):
                for dx in range(w):
                    target_set.add((y0 + dy) * grid_size + (x0 + dx))
        # Sample one large visible block
        vis_size = max(1, int(round(grid_size * visible_block_scale)))
        vy0 = torch.randint(0, grid_size - vis_size + 1, (1,)).item()
        vx0 = torch.randint(0, grid_size - vis_size + 1, (1,)).item()
        visible_set = set()
        for dy in range(vis_size):
            for dx in range(vis_size):
                idx = (vy0 + dy) * grid_size + (vx0 + dx)
                if idx not in target_set:
                    visible_set.add(idx)
        v_list = sorted(visible_set)
        t_list = sorted(target_set)
        visible_lists.append(v_list)
        target_lists.append(t_list)
        max_vis = max(max_vis, len(v_list))
        max_tgt = max(max_tgt, len(t_list))

    # Pad with -1 to common length
    visible_idx = torch.full((batch_size, max_vis), -1, dtype=torch.long, device=device)
    target_idx = torch.full((batch_size, max_tgt), -1, dtype=torch.long, device=device)
    for i, (v, t) in enumerate(zip(visible_lists, target_lists)):
        visible_idx[i, :len(v)] = torch.tensor(v, device=device)
        target_idx[i, :len(t)] = torch.tensor(t, device=device)
    return visible_idx, target_idx


# ----------------------------- Downstream classifiers ----------------------------- #


class JEPAClassifier(nn.Module):
    """Linear probe (mean pool over patches + linear head) on JEPA encoder."""
    def __init__(self, encoder: JEPAEncoder, num_classes: int = 10,
                 freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.LayerNorm(encoder.embed_dim),
            nn.Linear(encoder.embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                tokens = self.encoder.forward_full(x)
        else:
            tokens = self.encoder.forward_full(x)
        pooled = tokens.mean(dim=1)
        return self.head(pooled)

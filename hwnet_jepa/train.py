"""
Training loops.

- supervised_train: train any classifier on (x, y) data with adaptive epochs
- jepa_pretrain: I-JEPA self-supervised pretraining with VICReg-style variance reg
- linear_probe / fine_tune: downstream eval helpers
"""
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .jepa import IJEPA, JEPAClassifier, sample_jepa_masks, sample_jepa_block_masks
from .augment import augment_batch


# ----------------------------- Supervised ----------------------------- #


def supervised_train(model: nn.Module,
                     train_x: torch.Tensor, train_y: torch.Tensor,
                     val_x: torch.Tensor, val_y: torch.Tensor,
                     n_epochs: int = 15, batch_size: int = 64,
                     lr: float = 3e-3, weight_decay: float = 1e-4,
                     device: str = "cpu",
                     frontend_lr_mult: float = 1.0,
                     verbose: bool = False) -> float:
    """Standard supervised training. Returns best validation accuracy."""
    model = model.to(device)
    train_x = train_x.to(device); train_y = train_y.to(device)
    val_x = val_x.to(device); val_y = val_y.to(device)

    if frontend_lr_mult != 1.0 and hasattr(model, "frontend") and hasattr(model, "head"):
        opt = optim.Adam([
            {"params": list(model.frontend.parameters()),
             "lr": lr * frontend_lr_mult},
            {"params": list(model.head.parameters()), "lr": lr},
        ], weight_decay=weight_decay)
    else:
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(TensorDataset(train_x, train_y),
                        batch_size=batch_size, shuffle=True)
    best = 0.0
    for ep in range(n_epochs):
        model.train()
        for xb, yb in loader:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = []
            for i in range(0, val_x.shape[0], 256):
                v.append(model(val_x[i:i + 256]))
            va = (torch.cat(v, 0).argmax(1) == val_y).float().mean().item()
        if va > best:
            best = va
        if verbose and ((ep + 1) % 5 == 0 or ep == 0):
            print(f"  ep {ep+1:3d}/{n_epochs}  val_acc={va:.3f}  best={best:.3f}")
    return best


# ----------------------------- JEPA pretraining ----------------------------- #


def jepa_pretrain(model: IJEPA, images: torch.Tensor,
                  n_epochs: int = 15, batch_size: int = 64, lr: float = 1e-3,
                  weight_decay: float = 1e-4,
                  use_augmentation: bool = False,
                  use_block_masks: bool = False,
                  ema_init: float = 0.95, ema_final: float = 0.999,
                  var_loss_weight: float = 1.0, var_target: float = 1.0,
                  device: str = "cpu", verbose: bool = True):
    """JEPA pretraining with VICReg-style variance regularization.

    Variance reg is essential here. Without it the EMA target tends toward a
    near-constant solution and the predictor trivially matches it (we observed
    pred_var dropping to 0.002 without this term in early experiments).

    Set use_augmentation=True for natural images. On synthetic data with class
    structure baked into specific orientations, augmentations like flip/crop
    can destroy class-relevant information.

    Set use_block_masks=True to use canonical I-JEPA block masking (4 target
    blocks of 3x3 from a 0.85-scale visible block). The default random masking
    is simpler to debug but the block version matches the original paper.
    """
    model = model.to(device)
    images = images.to(device)
    optimiser = optim.AdamW(
        list(model.context_encoder.parameters()) +
        list(model.predictor.parameters()),
        lr=lr, weight_decay=weight_decay)
    n = images.shape[0]
    n_batches = max(1, n // batch_size)
    total_steps = n_epochs * n_batches
    step = 0
    history = []

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        epoch_var_loss = 0.0
        epoch_pred_var = 0.0
        epoch_tgt_var = 0.0
        for i in range(n_batches):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            xb = images[idx]
            if use_augmentation:
                xb = augment_batch(xb)
            if use_block_masks:
                visible_idx, target_idx = sample_jepa_block_masks(
                    batch_size=xb.shape[0], grid_size=model.grid_size, device=device)
                # Block masks may have -1 padding. For now we strip and use only
                # the first n_visible/n_target for each sample. (TODO: full padding-aware
                # implementation with attention masks.)
                # Minimum length across samples:
                min_vis = (visible_idx >= 0).sum(dim=1).min().item()
                min_tgt = (target_idx >= 0).sum(dim=1).min().item()
                visible_idx = visible_idx[:, :min_vis]
                target_idx = target_idx[:, :min_tgt]
            else:
                visible_idx, target_idx = sample_jepa_masks(
                    batch_size=xb.shape[0], grid_size=model.grid_size, device=device)
            preds, targets = model(xb, visible_idx, target_idx)
            pred_loss = F.smooth_l1_loss(preds, targets)
            preds_flat = preds.reshape(-1, preds.shape[-1])
            std_preds = torch.sqrt(preds_flat.var(dim=0) + 1e-4)
            var_loss = F.relu(var_target - std_preds).mean()
            # Also penalise low-variance CONTEXT encoder features so the EMA target
            # can't drift to a constant. Forwarding context_encoder full-pass with
            # gradients enabled (separate from the no-grad target_encoder pass).
            ctx_full = model.context_encoder.forward_full(xb)
            ctx_flat = ctx_full.reshape(-1, ctx_full.shape[-1])
            std_ctx = torch.sqrt(ctx_flat.var(dim=0) + 1e-4)
            ctx_var_loss = F.relu(var_target - std_ctx).mean()
            loss = pred_loss + var_loss_weight * (var_loss + ctx_var_loss)
            optimiser.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.context_encoder.parameters()) +
                list(model.predictor.parameters()), max_norm=1.0)
            optimiser.step()
            ema_decay = ema_init + (ema_final - ema_init) * (step / max(1, total_steps - 1))
            model.ema_decay = ema_decay
            model.update_target()
            epoch_loss += pred_loss.item()
            epoch_var_loss += var_loss.item()
            with torch.no_grad():
                epoch_pred_var += preds.var(dim=0).mean().item()
                epoch_tgt_var += targets.var(dim=0).mean().item()
            step += 1
        if verbose and ((epoch + 1) % 2 == 0 or epoch == 0):
            print(f"  ep {epoch+1:3d}/{n_epochs}  pred_loss={epoch_loss/n_batches:.4f}  "
                  f"var_loss={epoch_var_loss/n_batches:.3f}  "
                  f"pred_var={epoch_pred_var/n_batches:.3f}  "
                  f"tgt_var={epoch_tgt_var/n_batches:.3f}  ema={ema_decay:.4f}")
        history.append({
            "epoch": epoch + 1,
            "pred_loss": epoch_loss / n_batches,
            "var_loss": epoch_var_loss / n_batches,
            "pred_var": epoch_pred_var / n_batches,
            "tgt_var": epoch_tgt_var / n_batches,
        })
    return model, history


# ----------------------------- Downstream evaluation ----------------------------- #


def linear_probe(encoder, train_x, train_y, val_x, val_y,
                 num_classes: int = 10,
                 readout: str = "linear", n_epochs: int = 25,
                 batch_size: int = 64, lr: float = 3e-3,
                 freeze_encoder: bool = True, device: str = "cpu") -> float:
    """Train a readout on a (frozen or fine-tuned) encoder. Returns best val acc.

    readout: 'linear' (mean-pool linear), 'attn_pool' (attention-pool linear), or 'knn'
    """
    if readout == "knn":
        from .augment import knn_eval
        return knn_eval(encoder, train_x, train_y, val_x, val_y,
                        num_classes=num_classes, device=device)

    from .jepa import JEPAClassifier
    from .augment import AttnPoolClassifier
    if readout == "linear":
        clf = JEPAClassifier(encoder, num_classes=num_classes,
                             freeze_encoder=freeze_encoder)
    elif readout == "attn_pool":
        clf = AttnPoolClassifier(encoder, num_classes=num_classes,
                                 freeze_encoder=freeze_encoder)
        if lr == 3e-3:
            lr = 1e-3  # attn-pool needs gentler LR
    else:
        raise ValueError(f"Unknown readout: {readout}")

    clf = clf.to(device)
    train_x = train_x.to(device); train_y = train_y.to(device)
    val_x = val_x.to(device); val_y = val_y.to(device)
    opt = optim.Adam([p for p in clf.parameters() if p.requires_grad],
                     lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(train_x, train_y),
                        batch_size=batch_size, shuffle=True)
    best = 0.0
    for ep in range(n_epochs):
        clf.train()
        for xb, yb in loader:
            logits = clf(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        clf.eval()
        with torch.no_grad():
            v = []
            for i in range(0, val_x.shape[0], 256):
                v.append(clf(val_x[i:i + 256]))
            va = (torch.cat(v, 0).argmax(1) == val_y).float().mean().item()
        if va > best:
            best = va
    return best

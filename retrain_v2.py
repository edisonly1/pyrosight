"""Retrain PyroSight v2 — improved architecture + enhanced augmentation.

Usage:
    .venv/bin/python retrain_v2.py

Changes from v1:
  Architecture: wider [64,96,128], bottleneck self-attention, SE blocks,
                learned weather upsampler, stochastic depth
  Augmentation: Gaussian blur on weather channels, channel noise,
                random erasing, cutmix
  Training:     longer (200 epochs), cosine LR, higher patience
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from pyrosight.config import Config
from pyrosight.data import _parse_tfrecord_files, WildfireDataset, _build_fire_sampler
from pyrosight.loss import EDLPhysicsLoss
from pyrosight.model_v2 import DualBranchUNetEDL_v2


# ---------------------------------------------------------------------------
# Enhanced augmentation
# ---------------------------------------------------------------------------

def augment_enhanced(x_norm, x_raw, label, mask):
    """Enhanced augmentation beyond flips/rotations.

    Adds:
    - Gaussian blur on weather channels (smooths 4km grid → grid-invariant)
    - Per-channel noise injection
    - Random brightness/contrast shift on normalized channels
    """
    # Standard spatial augmentation first
    x_norm, x_raw, label, mask = WildfireDataset._augment(x_norm, x_raw, label, mask)

    # --- Gaussian blur on weather channels (indices 1-7, 10 in x_norm) ---
    # This teaches the model to not rely on the 4km grid pattern
    if torch.rand(1).item() > 0.5:
        weather_idx = [1, 2, 3, 4, 5, 6, 7, 10]
        sigma = torch.rand(1).item() * 1.5 + 0.5  # σ ∈ [0.5, 2.0]
        ks = int(sigma * 4) | 1  # kernel size (odd)
        ks = max(3, min(ks, 7))
        for ci in weather_idx:
            ch = x_norm[ci].unsqueeze(0).unsqueeze(0)
            blurred = _gaussian_blur(ch, ks, sigma)
            x_norm[ci] = blurred.squeeze()

    # --- Channel noise injection ---
    if torch.rand(1).item() > 0.5:
        noise_scale = torch.rand(1).item() * 0.1  # up to 10% noise
        noise = torch.randn_like(x_norm) * noise_scale
        # Don't add noise to PrevFireMask (binary, index 11)
        noise[11] = 0
        x_norm = x_norm + noise

    # --- Random channel dropout (zero out 1-2 weather channels) ---
    if torch.rand(1).item() > 0.8:  # 20% chance
        weather_idx = [1, 2, 3, 4, 5, 6, 7, 10]
        n_drop = torch.randint(1, 3, (1,)).item()
        drop_idx = torch.randperm(len(weather_idx))[:n_drop]
        for i in drop_idx:
            x_norm[weather_idx[i]] = 0.0

    return x_norm, x_raw, label, mask


def _gaussian_blur(x, kernel_size, sigma):
    """Apply Gaussian blur to a 4D tensor."""
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
    kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    return F.conv2d(x, kernel_2d, padding=pad)


# ---------------------------------------------------------------------------
# Custom dataset with enhanced augmentation
# ---------------------------------------------------------------------------

class EnhancedDataset(WildfireDataset):
    def __getitem__(self, idx):
        x_norm, x_raw, label, mask = super().__getitem__(idx)
        if self.augment:
            x_norm, x_raw, label, mask = augment_enhanced(x_norm, x_raw, label, mask)
        return x_norm, x_raw, label, mask


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[v2] Device: {device}")

    cfg = Config()
    cfg.max_epochs = 200
    cfg.warmup_epochs = 5
    cfg.lr = 3e-4
    cfg.batch_size = 24  # Slightly smaller for wider model
    cfg.patience = 40
    cfg.kl_anneal_epochs = 40
    cfg.checkpoint_dir = "checkpoints_v3"
    cfg.checkpoint_every = 10

    # Wider architecture
    widths = (64, 96, 128)
    bottleneck_ch = 256

    # Data
    print("[v2] Loading data...")
    num_workers = 0 if device == "mps" else 2
    raw_train = _parse_tfrecord_files(cfg.train_pattern, cfg)
    raw_val = _parse_tfrecord_files(cfg.val_pattern, cfg)
    print(f"[v2] Train: {len(raw_train)}, Val: {len(raw_val)}")

    train_ds = EnhancedDataset(raw_train, cfg, augment=True)
    val_ds = WildfireDataset(raw_val, cfg, augment=False)

    # Fire-patch oversampling
    n_fire = sum(1 for s in raw_train if (s[cfg.label_key] == 1).any())
    fire_frac = n_fire / len(raw_train)
    print(f"[v2] Fire patches: {n_fire}/{len(raw_train)} ({100*fire_frac:.1f}%)")

    if fire_frac < 0.4:
        sampler = _build_fire_sampler(raw_train, cfg.label_key)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=cfg.batch_size, sampler=sampler,
            num_workers=num_workers, drop_last=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True,
        )

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers,
    )

    print(f"[v2] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = DualBranchUNetEDL_v2(
        widths=widths, bottleneck_ch=bottleneck_ch,
        num_classes=cfg.num_classes, dropout=cfg.dropout_rate,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[v2] Parameters: {param_count:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cfg.max_epochs - cfg.warmup_epochs, eta_min=cfg.min_lr)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.warmup_epochs])

    criterion = EDLPhysicsLoss(
        w_fire=cfg.w_fire, w_nofire=cfg.w_nofire,
        dice_weight=cfg.dice_weight,
        aux_weights=tuple(cfg.aux_loss_weights),
        physics_weights=cfg.physics_loss_weights,
    )

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Checkpoint
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    patience_counter = 0
    start_epoch = 0

    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        print(f"[v2] Resuming from {latest}")
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            print("  Optimizer/scheduler state mismatch, starting fresh")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"  Epoch {start_epoch}, best F1={best_f1:.4f}")

    # Train
    print(f"\n[v2] Training epochs {start_epoch}–{cfg.max_epochs}")
    print(f"  Widths: {widths}, Bottleneck: {bottleneck_ch}")
    print(f"  LR: {cfg.lr}, Patience: {cfg.patience}")
    print(f"  Enhanced augmentation: blur + noise + channel dropout")
    print()

    for epoch in range(start_epoch, cfg.max_epochs):
        kl_weight = cfg.get_kl_weight(epoch)
        t0 = time.time()

        model.train()
        train_loss = 0.0
        train_S = 0.0
        n = 0

        for x_norm, x_raw, y, mask in tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False):
            x_norm = x_norm.to(device)
            x_raw = x_raw.to(device)
            y = y.to(device)
            mask = mask.to(device)

            with torch.amp.autocast(device if device != "mps" else "cpu", enabled=use_amp):
                outputs = model(x_norm, x_raw)
                loss = criterion(outputs, y, mask, kl_weight, x_raw=x_raw)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            with torch.no_grad():
                alpha = outputs[0] if isinstance(outputs, tuple) else outputs
                train_S += alpha.sum(dim=1).mean().item()
            n += 1

        scheduler.step()
        avg_loss = train_loss / max(n, 1)
        avg_S = train_S / max(n, 1)

        # Validate
        val_f1, val_loss = validate(model, val_loader, criterion, kl_weight, device)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d} | loss={avg_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_F1={val_f1:.4f} | S={avg_S:.1f} | λ={kl_weight:.4f} | "
            f"lr={lr_now:.2e} | {elapsed:.0f}s"
        )

        ckpt_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
            "patience_counter": patience_counter,
            "widths": widths,
            "bottleneck_ch": bottleneck_ch,
        }

        if (epoch + 1) % cfg.checkpoint_every == 0:
            torch.save(ckpt_state, ckpt_dir / f"epoch_{epoch:03d}.pt")
        torch.save(ckpt_state, ckpt_dir / "latest.pt")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            ckpt_state["best_f1"] = best_f1
            torch.save(ckpt_state, ckpt_dir / "best.pt")
            print(f"  ★ New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    print(f"\n[v2] Done. Best F1: {best_f1:.4f}")
    print(f"  Checkpoints: {ckpt_dir}/")


def validate(model, val_loader, criterion, kl_weight, device):
    model.eval()
    total_loss = 0.0
    n = 0
    all_probs, all_targets, all_valid = [], [], []

    with torch.no_grad():
        for x_norm, x_raw, y, mask in val_loader:
            x_norm = x_norm.to(device)
            x_raw = x_raw.to(device)
            y = y.to(device)
            mask = mask.to(device)

            alpha = model(x_norm, x_raw)
            loss = criterion(alpha, y, mask, kl_weight, x_raw=x_raw)
            total_loss += loss.item()
            n += 1

            probs = DualBranchUNetEDL_v2.get_probabilities(alpha)
            all_probs.append(probs[:, 1].cpu())
            all_targets.append(y.cpu())
            all_valid.append(mask.bool().cpu())

    probs_cat = torch.cat(all_probs)
    targets_cat = torch.cat(all_targets)
    valid_cat = torch.cat(all_valid)

    best_f1 = 0.0
    for thresh in [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        preds = (probs_cat > thresh).long()
        tp = ((preds == 1) & (targets_cat == 1) & valid_cat).sum().item()
        fp = ((preds == 1) & (targets_cat == 0) & valid_cat).sum().item()
        fn = ((preds == 0) & (targets_cat == 1) & valid_cat).sum().item()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        if f1 > best_f1:
            best_f1 = f1

    return best_f1, total_loss / max(n, 1)


if __name__ == "__main__":
    main()

"""Retrain PyroSight with improved hyperparameters on Apple Silicon MPS.

Usage:
    .venv/bin/python retrain.py

Saves checkpoints to checkpoints_v2/ so the original model is preserved.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from pyrosight.config import Config
from pyrosight.data import build_dataloaders
from pyrosight.loss import EDLPhysicsLoss
from pyrosight.model import DualBranchUNetEDL


def main():
    # ---- Device ----
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[retrain] Device: {device}")

    # ---- Config (with improvements) ----
    cfg = Config()
    cfg.max_epochs = 100          # Train longer (was effectively ~10 before)
    cfg.warmup_epochs = 3         # Shorter warmup
    cfg.lr = 5e-4                 # Slightly lower LR for stability
    cfg.batch_size = 32           # Keep at 32
    cfg.patience = 25             # Stop if no improvement for 25 epochs
    cfg.kl_anneal_epochs = 30     # Faster KL ramp-up
    cfg.checkpoint_dir = "checkpoints_v2"
    cfg.checkpoint_every = 10

    # ---- Data ----
    print("[retrain] Loading data...")
    num_workers = 0 if device == "mps" else 2  # MPS doesn't like multiprocess
    train_loader, val_loader, _ = build_dataloaders(cfg, num_workers=num_workers)
    print(f"[retrain] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---- Model ----
    model = DualBranchUNetEDL(
        widths=tuple(cfg.encoder_widths),
        bottleneck_ch=cfg.bottleneck_channels,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout_rate,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[retrain] Parameters: {param_count:,}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_epochs - cfg.warmup_epochs,  # This was the bug before
        eta_min=cfg.min_lr,
    )
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.warmup_epochs])

    criterion = EDLPhysicsLoss(
        w_fire=cfg.w_fire, w_nofire=cfg.w_nofire,
        dice_weight=cfg.dice_weight,
        aux_weights=tuple(cfg.aux_loss_weights),
        physics_weights=cfg.physics_loss_weights,
    )

    # ---- Mixed precision ----
    use_amp = device == "cuda"  # AMP only for CUDA, not MPS
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ---- Checkpointing ----
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    patience_counter = 0
    start_epoch = 0

    # Resume from v2 checkpoint if exists
    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists():
        print(f"[retrain] Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                print("  Scheduler state mismatch, starting fresh schedule")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"  Epoch {start_epoch}, best F1={best_f1:.4f}, patience={patience_counter}")

    # ---- Training Loop ----
    print(f"\n[retrain] Starting from epoch {start_epoch}, training to {cfg.max_epochs}")
    print(f"  LR: {cfg.lr}, warmup: {cfg.warmup_epochs}, cosine T_max: {cfg.max_epochs - cfg.warmup_epochs}")
    print(f"  KL anneal: {cfg.kl_anneal_epochs} epochs, max: {cfg.kl_max}")
    print(f"  Patience: {cfg.patience}")
    print()

    for epoch in range(start_epoch, cfg.max_epochs):
        kl_weight = cfg.get_kl_weight(epoch)
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_S = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False)
        for x_norm, x_raw, y, mask in pbar:
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
            n_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = train_loss / max(n_batches, 1)
        avg_S = train_S / max(n_batches, 1)

        # --- Validate ---
        val_f1, val_loss = validate(model, val_loader, criterion, kl_weight, device)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d} | "
            f"loss={avg_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_F1={val_f1:.4f} | "
            f"S={avg_S:.1f} | "
            f"λ_kl={kl_weight:.4f} | "
            f"lr={lr_now:.2e} | "
            f"{elapsed:.0f}s"
        )

        if epoch > 10 and avg_S < cfg.num_classes + 0.5:
            print("  ⚠ Evidence collapse warning: S ≈ K")

        # --- Checkpoint ---
        ckpt_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
            "patience_counter": patience_counter,
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
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {cfg.patience} epochs)")
                break

    print(f"\n[retrain] Done. Best F1: {best_f1:.4f}")
    print(f"  Checkpoints in: {ckpt_dir}/")
    print(f"  To use the new model, update server.py to load from '{ckpt_dir}/best.pt'")


def validate(model, val_loader, criterion, kl_weight, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_targets = []
    all_valid = []

    with torch.no_grad():
        for x_norm, x_raw, y, mask in val_loader:
            x_norm = x_norm.to(device)
            x_raw = x_raw.to(device)
            y = y.to(device)
            mask = mask.to(device)

            alpha = model(x_norm, x_raw)
            loss = criterion(alpha, y, mask, kl_weight, x_raw=x_raw)
            total_loss += loss.item()
            n_batches += 1

            probs = DualBranchUNetEDL.get_probabilities(alpha)
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
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        if f1 > best_f1:
            best_f1 = f1

    return best_f1, total_loss / max(n_batches, 1)


if __name__ == "__main__":
    main()

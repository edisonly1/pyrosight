"""Training loop for DualBranchUNetEDL with physics-informed loss."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from .config import Config
from .loss import EDLPhysicsLoss
from .model import DualBranchUNetEDL


def train(cfg: Config, train_loader, val_loader, device: str = "cuda"):
    """Run full training loop."""
    model = DualBranchUNetEDL(
        widths=tuple(cfg.encoder_widths),
        bottleneck_ch=cfg.bottleneck_channels,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout_rate,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    warmup = LinearLR(optimizer, start_factor=1e-4, total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs - cfg.warmup_epochs, eta_min=cfg.min_lr
    )
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.warmup_epochs])

    criterion = EDLPhysicsLoss(
        w_fire=cfg.w_fire, w_nofire=cfg.w_nofire,
        dice_weight=cfg.dice_weight,
        aux_weights=tuple(cfg.aux_loss_weights),
        physics_weights=cfg.physics_loss_weights,
    )

    # Mixed precision — CUDA only
    device_type = "cuda" if "cuda" in device else "cpu"
    use_amp = device_type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Checkpointing
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    patience_counter = 0
    start_epoch = 0

    # Resume
    latest_ckpt = ckpt_dir / "latest.pt"
    best_ckpt = ckpt_dir / "best.pt"
    if latest_ckpt.exists():
        print(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt["best_f1"]
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"  Resumed at epoch {start_epoch}, best F1={best_f1:.4f}")

    for epoch in range(start_epoch, cfg.max_epochs):
        kl_weight = cfg.get_kl_weight(epoch)
        t0 = time.time()

        # --- Training ---
        model.train()
        train_loss = 0.0
        train_S = 0.0
        n_batches = 0

        for x_norm, x_raw, y, mask in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x_norm = x_norm.to(device, non_blocking=True)
            x_raw = x_raw.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with torch.amp.autocast(device_type, enabled=use_amp):
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

        scheduler.step()

        avg_train_loss = train_loss / max(n_batches, 1)
        avg_S = train_S / max(n_batches, 1)

        # --- Validation ---
        val_f1, val_loss = _validate(model, val_loader, criterion, kl_weight,
                                     device, device_type, use_amp)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_F1={val_f1:.4f} | "
            f"mean_S={avg_S:.1f} | "
            f"λ_kl={kl_weight:.4f} | "
            f"lr={lr_now:.2e} | "
            f"{elapsed:.0f}s"
        )

        if epoch > 10 and avg_S < cfg.num_classes + 0.5:
            print("  WARNING: Mean Dirichlet strength S near K — possible evidence collapse!")

        # --- Checkpointing ---
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
        torch.save(ckpt_state, latest_ckpt)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            ckpt_state["best_f1"] = best_f1
            torch.save(ckpt_state, best_ckpt)
            print(f"  ★ New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (patience={cfg.patience})")
                break

    print(f"\nTraining complete. Best validation F1: {best_f1:.4f}")
    return model


def _validate(model, val_loader, criterion, kl_weight, device, device_type, use_amp):
    """Run validation, sweep thresholds for best F1."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_targets = []
    all_valid = []

    with torch.no_grad():
        for x_norm, x_raw, y, mask in val_loader:
            x_norm = x_norm.to(device, non_blocking=True)
            x_raw = x_raw.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with torch.amp.autocast(device_type, enabled=use_amp):
                alpha = model(x_norm, x_raw)
                loss = criterion(alpha, y, mask, kl_weight, x_raw=x_raw)

            total_loss += loss.item()
            n_batches += 1

            probs = DualBranchUNetEDL.get_probabilities(alpha)
            all_probs.append(probs[:, 1].cpu())
            all_targets.append(y.cpu())
            all_valid.append(mask.bool().cpu())

    # Sweep thresholds
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

    avg_loss = total_loss / max(n_batches, 1)
    return best_f1, avg_loss

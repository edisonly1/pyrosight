"""Quick hyperparameter sweep — test w_fire values for 10 epochs each."""

import torch
from pyrosight.config import Config
from pyrosight.data import build_dataloaders
from pyrosight.model import UNetEDL
from pyrosight.loss import EDLSegmentationLoss
from pyrosight.train import _validate

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
device_type = "cuda" if "cuda" in device else "cpu"

cfg = Config(batch_size=16)
train_loader, val_loader, _ = build_dataloaders(cfg, num_workers=0)

weights_to_try = [5, 8, 12, 18]
results = []

for w in weights_to_try:
    print(f"\n{'='*50}")
    print(f"Testing w_fire={w}")
    print(f"{'='*50}")

    model = UNetEDL(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_filters=cfg.base_filters,
        num_groups=cfg.num_groups,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = EDLSegmentationLoss(w_fire=w, w_nofire=1.0)

    best_f1 = 0
    for epoch in range(10):
        model.train()
        kl_weight = cfg.get_kl_weight(epoch)

        for x, y, mask in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            alpha = model(x)
            loss = criterion(alpha, y, mask, kl_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        val_f1, val_loss = _validate(model, val_loader, criterion, kl_weight, device, device_type, False)
        if val_f1 > best_f1:
            best_f1 = val_f1
        print(f"  Epoch {epoch}: val_F1={val_f1:.4f} (best={best_f1:.4f})")

    results.append((w, best_f1))
    print(f"  w_fire={w} → best F1={best_f1:.4f}")

print(f"\n{'='*50}")
print("RESULTS:")
for w, f1 in sorted(results, key=lambda x: -x[1]):
    print(f"  w_fire={w:>2d} → F1={f1:.4f}")
print(f"\nBest: w_fire={max(results, key=lambda x: x[1])[0]}")

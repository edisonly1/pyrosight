"""Central configuration for PyroSight v2 training and inference."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Config:
    # Data
    in_channels: int = 12
    num_classes: int = 2
    image_size: int = 64

    # Architecture — dual-branch
    encoder_widths: List[int] = field(default_factory=lambda: [48, 72, 96])
    bottleneck_channels: int = 192
    fuel_indices: List[int] = field(default_factory=lambda: [0, 8, 9, 11])
    weather_indices: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 10])
    dropout_rate: float = 0.1

    # Deep supervision
    deep_supervision: bool = True
    aux_loss_weights: List[float] = field(default_factory=lambda: [0.3, 0.1])

    # Physics
    use_rothermel: bool = True
    physics_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "wind": 0.1, "slope": 0.05, "fuel": 0.05,
    })
    use_ca_postprocess: bool = True
    ca_steps: int = 2

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 5e-4
    batch_size: int = 32
    max_epochs: int = 200
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    grad_clip: float = 1.0

    # EDL
    kl_max: float = 0.1
    kl_anneal_epochs: int = 50

    # Class imbalance — low weight since oversampling already balances patches
    w_fire: float = 5.0
    w_nofire: float = 1.0
    dice_weight: float = 1.0

    # Early stopping
    patience: int = 30

    # Checkpointing
    checkpoint_every: int = 5
    checkpoint_dir: str = "checkpoints"

    # Dataset paths
    train_pattern: str = "data/next_day_wildfire_spread_train_*.tfrecord*"
    val_pattern: str = "data/next_day_wildfire_spread_eval_*.tfrecord*"
    test_pattern: str = "data/next_day_wildfire_spread_test_*.tfrecord*"

    # Channel ordering
    feature_keys: List[str] = field(default_factory=lambda: [
        "elevation", "th", "vs", "tmmn", "tmmx",
        "sph", "pr", "pdsi", "NDVI", "population",
        "erc", "PrevFireMask",
    ])
    label_key: str = "FireMask"

    # Per-channel normalization stats (computed from training set)
    channel_stats: List[tuple] = field(default_factory=lambda: [
        (896.44,   842.19),   # elevation
        (199.43,    71.60),   # th
        (  3.63,     1.30),   # vs
        (281.85,    18.50),   # tmmn
        (297.72,    19.46),   # tmmx
        (  0.00653,  0.00373),# sph
        (  0.318,    1.439),  # pr
        ( -0.773,    2.437),  # pdsi
        (5351.79,  2179.28),  # NDVI (raw scaled)
        ( 29.36,   190.03),   # population
        ( 53.47,    25.08),   # erc
        (  0.0,      1.0),    # PrevFireMask (ternary, not normalized)
    ])

    def get_kl_weight(self, epoch: int) -> float:
        """Quadratic KL annealing: λ = kl_max * min(1, epoch/kl_anneal_epochs)²."""
        t = min(1.0, epoch / self.kl_anneal_epochs)
        return self.kl_max * t * t

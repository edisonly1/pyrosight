"""Inference utilities with CA post-processing and Bayesian evidential fusion."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import Config
from .model import DualBranchUNetEDL, CAPostProcessor, evidential_fusion


def load_model(checkpoint_path: str, cfg: Config = None,
               device: str = "cpu") -> DualBranchUNetEDL:
    """Load a trained model from checkpoint."""
    if cfg is None:
        cfg = Config()

    model = DualBranchUNetEDL(
        widths=tuple(cfg.encoder_widths),
        bottleneck_ch=cfg.bottleneck_channels,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout_rate,  # same as training; eval() disables it
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model: DualBranchUNetEDL, x_norm: torch.Tensor,
            x_raw: torch.Tensor, device: str = "cpu",
            use_ca: bool = False, use_fusion: bool = False,
            ca_steps: int = 2) -> dict:
    """Run inference with optional physics post-processing.

    Args:
        model: Trained DualBranchUNetEDL.
        x_norm: Normalized input (B, 12, 64, 64) or (12, 64, 64).
        x_raw: Raw input (same shape).
        device: Device.
        use_ca: Apply cellular automata post-processing.
        use_fusion: Apply Bayesian evidential fusion with Rothermel.

    Returns:
        Dict with fire_prob, uncertainty, alpha.
    """
    if x_norm.dim() == 3:
        x_norm = x_norm.unsqueeze(0)
        x_raw = x_raw.unsqueeze(0)

    x_norm = x_norm.to(device)
    x_raw = x_raw.to(device)

    alpha = model(x_norm, x_raw)

    # Optional Bayesian fusion with Rothermel physics
    if use_fusion:
        physics = model.rothermel(x_raw)
        spread_rate = physics[:, 0:1]  # first channel is spread rate
        alpha = evidential_fusion(alpha, spread_rate)

    probs = DualBranchUNetEDL.get_probabilities(alpha)
    uncertainty = DualBranchUNetEDL.get_uncertainty(alpha)

    fire_prob = probs[:, 1].cpu().numpy()
    unc = uncertainty[:, 0].cpu().numpy()

    # Optional CA post-processing
    if use_ca:
        ca = CAPostProcessor(n_steps=ca_steps)
        ndvi = x_raw[:, 8:9].cpu()
        prev_fire = (x_raw[:, 11:12] > 0).float().cpu()
        fire_prob_t = torch.from_numpy(fire_prob)
        unc_t = torch.from_numpy(unc)
        fire_prob_t, unc_t = ca(fire_prob_t, unc_t, ndvi.squeeze(1), prev_fire.squeeze(1))
        fire_prob = fire_prob_t.numpy()
        unc = unc_t.numpy()

    return {
        "fire_prob": fire_prob,
        "uncertainty": unc,
        "alpha": alpha.cpu().numpy(),
    }


def render_prediction(fire_prob: np.ndarray, uncertainty: np.ndarray,
                      save_path: Optional[str] = None,
                      label: Optional[np.ndarray] = None) -> plt.Figure:
    """Render fire probability and uncertainty maps side by side."""
    n_cols = 3 if label is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))

    im0 = axes[0].imshow(fire_prob, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title("Fire Probability")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(uncertainty, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Epistemic Uncertainty")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    if label is not None:
        im2 = axes[2].imshow(label, cmap="Reds", vmin=0, vmax=1)
        axes[2].set_title("Ground Truth")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def export_jit(model: DualBranchUNetEDL, save_path: str,
               device: str = "cpu") -> None:
    """Export model as TorchScript for deployment."""
    model.eval().to(device)
    dummy_norm = torch.randn(1, 12, 64, 64, device=device)
    dummy_raw = torch.randn(1, 12, 64, 64, device=device)
    traced = torch.jit.trace(model, (dummy_norm, dummy_raw))
    traced.save(save_path)
    print(f"Exported JIT model to {save_path}")

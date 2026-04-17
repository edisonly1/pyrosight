"""Physics-informed EDL loss for wildfire segmentation.

Combines:
  - MSE Bayes risk + KL divergence (evidential deep learning)
  - Soft Dice loss (directly optimizes F1/overlap)
  - Wind-directional consistency (fire spreads downwind)
  - Slope consistency (fire spreads uphill)
  - Fuel continuity (fire requires fuel/vegetation)
  - Deep supervision (auxiliary losses at intermediate scales)
"""

import torch
import torch.nn.functional as F


class EDLPhysicsLoss:
    """Full physics-informed EDL loss with deep supervision.

    Args:
        w_fire: Class weight for fire pixels.
        w_nofire: Class weight for no-fire pixels.
        dice_weight: Weight for Dice loss component.
        aux_weights: Weights for deep supervision auxiliary losses [aux2, aux3].
        physics_weights: Dict with wind, slope, fuel loss weights.
    """

    def __init__(self, w_fire: float = 5.0, w_nofire: float = 1.0,
                 dice_weight: float = 1.0,
                 aux_weights: tuple = (0.3, 0.1),
                 physics_weights: dict = None):
        self.w_fire = w_fire
        self.w_nofire = w_nofire
        self.dice_weight = dice_weight
        self.aux_weights = aux_weights
        self.pw = physics_weights or {"wind": 0.1, "slope": 0.05, "fuel": 0.05}

        # Sobel kernels for spatial gradients (registered at first call)
        self._sobel_x = None
        self._sobel_y = None

    def __call__(self, outputs, target, valid_mask, kl_weight,
                 x_raw=None):
        """Compute full loss.

        Args:
            outputs: alpha (B,K,H,W) or tuple (alpha_main, alpha_aux2, alpha_aux3).
            target: (B, H, W) ground truth {0, 1}.
            valid_mask: (B, H, W) boolean mask.
            kl_weight: Current KL annealing weight.
            x_raw: (B, 12, H, W) raw inputs for physics losses.
        """
        if isinstance(outputs, tuple):
            alpha_main, alpha_aux2, alpha_aux3 = outputs
            loss = self._compute_edl_loss(alpha_main, target, valid_mask, kl_weight)
            loss += self.aux_weights[0] * self._compute_edl_loss(
                alpha_aux2, target, valid_mask, kl_weight)
            loss += self.aux_weights[1] * self._compute_edl_loss(
                alpha_aux3, target, valid_mask, kl_weight)
        else:
            alpha_main = outputs
            loss = self._compute_edl_loss(alpha_main, target, valid_mask, kl_weight)

        # Physics-informed losses (annealed with KL weight)
        if x_raw is not None and kl_weight > 0:
            phys_scale = min(kl_weight / 0.1, 1.0)  # ramp with KL annealing
            loss += phys_scale * self._physics_losses(
                alpha_main, x_raw, target, valid_mask)

        return loss

    def _compute_edl_loss(self, alpha, target, valid_mask, kl_weight):
        """EDL MSE Bayes risk + KL + Dice for a single alpha output."""
        B, K, H, W = alpha.shape

        alpha = alpha.float().permute(0, 2, 3, 1)  # (B, H, W, K)
        y = F.one_hot(target.long(), K).float()

        # MSE Bayes Risk
        S = alpha.sum(dim=-1, keepdim=True)
        p_hat = alpha / S
        mse = (y - p_hat) ** 2 + alpha * (S - alpha) / (S ** 2 * (S + 1))
        mse = mse.sum(dim=-1)

        # KL Divergence
        kl = self._kl_divergence(alpha, y, K)

        # Class weighting + masking
        weights = torch.where(target == 1, self.w_fire, self.w_nofire)
        per_pixel = weights * (mse + kl_weight * kl) * valid_mask.float()
        num_valid = valid_mask.sum().clamp(min=1)
        edl_loss = per_pixel.sum() / num_valid

        # Soft Dice
        p_fire = p_hat[..., 1]
        y_fire = y[..., 1]
        mask_f = valid_mask.float()
        intersection = (p_fire * y_fire * mask_f).sum()
        union = (p_fire * mask_f).sum() + (y_fire * mask_f).sum()
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)

        return edl_loss + self.dice_weight * dice_loss

    def _physics_losses(self, alpha, x_raw, target, valid_mask):
        """Compute physics-informed loss terms from raw inputs."""
        device = alpha.device
        self._ensure_sobel(device)

        # Extract raw channels
        elevation = x_raw[:, 0:1]
        wind_dir = x_raw[:, 1:2]
        wind_speed = x_raw[:, 2:3]
        ndvi = x_raw[:, 8:9]
        prev_fire = (x_raw[:, 11:12] > 0).float()

        # Fire probability from alpha
        S = alpha.sum(dim=1, keepdim=True)
        p_fire = alpha[:, 1:2] / S  # (B, 1, H, W)

        # Elevation gradients
        dz_dx = F.conv2d(elevation, self._sobel_x, padding=1)
        dz_dy = F.conv2d(elevation, self._sobel_y, padding=1)

        # Wind components
        wind_rad = wind_dir * (3.14159265 / 180.0)
        wind_dx = -torch.sin(wind_rad) * wind_speed
        wind_dy = -torch.cos(wind_rad) * wind_speed

        loss = torch.tensor(0.0, device=device)

        # --- Wind consistency ---
        loss += self.pw["wind"] * _wind_consistency(
            p_fire, wind_dx, wind_dy, prev_fire, valid_mask,
            self._sobel_x, self._sobel_y)

        # --- Slope consistency ---
        loss += self.pw["slope"] * _slope_consistency(
            p_fire, dz_dx, dz_dy, wind_speed, prev_fire, valid_mask,
            self._sobel_x, self._sobel_y)

        # --- Fuel continuity ---
        loss += self.pw["fuel"] * _fuel_continuity(
            p_fire, ndvi, prev_fire, valid_mask)

        return loss

    def _ensure_sobel(self, device):
        if self._sobel_x is None or self._sobel_x.device != device:
            self._sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=torch.float32, device=device
            ).reshape(1, 1, 3, 3)
            self._sobel_y = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=torch.float32, device=device
            ).reshape(1, 1, 3, 3)

    def _kl_divergence(self, alpha, y, K):
        alpha_tilde = y + (1.0 - y) * alpha + 1e-6
        S_tilde = alpha_tilde.sum(dim=-1)
        kl = (
            torch.lgamma(S_tilde)
            - torch.lgamma(torch.tensor(float(K), device=alpha.device))
            - torch.lgamma(alpha_tilde).sum(dim=-1)
            + ((alpha_tilde - 1.0) *
               (torch.digamma(alpha_tilde) - torch.digamma(S_tilde.unsqueeze(-1)))
               ).sum(dim=-1)
        )
        return kl


# ---------------------------------------------------------------------------
# Physics loss functions
# ---------------------------------------------------------------------------

def _wind_consistency(p_fire, wind_dx, wind_dy, prev_fire, valid_mask,
                      sobel_x, sobel_y):
    """Penalize fire predictions that appear upwind of existing fire."""
    dp_dx = F.conv2d(p_fire, sobel_x, padding=1)
    dp_dy = F.conv2d(p_fire, sobel_y, padding=1)

    # Dot product: positive = downwind spread (good), negative = upwind (bad)
    alignment = dp_dx * wind_dx + dp_dy * wind_dy

    # Only penalize near existing fire perimeters
    near_fire = F.max_pool2d(prev_fire, 3, stride=1, padding=1) > 0
    penalty = F.relu(-alignment) * near_fire.float() * valid_mask.float().unsqueeze(1)

    return penalty.sum() / valid_mask.sum().clamp(min=1)


def _slope_consistency(p_fire, dz_dx, dz_dy, wind_speed, prev_fire, valid_mask,
                       sobel_x, sobel_y):
    """Penalize fire spreading downhill when wind is calm."""
    dp_dx = F.conv2d(p_fire, sobel_x, padding=1)
    dp_dy = F.conv2d(p_fire, sobel_y, padding=1)

    # Downhill = fire gradient opposes elevation gradient
    downhill = -(dp_dx * dz_dx + dp_dy * dz_dy)

    # Only penalize when wind is calm
    calm = (wind_speed < 2.0).float()
    near_fire = F.max_pool2d(prev_fire, 3, stride=1, padding=1) > 0
    penalty = F.relu(downhill) * calm * near_fire.float() * valid_mask.float().unsqueeze(1)

    return penalty.sum() / valid_mask.sum().clamp(min=1)


def _fuel_continuity(p_fire, ndvi, prev_fire, valid_mask):
    """Penalize fire predictions in areas with no fuel (low NDVI)."""
    no_fuel = (ndvi < 500.0).float()  # raw NDVI scale, ~500 ≈ 0.05 in normalized
    not_burning = (1.0 - prev_fire)
    penalty = p_fire * no_fuel * not_burning * valid_mask.float().unsqueeze(1)

    return penalty.sum() / valid_mask.sum().clamp(min=1)

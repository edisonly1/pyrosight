"""Dual-Branch U-Net with CAFIM, Rothermel Physics, and EDL Head.

Physics-informed architecture for wildfire spread prediction with
calibrated uncertainty maps via Evidential Deep Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _gn_groups(channels: int, target: int = 16) -> int:
    """Find largest valid GroupNorm group count ≤ target."""
    for g in range(target, 0, -1):
        if channels % g == 0:
            return g
    return 1


class BranchConvBlock(nn.Module):
    """Two convolutions with GroupNorm, GELU, and residual connection.

    Supports 3x3 (fuel/terrain) and 5x5 (weather) kernel sizes.
    5x5 uses depthwise-separable convolution to save parameters.
    Adds a residual skip when in_ch == out_ch for better gradient flow.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        groups = _gn_groups(out_ch)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_ch)

        if kernel_size == 5:
            # Depthwise-separable for 5x5 to cut params ~5x
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 5, padding=2, groups=out_ch, bias=False),
                nn.Conv2d(out_ch, out_ch, 1, bias=False),
            )
        else:
            self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad, bias=False)

        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Residual connection: identity when dims match, 1x1 projection otherwise
        if in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.act(self.gn2(self.conv2(x)))
        x = self.dropout(x)
        return x + identity


class CAFIM(nn.Module):
    """Cross-Attentive Feature Interaction Module.

    Each branch generates a spatial attention gate from the OTHER branch,
    then features are fused via concatenation + projection for skip connections.
    """

    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 4, 8)

        # Fuel attends to weather (learns "where weather amplifies fuel risk")
        self.gate_fuel = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        # Weather attends to fuel (learns "where terrain affects weather impact")
        self.gate_wx = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        # Fuse attended branches into a single skip connection
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(_gn_groups(channels), channels),
            nn.GELU(),
        )

        # Initialize gates near 0.5 (sigmoid(0)) for balanced start
        for gate in [self.gate_fuel, self.gate_wx]:
            nn.init.normal_(gate[-2].weight, std=0.01)

    def forward(self, fuel: torch.Tensor, wx: torch.Tensor):
        """Returns (fuel_out, wx_out, fused_skip)."""
        fuel_out = fuel + fuel * self.gate_fuel(wx)
        wx_out = wx + wx * self.gate_wx(fuel)
        fused = self.fuse(torch.cat([fuel_out, wx_out], dim=1))
        return fuel_out, wx_out, fused


# ---------------------------------------------------------------------------
# Rothermel fire spread physics (no learned parameters)
# ---------------------------------------------------------------------------

class RothermelPhysics(nn.Module):
    """Deterministic Rothermel-inspired fire spread physics branch.

    Computes physically-derived features from raw (unnormalized) inputs:
      - spread_rate: (B, 1, H, W) — fire spread rate proxy
      - wind_bias:   (B, 2, H, W) — directional wind vector
      - slope_bias:  (B, 2, H, W) — uphill spread tendency

    No learned parameters — pure physics computation.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels as buffers (not parameters)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32
        ).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32
        ).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw: (B, 12, 64, 64) raw unnormalized inputs.

        Returns:
            (B, 5, H, W) physics features: [spread_rate, wind_dx, wind_dy, slope_dx, slope_dy]
        """
        elevation = raw[:, 0:1]    # meters
        wind_dir = raw[:, 1:2]     # degrees azimuth
        wind_speed = raw[:, 2:3]   # m/s
        humidity = raw[:, 5:6]     # kg/kg
        erc = raw[:, 10:11]        # energy release component

        # --- Slope from elevation gradient (Sobel) ---
        dz_dx = F.conv2d(elevation, self.sobel_x, padding=1)
        dz_dy = F.conv2d(elevation, self.sobel_y, padding=1)
        slope_mag = torch.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1e-8)

        # Rothermel slope factor: φ_s = 5.275 * tan²(slope)
        phi_s = 5.275 * slope_mag ** 2

        # --- Wind factor ---
        wind_rad = wind_dir * (3.14159265 / 180.0)
        wind_dx = -torch.sin(wind_rad) * wind_speed
        wind_dy = -torch.cos(wind_rad) * wind_speed

        # Rothermel wind factor (simplified): φ_w ∝ wind_speed^1.5
        ws_safe = wind_speed.clamp(min=0)
        phi_w = 0.4 * ws_safe ** 1.5

        # --- Fuel moisture / reaction intensity proxy ---
        fuel_dryness = torch.sigmoid(erc / 50.0)
        hum_max = humidity.abs().amax(dim=(2, 3), keepdim=True) + 1e-8
        moisture = 1.0 - 0.5 * humidity.clamp(min=0) / hum_max
        IR = fuel_dryness * moisture

        # --- Rothermel spread rate ---
        Q_ig = 250.0 + 1116.0 * (1.0 - fuel_dryness)
        R = IR * (1.0 + phi_w + phi_s) / (Q_ig / 250.0 + 1e-8)
        R = R.clamp(-100, 100)  # prevent extreme values before sigmoid

        # Normalize to [0, 1]
        spread_rate = torch.sigmoid(R - R.mean(dim=(2, 3), keepdim=True))

        # --- Normalize directional biases ---
        wind_mag = torch.sqrt(wind_dx ** 2 + wind_dy ** 2 + 1e-8)
        wind_bias = torch.cat([wind_dx, wind_dy], dim=1) / (wind_mag.amax(dim=(2, 3), keepdim=True) + 1e-8)

        slope_bias = torch.cat([dz_dx, dz_dy], dim=1)
        slope_bias = slope_bias / (slope_bias.abs().amax(dim=(2, 3), keepdim=True) + 1e-8)

        return torch.cat([spread_rate, wind_bias, slope_bias], dim=1)  # (B, 5, H, W)


# ---------------------------------------------------------------------------
# Wind-aligned convolution
# ---------------------------------------------------------------------------

class WindAlignedConv(nn.Module):
    """Directionally-biased convolution modulated by wind direction.

    Applies standard convolution then adds a wind-direction-weighted residual,
    so the output is subtly biased toward the downwind direction.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn = nn.GroupNorm(_gn_groups(channels), channels)
        self.scale = nn.Parameter(torch.tensor(0.1))  # learnable modulation strength

    def forward(self, x: torch.Tensor, wind_dx: torch.Tensor,
                wind_dy: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Directional modulation: shift feature response in wind direction
        # wind_dx/dy are (B, 1, H, W) — broadcast across channels
        out = out + self.scale * (out * wind_dx + out * wind_dy)
        return F.gelu(self.gn(out))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DualBranchUNetEDL(nn.Module):
    """Dual-branch U-Net with CAFIM fusion, Rothermel physics, and EDL head.

    Input:  x_norm (B, 12, 64, 64) — normalized inputs
            x_raw  (B, 12, 64, 64) — raw inputs for physics branch
    Output: alpha  (B, 2, 64, 64)  — Dirichlet concentration params (α ≥ 1)
            During training: (alpha_main, alpha_aux2, alpha_aux3) for deep supervision
    """

    # Channel indices
    FUEL_IDX = [0, 8, 9, 11]       # elevation, NDVI, population, PrevFireMask
    WEATHER_IDX = [1, 2, 3, 4, 5, 6, 7, 10]  # th, vs, tmmn, tmmx, sph, pr, pdsi, erc

    def __init__(self, widths=(48, 72, 96), bottleneck_ch: int = 192,
                 num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        w1, w2, w3 = widths
        self.num_classes = num_classes

        # --- Dual-branch encoder ---
        self.fuel_enc1 = BranchConvBlock(4, w1, kernel_size=3, dropout=dropout)
        self.fuel_enc2 = BranchConvBlock(w1, w2, kernel_size=3, dropout=dropout)
        self.fuel_enc3 = BranchConvBlock(w2, w3, kernel_size=3, dropout=dropout)

        self.wx_enc1 = BranchConvBlock(8, w1, kernel_size=5, dropout=dropout)
        self.wx_enc2 = BranchConvBlock(w1, w2, kernel_size=5, dropout=dropout)
        self.wx_enc3 = BranchConvBlock(w2, w3, kernel_size=5, dropout=dropout)

        self.pool = nn.MaxPool2d(2)

        # --- CAFIM cross-attention at each scale ---
        self.cafim1 = CAFIM(w1)
        self.cafim2 = CAFIM(w2)
        self.cafim3 = CAFIM(w3)

        # --- Bottleneck ---
        self.bottleneck = BranchConvBlock(w3 * 2, bottleneck_ch, kernel_size=3, dropout=dropout)

        # --- Shared decoder ---
        self.up3 = nn.ConvTranspose2d(bottleneck_ch, w3, 2, stride=2)
        self.dec3 = BranchConvBlock(w3 * 2, w3, kernel_size=3, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(w3, w2, 2, stride=2)
        self.dec2 = BranchConvBlock(w2 * 2, w2, kernel_size=3, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(w2, w1, 2, stride=2)
        self.dec1 = BranchConvBlock(w1 * 2, w1, kernel_size=3, dropout=dropout)

        # --- Wind-aligned convolution in decoder ---
        self.wind_conv = WindAlignedConv(w2)

        # --- Rothermel physics branch ---
        self.rothermel = RothermelPhysics()

        # Physics fusion: cat(decoder_output, physics_features) → decoder width
        self.physics_fuse = nn.Sequential(
            nn.Conv2d(w1 + 5, w1, 1, bias=False),
            nn.GroupNorm(_gn_groups(w1), w1),
            nn.GELU(),
        )

        # --- EDL head ---
        self.evidence_conv = nn.Conv2d(w1, num_classes, 1)
        self.softplus = nn.Softplus()

        # --- Deep supervision auxiliary heads ---
        self.aux_head3 = nn.Conv2d(w3, num_classes, 1)
        self.aux_head2 = nn.Conv2d(w2, num_classes, 1)

        # Initialize evidence heads with small weights so early predictions
        # are near-uniform (low evidence), matching the KL annealing ramp-up
        for head in [self.evidence_conv, self.aux_head3, self.aux_head2]:
            nn.init.normal_(head.weight, std=0.01)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(self, x_norm: torch.Tensor,
                x_raw: torch.Tensor) -> torch.Tensor:
        # --- Channel split ---
        fuel_x = x_norm[:, self.FUEL_IDX]        # (B, 4, 64, 64)
        wx_x = x_norm[:, self.WEATHER_IDX]        # (B, 8, 64, 64)

        # --- Encoder with CAFIM ---
        f1 = self.fuel_enc1(fuel_x)
        w1 = self.wx_enc1(wx_x)
        f1, w1, skip1 = self.cafim1(f1, w1)        # 64×64

        f2 = self.fuel_enc2(self.pool(f1))
        w2 = self.wx_enc2(self.pool(w1))
        f2, w2, skip2 = self.cafim2(f2, w2)        # 32×32

        f3 = self.fuel_enc3(self.pool(f2))
        w3 = self.wx_enc3(self.pool(w2))
        f3, w3, skip3 = self.cafim3(f3, w3)        # 16×16

        # --- Bottleneck ---
        b = self.bottleneck(torch.cat([self.pool(f3), self.pool(w3)], dim=1))  # 8×8

        # --- Decoder ---
        d3 = self.dec3(torch.cat([self.up3(b), skip3], dim=1))     # 16×16

        d2 = self.up2(d3)
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.dec2(d2)                                          # 32×32

        # Wind-aligned convolution at 32×32 scale
        # Detach physics: Rothermel has no learned params, so no gradients needed
        physics = self.rothermel(x_raw).detach()
        wind_dx = F.interpolate(physics[:, 1:2], size=d2.shape[2:], mode='bilinear', align_corners=False)
        wind_dy = F.interpolate(physics[:, 2:3], size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.wind_conv(d2, wind_dx, wind_dy)

        d1 = self.dec1(torch.cat([self.up1(d2), skip1], dim=1))    # 64×64

        # --- Physics fusion ---
        d1 = torch.cat([d1, physics], dim=1)       # (B, w1+5, 64, 64)
        d1 = self.physics_fuse(d1)                  # (B, w1, 64, 64)

        # --- EDL head ---
        alpha_main = self.softplus(self.evidence_conv(d1)) + 1.0

        if self.training:
            # Deep supervision: auxiliary EDL heads at intermediate scales, upsampled to 64×64
            aux3 = self.softplus(self.aux_head3(d3)) + 1.0
            aux3 = F.interpolate(aux3, size=(64, 64), mode='bilinear', align_corners=False)

            aux2 = self.softplus(self.aux_head2(d2)) + 1.0
            aux2 = F.interpolate(aux2, size=(64, 64), mode='bilinear', align_corners=False)

            return alpha_main, aux2, aux3

        return alpha_main

    # --- Static utility methods (unchanged from v1) ---

    @staticmethod
    def get_probabilities(alpha: torch.Tensor) -> torch.Tensor:
        """Expected class probabilities: E[π_k] = α_k / S."""
        S = alpha.sum(dim=1, keepdim=True)
        return alpha / S

    @staticmethod
    def get_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
        """Epistemic uncertainty (vacuity): u = K / S."""
        K = alpha.shape[1]
        S = alpha.sum(dim=1, keepdim=True)
        return K / S

    @staticmethod
    def get_evidence(alpha: torch.Tensor) -> torch.Tensor:
        """Evidence: e_k = α_k - 1."""
        return alpha - 1.0


# ---------------------------------------------------------------------------
# Inference post-processing
# ---------------------------------------------------------------------------

class CAPostProcessor:
    """Cellular Automata post-processing for spatial coherence.

    Applied at inference only (non-differentiable).
    Rules:
      1. Suppress fire in no-fuel areas (low NDVI)
      2. Suppress isolated single-pixel fire predictions
      3. Increase uncertainty for CA-modified pixels
    """

    def __init__(self, n_steps: int = 2, ndvi_threshold: float = 500.0):
        self.n_steps = n_steps
        self.ndvi_threshold = ndvi_threshold  # raw NDVI scale: 500 ≈ 0.05 real NDVI

    def __call__(self, fire_prob, uncertainty, ndvi, prev_fire):
        p = fire_prob.clone()
        u = uncertainty.clone()
        ones_kernel = torch.ones(1, 1, 3, 3, device=p.device) / 9.0

        for _ in range(self.n_steps):
            # Neighborhood smoothing
            p_in = p.unsqueeze(1) if p.dim() == 3 else p
            p_smooth = F.conv2d(p_in, ones_kernel, padding=1)
            if p.dim() == 3:
                p_smooth = p_smooth.squeeze(1)

            # Suppress fire where no fuel (raw NDVI scale)
            no_fuel = ndvi < self.ndvi_threshold
            not_burning = prev_fire < 0.5
            p_smooth = torch.where(no_fuel & not_burning, p_smooth * 0.1, p_smooth)

            # Suppress isolated predictions
            fire_count = F.conv2d(
                (p > 0.3).float().unsqueeze(1), ones_kernel, padding=1
            ).squeeze(1)
            isolated = fire_count < 0.2
            p_smooth = torch.where(isolated, p_smooth * 0.5, p_smooth)

            # Increase uncertainty where predictions changed significantly
            modified = (p - p_smooth).abs() > 0.1
            u = torch.where(modified, u * 1.5, u)

            p = p_smooth

        return p, u


def evidential_fusion(alpha_nn: torch.Tensor,
                      spread_rate: torch.Tensor) -> torch.Tensor:
    """Bayesian fusion of neural EDL with Rothermel physics via evidence addition.

    Args:
        alpha_nn: (B, 2, H, W) neural Dirichlet params.
        spread_rate: (B, 1, H, W) Rothermel spread rate in [0, 1].

    Returns:
        (B, 2, H, W) fused Dirichlet params.
    """
    # Convert spread rate to weak evidence
    e_fire = spread_rate * 2.0
    e_nofire = (1.0 - spread_rate) * 2.0

    # Additive evidence fusion (Dempster's rule for Dirichlet)
    e_nn = alpha_nn - 1.0
    e_fused = e_nn + torch.cat([e_nofire, e_fire], dim=1)

    return e_fused + 1.0

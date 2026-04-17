"""DualBranchUNetEDL v2 — Improved architecture.

Changes from v1:
- Wider encoder: [64, 96, 128] instead of [48, 72, 96]
- Bottleneck self-attention (lightweight spatial attention)
- Learned weather channel upsampler (super-resolves 4km GRIDMET to 1km)
- Squeeze-and-Excitation channel attention in decoder blocks
- Stochastic depth in encoder for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, groups: int = 16) -> nn.GroupNorm:
    for g in range(groups, 0, -1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


class SE(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, ch: int, ratio: int = 4):
        super().__init__()
        mid = max(ch // ratio, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class ConvBlock(nn.Module):
    """Conv block with GroupNorm, GELU, residual, SE attention, stochastic depth."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.gn1 = _gn(out_ch)

        if kernel_size == 5:
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 5, padding=2, groups=out_ch, bias=False),
                nn.Conv2d(out_ch, out_ch, 1, bias=False),
            )
        else:
            self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad, bias=False)

        self.gn2 = _gn(out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.se = SE(out_ch)

        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.gn1(self.conv1(x)))
        out = self.act(self.gn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.se(out)

        # Stochastic depth
        if self.training and self.drop_path > 0:
            if torch.rand(1).item() < self.drop_path:
                return identity

        return out + identity


class CAFIM(nn.Module):
    """Cross-Attentive Feature Interaction Module (same as v1)."""
    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 4, 8)
        self.gate_fuel = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(mid, 1, 1, bias=False), nn.Sigmoid(),
        )
        self.gate_wx = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(mid, 1, 1, bias=False), nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            _gn(channels), nn.GELU(),
        )
        for gate in [self.gate_fuel, self.gate_wx]:
            nn.init.normal_(gate[-2].weight, std=0.01)

    def forward(self, fuel, wx):
        fuel_out = fuel + fuel * self.gate_fuel(wx)
        wx_out = wx + wx * self.gate_wx(fuel)
        fused = self.fuse(torch.cat([fuel_out, wx_out], dim=1))
        return fuel_out, wx_out, fused


class SpatialAttention(nn.Module):
    """Lightweight spatial self-attention for bottleneck."""
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = _gn(channels)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)

        return x + self.norm(self.proj(out))


class WeatherUpsampler(nn.Module):
    """Learned upsampler for coarse weather channels.

    GRIDMET is 4km while the model expects 1km. Instead of bilinear interpolation,
    this learns to super-resolve weather features, reducing grid artifacts.
    """
    def __init__(self, in_ch: int = 8, out_ch: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            _gn(32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            _gn(32), nn.GELU(),
            nn.Conv2d(32, out_ch, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)  # Residual: learn the correction, not the full signal


# Import physics from original model
from .model import RothermelPhysics, WindAlignedConv


class DualBranchUNetEDL_v2(nn.Module):
    """Improved DualBranchUNetEDL with wider encoder, attention, and weather upsampler.

    Input:  x_norm (B, 12, 64, 64), x_raw (B, 12, 64, 64)
    Output: alpha (B, 2, 64, 64) Dirichlet params
            Training: (alpha_main, alpha_aux2, alpha_aux3)
    """

    FUEL_IDX = [0, 8, 9, 11]
    WEATHER_IDX = [1, 2, 3, 4, 5, 6, 7, 10]

    def __init__(self, widths=(64, 96, 128), bottleneck_ch: int = 256,
                 num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        w1, w2, w3 = widths
        self.num_classes = num_classes

        # Weather channel upsampler (learns to smooth 4km grid artifacts)
        self.wx_upsample = WeatherUpsampler(8, 8)

        # Dual-branch encoder with stochastic depth
        self.fuel_enc1 = ConvBlock(4, w1, 3, dropout, drop_path=0.0)
        self.fuel_enc2 = ConvBlock(w1, w2, 3, dropout, drop_path=0.05)
        self.fuel_enc3 = ConvBlock(w2, w3, 3, dropout, drop_path=0.1)

        self.wx_enc1 = ConvBlock(8, w1, 5, dropout, drop_path=0.0)
        self.wx_enc2 = ConvBlock(w1, w2, 5, dropout, drop_path=0.05)
        self.wx_enc3 = ConvBlock(w2, w3, 5, dropout, drop_path=0.1)

        self.pool = nn.MaxPool2d(2)

        # CAFIM at each scale
        self.cafim1 = CAFIM(w1)
        self.cafim2 = CAFIM(w2)
        self.cafim3 = CAFIM(w3)

        # Bottleneck with spatial attention
        self.bottleneck = ConvBlock(w3 * 2, bottleneck_ch, 3, dropout)
        self.bottleneck_attn = SpatialAttention(bottleneck_ch, heads=4)

        # Decoder with SE attention
        self.up3 = nn.ConvTranspose2d(bottleneck_ch, w3, 2, stride=2)
        self.dec3 = ConvBlock(w3 * 2, w3, 3, dropout)

        self.up2 = nn.ConvTranspose2d(w3, w2, 2, stride=2)
        self.dec2 = ConvBlock(w2 * 2, w2, 3, dropout)

        self.up1 = nn.ConvTranspose2d(w2, w1, 2, stride=2)
        self.dec1 = ConvBlock(w1 * 2, w1, 3, dropout)

        # Wind-aligned conv
        self.wind_conv = WindAlignedConv(w2)

        # Physics
        self.rothermel = RothermelPhysics()
        self.physics_fuse = nn.Sequential(
            nn.Conv2d(w1 + 5, w1, 1, bias=False),
            _gn(w1), nn.GELU(),
        )

        # EDL head
        self.evidence_conv = nn.Conv2d(w1, num_classes, 1)
        self.softplus = nn.Softplus()

        # Deep supervision
        self.aux_head3 = nn.Conv2d(w3, num_classes, 1)
        self.aux_head2 = nn.Conv2d(w2, num_classes, 1)

        # Init evidence heads small
        for head in [self.evidence_conv, self.aux_head3, self.aux_head2]:
            nn.init.normal_(head.weight, std=0.01)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(self, x_norm, x_raw):
        fuel_x = x_norm[:, self.FUEL_IDX]
        wx_x = x_norm[:, self.WEATHER_IDX]

        # Learned weather upsampling (reduces 4km grid artifacts)
        wx_x = self.wx_upsample(wx_x)

        # Encoder
        f1 = self.fuel_enc1(fuel_x)
        w1 = self.wx_enc1(wx_x)
        f1, w1, skip1 = self.cafim1(f1, w1)

        f2 = self.fuel_enc2(self.pool(f1))
        w2 = self.wx_enc2(self.pool(w1))
        f2, w2, skip2 = self.cafim2(f2, w2)

        f3 = self.fuel_enc3(self.pool(f2))
        w3 = self.wx_enc3(self.pool(w2))
        f3, w3, skip3 = self.cafim3(f3, w3)

        # Bottleneck with attention
        b = self.bottleneck(torch.cat([self.pool(f3), self.pool(w3)], dim=1))
        b = self.bottleneck_attn(b)

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b), skip3], dim=1))

        d2 = self.dec2(torch.cat([self.up2(d3), skip2], dim=1))

        physics = self.rothermel(x_raw).detach()
        wind_dx = F.interpolate(physics[:, 1:2], size=d2.shape[2:], mode='bilinear', align_corners=False)
        wind_dy = F.interpolate(physics[:, 2:3], size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.wind_conv(d2, wind_dx, wind_dy)

        d1 = self.dec1(torch.cat([self.up1(d2), skip1], dim=1))

        # Physics fusion
        d1 = self.physics_fuse(torch.cat([d1, physics], dim=1))

        # EDL
        alpha_main = self.softplus(self.evidence_conv(d1)) + 1.0

        if self.training:
            aux3 = self.softplus(self.aux_head3(d3)) + 1.0
            aux3 = F.interpolate(aux3, size=(64, 64), mode='bilinear', align_corners=False)
            aux2 = self.softplus(self.aux_head2(d2)) + 1.0
            aux2 = F.interpolate(aux2, size=(64, 64), mode='bilinear', align_corners=False)
            return alpha_main, aux2, aux3

        return alpha_main

    @staticmethod
    def get_probabilities(alpha):
        S = alpha.sum(dim=1, keepdim=True)
        return alpha / S

    @staticmethod
    def get_uncertainty(alpha):
        K = alpha.shape[1]
        S = alpha.sum(dim=1, keepdim=True)
        return K / S

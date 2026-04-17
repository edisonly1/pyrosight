"""Tile assembly — builds a complete 12-channel input from any US lat/lng.

This is the core of the live data pipeline. Given a center coordinate,
it computes a 64×64 km bounding box, fetches all 12 channels from their
respective sources, normalizes them, and returns tensors ready for inference.
"""

import math
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import torch

from ..config import Config
from .gridmet import fetch_gridmet, fetch_gridmet_drought
from .firms import fetch_active_fires
from .dem import extract_dem
from .ndvi import fetch_ndvi
from .population import extract_population


def compute_bbox(lat: float, lng: float) -> Tuple[float, float, float, float]:
    """Compute a 64×64 km bounding box centered on (lat, lng).

    Each pixel = 1 km, so the box extends ±32 km in each direction.
    32 km ≈ 0.288 degrees latitude.

    Returns:
        (west, south, east, north) in degrees
    """
    half_lat = 0.288  # ~32 km in latitude degrees
    half_lng = half_lat / math.cos(math.radians(lat))  # Adjust for latitude

    return (
        lng - half_lng,  # west
        lat - half_lat,  # south
        lng + half_lng,  # east
        lat + half_lat,  # north
    )


def build_tile(
    lat: float,
    lng: float,
    date: str = None,
    cfg: Config = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Build a complete 12-channel tile for model inference.

    Args:
        lat: Center latitude
        lng: Center longitude
        date: Date string "YYYY-MM-DD" (default: today)
        cfg: Config instance (default: Config())

    Returns:
        x_norm: (12, 64, 64) tensor, z-score normalized for neural network
        x_raw: (12, 64, 64) tensor, raw physical units for Rothermel physics
        meta: dict with bbox, date, data freshness info
    """
    if cfg is None:
        cfg = Config()
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    bbox = compute_bbox(lat, lng)

    # Use yesterday's weather (GRIDMET has ~2 day latency)
    dt = datetime.strptime(date, "%Y-%m-%d")
    weather_date = (dt - timedelta(days=2)).strftime("%Y-%m-%d")
    fire_date = (dt - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"[tile] Building tile for ({lat:.3f}, {lng:.3f}) on {date}")
    print(f"[tile] Bbox: {bbox}")
    print(f"[tile] Weather date: {weather_date}, Fire date: {fire_date}")

    # Fetch all 12 channels
    channels = {}

    # Static layers
    print("[tile] Fetching elevation...")
    channels["elevation"] = extract_dem(bbox)

    print("[tile] Fetching population...")
    channels["population"] = extract_population(bbox)

    # Daily weather (GRIDMET)
    for var in ["th", "vs", "tmmn", "tmmx", "sph", "pr", "erc"]:
        print(f"[tile] Fetching GRIDMET {var}...")
        try:
            channels[var] = fetch_gridmet(var, weather_date, bbox)
        except Exception as e:
            print(f"[tile] Warning: GRIDMET {var} failed ({e}), using zeros")
            channels[var] = np.zeros((64, 64), dtype=np.float32)

    # Drought index (5-day lag)
    print("[tile] Fetching PDSI...")
    try:
        channels["pdsi"] = fetch_gridmet_drought(weather_date, bbox)
    except Exception as e:
        print(f"[tile] Warning: PDSI failed ({e}), using zeros")
        channels["pdsi"] = np.zeros((64, 64), dtype=np.float32)

    # Vegetation (8-day composite — use most recent)
    print("[tile] Fetching NDVI...")
    channels["NDVI"] = fetch_ndvi(bbox)

    # Fire detections from previous day
    print("[tile] Fetching fire detections...")
    channels["PrevFireMask"] = fetch_active_fires(bbox, fire_date)

    # Stack into (12, 64, 64) in the correct channel order
    raw_stack = []
    for key in cfg.feature_keys:
        ch = channels[key]
        # Replace NaN
        if np.isnan(ch).any():
            ch = np.nan_to_num(ch, nan=0.0)
        raw_stack.append(ch)

    x_raw_np = np.stack(raw_stack, axis=0).astype(np.float32)

    # Z-score normalize using training statistics
    norm_stack = []
    for i, key in enumerate(cfg.feature_keys):
        ch = x_raw_np[i].copy()
        mean, std = cfg.channel_stats[i]
        if key == "PrevFireMask":
            ch = (ch > 0).astype(np.float32)
        else:
            ch = (ch - mean) / (std + 1e-8)
            ch = np.clip(ch, -10.0, 10.0)
        norm_stack.append(ch)

    x_norm_np = np.stack(norm_stack, axis=0).astype(np.float32)

    # Convert to tensors
    x_norm = torch.from_numpy(x_norm_np)
    x_raw = torch.from_numpy(x_raw_np)

    meta = {
        "bbox": {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]},
        "center": {"lat": lat, "lng": lng},
        "date": date,
        "weather_date": weather_date,
        "fire_date": fire_date,
    }

    print("[tile] Tile built successfully.")
    return x_norm, x_raw, meta

"""NASA FIRMS active fire detection fetcher.

Fetches VIIRS/MODIS fire detections from NASA's Fire Information for
Resource Management System and rasterizes them onto a 64×64 grid.

API: https://firms.modaps.eosdis.nasa.gov/api/
Free tier: requires API key (instant signup), 10 requests/minute.
"""

import os
from typing import Tuple

import numpy as np
import requests

# Get API key from environment, or use the demo key (limited)
_API_KEY = os.environ.get("FIRMS_API_KEY", "DEMO_KEY")

# FIRMS API endpoint
_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def fetch_active_fires(
    bbox: Tuple[float, float, float, float],
    date: str,
    source: str = "VIIRS_SNPP_NRT",
) -> np.ndarray:
    """Fetch fire detections and rasterize to a 64×64 binary grid.

    Args:
        bbox: (west, south, east, north) in degrees
        date: Date string "YYYY-MM-DD"
        source: FIRMS data source (VIIRS_SNPP_NRT, MODIS_NRT, etc.)

    Returns:
        numpy array (64, 64) float32, 1.0 where fire detected, 0.0 otherwise
    """
    west, south, east, north = bbox

    # FIRMS API: /api/area/csv/{key}/{source}/{west},{south},{east},{north}/{days}/{date}
    url = f"{_BASE_URL}/{_API_KEY}/{source}/{west},{south},{east},{north}/1/{date}"

    grid = np.zeros((64, 64), dtype=np.float32)

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 403:
            # API key issue — try with MAP_KEY source format
            print(f"[FIRMS] API key may be invalid (403). Using empty fire mask.")
            return grid
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[FIRMS] Request failed: {e}. Using empty fire mask.")
        return grid

    lines = resp.text.strip().split("\n")
    if len(lines) < 2:
        return grid  # No detections (just header)

    # Parse header to find lat/lng columns
    header = lines[0].split(",")
    try:
        lat_idx = header.index("latitude")
        lng_idx = header.index("longitude")
    except ValueError:
        # Try alternate column names
        lat_idx = 0
        lng_idx = 1

    for line in lines[1:]:
        parts = line.split(",")
        try:
            fire_lat = float(parts[lat_idx])
            fire_lng = float(parts[lng_idx])
        except (IndexError, ValueError):
            continue

        # Map to pixel coordinates
        py = int((north - fire_lat) / (north - south) * 64)
        px = int((fire_lng - west) / (east - west) * 64)

        if 0 <= py < 64 and 0 <= px < 64:
            grid[py, px] = 1.0

    return grid

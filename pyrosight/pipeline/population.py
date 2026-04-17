"""GPWv4 population density extractor.

Pre-downloaded raster from SEDAC (Columbia University / NASA).
Source: https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-rev11
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np

_CACHE_DIR = Path(os.environ.get("PYROSIGHT_CACHE", "static_rasters"))
_POP_PATH = _CACHE_DIR / "population_conus.tif"


def extract_population(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Extract population density for a bounding box, resampled to 64×64.

    Args:
        bbox: (west, south, east, north) in degrees

    Returns:
        numpy array (64, 64) float32, people/km²
    """
    west, south, east, north = bbox

    if _POP_PATH.exists():
        import rasterio
        from rasterio.windows import from_bounds

        with rasterio.open(_POP_PATH) as src:
            window = from_bounds(west, south, east, north, src.transform)
            data = src.read(
                1,
                window=window,
                out_shape=(64, 64),
                resampling=rasterio.enums.Resampling.bilinear,
            )
        result = data.astype(np.float32)
        result = np.nan_to_num(result, nan=0.0)
        return np.clip(result, 0, None)

    # Fallback: estimate from location
    return _estimate_population(bbox)


def _estimate_population(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Rough population estimate from location. NOT real data."""
    west, south, east, north = bbox

    # Most wildfire areas are rural — low population
    rng = np.random.RandomState(int(abs(west * 100 + south * 10)))
    base = rng.exponential(10, (64, 64)).astype(np.float32)

    # Urban proximity boost for known metro areas
    center_lat = (north + south) / 2
    center_lng = (east + west) / 2

    # Simple: if near a major city, bump up
    metros = [
        (34.05, -118.24, 500),   # LA
        (37.77, -122.42, 300),   # SF
        (47.61, -122.33, 200),   # Seattle
        (33.45, -112.07, 200),   # Phoenix
        (39.74, -104.99, 200),   # Denver
    ]
    for mlat, mlng, pop in metros:
        dist = np.sqrt((center_lat - mlat)**2 + (center_lng - mlng)**2)
        if dist < 2:
            base += pop * np.exp(-dist)

    print("[POP] Warning: using estimated population. Download GPWv4 for production.")
    return base

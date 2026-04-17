"""VIIRS NDVI (vegetation index) fetcher.

Fetches the most recent NDVI composite from NASA LANCE/LAADS DAAC.
The training data uses VIIRS VNP13A1 (500m, 8-day composite) scaled ×10000.

For real-time use, we can also use MODIS MOD13A2 (1km, 16-day) which is
easier to access via OPeNDAP.
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np

_CACHE_DIR = Path(os.environ.get("PYROSIGHT_CACHE", "static_rasters"))
_NDVI_PATH = _CACHE_DIR / "ndvi_conus.tif"


def fetch_ndvi(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Fetch NDVI data for a bounding box, resampled to 64×64.

    The training data stores NDVI as raw VIIRS values scaled ×10000
    (so typical values are 2000–8000, not -1 to +1).

    Args:
        bbox: (west, south, east, north) in degrees

    Returns:
        numpy array (64, 64) float32, NDVI in training-data scale (×10000)
    """
    west, south, east, north = bbox

    # Try pre-downloaded NDVI composite
    if _NDVI_PATH.exists():
        return _extract_from_geotiff(_NDVI_PATH, bbox)

    # Fall back to estimation from location
    return _estimate_ndvi(bbox)


def _extract_from_geotiff(path: Path, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Extract and resample from a GeoTIFF file."""
    import rasterio
    from rasterio.windows import from_bounds

    west, south, east, north = bbox

    with rasterio.open(path) as src:
        window = from_bounds(west, south, east, north, src.transform)
        data = src.read(
            1,
            window=window,
            out_shape=(64, 64),
            resampling=rasterio.enums.Resampling.bilinear,
        )

    result = data.astype(np.float32)

    # Ensure in ×10000 scale
    if result.max() <= 1.0:
        result *= 10000

    result = np.nan_to_num(result, nan=0.0)
    return result


def _estimate_ndvi(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Estimate NDVI from location characteristics.

    This is a rough proxy — NOT real satellite data.
    For production, download VIIRS VNP13A1 composites.
    """
    west, south, east, north = bbox
    lats = np.linspace(north, south, 64)
    lons = np.linspace(west, east, 64)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    # Rough NDVI model for CONUS:
    # - Pacific NW forests: high (7000-8000)
    # - Southwest desert: low (1000-2000)
    # - Central plains: medium (4000-5000)
    # - Southeast: medium-high (5000-6000)
    base = 5000.0

    # Longitude effect: drier in the west-central
    base += 1500 * np.clip((lon_grid + 100) / 20, -1, 1)

    # Latitude effect: higher in north (forests)
    base += 500 * np.clip((lat_grid - 37) / 10, -1, 1)

    # Pacific NW boost
    mask_pnw = (lon_grid < -118) & (lat_grid > 42)
    base[mask_pnw] += 2000

    # Desert reduction
    mask_desert = (lon_grid > -115) & (lon_grid < -103) & (lat_grid < 37)
    base[mask_desert] -= 2500

    # Add noise
    rng = np.random.RandomState(int(abs(west * 100 + south * 10)))
    base += rng.normal(0, 500, (64, 64))
    base = np.clip(base, 500, 9000)

    print("[NDVI] Warning: using estimated NDVI. Download VIIRS for production.")
    return base.astype(np.float32)


def download_ndvi_composite():
    """Download a recent NDVI composite for CONUS.

    Instructions:
    1. Use Google Earth Engine:
        import ee
        ee.Initialize()
        ndvi = ee.ImageCollection('NOAA/VIIRS/001/VNP13A1') \\
            .filterDate('2025-01-01', '2025-12-31') \\
            .select('NDVI') \\
            .median() \\
            .multiply(10000)  # Scale to match training data
        task = ee.batch.Export.image.toDrive(
            ndvi, description='ndvi_conus',
            region=ee.Geometry.Rectangle([-125, 24.5, -66, 49.5]),
            scale=1000, crs='EPSG:4326'
        )
        task.start()

    2. Save the GeoTIFF to static_rasters/ndvi_conus.tif
    """
    print("See docstring for download instructions.")
    print(f"Save the GeoTIFF to: {_NDVI_PATH}")

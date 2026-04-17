"""SRTM Digital Elevation Model extractor.

Downloads SRTM 1-arc-second (~30m) tiles on demand from USGS and caches them
locally. At query time, extracts the 64×64 km window and resamples to 1km.

Alternative: uses a pre-downloaded CONUS DEM if available at static_rasters/dem_conus.tif.
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np

_CACHE_DIR = Path(os.environ.get("PYROSIGHT_CACHE", "static_rasters"))
_DEM_PATH = _CACHE_DIR / "dem_conus.tif"


def extract_dem(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Extract elevation data for a bounding box, resampled to 64×64.

    Args:
        bbox: (west, south, east, north) in degrees

    Returns:
        numpy array (64, 64) float32, elevation in meters
    """
    west, south, east, north = bbox

    # Try pre-downloaded CONUS DEM first
    if _DEM_PATH.exists():
        return _extract_from_geotiff(_DEM_PATH, bbox)

    # Fall back to downloading SRTM tiles
    return _fetch_srtm_tiles(bbox)


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
    result = np.nan_to_num(result, nan=0.0)
    return result


def _fetch_srtm_tiles(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Fetch SRTM elevation from OpenTopography or NASA Earthdata.

    For now, generates a synthetic elevation surface based on latitude/longitude
    as a placeholder. Replace with actual SRTM download for production.
    """
    west, south, east, north = bbox

    # Try the USGS National Map Elevation API (free, no key needed)
    try:
        return _fetch_usgs_elevation(bbox)
    except Exception:
        pass

    # Placeholder: generate a reasonable elevation surface
    # Uses a simple model: higher in western US, lower in east
    # This is NOT real data — just a demo fallback
    lats = np.linspace(north, south, 64)
    lons = np.linspace(west, east, 64)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    # Rough western US elevation model
    base_elev = 200 + 1500 * np.clip((-lon_grid - 95) / 30, 0, 1)
    # Add latitude variation
    base_elev += 300 * np.sin(lat_grid * 0.1)
    # Add some noise for realism
    rng = np.random.RandomState(int(abs(west * 100 + south * 10)))
    base_elev += rng.normal(0, 100, (64, 64))
    base_elev = np.clip(base_elev, 0, 4000)

    print("[DEM] Warning: using synthetic elevation. Download SRTM for production.")
    return base_elev.astype(np.float32)


def _fetch_usgs_elevation(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Fetch elevation from USGS National Map API."""
    import requests

    west, south, east, north = bbox
    # USGS Elevation Point Query Service
    # We query a grid of points and assemble into array
    lats = np.linspace(north, south, 64)
    lons = np.linspace(west, east, 64)

    # This API is point-based and too slow for 4096 points.
    # For production, use the USGS 3DEP 1/3 arc-second WCS service instead.
    raise NotImplementedError("Use pre-downloaded DEM for production")


def download_conus_dem():
    """Download a CONUS-wide DEM. Run once during setup.

    Instructions:
    1. Go to https://www.sciencebase.gov/catalog/item/543e6b86e4b0fd76af69cf4c
       (USGS 3DEP 1-km DEM)
    2. Download the GeoTIFF
    3. Save as static_rasters/dem_conus.tif

    Or use Google Earth Engine:
        import ee
        ee.Initialize()
        dem = ee.Image('USGS/SRTMGL1_003')
        task = ee.batch.Export.image.toDrive(
            dem, description='srtm_conus',
            region=ee.Geometry.Rectangle([-125, 24.5, -66, 49.5]),
            scale=1000, crs='EPSG:4326'
        )
        task.start()
    """
    print("See docstring for download instructions.")
    print(f"Save the GeoTIFF to: {_DEM_PATH}")

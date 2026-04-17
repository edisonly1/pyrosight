"""GRIDMET weather data fetcher.

Fetches daily weather from University of Idaho GRIDMET by downloading
yearly NetCDF files and caching them locally. Subsequent requests for
the same year use the cached file.

Source: https://www.northwestknowledge.net/metdata/data/
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import requests
import xarray as xr

_GRIDMET_VARS = {
    "th":   "wind_from_direction",
    "vs":   "wind_speed",
    "tmmn": "air_temperature",
    "tmmx": "air_temperature",
    "sph":  "specific_humidity",
    "pr":   "precipitation_amount",
    "erc":  "energy_release_component-g",
}

_BASE_URL = "https://www.northwestknowledge.net/metdata/data"
_CACHE_DIR = Path(os.environ.get("PYROSIGHT_CACHE", "static_rasters")) / "gridmet"


def _ensure_cached(variable: str, year: int) -> Path:
    """Download a GRIDMET NetCDF file if not already cached."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _CACHE_DIR / f"{variable}_{year}.nc"

    if local_path.exists():
        return local_path

    url = f"{_BASE_URL}/{variable}_{year}.nc"
    print(f"[GRIDMET] Downloading {url} ...")

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192 * 16):
            f.write(chunk)

    size_mb = local_path.stat().st_size / 1e6
    print(f"[GRIDMET] Saved {local_path.name} ({size_mb:.0f} MB)")
    return local_path


def fetch_gridmet(
    variable: str,
    date: str,
    bbox: Tuple[float, float, float, float],
) -> np.ndarray:
    """Fetch a GRIDMET variable for a date and bounding box, resampled to 64x64.

    Args:
        variable: One of "th", "vs", "tmmn", "tmmx", "sph", "pr", "erc"
        date: Date string "YYYY-MM-DD"
        bbox: (west, south, east, north) in degrees

    Returns:
        numpy array (64, 64) float32 in raw physical units
    """
    west, south, east, north = bbox
    year = int(date[:4])

    # Download if needed
    nc_path = _ensure_cached(variable, year)

    ds = xr.open_dataset(nc_path, engine="netcdf4")

    # Find the data variable
    actual_var = None
    expected = _GRIDMET_VARS.get(variable, variable)
    for v in ds.data_vars:
        if v == expected or variable in v.lower():
            actual_var = v
            break
    if actual_var is None:
        actual_var = list(ds.data_vars)[0]

    # Select date
    time_dim = None
    for dim in ds.dims:
        if "day" in dim.lower() or "time" in dim.lower():
            time_dim = dim
            break

    data = ds[actual_var]
    if time_dim:
        data = data.sel({time_dim: date}, method="nearest")

    # Find lat/lon dim names
    lat_dim = next((d for d in data.dims if "lat" in d.lower()), None)
    lon_dim = next((d for d in data.dims if "lon" in d.lower()), None)

    if lat_dim is None or lon_dim is None:
        ds.close()
        raise ValueError(f"Cannot find lat/lon dims in {nc_path}")

    # GRIDMET lat is typically descending
    lat_ascending = float(data[lat_dim][0]) < float(data[lat_dim][-1])
    if lat_ascending:
        data = data.sel(**{lat_dim: slice(south, north), lon_dim: slice(west, east)})
    else:
        data = data.sel(**{lat_dim: slice(north, south), lon_dim: slice(west, east)})

    # Resample to 64x64 using cubic interpolation (smoother than linear)
    target_lats = np.linspace(north, south, 64)
    target_lons = np.linspace(west, east, 64)

    resampled = data.interp(
        **{lat_dim: target_lats, lon_dim: target_lons},
        method="cubic",
    )

    result = resampled.values.astype(np.float32)
    ds.close()

    result = np.nan_to_num(result, nan=0.0)

    # Apply Gaussian smoothing to eliminate 4km grid artifacts
    # GRIDMET is ~4km, our grid is 1km, so σ=2 smooths the 4-pixel grid pattern
    result = _smooth(result, sigma=2.0)

    return result


def _smooth(data: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply Gaussian smoothing to remove grid artifacts."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(data, sigma=sigma, mode='nearest')


def fetch_gridmet_drought(date: str, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Fetch PDSI drought index."""
    return fetch_gridmet("pdsi", date, bbox)

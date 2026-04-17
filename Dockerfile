FROM python:3.12-slim

WORKDIR /app

# System deps for rasterio/GDAL and scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY server.py .
COPY pyrosight/ pyrosight/
COPY retrain.py .
COPY retrain_v2.py .

# Model checkpoint
COPY checkpoints/best.pt checkpoints/best.pt

# Static rasters (elevation, NDVI, population — NOT gridmet cache)
COPY static_rasters/dem_conus.tif static_rasters/dem_conus.tif
COPY static_rasters/ndvi_conus.tif static_rasters/ndvi_conus.tif
COPY static_rasters/population_conus.tif static_rasters/population_conus.tif

# GRIDMET cache directory (will be populated on first request)
RUN mkdir -p static_rasters/gridmet

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYROSIGHT_CACHE=/app/static_rasters

EXPOSE 8000

COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]

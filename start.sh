#!/bin/bash
set -e

# Download test data if not already present
mkdir -p data
if [ ! -f data/next_day_wildfire_spread_test_00.tfrecord ]; then
    echo "[startup] Downloading test data..."
    # Replace these URLs with your actual download links
    # Option A: GitHub Release assets
    curl -L -o data/next_day_wildfire_spread_test_00.tfrecord "https://github.com/edisonly1/pyrosight/releases/download/v1.0/next_day_wildfire_spread_eval_00.tfrecord"
    curl -L -o data/next_day_wildfire_spread_test_01.tfrecord "https://github.com/edisonly1/pyrosight/releases/download/v1.0/next_day_wildfire_spread_eval_01.tfrecord"

    # Option B: Google Drive (replace FILE_ID with your sharing IDs)
    # curl -L -o data/next_day_wildfire_spread_test_00.tfrecord "https://drive.google.com/uc?export=download&id=FILE_ID_00"
    # curl -L -o data/next_day_wildfire_spread_test_01.tfrecord "https://drive.google.com/uc?export=download&id=FILE_ID_01"

    echo "[startup] Test data downloaded."
else
    echo "[startup] Test data already present."
fi

exec uvicorn server:app --host 0.0.0.0 --port 8000

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$PROJECT_ROOT/data/raw"
ZIP_PATH="$RAW_DIR/house-rooms-streets-image-dataset.zip"
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/mikhailma/house-rooms-streets-image-dataset?datasetVersionNumber=1"

mkdir -p "$RAW_DIR"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Downloading Kaggle dataset to $ZIP_PATH"
  curl -L --fail --show-error --progress-bar -o "$ZIP_PATH" "$DATASET_URL"
else
  echo "Dataset ZIP already exists: $ZIP_PATH"
fi

echo "Extracting dataset to $RAW_DIR"
unzip -q -o "$ZIP_PATH" -d "$RAW_DIR"

HOUSE_DIR="$RAW_DIR/kaggle_room_street_data/house_data"
if [[ ! -d "$HOUSE_DIR" ]]; then
  echo "Expected folder not found: $HOUSE_DIR" >&2
  exit 1
fi

COUNT="$(find "$HOUSE_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l | tr -d ' ')"
echo "house_data JPG count: $COUNT"

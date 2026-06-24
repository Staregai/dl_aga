#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p cluster/logs

submit() {
  local job_name="$1"
  shift
  local export_arg="ALL"
  local item
  for item in "$@"; do
    export_arg="${export_arg},${item}"
  done
  sbatch --parsable --job-name="$job_name" --export="$export_arg" cluster/train_one.sbatch
}

submit room-cnn-224 \
  RUN_NAME=cnn_224 \
  TRAIN_SCRIPT=train_cnn.py \
  IMG_SIZE=224 \
  BATCH_SIZE=32 \
  EPOCHS=500 \
  PATIENCE=45 \
  NUM_WORKERS=4

submit room-cnnaug-224 \
  RUN_NAME=cnn_aug_224_strong \
  TRAIN_SCRIPT=train_cnn_aug.py \
  IMG_SIZE=224 \
  BATCH_SIZE=32 \
  EPOCHS=500 \
  PATIENCE=60 \
  NUM_WORKERS=4 \
  EXTRA_ARGS=--augment-strength=strong

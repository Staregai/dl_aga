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

submit room-gs-cnn-m \
  RUN_NAME=grid_cnn_medium_lr5e4_wd5e4_d40 \
  TRAIN_SCRIPT=train_cnn.py \
  IMG_SIZE=224 \
  BATCH_SIZE=32 \
  EPOCHS=500 \
  PATIENCE=40 \
  NUM_WORKERS=4 \
  EXTRA_ARGS="--arch=medium --lr=5e-4 --weight-decay=5e-4 --dropout=0.40 --label-smoothing=0.05"

submit room-gs-aug-m \
  RUN_NAME=grid_aug_medium_lr5e4_wd5e4_d35_ls08 \
  TRAIN_SCRIPT=train_cnn_aug.py \
  IMG_SIZE=224 \
  BATCH_SIZE=32 \
  EPOCHS=500 \
  PATIENCE=60 \
  NUM_WORKERS=4 \
  EXTRA_ARGS="--arch=medium --augment-strength=strong --lr=5e-4 --weight-decay=5e-4 --dropout=0.35 --label-smoothing=0.08"

submit room-gs-aug-l1 \
  RUN_NAME=grid_aug_large_lr3e4_wd1e3_d45_ls10 \
  TRAIN_SCRIPT=train_cnn_aug.py \
  IMG_SIZE=224 \
  BATCH_SIZE=32 \
  EPOCHS=500 \
  PATIENCE=65 \
  NUM_WORKERS=4 \
  EXTRA_ARGS="--arch=large --augment-strength=strong --lr=3e-4 --weight-decay=1e-3 --dropout=0.45 --label-smoothing=0.10"

submit room-gs-aug-l2 \
  RUN_NAME=grid_aug_large_lr2e4_wd5e4_d40_ls08 \
  TRAIN_SCRIPT=train_cnn_aug.py \
  IMG_SIZE=224 \
  BATCH_SIZE=32 \
  EPOCHS=500 \
  PATIENCE=65 \
  NUM_WORKERS=4 \
  EXTRA_ARGS="--arch=large --augment-strength=strong --lr=2e-4 --weight-decay=5e-4 --dropout=0.40 --label-smoothing=0.08"

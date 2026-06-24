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

submit room-gs-mlp-96 \
  RUN_NAME=grid_mlp_96_lr5e4_wd1e4_d50 \
  TRAIN_SCRIPT=train_mlp.py \
  IMG_SIZE=96 \
  BATCH_SIZE=64 \
  EPOCHS=200 \
  PATIENCE=25 \
  NUM_WORKERS=4 \
  EXTRA_ARGS="--lr=5e-4 --weight-decay=1e-4 --dropout=0.50 --label-smoothing=0.03"

submit room-gs-mlp-128 \
  RUN_NAME=grid_mlp_128_lr2e4_wd5e4_d60 \
  TRAIN_SCRIPT=train_mlp.py \
  IMG_SIZE=128 \
  BATCH_SIZE=64 \
  EPOCHS=200 \
  PATIENCE=25 \
  NUM_WORKERS=4 \
  EXTRA_ARGS="--lr=2e-4 --weight-decay=5e-4 --dropout=0.60 --label-smoothing=0.05"

submit room-gs-cnn-l \
  RUN_NAME=grid_cnn_large_lr3e4_wd1e3_d50 \
  TRAIN_SCRIPT=train_cnn.py \
  IMG_SIZE=224 \
  BATCH_SIZE=32 \
  EPOCHS=500 \
  PATIENCE=40 \
  NUM_WORKERS=4 \
  EXTRA_ARGS="--arch=large --lr=3e-4 --weight-decay=1e-3 --dropout=0.50 --label-smoothing=0.06"

submit room-gs-aug-basic \
  RUN_NAME=grid_aug_medium_basic_lr5e4_wd5e4_d35_ls08 \
  TRAIN_SCRIPT=train_cnn_aug.py \
  IMG_SIZE=224 \
  BATCH_SIZE=32 \
  EPOCHS=500 \
  PATIENCE=60 \
  NUM_WORKERS=4 \
  EXTRA_ARGS="--arch=medium --augment-strength=basic --lr=5e-4 --weight-decay=5e-4 --dropout=0.35 --label-smoothing=0.08"

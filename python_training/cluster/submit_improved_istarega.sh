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

submit room-imp-cnn-m256 \
  RUN_NAME=imp_cnn_medium_256_cos_ema \
  TRAIN_SCRIPT=train_cnn.py \
  IMG_SIZE=256 \
  BATCH_SIZE=32 \
  EPOCHS=500 \
  PATIENCE=55 \
  NUM_WORKERS=4 \
  TTA_PASSES=1 \
  EXTRA_ARGS="--arch=medium --lr=4e-4 --weight-decay=8e-4 --dropout=0.45 --label-smoothing=0.06 --scheduler=cosine --warmup-epochs=12 --min-lr=1e-6 --ema-decay=0.999"

submit room-imp-cnn-l288 \
  RUN_NAME=imp_cnn_large_288_cos_ema \
  TRAIN_SCRIPT=train_cnn.py \
  IMG_SIZE=288 \
  BATCH_SIZE=16 \
  EPOCHS=500 \
  PATIENCE=55 \
  NUM_WORKERS=4 \
  TTA_PASSES=1 \
  EXTRA_ARGS="--arch=large --lr=2.5e-4 --weight-decay=1e-3 --dropout=0.50 --label-smoothing=0.06 --scheduler=cosine --warmup-epochs=12 --min-lr=1e-6 --ema-decay=0.999"

submit room-imp-aug-m256 \
  RUN_NAME=imp_aug_medium_256_cos_ema_mix \
  TRAIN_SCRIPT=train_cnn_aug.py \
  IMG_SIZE=256 \
  BATCH_SIZE=32 \
  EPOCHS=500 \
  PATIENCE=75 \
  NUM_WORKERS=4 \
  TTA_PASSES=5 \
  EXTRA_ARGS="--arch=medium --augment-strength=strong --lr=4e-4 --weight-decay=8e-4 --dropout=0.40 --label-smoothing=0.09 --scheduler=cosine --warmup-epochs=12 --min-lr=1e-6 --ema-decay=0.999 --mixup-alpha=0.2 --cutmix-alpha=1.0 --mix-prob=0.40 --cutmix-prob=0.7"

submit room-imp-aug-l256-s123 \
  RUN_NAME=imp_aug_large_256_seed123_cos_ema_mix \
  TRAIN_SCRIPT=train_cnn_aug.py \
  IMG_SIZE=256 \
  BATCH_SIZE=24 \
  EPOCHS=500 \
  PATIENCE=75 \
  NUM_WORKERS=4 \
  TTA_PASSES=5 \
  EXTRA_ARGS="--arch=large --augment-strength=strong --lr=2.5e-4 --weight-decay=1e-3 --dropout=0.45 --label-smoothing=0.10 --scheduler=cosine --warmup-epochs=12 --min-lr=1e-6 --ema-decay=0.999 --mixup-alpha=0.2 --cutmix-alpha=1.0 --mix-prob=0.40 --cutmix-prob=0.7 --seed=123"

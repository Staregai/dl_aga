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
  local job_id
  job_id=$(sbatch --parsable --job-name="$job_name" --export="$export_arg" cluster/train_one.sbatch)
  echo "submitted $job_name as $job_id" >&2
  printf "%s" "$job_id"
}

submit_ensemble() {
  local job_name="$1"
  local dependency="$2"
  shift 2
  local export_arg="ALL"
  local item
  for item in "$@"; do
    export_arg="${export_arg},${item}"
  done
  local job_id
  job_id=$(
    sbatch \
      --parsable \
      --job-name="$job_name" \
      --dependency="afterok:$dependency" \
      --export="$export_arg" \
      cluster/evaluate_ensemble.sbatch
  )
  echo "submitted $job_name as $job_id afterok:$dependency" >&2
  printf "%s" "$job_id"
}

JID_CNN_L256=$(
  submit room-imp-cnn-l256 \
    RUN_NAME=imp_cnn_large_256_cos_ema \
    TRAIN_SCRIPT=train_cnn.py \
    IMG_SIZE=256 \
    BATCH_SIZE=24 \
    EPOCHS=500 \
    PATIENCE=55 \
    NUM_WORKERS=4 \
    TTA_PASSES=1 \
    EXTRA_ARGS="--arch=large --lr=3e-4 --weight-decay=1e-3 --dropout=0.50 --label-smoothing=0.06 --scheduler=cosine --warmup-epochs=12 --min-lr=1e-6 --ema-decay=0.999"
)

JID_AUG_L256=$(
  submit room-imp-aug-l256 \
    RUN_NAME=imp_aug_large_256_cos_ema_mix \
    TRAIN_SCRIPT=train_cnn_aug.py \
    IMG_SIZE=256 \
    BATCH_SIZE=24 \
    EPOCHS=500 \
    PATIENCE=75 \
    NUM_WORKERS=4 \
    TTA_PASSES=5 \
    EXTRA_ARGS="--arch=large --augment-strength=strong --lr=3e-4 --weight-decay=1e-3 --dropout=0.45 --label-smoothing=0.10 --scheduler=cosine --warmup-epochs=12 --min-lr=1e-6 --ema-decay=0.999 --mixup-alpha=0.2 --cutmix-alpha=1.0 --mix-prob=0.45 --cutmix-prob=0.7"
)

JID_AUG_L288=$(
  submit room-imp-aug-l288 \
    RUN_NAME=imp_aug_large_288_cos_ema_mix \
    TRAIN_SCRIPT=train_cnn_aug.py \
    IMG_SIZE=288 \
    BATCH_SIZE=16 \
    EPOCHS=500 \
    PATIENCE=75 \
    NUM_WORKERS=4 \
    TTA_PASSES=5 \
    EXTRA_ARGS="--arch=large --augment-strength=strong --lr=2.5e-4 --weight-decay=1e-3 --dropout=0.45 --label-smoothing=0.10 --scheduler=cosine --warmup-epochs=12 --min-lr=1e-6 --ema-decay=0.999 --mixup-alpha=0.2 --cutmix-alpha=1.0 --mix-prob=0.40 --cutmix-prob=0.7"
)

submit_ensemble room-imp-ens-aug "${JID_AUG_L256}:${JID_AUG_L288}" \
  RUN_NAME=imp_ensemble_aug_large_256_288_tta5 \
  CHECKPOINTS="outputs/checkpoints/imp_aug_large_256_cos_ema_mix/cnn_aug_best.pt,outputs/checkpoints/imp_aug_large_288_cos_ema_mix/cnn_aug_best.pt" \
  BATCH_SIZE=16 \
  NUM_WORKERS=4 \
  TTA_PASSES=5 \
  HARD_EXAMPLES=120 >/dev/null

echo "plain_cnn_job=$JID_CNN_L256"
echo "aug_256_job=$JID_AUG_L256"
echo "aug_288_job=$JID_AUG_L288"

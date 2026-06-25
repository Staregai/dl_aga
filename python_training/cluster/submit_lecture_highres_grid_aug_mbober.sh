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

for variant in wide deep; do
  for lr_name in 1e3 5e4; do
    case "$lr_name" in
      1e3) lr="1e-3" ;;
      5e4) lr="5e-4" ;;
      *) echo "Unknown lr_name=$lr_name" >&2; exit 1 ;;
    esac

    for dropout_name in d30 d45; do
      case "$dropout_name" in
        d30) dropout="0.30" ;;
        d45) dropout="0.45" ;;
        *) echo "Unknown dropout_name=$dropout_name" >&2; exit 1 ;;
      esac

      run_name="lecture_hr_aug_${variant}_224_lr${lr_name}_${dropout_name}_bn1"
      submit "lh-aug-${variant:0:1}-${lr_name}-${dropout_name}" \
        RUN_NAME="$run_name" \
        TRAIN_SCRIPT=train_lecture_cnn_aug.py \
        IMG_SIZE=224 \
        BATCH_SIZE=32 \
        EPOCHS=500 \
        PATIENCE=65 \
        NUM_WORKERS=4 \
        TTA_PASSES=1 \
        HARD_EXAMPLES=80 \
        EXTRA_ARGS="--variant=$variant --batch-norm --lr=$lr --dropout=$dropout"
    done
  done
done

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

for variant in highres_wide highres_deep; do
  case "$variant" in
    highres_wide) variant_label="wide"; variant_short="w" ;;
    highres_deep) variant_label="deep"; variant_short="d" ;;
    *) echo "Unknown variant=$variant" >&2; exit 1 ;;
  esac

  for lr_name in 3e4 1e4; do
    case "$lr_name" in
      3e4) lr="3e-4" ;;
      1e4) lr="1e-4" ;;
      *) echo "Unknown lr_name=$lr_name" >&2; exit 1 ;;
    esac

    for dropout_name in d30 d45; do
      case "$dropout_name" in
        d30) dropout="0.30" ;;
        d45) dropout="0.45" ;;
        *) echo "Unknown dropout_name=$dropout_name" >&2; exit 1 ;;
      esac

      run_name="lecture_hr2_cnn_${variant_label}_224_lr${lr_name}_${dropout_name}_bn1"
      submit "lh2-cnn-${variant_short}-${lr_name}-${dropout_name}" \
        RUN_NAME="$run_name" \
        TRAIN_SCRIPT=train_lecture_cnn.py \
        IMG_SIZE=224 \
        BATCH_SIZE=24 \
        EPOCHS=500 \
        PATIENCE=55 \
        NUM_WORKERS=4 \
        TTA_PASSES=1 \
        HARD_EXAMPLES=80 \
        EXTRA_ARGS="--variant=$variant --batch-norm --lr=$lr --dropout=$dropout"
    done
  done
done

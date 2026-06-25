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

for topology in 1 2 3; do
  for lr_name in 1e3 5e4; do
    case "$lr_name" in
      1e3) lr="1e-3" ;;
      5e4) lr="5e-4" ;;
      *) echo "Unknown lr_name=$lr_name" >&2; exit 1 ;;
    esac

    for dropout_name in d30 d50; do
      case "$dropout_name" in
        d30) dropout="0.30" ;;
        d50) dropout="0.50" ;;
        *) echo "Unknown dropout_name=$dropout_name" >&2; exit 1 ;;
      esac

      for bn in 0 1; do
        bn_arg=""
        if [[ "$bn" == "1" ]]; then
          bn_arg=" --batch-norm"
        fi

        run_name="lecture_aug_t${topology}_64_lr${lr_name}_${dropout_name}_bn${bn}"
        submit "lect-aug-t${topology}-${lr_name}-${dropout_name}-bn${bn}" \
          RUN_NAME="$run_name" \
          TRAIN_SCRIPT=train_lecture_cnn_aug.py \
          IMG_SIZE=64 \
          BATCH_SIZE=32 \
          EPOCHS=500 \
          PATIENCE=55 \
          NUM_WORKERS=4 \
          TTA_PASSES=1 \
          HARD_EXAMPLES=80 \
          EXTRA_ARGS="--topology=$topology --lr=$lr --dropout=$dropout$bn_arg"
      done
    done
  done
done

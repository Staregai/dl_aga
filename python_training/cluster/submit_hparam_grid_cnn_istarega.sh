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

for arch in medium large; do
  for lr_name in 3e4 5e4; do
    case "$lr_name" in
      3e4) lr="3e-4" ;;
      5e4) lr="5e-4" ;;
      *) echo "Unknown lr_name=$lr_name" >&2; exit 1 ;;
    esac

    for wd_name in 5e4 1e3; do
      case "$wd_name" in
        5e4) wd="5e-4" ;;
        1e3) wd="1e-3" ;;
        *) echo "Unknown wd_name=$wd_name" >&2; exit 1 ;;
      esac

      for dropout_name in d40 d50; do
        case "$dropout_name" in
          d40) dropout="0.40" ;;
          d50) dropout="0.50" ;;
          *) echo "Unknown dropout_name=$dropout_name" >&2; exit 1 ;;
        esac

        run_name="hgrid_cnn_${arch}_256_lr${lr_name}_wd${wd_name}_${dropout_name}_cos_ema"
        submit "hg-cnn-${arch:0:1}-${lr_name}-${wd_name}-${dropout_name}" \
          RUN_NAME="$run_name" \
          TRAIN_SCRIPT=train_cnn.py \
          IMG_SIZE=256 \
          BATCH_SIZE=24 \
          EPOCHS=500 \
          PATIENCE=55 \
          NUM_WORKERS=4 \
          TTA_PASSES=1 \
          EXTRA_ARGS="--arch=$arch --lr=$lr --weight-decay=$wd --dropout=$dropout --label-smoothing=0.06 --scheduler=cosine --warmup-epochs=12 --min-lr=1e-6 --ema-decay=0.999"
      done
    done
  done
done

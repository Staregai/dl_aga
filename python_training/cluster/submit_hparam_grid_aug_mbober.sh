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

for lr_name in 2e4 3e4; do
  case "$lr_name" in
    2e4) lr="2e-4" ;;
    3e4) lr="3e-4" ;;
    *) echo "Unknown lr_name=$lr_name" >&2; exit 1 ;;
  esac

  for wd_name in 5e4 1e3; do
    case "$wd_name" in
      5e4) wd="5e-4" ;;
      1e3) wd="1e-3" ;;
      *) echo "Unknown wd_name=$wd_name" >&2; exit 1 ;;
    esac

    for dropout_name in d40 d45; do
      case "$dropout_name" in
        d40) dropout="0.40" ;;
        d45) dropout="0.45" ;;
        *) echo "Unknown dropout_name=$dropout_name" >&2; exit 1 ;;
      esac

      for mix_name in m30 m45; do
        case "$mix_name" in
          m30) mix_prob="0.30" ;;
          m45) mix_prob="0.45" ;;
          *) echo "Unknown mix_name=$mix_name" >&2; exit 1 ;;
        esac

        run_name="hgrid_aug_large_256_lr${lr_name}_wd${wd_name}_${dropout_name}_${mix_name}_cos_ema"
        submit "hg-aug-l-${lr_name}-${wd_name}-${dropout_name}-${mix_name}" \
          RUN_NAME="$run_name" \
          TRAIN_SCRIPT=train_cnn_aug.py \
          IMG_SIZE=256 \
          BATCH_SIZE=24 \
          EPOCHS=500 \
          PATIENCE=75 \
          NUM_WORKERS=4 \
          TTA_PASSES=5 \
          HARD_EXAMPLES=120 \
          EXTRA_ARGS="--arch=large --augment-strength=strong --lr=$lr --weight-decay=$wd --dropout=$dropout --label-smoothing=0.10 --scheduler=cosine --warmup-epochs=12 --min-lr=1e-6 --ema-decay=0.999 --mixup-alpha=0.2 --cutmix-alpha=1.0 --mix-prob=$mix_prob --cutmix-prob=0.7"
      done
    done
  done
done

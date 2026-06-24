#!/bin/bash
#SBATCH --account=stud-2526-l-03
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl_aga/python_training}"
cd "$PROJECT_DIR"

source .venv/bin/activate

mkdir -p cluster/logs outputs/checkpoints outputs/reports splits

: "${RUN_NAME:?RUN_NAME is required}"
: "${TRAIN_SCRIPT:?TRAIN_SCRIPT is required}"

DATA_DIR="${DATA_DIR:-$PROJECT_DIR/../data/raw/kaggle_room_street_data/house_data}"
SPLIT_CSV="${SPLIT_CSV:-$PROJECT_DIR/splits/room_split_seed42_${RUN_NAME}.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/checkpoints/$RUN_NAME}"
REPORT_DIR="${REPORT_DIR:-$PROJECT_DIR/outputs/reports/$RUN_NAME}"
IMG_SIZE="${IMG_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-500}"
PATIENCE="${PATIENCE:-60}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUTPUT_DIR" "$REPORT_DIR"

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "user=$(whoami)"
echo "project_dir=$PROJECT_DIR"
echo "run_name=$RUN_NAME"
echo "train_script=$TRAIN_SCRIPT"
echo "data_dir=$DATA_DIR"
echo "split_csv=$SPLIT_CSV"
echo "output_dir=$OUTPUT_DIR"
echo "img_size=$IMG_SIZE batch_size=$BATCH_SIZE epochs=$EPOCHS patience=$PATIENCE num_workers=$NUM_WORKERS"
nvidia-smi || true

python - <<'PY'
import torch
import torchvision

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_runtime", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

# shellcheck disable=SC2086
python "$TRAIN_SCRIPT" \
  --data-dir "$DATA_DIR" \
  --split-csv "$SPLIT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --img-size "$IMG_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --num-workers "$NUM_WORKERS" \
  --device cuda \
  $EXTRA_ARGS

python evaluate_checkpoints.py \
  --data-dir "$DATA_DIR" \
  --split-csv "$SPLIT_CSV" \
  --checkpoint-dir "$OUTPUT_DIR" \
  --output-dir "$REPORT_DIR" \
  --split test \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device cuda

echo "finished_at=$(date -Is)"

#!/bin/bash
#SBATCH --account=stud-2526-l-03
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl_aga/python_training}"
cd "$PROJECT_DIR"

source .venv/bin/activate

mkdir -p cluster/logs outputs/reports splits

: "${RUN_NAME:?RUN_NAME is required}"
: "${CHECKPOINTS:?CHECKPOINTS is required as comma-separated paths}"

DATA_DIR="${DATA_DIR:-$PROJECT_DIR/../data/raw/kaggle_room_street_data/house_data}"
SPLIT_CSV="${SPLIT_CSV:-$PROJECT_DIR/splits/room_split_seed42_${RUN_NAME}.csv}"
REPORT_DIR="${REPORT_DIR:-$PROJECT_DIR/outputs/reports/$RUN_NAME}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TTA_PASSES="${TTA_PASSES:-5}"
HARD_EXAMPLES="${HARD_EXAMPLES:-120}"

mkdir -p "$REPORT_DIR"

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "user=$(whoami)"
echo "project_dir=$PROJECT_DIR"
echo "run_name=$RUN_NAME"
echo "checkpoints=$CHECKPOINTS"
echo "data_dir=$DATA_DIR"
echo "split_csv=$SPLIT_CSV"
echo "report_dir=$REPORT_DIR"
echo "batch_size=$BATCH_SIZE num_workers=$NUM_WORKERS tta_passes=$TTA_PASSES"
nvidia-smi || true

IFS=',' read -r -a checkpoint_array <<< "$CHECKPOINTS"
checkpoint_args=()
for checkpoint in "${checkpoint_array[@]}"; do
  checkpoint_args+=(--checkpoint "$checkpoint")
done

python evaluate_ensemble.py \
  "${checkpoint_args[@]}" \
  --data-dir "$DATA_DIR" \
  --split-csv "$SPLIT_CSV" \
  --output-dir "$REPORT_DIR" \
  --name "$RUN_NAME" \
  --split test \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device cuda \
  --tta-passes "$TTA_PASSES" \
  --hard-examples "$HARD_EXAMPLES"

echo "finished_at=$(date -Is)"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p cluster/logs

sbatch --parsable \
  --job-name=room-mlp-128 \
  --export=ALL,RUN_NAME=mlp_128,TRAIN_SCRIPT=train_mlp.py,IMG_SIZE=128,BATCH_SIZE=64,EPOCHS=200,PATIENCE=25,NUM_WORKERS=4 \
  cluster/train_one.sbatch

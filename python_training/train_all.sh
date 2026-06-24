#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python train_mlp.py
python train_cnn.py
python train_cnn_aug.py
python evaluate_checkpoints.py --split test

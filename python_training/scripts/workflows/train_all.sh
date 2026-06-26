#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python scripts/train/train_mlp.py
python scripts/train/train_cnn.py
python scripts/train/train_cnn_aug.py
python scripts/evaluate/evaluate_checkpoints.py --split test

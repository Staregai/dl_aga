# Scripts

```text
train/      trening pojedynczych modeli
evaluate/   ewaluacja checkpointów i ensemble
workflows/  skrypty łączące kilka kroków
```

Uruchamiaj z katalogu `python_training/`, np.:

```bash
python scripts/train/train_cnn_aug.py --seed 42
python scripts/evaluate/evaluate_checkpoints.py --split test
bash scripts/workflows/train_all.sh
```

# Python/PyTorch Training

Ten katalog zawiera uporządkowaną wersję treningu i ewaluacji modeli w PyTorch.

## Struktura

```text
python_training/
├── notebooks/
│   └── evaluate_trained_models.ipynb
├── scripts/
│   ├── train/
│   │   ├── train_mlp.py
│   │   ├── train_cnn.py
│   │   ├── train_cnn_aug.py
│   │   ├── train_simple_cnn.py
│   │   └── train_simple_cnn_aug.py
│   ├── evaluate/
│   │   ├── evaluate_checkpoints.py
│   │   └── evaluate_ensemble.py
│   └── workflows/
│       └── train_all.sh
├── src/room_classifier/
├── splits/
└── requirements.txt
```

Skrypty można odpalać z katalogu `python_training/`; same wykrywają root projektu treningowego i ustawiają importy z `src/`.

## Setup

```bash
cd python_training
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dane są domyślnie czytane z:

```text
../data/raw/kaggle_room_street_data/house_data
```

Domyślnie usuwana jest klasa `bath`, żeby zachować porównywalność z finalnym eksperymentem.

## Modele

- `scripts/train/train_mlp.py` - baseline MLP na spłaszczonych pikselach.
- `scripts/train/train_cnn.py` - własny `RoomResNet` trenowany od zera, bez augmentacji danych.
- `scripts/train/train_cnn_aug.py` - ta sama rodzina `RoomResNet`, ale z augmentacją, EMA, schedulerem cosine oraz opcjonalnym MixUp/CutMix.
- `scripts/train/train_simple_cnn.py` - proste CNN inspirowane pierwotnym notebookiem, bez augmentacji.
- `scripts/train/train_simple_cnn_aug.py` - ten sam typ prostego CNN, ale z lekką augmentacją danych.

Wszystkie skrypty treningowe przyjmują `--seed`. Historia treningu zapisuje się obok checkpointu jako `*.history.csv`; TensorBoard można włączyć przez `--tensorboard-dir`.

## Trening

```bash
python scripts/train/train_mlp.py
python scripts/train/train_cnn.py
python scripts/train/train_cnn_aug.py
```

Przykład szybkiego smoke testu:

```bash
python scripts/train/train_mlp.py --epochs 1 --max-samples 160 --batch-size 16 --output-dir outputs/smoke
python scripts/train/train_cnn.py --epochs 1 --max-samples 160 --batch-size 16 --img-size 96 --output-dir outputs/smoke
python scripts/train/train_cnn_aug.py --epochs 1 --max-samples 160 --batch-size 16 --img-size 96 --output-dir outputs/smoke
```

Zbiorcze uruchomienie podstawowego treningu i ewaluacji:

```bash
bash scripts/workflows/train_all.sh
```

## Ewaluacja

```bash
python scripts/evaluate/evaluate_checkpoints.py --split test
```

Ensemble:

```bash
python scripts/evaluate/evaluate_ensemble.py \
  --checkpoint outputs/checkpoints/model_a/cnn_aug_best.pt \
  --checkpoint outputs/checkpoints/model_b/cnn_aug_best.pt \
  --checkpoint outputs/checkpoints/model_c/cnn_aug_best.pt \
  --tta-passes 5 \
  --name ensemble_top3
```

Notebook:

```text
notebooks/evaluate_trained_models.ipynb
```

Notebook ładuje checkpointy i zapisane metryki, agreguje wyniki po seedach oraz generuje finalne wykresy do katalogu `../figures/`.

## Outputy

Domyślne lokalizacje:

```text
outputs/checkpoints/   # checkpointy, historie treningu
outputs/reports/       # metryki, macierze pomyłek, CSV
outputs/tensorboard/   # logi TensorBoard, jeśli włączone
```

`outputs/` jest ignorowany przez git. Do raportu potrzebne są tylko finalne grafiki w katalogu rootowym `figures/`.

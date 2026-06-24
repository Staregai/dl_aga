# Python/PyTorch training

Ten folder jest czystą wersją projektu w Pythonie. Dane są czytane z:

```text
../data/raw/kaggle_room_street_data/house_data
```

Domyślnie usuwana jest klasa `bath`, żeby zachować porównywalność z notebookiem Julii.

## Modele

- `train_mlp.py` - mocniejszy baseline MLP na pikselach, `128x128`.
- `train_cnn.py` - własny `room_resnet_small/medium/large` trenowany od zera: bloki rezydualne, BatchNorm, Squeeze-and-Excitation, global average pooling. Ten wariant nie używa żadnych augmentacji danych.
- `train_cnn_aug.py` - ta sama własna architektura `room_resnet_small/medium/large`, ale z augmentacją danych w treningu.

Jeżeli chcesz porównać wpływ augmentacji, najważniejszy jest trzeci wariant. Domyślnie używa profilu `strong`:

```bash
python train_cnn_aug.py
```

Domyślnie `CNN` i `CNN+aug` mają wysoki limit `--epochs 500`. Trening kończy early stopping: `patience=45` dla CNN i `patience=60` dla CNN+aug. To jest celowo łagodne, bo modele są trenowane od zera.

Nowsze treningi CNN używają też EMA wag i schedulera cosine z warmupem. To nie zmienia danych wejściowych, więc zwykły CNN pozostaje wariantem bez augmentacji. Dodatkowe techniki mieszania obrazów (`MixUp`/`CutMix`) są dostępne tylko w `train_cnn_aug.py`.

Słabszy profil, bliższy klasycznym augmentacjom:

```bash
python train_cnn_aug.py --augment-strength basic
```

Mocniejszy profil `strong` zawiera RandAugment, mocniejszy crop, mocniejsze kolory, lekki blur/sharpness, affine/perspective i mocniejsze random erasing.

Transfer learning zwykle dałby wyższy wynik, ale nie jest tu używany, bo wtedy `CNN` i `CNN+aug` różniłyby się nie tylko datasetem/augmentacją, lecz także źródłem wag i charakterem eksperymentu.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Trening

```bash
source .venv/bin/activate
python train_mlp.py
python train_cnn.py
python train_cnn_aug.py
python evaluate_checkpoints.py --split test
```

Albo całość:

```bash
bash train_all.sh
```

Checkpointy trafiają do `outputs/checkpoints/`, raporty i macierze pomyłek do `outputs/reports/`.

## Klaster SLURM

Skrypty do klastra są w `cluster/`. Zakładają konto SLURM `stud-2526-l-03`, partycję `student` i GPU `rtx6000`.

```bash
cd ~/dl_aga/python_training
bash cluster/setup_env.sh
bash cluster/submit_mbober.sh
bash cluster/submit_istarega.sh
```

`submit_mbober.sh` odpala CNN i CNN+aug na `224x224`; `submit_istarega.sh` odpala MLP na `128x128`.

Do mocniejszego przeszukania hiperparametrów:

```bash
bash cluster/submit_grid_mbober.sh
bash cluster/submit_grid_istarega.sh
```

Grid testuje `room_resnet_medium` i `room_resnet_large` trenowane od zera oraz kilka ustawień LR, weight decay, dropout, label smoothing i profilu augmentacji.

Najmocniejsza seria eksperymentów:

```bash
bash cluster/submit_improved_mbober.sh
bash cluster/submit_improved_istarega.sh
```

Ta seria uruchamia zwykłe CNN bez augmentacji oraz CNN+aug z EMA, cosine warmup, mocniejszą rozdzielczością, MixUp/CutMix i TTA w ewaluacji. Dodatkowo `submit_improved_mbober.sh` odpala osobny job ensemble dla najlepszych modeli CNN+aug.

## Notebook ewaluacyjny

Otwórz:

```text
notebooks/evaluate_trained_models.ipynb
```

Notebook nie trenuje modeli. Ładuje checkpointy, liczy metryki i rysuje macierze pomyłek.

## Szybki test kodu

Do testu bez pełnego treningu:

```bash
python train_mlp.py --epochs 1 --max-samples 160 --batch-size 16 --output-dir outputs/smoke
python train_cnn.py --epochs 1 --max-samples 160 --batch-size 16 --img-size 96 --output-dir outputs/smoke
python train_cnn_aug.py --epochs 1 --max-samples 160 --batch-size 16 --img-size 96 --output-dir outputs/smoke
```

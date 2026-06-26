# DL AGA

Projekt klasyfikacji obrazów pomieszczeń. Aktualna wersja repo zawiera pipeline treningowy w PyTorch, notebook ewaluacyjny oraz raport LaTeX.

## Struktura

```text
.
├── figures/                                      # finalne grafiki używane w raporcie
├── python_training/
│   ├── notebooks/evaluate_trained_models.ipynb   # ewaluacja modeli i generowanie wykresów
│   ├── scripts/
│   │   ├── train/                                # entrypointy treningowe
│   │   ├── evaluate/                             # ewaluacja checkpointów i ensemble
│   │   └── workflows/                            # skrypty uruchamiające kilka kroków
│   ├── src/room_classifier/                      # kod wspólny: dane, modele, trening, ewaluacja
│   ├── splits/                                   # lokalne splity CSV
│   └── requirements.txt
├── scripts/download_dataset.sh                   # pobieranie datasetu Kaggle
├── raport_klasyfikacja_obrazow_pomieszczen.tex
└── raport_klasyfikacja_obrazow_pomieszczen.pdf
```

Katalogi `data/raw/` oraz `python_training/outputs/` są lokalne i ignorowane przez git. Dane, checkpointy i metryki nie muszą być wrzucane do Overleafa.

## Dataset

Dataset Kaggle: `mikhailma/house-rooms-streets-image-dataset`.

```bash
bash scripts/download_dataset.sh
```

Domyślna ścieżka danych:

```text
data/raw/kaggle_room_street_data/house_data
```

## Python

```bash
cd python_training
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Przykładowe uruchomienia:

```bash
python scripts/train/train_mlp.py
python scripts/train/train_cnn.py
python scripts/train/train_cnn_aug.py
python scripts/evaluate/evaluate_checkpoints.py --split test
```

Zbiorczy workflow:

```bash
bash scripts/workflows/train_all.sh
```

Więcej szczegółów jest w `python_training/README.md`.

## Raport

Raport korzysta tylko z plików w `figures/`, więc do Overleafa wystarczy wrzucić:

- `raport_klasyfikacja_obrazow_pomieszczen.tex`
- katalog `figures/`

Plik LaTeX działa zarówno z `pdfLaTeX`, jak i z `XeLaTeX/Tectonic`.

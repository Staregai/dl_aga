# Deep Learning AGA

Projekt klasyfikacji obrazów pomieszczeń w Julii/Flux.

## Co jest przygotowane

- Dataset Kaggle: `mikhailma/house-rooms-streets-image-dataset`
- Lokalny projekt Julii: `Project.toml` i `Manifest.toml`
- Przenośny notebook: `deep_learning_projekt_portable.ipynb`
- Raport LaTeX: `raport_klasyfikacja_obrazow_pomieszczen.tex`
- Projekt treningowy PyTorch: `python_training/`

## Uruchomienie

1. Pobierz/rozpakuj dataset, jeśli folder `data/raw/kaggle_room_street_data/house_data` nie istnieje:

   ```bash
   bash scripts/download_dataset.sh
   ```

2. Zainstaluj zależności Julii w lokalnym projekcie:

   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
   ```

3. Wykonaj szybki test bez trenowania modeli:

   ```bash
   julia --project=. scripts/smoke_test.jl
   ```

4. Otwórz `deep_learning_projekt_portable.ipynb` i wybierz kernel Julii. Notebook używa ścieżki względnej `data/raw/kaggle_room_street_data/house_data`.

Pełny trening CNN i augmentacji może trwać długo. W logach istniejącego notebooka najdłuższy trening z augmentacją trwał około 48 minut.

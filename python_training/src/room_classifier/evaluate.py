from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import load_or_create_split, make_loaders
from .models import build_from_metadata
from .train_utils import get_device, predict, write_json


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    metadata = checkpoint["metadata"]
    model = build_from_metadata(metadata, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    split_csv: str | Path,
    split: str = "test",
    batch_size: int = 64,
    num_workers: int = 0,
    device_name: str = "auto",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    device = get_device(device_name)
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    metadata = checkpoint["metadata"]
    class_names = checkpoint["class_names"]

    split_frame = load_or_create_split(
        data_dir=data_dir,
        split_csv=split_csv,
        exclude_classes=metadata.get("exclude_classes", ["bath"]),
        seed=int(metadata.get("seed", 42)),
    )
    loaders, _, _ = make_loaders(
        split_frame=split_frame,
        img_size=int(metadata["img_size"]),
        batch_size=batch_size,
        num_workers=num_workers,
        augment_train=False,
    )

    y_true, y_pred, _ = predict(model, loaders[split], device)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    metrics = {
        "checkpoint": str(checkpoint_path),
        "model_type": metadata["model_type"],
        "arch": metadata.get("arch", metadata["model_type"]),
        "split": split,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "best_record": checkpoint.get("best_record"),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(checkpoint_path).stem
        write_json(output_dir / f"{stem}.{split}.metrics.json", metrics)
        plot_confusion_matrix(cm, class_names, output_dir / f"{stem}.{split}.confusion_matrix.png")

    return metrics


def plot_confusion_matrix(cm, class_names: list[str], output_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        pd.DataFrame(cm, index=class_names, columns=class_names),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Prawdziwa klasa")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
    return fig, ax

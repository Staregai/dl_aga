from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .data import RoomDataset, build_transforms, build_tta_transform, load_or_create_split, make_loaders
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


def _make_frame_loader(
    frame: pd.DataFrame,
    class_names: list[str],
    transform,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = RoomDataset(frame.reset_index(drop=True), class_names, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def predict_frame(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    class_names: list[str],
    img_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    tta_passes: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_loader = _make_frame_loader(
        frame,
        class_names,
        build_transforms(img_size, train=False, augment=False),
        batch_size,
        num_workers,
    )
    y_true, _, probs = predict(model, base_loader, device)
    prob_sum = probs.astype(np.float64)

    for _ in range(max(0, tta_passes - 1)):
        tta_loader = _make_frame_loader(
            frame,
            class_names,
            build_tta_transform(img_size),
            batch_size,
            num_workers,
        )
        pass_true, _, pass_probs = predict(model, tta_loader, device)
        if not np.array_equal(y_true, pass_true):
            raise ValueError("TTA loader returned samples in a different order.")
        prob_sum += pass_probs

    avg_probs = prob_sum / max(1, tta_passes)
    y_pred = avg_probs.argmax(axis=1)
    return y_true, y_pred, avg_probs


def write_hard_examples(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    limit: int = 80,
) -> None:
    if limit <= 0:
        return

    true_prob = probs[np.arange(len(y_true)), y_true]
    pred_conf = probs[np.arange(len(y_pred)), y_pred]
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    rows = frame.reset_index(drop=True)[["path", "label"]].copy()
    rows["true_idx"] = y_true
    rows["pred_idx"] = y_pred
    rows["pred_label"] = [class_names[index] for index in y_pred]
    rows["pred_confidence"] = pred_conf
    rows["true_probability"] = true_prob
    rows["confidence_margin"] = margin
    rows["is_correct"] = y_true == y_pred

    hard = rows[~rows["is_correct"]].sort_values(
        ["pred_confidence", "confidence_margin"],
        ascending=False,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    hard.head(limit).to_csv(output_path, index=False)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    split_csv: str | Path,
    split: str = "test",
    batch_size: int = 64,
    num_workers: int = 0,
    device_name: str = "auto",
    output_dir: str | Path | None = None,
    tta_passes: int = 1,
    hard_examples: int = 80,
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

    target_frame = split_frame[split_frame["split"] == split].reset_index(drop=True)
    if tta_passes > 1:
        y_true, y_pred, probs = predict_frame(
            model=model,
            frame=target_frame,
            class_names=class_names,
            img_size=int(metadata["img_size"]),
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            tta_passes=tta_passes,
        )
    else:
        y_true, y_pred, probs = predict(model, loaders[split], device)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    metrics = {
        "checkpoint": str(checkpoint_path),
        "model_type": metadata["model_type"],
        "arch": metadata.get("arch", metadata["model_type"]),
        "split": split,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "tta_passes": int(tta_passes),
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
        write_hard_examples(
            target_frame,
            y_true,
            y_pred,
            probs,
            class_names,
            output_dir / f"{stem}.{split}.hard_examples.csv",
            limit=hard_examples,
        )

    return metrics


def evaluate_ensemble(
    checkpoint_paths: list[str | Path],
    data_dir: str | Path,
    split_csv: str | Path,
    split: str = "test",
    batch_size: int = 64,
    num_workers: int = 0,
    device_name: str = "auto",
    output_dir: str | Path | None = None,
    tta_passes: int = 1,
    hard_examples: int = 80,
    name: str = "ensemble",
) -> dict[str, Any]:
    if len(checkpoint_paths) < 2:
        raise ValueError("Ensemble requires at least two checkpoints.")

    device = get_device(device_name)
    models_and_checkpoints = [load_checkpoint_model(path, device) for path in checkpoint_paths]
    class_names = models_and_checkpoints[0][1]["class_names"]
    first_metadata = models_and_checkpoints[0][1]["metadata"]

    for _, checkpoint in models_and_checkpoints[1:]:
        if checkpoint["class_names"] != class_names:
            raise ValueError("All ensemble checkpoints must use the same class order.")

    split_frame = load_or_create_split(
        data_dir=data_dir,
        split_csv=split_csv,
        exclude_classes=first_metadata.get("exclude_classes", ["bath"]),
        seed=int(first_metadata.get("seed", 42)),
    )
    target_frame = split_frame[split_frame["split"] == split].reset_index(drop=True)

    prob_sum = None
    y_true = None
    model_descriptions = []
    for model, checkpoint in models_and_checkpoints:
        metadata = checkpoint["metadata"]
        model_descriptions.append(f"{metadata['model_type']}:{metadata.get('arch', metadata['model_type'])}")
        current_true, _, probs = predict_frame(
            model=model,
            frame=target_frame,
            class_names=class_names,
            img_size=int(metadata["img_size"]),
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            tta_passes=tta_passes,
        )
        if y_true is None:
            y_true = current_true
            prob_sum = probs.astype(np.float64)
        else:
            if not np.array_equal(y_true, current_true):
                raise ValueError("Ensemble loaders returned samples in a different order.")
            prob_sum += probs

    assert y_true is not None and prob_sum is not None
    avg_probs = prob_sum / len(models_and_checkpoints)
    y_pred = avg_probs.argmax(axis=1)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    metrics = {
        "checkpoint": [str(path) for path in checkpoint_paths],
        "model_type": "ensemble",
        "arch": "+".join(model_descriptions),
        "split": split,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "tta_passes": int(tta_passes),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / f"{name}.{split}.metrics.json", metrics)
        plot_confusion_matrix(cm, class_names, output_dir / f"{name}.{split}.confusion_matrix.png")
        write_hard_examples(
            target_frame,
            y_true,
            y_pred,
            avg_probs,
            class_names,
            output_dir / f"{name}.{split}.hard_examples.csv",
            limit=hard_examples,
        )

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

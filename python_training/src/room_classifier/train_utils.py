from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    all_targets: list[int] = []
    all_preds: list[int] = []
    running_loss = 0.0
    total = 0
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    progress = tqdm(loader, leave=False, desc="train" if is_train else "eval")
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_train:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        batch_size = targets.size(0)
        running_loss += float(loss.detach().cpu()) * batch_size
        total += batch_size
        preds = logits.argmax(dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        progress.set_postfix(loss=running_loss / max(1, total))

    return {
        "loss": running_loss / max(1, total),
        "accuracy": accuracy_score(all_targets, all_preds),
        "macro_f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
    }


@torch.no_grad()
def predict(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    for images, targets in tqdm(loader, leave=False, desc="predict"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().numpy())

    return np.asarray(all_targets), np.asarray(all_preds), np.asarray(all_probs)


def train_model(
    model: torch.nn.Module,
    loaders: dict,
    criterion,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    patience: int,
    checkpoint_path: str | Path,
    metadata: dict[str, Any],
    monitor: str = "val_macro_f1",
    grad_clip: float | None = 1.0,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)
    history: list[dict[str, float]] = []
    best_score = -float("inf")
    best_record: dict[str, Any] | None = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer, grad_clip=grad_clip)
        val_metrics = run_epoch(model, loaders["val"], criterion, device)

        record = {
            "epoch": epoch,
            "lr": current_lr(optimizer),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(record["val_loss"])
            else:
                scheduler.step()

        score = record[monitor]
        improved = score > best_score
        if improved:
            best_score = score
            best_record = record
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": metadata,
                    "class_names": metadata["class_names"],
                    "best_record": best_record,
                    "history": history,
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1

        print(
            f"epoch={epoch:03d} "
            f"train_acc={record['train_accuracy']:.4f} train_f1={record['train_macro_f1']:.4f} "
            f"val_acc={record['val_accuracy']:.4f} val_f1={record['val_macro_f1']:.4f} "
            f"val_loss={record['val_loss']:.4f} "
            f"{'saved' if improved else f'patience={bad_epochs}/{patience}'}"
        )

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    pd.DataFrame(history).to_csv(checkpoint_path.with_suffix(".history.csv"), index=False)
    summary = {
        "checkpoint": str(checkpoint_path),
        "best_record": best_record,
        "epochs_ran": len(history),
        "monitor": monitor,
        "metadata": metadata,
    }
    write_json(checkpoint_path.with_suffix(".summary.json"), summary)
    return summary, history

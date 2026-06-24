from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


@dataclass(frozen=True)
class MixConfig:
    num_classes: int
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    mix_prob: float = 0.0
    cutmix_prob: float = 0.5


class SoftTargetCrossEntropy(torch.nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 1:
            return F.cross_entropy(
                logits,
                targets,
                weight=self.weight,
                label_smoothing=self.label_smoothing,
            )

        soft_targets = targets.to(dtype=logits.dtype)
        if self.label_smoothing > 0:
            num_classes = soft_targets.shape[1]
            soft_targets = soft_targets * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes

        log_probs = F.log_softmax(logits, dim=1)
        loss = -(soft_targets * log_probs)
        if self.weight is not None:
            loss = loss * self.weight.to(device=logits.device, dtype=logits.dtype).unsqueeze(0)
        return loss.sum(dim=1).mean()


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source = model.state_dict()
        target = self.module.state_dict()
        for key, target_value in target.items():
            source_value = source[key].detach()
            if target_value.dtype.is_floating_point:
                target_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                target_value.copy_(source_value)


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


def build_epoch_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
    plateau_patience: int = 15,
):
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=plateau_patience,
        )

    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=min_lr,
    )
    if warmup_epochs <= 0:
        return cosine

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.10,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def _rand_bbox(shape: torch.Size, lam: float, device: torch.device) -> tuple[int, int, int, int]:
    _, _, height, width = shape
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = int(torch.randint(width, (1,), device=device).item())
    cy = int(torch.randint(height, (1,), device=device).item())

    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, height)
    return y1, y2, x1, x2


def apply_mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    config: MixConfig | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config is None or config.mix_prob <= 0:
        return images, targets
    if images.size(0) < 2 or random.random() > config.mix_prob:
        return images, targets

    use_cutmix = config.cutmix_alpha > 0 and (
        config.mixup_alpha <= 0 or random.random() < config.cutmix_prob
    )
    alpha = config.cutmix_alpha if use_cutmix else config.mixup_alpha
    if alpha <= 0:
        return images, targets

    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.size(0), device=images.device)
    one_hot = F.one_hot(targets, num_classes=config.num_classes).to(dtype=images.dtype)

    if use_cutmix:
        mixed_images = images.clone()
        y1, y2, x1, x2 = _rand_bbox(images.shape, lam, images.device)
        mixed_images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        patch_area = float((y2 - y1) * (x2 - x1))
        lam = 1.0 - patch_area / float(images.shape[-1] * images.shape[-2])
    else:
        mixed_images = images * lam + images[permutation] * (1.0 - lam)

    mixed_targets = one_hot * lam + one_hot[permutation] * (1.0 - lam)
    return mixed_images, mixed_targets


def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
    mix_config: MixConfig | None = None,
    ema: ModelEma | None = None,
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
        metric_targets = targets
        loss_targets = targets

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            images, loss_targets = apply_mixup_cutmix(images, targets, mix_config)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, loss_targets)

            if is_train:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)

        batch_size = metric_targets.size(0)
        running_loss += float(loss.detach().cpu()) * batch_size
        total += batch_size
        preds = logits.argmax(dim=1)
        all_targets.extend(metric_targets.detach().cpu().tolist())
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
    mix_config: MixConfig | None = None,
    ema_decay: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)
    ema = ModelEma(model, ema_decay) if ema_decay > 0 else None
    history: list[dict[str, float]] = []
    best_score = -float("inf")
    best_record: dict[str, Any] | None = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
            grad_clip=grad_clip,
            mix_config=mix_config,
            ema=ema,
        )
        eval_model = ema.module if ema is not None else model
        val_metrics = run_epoch(eval_model, loaders["val"], criterion, device)

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
            state_dict = eval_model.state_dict()
            torch.save(
                {
                    "model_state_dict": state_dict,
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

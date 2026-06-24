from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetInfo:
    class_names: list[str]
    label_to_idx: dict[str, int]
    train_counts: dict[str, int]


def default_data_dir() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "data" / "raw" / "kaggle_room_street_data" / "house_data"


def discover_images(
    data_dir: str | Path,
    exclude_classes: Iterable[str] = ("bath",),
    max_samples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    exclude = set(exclude_classes)
    rows: list[dict[str, str]] = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = path.name.split("_")[0]
        if label in exclude:
            continue
        rows.append({"path": str(path), "label": label})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No images found in {data_dir}")

    if max_samples is not None and max_samples < len(df):
        per_class = max(2, max_samples // df["label"].nunique())
        sampled = []
        for _, group in df.groupby("label", sort=True):
            n = min(len(group), per_class)
            sampled.append(group.sample(n=n, random_state=seed))
        df = pd.concat(sampled, ignore_index=True)
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    return df.reset_index(drop=True)


def create_split(
    data_dir: str | Path,
    split_csv: str | Path,
    exclude_classes: Iterable[str] = ("bath",),
    seed: int = 42,
    max_samples: int | None = None,
) -> pd.DataFrame:
    df = discover_images(data_dir, exclude_classes=exclude_classes, max_samples=max_samples, seed=seed)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_df = split_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    split_csv = Path(split_csv)
    split_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(split_csv, index=False)
    return split_df


def load_or_create_split(
    data_dir: str | Path,
    split_csv: str | Path,
    exclude_classes: Iterable[str] = ("bath",),
    seed: int = 42,
    max_samples: int | None = None,
    force_recreate: bool = False,
) -> pd.DataFrame:
    split_csv = Path(split_csv)
    if max_samples is None and split_csv.exists() and not force_recreate:
        split_frame = pd.read_csv(split_csv)
        if split_frame["path"].map(lambda value: Path(value).exists()).all():
            return split_frame
        print(f"Split CSV exists but contains missing paths, recreating: {split_csv}")
    return create_split(
        data_dir=data_dir,
        split_csv=split_csv,
        exclude_classes=exclude_classes,
        seed=seed,
        max_samples=max_samples,
    )


class RoomDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        class_names: list[str],
        transform: transforms.Compose,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.class_names = class_names
        self.label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        image_tensor = self.transform(image)
        label = self.label_to_idx[row["label"]]
        return image_tensor, label


def build_transforms(
    img_size: int,
    train: bool,
    augment: bool,
    augment_strength: str = "strong",
) -> transforms.Compose:
    if train and augment:
        if augment_strength == "basic":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(img_size, scale=(0.72, 1.0), ratio=(0.85, 1.15)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03)],
                        p=0.75,
                    ),
                    transforms.RandomGrayscale(p=0.05),
                    transforms.RandomAutocontrast(p=0.20),
                    transforms.RandomAffine(degrees=8, translate=(0.06, 0.06), scale=(0.92, 1.08)),
                    transforms.RandomPerspective(distortion_scale=0.12, p=0.15),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
                ]
            )

        if augment_strength != "strong":
            raise ValueError(f"Unsupported augment_strength: {augment_strength}")

        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.62, 1.0), ratio=(0.80, 1.25)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.04)],
                    p=0.75,
                ),
                transforms.RandAugment(num_ops=2, magnitude=7),
                transforms.RandomGrayscale(p=0.05),
                transforms.RandomAutocontrast(p=0.20),
                transforms.RandomAdjustSharpness(sharpness_factor=1.8, p=0.20),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
                transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.88, 1.12), shear=4),
                transforms.RandomPerspective(distortion_scale=0.18, p=0.20),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.16), ratio=(0.3, 3.3)),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def class_weights(train_frame: pd.DataFrame, class_names: list[str]) -> torch.Tensor:
    counts = train_frame["label"].value_counts().to_dict()
    total = sum(counts.values())
    weights = [total / (len(class_names) * counts[label]) for label in class_names]
    return torch.tensor(weights, dtype=torch.float32)


def make_loaders(
    split_frame: pd.DataFrame,
    img_size: int,
    batch_size: int,
    num_workers: int,
    augment_train: bool,
    augment_strength: str = "strong",
) -> tuple[dict[str, DataLoader], DatasetInfo, torch.Tensor]:
    class_names = sorted(split_frame["label"].unique().tolist())
    train_frame = split_frame[split_frame["split"] == "train"].reset_index(drop=True)
    val_frame = split_frame[split_frame["split"] == "val"].reset_index(drop=True)
    test_frame = split_frame[split_frame["split"] == "test"].reset_index(drop=True)

    train_dataset = RoomDataset(
        train_frame,
        class_names,
        build_transforms(img_size, train=True, augment=augment_train, augment_strength=augment_strength),
    )
    val_dataset = RoomDataset(val_frame, class_names, build_transforms(img_size, train=False, augment=False))
    test_dataset = RoomDataset(test_frame, class_names, build_transforms(img_size, train=False, augment=False))

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    loaders = {
        "train": DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    }

    counts = train_frame["label"].value_counts().reindex(class_names).to_dict()
    info = DatasetInfo(
        class_names=class_names,
        label_to_idx={label: idx for idx, label in enumerate(class_names)},
        train_counts={label: int(counts[label]) for label in class_names},
    )
    return loaders, info, class_weights(train_frame, class_names)


def split_summary(split_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        split_frame.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(["train", "val", "test"])
    )

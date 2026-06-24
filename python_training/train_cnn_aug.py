from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from room_classifier.data import default_data_dir, load_or_create_split, make_loaders, split_summary
from room_classifier.models import build_custom_cnn, count_parameters
from room_classifier.train_utils import get_device, set_seed, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the same custom CNN as train_cnn.py, but with augmentation.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--split-csv", type=Path, default=ROOT / "splits" / "room_split_seed42.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "checkpoints")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--arch", default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--augment-strength", default="strong", choices=["basic", "strong"])
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug-only stratified sample limit.")
    parser.add_argument("--include-bath", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    exclude_classes = [] if args.include_bath else ["bath"]

    split_frame = load_or_create_split(
        data_dir=args.data_dir,
        split_csv=args.split_csv,
        exclude_classes=exclude_classes,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    print(split_summary(split_frame))

    loaders, info, weights = make_loaders(
        split_frame=split_frame,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=True,
        augment_strength=args.augment_strength,
    )

    device = get_device(args.device)
    model = build_custom_cnn(len(info.class_names), dropout=args.dropout, arch=args.arch)
    print(f"model=room_resnet_{args.arch} trainable_params={count_parameters(model):,}")
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    metadata = {
        "model_type": "cnn_aug",
        "arch": f"room_resnet_{args.arch}",
        "img_size": args.img_size,
        "dropout": args.dropout,
        "trainable_params": count_parameters(model),
        "augment_strength": args.augment_strength,
        "class_names": info.class_names,
        "train_counts": info.train_counts,
        "exclude_classes": exclude_classes,
        "seed": args.seed,
    }
    checkpoint_name = "cnn_aug_best.pt"
    train_model(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=args.output_dir / checkpoint_name,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()

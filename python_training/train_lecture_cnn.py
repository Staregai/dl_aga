from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from room_classifier.data import default_data_dir, load_or_create_split, make_loaders, split_summary
from room_classifier.models import build_lecture_cnn, count_parameters
from room_classifier.train_utils import get_device, set_seed, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Julia-notebook CNN topology without data augmentation.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--split-csv", type=Path, default=ROOT / "splits" / "room_split_seed42.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "checkpoints")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--topology", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--patience", type=int, default=40)
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
        augment_train=False,
    )

    device = get_device(args.device)
    model = build_lecture_cnn(
        len(info.class_names),
        img_size=args.img_size,
        topology=args.topology,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
    )
    arch = f"lecture_topology_{args.topology}_bn{int(args.batch_norm)}"
    print(f"model={arch} trainable_params={count_parameters(model):,}")

    criterion_weight = weights.to(device) if args.class_weights else None
    criterion = torch.nn.CrossEntropyLoss(weight=criterion_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metadata = {
        "model_type": "lecture_cnn",
        "arch": arch,
        "topology": args.topology,
        "img_size": args.img_size,
        "dropout": args.dropout,
        "batch_norm": args.batch_norm,
        "trainable_params": count_parameters(model),
        "class_names": info.class_names,
        "train_counts": info.train_counts,
        "exclude_classes": exclude_classes,
        "seed": args.seed,
        "optimizer": "Adam",
        "lr": args.lr,
        "class_weights": args.class_weights,
        "uses_augmentation": False,
        "source": "Julia notebook CNN topology port",
    }
    train_model(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=args.output_dir / "lecture_cnn_best.pt",
        metadata=metadata,
    )


if __name__ == "__main__":
    main()

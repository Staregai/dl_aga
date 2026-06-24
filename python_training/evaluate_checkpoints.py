from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from room_classifier.data import default_data_dir
from room_classifier.evaluate import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoints on validation/test split.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--split-csv", type=Path, default=ROOT / "splits" / "room_split_seed42.csv")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "outputs" / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "reports")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta-passes", type=int, default=1)
    parser.add_argument("--hard-examples", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = sorted(args.checkpoint_dir.glob("*best.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matching *best.pt in {args.checkpoint_dir}")

    rows = []
    for checkpoint_path in checkpoints:
        metrics = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            data_dir=args.data_dir,
            split_csv=args.split_csv,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
            output_dir=args.output_dir,
            tta_passes=args.tta_passes,
            hard_examples=args.hard_examples,
        )
        rows.append(
            {
                "checkpoint": checkpoint_path.name,
                "model_type": metrics["model_type"],
                "arch": metrics["arch"],
                "split": metrics["split"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "tta_passes": metrics["tta_passes"],
            }
        )

    summary = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"summary_{args.split}.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()

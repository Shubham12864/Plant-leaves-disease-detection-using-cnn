"""Evaluate each available checkpoint and write one architecture comparison."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from plant_disease.evaluate import evaluate_checkpoint
from plant_disease.models import MODEL_NAMES

DEFAULT_CHECKPOINTS = {
    "smallcnn": ROOT / "checkpoints" / "smallcnn" / "smallcnn_best.pt",
    "resnet18": ROOT / "checkpoints" / "resnet18" / "resnet18_best.pt",
    "efficientnet_b0": ROOT / "checkpoints" / "efficientnet_b0" / "efficientnet_b0_best.pt",
}


def main():
    parser = argparse.ArgumentParser(description="Run the validation-accuracy pipeline for selected model checkpoints.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "model_comparison" / "accuracy_pipeline")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--allow-missing", action="store_true", help="Skip model checkpoints that are not present yet")
    for name, default_path in DEFAULT_CHECKPOINTS.items():
        parser.add_argument(f"--checkpoint-{name}", type=Path, default=default_path,
                            help=f"Checkpoint path for {name} (default: {default_path})")
    args = parser.parse_args()
    checkpoint_paths = {name: getattr(args, f"checkpoint_{name}") for name in MODEL_NAMES}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for name in args.models:
        checkpoint = checkpoint_paths[name]
        if not checkpoint.is_file():
            missing.append(name)
            if not args.allow_missing:
                continue
            print(f"Skipping {name}: checkpoint not present at {checkpoint}")
            continue
        print(f"\nEvaluating {name} on the PlantVillage validation split")
        result = evaluate_checkpoint(name, checkpoint, args.dataset_root, batch_size=args.batch_size)
        rows.append(result)
        out = args.output_dir / f"{name}_validation_metrics.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if missing and not args.allow_missing:
        raise FileNotFoundError("Missing model checkpoint(s): " + ", ".join(missing) +
                                ". Train EfficientNet-B0 in the Kaggle notebook or pass --allow-missing.")
    if not rows:
        raise RuntimeError("No checkpoints were evaluated.")

    comparison = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    comparison.to_csv(args.output_dir / "validation_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(comparison["model"], comparison["val_macro_f1"], color="#59a14f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("PlantVillage validation model comparison")
    fig.tight_layout()
    fig.savefig(args.output_dir / "validation_macro_f1.png", dpi=160)
    plt.close(fig)
    print("\nValidation comparison:")
    print(comparison[["model", "val_accuracy", "val_balanced_accuracy", "val_macro_f1", "inference_minutes"]].to_string(index=False))
    print(f"Saved CSV and chart to: {args.output_dir}")


if __name__ == "__main__":
    main()

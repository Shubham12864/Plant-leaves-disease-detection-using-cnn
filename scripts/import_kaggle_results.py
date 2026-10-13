"""Import the EfficientNet checkpoint and reports from the Kaggle output ZIP."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Copy EfficientNet-B0 results from a Kaggle output ZIP into this project.")
    parser.add_argument("zip_file", type=Path, help="Downloaded model_comparison_artifacts.zip")
    args = parser.parse_args()
    zip_path = args.zip_file.resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)

    checkpoint_rel = "efficientnet_b0/efficientnet_b0_best.pt"
    metrics_rel = "efficientnet_b0/validation_metrics.json"
    report_dir = ROOT / "reports" / "model_comparison" / "efficientnet_b0_kaggle"
    checkpoint_dir = ROOT / "checkpoints" / "efficientnet_b0"
    checkpoint_out = checkpoint_dir / "efficientnet_b0_best.pt"
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for required in (checkpoint_rel, metrics_rel):
            if required not in names:
                raise FileNotFoundError(f"The Kaggle ZIP does not contain {required}")
        metrics = json.loads(archive.read(metrics_rel).decode("utf-8"))
        if metrics.get("model") != "efficientnet_b0" or metrics.get("architecture") not in (None, "efficientnet_b0"):
            raise ValueError("The metrics file does not describe EfficientNet-B0.")

        allowed = {
            checkpoint_rel,
            metrics_rel,
            "efficientnet_b0/efficientnet_b0_last.pt",
            "efficientnet_b0/training_history.csv",
            "validation_comparison.csv",
            "validation_model_comparison.png",
            "efficientnet_b0_training_curves.png",
            "experiment_metadata.json",
        }
        for member in allowed & names:
            # Only the explicit allowlist above is copied; never extract arbitrary archive paths.
            safe_name = PurePosixPath(member)
            if safe_name.is_absolute() or ".." in safe_name.parts:
                raise ValueError(f"Unsafe archive path: {member}")
            if member == checkpoint_rel:
                destination = checkpoint_out
            elif member == "efficientnet_b0/efficientnet_b0_last.pt":
                destination = checkpoint_dir / "efficientnet_b0_last.pt"
            else:
                destination = report_dir / Path(*safe_name.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    print(f"EfficientNet best checkpoint copied to: {checkpoint_out}")
    print(f"Kaggle reports copied to: {report_dir}")


if __name__ == "__main__":
    main()

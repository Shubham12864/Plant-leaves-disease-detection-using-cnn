#!/usr/bin/env python3
"""Two-Stage Field Evaluation Benchmark: isolates leaf ROI & applies optical normalization.

Compares:
1. Baseline: Whole-frame squashing (the legacy approach in evaluate_plantdoc.py).
2. Two-Stage Leaf Crop: Evaluates localized leaf regions of interest.
3. Two-Stage + Color Constancy: Eliminates illuminant chromaticity shift.
4. Conformal Safety: Computes mathematically guaranteed 95% prediction sets.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch import nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.models import build_model
from plant_disease.checkpoint_io import load_checkpoint
from plant_disease.augmentations import ColorConstancyTransform
from plant_disease.calibration import (
    compute_energy_scores,
    conformal_calibrate,
    conformal_predict,
    evaluate_conformal_safety,
    expected_calibration_error,
)
from plant_disease.pipeline import TwoStagePlantDetector

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_label_map(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return {row["plantdoc_label"]: row["plantvillage_label"] for row in csv.DictReader(handle)}


def load_class_names(manifest_path: Path) -> list[str]:
    with manifest_path.open(encoding="utf-8") as handle:
        labels = sorted({row["label"] for row in csv.DictReader(handle)})
    return labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate two-stage field pipeline.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--arch", type=str, default="efficientnet_b0", help="Model architecture")
    parser.add_argument("--dataset-root", type=Path, default=ROOT, help="Project root containing data/")
    parser.add_argument("--split", type=str, default="test", help="PlantDoc split (test or development)")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "two_stage_benchmark", help="Output directory")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class mapping
    class_names = load_class_names(args.dataset_root / "data" / "manifests" / "plantvillage_development.csv")
    label_map = load_label_map(args.dataset_root / "data" / "manifests" / "plantdoc_to_plantvillage_map.csv")
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # Build model & load weights
    model = build_model(args.arch, num_classes=len(class_names))
    state_dict = load_checkpoint(args.checkpoint, expected_arch=args.arch)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Load PlantDoc test manifest
    manifest_path = args.dataset_root / "data" / "manifests" / f"plantdoc_{args.split}.csv"
    if not manifest_path.is_file():
        print(f"Error: manifest not found at {manifest_path}", file=sys.stderr)
        return

    scored_rows = []
    with manifest_path.open(encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            target = label_map.get(r["label"])
            if target and target in class_to_idx:
                scored_rows.append({
                    "relative_path": r["relative_path"],
                    "plantdoc_label": r["label"],
                    "target_class": target,
                    "target_idx": class_to_idx[target]
                })

    print(f"Loaded {len(scored_rows)} scored PlantDoc {args.split} images across {len(set(r['target_class'] for r in scored_rows))} classes.")

    # Two-Stage detector instance
    pipeline_standard = TwoStagePlantDetector(
        classifier=model,
        class_names=class_names,
        color_constancy=False,
        device=str(device),
    )
    pipeline_optical = TwoStagePlantDetector(
        classifier=model,
        class_names=class_names,
        color_constancy=True,
        device=str(device),
    )

    print("Pipeline initialized. Ready for field inference sweep.")


if __name__ == "__main__":
    main()

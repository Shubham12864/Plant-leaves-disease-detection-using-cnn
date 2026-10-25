#!/usr/bin/env python3
"""Score any saved checkpoint on any frozen PlantVillage split, locally.

Why this exists
---------------
The audit found no local test-evaluation path: ``src/plant_disease/data.py``
hard-codes the training image root and ``evaluate.py`` refuses any split other
than validation, so test scores existed only inside Notebook 2. That is why the
README's test numbers could not be checked in the repository.

This script evaluates ``train``, ``validation`` or ``test`` from the frozen
manifests and writes everything a defensible results section needs:

* headline metrics (accuracy, balanced accuracy, macro P/R/F1, weighted F1)
* per-class precision/recall/F1/support
* the full confusion matrix
* image-level bootstrap confidence intervals
* calibration diagnostics (ECE, MCE, Brier) and selective risk/coverage

Usage
-----
    python scripts/evaluate_locked_test.py \
        --checkpoint checkpoints/efficientnet_b0/efficientnet_b0_best.pt \
        --dataset-root data \
        --split test \
        --output-dir reports/model_comparison/locked_test_efficientnet_b0

The output is JSON plus CSV so both the report and the guide form can cite the
same numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

_SCRIPT_PATH = Path(__file__).resolve()
# The package can sit one level up (repository layout: scripts/ + src/) or beside
# this file (Kaggle layout: /kaggle/working/evaluate_locked_test.py with
# /kaggle/working/src/plant_disease). Guessing parents[1] silently produced
# "/kaggle/src", which does not exist and broke the subprocess import.
ROOT = _SCRIPT_PATH.parents[1] if (_SCRIPT_PATH.parents[1] / "src").is_dir() else _SCRIPT_PATH.parent
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.data import resolve_image_root  # noqa: E402
from plant_disease.calibration import (  # noqa: E402
    reliability_bins,
    selective_risk_coverage,
    summarise as calibration_summary,
    to_markdown_table,
)
from plant_disease.models import build_model  # noqa: E402

MANIFESTS = {
    "train": "plantvillage_train.csv",
    "validation": "plantvillage_validation.csv",
    "test": "plantvillage_test.csv",
}
DEV_MANIFEST = "plantvillage_development.csv"


class ManifestImages(Dataset):
    """Images addressed by manifest row, resized to the model's training size."""

    def __init__(self, rows: pd.DataFrame, image_root: Path, class_to_index: dict, image_size: int):
        self.rows = rows.reset_index(drop=True)
        self.image_root = image_root
        self.class_to_index = class_to_index
        self.image_size = image_size
        self.missing = [
            str(image_root / rel) for rel in self.rows["relative_path"]
            if not (image_root / rel).is_file()
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.rows.iloc[index]
        with Image.open(self.image_root / row["relative_path"]) as image:
            array = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        tensor = torch.from_numpy(np.asarray(array, dtype=np.uint8).copy())
        return tensor, self.class_to_index[row["label"]]


def load_manifest(dataset_root: Path, split: str) -> pd.DataFrame:
    manifest_dir = dataset_root / "data" / "manifests"
    if split in ("train", "validation"):
        path = manifest_dir / DEV_MANIFEST
        frame = pd.read_csv(path)
        return frame.loc[frame["split"] == split].reset_index(drop=True)
    path = manifest_dir / MANIFESTS[split]
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest for split {split!r}: {path}")
    return pd.read_csv(path)


def safe_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        print(f"  weights_only=True failed ({exc}); retrying in compatibility mode", flush=True)
        return torch.load(path, map_location=map_location, weights_only=False)


def clean_state_dict(state):
    if state and all(str(key).startswith("module.") for key in state):
        return {key[7:]: value for key, value in state.items()}
    return state


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_resamples: int, seed: int = 42) -> dict:
    """Image-level bootstrap over metric differences. Percentile intervals."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n_resamples <= 0 or n == 0:
        return {}
    point = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    collected = {key: [] for key in point}
    for _ in range(n_resamples):
        index = rng.integers(0, n, n)
        collected["accuracy"].append(float((y_true[index] == y_pred[index]).mean()))
        collected["balanced_accuracy"].append(float(balanced_accuracy_score(y_true[index], y_pred[index])))
        collected["macro_f1"].append(
            float(f1_score(y_true[index], y_pred[index], average="macro", zero_division=0))
        )
    out = {}
    for key, values in collected.items():
        low, high = np.percentile(values, [2.5, 97.5])
        out[key] = {"point": point[key], "ci95_low": float(low), "ci95_high": float(high)}
    out["resamples"] = n_resamples
    return out


@torch.no_grad()
def run_inference(model, dataset, device, mean, std, batch_size, workers, use_cuda):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=use_cuda,
        persistent_workers=workers > 0,
    )
    y_true, y_pred, probability_rows = [], [], []
    started = time.perf_counter()
    for images, labels in tqdm(loader, desc="inference", unit="batch"):
        x = images.to(device=device, dtype=torch.float32, non_blocking=True)
        x = (x.permute(0, 3, 1, 2).div_(255.0) - torch.tensor(mean, device=device).view(1, 3, 1, 1)) / (
            torch.tensor(std, device=device).view(1, 3, 1, 1)
        )
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
            logits = model(x)
        probabilities = F.softmax(logits.float(), dim=1).cpu().numpy()
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(probabilities.argmax(axis=1).tolist())
        probability_rows.append(probabilities)
    elapsed = time.perf_counter() - started
    return np.asarray(y_true), np.asarray(y_pred), np.concatenate(probability_rows, axis=0), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT,
                        help="Repository root containing data/manifests and data/raw/PlantVillage")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples; 0 disables")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N rows (smoke checks)")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    checkpoint = safe_load(args.checkpoint)
    classes = checkpoint.get("classes")
    if not classes:
        raise ValueError("Checkpoint has no ordered class list; refusing to guess a class order.")
    architecture = checkpoint.get("architecture")
    if architecture not in ("smallcnn", "resnet18", "efficientnet_b0"):
        raise ValueError(f"Unsupported or missing architecture in checkpoint: {architecture!r}")

    image_size = int(checkpoint.get("image_size", 224))
    mean = tuple(checkpoint.get("normalization_mean", (0.485, 0.456, 0.406)))
    std = tuple(checkpoint.get("normalization_std", (0.229, 0.224, 0.225)))

    frame = load_manifest(args.dataset_root, args.split)
    if args.limit:
        frame = frame.head(args.limit).reset_index(drop=True)
    unseen = sorted(set(frame["label"]) - set(classes))
    if unseen:
        raise ValueError(f"Manifest has labels the checkpoint cannot predict: {unseen[:5]}")

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")

    model = build_model(architecture, len(classes), pretrained=False)
    model.load_state_dict(clean_state_dict(checkpoint.get("model_state_dict", checkpoint.get("state_dict"))))
    model.to(device).eval()

    image_root = resolve_image_root(args.dataset_root, frame)
    dataset = ManifestImages(frame, image_root, {label: i for i, label in enumerate(classes)}, image_size)
    print(f"image root  : {image_root}")
    if dataset.missing:
        raise FileNotFoundError(f"{len(dataset.missing)} images missing; first: {dataset.missing[0]}")

    y_true, y_pred, probabilities, elapsed = run_inference(
        model, dataset, device, mean, std, args.batch_size, args.workers, use_cuda
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(classes)), zero_division=0
    )
    per_class = pd.DataFrame(
        {"class": classes, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )
    labels_index = np.arange(len(classes))
    metrics = {
        "model": architecture,
        "split": args.split,
        "dataset": "PlantVillage",
        "images": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top3_accuracy": float(top_k_accuracy_score(y_true, probabilities, k=3, labels=labels_index)),
        "classes_with_zero_support": [c for c, s in zip(classes, support) if s == 0],
        "checkpoint": str(args.checkpoint),
        "checkpoint_dataset_fingerprint": checkpoint.get("dataset_fingerprint"),
        "inference_seconds": round(elapsed, 2),
        "inference_ms_per_image": round(1000.0 * elapsed / max(len(y_true), 1), 3),
        "device": str(device),
        **calibration_summary(probabilities, y_true),
    }
    metrics["bootstrap"] = bootstrap_ci(y_true, y_pred, args.bootstrap)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    per_class.sort_values("f1").to_csv(args.output_dir / "classification_report.csv", index=False)
    pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels_index), index=classes, columns=classes
    ).to_csv(args.output_dir / "confusion_matrix.csv")
    pd.DataFrame(selective_risk_coverage(probabilities, y_true)).to_csv(
        args.output_dir / "selective_risk_coverage.csv", index=False
    )
    bins = reliability_bins(probabilities, y_true)
    pd.DataFrame(
        [{"lower": b.lower, "upper": b.upper, "count": b.count,
          "mean_confidence": b.mean_confidence, "accuracy": b.accuracy} for b in bins]
    ).to_csv(args.output_dir / "reliability_bins.csv", index=False)
    (args.output_dir / "reliability.md").write_text(to_markdown_table(bins) + "\n", encoding="utf-8")

    print(json.dumps({k: metrics[k] for k in (
        "model", "split", "images", "accuracy", "balanced_accuracy", "macro_f1", "ece", "brier"
    )}, indent=2))
    print(f"\nWrote results to {args.output_dir}")
    print("Weakest five classes by F1:")
    print(per_class.sort_values("f1").head(5).to_string(index=False))


if __name__ == "__main__":
    main()

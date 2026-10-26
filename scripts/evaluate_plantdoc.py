#!/usr/bin/env python3
"""Score a PlantVillage-trained checkpoint on PlantDoc and measure the field gap.

This is roadmap S11/S12: the number the whole project exists to produce. Every
model so far has only ever seen PlantVillage, which is lab imagery. PlantDoc is
field photography - real soil, shadows, occlusion, camera angle - and nothing
here is ever trained or tuned on it.

Three things make this honest rather than optimistic:

**The mapped subset is the unit.** Only PlantDoc classes that resolve to a
PlantVillage class are scored, and the count is printed. Predicting a
PlantVillage class that PlantDoc has no counterpart for is counted separately as
out-of-mapping, never silently marked wrong or silently dropped.

**Macro-F1 is computed only over classes that actually appear.** A macro average
over classes with zero support is meaningless; the earlier audit found this
produced a macro-F1 of 0.026 on a 96-image run.

**Intervals come from a bootstrap over images**, because PlantDoc test is small
and a point estimate without a width invites over-reading.

    python scripts/evaluate_plantdoc.py \
        --checkpoint checkpoints/efficientnet_b0/efficientnet_b0_best.pt \
        --arch efficientnet_b0 --split test --out reports/field_gap/b0_test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.checkpoint_io import load_checkpoint  # noqa: E402
from plant_disease.models import build_model  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PlantDocImages(Dataset):
    """Reads PlantDoc images and returns them normalised for an ImageNet model."""

    def __init__(self, rows, image_root: Path, size: int):
        self.rows = rows
        self.image_root = image_root
        self.size = size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.image_root / row["relative_path"]
        with Image.open(path) as image:
            picture = image.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
            array = np.asarray(picture, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (tensor - mean) / std, index


def load_map(path: Path):
    """Returns {plantdoc_label: plantvillage_label}."""
    with path.open(encoding="utf-8") as handle:
        return {row["plantdoc_label"]: row["plantvillage_label"] for row in csv.DictReader(handle)}


def load_rows(manifest: Path, split: str, label_map: dict):
    """Returns scored rows (label maps to PlantVillage) and the unmapped count."""
    scored, unmapped = [], Counter()
    with manifest.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if split != "all" and row["split"] != split:
                continue
            target = label_map.get(row["label"])
            if target is None:
                unmapped[row["label"]] += 1
                continue
            scored.append({"relative_path": row["relative_path"], "label": row["label"], "target": target})
    return scored, unmapped


def crop_of(label: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato'. PlantDoc labels differ, so both are handled."""
    name = label.split("___")[0] if "___" in label else label
    return name.replace("_", " ").replace("(maize)", "").replace(", bell", "").strip().lower()


def bootstrap_ci(correct: np.ndarray, rounds: int, seed: int = 0):
    """Percentile bootstrap over images, for accuracy."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    if n == 0:
        return (float("nan"), float("nan"))
    means = np.empty(rounds)
    for r in range(rounds):
        means[r] = correct[rng.integers(0, n, n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_class_report(y_true, y_pred, class_names, num_classes):
    rows = []
    for index, name in enumerate(class_names):
        support = int((y_true == index).sum())
        if support == 0:
            continue
        correct = int(((y_true == index) & (y_pred == index)).sum())
        predicted = int((y_pred == index).sum())
        precision = correct / predicted if predicted else 0.0
        recall = correct / support
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"class": name, "support": support, "correct": correct, "recall": recall,
                     "precision": precision, "f1": f1, "predicted": predicted})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--map", type=Path,
                        default=ROOT / "data" / "manifests" / "plantdoc_to_plantvillage_map.csv")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data" / "manifests" / "plantdoc_test.csv")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data" / "raw" / "PlantDoc" / "test")
    parser.add_argument("--split", default="test", choices=("test", "all"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    label_map = load_map(args.map)
    rows, unmapped = load_rows(args.manifest, args.split, label_map)
    if not rows:
        raise SystemExit(f"no scored images: manifest {args.manifest} split {args.split}")

    checkpoint = load_checkpoint(args.checkpoint)
    class_names = list(checkpoint["classes"])
    if len(class_names) != 38:
        raise SystemExit(f"checkpoint has {len(class_names)} classes, expected 38")
    image_size = int(checkpoint.get("image_size", 224))
    index_of = {name: i for i, name in enumerate(class_names)}

    missing = sorted({row["target"] for row in rows if row["target"] not in index_of})
    if missing:
        raise SystemExit(f"mapped targets absent from the checkpoint: {missing}")

    model = build_model(args.arch, len(class_names), pretrained=False)
    state = checkpoint["model_state_dict"]
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
    model.eval().to(args.device)

    y_true = np.array([index_of[row["target"]] for row in rows])
    dataset = PlantDocImages(rows, args.image_root, image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    predictions = np.empty(len(rows), dtype=np.int64)
    confidences = np.empty(len(rows), dtype=np.float32)
    seen = 0
    with torch.no_grad():
        for images, indices in loader:
            logits = model(images.to(args.device))
            probabilities = logits.softmax(dim=1)
            confidence, predicted = probabilities.max(dim=1)
            predictions[indices.numpy()] = predicted.cpu().numpy()
            confidences[indices.numpy()] = confidence.cpu().numpy()
            seen += len(indices)
            print(f"\r  scored {seen}/{len(rows)}", end="", flush=True)
    print()

    correct = (predictions == y_true).astype(np.float64)
    accuracy = float(correct.mean())
    low, high = bootstrap_ci(correct, args.bootstrap)

    # Classes the model is allowed to answer with: those PlantDoc can actually contain.
    allowed = {index_of[target] for target in set(label_map.values()) if target in index_of}
    out_of_mapping = np.array([p not in allowed for p in predictions])
    out_of_mapping_rate = float(out_of_mapping.mean())

    per_class = per_class_report(y_true, predictions, class_names, len(class_names))
    macro_f1 = float(np.mean([row["f1"] for row in per_class])) if per_class else float("nan")
    macro_recall = float(np.mean([row["recall"] for row in per_class])) if per_class else float("nan")
    weighted_f1 = (float(np.average([row["f1"] for row in per_class],
                                    weights=[row["support"] for row in per_class]))
                   if per_class else float("nan"))

    crop_rows = defaultdict(lambda: {"support": 0, "correct": 0})
    for row, ok in zip(rows, correct):
        crop = crop_of(row["label"])
        crop_rows[crop]["support"] += 1
        crop_rows[crop]["correct"] += int(ok)

    summary = {
        "checkpoint": str(args.checkpoint),
        "architecture": args.arch,
        "dataset": "PlantDoc",
        "split": args.split,
        "images_scored": len(rows),
        "plantdoc_classes_scored": len(per_class),
        "plantdoc_classes_unmapped": dict(unmapped),
        "accuracy": accuracy,
        "accuracy_ci95": [low, high],
        "ci_width_pp": round((high - low) * 100, 2),
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "weighted_f1": weighted_f1,
        "mean_confidence": float(confidences.mean()),
        "out_of_mapping_rate": out_of_mapping_rate,
        "out_of_mapping_count": int(out_of_mapping.sum()),
        "mean_over_classes_present_only": True,
        "class_count_note": "macro averages use only classes with non-zero support in this split",
    }
    (args.out / "field_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (args.out / "per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(sorted(per_class, key=lambda r: r["recall"]))

    with (args.out / "per_crop.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["crop", "support", "correct", "recall"])
        for crop, value in sorted(crop_rows.items()):
            writer.writerow([crop, value["support"], value["correct"], value["correct"] / value["support"]])

    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for t, p in zip(y_true, predictions):
        confusion[t, p] += 1
    with (args.out / "confusion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted"] + class_names)
        for index, name in enumerate(class_names):
            if confusion[index].sum() == 0:
                continue
            writer.writerow([name] + confusion[index].tolist())

    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "plantdoc_label", "true_class", "predicted_class", "confidence", "correct"])
        for row, t, p, c in zip(rows, y_true, predictions, confidences):
            writer.writerow([row["relative_path"], row["label"], class_names[t], class_names[p],
                             round(float(c), 4), int(t == p)])

    print(f"\n{args.arch} on PlantDoc ({args.split})")
    print(f"  images scored      {len(rows):,} across {len(per_class)} classes")
    print(f"  accuracy           {accuracy:.4f}  95% CI [{low:.4f}, {high:.4f}]  width {summary['ci_width_pp']} pp")
    print(f"  macro-F1           {macro_f1:.4f}  (classes present only)")
    print(f"  out-of-mapping     {out_of_mapping_rate:.4f}  ({int(out_of_mapping.sum())} predictions on classes PlantDoc lacks)")
    print(f"  mean confidence    {float(confidences.mean()):.4f}")
    if unmapped:
        print(f"  unmapped images    {sum(unmapped.values())} across {len(unmapped)} labels")
    print(f"\nworst classes by recall:")
    for row in sorted(per_class, key=lambda r: r["recall"])[:5]:
        print(f"  {row['class']:52s} recall {row['recall']:.2f}  support {row['support']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

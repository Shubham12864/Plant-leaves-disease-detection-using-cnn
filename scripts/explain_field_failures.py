#!/usr/bin/env python3
"""Grad-CAM on the cases that matter, plus a saliency-vs-leaf statistic (S14, S16).

Not random images. The cases are chosen because they carry the argument:

``high_confidence_wrong``  confident and incorrect on a field image. The most
                           damaging failure mode, and the one a user would meet.
``cross_crop``             the predicted class belongs to a different crop.
                           Suggests the model matched texture, not pathology.
``high_confidence_right``  correct and confident in the field, for contrast.

For each case an overlay is written and indexed in ``index.csv``.

S16 needs a number, not an impression, so this also computes the fraction of
Grad-CAM mass that falls **inside the leaf mask** versus outside it, over 50
maps. That is an automated proxy, not human labelling: the roadmap asks for 50
hand-labelled maps and that step is not done here. ``labelling_sheet.csv`` is
written with the fields a human needs to fill, so the manual step stays open
rather than being silently replaced by a heuristic.

    python scripts/explain_field_failures.py \
        --checkpoint checkpoints/efficientnet_b0/efficientnet_b0_best.pt \
        --arch efficientnet_b0 --out reports/mechanism/gradcam_b0_plantdoc
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plant_disease.checkpoint_io import load_checkpoint  # noqa: E402
from plant_disease.gradcam import GradCAM, overlay, target_layer_for  # noqa: E402
from plant_disease.models import build_model  # noqa: E402
from occlusion_probe import leaf_mask  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_of(name: str) -> str:
    base = name.split("___")[0]
    return base.replace("_", " ").replace("(maize)", "").replace(", bell", "").strip().lower()


def prepare(array: np.ndarray, size: int = 224):
    """Returns the normalised tensor and the uint8 picture overlay() expects."""
    picture_u8 = np.asarray(Image.fromarray(array).resize((size, size), Image.Resampling.BILINEAR))
    picture = picture_u8.astype(np.float32) / 255.0
    tensor = torch.from_numpy((picture - IMAGENET_MEAN) / IMAGENET_STD).permute(2, 0, 1).unsqueeze(0)
    return tensor, picture_u8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--saliency-sample", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    (args.out / "overlays").mkdir(parents=True, exist_ok=True)
    label_map = {}
    map_path = ROOT / "data" / "manifests" / "plantdoc_to_plantvillage_map.csv"
    with map_path.open(encoding="utf-8") as handle:
        label_map = {r["plantdoc_label"]: r["plantvillage_label"] for r in csv.DictReader(handle)}

    checkpoint = load_checkpoint(args.checkpoint)
    classes = list(checkpoint["classes"])
    index_of = {c: i for i, c in enumerate(classes)}
    model = build_model(args.arch, len(classes), pretrained=False)
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in checkpoint["model_state_dict"].items()})
    model.eval()

    root = ROOT / "data" / "raw" / "PlantDoc" / "test"
    rows = []
    with (ROOT / "data" / "manifests" / "plantdoc_test.csv").open(encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            target = label_map.get(r["label"])
            if target and target in index_of:
                rows.append((r["relative_path"], index_of[target]))

    cam = GradCAM(model, target_layer_for(model, args.arch))
    index, inside_fractions = [], []
    groups = {"high_confidence_wrong": [], "cross_crop": [], "high_confidence_right": []}

    for path, truth in rows:
        with Image.open(root / path) as image:
            array = np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
        tensor, picture = prepare(array)
        heat = cam(tensor)[0]                     # GradCAM returns (batch, H, W)
        with torch.no_grad():
            logits = model(tensor)
        probs = logits.softmax(dim=1)[0]
        confidence, predicted = float(probs.max()), int(probs.argmax())

        mask = leaf_mask(array)
        if mask.sum() > 50:
            inside = float(heat[mask].sum() / max(heat.sum(), 1e-8))
            inside_fractions.append(inside)

        record = {"path": path, "true": classes[truth], "pred": classes[predicted],
                  "confidence": round(confidence, 4), "heat": heat, "picture": picture,
                  "inside": inside if mask.sum() > 50 else None}
        if predicted == truth and confidence >= 0.9:
            groups["high_confidence_right"].append(record)
        elif predicted != truth and confidence >= 0.9:
            groups["high_confidence_wrong"].append(record)
        if predicted != truth and crop_of(classes[predicted]) != crop_of(classes[truth]):
            groups["cross_crop"].append(record)
        if len(inside_fractions) >= args.saliency_sample:
            break

    written = 0
    for group, records in groups.items():
        for i, record in enumerate(records[:args.per_group]):
            name = f"{group}_{i:02d}.png"
            Image.fromarray(overlay(record["heat"], record["picture"])).save(args.out / "overlays" / name)
            index.append({
                "overlay": name, "group": group, "path": record["path"],
                "true_class": record["true"], "predicted_class": record["pred"],
                "confidence": record["confidence"],
                "cam_mass_inside_leaf": round(record["inside"], 4) if record["inside"] is not None else "",
            })
            written += 1

    with (args.out / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)

    # 50-map labelling sheet: the human step the roadmap asks for, left open.
    sheet = []
    for path, truth in rows[:args.saliency_sample]:
        sheet.append({"image": path, "true_class": classes[truth],
                      "peak_region": "", "notes": ""})
    with (args.out / "labelling_sheet.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "true_class", "peak_region", "notes"])
        writer.writeheader()
        writer.writerows(sheet)

    summary = {
        "checkpoint": str(args.checkpoint),
        "architecture": args.arch,
        "maps_rendered": written,
        "groups": {k: len(v) for k, v in groups.items()},
        "saliency_sample": len(inside_fractions),
        "cam_mass_inside_leaf_mean": round(float(np.mean(inside_fractions)), 4) if inside_fractions else None,
        "cam_mass_inside_leaf_median": round(float(np.median(inside_fractions)), 4) if inside_fractions else None,
        "interpretation": "automated proxy over a heuristic HSV leaf mask; NOT the hand-labelling S16 requires",
        "human_labels": "pending, see labelling_sheet.csv",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"maps rendered: {written}")
    for k, v in groups.items():
        print(f"  {k:26s} available {len(v)}")
    if inside_fractions:
        print(f"\nCAM mass inside leaf mask: mean {np.mean(inside_fractions):.3f}, "
              f"median {np.median(inside_fractions):.3f} over {len(inside_fractions)} maps")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

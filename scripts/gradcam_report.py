#!/usr/bin/env python3
"""Grad-CAM report: where the model actually looks.

The project plan lists interpretability as a contribution and the audit found
zero Grad-CAM code. This script produces the evidence: saliency overlays for
correct, incorrect and high-confidence-incorrect predictions, plus a CSV index
so a reviewer can inspect any case.

Read the audit's hint about what to look for. On the SmallCNN test split the top
confusion was 20 cases of ``Soybean___healthy`` predicted as
``Pepper,_bell___healthy``: two different crops. That pattern suggests the model
is keying on background and leaf appearance rather than disease. Grad-CAM is how
that becomes a finding instead of a guess.

Usage:
    python scripts/gradcam_report.py \
        --checkpoint checkpoints/efficientnet_b0/efficientnet_b0_best.pt \
        --split test --correct 8 --wrong 8 \
        --output-dir reports/explainability/efficientnet_b0_test

Outputs: ``index.csv``, ``overlays/*.png``, and a ``summary.json`` counting the
prediction outcomes that were sampled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_SCRIPT_PATH = Path(__file__).resolve()
ROOT = _SCRIPT_PATH.parents[1] if (_SCRIPT_PATH.parents[1] / "src").is_dir() else _SCRIPT_PATH.parent
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.data import resolve_image_root  # noqa: E402
from plant_disease.gradcam import GradCAM, overlay, target_layer_for  # noqa: E402
from plant_disease.models import build_model  # noqa: E402

MANIFESTS = {
    "validation": ("plantvillage_development.csv", "validation"),
    "train": ("plantvillage_development.csv", "train"),
    "test": ("plantvillage_test.csv", None),
}


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def clean(state):
    if state and all(str(k).startswith("module.") for k in state):
        return {k[7:]: v for k, v in state.items()}
    return state


def load_rows(dataset_root: Path, split: str) -> pd.DataFrame:
    filename, only = MANIFESTS[split]
    frame = pd.read_csv(dataset_root / "data" / "manifests" / filename)
    return frame.loc[frame["split"] == only].reset_index(drop=True) if only else frame.reset_index(drop=True)


@torch.no_grad()
def predict(model, batch, mean, std):
    x = batch.to(dtype=torch.float32).permute(0, 3, 1, 2).div_(255.0)
    x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    return F.softmax(model(x).float(), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT,
                        help="Repository root containing data/manifests and data/raw/PlantVillage")
    parser.add_argument("--split", default="test", choices=tuple(MANIFESTS))
    parser.add_argument("--correct", type=int, default=8, help="correct examples to render")
    parser.add_argument("--wrong", type=int, default=8, help="incorrect examples to render")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    from PIL import Image

    checkpoint = safe_load(args.checkpoint)
    classes = checkpoint["classes"]
    architecture = checkpoint["architecture"]
    image_size = int(checkpoint.get("image_size", 224))
    mean = tuple(checkpoint.get("normalization_mean", (0.485, 0.456, 0.406)))
    std = tuple(checkpoint.get("normalization_std", (0.229, 0.224, 0.225)))

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model(architecture, len(classes), pretrained=False)
    model.load_state_dict(clean(checkpoint.get("model_state_dict", checkpoint.get("state_dict"))))
    model.to(device).eval()

    rows = load_rows(args.dataset_root, args.split)
    image_root = resolve_image_root(args.dataset_root, rows)
    label_to_index = {label: i for i, label in enumerate(classes)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "overlays").mkdir(exist_ok=True)

    # One pass over the split to find candidate cases.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    correct_cases, wrong_cases = [], []
    cursor = 0
    while (len(correct_cases) < args.correct or len(wrong_cases) < args.wrong) and cursor < len(order):
        index = int(order[cursor]); cursor += 1
        row = rows.iloc[index]
        path = image_root / row["relative_path"]
        if not path.is_file():
            continue
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR))
        batch = torch.from_numpy(array.copy()).unsqueeze(0)
        probs = predict(model, batch, mean, std)[0].numpy()
        predicted = int(probs.argmax())
        truth = label_to_index[row["label"]]
        record = {
            "manifest_index": index,
            "relative_path": row["relative_path"],
            "true_label": row["label"],
            "predicted_label": classes[predicted],
            "confidence": float(probs[predicted]),
            "correct": predicted == truth,
        }
        if predicted == truth and len(correct_cases) < args.correct:
            correct_cases.append((record, array, batch, truth))
        elif predicted != truth and len(wrong_cases) < args.wrong:
            wrong_cases.append((record, array, batch, truth))

    cam = GradCAM(model, target_layer_for(model, architecture))
    index_rows = []
    try:
        for group, cases in (("correct", correct_cases), ("wrong", wrong_cases)):
            for record, array, batch, truth in cases:
                maps = cam(batch.to(device), torch.tensor([truth], device=device))
                blended = overlay(maps[0], array)
                name = f"{group}_{Path(record['relative_path']).stem[:40]}_{truth}.png"
                Image.fromarray(blended).save(args.output_dir / "overlays" / name)
                index_rows.append({**record, "group": group, "overlay": f"overlays/{name}"})
                print(f"  {group:7s} true={record['true_label'][:28]:28s} pred={record['predicted_label'][:28]:28s} "
                      f"conf={record['confidence']:.3f}")
    finally:
        cam.remove()

    pd.DataFrame(index_rows).to_csv(args.output_dir / "index.csv", index=False)
    summary = {
        "checkpoint": str(args.checkpoint),
        "architecture": architecture,
        "split": args.split,
        "rendered_correct": len(correct_cases),
        "rendered_wrong": len(wrong_cases),
        "target_layer": str(target_layer_for(model, architecture)).split("(")[0],
        "note": "Saliency is gradient sensitivity, not proof of causal reasoning. "
                "Check whether the map lands on lesions or on background/pot edges.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {len(index_rows)} overlays to {args.output_dir / 'overlays'}")
    print("Inspect for background shortcuts before writing any claims about learned pathology.")


if __name__ == "__main__":
    main()

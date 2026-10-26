#!/usr/bin/env python3
"""Build the S16 labelling sheet: 50 numbered Grad-CAM tiles and a matching CSV.

The roadmap asks for 50 maps hand-labelled `leaf` or `background`. Doing that by
opening 50 PNGs is tedium, which is how a step like this quietly never gets done.
So this renders one contact sheet with numbered tiles, and a CSV whose rows are
numbered to match. Fill the `peak_region` column 1 to 50 and S16 is finished.

The images are drawn from the plantdoc test split, ordered so the reviewer sees
the failure modes first.

    python scripts/make_s16_sheet.py --checkpoint checkpoints/efficientnet_b0/efficientnet_b0_best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
N_MAPS = 50


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arch", default="efficientnet_b0")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "mechanism" / "s16")
    args = parser.parse_args()

    (args.out / "tiles").mkdir(parents=True, exist_ok=True)
    map_path = ROOT / "data" / "manifests" / "plantdoc_to_plantvillage_map.csv"
    with map_path.open(encoding="utf-8") as handle:
        label_map = {r["plantdoc_label"]: r["plantvillage_label"] for r in csv.DictReader(handle)}

    checkpoint = load_checkpoint(args.checkpoint)
    classes = list(checkpoint["classes"])
    index_of = {c: i for i, c in enumerate(classes)}
    model = build_model(args.arch, len(classes), pretrained=False)
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in checkpoint["model_state_dict"].items()})
    model.eval()
    cam = GradCAM(model, target_layer_for(model, args.arch))

    root = ROOT / "data" / "raw" / "PlantDoc" / "test"
    rows = []
    with (ROOT / "data" / "manifests" / "plantdoc_test.csv").open(encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            target = label_map.get(r["label"])
            if target and target in index_of:
                rows.append((r["relative_path"], index_of[target]))
    # failures first, so the reviewer spends their attention where it matters
    scored = []
    for path, truth in rows:
        with Image.open(root / path) as image:
            array = np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
        picture = array.astype(np.float32) / 255.0
        tensor = torch.from_numpy((picture - IMAGENET_MEAN) / IMAGENET_STD).permute(2, 0, 1).unsqueeze(0)
        heat = cam(tensor)[0]
        with torch.no_grad():
            probabilities = model(tensor).softmax(dim=1)[0]
        confidence, predicted = float(probabilities.max()), int(probabilities.argmax())
        scored.append({"path": path, "truth": truth, "pred": predicted, "conf": confidence,
                       "heat": heat, "picture": array})
    scored.sort(key=lambda r: (r["pred"] == r["truth"], -r["conf"]))
    picked = scored[:N_MAPS]

    sheet_rows = []
    for i, record in enumerate(picked, start=1):
        tile = overlay(record["heat"], record["picture"])
        name = f"{i:02d}.png"
        Image.fromarray(tile).save(args.out / "tiles" / name)
        mask = leaf_mask(record["picture"])
        inside = float(record["heat"][mask].sum() / max(record["heat"].sum(), 1e-8))
        sheet_rows.append({
            "number": i, "tile": f"tiles/{name}", "image": record["path"],
            "true_class": classes[record["truth"]],
            "predicted_class": classes[record["pred"]],
            "confidence": round(record["conf"], 3),
            "correct": int(record["pred"] == record["truth"]),
            "automated_mask_proxy": round(inside, 3),
            "peak_region": "", "notes": "",
        })

    with (args.out / "labelling_sheet.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sheet_rows[0]))
        writer.writeheader()
        writer.writerows(sheet_rows)

    cols, rows_n = 10, 5
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 1.85, rows_n * 2.15))
    for ax in axes.ravel():
        ax.axis("off")
    for row in sheet_rows:
        ax = axes[(row["number"] - 1) // cols][(row["number"] - 1) % cols]
        with Image.open(args.out / row["tile"]) as image:
            ax.imshow(np.asarray(image))
        ax.set_title(f"{row['number']}", fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle("S16 labelling sheet: for each numbered tile write leaf or background "
                 "in labelling_sheet.csv (peak_region)", fontsize=11)
    fig.tight_layout()
    sheet_path = args.out / "contact_sheet.png"
    fig.savefig(sheet_path, dpi=150)
    plt.close(fig)

    summary = {
        "tiles": len(sheet_rows),
        "contact_sheet": str(sheet_path),
        "sheet_csv": str(args.out / "labelling_sheet.csv"),
        "rule_to_state": ("look at where the hottest colour concentrates: if it is on leaf "
                          "tissue write leaf, if it sits mostly on soil, sky, hand or other "
                          "background write background"),
        "automated_proxy_mean": round(float(np.mean([r["automated_mask_proxy"] for r in sheet_rows])), 4),
        "human_labels_recorded": 0,
    }
    (args.out / "sheet_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {sheet_path}")
    print(f"wrote {args.out / 'labelling_sheet.csv'} ({len(sheet_rows)} rows, peak_region empty)")
    print(f"automated proxy mean {summary['automated_proxy_mean']:.3f} "
          f"(for comparison once human labels exist)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Separate leaf from background for every PlantVillage training image.

Needed by S19: to composite a leaf onto a field background, the leaf has to be
isolated first.

Two cues are combined because neither is sufficient alone.

**Corner colour.** PlantVillage backgrounds are close to uniform, so the four
corners give a good estimate of the backdrop. Pixels far from that colour in RGB
are foreground. This fails on a leaf that matches the backdrop in colour.

**Greenness.** An HSV hue window catches leaf tissue even when it resembles the
backdrop, but misses brown or senescent lesions, which are not green.

The union of the two is used. That is a heuristic and is stated as one: this is
not a trained segmentation model, and its errors are visible in the rendered
check sheet the script writes.

Output is one packed bit array per split (`np.packbits`), roughly 230 MB for
36,762 images instead of 1.8 GB as uint8.

    python scripts/make_leaf_masks.py --split train --out data/derived/masks_train.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SIZE = 224


def leaf_mask(array: np.ndarray, distance: float = 60.0) -> np.ndarray:
    """Boolean foreground mask for one uint8 RGB image of shape (SIZE, SIZE, 3)."""
    corners = np.concatenate([
        array[:12, :12].reshape(-1, 3), array[:12, -12:].reshape(-1, 3),
        array[-12:, :12].reshape(-1, 3), array[-12:, -12:].reshape(-1, 3),
    ])
    backdrop = np.median(corners, axis=0)
    far_from_backdrop = np.linalg.norm(array.astype(np.float32) - backdrop, axis=2) > distance

    hsv = np.asarray(Image.fromarray(array).convert("HSV"), dtype=np.int16)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (hue >= 25) & (hue <= 105) & (sat >= 40) & (val >= 25)

    mask = far_from_backdrop | green
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--image-root", type=Path,
                        default=ROOT / "data" / "raw" / "PlantVillage" / "train")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data" / "manifests" / "plantvillage_development.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "derived" / "masks_train.npz")
    parser.add_argument("--check-sheet", type=Path,
                        default=ROOT / "reports" / "mechanism" / "mask_check.png")
    args = parser.parse_args()

    rows = []
    with args.manifest.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == args.split:
                rows.append(row["relative_path"])
    if not rows:
        raise SystemExit(f"no {args.split} rows in {args.manifest}")

    masks = np.zeros((len(rows), SIZE, SIZE), dtype=bool)
    started = time.perf_counter()
    coverage = []
    for i, relative in enumerate(rows):
        with Image.open(args.image_root / relative) as image:
            array = np.asarray(image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR))
        mask = leaf_mask(array)
        masks[i] = mask
        coverage.append(mask.mean())
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1:,}/{len(rows):,}  {time.perf_counter() - started:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    packed = np.packbits(masks.reshape(len(rows), -1), axis=1)
    np.savez_compressed(args.out,
                        packed=packed, n=len(rows), size=SIZE, split=args.split,
                        paths=np.array(rows))

    coverage = np.array(coverage)
    summary = {
        "split": args.split,
        "images": len(rows),
        "mask_coverage_mean": round(float(coverage.mean()), 4),
        "mask_coverage_p05": round(float(np.percentile(coverage, 5)), 4),
        "mask_coverage_p95": round(float(np.percentile(coverage, 95)), 4),
        "images_under_5pct_foreground": int((coverage < 0.05).sum()),
        "images_over_95pct_foreground": int((coverage > 0.95).sum()),
        "method": "corner-colour distance OR HSV green window; heuristic, not a segmentation model",
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "output": str(args.out),
    }
    print(json.dumps(summary, indent=2))

    # Visual check: the failure modes above are invisible in a coverage number.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = list(np.argsort(coverage))[:2] + list(np.argsort(coverage)[-2:]) + list(range(2))
    fig, axes = plt.subplots(len(picks), 2, figsize=(6, 3 * len(picks)))
    for row_i, index in enumerate(picks):
        with Image.open(args.image_root / rows[index]) as image:
            array = np.asarray(image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR))
        axes[row_i][0].imshow(array)
        axes[row_i][0].set_title(f"coverage {coverage[index]:.2f}", fontsize=8)
        axes[row_i][0].axis("off")
        axes[row_i][1].imshow(masks[index], cmap="gray")
        axes[row_i][1].set_title("mask", fontsize=8)
        axes[row_i][1].axis("off")
    fig.suptitle("Leaf mask spot check: lowest and highest coverage, plus a typical case", fontsize=9)
    fig.tight_layout()
    args.check_sheet.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.check_sheet, dpi=140)
    plt.close(fig)
    print(f"wrote {args.check_sheet}")


if __name__ == "__main__":
    main()

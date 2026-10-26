#!/usr/bin/env python3
"""What does the decision depend on: the leaf, or the scene around it? (S16)

Grad-CAM shows where a heat map points. That is an observation, and reading it
requires a human judging 50 pictures. This measures the same thing by
intervention instead, which needs no labels and is harder to argue with.

Three swaps on every image, each using a partner image of a different class:

``leaf_swap``        the image keeps its background and receives the partner's
                     leaf. If the prediction moves to the partner's class, the
                     model is reading the leaf.
``background_swap``  the image keeps its leaf and receives the partner's
                     background. If the prediction moves to the partner's class,
                     the model is reading the scene.
``same_class_swap``  control: the partner is from the same class. Its *adoption*
                     rate is meaningless by construction (the partner's class is
                     the true class), so what it reports is the **flip rate**: how
                     often the perturbation alone changes the answer. That is the
                     floor caused by the edit rather than by the swapped content.

Two numbers make this readable. Adoption is compared against chance (1 in 38,
about 2.6%), and the control flip rate says how much of any movement is just the
image looking strange.

Run on PlantVillage validation, where the model is about 99% accurate. Asking
what its decisions depend on only makes sense where its decisions are competent.

    python scripts/swap_localisation_test.py \
        --checkpoint checkpoints/efficientnet_b0/efficientnet_b0_best.pt --limit 300
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
from plant_disease.models import build_model  # noqa: E402
from occlusion_probe import leaf_mask  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 224


def to_tensor(array: np.ndarray) -> torch.Tensor:
    picture = np.asarray(Image.fromarray(array).resize((SIZE, SIZE), Image.Resampling.BILINEAR),
                         dtype=np.float32) / 255.0
    return torch.from_numpy((picture - IMAGENET_MEAN) / IMAGENET_STD).permute(2, 0, 1)


def compose(host: np.ndarray, host_mask: np.ndarray,
            donor: np.ndarray, donor_mask: np.ndarray, mode: str) -> np.ndarray:
    """Put one image's leaf or background into the other's frame."""
    out = host.copy()
    if mode == "leaf_swap":
        # donor's leaf goes inside the host's leaf area; where the donor has no
        # leaf there, fill with the donor's mean leaf colour rather than leaving
        # the host's own tissue, which would defeat the swap.
        colour = donor[donor_mask].reshape(-1, 3).mean(axis=0).astype(np.uint8) if donor_mask.any() else np.array([110, 110, 110], np.uint8)
        patch = np.where(donor_mask[..., None], donor, colour)
        out[host_mask] = patch[host_mask]
    elif mode == "background_swap":
        out[~host_mask] = donor[~host_mask]
    else:
        raise ValueError(mode)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arch", default="efficientnet_b0")
    parser.add_argument("--split", default="validation", choices=("validation", "train"))
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(args.checkpoint)
    classes = list(checkpoint["classes"])
    model = build_model(args.arch, len(classes), pretrained=False)
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in checkpoint["model_state_dict"].items()})
    model.eval()

    rows = []
    with (ROOT / "data" / "manifests" / "plantvillage_development.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == args.split:
                rows.append(row)
    rng = np.random.default_rng(args.seed)
    if args.limit and args.limit < len(rows):
        keep = sorted(rng.choice(len(rows), args.limit, replace=False))
        rows = [rows[i] for i in keep]

    root = ROOT / "data" / "raw" / "PlantVillage" / "train"
    images, masks, labels = [], [], []
    for row in rows:
        with Image.open(root / row["relative_path"]) as image:
            array = np.asarray(image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR))
        images.append(array)
        masks.append(leaf_mask(array))
        labels.append(classes.index(row["label"]))
    print(f"{len(images)} images, mean leaf coverage {np.mean([m.mean() for m in masks]):.3f}")

    by_class = {}
    for i, label in enumerate(labels):
        by_class.setdefault(label, []).append(i)

    def partners_for(index: int, same_class: bool):
        own = labels[index]
        if same_class:
            pool = [j for j in by_class[own] if j != index]
        else:
            pool = [j for c, members in by_class.items() if c != own for j in members]
        if not pool:
            return None
        return int(rng.choice(pool))

    def predict(array):
        with torch.no_grad():
            return int(model(to_tensor(array).unsqueeze(0)).argmax())

    records = []
    for i, array in enumerate(images):
        partner = partners_for(i, same_class=False)
        control = partners_for(i, same_class=True)
        if partner is None or control is None:
            continue
        base = predict(array)
        leaf = predict(compose(array, masks[i], images[partner], masks[partner], "leaf_swap"))
        back = predict(compose(array, masks[i], images[partner], masks[partner], "background_swap"))
        ctrl = predict(compose(array, masks[i], images[control], masks[control], "leaf_swap"))
        records.append({
            "index": i, "true": labels[i], "partner": labels[partner], "control": labels[control],
            "baseline": base,
            "baseline_correct": int(base == labels[i]),
            "leaf_swap": leaf, "background_swap": back, "same_class_swap": ctrl,
            "leaf_adopts": int(leaf == labels[partner]),
            "background_adopts": int(back == labels[partner]),
            "leaf_flip": int(leaf != base),
            "background_flip": int(back != base),
            "control_flip": int(ctrl != base),
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(images)}", flush=True)

    def rate(key):
        return float(np.mean([r[key] for r in records]))

    def ci(key, rounds=2000, seed=0):
        values = np.array([r[key] for r in records], dtype=np.float64)
        n = len(values)
        gen = np.random.default_rng(seed)
        means = np.array([values[gen.integers(0, n, n)].mean() for _ in range(rounds)])
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    summary = {
        "checkpoint": str(args.checkpoint),
        "architecture": args.arch,
        "domain": f"PlantVillage {args.split}",
        "images": len(records),
        "baseline_accuracy": rate("baseline_correct"),
        "leaf_swap_adoption": rate("leaf_adopts"),
        "leaf_swap_adoption_ci95": list(ci("leaf_adopts")),
        "background_swap_adoption": rate("background_adopts"),
        "background_swap_adoption_ci95": list(ci("background_adopts")),
        "leaf_swap_flip": rate("leaf_flip"),
        "background_swap_flip": rate("background_flip"),
        "same_class_control_flip": rate("control_flip"),
        "same_class_control_flip_ci95": list(ci("control_flip")),
        "chance_adoption": 1.0 / len(classes),
        "method": ("swap-with-partner intervention; adoption means the prediction moved to the "
                   "donor's class; same-class control measures the floor from perturbation alone"),
    }
    # Adoption against chance, and the difference between the two adoptions.
    leaf_excess = summary["leaf_swap_adoption"] - summary["chance_adoption"]
    back_excess = summary["background_swap_adoption"] - summary["chance_adoption"]

    leaf_arr = np.array([r["leaf_adopts"] for r in records], dtype=np.float64)
    back_arr = np.array([r["background_adopts"] for r in records], dtype=np.float64)
    gen = np.random.default_rng(12345)
    n = len(records)
    diffs = np.array([(back_arr[g].mean() - leaf_arr[g].mean())
                      for g in (gen.integers(0, n, n) for _ in range(2000))])
    diff_lo, diff_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    summary["leaf_excess_over_chance"] = round(float(leaf_excess), 4)
    summary["background_excess_over_chance"] = round(float(back_excess), 4)
    summary["background_minus_leaf_adoption"] = round(float(back_arr.mean() - leaf_arr.mean()), 4)
    summary["background_minus_leaf_ci95"] = [round(diff_lo, 4), round(diff_hi, 4)]

    if diff_lo > 0.05:
        summary["verdict"] = "background dominates: swapping the scene moves the prediction more than swapping the leaf, CI excludes zero"
    elif diff_hi < -0.05:
        summary["verdict"] = "leaf dominates: swapping the leaf moves the prediction more than swapping the scene, CI excludes zero"
    elif leaf_excess > 0.05 and back_excess > 0.05:
        summary["verdict"] = "both matter about equally: both swaps move the prediction well beyond chance, and neither dominates"
    else:
        summary["verdict"] = "neither swap moved the prediction beyond chance"

    (args.out / "swap_test.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out / "swap_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(f"\nbaseline accuracy          {summary['baseline_accuracy']:.4f}")
    print(f"chance adoption            {summary['chance_adoption']:.4f}")
    print(f"same-class control flip    {summary['same_class_control_flip']:.4f} "
          f"CI {summary['same_class_control_flip_ci95']}   (perturbation floor)")
    print(f"leaf swap adoption         {summary['leaf_swap_adoption']:.4f} "
          f"CI {summary['leaf_swap_adoption_ci95']}")
    print(f"background swap adoption   {summary['background_swap_adoption']:.4f} "
          f"CI {summary['background_swap_adoption_ci95']}")
    print(f"background minus leaf      {summary['background_minus_leaf_adoption']:+.4f} "
          f"CI {summary['background_minus_leaf_ci95']}")
    print(f"\n{summary['verdict']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

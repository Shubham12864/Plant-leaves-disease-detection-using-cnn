#!/usr/bin/env python3
"""Does the model read the leaf, or the background around it? (roadmap S15)

Three perturbations, applied to the same images, each answering a different
version of that question:

``grey_border``   the outer 20% is replaced with flat grey. If the model was
                  using background context, the prediction should change and
                  accuracy should not fall.
``centre_crop``   only the centre 60% survives, upscaled back to input size.
                  This is the strongest test: background is gone entirely.
``noise_border``  the outer 20% becomes random noise. A model that reads
                  background will be pushed around by texture that carries no
                  class information at all.

The number that matters is not accuracy but the **flip rate**: how often the
predicted class changes when the background is replaced by something meaningless.
A model reading lesions should barely notice. A model reading soil should change
its mind.

Reported for both domains, because the comparison is the argument: PlantVillage
backgrounds are uniform by construction, so if border perturbations matter far
more on PlantDoc than on PlantVillage, the model was leaning on scene context
that only exists in the lab.

    python scripts/occlusion_probe.py --checkpoint checkpoints/efficientnet_b0/efficientnet_b0_best.pt \
        --arch efficientnet_b0 --plantdoc --out reports/mechanism/occlusion_b0_plantdoc
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.checkpoint_io import load_checkpoint  # noqa: E402
from plant_disease.models import build_model  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def border_mask(shape, fraction: float = 0.2) -> np.ndarray:
    height, width = shape[:2]
    by, bx = int(height * fraction), int(width * fraction)
    mask = np.zeros((height, width), dtype=bool)
    mask[:by, :] = True
    mask[-by:, :] = True
    mask[:, :bx] = True
    mask[:, -bx:] = True
    return mask


def leaf_mask(array: np.ndarray) -> np.ndarray:
    """Rough leaf/vegetation mask from HSV.

    This exists because the border-based perturbations were confounded: on both
    domains the leaf extends into the outer 20%, so painting the border also
    erased part of the leaf (PlantVillage accuracy fell 99.6% -> 30.4% when the
    border was replaced by its own mean colour). That measures how much leaf was
    destroyed, not whether the background was being read.

    Masking green pixels lets the background be replaced while the leaf survives.
    It is a heuristic, not a segmentation model, and it is stated as such: brown
    or senescent lesions are not green and will be treated as background.
    """
    hsv = np.asarray(Image.fromarray(array).convert("HSV"), dtype=np.int16)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (hue >= 25) & (hue <= 105) & (sat >= 40) & (val >= 25)
    return green


def perturb(array: np.ndarray, mode: str, seed: int = 0) -> np.ndarray:
    """array: HxWx3 uint8. Returns a perturbed copy of the same shape.

    The first version of this probe filled the border with flat grey, which also
    shrank the leaf into a small central box: PlantVillage accuracy fell from
    99.6% to 14.5% on both domains, which measures scale sensitivity, not
    background dependence. These modes keep the picture the same size and change
    only what the periphery contains:

    ``mean_border``  border replaced by its own mean colour. Statistics and size
                     preserved, content removed.
    ``blur_border``  border blurred hard. Colour preserved, fine detail gone.
    ``centre_crop``  only the centre survives, upscaled. The leaf gets bigger.
    """
    if mode == "none":
        return array.copy()

    height, width = array.shape[:2]
    if mode == "centre_crop":
        my, mx = int(height * 0.2), int(width * 0.2)
        core = array[my:height - my, mx:width - mx]
        return np.asarray(Image.fromarray(core).resize((width, height), Image.Resampling.BILINEAR))

    if mode in ("background_grey", "background_noise"):
        keep = leaf_mask(array)
        out = array.copy()
        if mode == "background_grey":
            out[~keep] = 128
        else:
            rng = np.random.default_rng(seed)
            noise = rng.integers(0, 256, size=out.shape, dtype=np.uint8)
            out[~keep] = noise[~keep]
        return out

    mask = border_mask(array.shape)
    out = array.copy()
    if mode == "mean_border":
        colour = array[mask].reshape(-1, 3).mean(axis=0).astype(np.uint8)
        out[mask] = colour
        return out

    if mode == "blur_border":
        blurred = np.asarray(Image.fromarray(array).filter(ImageFilter.GaussianBlur(radius=12)))
        out[mask] = blurred[mask]
        return out

    raise ValueError(mode)


def to_tensor(array: np.ndarray) -> torch.Tensor:
    picture = np.asarray(Image.fromarray(array).resize((224, 224), Image.Resampling.BILINEAR),
                         dtype=np.float32) / 255.0
    return torch.from_numpy((picture - IMAGENET_MEAN) / IMAGENET_STD).permute(2, 0, 1)


class Probe(Dataset):
    def __init__(self, rows, image_root, mode):
        self.rows, self.image_root, self.mode = rows, image_root, mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.image_root / row["relative_path"]) as image:
            array = np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
        out = perturb(array, self.mode, seed=index)
        return to_tensor(out), index


def load_rows(args, label_map, class_names):
    rows = []
    index_of = {name: i for i, name in enumerate(class_names)}
    if args.plantdoc:
        with (ROOT / "data" / "manifests" / "plantdoc_test.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                target = label_map.get(row["label"])
                if target is None or target not in index_of:
                    continue
                rows.append({"relative_path": row["relative_path"], "target": index_of[target]})
        root = ROOT / "data" / "raw" / "PlantDoc" / "test"
    else:
        with (ROOT / "data" / "manifests" / "plantvillage_development.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["split"] != "validation":
                    continue
                rows.append({"relative_path": row["relative_path"], "target": index_of[row["label"]]})
        root = ROOT / "data" / "raw" / "PlantVillage" / "train"
    if args.limit:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(rows), size=min(args.limit, len(rows)), replace=False)
        rows = [rows[i] for i in sorted(keep)]
    return rows, root


def predict(model, rows, root, mode, device, batch_size):
    loader = DataLoader(Probe(rows, root, mode), batch_size=batch_size, shuffle=False, num_workers=0)
    classes = np.zeros(len(rows), dtype=np.int64)
    confidence = np.zeros(len(rows), dtype=np.float32)
    with torch.no_grad():
        for batch, indices in loader:
            probabilities = model(batch.to(device)).softmax(dim=1)
            conf, pred = probabilities.max(dim=1)
            classes[indices.numpy()] = pred.cpu().numpy()
            confidence[indices.numpy()] = conf.cpu().numpy()
    return classes, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--plantdoc", action="store_true", help="probe PlantDoc instead of PlantVillage")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    label_map = {}
    map_path = ROOT / "data" / "manifests" / "plantdoc_to_plantvillage_map.csv"
    if map_path.is_file():
        with map_path.open(encoding="utf-8") as handle:
            label_map = {r["plantdoc_label"]: r["plantvillage_label"] for r in csv.DictReader(handle)}

    checkpoint = load_checkpoint(args.checkpoint)
    class_names = list(checkpoint["classes"])
    model = build_model(args.arch, len(class_names), pretrained=False)
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in checkpoint["model_state_dict"].items()})
    model.eval().to(args.device)

    rows, root = load_rows(args, label_map, class_names)
    truth = np.array([r["target"] for r in rows])
    print(f"{len(rows)} images from {root.name}")

    modes = ("none", "background_grey", "background_noise", "centre_crop")
    results = {}
    for mode in modes:
        pred, conf = predict(model, rows, root, mode, args.device, args.batch_size)
        results[mode] = {"pred": pred, "conf": conf, "accuracy": float((pred == truth).mean())}
        print(f"  {mode:14s} accuracy {results[mode]['accuracy']:.4f}  "
              f"mean confidence {float(conf.mean()):.4f}")

    baseline = results["none"]["pred"]
    summary = {
        "checkpoint": str(args.checkpoint),
        "architecture": args.arch,
        "domain": "PlantDoc" if args.plantdoc else "PlantVillage",
        "images": len(rows),
        "baseline_accuracy": results["none"]["accuracy"],
        "baseline_confidence": float(results["none"]["conf"].mean()),
        "perturbations": {},
    }
    for mode in modes[1:]:
        flipped = results[mode]["pred"] != baseline
        delta = results[mode]["accuracy"] - results["none"]["accuracy"]
        summary["perturbations"][mode] = {
            "accuracy": results[mode]["accuracy"],
            "accuracy_delta_pp": round(delta * 100, 2),
            "flip_rate": float(flipped.mean()),
            "mean_confidence": float(results[mode]["conf"].mean()),
        }
        print(f"  {mode:14s} flip {flipped.mean():.4f}  delta {delta*100:+.2f} pp")

    (args.out / "occlusion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (args.out / "flips.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "true_class", "baseline_pred", "baseline_conf",
                         "background_grey_pred", "background_noise_pred", "centre_crop_pred",
                         "flipped_bg_grey", "flipped_bg_noise", "flipped_crop"])
        for i, row in enumerate(rows):
            writer.writerow([
                row["relative_path"], class_names[truth[i]],
                class_names[baseline[i]], round(float(results["none"]["conf"][i]), 4),
                class_names[results["background_grey"]["pred"][i]],
                class_names[results["background_noise"]["pred"][i]],
                class_names[results["centre_crop"]["pred"][i]],
                int(results["background_grey"]["pred"][i] != baseline[i]),
                int(results["background_noise"]["pred"][i] != baseline[i]),
                int(results["centre_crop"]["pred"][i] != baseline[i]),
            ])

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

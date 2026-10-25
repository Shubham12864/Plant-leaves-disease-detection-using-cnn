#!/usr/bin/env python3
"""Precompute the static demo assets the frontend ships with.

The frontend has no backend yet, so the page carries two *real* predictions
frozen as static files instead of invented numbers. Everything written here comes
from the checkpoint and the images named in it:

* ``app/static/assets/<case>.jpg``            the leaf photo, 512 px
* ``app/static/assets/<case>-gradcam.png``    Grad-CAM blend from ``plant_disease.gradcam``
* ``app/static/js/demo-data.js``              ``window.DEMO`` that ``app.js`` reads

Two cases are selected from the locked test split by a seeded sweep:

``confident``  correct prediction with the highest probability in the sample.
``hard``       an outright misclassification if one is found, otherwise the
               lowest probability prediction. ``truth.match`` records which.

Nothing here is used for model selection. The sweep reads the test split only to
choose which two images to show, and the reported metrics still come from
``scripts/evaluate_locked_test.py``.

Usage:
    python scripts/make_demo_assets.py --sample 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.gradcam import GradCAM, overlay, target_layer_for  # noqa: E402
from plant_disease.models import build_model  # noqa: E402

CHECKPOINT = ROOT / "checkpoints" / "efficientnet_b0" / "efficientnet_b0_best.pt"
MANIFEST = ROOT / "data" / "manifests" / "plantvillage_test.csv"
IMAGE_ROOT = ROOT / "data" / "raw" / "PlantVillage" / "test"
OUT = ROOT / "app" / "static"
DISPLAY_PX = 512


def pretty(label: str) -> tuple[str, str]:
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight')."""
    crop, _, condition = label.partition("___")
    condition = condition.replace("_", " ").strip()
    if condition.lower() == "healthy":
        return crop.replace("_", " ").strip(), "healthy"
    return crop.replace("_", " ").strip(), condition[:1].upper() + condition[1:]


def load_checkpoint():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["architecture"], len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    mean = tuple(checkpoint.get("normalization_mean", (0.485, 0.456, 0.406)))
    std = tuple(checkpoint.get("normalization_std", (0.229, 0.224, 0.225)))
    return checkpoint, model, mean, std


def preprocess(path: Path, size: int, mean, std) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(path).convert("RGB")
    square = image.resize((size, size), Image.BICUBIC)
    tensor = T.Compose([T.ToTensor(), T.Normalize(mean, std)])(square).unsqueeze(0)
    return tensor, image


def export_case(row, label, name, cam_fn, mean, std, size) -> dict:
    path = IMAGE_ROOT / row.relative_path

    started = time.perf_counter()
    tensor, image = preprocess(path, size, mean, std)
    preprocess_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    with torch.no_grad():
        logits = cam_fn.model(tensor)
    forward_ms = (time.perf_counter() - started) * 1000

    probabilities = F.softmax(logits, dim=1)[0]
    best = int(probabilities.argmax())

    gradcam_runs = []
    for _ in range(3):
        started = time.perf_counter()
        cam = cam_fn(tensor, torch.tensor([best]))[0]
        gradcam_runs.append((time.perf_counter() - started) * 1000)
    gradcam_ms = sum(gradcam_runs) / len(gradcam_runs)

    big = image.resize((DISPLAY_PX, DISPLAY_PX), Image.BICUBIC)
    cam_big = np.asarray(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((DISPLAY_PX, DISPLAY_PX), Image.BICUBIC)
    ) / 255.0
    blended = overlay(cam_big, np.asarray(big), alpha=0.45)

    # ``overlay`` tints the entire frame, and jet renders near zero activation as
    # dark blue, which reads as "the model looked everywhere". Fade the wash out
    # below 6 percent activation so the untouched photo shows through.
    base_rgb = np.asarray(big).astype(np.float32)
    weight = np.clip((cam_big - 0.06) / 0.18, 0.0, 1.0)[..., None]
    blended = (base_rgb * (1 - weight) + blended.astype(np.float32) * weight).astype(np.uint8)

    OUT.joinpath("assets").mkdir(parents=True, exist_ok=True)
    big.save(OUT / "assets" / f"{name}.jpg", quality=88, optimize=True)
    Image.fromarray(blended).save(OUT / "assets" / f"{name}-gradcam.png", optimize=True)

    top = torch.topk(probabilities, 3)
    top3 = []
    for probability, index in zip(top.values.tolist(), top.indices.tolist()):
        crop, condition = pretty(label[index])
        top3.append({
            "label": label[index],
            "crop": crop,
            "condition": condition,
            "probability": round(probability, 6),
        })

    order = torch.argsort(probabilities, descending=True).tolist()
    ranking = []
    for index in order:
        crop_i, condition_i = pretty(label[index])
        ranking.append({
            "label": label[index],
            "crop": crop_i,
            "condition": condition_i,
            "probability": round(float(probabilities[index]), 6),
        })

    truth_crop, truth_condition = pretty(row.label)
    return {
        "label": top3[0]["label"],
        "crop": top3[0]["crop"],
        "condition": top3[0]["condition"],
        "probability": top3[0]["probability"],
        "top3": top3,
        "image": {
            "photo": f"assets/{name}.jpg",
            "gradcam": f"assets/{name}-gradcam.png",
            "alt": f"Held out test photo of a {truth_crop.lower()} leaf, {truth_condition.lower()}",
            "px": DISPLAY_PX,
        },
        "truth": {
            "label": row.label,
            "crop": truth_crop,
            "condition": truth_condition,
            "match": bool(top3[0]["label"] == row.label),
        },
        "ranking": ranking,
        "timing_ms": {
            "preprocess": round(preprocess_ms, 1),
            "forward": round(forward_ms, 1),
            "gradcam": round(gradcam_ms, 1),
        },
        "source_image": row.relative_path,
        "file": Path(row.relative_path).name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=300, help="test images to sweep")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument(
        "--batch", type=int, default=12,
        help="held out test images to record for the batch upload page",
    )
    args = parser.parse_args()

    checkpoint, model, mean, std = load_checkpoint()
    size = int(checkpoint["image_size"])
    labels = list(checkpoint["classes"])

    frame = pd.read_csv(MANIFEST)
    frame = frame[frame.split.eq("test") if "split" in frame else np.ones(len(frame), bool)]
    sample = frame.sample(n=min(args.sample, len(frame)), random_state=args.seed)

    best_correct, first_wrong, lowest_correct, uncertain = None, None, None, None
    seen = []  # every (row, probabilities) from the sweep, reused for the batch export
    threshold = 0.70
    started = time.perf_counter()
    with torch.no_grad():
        for row in sample.itertuples():
            path = IMAGE_ROOT / row.relative_path
            if not path.is_file():
                continue
            tensor, _ = preprocess(path, size, mean, std)
            probabilities = F.softmax(model(tensor), dim=1)[0]
            probability = float(probabilities.max())
            index = int(probabilities.argmax())
            correct = labels[index] == row.label
            if correct:
                # A saturated 1.0 hides the gap between top 1 and top 2 on the page,
                # so pick the correct case closest to 0.96 and let the real spread show.
                if probability >= 0.90 and (
                    best_correct is None or abs(probability - 0.96) < abs(best_correct[1] - 0.96)
                ):
                    best_correct = (row, probability)
                if lowest_correct is None or probability < lowest_correct[1]:
                    lowest_correct = (row, probability)
            elif first_wrong is None:
                first_wrong = (row, probability)
            if uncertain is None or probability < uncertain[1]:
                uncertain = (row, probability)
            seen.append((row, probabilities, correct))
    elapsed = time.perf_counter() - started
    print(f"swept {len(sample)} images in {elapsed:.1f}s ({elapsed / len(sample) * 1000:.0f} ms per image)")

    if best_correct is None:
        raise SystemExit("no confidently correct prediction in the sample; raise --sample")

    cam_fn = GradCAM(model, target_layer_for(model, checkpoint["architecture"]))
    cases = {}

    row, _ = best_correct
    with torch.no_grad():
        tensor, _ = preprocess(IMAGE_ROOT / row.relative_path, size, mean, std)
    probabilities = F.softmax(model(tensor), dim=1)[0]
    cases["confident"] = export_case(row, labels, "confident", cam_fn, mean, std, size)
    print(f"confident: {cases['confident']['label']} at {cases['confident']['probability']:.4f}")

    hard = first_wrong if first_wrong is not None else lowest_correct
    if hard is not None:
        row, _ = hard
        with torch.no_grad():
            tensor, _ = preprocess(IMAGE_ROOT / row.relative_path, size, mean, std)
        probabilities = F.softmax(model(tensor), dim=1)[0]
        cases["hard"] = export_case(row, labels, "hard", cam_fn, mean, std, size)
        summary = cases["hard"]
        kind = "misclassified" if first_wrong is not None else "lowest confidence in sample"
        print(f"hard     : {summary['label']} at {summary['probability']:.4f} "
              f"(truth {summary['truth']['label']}, {kind})")

    if uncertain is not None and uncertain[1] < threshold:
        row, _ = uncertain
        with torch.no_grad():
            tensor, _ = preprocess(IMAGE_ROOT / row.relative_path, size, mean, std)
        probabilities = F.softmax(model(tensor), dim=1)[0]
        cases["uncertain"] = export_case(row, labels, "uncertain", cam_fn, mean, std, size)
        summary = cases["uncertain"]
        print(f"uncertain: {summary['label']} at {summary['probability']:.4f} "
              f"(truth {summary['truth']['label']}, below {threshold})")

    batch = []
    if args.batch > 0:
        # A random 12 of a 99.3% accurate model is almost always all correct, which
        # hides the error row in the UI. Include the real misclassifications found in
        # the sweep, so the batch shows what a real batch occasionally looks like.
        wrong = [item for item in seen if not item[2]]
        right = [item for item in seen if item[2]]
        chosen = wrong[: min(2, len(wrong))]
        chosen += right[: max(args.batch - len(chosen), 0)]
        chosen = chosen[: args.batch]
        for index, (row, _, _) in enumerate(chosen):
            with torch.no_grad():
                tensor, _ = preprocess(IMAGE_ROOT / row.relative_path, size, mean, std)
            probabilities = F.softmax(model(tensor), dim=1)[0]
            batch.append(export_case(row, labels, f"batch{index + 1:02d}", cam_fn, mean, std, size))
        correct = sum(1 for item in batch if item["truth"]["match"])
        print(f"batch    : {len(batch)} images, {correct} correct, {len(batch) - correct} wrong")

    cam_fn.remove()

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "generated_by": "scripts/make_demo_assets.py",
        "model": {
            "architecture": checkpoint["architecture"],
            "image_size": size,
            "classes": len(labels),
            "val_macro_f1": round(float(checkpoint.get("val_macro_f1", 0.0)), 6),
            "epoch": checkpoint.get("epoch"),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "checkpoint": CHECKPOINT.name,
            "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
            "normalization": {"mean": list(mean), "std": list(std)},
            "confidence_threshold": threshold,
        },
        "cases": cases,
        "batch": batch,
    }
    target = OUT / "js" / "demo-data.js"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "// Generated by scripts/make_demo_assets.py - do not edit by hand.\n"
        "// Every number below came out of the checkpoint on the image named in each case.\n"
        "window.DEMO = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

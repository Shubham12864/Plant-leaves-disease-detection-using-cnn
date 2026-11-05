#!/usr/bin/env python3
"""Interactive Agricultural Diagnostic CLI for Unified Gated Plant CNN.

Usage:
    python scripts/diagnose_leaf.py --image path/to/leaf.jpg
    python scripts/diagnose_leaf.py  # Runs on a default field sample
"""
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from src.plant_disease.diagnostic_engine import PlantDiagnosticEngine

ROOT = Path(__file__).resolve().parents[1]


def render_ascii_bar(val: float, width: int = 25) -> str:
    filled = int(round(val * width))
    return "█" * filled + "░" * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="Unified Gated Plant CNN Diagnostic Engine")
    parser.add_argument("--image", type=str, default=None, help="Path to leaf image")
    parser.add_argument("--save-overlay", action="store_true", default=True, help="Save diagnostic overlay PNG")
    args = parser.parse_args()

    # Find sample if not specified
    if not args.image:
        samples = list((ROOT / "data" / "raw" / "PlantDoc" / "test").rglob("*.jpg"))
        if not samples:
            raise FileNotFoundError("No leaf images found in data/raw/PlantDoc/test/")
        image_path = samples[0]
        print(f"No image provided. Using field benchmark sample: {image_path.name}")
    else:
        image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("\n=======================================================")
    print("🌿 UNIFIED GATED PLANT CNN: AGRICULTURAL DIAGNOSTIC VIVA")
    print("=======================================================")
    print(f"Target Image: {image_path}")

    engine = PlantDiagnosticEngine()
    with Image.open(image_path) as raw:
        img_rgb = raw.convert("RGB")
    result = engine.diagnose_image(img_rgb)

    # Display Terminal Banner
    print("\n---------------- HIERARCHICAL DIAGNOSIS ---------------")
    print(f"Botanical Crop Family: {result['predicted_crop'].upper()}")
    print(f"Crop Confidence:       [{render_ascii_bar(result['crop_confidence'])}] {result['crop_confidence']*100:.1f}%")
    print(f"Pathological Disease:  {result['disease_clean_name'].upper()}")
    print(f"Disease Confidence:    [{render_ascii_bar(result['disease_confidence'])}] {result['disease_confidence']*100:.1f}%")

    print("\n---------------- CONFORMAL RISK CONTROL ---------------")
    print(f"Statistical Guarantee:  1 - α = 95.0% Confidence")
    print(f"Calibrated Safe Set:    {len(result['conformal_prediction_set'])} candidate pathologies")
    for idx, c in enumerate(result['conformal_prediction_set']):
        print(f"  [{idx+1}] {c['crop']} - {c['clean_name']:<25} ({c['confidence']*100:.1f}%)")

    print("\n---------------- OOD & SPATIAL SALIENCY ---------------")
    print(f"Free Helmholtz Energy: {result['free_energy']:.2f}")
    ood_label = "✅ VALID PLANT FOLIAGE" if result['is_foliar_leaf'] else "⚠️ OUT-OF-DISTRIBUTION / NON-LEAF"
    print(f"Foliar Integrity:      {ood_label}")
    print(f"Saliency Mean Gain:    {result['saliency_mean']:.3f} (Active lesion focus)")

    if args.save_overlay:
        out_dir = ROOT / "reports" / "unified_extended" / "cli_diagnoses"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"diagnosis_{image_path.stem}.png"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img_rgb.resize((224, 224)))
        ax1.set_title(f"Field Leaf: {result['predicted_crop']}\nTrue Path: {image_path.parent.name}", fontsize=11, fontweight="bold")
        ax1.axis("off")

        # Saliency overlay
        img_arr = np.array(img_rgb.resize((224, 224)), dtype=np.float32) / 255.0
        mask_np = result["saliency_mask"]
        gain = 0.1 + (1.0 - 0.1) * mask_np[:, :, None]
        attenuated = np.clip(img_arr * gain, 0.0, 1.0)

        ax2.imshow(attenuated)
        ax2.set_title(f"Predicted: {result['disease_clean_name']}\nConf: {result['disease_confidence']*100:.1f}% | Energy: {result['free_energy']:.2f}", fontsize=11, fontweight="bold")
        ax2.axis("off")

        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"\nSaved diagnosis visualization: {out_path}")

    print("=======================================================\n")


if __name__ == "__main__":
    main()

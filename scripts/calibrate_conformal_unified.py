#!/usr/bin/env python3
"""Split Conformal Calibration & Statistical Safety for Unified Gated Plant CNN.

Applies Inductive Split Conformal Prediction (alpha = 0.05, 95% confidence guarantee)
and Free Helmholtz Energy OOD Detection to the Unified Gated Plant CNN.
"""
from pathlib import Path
import json
import random
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plant_disease.diagnostic_engine import compute_joint_probabilities
from src.plant_disease.models.unified_gated_cnn import build_unified_plant_cnn
from src.plant_disease.taxonomy import (
    CLASSES_38,
    CLASS_TO_IDX,
    CROPS_14,
    CLASS_TO_CROP_IDX,
    PLANTDOC_TO_PV_MAP,
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_pil_image,
)

CHECKPOINT_PATH = ROOT / "reports" / "unified_extended" / "unified_gated_plant_cnn_extended_best.pt"
OUTPUT_DIR = ROOT / "reports" / "unified_extended"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    model = build_unified_plant_cnn(num_crops=len(CROPS_14), num_diseases=len(CLASSES_38))
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    model = load_model()
    test_root = ROOT / "data" / "raw" / "PlantDoc" / "test"

    samples = []
    for cat_dir in test_root.iterdir():
        if not cat_dir.is_dir():
            continue
        mapped = PLANTDOC_TO_PV_MAP.get(cat_dir.name)
        if not mapped or mapped not in CLASS_TO_IDX:
            continue
        label_idx = CLASS_TO_IDX[mapped]
        crop_idx = CLASS_TO_CROP_IDX[label_idx]

        for img_p in list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.jpeg")):
            samples.append((str(img_p), label_idx, crop_idx))

    random.seed(42)
    random.shuffle(samples)
    print(f"Total Evaluated Test Samples: {len(samples)}")

    # Forward pass across test samples
    true_labels = []
    probs_list = []
    energies_list = []

    for img_p, label_idx, _ in samples:
        try:
            with Image.open(img_p) as raw:
                img_rgb = preprocess_pil_image(raw, target_size=224, center_crop=True)
            tensor = TF.to_tensor(img_rgb)
            tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0)

            with torch.no_grad():
                d_logits, c_logits = model(tensor, return_mask=False)
                d_probs = torch.softmax(d_logits, dim=-1)[0].numpy()
                c_probs = torch.softmax(c_logits, dim=-1)[0].numpy()
                probs = compute_joint_probabilities(d_probs, c_probs)
                energy = model.compute_energy(d_logits).item()

            true_labels.append(label_idx)
            probs_list.append(probs)
            energies_list.append(energy)
        except Exception:
            continue

    true_labels = np.array(true_labels)
    probs_mat = np.array(probs_list)
    energies = np.array(energies_list)

    n_total = len(true_labels)
    n_calib = n_total // 2
    calib_idx = np.arange(0, n_calib)
    eval_idx = np.arange(n_calib, n_total)

    # 1. Adaptive Prediction Sets (APS) Non-conformity Scores
    calib_aps_scores = []
    for i in calib_idx:
        p_vec = probs_mat[i]
        true_y = true_labels[i]
        sorted_idx = np.argsort(p_vec)[::-1]
        sorted_probs = p_vec[sorted_idx]
        cum_probs = np.cumsum(sorted_probs)
        rank = np.where(sorted_idx == true_y)[0][0]
        calib_aps_scores.append(cum_probs[rank])

    # 2. Conformal Quantile at alpha=0.05 (95% coverage)
    alpha = 0.05
    n = len(calib_aps_scores)
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    q_hat_aps = float(np.quantile(calib_aps_scores, q_level, method="higher"))
    print(f"Calibrated 95% APS Non-Conformity Quantile (q_hat): {q_hat_aps:.4f}")

    # 3. Evaluate APS prediction sets on evaluation set
    covered = 0
    set_sizes = []
    consistent_sets = 0

    for i in eval_idx:
        p_vec = probs_mat[i]
        sorted_idx = np.argsort(p_vec)[::-1]
        sorted_probs = p_vec[sorted_idx]
        cum_probs = np.cumsum(sorted_probs)

        # Cutoff: include top classes until cum_prob >= q_hat (clamped to valid range)
        cutoff = min(int(np.searchsorted(cum_probs, q_hat_aps)), len(sorted_idx) - 1)
        pred_set = sorted_idx[:cutoff + 1]
        set_sizes.append(len(pred_set))

        if true_labels[i] in pred_set:
            covered += 1

        # Check taxonomic consistency
        crops_in_set = set(CLASS_TO_CROP_IDX[k] for k in pred_set)
        if len(crops_in_set) <= 1:
            consistent_sets += 1

    emp_coverage = covered / len(eval_idx)
    avg_set_size = np.mean(set_sizes)
    singleton_rate = np.mean([1 if s == 1 else 0 for s in set_sizes])
    cons_rate = consistent_sets / len(eval_idx)

    print("\n================ CONFORMAL EVALUATION ================")
    print(f"Target Coverage:             {100*(1-alpha):.2f}%")
    print(f"Empirical Field Coverage:    {emp_coverage*100:.2f}%")
    print(f"Average Prediction Set Size: {avg_set_size:.2f} classes")
    print(f"Singleton (Exact 1-Class):   {singleton_rate*100:.2f}%")
    print(f"Taxonomic Consistency:       {cons_rate*100:.2f}%")
    print(f"Mean Free Energy (Field):    {np.mean(energies):.2f}")
    print("======================================================")

    # Save results to JSON
    conformal_summary = {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "empirical_field_coverage": float(emp_coverage),
        "q_hat_threshold": q_hat_aps,
        "average_set_size": float(avg_set_size),
        "singleton_rate": float(singleton_rate),
        "taxonomic_consistency": float(cons_rate),
        "mean_free_energy": float(np.mean(energies)),
        "std_free_energy": float(np.std(energies))
    }
    with open(OUTPUT_DIR / "conformal_calibration_results.json", "w") as f:
        json.dump(conformal_summary, f, indent=2)

    # Plot Set Size Distribution & Free Energy Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    counts, bins, _ = ax1.hist(set_sizes, bins=range(1, max(set_sizes) + 2), color="#3498db", edgecolor="black", align="left")
    ax1.axvline(avg_set_size, color="red", linestyle="--", label=f"Mean Size: {avg_set_size:.2f}")
    ax1.set_xlabel("Prediction Set Size |C(X)|", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Conformal Prediction Set Sizing (1 - α = 95%)", fontsize=12, fontweight="bold")
    ax1.legend()

    ax2.hist(energies, bins=25, color="#2ecc71", edgecolor="black")
    ax2.axvline(np.mean(energies), color="darkgreen", linestyle="--", label=f"Mean E(x): {np.mean(energies):.2f}")
    ax2.set_xlabel("Free Helmholtz Energy E(x)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Free Helmholtz Energy Distribution (Foliar Leaves)", fontsize=12, fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "conformal_coverage_plot.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"✅ Conformal calibration artifacts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

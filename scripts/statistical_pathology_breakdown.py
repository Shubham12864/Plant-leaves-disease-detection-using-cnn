#!/usr/bin/env python3
"""Fine-Grained Statistical Pathology & Crop Breakdown for Unified Gated Plant CNN.

Evaluates the Unified Gated Plant CNN on the field test benchmark:
- Top-1, Top-3, Top-5 Diagnostic Accuracy
- Crop Family Accuracy (14 botanical families)
- Macro & Micro Precision, Recall, and F1-score
- Generates high-resolution Confusion Matrix heatmap across all 14 crop families
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
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
            samples.append((str(img_p), label_idx, crop_idx, mapped, cat_dir.name))

    print(f"Total Evaluated Test Samples: {len(samples)}")

    y_true_d = []
    y_true_c = []
    y_pred_d = []
    y_pred_joint = []
    y_pred_c = []
    top3_correct = 0
    top5_correct = 0

    for img_p, d_idx, c_idx, _, _ in samples:
        try:
            with Image.open(img_p) as raw:
                img_rgb = preprocess_pil_image(raw, target_size=224, center_crop=True)
            tensor = TF.to_tensor(img_rgb)
            tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0)

            with torch.no_grad():
                d_logits, c_logits = model(tensor, return_mask=False)
                d_probs = torch.softmax(d_logits, dim=-1)[0].numpy()
                c_probs = torch.softmax(c_logits, dim=-1)[0].numpy()

            joint_probs = compute_joint_probabilities(d_probs, c_probs)
            pred_d = int(np.argmax(d_probs))
            pred_j = int(np.argmax(joint_probs))
            pred_c = int(np.argmax(c_probs))

            y_true_d.append(d_idx)
            y_true_c.append(c_idx)
            y_pred_d.append(pred_d)
            y_pred_joint.append(pred_j)
            y_pred_c.append(pred_c)

            top3 = np.argsort(joint_probs)[::-1][:3]
            top5 = np.argsort(joint_probs)[::-1][:5]
            if d_idx in top3:
                top3_correct += 1
            if d_idx in top5:
                top5_correct += 1
        except Exception:
            continue

    y_true_d = np.array(y_true_d)
    y_true_c = np.array(y_true_c)
    y_pred_d = np.array(y_pred_d)
    y_pred_joint = np.array(y_pred_joint)
    y_pred_c = np.array(y_pred_c)
    n = len(y_true_d)

    top1_acc = np.mean(y_true_d == y_pred_d)
    joint_top1_acc = np.mean(y_true_d == y_pred_joint)
    top3_acc = top3_correct / n
    top5_acc = top5_correct / n
    crop_acc = np.mean(y_true_c == y_pred_c)
    consistency = np.mean([1 if CLASS_TO_CROP_IDX[pd] == pc else 0 for pd, pc in zip(y_pred_d, y_pred_c)])

    print("\n================ STATISTICAL BENCHMARK ================")
    print(f"Top-1 Disease Accuracy:      {top1_acc*100:.2f}%")
    print(f"Top-3 Disease Accuracy:      {top3_acc*100:.2f}%")
    print(f"Top-5 Disease Accuracy:      {top5_acc*100:.2f}%")
    print(f"Crop Family Accuracy:        {crop_acc*100:.2f}%")
    print(f"Biological Consistency:      {consistency*100:.2f}%")
    print("========================================================")

    # Confusion Matrix across Crops (14x14)
    n_crops = len(CROPS_14)
    cm = np.zeros((n_crops, n_crops), dtype=int)
    for t, p in zip(y_true_c, y_pred_c):
        cm[t, p] += 1

    # Per-crop metrics
    crop_stats = []
    for c_i, c_name in enumerate(CROPS_14):
        tp = cm[c_i, c_i]
        fn = np.sum(cm[c_i, :]) - tp
        fp = np.sum(cm[:, c_i]) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        crop_stats.append({
            "crop_family": c_name,
            "total_samples": int(tp + fn),
            "true_positives": int(tp),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        })

    df_crops = pd.DataFrame(crop_stats)
    df_crops.to_csv(OUTPUT_DIR / "per_crop_metrics.csv", index=False)
    print(f"Saved per-crop metrics: {OUTPUT_DIR / 'per_crop_metrics.csv'}")

    # Plot Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Field Crop Family Confusion Matrix (14 Crops)", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(n_crops)
    short_crops = [c.replace("_", " ")[:10] for c in CROPS_14]
    plt.xticks(tick_marks, short_crops, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, short_crops, fontsize=9)
    plt.xlabel("Predicted Crop Family", fontsize=11, fontweight="bold")
    plt.ylabel("True Botanical Crop Family", fontsize=11, fontweight="bold")

    # Annotate numbers in heatmap
    thresh = cm.max() / 2.0
    for i in range(n_crops):
        for j in range(n_crops):
            if cm[i, j] > 0:
                plt.text(j, i, format(cm[i, j], "d"),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontsize=8)

    plt.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix_crops.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved crop confusion matrix: {cm_path}")

    # Save summary
    benchmark_summary = {
        "evaluated_samples": int(n),
        "top1_disease_accuracy": float(top1_acc),
        "joint_conditioned_top1_accuracy": float(joint_top1_acc),
        "top3_disease_accuracy": float(top3_acc),
        "top5_disease_accuracy": float(top5_acc),
        "crop_family_accuracy": float(crop_acc),
        "biological_consistency": float(consistency),
        "macro_crop_f1": float(df_crops["f1_score"].mean())
    }
    with open(OUTPUT_DIR / "statistical_benchmark_summary.json", "w") as f:
        json.dump(benchmark_summary, f, indent=2)
    print(f"✅ Statistical breakdown complete: {OUTPUT_DIR / 'statistical_benchmark_summary.json'}")


if __name__ == "__main__":
    main()

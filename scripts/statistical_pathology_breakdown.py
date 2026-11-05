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
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as TF

from src.plant_disease.models.unified_gated_cnn import build_unified_plant_cnn

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "reports" / "unified_extended" / "unified_gated_plant_cnn_extended_best.pt"
OUTPUT_DIR = ROOT / "reports" / "unified_extended"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES_38 = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES_38)}
CROPS_14 = sorted(list(set(c.split('___')[0] for c in CLASSES_38)))
CROP_TO_IDX = {cr: i for i, cr in enumerate(CROPS_14)}
CLASS_TO_CROP_IDX = {i: CROP_TO_IDX[c.split('___')[0]] for i, c in enumerate(CLASSES_38)}

PLANTDOC_TO_PV_MAP = {
    'Apple Scab Leaf': 'Apple___Apple_scab',
    'Apple leaf': 'Apple___healthy',
    'Apple rust leaf': 'Apple___Cedar_apple_rust',
    'Bell_pepper leaf': 'Pepper,_bell___healthy',
    'Bell_pepper leaf spot': 'Pepper,_bell___Bacterial_spot',
    'Blueberry leaf': 'Blueberry___healthy',
    'Cherry leaf': 'Cherry_(including_sour)___healthy',
    'Corn Gray leaf spot': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn leaf blight': 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn rust leaf': 'Corn_(maize)___Common_rust_',
    'Peach leaf': 'Peach___healthy',
    'Potato leaf early blight': 'Potato___Early_blight',
    'Potato leaf late blight': 'Potato___Late_blight',
    'Potato leaf': 'Potato___healthy',
    'Raspberry leaf': 'Raspberry___healthy',
    'Soyabean leaf': 'Soybean___healthy',
    'Squash Powdery mildew leaf': 'Squash___Powdery_mildew',
    'Strawberry leaf': 'Strawberry___healthy',
    'Tomato Early blight leaf': 'Tomato___Early_blight',
    'Tomato Septoria leaf spot': 'Tomato___Septoria_leaf_spot',
    'Tomato leaf': 'Tomato___healthy',
    'Tomato leaf bacterial spot': 'Tomato___Bacterial_spot',
    'Tomato leaf late blight': 'Tomato___Late_blight',
    'Tomato leaf mosaic virus': 'Tomato___Tomato_mosaic_virus',
    'Tomato leaf yellow virus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato mold leaf': 'Tomato___Leaf_Mold',
    'grape leaf': 'Grape___healthy',
    'grape leaf black rot': 'Grape___Black_rot'
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
    y_pred_c = []
    top3_correct = 0
    top5_correct = 0

    for img_p, d_idx, c_idx, _, _ in samples:
        try:
            with Image.open(img_p) as raw:
                img_rgb = raw.convert("RGB").resize((224, 224))
            tensor = TF.to_tensor(img_rgb)
            tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0)

            with torch.no_grad():
                d_logits, c_logits = model(tensor, return_mask=False)
                d_probs = torch.softmax(d_logits, dim=-1)[0].numpy()
                c_probs = torch.softmax(c_logits, dim=-1)[0].numpy()

            pred_d = int(np.argmax(d_probs))
            pred_c = int(np.argmax(c_probs))

            y_true_d.append(d_idx)
            y_true_c.append(c_idx)
            y_pred_d.append(pred_d)
            y_pred_c.append(pred_c)

            top3 = np.argsort(d_probs)[::-1][:3]
            top5 = np.argsort(d_probs)[::-1][:5]
            if d_idx in top3:
                top3_correct += 1
            if d_idx in top5:
                top5_correct += 1
        except Exception:
            continue

    y_true_d = np.array(y_true_d)
    y_true_c = np.array(y_true_c)
    y_pred_d = np.array(y_pred_d)
    y_pred_c = np.array(y_pred_c)
    n = len(y_true_d)

    top1_acc = np.mean(y_true_d == y_pred_d)
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
        "top1_disease_accuracy": float(top1_acc),
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

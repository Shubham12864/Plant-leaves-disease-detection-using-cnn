#!/usr/bin/env python3
"""Generate the Phase 1 Spatial Isolation & Diagnostic Benchmarking Notebook.

Generates: notebooks/11_Phase1_Spatial_Isolation.ipynb

Phase 1 isolates the pure spatial effect:
- Discovers both raw image directories and canonical manifests.
- Loads the trained PlantVillage checkpoints (EfficientNet-B0 and ConvNeXt-Tiny).
- Measures baseline whole-image accuracy on PlantDoc.
- Runs YOLO11 leaf ROI detection and context-padded cropping.
- Measures cropped leaf accuracy on the same models.
- Quantifies Delta_spatial to answer: 'How much of the field gap is background clutter?'
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "11_Phase1_Spatial_Isolation.ipynb"

cells = []

def add_md(source: str):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()]
    })

def add_code(source: str):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()]
    })

# 1. Header
add_md("""# Phase 1: Spatial Isolation & Diagnostic Benchmarking

### Research Question:
*How much of the 78.4 percentage point Field Gap is caused purely by background clutter (soil, weeds, hands, sky) vs. leaf appearance itself?*

### What this notebook executes:
1. **Whole-Image Baseline Benchmark:** Scores PlantVillage-trained models (EfficientNet-B0 and ConvNeXt-Tiny) on raw uncropped PlantDoc field images.
2. **YOLO11 Leaf Detection & Cropping:** Runs Ultralytics YOLO11 (`yolo11n.pt`) to detect foliage ROIs and crop leaves with 10% context padding.
3. **Cropped-Leaf Benchmark:** Evaluates the exact same models on the isolated leaf crops.
4. **Spatial Delta Calculation (\\(\\Delta_{\\text{spatial}}\\)):**
   $$\\Delta_{\\text{spatial}} = \\text{Accuracy}(\\text{YOLO11 Crop}) - \\text{Accuracy}(\\text{Whole Image})$$
5. **Generates Phase 1 Results Bundle:** Exports `phase1_comparison.csv`, `phase1_metrics.json`, and packaged zip.
""")

# 2. Setup & Environment Verification
add_code("""import os, sys, glob, json, time, math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
import torchvision.transforms as T
import torchvision.transforms.functional as TF

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Check Ultralytics YOLO availability
try:
    from ultralytics import YOLO
    print("✅ Ultralytics YOLO successfully imported!")
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics -q")
    from ultralytics import YOLO
    print("✅ Ultralytics installed and imported!")
""")

# 3. Discover Dataset & Model Paths
add_code("""# Scan Kaggle input directories
print("Input folders detected under /kaggle/input:")
for d in sorted(glob.glob("/kaggle/input/*")):
    print(" -", d)

def find_first(patterns):
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            return Path(matches[0])
    return None

def find_checkpoint(filename):
    patterns = [
        f"/kaggle/input/**/{filename}",
        f"kaggle input/checkpoints/{filename}",
        f"kaggle input/models/{filename}",
        f"checkpoints/{filename}",
    ]
    res = find_first(patterns)
    return str(res) if res else None

# Checkpoints
ckpt_b0 = find_checkpoint("efficientnet_b0_best.pt")
ckpt_convnext = find_checkpoint("convnext_tiny_best.pt")
print(f"Checkpoint EfficientNet-B0: {ckpt_b0}")
print(f"Checkpoint ConvNeXt-Tiny:   {ckpt_convnext}")

# YOLO weights
yolo_weight_path = find_checkpoint("yolo11n.pt") or "yolo11n.pt"
print(f"YOLO11 Weight Path:         {yolo_weight_path}")
""")

# 4. Class Taxonomy & Label Mapping
add_code("""# 38 PlantVillage Classes
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

# Canonical PlantDoc to PlantVillage mapping dictionary
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

def normalize_name(s: str) -> str:
    \"\"\"Standardizes string: lowercase, converts underscores and dashes to space, collapses whitespace.\"\"\"
    return " ".join(s.lower().replace("_", " ").replace("-", " ").split())

NORMALIZED_MAP = {normalize_name(k): v for k, v in PLANTDOC_TO_PV_MAP.items()}

def resolve_plantdoc_class(name: str):
    \"\"\"Multi-tier robust label resolver mapping folder or label names to PlantVillage classes.\"\"\"
    norm = normalize_name(name)
    # Tier 1: Direct normalized hit
    if norm in NORMALIZED_MAP:
        return NORMALIZED_MAP[norm]
    
    # Tier 2: Match after stripping 'leaf' / 'leaves'
    norm_no_leaf = " ".join([w for w in norm.split() if w not in ("leaf", "leaves")])
    for k, v in NORMALIZED_MAP.items():
        k_no_leaf = " ".join([w for w in k.split() if w not in ("leaf", "leaves")])
        if norm_no_leaf == k_no_leaf:
            return v
            
    # Tier 3: Multi-token overlap >= 2
    norm_tokens = set(norm_no_leaf.split())
    best_match, best_score = None, 0
    for k, v in NORMALIZED_MAP.items():
        k_tokens = set(w for w in k.split() if w not in ("leaf", "leaves"))
        overlap = len(norm_tokens.intersection(k_tokens))
        if overlap >= 2 and overlap > best_score:
            best_score = overlap
            best_match = v
            
    return best_match

print(f"Total PV target classes: {len(CLASSES_38)}")
print(f"Total mapped PlantDoc field classes: {len(PLANTDOC_TO_PV_MAP)}")
""")

# 5. Build Models (EfficientNet-B0 and ConvNeXt-Tiny)
add_code("""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint_weights(model, ckpt_path):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"  Note: Checkpoint {ckpt_path} not found. Running with ImageNet backbone.")
        return False
    try:
        data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
    state = data.get("model_state_dict", data) if isinstance(data, dict) else data
    if isinstance(state, dict):
        cleaned_state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
        print(f"  Successfully loaded trained checkpoint from: {ckpt_path}")
        if missing:
            print(f"    Missing keys: {len(missing)}")
        if unexpected:
            print(f"    Unexpected keys: {len(unexpected)}")
        return True
    return False

def load_b0():
    m = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 38)
    load_checkpoint_weights(m, ckpt_b0)
    return m.to(device).eval()

def load_convnext():
    m = tv_models.convnext_tiny(weights=tv_models.ConvNeXt_Tiny_Weights.DEFAULT)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 38)
    load_checkpoint_weights(m, ckpt_convnext)
    return m.to(device).eval()

models_dict = {
    "EfficientNet-B0": load_b0(),
    "ConvNeXt-Tiny": load_convnext(),
}
print("Models ready for benchmarking.")
""")

# 6. Load YOLO11 Leaf Cropper
add_code("""# Load YOLO11 Detector
print(f"Loading YOLO from: {yolo_weight_path}")
yolo_detector = YOLO(yolo_weight_path)
print("YOLO11 loaded successfully!")

def extract_leaf_crops(pil_img, pad_ratio=0.10, min_size=32):
    w, h = pil_img.size
    results = yolo_detector(pil_img, verbose=False)
    boxes = []
    for r in results:
        for b in r.boxes.xyxy.cpu().numpy():
            boxes.append(tuple(map(int, b)))
    
    crops = []
    if not boxes:
        # Fallback to multi-crop if no bbox detected
        boxes = [
            (0, 0, w, h),
            (int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))
        ]
    
    for (x1, y1, x2, y2) in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bw < min_size or bh < min_size:
            continue
        px, py = int(bw * pad_ratio), int(bh * pad_ratio)
        cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
        cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
        crops.append(pil_img.crop((cx1, cy1, cx2, cy2)))
    
    return crops if crops else [pil_img]
""")

# 7. Collect PlantDoc Test Images (Manifest + Recursive Fallback)
add_code("""test_samples = []
print("Searching for field test images inside PlantDoc...")

# Priority 1: Canonical manifest file if present
manifest_paths = glob.glob("/kaggle/input/**/plantdoc_test.csv", recursive=True)
manifest_used = False

if manifest_paths:
    m_path = manifest_paths[0]
    print(f"Found canonical test manifest: {m_path}")
    try:
        df_manifest = pd.read_csv(m_path)
        # Search base directory containing test images
        base_dirs = glob.glob("/kaggle/input/**/test", recursive=True) + glob.glob("/kaggle/input/**/PlantDoc", recursive=True)
        for _, row in df_manifest.iterrows():
            rel_p = str(row['relative_path']).strip()
            lbl = str(row['label']).strip()
            target = resolve_plantdoc_class(lbl)
            if not target or target not in CLASS_TO_IDX:
                continue
            
            # Find the image file
            found_p = None
            for b in base_dirs:
                cand = Path(b) / rel_p
                if cand.is_file():
                    found_p = str(cand)
                    break
                # Try stripped class folder
                cand2 = Path(b) / Path(rel_p).name
                if cand2.is_file():
                    found_p = str(cand2)
                    break
                    
            if found_p:
                test_samples.append({
                    "path": found_p,
                    "plantdoc_class": lbl,
                    "target_class": target,
                    "target_idx": CLASS_TO_IDX[target]
                })
        if test_samples:
            manifest_used = True
            print(f"Loaded {len(test_samples)} test samples from manifest.")
    except Exception as exc:
        print(f"Manifest loading notice: {exc}")

# Priority 2: Recursive folder walk if manifest was not used or yielded 0
if not manifest_used or len(test_samples) == 0:
    print("Performing comprehensive recursive directory scan...")
    search_dirs = [Path(d) for d in glob.glob("/kaggle/input/**/test", recursive=True) if os.path.isdir(d)]
    search_dirs += [Path(d) for d in glob.glob("/kaggle/input/**/plantdoc*", recursive=True) if os.path.isdir(d)]
    
    seen_paths = set()
    for s_dir in search_dirs:
        for img_p in s_dir.rglob("*.*"):
            if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            p_str = str(img_p.resolve())
            if p_str in seen_paths:
                continue
                
            parts_lower = [part.lower() for part in img_p.parts]
            # Must be in a test partition or under plantdoc test
            if "test" not in parts_lower:
                continue
                
            parent_folder = img_p.parent.name
            target = resolve_plantdoc_class(parent_folder)
            if target and target in CLASS_TO_IDX:
                seen_paths.add(p_str)
                test_samples.append({
                    "path": str(img_p),
                    "plantdoc_class": parent_folder,
                    "target_class": target,
                    "target_idx": CLASS_TO_IDX[target]
                })

print(f"Total scored field test images gathered: {len(test_samples)}")
df_samples = pd.DataFrame(test_samples)
if not df_samples.empty and 'target_class' in df_samples:
    print("Class counts in field test set:")
    print(df_samples['target_class'].value_counts())
else:
    raise RuntimeError("Zero field test samples found. Please verify dataset attachments.")
""")

# 8. Evaluation Engine (Whole Image vs YOLO11 Crop)
add_code("""IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def preprocess_pil(img_pil, size=224):
    resized = img_pil.resize((size, size), Image.BILINEAR)
    tensor = TF.to_tensor(resized)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std

def evaluate_condition(model, samples, use_crop=False):
    correct_list = []
    y_true_list = []
    y_pred_list = []
    conf_list = []

    with torch.no_grad():
        for s in samples:
            target_idx = s["target_idx"]
            y_true_list.append(target_idx)
            try:
                with Image.open(s["path"]) as orig:
                    img = orig.convert("RGB")
            except Exception as e:
                correct_list.append(0)
                y_pred_list.append(-1)
                conf_list.append(0.0)
                continue

            if not use_crop:
                # Whole image baseline
                inp = preprocess_pil(img).unsqueeze(0).to(device)
                logits = model(inp)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])
            else:
                # YOLO11 leaf crops with soft max-pooling
                crops = extract_leaf_crops(img)
                crop_tensors = torch.stack([preprocess_pil(c) for c in crops]).to(device)
                logits = model(crop_tensors)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                # Aggregate across crops via mean probabilities
                mean_probs = np.mean(probs, axis=0)
                pred_idx = int(np.argmax(mean_probs))
                conf = float(mean_probs[pred_idx])

            is_correct = int(pred_idx == target_idx)
            correct_list.append(is_correct)
            y_pred_list.append(pred_idx)
            conf_list.append(conf)

    acc = np.mean(correct_list)
    return {
        "accuracy": float(acc),
        "mean_confidence": float(np.mean(conf_list)),
        "y_true": np.array(y_true_list),
        "y_pred": np.array(y_pred_list),
        "correct": np.array(correct_list)
    }
""")

# 9. Run Phase 1 Comparative Benchmark
add_code("""print("==================================================")
print("RUNNING PHASE 1 SPATIAL ISOLATION BENCHMARK...")
print("==================================================")

results = {}

for model_name, model in models_dict.items():
    print(f"\\n--- Evaluating {model_name} on Whole-Image Baseline ---")
    t0 = time.time()
    res_whole = evaluate_condition(model, test_samples, use_crop=False)
    print(f"Whole Image Accuracy: {res_whole['accuracy']*100:.2f}% (Time: {time.time()-t0:.1f}s)")

    print(f"--- Evaluating {model_name} on YOLO11 Leaf Crops ---")
    t0 = time.time()
    res_crop = evaluate_condition(model, test_samples, use_crop=True)
    print(f"YOLO11 Cropped Accuracy: {res_crop['accuracy']*100:.2f}% (Time: {time.time()-t0:.1f}s)")

    delta = res_crop['accuracy'] - res_whole['accuracy']
    print(f"Spatial Delta (Delta_spatial): {delta*100:+.2f} percentage points!")

    results[model_name] = {
        "whole_acc": res_whole['accuracy'],
        "crop_acc": res_crop['accuracy'],
        "delta_pp": delta * 100.0,
        "whole_conf": res_whole['mean_confidence'],
        "crop_conf": res_crop['mean_confidence']
    }
""")

# 10. Generate Comparison Table & Package Output
add_code("""out_dir = Path("/kaggle/working/phase1_results")
out_dir.mkdir(parents=True, exist_ok=True)

df_comparison = pd.DataFrame([
    {
        "Model": k,
        "Whole_Image_Acc (%)": f"{v['whole_acc']*100:.2f}%",
        "YOLO11_Cropped_Acc (%)": f"{v['crop_acc']*100:.2f}%",
        "Spatial_Delta (pp)": f"{v['delta_pp']:+.2f} pp",
        "Whole_Confidence": f"{v['whole_conf']:.3f}",
        "Crop_Confidence": f"{v['crop_conf']:.3f}"
    }
    for k, v in results.items()
])

print("\\n" + "="*60)
print("PHASE 1 SPATIAL ABLATION BENCHMARK RESULTS")
print("="*60)
print(df_comparison.to_string(index=False))

df_comparison.to_csv(out_dir / "phase1_comparison.csv", index=False)
with open(out_dir / "phase1_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

import shutil
shutil.make_archive("/kaggle/working/phase1_results", "zip", "/kaggle/working/phase1_results")
print("\\n✅ Phase 1 Benchmark Complete! Saved to /kaggle/working/phase1_results.zip")
""")

notebook_json = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2)

print(f"Generated Phase 1 Notebook: {NOTEBOOK_PATH}")

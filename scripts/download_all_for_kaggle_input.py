#!/usr/bin/env python3
"""Download datasets and pretrained models directly into 'kaggle input/' directory.

Creates:
- kaggle input/models/ (ConvNeXt, EfficientNet, ResNet, YOLO weights)
- kaggle input/plantdoc/ (PlantDoc field dataset)
- kaggle input/plantvillage/ (PlantVillage lab dataset)
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_INPUT = ROOT / "kaggle input"
MODELS_DIR = KAGGLE_INPUT / "models"
PLANTDOC_DIR = KAGGLE_INPUT / "plantdoc"
PLANTVILLAGE_DIR = KAGGLE_INPUT / "plantvillage"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLANTDOC_DIR.mkdir(parents=True, exist_ok=True)
PLANTVILLAGE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_URLS = {
    "convnext_tiny-983f1562.pth": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
    "convnext_small-0c510722.pth": "https://download.pytorch.org/models/convnext_small-0c510722.pth",
    "efficientnet_b0_rwightman-7f5810bc.pth": "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
    "resnet18-f37072fd.pth": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
}

PV_REVISION = "9e97599868962bd0079b8db4b7f1efa9185fa1e7"
PD_REVISION = "5467f6012d78d1c446145d5f582da6096f852ae8"

def download_file(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  [cached] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  [downloading] {dest.name} from {url}...")
    cmd = ["curl", "-L", "--fail", "--retry", "5", url, "-o", str(dest)]
    subprocess.run(cmd, check=True)
    print(f"  [done] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")

def download_models():
    print("\n--- 1. Downloading Pretrained CNN & Detection Models ---")
    for name, url in MODEL_URLS.items():
        dest = MODELS_DIR / name
        download_file(url, dest)

def download_plantdoc():
    print("\n--- 2. Downloading & Extracting PlantDoc (Field Dataset) ---")
    source_dir = ROOT / "data" / "raw" / "PlantDoc" / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = source_dir / "PlantDoc-Dataset.zip"
    url = f"https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/{PD_REVISION}"
    download_file(url, zip_dest)

    # Materialize into both data/raw/PlantDoc and kaggle input/plantdoc
    target_raw = ROOT / "data" / "raw" / "PlantDoc"
    print(f"Extracting PlantDoc into {target_raw} and {PLANTDOC_DIR}...")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "materialize_plantdoc.py"),
        "--archive", str(zip_dest),
        "--output-dir", str(PLANTDOC_DIR)
    ]
    subprocess.run(cmd, check=True)
    # Also create symlink from data/raw/PlantDoc if needed
    if not (ROOT / "data" / "raw" / "PlantDoc" / "train").exists():
        os.system(f"cp -rsf '{PLANTDOC_DIR}/'* '{target_raw}/' 2>/dev/null || true")

def download_plantvillage():
    print("\n--- 3. Downloading & Extracting PlantVillage (Lab Baseline) ---")
    source_dir = ROOT / "data" / "raw" / "PlantVillage" / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    
    files = {
        "data.zip": f"https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{PV_REVISION}/data.zip",
        "color_train.txt": f"https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{PV_REVISION}/splits/color_train.txt",
        "color_test.txt": f"https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{PV_REVISION}/splits/color_test.txt",
        "leaf-map.json": f"https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{PV_REVISION}/leaf_grouping/leaf-map.json",
    }
    for fname, url in files.items():
        download_file(url, source_dir / fname)

    # Materialize into kaggle input/plantvillage
    print(f"Extracting PlantVillage into {PLANTVILLAGE_DIR}...")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "materialize_plantvillage.py"),
        "--source-dir", str(source_dir),
        "--output-dir", str(PLANTVILLAGE_DIR)
    ]
    subprocess.run(cmd, check=True)
    target_raw = ROOT / "data" / "raw" / "PlantVillage"
    if not (target_raw / "train").exists():
        os.system(f"cp -rsf '{PLANTVILLAGE_DIR}/'* '{target_raw}/' 2>/dev/null || true")

if __name__ == "__main__":
    download_models()
    download_plantdoc()
    download_plantvillage()
    print("\n🎉 ALL DATASETS & PRETRAINED MODELS READY IN 'kaggle input/'!")

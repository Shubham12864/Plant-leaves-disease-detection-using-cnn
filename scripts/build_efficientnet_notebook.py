"""Build the single-model Kaggle notebook from the maintained training script."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_Train_EfficientNet_B0.ipynb"
SCRIPT = ROOT / "scripts" / "train_efficientnet_b0_ddp.py"
RESULTS = ROOT / "reports" / "model_comparison" / "previous" / "validation_comparison.csv"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def main() -> None:
    import pandas as pd

    previous = pd.read_csv(RESULTS).to_dict(orient="records")
    # Embed JSON as a string and parse it at runtime. A raw JSON object uses
    # lowercase true/false/null, which are not valid Python literals.
    baseline_json = json.dumps(previous, indent=2, ensure_ascii=False)
    script_text = SCRIPT.read_text(encoding="utf-8")
    cells = [
        md("""# Plant leaves disease detection using CNN — EfficientNet-B0\n
This notebook trains **only EfficientNet-B0** on the PlantVillage development split. It uses two T4 GPUs through DistributedDataParallel (DDP), ImageNet-pretrained weights, mixed precision, a one-time disk-backed image cache, and a live batch progress bar.\n
The notebook reuses the saved SmallCNN and ResNet18 validation results. It does not train those models again. It skips MobileNet. It does not open the locked PlantVillage test manifest or use PlantDoc for training.\n
**Run the cells from top to bottom.** The training cell can resume from its latest complete epoch if Kaggle stops the session. Save or download the results ZIP before the Kaggle session ends."""),
        md("""## 1. Attach input and check the run
In Kaggle, attach the prepared **PlantVillage** dataset, select **GPU T4 x2**, and enable Internet so PyTorch can download the ImageNet EfficientNet-B0 weights if Kaggle has not cached them. This cell checks the expected input paths and the two GPUs. It does not run a training smoke test."""),
        code('''from pathlib import Path
import json, os, shutil, hashlib, time, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
import torch

SEED = 42
IMAGE_SIZE = 224
DATASET_ROOT = Path("/kaggle/input/datasets/shubhamkumar12864/plantvillage")
MANIFEST_DIR = DATASET_ROOT / "data" / "manifests"
IMAGE_ROOT = DATASET_ROOT / "data" / "raw" / "PlantVillage"
TRAIN_MANIFEST = MANIFEST_DIR / "plantvillage_development.csv"
CACHE_DIR = Path("/kaggle/temp/plantvillage_effnet_cache")
OUTPUT_ROOT = Path("/kaggle/working/model_comparison")
MODEL_DIR = OUTPUT_ROOT / "efficientnet_b0"
SCRIPT_PATH = Path("/kaggle/working/train_efficientnet_b0_ddp.py")

if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("Select GPU T4 x2 in Kaggle Settings, then rerun this cell.")
gpu_names = [torch.cuda.get_device_name(i) for i in range(2)]
if not all("T4" in name for name in gpu_names):
    print("Warning: this notebook is tuned for 2× T4; detected:", gpu_names)
if not TRAIN_MANIFEST.is_file() or not IMAGE_ROOT.is_dir():
    raise FileNotFoundError(f"PlantVillage input is missing. Expected {TRAIN_MANIFEST} and {IMAGE_ROOT}")

# This notebook intentionally reads only the development manifest.
dev = pd.read_csv(TRAIN_MANIFEST)
required = {"split", "relative_path", "label"}
if not required.issubset(dev.columns):
    raise ValueError(f"Development manifest needs columns: {sorted(required)}")
train_df = dev.loc[dev.split == "train"].reset_index(drop=True)
val_df = dev.loc[dev.split == "validation"].reset_index(drop=True)
classes = sorted(train_df.label.unique().tolist())
if len(classes) != 38 or set(val_df.label) != set(classes):
    raise ValueError(f"Unexpected class split: {len(classes)} train classes, {val_df.label.nunique()} validation classes")
if set(train_df.relative_path) & set(val_df.relative_path):
    raise ValueError("Train and validation contain shared image paths.")

def paths_for(frame):
    return [IMAGE_ROOT / "train" / rel for rel in frame.relative_path]

train_paths, val_paths = paths_for(train_df), paths_for(val_df)
missing = [str(p) for p in train_paths + val_paths if not p.is_file()]
if missing:
    raise FileNotFoundError(f"{len(missing)} manifest images are missing; first: {missing[0]}")
required_cache_bytes = (len(train_df) + len(val_df)) * IMAGE_SIZE * IMAGE_SIZE * 3
CACHE_DIR.mkdir(parents=True, exist_ok=True)
free_cache = shutil.disk_usage(CACHE_DIR).free
if free_cache < required_cache_bytes * 1.10:
    raise OSError(f"Image cache needs about {required_cache_bytes/1e9:.1f} GB; only {free_cache/1e9:.1f} GB is free in {CACHE_DIR}")

print(f"GPUs: {gpu_names}")
print(f"PlantVillage: {len(train_df):,} train, {len(val_df):,} validation, {len(classes)} classes")
print(f"Image size: {IMAGE_SIZE}×{IMAGE_SIZE} | DDP global batch: 32 (16 per GPU)")
print(f"Cache space: {required_cache_bytes/1e9:.1f} GB required; {free_cache/1e9:.1f} GB available")
print("Comparison baselines: SmallCNN and ResNet18 results will be reused; neither will be retrained.")
print("Locked PlantVillage test manifest: not opened.")
'''),
        md("""## 2. Build or reuse the image cache
This cell decodes each training and validation image once, resizes it to 224×224, and saves RGB pixels as uint8 arrays in `/kaggle/temp`. EfficientNet training reads from these memory-mapped arrays. The cache keeps decoded images out of RAM and avoids DataLoader worker processes, which caused earlier worker and kernel failures. If this cell stops during cache creation, rerun it; incomplete caches are rebuilt."""),
        code('''def cache_split(frame, paths, tag):
    labels = np.asarray([classes.index(label) for label in frame.label], dtype=np.int64)
    path_key = hashlib.sha256("\\n".join(map(str, paths)).encode()).hexdigest()[:16]
    stem = CACHE_DIR / f"{tag}_{IMAGE_SIZE}_{len(paths)}_{path_key}"
    x_path, y_path, done_path = Path(f"{stem}_x.npy"), Path(f"{stem}_y.npy"), Path(f"{stem}.done")
    shape = (len(paths), IMAGE_SIZE, IMAGE_SIZE, 3)
    reusable = False
    if done_path.is_file() and x_path.is_file() and y_path.is_file():
        try:
            x_existing = np.load(x_path, mmap_mode="r", allow_pickle=False)
            y_existing = np.load(y_path, allow_pickle=False)
            reusable = x_existing.shape == shape and x_existing.dtype == np.uint8 and np.array_equal(y_existing, labels)
            del x_existing, y_existing
        except Exception:
            reusable = False
    if not reusable:
        done_path.unlink(missing_ok=True)
        x_path.unlink(missing_ok=True)
        y_path.unlink(missing_ok=True)
        array = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.uint8, shape=shape)
        started = time.perf_counter()
        def decode(item):
            i, path = item
            with Image.open(path) as image:
                array[i] = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR), dtype=np.uint8)
            return i + 1
        with ThreadPoolExecutor(max_workers=4) as pool:
            progress = tqdm(pool.map(decode, enumerate(paths)), total=len(paths), desc=f"Cache {tag}", unit="img")
            for count in progress:
                if count % 1000 == 0 or count == len(paths):
                    progress.set_postfix(img_s=f"{count/max(time.perf_counter()-started,1e-6):.0f}")
        array.flush()
        del array
        np.save(y_path, labels, allow_pickle=False)
        done_path.write_text("complete\\n", encoding="utf-8")
        print(f"Wrote {{tag}} cache: {{x_path}}")
    else:
        print(f"Reused verified {{tag}} cache: {{x_path}}")
    return x_path, y_path

TRAIN_X_PATH, TRAIN_Y_PATH = cache_split(train_df, train_paths, "train")
VAL_X_PATH, VAL_Y_PATH = cache_split(val_df, val_paths, "validation")
CLASSES_PATH = CACHE_DIR / "classes.json"
CLASSES_PATH.write_text(json.dumps(classes, indent=2), encoding="utf-8")
print("Image cache is ready.")
'''),
        md("""## 3. Set up the training program
The training program uses DistributedDataParallel (one process per T4), a global batch of 32, local batch 16 per GPU, FP16 autocast, AdamW, ImageNet normalization and GPU-side flip/rotation/brightness augmentation. It saves the best validation checkpoint and a full resume checkpoint after every completed epoch. It reports live batches, current/average loss, image throughput, and per-epoch validation accuracy and macro-F1.\n
The existing ResNet18 and SmallCNN are comparison results only. Their checkpoint files are stored in this project, but this training run does not need to load or modify them."""),
        code("%%writefile /kaggle/working/train_efficientnet_b0_ddp.py\n" + script_text),
        md("""## 4. Train EfficientNet-B0 on both T4 GPUs
Run this cell once after the cache and training-program cells finish. It launches one two-process DDP run. If Kaggle ends the session, start a new session, rerun setup and cache cells, and rerun this training cell; it continues from the last completed epoch in `/kaggle/working` only if Kaggle preserves that working directory. Download the output ZIP after training."""),
        code('''OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
    "--nproc_per_node=2", str(SCRIPT_PATH),
    "--train-images", str(TRAIN_X_PATH), "--train-labels", str(TRAIN_Y_PATH),
    "--val-images", str(VAL_X_PATH), "--val-labels", str(VAL_Y_PATH),
    "--classes-json", str(CLASSES_PATH), "--output-dir", str(MODEL_DIR),
]
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
print("Starting EfficientNet-B0 DDP with 2 GPUs. No other architecture will train.", flush=True)
result = subprocess.run(command, env=env, check=False)
if result.returncode != 0:
    raise RuntimeError(f"EfficientNet training process exited with code {result.returncode}. Check the last log lines above.")
print("EfficientNet-B0 training completed.", flush=True)
'''),
        md("""## 5. Compare results and create the download bundle
The table uses the saved validation results from the prior ZIP for SmallCNN and ResNet18, plus the new EfficientNet-B0 validation result. It does not use PlantVillage test or PlantDoc scores to select a model. The output includes the EfficientNet best checkpoint, resumable checkpoint, metrics, epoch history, and comparison files."""),
        code(f'''# These are the two completed models from the supplied comparison ZIP.
previous_results = json.loads({baseline_json!r})
effnet_metrics_path = MODEL_DIR / "validation_metrics.json"
if not effnet_metrics_path.is_file():
    raise FileNotFoundError(f"Training metrics were not created: {{effnet_metrics_path}}")
effnet_result = json.loads(effnet_metrics_path.read_text(encoding="utf-8"))
comparison = pd.DataFrame(previous_results + [effnet_result]).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
comparison_path = OUTPUT_ROOT / "validation_comparison.csv"
comparison.to_csv(comparison_path, index=False)
display(comparison[["model", "val_accuracy", "val_balanced_accuracy", "val_macro_f1", "best_epoch", "parameters", "checkpoint_mb", "train_minutes"]])

history = pd.read_csv(MODEL_DIR / "training_history.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history["epoch"], history["train_loss"], marker="o", label="EfficientNet-B0 train")
ax[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
ax[1].plot(history["epoch"], history["val_macro_f1"], marker="o", label="EfficientNet-B0 validation")
ax[1].set(title="Validation macro-F1", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1))
for axis in ax: axis.legend()
fig.tight_layout()
fig.savefig(OUTPUT_ROOT / "efficientnet_b0_training_curves.png", dpi=160)
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(comparison["model"], comparison["val_macro_f1"], color=["#377eb8" if x == "efficientnet_b0" else "#70ad47" for x in comparison["model"]])
ax.set_ylim(0, 1)
ax.set_ylabel("Validation macro-F1")
ax.set_title("PlantVillage validation comparison (locked test not used)")
fig.tight_layout()
fig.savefig(OUTPUT_ROOT / "validation_model_comparison.png", dpi=160)
plt.show()
plt.close(fig)

metadata = {{
    "dataset": "PlantVillage", "train_images": len(train_df), "validation_images": len(val_df),
    "classes": classes, "locked_test_manifest_opened": False, "plantdoc_used_for_training": False,
    "new_training_model": "efficientnet_b0", "skipped_model": "mobilenet_v3_large",
    "reused_validation_baselines": ["smallcnn", "resnet18"], "gpu_names": gpu_names,
    "training_mode": "torchrun DistributedDataParallel", "global_batch_size": 32,
    "per_gpu_batch_size": 16, "image_size": IMAGE_SIZE, "seed": SEED,
}}
(OUTPUT_ROOT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
zip_path = shutil.make_archive("/kaggle/working/model_comparison_artifacts", "zip", root_dir=OUTPUT_ROOT)
print(f"Download this file from the Kaggle Output panel: {{zip_path}}")
print(f"Best EfficientNet checkpoint: {{MODEL_DIR / 'efficientnet_b0_best.pt'}}")
'''),
        md("""## What to do next
1. Download `/kaggle/working/model_comparison_artifacts.zip` from the Output panel.\n2. Share the ZIP here so we can verify the EffNet training curve and comparison.\n3. Select the final model using validation macro-F1, balanced accuracy, size and speed.\n4. Then run the final evaluation notebook once on the locked PlantVillage test split."""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {NOTEBOOK}")


if __name__ == "__main__":
    main()

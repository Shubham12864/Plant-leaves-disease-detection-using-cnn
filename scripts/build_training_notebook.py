#!/usr/bin/env python3
"""Generate the Kaggle training notebook from the canonical training script.

The pilot notebook embedded its training program as a literal string, and the
separate ``build_efficientnet_notebook.py`` that was supposed to regenerate it
read a CSV that no longer exists. That is why the audit flagged the notebook and
``scripts/`` as a possible source of drift.

This builder removes the duplication: ``scripts/train_ddp.py`` is the single
source of truth, and the notebook simply writes that exact text to disk inside
Kaggle before launching it. Regenerate after editing the training script with:

    python scripts/build_training_notebook.py

and re-commit the notebook.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_ddp.py"
NOTEBOOK = ROOT / "notebooks" / "01_Train_EfficientNet_B0_v2.ipynb"

MARKDOWN = "markdown"
CODE = "code"


def md(text: str) -> dict:
    return {"cell_type": MARKDOWN, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": CODE,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def writefile_cell(script_text: str, target: str) -> dict:
    """A ``%%writefile`` cell that reproduces the canonical script verbatim.

    Built by hand instead of through ``code()`` because ``code()`` trims outer
    newlines, which would make the embedded copy differ from the source by a
    trailing newline and defeat the byte-for-byte check below.
    """
    lines = [f"%%writefile {target}\n"]
    body_lines = script_text.splitlines(keepends=True)
    if body_lines and not body_lines[-1].endswith("\n"):
        body_lines[-1] += "\n"
    lines.extend(body_lines)
    return {"cell_type": CODE, "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


HEADER = """# Training: EfficientNet-B0 on PlantVillage (v2, position-fixed)

This notebook trains **one** architecture at a time on the PlantVillage development split, in Kaggle, on two T4 GPUs.

It replaces `01_Train_EfficientNet_B0.ipynb`. That notebook worked but was I/O bound: it read a 5.5 GB memory-mapped cache with `num_workers=0`, so the GPUs idled while the main process page-faulted. Per-epoch time grew from 1.7 to 15.65 minutes as the page cache evicted. The measured throughput was about 450 ms per global batch of 32.

v2 changes the data path, not the science. Same split, same 224x224 inputs, same ImageNet normalisation, same validation-based model selection.

| Change | Pilot | v2 | Why |
|---|---|---|---|
| DataLoader workers | 0 | 4 per rank, prefetch 4 | overlap I/O with compute; this is the ~5-8x fix |
| Per-GPU batch | 16 | 64 | fewer, larger steps; AdamW tolerates it at this LR |
| cuDNN benchmark | off | on | fixed input shape, free convolution autotune |
| Loss | plain CE | class-balanced CE, label smoothing 0.05 | training set is 38.4x imbalanced; macro-F1 is the selection metric |
| LR schedule | constant | 10% warmup then cosine | standard fine-tuning schedule |
| Grad clip | none | 5.0 | stabilises FP16 fine-tuning |
| Weight averaging | none | EMA 0.999 | usually worth about 0.5-1 point; kept only if it wins on validation |
| Validation output | headline numbers | + per-class table, confusion matrix, ECE, Brier, top-3 | the report and the guide review need these |

Everything here targets the **development** manifest only. The locked test manifest is never opened and PlantDoc is never used for training.

**Run top to bottom.** The training cell resumes from the last completed epoch if Kaggle restarts the session and `/kaggle/working` survives. Download the output ZIP before the session ends.
"""

SETUP_MD = """## 1. Setup and preflight

Attach the prepared **PlantVillage** dataset, set **Accelerator → GPU T4 x2**, and enable Internet once so torchvision can fetch the ImageNet weights.

The tunables live in the last cell of this section. Change them there, not inside the training program.
"""

SETUP_CODE = '''from pathlib import Path
import json, os, shutil, hashlib, time, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
import torch, torchvision

SEED = 42
IMAGE_SIZE = 224

# ---- inputs (edit if your Kaggle dataset slug differs) ----------------------
DATASET_ROOT = Path("/kaggle/input/datasets/shubhamkumar12864/plantvillage")
MANIFEST_DIR = DATASET_ROOT / "data" / "manifests"
IMAGE_ROOT = DATASET_ROOT / "data" / "raw" / "PlantVillage"
TRAIN_MANIFEST = MANIFEST_DIR / "plantvillage_development.csv"

# ---- working paths ----------------------------------------------------------
CACHE_DIR = Path("/kaggle/temp/plantvillage_effnet_cache")
OUTPUT_ROOT = Path("/kaggle/working/model_comparison")
MODEL_DIR = OUTPUT_ROOT / "efficientnet_b0"
SCRIPT_PATH = Path("/kaggle/working/train_ddp.py")

# ---- training tunables (the only place to change them) ----------------------
ARCH = "efficientnet_b0"
EPOCHS = 12
PATIENCE = 3
LOCAL_BATCH_SIZE = 64      # per GPU; global = 64 x 2 = 128
WORKERS = 4                # DataLoader workers per rank
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
CLASS_WEIGHT_ALPHA = 0.5   # 0 disables class-balanced loss
EMA_DECAY = 0.999          # 0 disables EMA
ROTATION_DEG = 12.0
USE_TORCH_COMPILE = False  # try True on a fresh session; adds warm-up variance
SHUFFLE_MODE = "blocked"   # keeps page-cache reads local

# ---- preflight --------------------------------------------------------------
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("Select Accelerator -> GPU T4 x2 in Kaggle Settings, then rerun this cell.")
gpu_names = [torch.cuda.get_device_name(i) for i in range(2)]
if not all("T4" in name for name in gpu_names):
    print("Warning: tuned for 2x T4; detected:", gpu_names)
if not TRAIN_MANIFEST.is_file() or not IMAGE_ROOT.is_dir():
    raise FileNotFoundError(f"PlantVillage input missing. Expected {TRAIN_MANIFEST} and {IMAGE_ROOT}")

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
    raise ValueError("Train and validation share image paths.")

def paths_for(frame):
    return [IMAGE_ROOT / "train" / rel for rel in frame.relative_path]

train_paths, val_paths = paths_for(train_df), paths_for(val_df)
missing = [str(p) for p in train_paths + val_paths if not p.is_file()]
if missing:
    raise FileNotFoundError(f"{len(missing)} manifest images are missing; first: {missing[0]}")

expected_bytes = (len(train_df) + len(val_df)) * IMAGE_SIZE * IMAGE_SIZE * 3
CACHE_DIR.mkdir(parents=True, exist_ok=True)
free_cache = shutil.disk_usage(CACHE_DIR).free
if free_cache < expected_bytes * 1.10:
    raise OSError(f"Cache needs about {expected_bytes/1e9:.1f} GB; {free_cache/1e9:.1f} GB free in {CACHE_DIR}")

print(f"torch {torch.__version__} | torchvision {torchvision.__version__}")
print(f"GPUs: {gpu_names}")
print(f"PlantVillage: {len(train_df):,} train, {len(val_df):,} validation, {len(classes)} classes")
print(f"Global batch {LOCAL_BATCH_SIZE * 2} ({LOCAL_BATCH_SIZE}/GPU), {WORKERS} workers/rank, lr={LEARNING_RATE}")
print(f"Cache: {expected_bytes/1e9:.1f} GB required, {free_cache/1e9:.1f} GB free")
print("Locked test manifest: not opened. PlantDoc: not used.")
'''

CACHE_MD = """## 2. Build or reuse the decoded image cache

Decode each image once into a uint8 `224x224x3` memory map, then let the DataLoader workers stream it. Decoding inside workers on every epoch is slower and makes epoch time depend on JPEG complexity.

The cache is keyed by a hash of the image path list, so changing the split invalidates it automatically. Re-running this cell reuses a verified cache instead of decoding 43k images again.

`/kaggle/temp` is faster than `/kaggle/working` and is wiped when the session ends. That is the right place for a disposable cache.
"""

CACHE_CODE = '''def cache_split(frame, paths, tag):
    labels = np.asarray([classes.index(label) for label in frame.label], dtype=np.int64)
    path_key = hashlib.sha256("\\n".join(map(str, paths)).encode()).hexdigest()[:16]
    stem = CACHE_DIR / f"{tag}_{IMAGE_SIZE}_{len(paths)}_{path_key}"
    x_path, y_path, done_path = Path(f"{stem}_x.npy"), Path(f"{stem}_y.npy"), Path(f"{stem}.done")
    shape = (len(paths), IMAGE_SIZE, IMAGE_SIZE, 3)

    reusable = False
    if done_path.is_file() and x_path.is_file() and y_path.is_file():
        try:
            existing = np.load(x_path, mmap_mode="r", allow_pickle=False)
            reusable = existing.shape == shape and existing.dtype == np.uint8 and np.array_equal(
                np.load(y_path, allow_pickle=False), labels
            )
            del existing
        except Exception:
            reusable = False

    if reusable:
        print(f"Reused verified {tag} cache: {x_path}")
        return x_path, y_path

    for stale in (done_path, x_path, y_path):
        stale.unlink(missing_ok=True)
    array = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.uint8, shape=shape)
    started = time.perf_counter()

    def decode(item):
        i, path = item
        with Image.open(path) as image:
            array[i] = np.asarray(
                image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR), dtype=np.uint8
            )
        return i + 1

    with ThreadPoolExecutor(max_workers=8) as pool:
        progress = tqdm(pool.map(decode, enumerate(paths)), total=len(paths), desc=f"Cache {tag}", unit="img")
        for count in progress:
            if count % 1000 == 0 or count == len(paths):
                progress.set_postfix(img_s=f"{count / max(time.perf_counter() - started, 1e-6):.0f}")
    array.flush()
    del array
    np.save(y_path, labels, allow_pickle=False)
    done_path.write_text("complete\\n", encoding="utf-8")
    print(f"Wrote {tag} cache in {time.perf_counter() - started:.1f}s: {x_path}")
    return x_path, y_path

TRAIN_X_PATH, TRAIN_Y_PATH = cache_split(train_df, train_paths, "train")
VAL_X_PATH, VAL_Y_PATH = cache_split(val_df, val_paths, "validation")
CLASSES_PATH = CACHE_DIR / "classes.json"
CLASSES_PATH.write_text(json.dumps(classes, indent=2), encoding="utf-8")
print("Image cache ready.")
'''

PROGRAM_MD = """## 3. Write the training program

This cell writes `scripts/train_ddp.py` from the repository into the Kaggle working directory, byte for byte. The notebook contains no second copy of the training logic, so the notebook and `scripts/` cannot drift apart.

The program is architecture-agnostic. The same file trains ResNet18 with `ARCH = "resnet18"`, which is how the ResNet18 training gap gets closed without a second, divergent script.
"""

RUN_MD = """## 4. Smoke test (three optimizer steps)

Run this before the full job. It proves the DDP handshake, the cache path, the loss weighting and the AMP scaler all work, and takes about a minute.

If this cell fails, the full run would have failed too, just 20 minutes later.
"""

RUN_CODE = '''def launch_training(*, dry_run: bool) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        "--nproc_per_node=2", str(SCRIPT_PATH),
        "--arch", ARCH,
        "--train-images", str(TRAIN_X_PATH), "--train-labels", str(TRAIN_Y_PATH),
        "--val-images", str(VAL_X_PATH), "--val-labels", str(VAL_Y_PATH),
        "--classes-json", str(CLASSES_PATH), "--output-dir", str(MODEL_DIR),
        "--epochs", str(EPOCHS), "--patience", str(PATIENCE),
        "--local-batch-size", str(LOCAL_BATCH_SIZE), "--workers", str(WORKERS),
        "--lr", str(LEARNING_RATE), "--weight-decay", str(WEIGHT_DECAY),
        "--label-smoothing", str(LABEL_SMOOTHING),
        "--class-weight-alpha", str(CLASS_WEIGHT_ALPHA),
        "--ema-decay", str(EMA_DECAY), "--rotation-deg", str(ROTATION_DEG),
        "--shuffle-mode", SHUFFLE_MODE, "--seed", str(SEED),
    ]
    if USE_TORCH_COMPILE:
        command.append("--compile")
    if dry_run:
        command.append("--dry-run")

    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    print("Launching:", " ".join(command[-8:]), flush=True)
    result = subprocess.run(command, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training process exited with code {result.returncode}. Read the last log lines above.")

launch_training(dry_run=True)
print("Smoke test passed. Continue to the full run.")
'''

FULL_MD = """## 5. Full training

Expected cost on two T4s with the fixed data path: about **1 to 1.5 minutes per epoch**, so a full 12-epoch run plus validation is roughly **15 to 25 minutes**. The pilot run took 104 minutes for the same work.

Early stopping fires when validation macro-F1 has not improved for `PATIENCE` epochs. The best checkpoint is chosen by validation macro-F1 only, never by test performance.

If Kaggle restarts the session, rerun cells 1-3 and then this cell. Training continues from the last completed epoch as long as `/kaggle/working` survived and the dataset fingerprint still matches.
"""

FULL_CODE = '''launch_training(dry_run=False)
print("Training finished.")
'''

RESULTS_MD = """## 6. Results, comparison and download bundle

Rebuilds the three-model validation comparison, plots the curves, and writes the artifact ZIP. This does **not** open the locked test split and does not train any other architecture.
"""

RESULTS_CODE = '''prior_results = [
    {
        "model": "resnet18", "pretrained": True, "validation_images": 6544,
        "val_accuracy": 0.9614914425427872, "val_balanced_accuracy": 0.9560818914471017,
        "val_macro_precision": 0.9606140206255522, "val_macro_recall": 0.9560818914471017,
        "val_macro_f1": 0.9545999368985558, "best_epoch": 8, "parameters": 11196006,
        "checkpoint_mb": 42.79, "train_minutes": 9.45, "training_mode": "imagenet_finetune",
        "checkpoint": "/kaggle/working/model_comparison/resnet18/resnet18_best.pt",
    },
    {
        "model": "smallcnn", "pretrained": False, "validation_images": 6544,
        "val_accuracy": 0.9601161369193154, "val_balanced_accuracy": 0.9539834655376552,
        "val_macro_precision": 0.9428519200510684, "val_macro_recall": 0.9539834655376552,
        "val_macro_f1": 0.9444106202645282, "best_epoch": 12, "parameters": 398662,
        "checkpoint_mb": 1.54, "train_minutes": float("nan"), "training_mode": "reused_phase2_checkpoint",
        "checkpoint": "/kaggle/working/model_comparison/smallcnn/smallcnn_best.pt",
    },
]

metrics_path = MODEL_DIR / "validation_metrics.json"
if not metrics_path.is_file():
    raise FileNotFoundError(f"Validation metrics missing: {metrics_path}")
new_result = json.loads(metrics_path.read_text(encoding="utf-8"))

comparison = (
    pd.DataFrame(prior_results + [new_result])
    .sort_values("val_macro_f1", ascending=False)
    .reset_index(drop=True)
)
comparison.to_csv(OUTPUT_ROOT / "validation_comparison.csv", index=False)
display(comparison[["model", "val_accuracy", "val_balanced_accuracy", "val_macro_f1",
                    "val_ece", "best_epoch", "parameters", "checkpoint_mb", "train_minutes"]])

history = pd.read_csv(MODEL_DIR / "training_history.csv")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(history.epoch, history.train_loss, marker="o")
ax[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
ax[1].plot(history.epoch, history.val_macro_f1, marker="o", label="raw")
if "val_macro_f1_ema" in history:
    ax[1].plot(history.epoch, history.val_macro_f1_ema, marker="s", label="ema")
ax[1].set(title="Validation macro-F1", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1))
ax[1].legend()
ax[2].plot(history.epoch, history.val_accuracy, marker="o")
ax[2].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
fig.tight_layout()
fig.savefig(OUTPUT_ROOT / "training_curves.png", dpi=160)
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(comparison.model, comparison.val_macro_f1,
       color=["#377eb8" if m == new_result["model"] else "#70ad47" for m in comparison.model])
ax.set_ylim(0, 1)
ax.set_ylabel("Validation macro-F1")
ax.set_title("PlantVillage validation comparison (locked test unused)")
fig.tight_layout()
fig.savefig(OUTPUT_ROOT / "validation_model_comparison.png", dpi=160)
plt.show()
plt.close(fig)

report_csv = MODEL_DIR / "validation_classification_report.csv"
if report_csv.is_file():
    report = pd.read_csv(report_csv).sort_values("f1")
    display(report.head(10))
    print("Weakest classes by F1 written to", report_csv)

metadata = {
    "dataset": "PlantVillage", "train_images": len(train_df), "validation_images": len(val_df),
    "classes": classes, "locked_test_manifest_opened": False, "plantdoc_used_for_training": False,
    "trained_models": [ARCH], "reused_validation_baselines": ["smallcnn", "resnet18"],
    "gpu_names": gpu_names, "training_mode": "torchrun DistributedDataParallel (v2)",
    "global_batch_size": LOCAL_BATCH_SIZE * 2, "per_gpu_batch_size": LOCAL_BATCH_SIZE,
    "workers_per_rank": WORKERS, "learning_rate": LEARNING_RATE,
    "class_weight_alpha": CLASS_WEIGHT_ALPHA, "label_smoothing": LABEL_SMOOTHING,
    "ema_decay": EMA_DECAY, "image_size": IMAGE_SIZE, "seed": SEED,
}
(OUTPUT_ROOT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

zip_path = shutil.make_archive("/kaggle/working/model_comparison_artifacts_v2", "zip", root_dir=OUTPUT_ROOT)
print("Download from the Kaggle Output panel:", zip_path)
print("Best checkpoint:", MODEL_DIR / f"{ARCH}_best.pt")
print("Per-class report:", MODEL_DIR / "validation_classification_report.csv")
print("Confusion matrix:", MODEL_DIR / "validation_confusion_matrix.csv")
'''

NEXT_MD = """## 7. What to do with the bundle

1. Download `model_comparison_artifacts_v2.zip` from the Output panel.
2. Unpack it into the repository so the new checkpoint lives at `checkpoints/efficientnet_b0/efficientnet_b0_best.pt`. That restores the artifact the audit marked missing and closes P0.1.
3. Re-run the locked-test evaluation locally with `python scripts/evaluate_locked_test.py`, which is what produces the verifiable test numbers and the bootstrap confidence intervals.
4. To close P0.2 (no ResNet18 training code), rerun this notebook with `ARCH = "resnet18"` and `MODEL_DIR = OUTPUT_ROOT / "resnet18"`. Same engine, same split, so the comparison becomes *per-architecture tuned on one code path* rather than three ad hoc runs.

Do not report the locked-test numbers in the guide form until step 3 has actually produced them.
"""


def build() -> dict:
    script_text = TRAIN_SCRIPT.read_text(encoding="utf-8")
    cells = [
        md(HEADER),
        md(SETUP_MD),
        code(SETUP_CODE),
        md(CACHE_MD),
        code(CACHE_CODE),
        md(PROGRAM_MD),
        writefile_cell(script_text, "/kaggle/working/train_ddp.py"),
        md(RUN_MD),
        code(RUN_CODE),
        md(FULL_MD),
        code(FULL_CODE),
        md(RESULTS_MD),
        code(RESULTS_CODE),
        md(NEXT_MD),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build()
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # Round-trip and verify the embedded program is byte-identical to the source.
    reloaded = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    embedded = next(
        "".join(cell["source"]) for cell in reloaded["cells"]
        if cell["cell_type"] == CODE and "%%writefile" in "".join(cell["source"])
    )
    body = embedded.split("\n", 1)[1]
    if body != TRAIN_SCRIPT.read_text(encoding="utf-8"):
        raise SystemExit("Embedded training program does not match scripts/train_ddp.py")
    print(f"Wrote {NOTEBOOK.relative_to(ROOT)} ({len(reloaded['cells'])} cells)")
    print("Embedded training program matches scripts/train_ddp.py byte for byte.")


if __name__ == "__main__":
    main()

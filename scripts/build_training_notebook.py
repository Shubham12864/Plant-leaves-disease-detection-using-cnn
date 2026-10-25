#!/usr/bin/env python3
"""Generate the Kaggle training notebook from the canonical script and protocol.

Same anti-drift rule as the evaluation notebook: the notebook embeds the real
files byte for byte instead of restating them, and this builder verifies the
embedded copies match. Regenerate with:

    python scripts/build_training_notebook.py
"""
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_Train_EfficientNet_B0_v2.ipynb"

# target path inside Kaggle -> repository source
EMBEDDED = [
    ("/kaggle/working/train_ddp.py", "scripts/train_ddp.py"),
    ("/kaggle/working/protocol.json", "configs/protocol.json"),
]

HEADER = """# Training: one protocol, four architectures

Trains a model on the PlantVillage development split, on Kaggle, on two T4 GPUs. Every architecture goes through the **same engine against the same frozen protocol** (`protocol.json`), which is what makes the comparison a controlled experiment rather than four ad hoc runs.

Nothing here touches the locked test split. Model selection uses validation macro-F1 only.

## Run modes

Set `RUN_MODE` in cell 1.

| Mode | What it does | When |
|---|---|---|
| `"pilot"` | 1 epoch per architecture, seed 42. Measures wall-clock and confirms no OOM. Writes `pilot_runtime.csv`. | **do this first** |
| `"single"` | one architecture, one seed, full epochs | any single run |
| `"sweep"` | every architecture x every seed in `protocol.json`, full epochs | the 12-run comparison |

The pilot exists because the real sweep is 12 training runs. Measuring one epoch first tells you whether the batch sizes in the protocol are safe on a T4 and how long the sweep will actually take, instead of guessing.

## Why the previous notebook was replaced

The pilot notebook read a 5.5 GB memory map with `num_workers=0`, so the GPUs idled while the main process page-faulted. Per-epoch time grew from 1.7 to 15.65 minutes and the run took 104 minutes. This engine uses worker processes with prefetch, which is the actual fix.

## What changed in the results

- class-balanced cross-entropy, because the training set is 38.4x imbalanced and macro-F1 is the selection metric
- label smoothing 0.05, cosine schedule with 10% warmup, gradient clipping 5.0
- an EMA copy of the weights, evaluated each epoch and kept only when it beats the raw weights
- per-class precision/recall/F1/support, a confusion matrix, ECE and Brier at every validation pass
- a `dataset_fingerprint` in every checkpoint, so a resumed run refuses to continue on a different split

## Run order

| Cell | Does |
|---|---|
| 1 | sets paths and `RUN_MODE`, auto-discovers the dataset mount |
| 2 | builds or reuses the decoded image cache |
| 3 | writes `train_ddp.py` and `protocol.json`, then checks they parse |
| 4 | smoke test: three optimizer steps, no checkpoint written |
| 5 | runs whatever `RUN_MODE` says |
| 6 | comparison table, curves, and the download bundle |
"""

PREPARE = '''from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
import torch, torchvision

# ---- what to run ------------------------------------------------------------
RUN_MODE = "pilot"        # "pilot" | "single" | "sweep"
SEED = 42                 # used by "single" only
ARCH = "efficientnet_b0"  # used by "single" only
# sweep runs every architecture unless this is set. Set it to run one
# architecture per session, which is the recommended way to survive restarts:
# ONLY_ARCH = "convnext_tiny"
ONLY_ARCH = None

# ---- inputs -----------------------------------------------------------------
DATASET_ROOT = Path("/kaggle/input/datasets/shubhamkumar12864/plantvillage")

KAGGLE_INPUT = Path("/kaggle/input")


def autodiscover_dataset_root(preferred: Path) -> Path:
    """Find the attached dataset that holds data/manifests."""
    if (preferred / "data" / "manifests" / "plantvillage_development.csv").is_file():
        return preferred
    if KAGGLE_INPUT.is_dir():
        for manifests in sorted(KAGGLE_INPUT.rglob("manifests")):
            if (manifests / "plantvillage_development.csv").is_file():
                return manifests.parent.parent
    return preferred


_resolved = autodiscover_dataset_root(DATASET_ROOT)
if _resolved != DATASET_ROOT:
    print(f"dataset root corrected: {DATASET_ROOT} -> {_resolved}")
DATASET_ROOT = _resolved

MANIFEST_DIR = DATASET_ROOT / "data" / "manifests"
IMAGE_ROOT = DATASET_ROOT / "data" / "raw" / "PlantVillage"
TRAIN_MANIFEST = MANIFEST_DIR / "plantvillage_development.csv"

WORKDIR = Path("/kaggle/working")
SCRIPT = WORKDIR / "train_ddp.py"
PROTOCOL_PATH = WORKDIR / "protocol.json"
CACHE_DIR = Path("/kaggle/temp/plantvillage_cache")
OUTPUT_ROOT = WORKDIR / "runs"

for folder in (CACHE_DIR, OUTPUT_ROOT, WORKDIR / "src"):
    folder.mkdir(parents=True, exist_ok=True)

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
    raise ValueError(f"Unexpected class split: {len(classes)} train, {val_df.label.nunique()} validation")
if set(train_df.relative_path) & set(val_df.relative_path):
    raise ValueError("Train and validation share image paths.")

IMAGE_SIZE = 224
train_paths = [IMAGE_ROOT / "train" / rel for rel in train_df.relative_path]
val_paths = [IMAGE_ROOT / "train" / rel for rel in val_df.relative_path]
missing = [str(p) for p in train_paths + val_paths if not p.is_file()]
if missing:
    raise FileNotFoundError(f"{len(missing)} manifest images missing; first: {missing[0]}")

expected_bytes = (len(train_df) + len(val_df)) * IMAGE_SIZE * IMAGE_SIZE * 3
free_cache = shutil.disk_usage(CACHE_DIR).free
if free_cache < expected_bytes * 1.10:
    raise OSError(f"Cache needs about {expected_bytes/1e9:.1f} GB; {free_cache/1e9:.1f} GB free")

print(f"torch {torch.__version__} | torchvision {torchvision.__version__}")
print(f"GPUs: {gpu_names}")
print(f"PlantVillage: {len(train_df):,} train, {len(val_df):,} validation, {len(classes)} classes")
print(f"RUN_MODE={RUN_MODE}" + (f" | ARCH={ARCH} SEED={SEED}" if RUN_MODE == "single" else ""))
print("Locked test manifest: not opened.")
'''

CACHE_MD = """## 2. Build or reuse the decoded image cache

Decodes each image once into a uint8 memory map, then the DataLoader workers stream it. The cache key is a hash of the image path list, so changing the split invalidates it automatically.

`/kaggle/temp` is wiped when the session ends, which is the right place for a disposable cache. Re-running this cell reuses a verified cache instead of decoding 43k images again.
"""

CACHE_CODE = '''def cache_split(frame, paths, tag):
    labels = np.asarray([classes.index(label) for label in frame.label], dtype=np.int64)
    path_key = hashlib.sha256("\\n".join(map(str, paths)).encode()).hexdigest()[:16]
    stem = CACHE_DIR / f"{tag}_{IMAGE_SIZE}_{len(paths)}_{path_key}"
    x_path, y_path, done_path = Path(f"{stem}_x.npy"), Path(f"{stem}_y.npy"), Path(f"{stem}.done")
    shape = (len(paths), IMAGE_SIZE, IMAGE_SIZE, 3)

    if done_path.is_file() and x_path.is_file() and y_path.is_file():
        try:
            existing = np.load(x_path, mmap_mode="r", allow_pickle=False)
            if (existing.shape == shape and existing.dtype == np.uint8
                    and np.array_equal(np.load(y_path, allow_pickle=False), labels)):
                del existing
                print(f"Reused verified {tag} cache: {x_path}")
                return x_path, y_path
            del existing
        except Exception:
            pass

    for stale in (done_path, x_path, y_path):
        stale.unlink(missing_ok=True)
    array = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.uint8, shape=shape)
    started = time.perf_counter()

    def decode(item):
        i, path = item
        with Image.open(path) as image:
            array[i] = np.asarray(
                image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR),
                dtype=np.uint8,
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

WRITE_MD = """## 3. Write the engine and the protocol

One cell writes both files and then checks that they parse. There is no ordering to get wrong: if this cell succeeded, the engine and the protocol exist and are valid.

The engine is architecture-agnostic. `smallcnn`, `resnet18`, `efficientnet_b0` and `convnext_tiny` all run through it, which is what lets the comparison claim a shared protocol instead of "same split, different everything".
"""

SMOKE_MD = """## 4. Smoke test

Three optimizer steps on the smallest architecture. Proves the DDP handshake, the cache path, the class-weighted loss and the AMP scaler work, in about a minute. No checkpoint is written.

If this fails, the full run would have failed too, twenty minutes later.
"""

RUN_MD = """## 5. Run

Behaviour depends on `RUN_MODE` from cell 1.

**pilot** — one epoch per architecture at seed 42. Writes `runs/pilot_runtime.csv` with wall-clock per architecture and the epoch-1 validation score.

Measured on 2026-10-25 (2x T4, batch per `protocol.json`): SmallCNN 0.67 min/epoch (914 img/s), ResNet18 0.78 (786), EfficientNet-B0 2.45 (250), ConvNeXt-Tiny 3.15 (194). No OOM at any batch size, so the protocol stands as written. Projected 12-run sweep: **4.2 h** as an upper bound, ignoring early stopping.

Epoch-1 accuracy in that table is **not a result** — with 10% warmup over 12 epochs the learning rate is still ramping, so 6–21% top-1 is expected. The pilot measures time and memory only.

**single** — `ARCH` at `SEED`, full epochs.

**sweep** — every architecture x every seed. 12 runs, about 4 hours total. Set `ONLY_ARCH` to a single architecture to run it per session, which is the recommended plan: `smallcnn` (0.4 h), `resnet18` (0.5 h), `efficientnet_b0` (1.5 h), `convnext_tiny` (1.9 h). Each run keeps a resumable `_last.pt`, so a restarted session continues from the last completed epoch.
"""

RESULTS_MD = """## 6. Results and download bundle

Gathers every `validation_metrics.json` under `runs/`, builds the comparison table, plots macro-F1 per architecture with the spread across seeds, and writes the artifact ZIP.

This never opens the locked test split. Test scoring is a separate notebook (`04_Locked_Test_Evaluation.ipynb`).
"""

NEXT_MD = """## What to do with the bundle

1. Download `training_runs.zip` from the Output panel.
2. Unpack into the repository so checkpoints land at `checkpoints/<arch>/<arch>_best.pt` and histories under `reports/model_comparison/runs/`.
3. Score the winner on the locked test split with `notebooks/04_Locked_Test_Evaluation.ipynb`.
4. Register every number in `results/registry.json` before any document quotes it.

Do not compare a validation score with a test score in the same table.
"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def files_cell(files: list[tuple[str, str]]) -> dict:
    """Write every embedded file from one literal dict, then verify it parses."""
    lines = ["# Writes the engine and the protocol, then checks both. One cell, so the",
             "# files always exist before anything imports or runs them.",
             "FILES = {"]
    for target, content in files:
        lines.append(f"    {target!r}: {content!r},")
    lines += [
        "}",
        "",
        "for _path, _content in FILES.items():",
        "    _target = Path(_path)",
        "    _target.parent.mkdir(parents=True, exist_ok=True)",
        "    _target.write_text(_content, encoding='utf-8')",
        "    print(f'wrote {_target.name:16s} {len(_content):7d} bytes')",
        "",
        "import py_compile",
        "py_compile.compile(str(SCRIPT), doraise=True)",
        "print('train_ddp.py compiles')",
        "",
        "PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))",
        "print('protocol.json loads:', list(PROTOCOL['architectures']))",
        "missing_keys = [a for a in ('smallcnn', 'resnet18', 'efficientnet_b0', 'convnext_tiny')",
        "                if a not in PROTOCOL['architectures']]",
        "if missing_keys:",
        "    raise ValueError(f'protocol.json is missing architectures: {missing_keys}')",
        "print('protocol covers all four architectures')",
    ]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]}


SMOKE_CODE = '''smoke_cmd = [
    sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
    "--nproc_per_node=2", str(SCRIPT),
    "--arch", "smallcnn", "--train-images", str(TRAIN_X_PATH), "--train-labels", str(TRAIN_Y_PATH),
    "--val-images", str(VAL_X_PATH), "--val-labels", str(VAL_Y_PATH),
    "--classes-json", str(CLASSES_PATH), "--output-dir", str(WORKDIR / "_smoke"),
    "--local-batch-size", "16", "--eval-batch-size", "32", "--workers", "2",
    "--image-size", str(PROTOCOL["image_size"]), "--dry-run",
]
print("Running:", " ".join(smoke_cmd[-10:]), flush=True)
result = subprocess.run(smoke_cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=False)
if result.returncode != 0:
    raise RuntimeError(f"Smoke test failed with code {result.returncode}. Read the log above.")
print("Smoke test passed. Continue to the run cell.")
'''

RUN_CODE = '''def launch(arch, seed, epochs, out_dir, dry_run=False):
    cfg = PROTOCOL["architectures"][arch]
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        "--nproc_per_node=2", str(SCRIPT),
        "--arch", arch,
        "--train-images", str(TRAIN_X_PATH), "--train-labels", str(TRAIN_Y_PATH),
        "--val-images", str(VAL_X_PATH), "--val-labels", str(VAL_Y_PATH),
        "--classes-json", str(CLASSES_PATH), "--output-dir", str(out_dir),
        "--epochs", str(epochs), "--patience", str(PROTOCOL["patience"]),
        "--local-batch-size", str(cfg["local_batch_size"]),
        "--eval-batch-size", str(PROTOCOL["eval_batch_size"]),
        "--lr", str(cfg["lr"]), "--weight-decay", str(cfg["weight_decay"]),
        "--workers", str(PROTOCOL["workers"]),
        "--label-smoothing", str(PROTOCOL["label_smoothing"]),
        "--class-weight-alpha", str(PROTOCOL["class_weight_alpha"]),
        "--ema-decay", str(PROTOCOL["ema_decay"]),
        "--grad-clip", str(PROTOCOL["grad_clip"]),
        "--warmup-fraction", str(PROTOCOL["warmup_fraction"]),
        "--rotation-deg", str(PROTOCOL["rotation_deg"]),
        "--shuffle-mode", PROTOCOL["shuffle_mode"],
        "--image-size", str(PROTOCOL["image_size"]),
        "--seed", str(seed),
    ]
    if dry_run:
        command.append("--dry-run")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"\\n=== {arch} seed {seed} epochs {epochs} -> {out_dir} ===", flush=True)
    result = subprocess.run(command, env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=False)
    minutes = (time.time() - started) / 60.0
    if result.returncode != 0:
        raise RuntimeError(f"{arch} seed {seed} failed with code {result.returncode}. Read the log above.")
    return minutes


if RUN_MODE == "pilot":
    rows = []
    for arch in PROTOCOL["architectures"]:
        out_dir = OUTPUT_ROOT / "pilot" / arch
        minutes = launch(arch, PROTOCOL["seeds"][0], epochs=1, out_dir=out_dir)
        metrics_path = out_dir / "validation_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        rows.append({
            "architecture": arch,
            "seed": PROTOCOL["seeds"][0],
            "epochs": 1,
            "wall_clock_minutes": round(minutes, 2),
            "local_batch_size": PROTOCOL["architectures"][arch]["local_batch_size"],
            "val_accuracy_after_1_epoch": metrics.get("val_accuracy"),
            "val_macro_f1_after_1_epoch": metrics.get("val_macro_f1"),
            "parameters": metrics.get("parameters"),
        })
        pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "pilot_runtime.csv", index=False)
    display(pd.DataFrame(rows))
    total = sum(r["wall_clock_minutes"] for r in rows) * len(PROTOCOL["seeds"]) * PROTOCOL["epochs"]
    print(f"\\nProjected full sweep: {total/60:.1f} hours of GPU time "
          f"({len(PROTOCOL['architectures'])} archs x {len(PROTOCOL['seeds'])} seeds x {PROTOCOL['epochs']} epochs)")

elif RUN_MODE == "single":
    launch(ARCH, SEED, PROTOCOL["epochs"], OUTPUT_ROOT / ARCH / f"seed{SEED}")

elif RUN_MODE == "sweep":
    architectures = [ONLY_ARCH] if ONLY_ARCH else list(PROTOCOL["architectures"])
    if ONLY_ARCH and ONLY_ARCH not in PROTOCOL["architectures"]:
        raise ValueError(f"ONLY_ARCH {ONLY_ARCH!r} is not in protocol.json")
    print(f"Sweeping {architectures} x seeds {PROTOCOL['seeds']} x {PROTOCOL['epochs']} epochs")
    for arch in architectures:
        for seed in PROTOCOL["seeds"]:
            launch(arch, seed, PROTOCOL["epochs"], OUTPUT_ROOT / arch / f"seed{seed}")

else:
    raise ValueError(f"Unknown RUN_MODE {RUN_MODE!r}")
print("\\nDone.")
'''

RESULTS_CODE = '''records = []
for metrics_path in sorted(OUTPUT_ROOT.rglob("validation_metrics.json")):
    run_dir = metrics_path.parent
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    records.append({
        "architecture": metrics.get("model"),
        "run": str(run_dir.relative_to(OUTPUT_ROOT)),
        "val_accuracy": metrics.get("val_accuracy"),
        "val_macro_f1": metrics.get("val_macro_f1"),
        "val_balanced_accuracy": metrics.get("val_balanced_accuracy"),
        "val_ece": metrics.get("val_ece"),
        "val_brier": metrics.get("val_brier"),
        "best_epoch": metrics.get("best_epoch"),
        "parameters": metrics.get("parameters"),
        "train_minutes": metrics.get("train_minutes"),
        "weights_are_ema": metrics.get("weights_are_ema"),
        "dataset_fingerprint": (metrics.get("dataset_fingerprint") or "")[:12],
    })

if not records:
    raise FileNotFoundError("No validation_metrics.json found under runs/. Did the run cell finish?")

comparison = pd.DataFrame(records).sort_values(["architecture", "run"]).reset_index(drop=True)
comparison.to_csv(OUTPUT_ROOT / "comparison.csv", index=False)
display(comparison)

# Per-architecture spread across seeds, so a single lucky run cannot be mistaken
# for a stable result.
summary = (comparison.groupby("architecture")["val_macro_f1"]
           .agg(["count", "mean", "std", "min", "max"]).round(6)
           .sort_values("mean", ascending=False))
print("\\nValidation macro-F1 per architecture (across seeds)")
display(summary)

if len(comparison) > 1:
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (arch, group) in enumerate(comparison.groupby("architecture")):
        ax.scatter([i] * len(group), group["val_macro_f1"], s=60, label=arch)
        ax.hlines(group["val_macro_f1"].mean(), i - 0.2, i + 0.2, colors="black", linewidth=2)
    ax.set_xticks(range(len(comparison["architecture"].unique())))
    ax.set_xticklabels(sorted(comparison["architecture"].unique()), rotation=15)
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Per-architecture validation macro-F1 (bar = mean across seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "macro_f1_by_architecture.png", dpi=160)
    plt.show()
    plt.close(fig)

zip_path = shutil.make_archive("/kaggle/working/training_runs", "zip", root_dir=OUTPUT_ROOT)
print(f"\\nDownload from the Kaggle Output panel: {zip_path}")
print("Train minutes per run:")
print(comparison[["architecture", "run", "train_minutes", "val_macro_f1"]].to_string(index=False))
'''

NEXT_MD_POST = NEXT_MD


def build() -> dict:
    files = [(target, (ROOT / source).read_text(encoding="utf-8")) for target, source in EMBEDDED]
    cells = [
        md(HEADER),
        md("## 1. Paths and mode"),
        code(PREPARE),
        md(CACHE_MD),
        code(CACHE_CODE),
        md(WRITE_MD),
        files_cell(files),
        md(SMOKE_MD),
        code(SMOKE_CODE),
        md(RUN_MD),
        code(RUN_CODE),
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

    reloaded = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    embedded = None
    for cell in reloaded["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] == "code" and "FILES = {" in text:
            start = text.index("FILES = {") + len("FILES = ")
            depth, end = 0, None
            for index, char in enumerate(text[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = index + 1
                        break
            if end is None:
                raise SystemExit("Could not parse the FILES literal")
            embedded = ast.literal_eval(text[start:end])
            break
    if embedded is None:
        raise SystemExit("Notebook contains no FILES writer cell")

    for target, source in EMBEDDED:
        expected = (ROOT / source).read_text(encoding="utf-8")
        if embedded.get(target) != expected:
            raise SystemExit(f"Embedded copy of {source} does not match")
    if len(embedded) != len(EMBEDDED):
        raise SystemExit(f"Writer cell has {len(embedded)} files, expected {len(EMBEDDED)}")

    sandbox = Path(tempfile.mkdtemp(prefix="writecheck_"))
    for target, content in embedded.items():
        destination = sandbox / target.lstrip("/").replace("kaggle/working/", "")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        if destination.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Round-trip write failed for {target}")

    print(f"Wrote {NOTEBOOK.relative_to(ROOT)} ({len(reloaded['cells'])} cells)")
    print(f"Verified {len(embedded)} embedded files byte for byte and by round-trip write.")


if __name__ == "__main__":
    main()

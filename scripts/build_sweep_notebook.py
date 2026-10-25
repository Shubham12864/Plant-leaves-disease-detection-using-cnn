#!/usr/bin/env python3
"""Generate the all-in-one sweep notebook.

One session, one Run All, twelve training runs, scheduled for wall-clock rather
than one-architecture-per-session. Embeds the real engine, worker and protocol
byte for byte and verifies them, same rule as the other generated notebooks.

    python scripts/build_sweep_notebook.py
"""
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "06_Sweep_All.ipynb"

EMBEDDED = [
    ("/kaggle/working/train_ddp.py", "scripts/train_ddp.py"),
    ("/kaggle/working/sweep_worker.py", "scripts/sweep_worker.py"),
    ("/kaggle/working/protocol.json", "configs/protocol.json"),
]

HEADER = """# Full sweep: four architectures, three seeds, one session

Runs all twelve training runs in one Kaggle session and writes a single bundle.

Every stage uses **both GPUs** through DDP. That is deliberate, and it is not the
obvious choice — see below.

## Why every stage uses both GPUs

I built a schedule that ran the two small models concurrently, one GPU each, to
"use both GPUs better". Measured against the pilot rates, it is not faster:

| Stage | min/epoch | 36 epochs |
|---|---:|---:|
| SmallCNN, 2 GPUs (DDP) | 0.67 | 24 min |
| SmallCNN, 1 GPU | ~1.34 (2x slower) | ~48 min |
| ResNet18, 2 GPUs (DDP) | 0.78 | 28 min |
| ResNet18, 1 GPU | ~1.56 | ~56 min |

Two concurrent single-GPU jobs finish the small models in `max(48, 56) = 56 min`.
Two sequential DDP jobs finish them in `24 + 28 = 52 min`. DDP scales close
enough to 2x that splitting the GPUs buys nothing, and the parallel path adds a
failure mode for free.

So the schedule is sequential DDP, ordered by importance:

1. **EfficientNet-B0** — 88 min. First because it is the reference model for H1–H4.
2. **ConvNeXt-Tiny** — 113 min.
3. **ResNet18** — 28 min.
4. **SmallCNN** — 24 min.

Total **~4.2 h**. The honest statement is that scheduling does not shorten this
sweep: 253 minutes of training is 253 minutes of training. The levers that do
change the run time are below.

## The real levers, in order of effect

| Lever | Saving | Cost |
|---|---|---|
| Early stopping (already on, patience 3) | 20–30%, so likely **3.0–3.4 h** instead of 4.2 | none |
| `SEED_PLAN` with 3 seeds for the reference only | **~2.4 h** | other architectures get no variance estimate |
| Raising EfficientNet-B0's batch above 64/GPU | unknown, likely <10% — it is bandwidth-bound, not compute-bound | needs a re-pilot |
| `torch.compile` | 10–15% on the two big models | adds per-run warm-up, can fail on Kaggle |

The first lever is free, which is why patience is 3. The second is a deliberate
downgrade and is commented, not applied, in cell 1.

## Resuming

Set `ONLY_STAGE` in cell 1 to re-run one stage. Each seed writes its own
resumable `_last.pt`, and a seed whose `validation_metrics.json` already exists
is skipped, so re-running a stage costs nothing for runs that finished.

| `ONLY_STAGE` | Runs | ~Time |
|---|---|---|
| `None` | everything | 4.2 h |
| `"efficientnet_b0"` | 3 seeds | 88 min |
| `"convnext_tiny"` | 3 seeds | 113 min |
| `"resnet18"` | 3 seeds | 28 min |
| `"smallcnn"` | 3 seeds | 24 min |

A parallel single-GPU stage is still available as a commented line in the
schedule cell. It is documented rather than removed so the measurement above is
reproducible.
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
ONLY_STAGE = None          # None | "efficientnet_b0" | "convnext_tiny" | "small"
SEED_PLAN = None           # None = protocol seeds for every architecture
# Cheaper alternative, if runtime matters more than per-architecture variance:
#   SEED_PLAN = {"efficientnet_b0": [42, 43, 44], "convnext_tiny": [42],
#                "resnet18": [42], "smallcnn": [42]}

# ---- inputs -----------------------------------------------------------------
DATASET_ROOT = Path("/kaggle/input/datasets/shubhamkumar12864/plantvillage")
KAGGLE_INPUT = Path("/kaggle/input")


def autodiscover_dataset_root(preferred: Path) -> Path:
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
WORKER = WORKDIR / "sweep_worker.py"
PROTOCOL_PATH = WORKDIR / "protocol.json"
CACHE_DIR = Path("/kaggle/temp/plantvillage_cache")
OUTPUT_ROOT = WORKDIR / "runs"

for folder in (CACHE_DIR, OUTPUT_ROOT):
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

print(f"GPUs: {gpu_names}")
print(f"PlantVillage: {len(train_df):,} train, {len(val_df):,} validation, {len(classes)} classes")
print(f"ONLY_STAGE={ONLY_STAGE!r} | SEED_PLAN={'protocol default' if SEED_PLAN is None else SEED_PLAN}")
print("Locked test manifest: not opened.")
'''

CACHE_MD = """## 2. Image cache

Built once per session and shared by all twelve runs. The cache key is a hash of the path list, so a changed split invalidates it automatically. This is the main reason not to split the sweep across sessions: each session would pay this decode again.
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
            ok = (existing.shape == shape and existing.dtype == np.uint8
                  and np.array_equal(np.load(y_path, allow_pickle=False), labels))
            del existing
            if ok:
                print(f"Reused verified {tag} cache: {x_path}")
                return x_path, y_path
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
    print(f"Wrote {tag} cache in {time.perf_counter() - started:.1f}s")
    return x_path, y_path


TRAIN_X_PATH, TRAIN_Y_PATH = cache_split(train_df, train_paths, "train")
VAL_X_PATH, VAL_Y_PATH = cache_split(val_df, val_paths, "val")
CLASSES_PATH = CACHE_DIR / "classes.json"
CLASSES_PATH.write_text(json.dumps(classes, indent=2), encoding="utf-8")
print("Image cache ready.")
'''

WRITE_MD = """## 3. Write the engine, the worker and the protocol

One cell, three files, then a parse check on each. If this cell succeeded, everything below can run.
"""

SCHEDULE_MD = """## 4. Run the schedule

Four stages in order, every one on both GPUs through DDP.

Progress is printed per stage and per seed. A seed that already has `validation_metrics.json` is skipped, so re-running after a failure costs only the seeds that did not finish.

If a stage fails the cell raises, but everything already written stays under `runs/`, so the results cell still works on partial data.
"""

RESULTS_MD = """## 5. Results and one bundle

Aggregates every sweep run into the comparison table, reports macro-F1 per architecture as mean plus spread across seeds, and zips `runs/`.

A single-number summary is not enough here: three seeds exist precisely so the spread is visible, and a difference smaller than that spread is not a result.
"""

NEXT_MD = """## After the sweep

1. Download the zip.
2. Unpack `runs/<arch>/seed<N>/` into `reports/model_comparison/runs/` and the best checkpoints into `checkpoints/<arch>/`.
3. Register the numbers in `results/registry.json` before any document quotes them.
4. Then Stage C: score PlantDoc and measure the field gap, which is the actual contribution.

Do not compare a validation score with a test score in the same table, and do not treat a per-architecture difference smaller than its seed spread as a finding.
"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def files_cell(files):
    lines = ["# Writes the engine, the worker and the protocol, then checks all three.",
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
        "    print(f'wrote {_target.name:18s} {len(_content):7d} bytes')",
        "",
        "import py_compile",
        "for _file in (SCRIPT, WORKER):",
        "    py_compile.compile(str(_file), doraise=True)",
        "    print(f'{_file.name} compiles')",
        "",
        "PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))",
        "ARCHITECTURES = list(PROTOCOL['architectures'])",
        "print('protocol architectures:', ARCHITECTURES)",
        "print('protocol seeds         :', PROTOCOL['seeds'])",
    ]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]}


RUN_CODE = '''# Measured on 2x T4 during the pilot, used only to print an estimate.
PILOT_MINUTES = {"smallcnn": 0.67, "resnet18": 0.78, "efficientnet_b0": 2.45, "convnext_tiny": 3.15}

seeds_for = (lambda arch: list(SEED_PLAN.get(arch, PROTOCOL["seeds"]))) if SEED_PLAN else \\
            (lambda arch: list(PROTOCOL["seeds"]))

# stage name -> (kind, architectures). "ddp" takes both GPUs; "parallel" runs one
# single-GPU process per architecture at the same time.
# Sequential DDP for every stage. Measured: two concurrent single-GPU jobs
# (max(1.56, 1.34) min/epoch) are no faster than two sequential DDP jobs
# (0.78 + 0.67 min/epoch) and add a failure mode. Kept as a comment so the
# measurement stays reproducible:
#   ("small", "parallel", ["resnet18", "smallcnn"]),
SCHEDULE = [
    ("efficientnet_b0", "ddp", ["efficientnet_b0"]),
    ("convnext_tiny", "ddp", ["convnext_tiny"]),
    ("resnet18", "ddp", ["resnet18"]),
    ("smallcnn", "ddp", ["smallcnn"]),
]

cache_args = [
    "--train-images", str(TRAIN_X_PATH), "--train-labels", str(TRAIN_Y_PATH),
    "--val-images", str(VAL_X_PATH), "--val-labels", str(VAL_Y_PATH),
    "--classes-json", str(CLASSES_PATH), "--output-root", str(OUTPUT_ROOT),
    "--protocol", str(PROTOCOL_PATH),
]


def worker_command(arch, mode):
    return [sys.executable, str(WORKER), "--arch", arch, "--mode", mode,
            "--seeds", *[str(s) for s in seeds_for(arch)], *cache_args]


def estimate_hours():
    total = 0.0
    for name, kind, archs in SCHEDULE:
        per_arch = [PILOT_MINUTES[a] * PROTOCOL["epochs"] * len(seeds_for(a)) / 60 for a in archs]
        total += max(per_arch) if kind == "parallel" else sum(per_arch)
    return total


print(f"Seed plan: {{{', '.join(f'{a}: {len(seeds_for(a))}' for a in ARCHITECTURES)}}}")
print(f"Estimated wall clock (pilot rates, no early stopping): {estimate_hours():.2f} h")
print(f"Running stage(s): {ONLY_STAGE or 'all'}\\n")

started_all = time.time()
for name, kind, archs in SCHEDULE:
    if ONLY_STAGE and name != ONLY_STAGE:
        continue
    print(f"\\n===== stage '{name}' ({kind}) : {archs} =====", flush=True)
    if kind == "ddp":
        for arch in archs:
            result = subprocess.run(worker_command(arch, "ddp"),
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"stage {name}: {arch} failed with code {result.returncode}")
    else:
        running = []
        for gpu, arch in enumerate(archs):
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": str(gpu)}
            print(f"  launching {arch} on GPU {gpu}", flush=True)
            running.append((arch, subprocess.Popen(worker_command(arch, "single"), env=env)))
        for arch, process in running:
            code_ = process.wait()
            print(f"  {arch} finished with code {code_}", flush=True)
            if code_ != 0:
                raise RuntimeError(f"stage {name}: {arch} failed with code {code_}")

print(f"\\nAll requested stages finished in {(time.time() - started_all)/60:.1f} min")
'''

RESULTS_CODE = '''records = []
for metrics_path in sorted(OUTPUT_ROOT.rglob("validation_metrics.json")):
    parts = metrics_path.relative_to(OUTPUT_ROOT).parts
    if "_smoke" in parts or "pilot" in parts:
        continue
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    records.append({
        "architecture": metrics.get("model"),
        "run": str(metrics_path.parent.relative_to(OUTPUT_ROOT)),
        "seed": int(metrics_path.parent.name.replace("seed", "") or 0),
        "val_accuracy": metrics.get("val_accuracy"),
        "val_macro_f1": metrics.get("val_macro_f1"),
        "val_balanced_accuracy": metrics.get("val_balanced_accuracy"),
        "val_ece": metrics.get("val_ece"),
        "val_brier": metrics.get("val_brier"),
        "best_epoch": metrics.get("best_epoch"),
        "parameters": metrics.get("parameters"),
        "train_minutes": metrics.get("train_minutes"),
        "weights_are_ema": metrics.get("weights_are_ema"),
        "fingerprint": (metrics.get("dataset_fingerprint") or "")[:12],
    })

if not records:
    raise FileNotFoundError("No sweep runs found under runs/. Did the schedule cell finish?")

comparison = pd.DataFrame(records).sort_values(["architecture", "seed"]).reset_index(drop=True)
comparison.to_csv(OUTPUT_ROOT / "comparison.csv", index=False)
display(comparison)

expected = sum(len(seeds_for(a)) for a in ARCHITECTURES)
print(f"\\n{len(comparison)} of {expected} planned runs present")
missing = [(a, s) for a in ARCHITECTURES for s in seeds_for(a)
           if not ((OUTPUT_ROOT / a / f"seed{s}" / "validation_metrics.json").is_file())]
if missing:
    print("Not yet run:", missing)
print("\\nFingerprints (all runs of one architecture must agree):")
print(comparison.groupby("architecture")["fingerprint"].apply(lambda col: sorted(set(col))).to_string())

summary = (comparison.groupby("architecture")["val_macro_f1"]
           .agg(runs="count", mean="mean", std="std", min="min", max="max").round(6)
           .sort_values("mean", ascending=False))
summary["spread_pp"] = ((summary["max"] - summary["min"]) * 100).round(2)
print("\\nValidation macro-F1 per architecture across seeds")
display(summary)

if summary["runs"].max() > 1:
    fig, ax = plt.subplots(figsize=(9, 4))
    order = list(summary.index)
    for i, arch in enumerate(order):
        values = comparison.loc[comparison.architecture == arch, "val_macro_f1"]
        ax.scatter([i] * len(values), values, s=70, zorder=3)
        ax.hlines(values.mean(), i - 0.22, i + 0.22, colors="black", linewidth=2, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=12)
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Per-architecture validation macro-F1 (points = seeds, bar = mean)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "macro_f1_by_architecture.png", dpi=160)
    plt.show()
    plt.close(fig)

zip_path = shutil.make_archive("/kaggle/working/sweep_all", "zip", root_dir=OUTPUT_ROOT)
print(f"\\nDownload from the Kaggle Output panel: {zip_path}")
print(f"Size: {Path(zip_path).stat().st_size/1e6:.0f} MB")
'''


def build() -> dict:
    files = [(target, (ROOT / source).read_text(encoding="utf-8")) for target, source in EMBEDDED]
    cells = [
        md(HEADER),
        md("## 1. Paths, stage and seed plan"),
        code(PREPARE),
        md(CACHE_MD),
        code(CACHE_CODE),
        md(WRITE_MD),
        files_cell(files),
        md(SCHEDULE_MD),
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
        if embedded.get(target) != (ROOT / source).read_text(encoding="utf-8"):
            raise SystemExit(f"Embedded copy of {source} does not match")
    if len(embedded) != len(EMBEDDED):
        raise SystemExit(f"Writer has {len(embedded)} files, expected {len(EMBEDDED)}")

    sandbox = Path(tempfile.mkdtemp(prefix="sweepcheck_"))
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

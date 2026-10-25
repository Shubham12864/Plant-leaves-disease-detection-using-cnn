#!/usr/bin/env python3
"""Generate the overnight notebook (notebooks/08).

Twelve full training runs in one unattended session, built so that no single
failure can waste the night:

  * the gate runs first and silences the sweep if the recipe cannot learn
  * every run is a subprocess with a timeout, caught and recorded
  * finished seeds are skipped, so a restart resumes instead of repeating
  * a wall-clock budget stops starting new runs before the session dies
  * the results cell always writes a zip, including on partial completion

    python scripts/build_overnight_notebook.py
"""
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "08_Overnight_Sweep.ipynb"

EMBEDDED = [
    ("/kaggle/working/train_ddp.py", "scripts/train_ddp.py"),
    ("/kaggle/working/sweep_worker.py", "scripts/sweep_worker.py"),
    ("/kaggle/working/protocol_v2.json", "configs/protocol_v2.json"),
]

HEADER = """# Overnight sweep — twelve runs, unattended

Start this before you sleep, with **Save Version → Save and Run All** so it keeps
going after the browser closes. Expect 5–7 hours.

## What it produces

| Stage | Runs | Purpose |
|---|---|---|
| gate | 1 × 2 epochs | proves the recipe learns before anything long starts |
| sweep | 4 architectures × 3 seeds × 12 epochs | the comparison that Stage B needs |
| extra seeds | 2 more EfficientNet-B0 seeds | tighter variance on the reference model, only if time allows |

Everything lands in one zip that the last cell writes **even if earlier stages
failed**.

## Why you can leave it running

Five things could waste a night. Each one is handled:

| Failure | Handling |
|---|---|
| The recipe still cannot learn | the gate stops it in ~5 minutes; the sweep never starts |
| One run crashes (OOM, NaN, bug) | caught per run, recorded in `_overnight_log.json`, the loop continues |
| One run hangs | subprocess timeout kills it and moves on |
| The session dies halfway | finished seeds are skipped on the next run, so a restart resumes |
| The session hits its time limit | a budget stops launching new runs ~45 minutes before the limit |

## The recipe, and why it is not negotiable

`protocol_v2` restores the settings that produced the 99.54% reference: AdamW
lr 1e-4, weight decay 1e-4, global batch 32, constant learning rate, no label
smoothing, no class weights, **random shuffling**.

That last one matters. v1 and the first v2 gate used blocked shuffling, and the
manifest is written in class order, so batches contained a median of 3 classes
instead of 12. That broke BatchNorm: the only architecture that trained was
ConvNeXt, which uses LayerNorm and ignores batch composition. Evidence in
`docs/KNOWN_ISSUES.md` R14.

**Do not "improve" the batch size or the learning rate in this notebook.** The
one time that was done, twelve runs and 5.6 GPU-hours produced nothing usable.
"""

SETUP = '''from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
import torch, torchvision

# ---- knobs ------------------------------------------------------------------
SESSION_BUDGET_H = 10.0    # stop starting new runs after this many hours
RESERVE_H = 0.75           # leave this much at the end for results + download
ALLOW_EXTRA_SEEDS = True   # 2 extra EfficientNet-B0 seeds if the budget allows
GATE_EPOCHS = 2
GATE_THRESHOLD = 0.95

STARTED_AT = time.time()

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
PROTOCOL_PATH = WORKDIR / "protocol_v2.json"
CACHE_DIR = Path("/kaggle/temp/plantvillage_cache")
OUTPUT_ROOT = WORKDIR / "runs"
GATE_ROOT = OUTPUT_ROOT / "_gate"
LOG_PATH = OUTPUT_ROOT / "_overnight_log.json"

for folder in (CACHE_DIR, OUTPUT_ROOT):
    folder.mkdir(parents=True, exist_ok=True)

# ---- preflight: fail loudly and early, never eight hours in ------------------
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("Set Accelerator -> GPU T4 x2 in Kaggle Settings, then Run All again.")
gpu_names = [torch.cuda.get_device_name(i) for i in range(2)]
print(f"GPUs: {gpu_names}")
if not all("T4" in name for name in gpu_names):
    print("Warning: timings are tuned for 2x T4.")

if not TRAIN_MANIFEST.is_file() or not IMAGE_ROOT.is_dir():
    raise FileNotFoundError(
        f"PlantVillage input missing. Expected\\n  {TRAIN_MANIFEST}\\n  {IMAGE_ROOT}\\n"
        "Attach the PlantVillage dataset to this notebook."
    )

dev = pd.read_csv(TRAIN_MANIFEST)
required = {"split", "relative_path", "label"}
if not required.issubset(dev.columns):
    raise ValueError(f"Development manifest needs columns {sorted(required)}, found {list(dev.columns)}")

train_df = dev.loc[dev.split == "train"].reset_index(drop=True)
val_df = dev.loc[dev.split == "validation"].reset_index(drop=True)
classes = sorted(train_df.label.unique().tolist())
if len(classes) != 38:
    raise ValueError(f"Expected 38 classes, found {len(classes)}")
if set(val_df.label) != set(classes):
    raise ValueError("Validation split does not cover the same 38 classes")
if set(train_df.relative_path) & set(val_df.relative_path):
    raise ValueError("Train and validation share image paths")

IMAGE_SIZE = 224
train_paths = [IMAGE_ROOT / "train" / rel for rel in train_df.relative_path]
val_paths = [IMAGE_ROOT / "train" / rel for rel in val_df.relative_path]
missing = [str(p) for p in train_paths[:200] + val_paths[:200] if not p.is_file()]
if missing:
    raise FileNotFoundError(f"Manifest paths do not resolve, e.g. {missing[0]}")

expected_bytes = (len(train_df) + len(val_df)) * IMAGE_SIZE * IMAGE_SIZE * 3
free_cache = shutil.disk_usage(CACHE_DIR).free
if free_cache < expected_bytes * 1.10:
    raise OSError(f"Cache needs ~{expected_bytes/1e9:.1f} GB, only {free_cache/1e9:.1f} GB free in /kaggle/temp")

def log(entry):
    """Append a progress record to disk. Defined here so no later cell depends
    on which cell ran first: an ordering dependency is exactly the kind of thing
    that kills an unattended run."""
    entries = json.loads(LOG_PATH.read_text()) if LOG_PATH.is_file() else []
    entries.append(entry)
    LOG_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


print(f"PlantVillage: {len(train_df):,} train / {len(val_df):,} validation / {len(classes)} classes")
print(f"Budget: {SESSION_BUDGET_H} h session, stopping new runs {RESERVE_H} h before the end")
print(f"Start time: {time.strftime('%H:%M')} — expected finish around "
      f"{time.strftime('%H:%M', time.localtime(time.time() + 6*3600))}")
print("Locked test manifest: not opened.")
'''

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
    spare = shutil.disk_usage(CACHE_DIR).free - int(shape[0] * shape[1] * shape[2] * 3 * 1.05)
    if spare < 1e9:
        raise OSError(f"Not enough space to build the {tag} cache; {spare/1e9:.2f} GB would remain")

    array = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.uint8, shape=shape)
    started = time.perf_counter()
    failures = []

    def decode(item):
        i, path = item
        try:
            with Image.open(path) as image:
                array[i] = np.asarray(
                    image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
            return i
        except Exception as exc:                      # keep going, report at the end
            failures.append((str(path), repr(exc)))
            array[i] = 0
            return i

    with ThreadPoolExecutor(max_workers=8) as pool:
        progress = tqdm(pool.map(decode, enumerate(paths)), total=len(paths), desc=f"Cache {tag}", unit="img")
        for count, _ in enumerate(progress, start=1):
            if count % 1000 == 0 or count == len(paths):
                progress.set_postfix(img_s=f"{count / max(time.perf_counter() - started, 1e-6):.0f}")
    array.flush()
    del array
    np.save(y_path, labels, allow_pickle=False)

    if failures:
        print(f"{len(failures)} image(s) failed to decode; first 3:")
        for path, err in failures[:3]:
            print(f"  {path}: {err}")
        raise RuntimeError(f"{len(failures)} undecodable images in the {tag} split; cache not marked complete")

    done_path.write_text("complete\\n", encoding="utf-8")
    print(f"Wrote {tag} cache in {time.perf_counter() - started:.1f}s")
    return x_path, y_path


TRAIN_X_PATH, TRAIN_Y_PATH = cache_split(train_df, train_paths, "train")
VAL_X_PATH, VAL_Y_PATH = cache_split(val_df, val_paths, "val")
CLASSES_PATH = CACHE_DIR / "classes.json"
CLASSES_PATH.write_text(json.dumps(classes, indent=2), encoding="utf-8")
print("Image cache ready.")


# Defined here, next to the paths they use, so neither the gate cell nor the
# sweep cell depends on the other having run first.
cache_args = [
    "--train-images", str(TRAIN_X_PATH), "--train-labels", str(TRAIN_Y_PATH),
    "--val-images", str(VAL_X_PATH), "--val-labels", str(VAL_Y_PATH),
    "--classes-json", str(CLASSES_PATH), "--protocol", str(PROTOCOL_PATH),
]


def worker_command(arch, mode, seeds, epochs, output_root):
    return [sys.executable, str(WORKER), "--arch", arch, "--mode", mode,
            "--seeds", *[str(s) for s in seeds], "--epochs", str(epochs),
            "--output-root", str(output_root), *cache_args]
'''

GATE_CODE = '''# Measured on 2x T4 during the pilot; only used to project the schedule.
PILOT_MINUTES = {"smallcnn": 0.67, "resnet18": 0.78, "efficientnet_b0": 2.45, "convnext_tiny": 3.15}

if GATE_ROOT.exists():
    shutil.rmtree(GATE_ROOT)                 # the gate always runs fresh

print("=" * 70)
print("GATE — EfficientNet-B0, seed 42, 2 epochs")
print("=" * 70, flush=True)
gate_started = time.time()
gate = subprocess.run(worker_command("efficientnet_b0", "ddp", [42], GATE_EPOCHS, GATE_ROOT),
                      env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=False)
gate_minutes = (time.time() - gate_started) / 60.0

GATE_PASSED = False
gate_history = None
if gate.returncode != 0:
    print(f"\\nGATE ERROR: the gate run crashed (exit {gate.returncode}).")
    log({"stage": "gate", "status": "crashed", "exit": gate.returncode,
         "minutes": round(gate_minutes, 2)})
else:
    gate_history = pd.read_csv(GATE_ROOT / "efficientnet_b0" / "seed42" / "training_history.csv")
    print("\\n" + gate_history[["epoch", "train_loss", "val_accuracy", "lr"]].to_string(index=False))
    final_accuracy = float(gate_history["val_accuracy"].iloc[-1])
    first_accuracy = float(gate_history["val_accuracy"].iloc[0])
    MINUTES_PER_EPOCH = gate_minutes / GATE_EPOCHS
    GATE_PASSED = final_accuracy >= GATE_THRESHOLD

    print(f"\\nepoch 1 accuracy : {first_accuracy:.4f}  (original recipe: 0.9826)")
    print(f"epoch 2 accuracy : {final_accuracy:.4f}  (threshold: {GATE_THRESHOLD})")
    print(f"throughput       : {MINUTES_PER_EPOCH:.2f} min/epoch")
    log({"stage": "gate", "status": "passed" if GATE_PASSED else "failed",
         "epoch1": round(first_accuracy, 4), "epoch2": round(final_accuracy, 4),
         "minutes_per_epoch": round(MINUTES_PER_EPOCH, 3), "minutes": round(gate_minutes, 2)})

if not GATE_PASSED:
    print("\\n" + "!" * 70)
    print("THE SWEEP IS CANCELLED. Nothing long will run tonight.")
    print("The gate did its job: it cost ~6 minutes instead of ~6 hours.")
    print("The results cell below still runs and still writes a zip containing")
    print("the gate history — that is the evidence needed to diagnose this.")
    print("!" * 70)
else:
    print(f"\\nGATE PASSED. Full sweep released.")
'''

SWEEP_MD = """## 4. The sweep

Runs in priority order: the reference model first, so if anything is cut short the
most important numbers exist. Each run is a separate subprocess with a timeout.
A failed or timed-out run is recorded and the loop moves on — one bad seed never
costs the night.

Finished seeds are skipped, so re-running this notebook after a crash resumes
rather than repeating work.

The last stage adds two extra EfficientNet-B0 seeds, but only if the remaining
budget clearly covers them.
"""

SWEEP_CODE = '''if not GATE_PASSED:
    print("Skipped: the gate did not pass.")
else:
    PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    SEEDS = PROTOCOL["seeds"]
    EPOCHS = PROTOCOL["epochs"]
    SCALE = {a: PILOT_MINUTES[a] / PILOT_MINUTES["efficientnet_b0"] for a in PILOT_MINUTES}

    # Reference model first: it carries H1-H4, so it must survive any cut-off.
    PLAN = [
        ("efficientnet_b0", SEEDS),
        ("convnext_tiny", SEEDS),
        ("resnet18", SEEDS),
        ("smallcnn", SEEDS),
    ]
    if ALLOW_EXTRA_SEEDS:
        PLAN.append(("efficientnet_b0", [45, 46]))

    def remaining_hours():
        return SESSION_BUDGET_H - (time.time() - STARTED_AT) / 3600.0

    def already_done(arch, seed):
        return (OUTPUT_ROOT / arch / f"seed{seed}" / "validation_metrics.json").is_file()

    failures, completed, skipped = [], [], []

    for arch, seeds in PLAN:
        for seed in seeds:
            if already_done(arch, seed):
                skipped.append(f"{arch}/seed{seed}")
                print(f"[skip] {arch} seed {seed} — already complete")
                continue

            estimate = EPOCHS * MINUTES_PER_EPOCH * SCALE[arch]
            budget_left = remaining_hours() - RESERVE_H
            if budget_left * 60 < estimate * 1.25:
                print(f"[cut] {arch} seed {seed} — needs ~{estimate:.0f} min, "
                      f"only {budget_left*60:.0f} min of budget left")
                failures.append({"arch": arch, "seed": seed, "reason": "budget"})
                continue

            timeout = max(1800, int(estimate * 60 * 4))
            print(f"\\n{'='*70}\\n[{arch} seed {seed}] start {time.strftime('%H:%M')} | "
                  f"estimate {estimate:.0f} min | timeout {timeout//60} min | "
                  f"budget left {budget_left:.1f} h\\n{'='*70}", flush=True)
            run_started = time.time()
            try:
                result = subprocess.run(worker_command(arch, "ddp", [seed], EPOCHS, OUTPUT_ROOT),
                                        env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                        check=False, timeout=timeout)
                elapsed = (time.time() - run_started) / 60.0
                if result.returncode == 0:
                    completed.append(f"{arch}/seed{seed}")
                    print(f"[done] {arch} seed {seed} in {elapsed:.1f} min")
                    log({"stage": "run", "arch": arch, "seed": seed, "status": "ok",
                         "minutes": round(elapsed, 2)})
                else:
                    failures.append({"arch": arch, "seed": seed, "reason": f"exit {result.returncode}"})
                    print(f"[FAIL] {arch} seed {seed} exit {result.returncode} after {elapsed:.1f} min")
                    log({"stage": "run", "arch": arch, "seed": seed, "status": "failed",
                         "exit": result.returncode, "minutes": round(elapsed, 2)})
            except subprocess.TimeoutExpired:
                elapsed = (time.time() - run_started) / 60.0
                failures.append({"arch": arch, "seed": seed, "reason": "timeout"})
                print(f"[TIMEOUT] {arch} seed {seed} killed after {elapsed:.1f} min")
                log({"stage": "run", "arch": arch, "seed": seed, "status": "timeout",
                     "minutes": round(elapsed, 2)})
            except Exception as exc:
                failures.append({"arch": arch, "seed": seed, "reason": repr(exc)})
                print(f"[ERROR] {arch} seed {seed}: {exc!r}")
                log({"stage": "run", "arch": arch, "seed": seed, "status": "error",
                     "error": repr(exc)})

    print("\\n" + "=" * 70)
    print(f"completed {len(completed)} | skipped {len(skipped)} | failed {len(failures)}")
    if failures:
        for item in failures:
            print(f"  failed: {item}")
    print(f"session time so far: {(time.time() - STARTED_AT)/3600:.2f} h")
    print("=" * 70)
'''

RESULTS_CODE = '''records = []
for metrics_path in sorted(OUTPUT_ROOT.rglob("validation_metrics.json")):
    parts = metrics_path.relative_to(OUTPUT_ROOT).parts
    if any(skip in parts for skip in ("_smoke", "pilot", "_gate")):
        continue
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    records.append({
        "architecture": metrics.get("model"),
        "seed": int(metrics_path.parent.name.replace("seed", "") or 0),
        "val_accuracy": metrics.get("val_accuracy"),
        "val_macro_f1": metrics.get("val_macro_f1"),
        "val_ece": metrics.get("val_ece"),
        "best_epoch": metrics.get("best_epoch"),
        "train_minutes": metrics.get("train_minutes"),
        "global_batch": metrics.get("global_batch_size"),
        "weight_decay": metrics.get("weight_decay"),
        "fingerprint": (metrics.get("dataset_fingerprint") or "")[:12],
    })

zip_name = "/kaggle/working/overnight_runs"
if records:
    comparison = pd.DataFrame(records).sort_values(["architecture", "seed"]).reset_index(drop=True)
    comparison.to_csv(OUTPUT_ROOT / "comparison.csv", index=False)
    display(comparison)

    print("\\nfingerprints (must agree within an architecture):")
    print(comparison.groupby("architecture")["fingerprint"].apply(lambda c: sorted(set(c))).to_string())

    summary = (comparison.groupby("architecture")["val_macro_f1"]
               .agg(runs="count", mean="mean", std="std", min="min", max="max").round(6)
               .sort_values("mean", ascending=False))
    summary["spread_pp"] = ((summary["max"] - summary["min"]) * 100).round(2)
    print("\\nValidation macro-F1 per architecture")
    display(summary)

    reference = comparison[comparison.architecture == "efficientnet_b0"]
    if not reference.empty:
        print(f"\\nReference model: {len(reference)} seeds, macro-F1 "
              f"{reference.val_macro_f1.mean():.4f} +/- {reference.val_macro_f1.std():.4f} "
              f"(published checkpoint: 0.99369)")

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
else:
    print("No completed runs to report.")
    print("If the gate failed, that is expected: the zip will still contain the gate evidence.")

# Always ship something, even if every run failed.
zip_path = shutil.make_archive(zip_name, "zip", root_dir=OUTPUT_ROOT)
print(f"\\nBundle: {zip_path}  ({Path(zip_path).stat().st_size/1e6:.0f} MB)")
print(f"Session total: {(time.time() - STARTED_AT)/3600:.2f} h")
print("Download it from the Output panel, or fetch it later with:")
print("  kaggle kernels output <user>/<notebook> -p ./out")
'''

MORNING_MD = """## In the morning

1. Download `overnight_runs.zip`.
2. Check the summary printed above against the published reference: the
   EfficientNet-B0 mean macro-F1 should land near **0.9937** at 12 epochs. If it
   is far below that, stop — do not build the comparison table on it.
3. Check `_overnight_log.json` inside the zip for any `failed` or `timeout`
   entries.
4. Read the seed spread. A difference between architectures that is smaller than
   the spread within an architecture is not a result, and must not be written up
   as one.

Nothing here may be quoted in a document until it is registered in
`results/registry.json` with its checkpoint hash and fingerprint.
"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def files_cell(files):
    lines = ["# Writes the engine, worker and protocol, then proves all three are intact.",
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
        "    print(f'wrote {_target.name:20s} {len(_content):7d} bytes')",
        "",
        "import py_compile",
        "for _file in (SCRIPT, WORKER):",
        "    py_compile.compile(str(_file), doraise=True)",
        "print('engine and worker compile')",
        "",
        "PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))",
        "assert PROTOCOL['version'] == 2, PROTOCOL['version']",
        "assert PROTOCOL['shuffle_mode'] == 'random', ('blocked shuffling breaks BatchNorm: see R14', PROTOCOL['shuffle_mode'])",
        "assert PROTOCOL['warmup_fraction'] == 0.0 and PROTOCOL['min_lr_factor'] == 1.0, 'v2 must hold the LR constant'",
        "assert PROTOCOL['label_smoothing'] == 0.0 and PROTOCOL['class_weight_alpha'] == 0.0, 'baseline arm uses no smoothing or class weights'",
        "b0 = PROTOCOL['architectures']['efficientnet_b0']",
        "assert b0['lr'] == 1e-4 and b0['weight_decay'] == 1e-4 and b0['local_batch_size'] == 16, b0",
        "print(f\"protocol v{PROTOCOL['version']} | shuffle={PROTOCOL['shuffle_mode']} | flat LR | \"",
        "      f\"b0 lr={b0['lr']} wd={b0['weight_decay']} batch/gpu={b0['local_batch_size']}\")",
        "print('protocol assertions passed')",
    ]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]}


def build() -> dict:
    files = [(target, (ROOT / source).read_text(encoding="utf-8")) for target, source in EMBEDDED]
    cells = [
        md(HEADER),
        md("## 1. Setup and preflight\n\nEverything that can be checked in ten seconds is checked here: two GPUs present, dataset attached, manifests agree, images resolve, disk space for the cache."),
        code(SETUP),
        md("## 2. Image cache\n\nBuilt once and shared by all fourteen runs. Any image that fails to decode aborts the cache build instead of silently training on a black square."),
        code(CACHE_CODE),
        md("## 3. Engine, worker and protocol\n\nWritten from source, then asserted: shuffle mode, flat learning rate, no label smoothing, no class weights, and the exact reference-model settings. If any of that is wrong the notebook stops now rather than at hour six."),
        files_cell(files),
        md("## 4. Gate"),
        code(GATE_CODE),
        md(SWEEP_MD),
        code(SWEEP_CODE),
        md("## 5. Results and bundle"),
        code(RESULTS_CODE),
        md(MORNING_MD),
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
    problems = []
    for index, cell in enumerate(reloaded["cells"]):
        if cell["cell_type"] == "code":
            try:
                compile("".join(cell["source"]), f"<cell {index}>", "exec")
            except SyntaxError as exc:
                problems.append(f"cell {index}: {exc}")
    if problems:
        raise SystemExit("generated notebook does not compile:\n" + "\n".join(problems))

    embedded = None
    for cell in reloaded["cells"]:
        text = "".join(cell["source"])
        if "FILES = {" in text:
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

    sandbox = Path(tempfile.mkdtemp(prefix="overnightcheck_"))
    for target, content in embedded.items():
        destination = sandbox / target.lstrip("/").replace("kaggle/working/", "")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        if destination.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Round-trip write failed for {target}")

    proto = json.loads(embedded["/kaggle/working/protocol_v2.json"])
    assert proto["shuffle_mode"] == "random", proto["shuffle_mode"]
    assert proto["warmup_fraction"] == 0.0 and proto["min_lr_factor"] == 1.0
    assert proto["architectures"]["efficientnet_b0"]["lr"] == 1e-4

    print(f"Wrote {NOTEBOOK.relative_to(ROOT)} ({len(reloaded['cells'])} cells, all compile)")
    print(f"Verified {len(embedded)} embedded files byte for byte; protocol assertions hold.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the protocol-v2 gate notebook (notebooks/07).

Two-epoch gate first, sweep second, in one notebook. The gate exists because a
five-and-a-half-hour sweep was once allowed to run on an unvalidated recipe and
came back invalid: see reports/model_comparison/SWEEP_V1_INVALID.md.

    python scripts/build_gate_notebook.py
"""
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "07_Protocol_v2_Gate.ipynb"

EMBEDDED = [
    ("/kaggle/working/train_ddp.py", "scripts/train_ddp.py"),
    ("/kaggle/working/sweep_worker.py", "scripts/sweep_worker.py"),
    ("/kaggle/working/protocol_v2.json", "configs/protocol_v2.json"),
]

HEADER = """# Protocol v2 — gate first, sweep second

Runs the fixed recipe for two epochs, checks it against a hard threshold, and
only then starts the twelve-run sweep.

## Why this notebook exists

The first sweep ran twelve trainings over 5.6 GPU-hours and none of it was
usable. EfficientNet-B0 came back at 0.6815 against a reference of 0.9954. The
cause was the recipe, not the data: whole-model distance from pristine ImageNet
weights was 0.49% for the broken run against 1.96% for the reference, meaning
the backbone never fine-tuned. Full evidence in
`reports/model_comparison/SWEEP_V1_INVALID.md`.

The mistake was not the wrong number. The mistake was freezing a protocol and
committing hours to it without a cheap check that it could reproduce the result
it was supposed to reproduce.

## What changed

| Setting | v1 (invalid) | v2 (restored) | Source of the v2 value |
|---|---|---|---|
| weight decay | 0.05 | **1e-4** | original B0 script |
| global batch | 128 | **32** | original B0 script |
| LR schedule | warmup + cosine to 0 | **constant** | original B0 script |
| label smoothing | 0.05 | **0** | original B0 script used none |
| class weights | alpha 0.5 | **0** | original B0 script used none |

SmallCNN keeps its own proven settings (lr 1e-3, weight decay 1e-4, global batch
64) from `reports/phase2_baseline/experiment_config.json`, which produced 96.01%.
ResNet18 and ConvNeXt-Tiny inherit the validated B0 recipe, because no original
configuration exists for them and inventing one is how v1 went wrong.

Class weighting and label smoothing are **not** in this baseline. They belong to
the H3 treatment arm, where they get introduced one variable at a time.

## The gate

One EfficientNet-B0 run, two epochs, seed 42:

**pass if validation accuracy at epoch 2 is at least 0.95**

The original reached 0.9826 in epoch 1, so this is loose on purpose. It is
tuned to catch a configuration that cannot learn, not one that is slightly
weaker than the old best.

If the gate fails, nothing else runs. That is the entire point: a 6-minute run
replaces a 5.6-hour one.

## Then

`MODE = "sweep"` re-runs the gate (it is cheap) and continues into all twelve
runs. Each stage is selectable, and a seed whose metrics already exist is
skipped, so an interrupted session resumes instead of restarting.
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
MODE = "gate"              # "gate" = two-epoch check only | "sweep" = gate then 12 runs
ONLY_STAGE = None          # None | "efficientnet_b0" | "convnext_tiny" | "resnet18" | "smallcnn"

GATE_ARCH = "efficientnet_b0"
GATE_SEED = 42
GATE_EPOCHS = 2
GATE_THRESHOLD = 0.95      # validation accuracy required at the final gate epoch

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
print(f"MODE={MODE!r} | gate: {GATE_ARCH} seed {GATE_SEED}, {GATE_EPOCHS} epochs, pass at >={GATE_THRESHOLD}")
print("Locked test manifest: not opened.")
'''

CACHE_MD = """## 2. Image cache

Built once per session and shared by every run in it. The key is a hash of the path list, so a changed split invalidates the cache automatically.
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

WRITE_MD = """## 3. Write the engine, the worker and protocol v2

One cell, three files, each parsed after writing. The protocol is the repaired recipe from `configs/protocol_v2.json` in the repository, embedded here byte for byte.
"""

GATE_MD = """## 4. The gate

One run: EfficientNet-B0, seed 42, two epochs. Output goes to `runs/_gate` so it can never be mistaken for a sweep result.

The threshold is 0.95 validation accuracy at epoch 2. For reference, the original run of this recipe reached **0.9826 after one epoch**, so a healthy configuration clears this comfortably. A configuration like v1 sat at **0.1226** after one epoch and needed twelve to crawl to 0.68.

If this cell raises, do not run the sweep. Send me the history table and I will find what else differs.
"""

GATE_CODE = '''# Measured on 2x T4 during the pilot; used only to project the sweep afterwards.
PILOT_MINUTES = {"smallcnn": 0.67, "resnet18": 0.78, "efficientnet_b0": 2.45, "convnext_tiny": 3.15}

cache_args = [
    "--train-images", str(TRAIN_X_PATH), "--train-labels", str(TRAIN_Y_PATH),
    "--val-images", str(VAL_X_PATH), "--val-labels", str(VAL_Y_PATH),
    "--classes-json", str(CLASSES_PATH), "--protocol", str(PROTOCOL_PATH),
]


def worker_command(arch, mode, seeds, epochs, output_root):
    return [sys.executable, str(WORKER), "--arch", arch, "--mode", mode,
            "--seeds", *[str(s) for s in seeds], "--epochs", str(epochs),
            "--output-root", str(output_root), *cache_args]


if GATE_ROOT.exists():
    shutil.rmtree(GATE_ROOT)          # the gate always runs fresh, never resumes
print(f"running gate: {GATE_ARCH} seed {GATE_SEED}, {GATE_EPOCHS} epochs\\n", flush=True)

started = time.time()
result = subprocess.run(worker_command(GATE_ARCH, "ddp", [GATE_SEED], GATE_EPOCHS, GATE_ROOT),
                        env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=False)
elapsed_min = (time.time() - started) / 60.0
if result.returncode != 0:
    raise RuntimeError(f"the gate run itself failed (exit {result.returncode}) before producing a verdict")

gate_dir = GATE_ROOT / GATE_ARCH / f"seed{GATE_SEED}"
history = pd.read_csv(gate_dir / "training_history.csv")
print("\\n" + history[["epoch", "train_loss", "val_accuracy", "val_macro_f1"]].to_string(index=False))

final_accuracy = float(history["val_accuracy"].iloc[-1])
first_accuracy = float(history["val_accuracy"].iloc[0])
minutes_per_epoch = elapsed_min / GATE_EPOCHS

print(f"\\nepoch 1 accuracy : {first_accuracy:.4f}   (original recipe: 0.9826)")
print(f"epoch {GATE_EPOCHS} accuracy : {final_accuracy:.4f}   (threshold: {GATE_THRESHOLD})")
print(f"wall clock       : {elapsed_min:.1f} min for {GATE_EPOCHS} epochs = {minutes_per_epoch:.1f} min/epoch")

if final_accuracy < GATE_THRESHOLD:
    print("\\n" + "=" * 70)
    print("GATE FAILED — do not run the sweep.")
    print(f"  expected >= {GATE_THRESHOLD} by epoch {GATE_EPOCHS}, got {final_accuracy:.4f}")
    print("  v1 of this protocol sat at 0.12 after epoch 1 and reached only 0.68 in twelve.")
    print("  A working recipe is at 0.98 after ONE epoch. This is a configuration fault,")
    print("  not a slow start. Nothing else in this notebook should run.")
    print("=" * 70)
    raise RuntimeError(f"gate failed: {final_accuracy:.4f} < {GATE_THRESHOLD}")

scale = {arch: PILOT_MINUTES[arch] / PILOT_MINUTES["efficientnet_b0"] for arch in PILOT_MINUTES}
protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
seeds = protocol["seeds"]
projected = sum(protocol["epochs"] * minutes_per_epoch * scale[arch] * len(seeds)
                for arch in protocol["architectures"])

print("\\n" + "=" * 70)
print(f"GATE PASSED at {final_accuracy:.4f}")
print(f"projected full sweep at this throughput: {projected/60:.1f} h "
      f"({projected:.0f} min for {len(protocol['architectures'])} architectures x {len(seeds)} seeds)")
print("=" * 70)
'''

SWEEP_MD = """## 5. The sweep

Skipped unless `MODE = "sweep"`. Four stages in order, each on both GPUs through DDP. A seed whose metrics already exist is skipped, so a restart costs only the seeds that did not finish.

Set `ONLY_STAGE` to run a single architecture.
"""

SWEEP_CODE = '''if MODE != "sweep":
    print('MODE is "gate": the sweep is intentionally not running.')
    print("Read the gate verdict above. If it passed and the throughput is acceptable,")
    print('set MODE = "sweep" in cell 1 and run cells 1, 3, 5, this cell and the results cell.')
else:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    seeds = protocol["seeds"]
    SCHEDULE = [
        ("efficientnet_b0", "efficientnet_b0"),
        ("convnext_tiny", "convnext_tiny"),
        ("resnet18", "resnet18"),
        ("smallcnn", "smallcnn"),
    ]
    started_all = time.time()
    for stage_name, arch in SCHEDULE:
        if ONLY_STAGE and stage_name != ONLY_STAGE:
            continue
        print(f"\\n===== {arch} : {len(seeds)} seeds =====", flush=True)
        result = subprocess.run(worker_command(arch, "ddp", seeds, protocol["epochs"], OUTPUT_ROOT),
                                env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"{arch} failed with code {result.returncode}")
    print(f"\\nAll requested stages finished in {(time.time() - started_all)/60:.1f} min")
'''

RESULTS_MD = """## 6. Results and one bundle

Aggregates every sweep run, reports macro-F1 per architecture as mean plus spread across seeds, and zips `runs/`.

Three seeds exist so the spread is visible. A difference smaller than that spread is not a result, and no number here may be quoted until it is registered in `results/registry.json`.
"""

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
        "fingerprint": (metrics.get("dataset_fingerprint") or "")[:12],
    })

if not records:
    print("No sweep runs on disk yet.")
    print('Gate-only session: that is expected. Set MODE = "sweep" when you are ready.')
else:
    comparison = pd.DataFrame(records).sort_values(["architecture", "seed"]).reset_index(drop=True)
    comparison.to_csv(OUTPUT_ROOT / "comparison.csv", index=False)
    display(comparison)

    print("\\nfingerprints (all runs of one architecture must agree):")
    print(comparison.groupby("architecture")["fingerprint"].apply(lambda c: sorted(set(c))).to_string())

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

    label = "gate" if MODE == "gate" else (ONLY_STAGE or "all")
    zip_path = shutil.make_archive(f"/kaggle/working/protocol_v2_{label}", "zip", root_dir=OUTPUT_ROOT)
    print(f"\\nDownload from the Kaggle Output panel: {zip_path}")
    print(f"Size: {Path(zip_path).stat().st_size/1e6:.0f} MB")
'''

NEXT_MD = """## After the gate

If it passed, one sentence is enough to record:

> Protocol v2 reached **X** validation accuracy after **N** epochs, against 0.9826 for the original run at the same point; the sweep was then released.

If it failed, stop. The pattern is diagnostic: near-chance accuracy that climbs slowly means the backbone still is not training, and the next suspects are the model construction path and the augmentation pipeline, not the learning rate.

Either way the gate result goes into `docs/KNOWN_ISSUES.md` under R12 and into the experiment log, because a validation step that catches a broken configuration is a result worth recording.
"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def files_cell(files):
    lines = ["# Writes the engine, the worker and protocol v2, then checks all three.",
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
        "    print(f'{_file.name} compiles')",
        "",
        "PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))",
        "assert PROTOCOL['version'] == 2, PROTOCOL['version']",
        "assert PROTOCOL['weight_decay' if 'weight_decay' in PROTOCOL else 'epochs']",
        "flat = PROTOCOL['warmup_fraction'] == 0.0 and PROTOCOL['min_lr_factor'] == 1.0",
        "print('protocol version :', PROTOCOL['version'])",
        "print('constant LR      :', flat)",
        "print('architectures    :', list(PROTOCOL['architectures']))",
        "for _arch, _cfg in PROTOCOL['architectures'].items():",
        "    print(f\"  {_arch:16s} lr={_cfg['lr']:<8} wd={_cfg['weight_decay']:<8} \"",
        "          f\"global_batch={_cfg['local_batch_size'] * 2}\")",
        "if not flat:",
        "    print('WARNING: this protocol decays the learning rate; v2 is meant to hold it constant.')",
    ]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]}


def build() -> dict:
    files = [(target, (ROOT / source).read_text(encoding="utf-8")) for target, source in EMBEDDED]
    cells = [
        md(HEADER),
        md("## 1. Paths, mode and the gate rule"),
        code(PREPARE),
        md(CACHE_MD),
        code(CACHE_CODE),
        md(WRITE_MD),
        files_cell(files),
        md(GATE_MD),
        code(GATE_CODE),
        md(SWEEP_MD),
        code(SWEEP_CODE),
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

    sandbox = Path(tempfile.mkdtemp(prefix="gatecheck_"))
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

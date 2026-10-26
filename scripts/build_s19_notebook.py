#!/usr/bin/env python3
"""Generate the S19 background-augmentation notebook (notebooks/09).

Six training runs: three baseline, three with leaves composited onto field
backgrounds. Same seeds, same protocol, same evaluator, so the comparison is
paired and only the background differs.

    python scripts/build_s19_notebook.py
"""
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "09_Background_Augmentation.ipynb"

EMBEDDED = [
    ("/kaggle/working/train_ddp.py", "scripts/train_ddp.py"),
    ("/kaggle/working/sweep_worker.py", "scripts/sweep_worker.py"),
    ("/kaggle/working/protocol_v2.json", "configs/protocol_v2.json"),
    ("/kaggle/working/make_leaf_masks.py", "scripts/make_leaf_masks.py"),
    ("/kaggle/working/build_composite_cache.py", "scripts/build_composite_cache.py"),
    ("/kaggle/working/check_background_disjointness.py", "scripts/check_background_disjointness.py"),
    ("/kaggle/working/evaluate_plantdoc.py", "scripts/evaluate_plantdoc.py"),
    ("/kaggle/working/paired_bootstrap.py", "scripts/paired_bootstrap.py"),
]

HEADER = """# S19: background augmentation, paired comparison

The treatment arm for H3. Does replacing the laboratory backdrop with field
backgrounds reduce the drop from PlantVillage to PlantDoc?

## Design

Six runs, three paired seeds. Everything is held constant except one thing.

| arm | training images | everything else |
|---|---|---|
| baseline | the existing clean cache | protocol v2, unchanged |
| composite | same leaves on random field backgrounds | protocol v2, unchanged |

Validation is **never** composited, so the PlantVillage score stays comparable to
the baseline. PlantDoc is untouched in both arms.

The pairwise comparison is done with `paired_bootstrap.py`, which resamples the
same field images for both models, so a seed that happens to see an easy 236
images cannot flatter the treatment.

## What this can and cannot show

Backgrounds come from DeepWeeds (CC BY 4.0), which is a different continent and a
different set of species from both PlantVillage and PlantDoc. Disjointness is
checked by hash before any training starts, and the notebook stops if that check
fails.

Two limits are built into the method and belong in the report:

**One random background per image, fixed for the run.** Re-sampling every epoch
would be stronger augmentation but needs the training engine to hold a background
pool and per-image masks during the loop. With a fixed random assignment the
background-to-class mapping is already broken, since each class receives random
backdrops. What is lost is per-image background diversity.

**Segmentation is a heuristic.** Corner-colour plus greenness fails on close-up
leaf textures, where the frame is almost entirely leaf. Masks outside a
foreground fraction of 0.25 to 0.90 are not trusted, and those images pass through
unaltered rather than being composited wrongly. On the validation split that was
2.7% of images.

## Cost

Roughly 2.5 hours of T4 time for six runs. Set `SEEDS = [42]` for a one-pair
smoke version at about 50 minutes.
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

SEEDS = [42, 43, 44]          # reduce to [42] for a one-pair smoke run
EPOCHS = 12
GATE_ACCURACY = 0.95

DATASET_ROOT = Path("/kaggle/input/datasets/shubhamkumar12864/plantvillage")
KAGGLE_INPUT = Path("/kaggle/input")
BACKGROUND_ROOT = Path("/kaggle/input/deepweeds")      # attach DeepWeeds here


def autodiscover_dataset_root(preferred: Path) -> Path:
    if (preferred / "data" / "manifests" / "plantvillage_development.csv").is_file():
        return preferred
    if KAGGLE_INPUT.is_dir():
        for manifests in sorted(KAGGLE_INPUT.rglob("manifests")):
            if (manifests / "plantvillage_development.csv").is_file():
                return manifests.parent.parent
    return preferred


DATASET_ROOT = autodiscover_dataset_root(DATASET_ROOT)
MANIFEST_DIR = DATASET_ROOT / "data" / "manifests"
IMAGE_ROOT = DATASET_ROOT / "data" / "raw" / "PlantVillage"
TRAIN_MANIFEST = MANIFEST_DIR / "plantvillage_development.csv"

WORKDIR = Path("/kaggle/working")
SCRIPT = WORKDIR / "train_ddp.py"
WORKER = WORKDIR / "sweep_worker.py"
PROTOCOL_PATH = WORKDIR / "protocol_v2.json"
MASK_SCRIPT = WORKDIR / "make_leaf_masks.py"
COMPOSITE_SCRIPT = WORKDIR / "build_composite_cache.py"
DISJOINT_SCRIPT = WORKDIR / "check_background_disjointness.py"
EVAL_SCRIPT = WORKDIR / "evaluate_plantdoc.py"
BOOTSTRAP_SCRIPT = WORKDIR / "paired_bootstrap.py"

CACHE_DIR = Path("/kaggle/temp/plantvillage_cache")
DERIVED = Path("/kaggle/temp/derived")
OUTPUT_ROOT = WORKDIR / "runs_bg"
IMAGE_SIZE = 224
STARTED_AT = time.time()

for folder in (CACHE_DIR, DERIVED, OUTPUT_ROOT):
    folder.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("Set Accelerator -> GPU T4 x2 in Kaggle Settings, then Run All again.")
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(2)])

if not TRAIN_MANIFEST.is_file():
    raise FileNotFoundError(f"PlantVillage not attached: expected {TRAIN_MANIFEST}")
if not BACKGROUND_ROOT.is_dir():
    raise FileNotFoundError(
        f"Background corpus not attached at {BACKGROUND_ROOT}.\\n"
        "S19 cannot run without a licence-clean corpus disjoint from PlantDoc. "
        "See docs/BACKGROUND_CORPUS_OPTIONS.md."
    )

dev = pd.read_csv(TRAIN_MANIFEST)
train_df = dev.loc[dev.split == "train"].reset_index(drop=True)
val_df = dev.loc[dev.split == "validation"].reset_index(drop=True)
classes = sorted(train_df.label.unique().tolist())
if len(classes) != 38:
    raise ValueError(f"expected 38 classes, found {len(classes)}")
print(f"PlantVillage: {len(train_df):,} train / {len(val_df):,} validation")
print(f"background corpus: {BACKGROUND_ROOT}")
print(f"seeds: {SEEDS}  |  epochs: {EPOCHS}  |  estimated "
      f"{len(SEEDS) * 2 * EPOCHS * 2.2 / 60:.1f} h")
'''

DISJOINT_CODE = '''print("=" * 68)
print("Disjointness gate: if a background came from PlantDoc, the whole")
print("comparison is void, so this runs before anything trains.")
print("=" * 68)

disjoint_out = DERIVED / "disjointness.json"
check = subprocess.run([sys.executable, str(DISJOINT_SCRIPT),
                        "--backgrounds", str(BACKGROUND_ROOT),
                        "--out", str(disjoint_out)],
                       env={**os.environ, "PYTHONUNBUFFERED": "1"})
if check.returncode != 0:
    raise RuntimeError(
        "DEDISJOINTNESS FAILED: a background image is byte-identical to a PlantDoc or "
        "PlantVillage image. Training on it would put the evaluation set into the training "
        "data. See " + str(disjoint_out)
)
DISJOINT = json.loads(disjoint_out.read_text())
print(f"\\nchecked {DISJOINT['candidate_images']:,} backgrounds against "
      f"{DISJOINT['plantdoc_hashes']:,} PlantDoc and {DISJOINT['plantvillage_hashes']:,} "
      f"PlantVillage hashes: {DISJOINT['verdict']}")
print("limitation:", DISJOINT["limitation"])
'''

CACHE_CODE = '''def cache_split(frame, image_root, tag):
    labels = np.asarray([classes.index(label) for label in frame.label], dtype=np.int64)
    paths = [image_root / rel for rel in frame.relative_path]
    path_key = hashlib.sha256("\\n".join(map(str, paths)).encode()).hexdigest()[:16]
    stem = CACHE_DIR / f"{tag}_{IMAGE_SIZE}_{len(paths)}_{path_key}"
    x_path, y_path, done = Path(f"{stem}_x.npy"), Path(f"{stem}_y.npy"), Path(f"{stem}.done")
    shape = (len(paths), IMAGE_SIZE, IMAGE_SIZE, 3)

    if done.is_file() and x_path.is_file() and y_path.is_file():
        existing = np.load(x_path, mmap_mode="r", allow_pickle=False)
        if existing.shape == shape and np.array_equal(np.load(y_path, allow_pickle=False), labels):
            print(f"reused {tag} cache")
            return x_path, y_path
        del existing

    array = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.uint8, shape=shape)
    failures = []

    def decode(item):
        i, path = item
        try:
            with Image.open(path) as image:
                array[i] = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE),
                                                                  Image.Resampling.BILINEAR))
        except Exception as exc:
            failures.append((str(path), repr(exc)))
        return i

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(tqdm(pool.map(decode, enumerate(paths)), total=len(paths), desc=tag, unit="img"))
    array.flush(); del array
    if failures:
        raise RuntimeError(f"{len(failures)} images failed to decode, first: {failures[0]}")
    np.save(y_path, labels, allow_pickle=False)
    done.write_text("complete")
    return x_path, y_path


TRAIN_X_PATH, TRAIN_Y_PATH = cache_split(train_df, IMAGE_ROOT / "train", "train")
VAL_X_PATH, VAL_Y_PATH = cache_split(val_df, IMAGE_ROOT / "train", "val")
CLASSES_PATH = CACHE_DIR / "classes.json"
CLASSES_PATH.write_text(json.dumps(classes, indent=2))
print("clean caches ready")
'''

COMPOSITE_CODE = '''MASKS_PATH = DERIVED / "masks_train.npz"
COMPOSITE_X = CACHE_DIR / "train_composite_x.npy"
COMPOSITE_Y = TRAIN_Y_PATH

if not MASKS_PATH.is_file():
    print("building leaf masks for the training split (~4 minutes)")
    subprocess.run([sys.executable, str(MASK_SCRIPT), "--split", "train",
                    "--manifest", str(TRAIN_MANIFEST), "--image-root", str(IMAGE_ROOT / "train"),
                    "--out", str(MASKS_PATH),
                    "--check-sheet", str(DERIVED / "mask_check.png")], check=True)

if not COMPOSITE_X.is_file():
    print("compositing leaves onto field backgrounds")
    subprocess.run([sys.executable, str(COMPOSITE_SCRIPT),
                    "--backgrounds", str(BACKGROUND_ROOT), "--masks", str(MASKS_PATH),
                    "--train-cache", str(TRAIN_X_PATH), "--out", str(COMPOSITE_X),
                    "--check-sheet", str(DERIVED / "composite_check.png")], check=True)

COMPOSITE_SUMMARY = json.loads(Path(str(COMPOSITE_X) + ".json").read_text())
print(f"composited {COMPOSITE_SUMMARY['images_composited']:,} images, "
      f"{COMPOSITE_SUMMARY['images_passed_through']:,} passed through "
      f"(mask outside {COMPOSITE_SUMMARY['mask_trust_band']})")
'''

FILES_CELL_TEMPLATE = "FILES = {"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


TRAIN_CODE = '''cache_args = [
    "--train-labels", str(TRAIN_Y_PATH),
    "--val-images", str(VAL_X_PATH), "--val-labels", str(VAL_Y_PATH),
    "--classes-json", str(CLASSES_PATH), "--protocol", str(PROTOCOL_PATH),
]


def worker(arch, seeds, epochs, output_root, train_images):
    return [sys.executable, str(WORKER), "--arch", arch, "--mode", "ddp",
            "--seeds", *[str(s) for s in seeds], "--epochs", str(epochs),
            "--train-images", str(train_images), "--output-root", str(output_root), *cache_args]


ARMS = {
    "baseline":  {"train_images": TRAIN_X_PATH,     "root": OUTPUT_ROOT / "baseline"},
    "composite": {"train_images": COMPOSITE_X,      "root": OUTPUT_ROOT / "composite"},
}

print("GATE: baseline seed 42, 2 epochs, must reach "
      f"{GATE_ACCURACY} before six runs are committed")
gate = subprocess.run(worker("efficientnet_b0", [SEEDS[0]], 2, ARMS["baseline"]["root"],
                             ARMS["baseline"]["train_images"]),
                      env={**os.environ, "PYTHONUNBUFFERED": "1"})
gate_history = pd.read_csv(ARMS["baseline"]["root"] / "efficientnet_b0" / f"seed{SEEDS[0]}"
                           / "training_history.csv")
accuracy = float(gate_history["val_accuracy"].iloc[-1])
print(gate_history[["epoch", "train_loss", "val_accuracy", "lr"]].to_string(index=False))
print(f"gate accuracy after 2 epochs: {accuracy:.4f}")

GATE_PASSED = gate.returncode == 0 and accuracy >= GATE_ACCURACY
if not GATE_PASSED:
    print("GATE FAILED - no long runs tonight. The gate evidence is still in the zip below.")

if GATE_PASSED:
    for arm, spec in ARMS.items():
        print(f"\\n===== {arm} : {len(SEEDS)} seeds =====", flush=True)
        result = subprocess.run(worker("efficientnet_b0", SEEDS, EPOCHS, spec["root"],
                                       spec["train_images"]),
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        if result.returncode != 0:
            print(f"  {arm} arm returned {result.returncode}, continuing so partial results survive")
    print(f"\\ntraining finished in {(time.time() - STARTED_AT)/3600:.2f} h")
'''

EVAL_CODE = '''if not GATE_PASSED:
    print("skipped: gate did not pass")
else:
    EVALS = {}
    for arm, spec in ARMS.items():
        for seed in SEEDS:
            checkpoint = spec["root"] / "efficientnet_b0" / f"seed{seed}" / "efficientnet_b0_best.pt"
            out_dir = OUTPUT_ROOT / "field" / arm / f"seed{seed}"
            if not checkpoint.is_file():
                print(f"  missing checkpoint: {checkpoint}")
                continue
            if (out_dir / "predictions.csv").is_file():
                print(f"  {arm} seed {seed}: cached")
            else:
                subprocess.run([sys.executable, str(EVAL_SCRIPT), "--checkpoint", str(checkpoint),
                                "--arch", "efficientnet_b0", "--split", "test", "--device", "cuda",
                                "--bootstrap", "1000", "--out", str(out_dir)], check=False)
            metrics_path = out_dir / "field_metrics.json"
            if metrics_path.is_file():
                m = json.loads(metrics_path.read_text())
                EVALS[(arm, seed)] = {"field_accuracy": m["accuracy"],
                                      "predictions": out_dir / "predictions.csv"}
                print(f"  {arm:10s} seed {seed}  field accuracy {m['accuracy']:.4f}")
'''

COMPARE_CODE = '''if not GATE_PASSED or not EVALS:
    print("nothing to compare yet")
else:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    results = []
    for seed in SEEDS:
        if ("baseline", seed) not in EVALS or ("composite", seed) not in EVALS:
            continue
        base_metrics = json.loads((OUTPUT_ROOT / "field" / "baseline" / f"seed{seed}"
                                   / "field_metrics.json").read_text())
        treat_metrics = json.loads((OUTPUT_ROOT / "field" / "composite" / f"seed{seed}"
                                    / "field_metrics.json").read_text())
        base_lab = json.loads((ARMS["baseline"]["root"] / "efficientnet_b0" / f"seed{seed}"
                               / "validation_metrics.json").read_text())["val_accuracy"]
        treat_lab = json.loads((ARMS["composite"]["root"] / "efficientnet_b0" / f"seed{seed}"
                                / "validation_metrics.json").read_text())["val_accuracy"]

        out_json = OUTPUT_ROOT / "paired" / f"seed{seed}.json"
        subprocess.run([sys.executable, str(BOOTSTRAP_SCRIPT),
                        "--baseline", str(EVALS[("baseline", seed)]["predictions"]),
                        "--treatment", str(EVALS[("composite", seed)]["predictions"]),
                        "--baseline-lab", str(base_lab), "--treatment-lab", str(treat_lab),
                        "--out", str(out_json)], check=False)
        if out_json.is_file():
            row = json.loads(out_json.read_text())
            row["seed"] = seed
            row["baseline_lab_accuracy"] = base_lab
            row["treatment_lab_accuracy"] = treat_lab
            results.append(row)

    if not results:
        print("no paired seed completed")
    else:
        table = pd.DataFrame([{
            "seed": r["seed"],
            "lab baseline": round(r["baseline_lab_accuracy"], 4),
            "lab composite": round(r["treatment_lab_accuracy"], 4),
            "field baseline": round(r["baseline_field_accuracy"], 4),
            "field composite": round(r["treatment_field_accuracy"], 4),
            "drop baseline pp": round(r["baseline_drop_pp"], 1),
            "drop composite pp": round(r["treatment_drop_pp"], 1),
            "drop improvement pp": r["drop_improvement_pp"],
            "CI": r["drop_improvement_ci95_pp"],
        } for r in results])
        display(table)
        (OUTPUT_ROOT / "paired_summary.csv").write_text(table.to_csv(index=False))

        supported = sum(1 for r in results if r["verdict"].startswith("H3 supported"))
        print(f"\\nseeds where the composite arm reduced the drop: {supported}/{len(results)}")
        for r in results:
            print(f"  seed {r['seed']}: {r['verdict']}")

        label = "s19" if len(SEEDS) > 1 else "s19_smoke"
        zip_path = shutil.make_archive(f"/kaggle/working/{label}_results", "zip",
                                       root_dir=OUTPUT_ROOT)
        print(f"\\nBundle: {zip_path} ({Path(zip_path).stat().st_size/1e6:.0f} MB)")
'''

NEXT_MD = """## Reading the result

A positive result means the composite arm's drop is smaller and the confidence
interval on the improvement excludes zero. Report it as "recovered X points of the
regression, 95% CI [a, b]", never as "solved".

A null result is equally publishable and pairs naturally with the calibration
finding: if randomising the background does not help, the gap is about leaf
appearance across domains rather than the scene around the leaf.

Either way, the result goes into `results/registry.json` with this run's artifact
before it appears in any document. And the two method limits at the top of this
notebook travel with the number.
"""


def build() -> dict:
    files = [(target, (ROOT / source).read_text(encoding="utf-8")) for target, source in EMBEDDED]
    lines = ["# Writes every script this notebook uses, then checks them.",
             FILES_CELL_TEMPLATE]
    for target, content in files:
        lines.append(f"    {target!r}: {content!r},")
    lines += [
        "}",
        "",
        "for _path, _content in FILES.items():",
        "    _target = Path(_path)",
        "    _target.parent.mkdir(parents=True, exist_ok=True)",
        "    _target.write_text(_content, encoding='utf-8')",
        "print(f'wrote {len(FILES)} files')",
        "",
        "import py_compile",
        "for _file in (SCRIPT, WORKER, MASK_SCRIPT, COMPOSITE_SCRIPT, DISJOINT_SCRIPT, EVAL_SCRIPT, BOOTSTRAP_SCRIPT):",
        "    py_compile.compile(str(_file), doraise=True)",
        "print('all scripts compile')",
        "",
        "PROTOCOL = json.loads(PROTOCOL_PATH.read_text())",
        "assert PROTOCOL['version'] == 2 and PROTOCOL['shuffle_mode'] == 'random'",
        "b0 = PROTOCOL['architectures']['efficientnet_b0']",
        "assert b0['lr'] == 1e-4 and b0['local_batch_size'] == 16, b0",
        "print('protocol v2 unchanged:', b0)",
    ]
    files_cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]}

    cells = [
        md(HEADER),
        md("## 1. Setup and preflight\n\nFails early and loudly if the GPUs, the dataset or the background corpus are missing."),
        code(SETUP),
        md("## 2. Disjointness gate\n\nA hash check, not an assertion. If any background is byte-identical to a PlantDoc or PlantVillage image, training never starts."),
        code(DISJOINT_CODE),
        md("## 3. Caches: clean, masks, composite\n\nThe validation cache is built once and never composited."),
        code(CACHE_CODE),
        code(COMPOSITE_CODE),
        md("## 4. Scripts and protocol"),
        files_cell,
        md("## 5. Gate, then six runs\n\nBaseline and composite arms, same seeds, same protocol. A failure in one arm does not stop the other, so partial results survive."),
        code(TRAIN_CODE),
        md("## 6. Field evaluation"),
        code(EVAL_CODE),
        md("## 7. Paired comparison"),
        code(COMPARE_CODE),
        md(NEXT_MD),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5,
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
        if FILES_CELL_TEMPLATE in text:
            start = text.index(FILES_CELL_TEMPLATE) + len("FILES = ")
            depth, end = 0, None
            for index, char in enumerate(text[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = index + 1
                        break
            embedded = ast.literal_eval(text[start:end])
            break
    if embedded is None:
        raise SystemExit("notebook has no FILES cell")
    for target, source in EMBEDDED:
        if embedded.get(target) != (ROOT / source).read_text(encoding="utf-8"):
            raise SystemExit(f"embedded copy of {source} does not match")

    print(f"Wrote {NOTEBOOK.relative_to(ROOT)} ({len(reloaded['cells'])} cells, all compile)")
    print(f"Verified {len(embedded)} embedded files byte for byte.")


if __name__ == "__main__":
    main()

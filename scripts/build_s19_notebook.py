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

HEADER = """# S19 — background augmentation, paired comparison (H3)

This notebook answers the last open question in the project: **if you train the
model on leaves sitting in real field backgrounds instead of laboratory backdrops,
does the accuracy loss on field images get smaller?**

Everything else in the project is finished. H1 is confirmed (field accuracy is 78
points below laboratory accuracy), H2 is partly supported (removing the background
costs 56.5 points in the lab), H4 is confirmed (a confidence threshold does not
rescue a usable subset). H3 is the only hypothesis with no result, because it needs
a training experiment rather than an evaluation.

## What it produces

Six trained models and one comparison table.

| arm | what the model trains on | everything else |
|---|---|---|
| **baseline** | the same PlantVillage images as always | protocol v2, untouched |
| **composite** | the same leaves, pasted onto random field backgrounds | protocol v2, untouched |

Three paired seeds each. Same architecture, same learning rate, same epochs, same
evaluator, same 236 field images. The only thing that differs is what is behind the
leaf.

The comparison is **paired**: for every field image, both models either get it
right or wrong, and the test resamples the same images for both. That stops a seed
that happened to see easy field images from flattering the treatment.

## What you need before pressing Run All

| requirement | how to set it |
|---|---|
| Accelerator | **GPU T4 x2** (Settings, right panel) |
| Dataset 1 | the PlantVillage dataset already used by notebooks 06 to 08 |
| Dataset 2 | **DeepWeeds**, Kaggle slug `imsparsh/deepweeds` |
| Internet | not needed |
| Run mode | **Save Version, then Save and Run All** so it survives the browser closing |

If DeepWeeds is not attached, cell 1 stops with a message and nothing else runs.
That is deliberate: without a licence-clean background corpus there is no
experiment, and training on PlantDoc imagery instead would corrupt the field
evaluation that produced the headline result.

## How long it takes

About 2.5 hours for six runs. If you want a fast smoke version first, set
`SEEDS = [42]` in cell 1: one pair of runs, roughly 50 minutes, enough to see
whether the direction of the effect is even interesting.

The notebook prints progress as it goes. You can close the browser once the commit
starts.

## What can go wrong, and what the notebook does about it

| failure | handling |
|---|---|
| DeepWeeds missing | cell 1 stops immediately, before any time is spent |
| a background image came from PlantDoc | the hash check in cell 2 aborts before training |
| the recipe cannot learn | the gate in cell 5 stops it in about 5 minutes |
| one arm crashes | the other arm still runs, so partial results survive |
| the session dies | finished seeds are skipped on restart |
| evaluation fails for one seed | that pair is skipped, the rest still compare |
| everything fails | the final cell still writes a bundle with whatever exists |

## Honest limits of the method

These belong in the report next to the result, not in a footnote.

**One random background per image, fixed for the run.** Re-sampling the background
every epoch would be stronger augmentation, but it needs the training engine to
carry a background pool and masks through the loop. With a fixed random assignment
the background-to-class mapping is already destroyed, because every class receives
random backdrops. What is given up is per-image background variety.

**The leaf mask is a heuristic.** Corner-colour plus greenness fails on close-up
leaf textures where the frame is almost entirely leaf. Masks covering less than 25%
or more than 90% of the frame are not trusted, and those images pass through
unaltered rather than being composited wrongly. On the validation split that was
2.7% of images.

**DeepWeeds is Australian rangeland.** Different continent, different species,
different cameras from both PlantVillage and PlantDoc. That is what makes it safe
to use, and it also means the backgrounds are not the same kind of field as
PlantDoc's, so a positive result would be a lower bound on what real in-domain
backgrounds could do.
"""

BEFORE_YOU_START = """## Before you press Run All

Work through this list once. Every item on it has cost this project time before.

1. **Accelerator set to GPU T4 x2** in the right-hand Settings panel. Cell 1 checks
   this and stops if only one GPU is present.
2. **PlantVillage attached.** Cell 1 checks for the development manifest and finds
   the dataset root automatically if the path has changed.
3. **DeepWeeds attached** at `/kaggle/input/deepweeds`. Cell 1 stops if it is
   missing, and tells you the slug to add.
4. **Run All, not one cell at a time.** Cells 3 to 7 depend on values defined
   earlier; the notebook is written for a top-to-bottom run.
5. **Save Version, then Save and Run All** if you want it to continue after you
   close the tab. An interactive session dies with the browser.

You can safely re-run this notebook. Finished seeds are skipped, the caches are
reused, and the composited images are not rebuilt if they already exist.
"""

TROUBLESHOOTING = """## If something goes wrong

**"Background corpus not attached"** in cell 1. Add the DeepWeeds dataset from the
right-hand panel, using the slug `imsparsh/deepweeds`, then Run All again.

**"DISJOINTNESS FAILED"** in cell 2. A background image is byte-identical to a
PlantDoc or PlantVillage image. Do not work around this. Check
`/kaggle/temp/derived/disjointness.json` for which file collided, and either remove
that file from the corpus or use a different corpus. Training on it would put the
evaluation set into the training data.

**"GATE FAILED"** in cell 5. The baseline arm could not reach 0.95 validation
accuracy in two epochs, which means the recipe is not reproducing the reference
behaviour (the original reached 0.9836 in epoch 1). Nothing long will run. Send me
the gate history printed above the error and I will diagnose it. Do not raise the
gate threshold and do not re-run hoping for a different outcome.

**A run shows `[FAIL]` or `[TIMEOUT]`** in cell 5. That seed is skipped and the
loop continues. The other seeds still run. Note which one failed and send the
bundle; one missing seed is not fatal.

**Field evaluation prints `missing checkpoint`** in cell 6. That arm's training did
not finish. The remaining pairs still compare.

**"no paired seed completed"** in cell 7. Both arms are needed for a pair. Check
that cell 5 printed results for both `baseline` and `composite`.

**Runtime is longer than expected.** Three seeds per arm is six runs. Set
`SEEDS = [42]` at the top of cell 1 and re-run; finished work is not repeated.
"""

SEND_BACK = """## What to send back

When the run finishes, the last cell prints the bundle path and size. Download
`/kaggle/working/s19_results.zip` from the Output panel on the right and send it
over.

The bundle contains, for each arm and seed, the checkpoint, the training history,
the validation metrics and the field predictions, plus one `paired/seed<N>.json`
per comparison with the confidence intervals.

With that in hand I can register the H3 verdict, update every document that
currently says "blocked", and fold the result into the results section. Until then
H3 stays marked as not tested, which is what the submission already says.
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

# The gate trains into its own directory. If it wrote into the baseline arm, it
# would leave a validation_metrics.json behind, the worker skips seeds that
# already have one, and baseline seed 42 would never train the full 12 epochs.
# The comparison would then be 2-epoch baseline against 12-epoch treatment.
GATE_ROOT = OUTPUT_ROOT / "_gate"
if GATE_ROOT.exists():
    shutil.rmtree(GATE_ROOT)

print("GATE: baseline seed 42, 2 epochs, must reach "
      f"{GATE_ACCURACY} before six runs are committed")
gate = subprocess.run(worker("efficientnet_b0", [SEEDS[0]], 2, GATE_ROOT,
                             ARMS["baseline"]["train_images"]),
                      env={**os.environ, "PYTHONUNBUFFERED": "1"})
gate_history = pd.read_csv(GATE_ROOT / "efficientnet_b0" / f"seed{SEEDS[0]}"
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

FINAL_CODE = '''# Always ship something. If the gate failed or a run crashed, this is the evidence.
manifest = {}
for path in sorted(OUTPUT_ROOT.rglob("*")):
    if path.is_file():
        manifest[str(path.relative_to(OUTPUT_ROOT))] = round(path.stat().st_size / 1e6, 2)
(OUTPUT_ROOT / "artifact_manifest.json").write_text(json.dumps({
    "files": manifest,
    "total_files": len(manifest),
    "session_hours": round((time.time() - STARTED_AT) / 3600, 2),
    "gate_passed": bool(GATE_PASSED),
    "seeds": SEEDS,
    "epochs": EPOCHS,
}, indent=2), encoding="utf-8")

print(f"{len(manifest)} artifacts in the run directory")
for name in list(manifest)[:15]:
    print(f"  {name}")
if len(manifest) > 15:
    print(f"  ... and {len(manifest) - 15} more")

bundle = shutil.make_archive("/kaggle/working/s19_results", "zip", root_dir=OUTPUT_ROOT)
print("Bundle: " + bundle + f"  ({Path(bundle).stat().st_size/1e6:.0f} MB)")
print("Download it from the Output panel on the right.")
if not GATE_PASSED:
    print("Note: the gate did not pass, so this bundle contains gate evidence, not results.")
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
        md(BEFORE_YOU_START),
        md("## 1. Setup and preflight\n\n"
           "Sets the seeds and epoch count, finds the dataset, and checks the three things that "
           "make the run possible: two GPUs, the PlantVillage manifest, and the background corpus.\n\n"
           "**Expected output:** the GPU names, the image counts, and the estimated runtime.\n\n"
           "**If it stops here:** read the error. It names the exact path or setting that is missing."),
        code(SETUP),
        md("## 2. Disjointness gate\n\n"
           "Hashes every background image and compares it against the 53,999 hashes already "
           "recorded in the PlantDoc and PlantVillage manifests.\n\n"
           "This is the check that makes H3 legitimate. If one background came from PlantDoc, the "
           "field evaluation would be seeing training data and the 78-point gap would be void. "
           "\"They are different datasets\" is not evidence; a hash comparison is.\n\n"
           "**Expected output:** `PASS: no byte-identical overlap`, then the limitation note that "
           "exact hashing will not catch a re-encoded near-duplicate."),
        code(DISJOINT_CODE),
        md("## 3a. Clean caches\n\n"
           "Decodes every image once into a uint8 memory map. The validation cache is built here and "
           "is **never** composited, so the PlantVillage score stays comparable to the baseline.\n\n"
           "**Expected output:** `clean caches ready`. Takes about 5 minutes the first time and is "
           "reused afterwards."),
        code(CACHE_CODE),
        md("## 3b. Leaf masks and the composite cache\n\n"
           "Two steps. First, segment the leaf from the backdrop in every training image using "
           "corner colour plus greenness. Second, paste each leaf onto a random field background.\n\n"
           "Images whose mask covers less than 25% or more than 90% of the frame are not trusted and "
           "pass through unchanged, because that is where the heuristic fails.\n\n"
           "**Expected output:** how many images were composited and how many passed through, plus "
           "two check images in `/kaggle/temp/derived/` you can open to confirm the leaves survive "
           "and the backgrounds actually changed."),
        code(COMPOSITE_CODE),
        md("## 4. Scripts and protocol\n\n"
           "Writes the eight scripts this notebook runs, compiles each one, and asserts that protocol "
           "v2 is unchanged. If this cell does not print `all scripts compile`, stop."),
        files_cell,
        md("## 5. Gate, then six runs\n\n"
           "The gate trains baseline seed 42 for two epochs and requires 0.95 validation accuracy. It "
           "writes into its own directory, so it cannot interfere with the six real runs.\n\n"
           "If the gate passes, both arms train three seeds each. A failure in one arm does not stop "
           "the other.\n\n"
           "**Expected output:** the gate history, then a line per completed run. Each run is about "
           "25 minutes."),
        code(TRAIN_CODE),
        md("## 6. Field evaluation\n\n"
           "Scores every trained checkpoint on PlantDoc: 236 images, 27 mapped classes, never seen "
           "during training.\n\n"
           "**Expected output:** one line per checkpoint with its field accuracy. Baseline should land "
           "near 0.21 to 0.25, consistent with the 0.2085 five-seed mean already on record."),
        code(EVAL_CODE),
        md("## 7. Paired comparison\n\n"
           "For each seed, compares the two arms on the same field images with a paired bootstrap, "
           "5,000 rounds. Reports the difference in field accuracy **and** the difference in the drop "
           "from PlantVillage, because a model can look better in the field while being worse in the "
           "lab.\n\n"
           "**Expected output:** a table, then one verdict per seed. The verdict is the H3 answer: "
           "supported if the confidence interval on the drop improvement excludes zero."),
        code(COMPARE_CODE),
        md("## 8. Bundle\n\n"
           "Runs whatever happened above. Writes an artifact manifest and zips the run directory, so "
           "a failed gate or a crashed arm still leaves something to diagnose from."),
        code(FINAL_CODE),
        md(NEXT_MD),
        md(TROUBLESHOOTING),
        md(SEND_BACK),
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

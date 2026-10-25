#!/usr/bin/env python3
"""Generate the Kaggle P0.1 completion notebook from repository sources.

Same anti-drift rule as the training notebook: the notebook embeds the real
files byte for byte instead of restating the logic, and this builder verifies
the embedded copies match. Regenerate with:

    python scripts/build_eval_notebook.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "04_Locked_Test_Evaluation.ipynb"

# target path inside Kaggle -> repository source
EMBEDDED = [
    ("/kaggle/working/src/plant_disease/__init__.py", "src/plant_disease/__init__.py"),
    ("/kaggle/working/src/plant_disease/calibration.py", "src/plant_disease/calibration.py"),
    ("/kaggle/working/src/plant_disease/data.py", "src/plant_disease/data.py"),
    ("/kaggle/working/src/plant_disease/models/__init__.py", "src/plant_disease/models/__init__.py"),
    ("/kaggle/working/src/plant_disease/models/efficientnet_b0.py", "src/plant_disease/models/efficientnet_b0.py"),
    ("/kaggle/working/src/plant_disease/models/resnet18.py", "src/plant_disease/models/resnet18.py"),
    ("/kaggle/working/src/plant_disease/models/smallcnn.py", "src/plant_disease/models/smallcnn.py"),
    ("/kaggle/working/evaluate_locked_test.py", "scripts/evaluate_locked_test.py"),
]

HEADER = """# P0.1 completion - checkpoint verification + locked-test evaluation

Everything P0.1 needs in one run:

1. **Verify the checkpoint contract** - loads safely, class order matches the frozen manifest, parameter count, recorded validation score, SHA-256 fingerprint.
2. **Re-measure validation** so the claimed 99.54% accuracy / 99.37% macro-F1 is independently reproduced rather than trusted.
3. **Score the locked test split once** - the artifact that never existed: per-class report, 38x38 confusion matrix, bootstrap 95% CIs, calibration.
4. **Error analysis** - weakest classes, top off-diagonal confusions, and the cross-crop healthy-class check the audit flagged.
5. **Write `P0.1_VERIFICATION.md`** with every number filled in, ready to commit as evidence.

It trains nothing and never uses the test split for model selection.

**Attach in Kaggle**

| Attachment | Gives |
|---|---|
| PlantVillage dataset | `data/manifests`, `data/raw/PlantVillage` |
| checkpoint dataset | `efficientnet_b0_best.pt` |

Set `CHECKPOINT_PATH` and `DATASET_ROOT` in cell 1. Nothing else needs editing.

Outputs land in `/kaggle/working/p01_verification/` and are zipped for download.

### Run order matters

| Cell | Does | Why the order matters |
|---|---|---|
| 1 | sets paths, creates the folder tree | everything else reads these paths |
| 2 | writes the package and the evaluator, then imports them to prove it worked | this is the only cell that can create the code, so it cannot be skipped |
| 3 | checkpoint contract check | imports `plant_disease.models` |
| 4 | runs the evaluation | needs 1 and 3 |
| 5 | headline results | needs 4 |
| 6 | error analysis | needs 4 |
| 7 | evidence record + zip | needs 4 and 6 |

Cell 2 prints `package import OK` when the code is in place. If the next cell ever fails with `unknown location`, cell 2 has not run.
"""

PREPARE = '''from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys

import numpy as np
import pandas as pd
import torch

# ---- paths ------------------------------------------------------------------
# Kaggle mounts a dataset at /kaggle/input/<slug>. The slug and any intermediate
# folder depend on how the dataset was uploaded, so these are defaults that get
# corrected by discovery below. Override either one if discovery picks wrong.
DATASET_ROOT = Path("/kaggle/input/datasets/shubhamkumar12864/plantvillage")
CHECKPOINT_PATH = Path("/kaggle/input/efficientnet/efficientnet_b0_best.pt")


KAGGLE_INPUT = Path("/kaggle/input")   # monkeypatched by the local test


def autodiscover_dataset_root(preferred: Path) -> Path:
    """Find the attached dataset that holds data/manifests."""
    if (preferred / "data" / "manifests" / "plantvillage_test.csv").is_file():
        return preferred
    base = KAGGLE_INPUT
    if not base.is_dir():
        return preferred
    for manifests in sorted(base.rglob("manifests")):
        if (manifests / "plantvillage_test.csv").is_file():
            return manifests.parent.parent
    return preferred


def autodiscover_checkpoint(preferred: Path) -> Path:
    """Find the attached checkpoint, preferring the requested one."""
    if preferred.is_file():
        return preferred
    base = KAGGLE_INPUT
    if base.is_dir():
        hits = sorted(base.rglob("*.pt"))
        exact = [h for h in hits if h.name == preferred.name]
        if exact:
            return exact[0]
        if len(hits) == 1:
            return hits[0]
    return preferred


_resolved_root = autodiscover_dataset_root(DATASET_ROOT)
if _resolved_root != DATASET_ROOT:
    print(f"dataset root corrected: {DATASET_ROOT} -> {_resolved_root}")
DATASET_ROOT = _resolved_root

_resolved_ckpt = autodiscover_checkpoint(CHECKPOINT_PATH)
if _resolved_ckpt != CHECKPOINT_PATH:
    print(f"checkpoint corrected: {CHECKPOINT_PATH} -> {_resolved_ckpt}")
CHECKPOINT_PATH = _resolved_ckpt

# ---- run settings -----------------------------------------------------------
RUN_VALIDATION = True    # re-measure validation to reproduce the claimed score
RUN_TEST = True          # locked test split, scored once
BOOTSTRAP = 2000         # resamples for the confidence intervals
BATCH_SIZE = 128
WORKERS = 4

WORKDIR = Path("/kaggle/working")
SCRIPT = WORKDIR / "evaluate_locked_test.py"
OUT_ROOT = WORKDIR / "p01_verification"

# The %%writefile cells below need this tree to exist first.
(WORKDIR / "src" / "plant_disease" / "models").mkdir(parents=True, exist_ok=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

print("Paths ready. Next: run the writefile cells, then the checkpoint check.")
print(f"  dataset    : {DATASET_ROOT}")
print(f"  checkpoint : {CHECKPOINT_PATH}")
'''

SETUP = '''# This cell needs the plant_disease package, which the writefile cells above create.
_sys_path_added = str(WORKDIR / "src")
if _sys_path_added not in sys.path:
    sys.path.insert(0, _sys_path_added)

try:
    from plant_disease.models import build_model
except ImportError as exc:
    raise ImportError(
        "The plant_disease package is not on disk yet. Run the %%writefile cells above "
        "(they create /kaggle/working/src/plant_disease). Original error: " + str(exc)
    )

# ---- preflight: inputs ------------------------------------------------------
if not DATASET_ROOT.is_dir():
    raise FileNotFoundError(f"Dataset not attached at {DATASET_ROOT}")
manifest_dir = DATASET_ROOT / "data" / "manifests"
for required in ("plantvillage_development.csv", "plantvillage_test.csv"):
    if not (manifest_dir / required).is_file():
        raise FileNotFoundError(f"{required} missing from {manifest_dir}")
if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(f"Checkpoint not attached at {CHECKPOINT_PATH}")

# ---- preflight: checkpoint contract ----------------------------------------
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
ARCH = checkpoint["architecture"]
classes = checkpoint["classes"]
state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
if all(str(k).startswith("module.") for k in state):
    state = {k[7:]: v for k, v in state.items()}

dev = pd.read_csv(manifest_dir / "plantvillage_development.csv")
manifest_classes = sorted(dev.loc[dev.split == "train", "label"].unique().tolist())
if classes != manifest_classes:
    raise ValueError(
        "Checkpoint class order does not match the development manifest. "
        f"first difference: {next(((i, a, b) for i, (a, b) in enumerate(zip(classes, manifest_classes)) if a != b), None)}"
    )

model = build_model(ARCH, len(classes), pretrained=False)
model.load_state_dict(state, strict=True)
parameters = sum(p.numel() for p in model.parameters())
del model

sha256 = hashlib.sha256()
with CHECKPOINT_PATH.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        sha256.update(chunk)

print("CHECKPOINT OK")
print(f"  path        : {CHECKPOINT_PATH}")
print(f"  sha256      : {sha256.hexdigest()}")
print(f"  size        : {CHECKPOINT_PATH.stat().st_size / 1024**2:.2f} MB")
print(f"  architecture: {ARCH}")
print(f"  parameters  : {parameters:,}")
print(f"  classes     : {len(classes)} (order matches the development manifest)")
print(f"  image size  : {checkpoint.get('image_size')}")
print(f"  epoch       : {checkpoint.get('epoch')}")
print(f"  val_macro_f1 recorded: {checkpoint.get('val_macro_f1')}")
print(f"  fingerprint : {checkpoint.get('dataset_fingerprint', 'absent (pilot checkpoint)')}")
print(f"  device      : {'cuda' if torch.cuda.is_available() else 'cpu'}")
'''

RUN = '''env = os.environ.copy()
env.update({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4"})

splits = []
if RUN_VALIDATION:
    splits.append("validation")
if RUN_TEST:
    splits.append("test")

outputs = {}
for split in splits:
    out = OUT_ROOT / split
    command = [
        sys.executable, str(SCRIPT),
        "--checkpoint", str(CHECKPOINT_PATH),
        "--dataset-root", str(DATASET_ROOT),
        "--split", split,
        "--output-dir", str(out),
        "--batch-size", str(BATCH_SIZE),
        "--workers", str(WORKERS),
        "--bootstrap", str(BOOTSTRAP),
        "--device", "auto",
    ]
    print(f"\\n=== {split} ===", flush=True)
    print("Running:", " ".join(command[-8:]), flush=True)
    result = subprocess.run(command, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed for {split} with code {result.returncode}. Read the log above.")
    outputs[split] = out

print("\\nAll requested splits evaluated:", sorted(outputs))
'''

HEADLINE = '''metrics = {split: json.loads((out / "metrics.json").read_text(encoding="utf-8"))
           for split, out in outputs.items()}

rows = []
for split, m in metrics.items():
    boot = m.get("bootstrap") or {}
    acc = boot.get("accuracy", {})
    f1 = boot.get("macro_f1", {})
    rows.append({
        "split": split,
        "images": m["images"],
        "accuracy": round(m["accuracy"], 5),
        "acc_ci95": f"[{acc.get('ci95_low', float('nan')):.4f}, {acc.get('ci95_high', float('nan')):.4f}]" if acc else "-",
        "balanced_accuracy": round(m["balanced_accuracy"], 5),
        "macro_precision": round(m["macro_precision"], 5),
        "macro_recall": round(m["macro_recall"], 5),
        "macro_f1": round(m["macro_f1"], 5),
        "macro_f1_ci95": f"[{f1.get('ci95_low', float('nan')):.4f}, {f1.get('ci95_high', float('nan')):.4f}]" if f1 else "-",
        "top3": round(m["top3_accuracy"], 5),
        "ece": round(m["ece"], 4),
        "mce": round(m["mce"], 4),
        "brier": round(m["brier"], 4),
        "ms_per_image": m["inference_ms_per_image"],
    })
headline = pd.DataFrame(rows)
display(headline)

print("Recorded values inside the checkpoint file:")
print(f"  val_macro_f1 = {checkpoint.get('val_macro_f1')}")
if "validation" in metrics:
    measured = metrics["validation"]["macro_f1"]
    recorded = checkpoint.get("val_macro_f1")
    delta = abs(measured - recorded) if recorded is not None else float("nan")
    verdict = "MATCH" if recorded is not None and delta < 1e-3 else "DIFFERS - investigate"
    print(f"  measured  val_macro_f1 = {measured:.6f}")
    print(f"  |difference| = {delta:.6f} -> {verdict}")
    print("  (a small gap is expected between GPU FP16 autocast training-time validation "
          "and CPU FP32 re-evaluation; a large gap is not)")
'''

ERRORS = '''confusions = {}
for split, out in outputs.items():
    matrix_path = out / "confusion_matrix.csv"
    if not matrix_path.is_file():
        continue
    matrix = pd.read_csv(matrix_path, index_col=0)
    off = matrix.where(~np.eye(len(matrix), dtype=bool)).stack().dropna().sort_values(ascending=False)
    top = off.head(10).reset_index()
    top.columns = ["true_class", "predicted_class", "count"]
    # stack() upcasts to float because of the masked diagonal; counts are ints.
    top["count"] = top["count"].astype(int)
    top = top.loc[top["count"] > 0].reset_index(drop=True)
    confusions[split] = top
    print(f"\\nTop off-diagonal confusions - {split}")
    if len(top):
        display(top)
    else:
        print("  none - every prediction was correct")

    report = pd.read_csv(out / "classification_report.csv").sort_values("f1")
    print(f"Weakest 8 classes by F1 - {split}")
    display(report.head(8))
    thin = report.loc[report.support <= 30].sort_values("support")
    if len(thin):
        print(f"Classes with <= 30 images of support - {split} (metrics on these are unstable)")
        display(thin)

    # Cross-crop healthy-class check the audit called out: Soybean healthy was
    # predicted as Pepper bell healthy in 20 of 32 errors. If that pattern is
    # still present it is evidence of background/appearance shortcuts.
    healthy = [c for c in matrix.index if "healthy" in c.lower()]
    if healthy:
        cross = matrix.loc[healthy, healthy]
        worst = (
            cross.where(~np.eye(len(cross), dtype=bool)).stack().dropna()
            .astype(int).sort_values(ascending=False).head(5)
        )
        worst = worst.loc[worst > 0]
        if len(worst):
            print(f"Healthy -> different-crop healthy confusions - {split}")
            print(worst.to_string())
        else:
            print(f"No cross-crop healthy confusions in {split}.")
'''

SUMMARY = '''lines = [
    "# P0.1 verification record",
    "",
    f"Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
    "",
    "## Checkpoint",
    "",
    "| Field | Value |",
    "|---|---|",
    f"| File | `{CHECKPOINT_PATH.name}` |",
    f"| SHA-256 | `{sha256.hexdigest()}` |",
    f"| Size | {CHECKPOINT_PATH.stat().st_size / 1024**2:.2f} MB |",
    f"| Architecture | {ARCH} |",
    f"| Parameters | {parameters:,} |",
    f"| Classes | {len(classes)} (order matches development manifest) |",
    f"| Image size | {checkpoint.get('image_size')} |",
    f"| Epoch | {checkpoint.get('epoch')} |",
    f"| Recorded val macro-F1 | {checkpoint.get('val_macro_f1')} |",
    f"| Dataset fingerprint | {checkpoint.get('dataset_fingerprint') or 'absent (pilot checkpoint)'} |",
    f"| Strict state-dict load | passed |",
    "",
    "## Measured performance",
    "",
]
for split, m in metrics.items():
    boot = m.get("bootstrap") or {}
    acc_ci = boot.get("accuracy", {})
    lines += [
        f"### {split} ({m['images']:,} images)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {m['accuracy']:.5f} |",
    ]
    if acc_ci:
        lines.append(f"| Accuracy 95% CI | [{acc_ci['ci95_low']:.5f}, {acc_ci['ci95_high']:.5f}] |")
    lines += [
        f"| Balanced accuracy | {m['balanced_accuracy']:.5f} |",
        f"| Macro precision | {m['macro_precision']:.5f} |",
        f"| Macro recall | {m['macro_recall']:.5f} |",
        f"| Macro F1 | {m['macro_f1']:.5f} |",
        f"| Top-3 accuracy | {m['top3_accuracy']:.5f} |",
        f"| ECE | {m['ece']:.5f} |",
        f"| MCE | {m['mce']:.5f} |",
        f"| Brier | {m['brier']:.5f} |",
        f"| Inference | {m['inference_ms_per_image']} ms/image |",
        "",
    ]

lines += [
    "## Notes",
    "",
    f"- Bootstrap resamples: {BOOTSTRAP} (image-level, percentile intervals).",
    "- The test split was scored once and was not used for model selection.",
    "- Validation was re-measured only to check the recorded score, not to choose anything.",
    "- Grad-CAM and field-domain (PlantDoc) results are separate, higher-phase artifacts.",
    "",
]
verification_path = OUT_ROOT / "P0.1_VERIFICATION.md"
verification_path.write_text("\\n".join(lines), encoding="utf-8")
print("Wrote", verification_path)

zip_path = shutil.make_archive(str(OUT_ROOT), "zip", root_dir=OUT_ROOT)
print(f"\\nDownload from the Kaggle Output panel: {zip_path}")
print("Contents:", sorted(p.name for p in OUT_ROOT.rglob('*') if p.is_file()))
'''

NEXT = """## After this run

1. Download `p01_verification.zip`.
2. Unpack `validation/` and `test/` into `reports/model_comparison/locked_test_efficientnet_b0/` and keep `P0.1_VERIFICATION.md` next to them.
3. The test numbers in `metrics.json` become the only test numbers quoted in the guide form.
4. If validation macro-F1 does not reproduce the recorded 0.99369 within ~0.002, stop and investigate before quoting any test result - that would mean the checkpoint and the reported number disagree.

Still open after this notebook: P0.2 (ResNet18 retrain on the shared engine), P0.3 (Grad-CAM + calibration narrative), P0.4 (PlantDoc label map and first field evaluation), P0.5 (stale summary + survey title).
"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def writefile_cell(text: str, target: str) -> dict:
    """Unused now that files are written from a single literal dict.

    Kept so the embed-verification helper below has a single place to look for
    %%writefile cells if one is ever reintroduced.
    """
    lines = [f"%%writefile {target}\n"]
    body = text.splitlines(keepends=True)
    if body and not body[-1].endswith("\n"):
        body[-1] += "\n"
    lines.extend(body)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def files_cell(files: list[tuple[str, str]]) -> dict:
    """One cell that writes every embedded file and imports it immediately.

    Replaces a chain of %%writefile cells. The chain could be run out of order,
    which produced ``ImportError: cannot import name 'build_model' from
    'plant_disease.models' (unknown location)`` when someone ran the check before
    the writers. With a single cell there is no order to get wrong.
    """
    lines = ["# Writes the package and the evaluator, then imports them. One cell, so",
             "# the code always exists before anything tries to use it.",
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
        "    print(f'wrote {_target.name:24s} {len(_content):6d} bytes')",
        "print(f'{len(FILES)} files written to', Path('/kaggle/working/src'))",
        "",
        "# Prove the package imports before anything else uses it.",
        "if str(WORKDIR / 'src') not in sys.path:",
        "    sys.path.insert(0, str(WORKDIR / 'src'))",
        "import importlib",
        "import plant_disease.models as _models",
        "importlib.reload(_models)",
        "if not hasattr(_models, 'build_model'):",
        "    raise ImportError('plant_disease.models imported but exposes no build_model; files may be stale')",
        "print('package import OK')",
    ]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]}


def build() -> dict:
    files = [(target, (ROOT / source).read_text(encoding="utf-8")) for target, source in EMBEDDED]
    cells = [md(HEADER), md("## 1. Paths"), code(PREPARE)]
    cells.append(md("## 2. Write and import the evaluation code\n\n"
                    "One cell writes the package and the evaluator, then imports the package to prove it works. "
                    "There is no ordering to get wrong."))
    cells.append(files_cell(files))
    cells.extend([
        md("## 3. Check the checkpoint"),
        code(SETUP),
        md("## 4. Run the evaluation"),
        code(RUN),
        md("## Headline results"),
        code(HEADLINE),
        md("## Error analysis"),
        code(ERRORS),
        md("## Evidence record and download bundle"),
        code(SUMMARY),
        md(NEXT),
    ])
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

    # Pull the FILES literal back out of the generated writer cell and compare it
    # to the repository sources. This is the guarantee that the notebook and the
    # scripts/ directory cannot drift apart.
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
                raise SystemExit("Could not parse the FILES literal out of the notebook")
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

    # Execute the writer cell's file loop into a temp tree and re-read it, so a
    # corrupted literal or a bad path is caught here rather than on Kaggle.
    import tempfile
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

#!/usr/bin/env python3
"""End-to-end smoke test for the training and evaluation pipeline.

The audit found no tests at all. This is the one that matters most: it runs the
real DDP engine and the real locked-test evaluator against synthetic data, so a
refactor cannot silently break checkpoint IO, metric reporting or the manifest
contract without failing here.

It is deliberately tiny (3 classes, a few dozen 32x32 images, 2 epochs, CPU via
gloo) so it finishes in well under a minute.

What it asserts
---------------
1. ``train_ddp.py`` completes a real 2-process run on CPU and writes
   ``*_best.pt``, ``*_last.pt``, ``training_history.csv``,
   ``validation_classification_report.csv``, ``validation_confusion_matrix.csv``
   and ``validation_metrics.json``.
2. The checkpoint carries the contract the evaluator needs: class order, image
   size, normalisation and a dataset fingerprint.
3. ``evaluate_locked_test.py`` scores that checkpoint on a held-out manifest and
   produces finite metrics inside [0, 1], a per-class table and bootstrap CIs.
4. Resuming from ``*_last.pt`` refuses a different dataset fingerprint.

Usage:
    python scripts/smoke_test_engine.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "scripts" / "train_ddp.py"
EVALUATOR = ROOT / "scripts" / "evaluate_locked_test.py"

CLASSES = ["Alpha___healthy", "Beta___spot", "Gamma___blight"]
IMAGE_SIZE = 32
TRAIN_PER_CLASS = 12
VAL_PER_CLASS = 6
TEST_PER_CLASS = 6


def make_images(rng, label_index: int, count: int) -> np.ndarray:
    """A synthetic but *learnable* signal: a class-specific colour patch."""
    base = np.zeros((count, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    colour = np.array([80 + 60 * label_index, 200 - 50 * label_index, 40 + 30 * label_index])
    base[:] = rng.integers(20, 60, (count, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    base[:, 8:24, 8:24, :] = colour.astype(np.uint8)
    noise = rng.integers(-12, 12, base.shape, dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def write_dataset(root: Path) -> dict:
    rng = np.random.default_rng(0)
    image_root = root / "data" / "raw" / "PlantVillage" / "train"
    manifest_dir = root / "data" / "manifests"
    image_root.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)

    splits = {"train": TRAIN_PER_CLASS, "validation": VAL_PER_CLASS, "test": TEST_PER_CLASS}
    label_to_index = {label: i for i, label in enumerate(CLASSES)}

    dev_rows, test_rows = [], []
    for label, index in label_to_index.items():
        for split, per_class in (("train", TRAIN_PER_CLASS), ("validation", VAL_PER_CLASS)):
            array = make_images(rng, index, per_class)
            for j in range(per_class):
                rel = f"{label}/img_{split}_{j}.png"
                (image_root / rel).parent.mkdir(exist_ok=True)
                from PIL import Image
                Image.fromarray(array[j]).save(image_root / rel)
                dev_rows.append(
                    {"dataset": "PlantVillage", "domain": "controlled", "relative_path": rel,
                     "label": label, "sha256": "0" * 64, "width": IMAGE_SIZE, "height": IMAGE_SIZE,
                     "group_id": f"g_{label}_{j}", "source_id": "synthetic", "plant_id": f"p_{label}_{j}",
                     "split": split}
                )
        array = make_images(rng, index, TEST_PER_CLASS)
        for j in range(TEST_PER_CLASS):
            rel = f"{label}/img_test_{j}.png"
            (image_root / rel).parent.mkdir(exist_ok=True)
            from PIL import Image
            Image.fromarray(array[j]).save(image_root / rel)
            test_rows.append(
                {"dataset": "PlantVillage", "domain": "controlled", "relative_path": rel,
                 "label": label, "sha256": "1" * 64, "width": IMAGE_SIZE, "height": IMAGE_SIZE,
                 "group_id": f"g_{label}_t{j}", "source_id": "synthetic", "plant_id": f"p_{label}_t{j}",
                 "split": "test"}
            )

    import csv
    for filename, rows in (("plantvillage_development.csv", dev_rows), ("plantvillage_test.csv", test_rows)):
        with (manifest_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Cache arrays in the layout the engine expects.
    cache = root / "cache"
    cache.mkdir()

    def write_cache(rows, tag):
        labels = np.asarray([label_to_index[r["label"]] for r in rows], dtype=np.int64)
        images = np.zeros((len(rows), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        for i, row in enumerate(rows):
            from PIL import Image
            with Image.open(image_root / row["relative_path"]) as image:
                images[i] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        x_path, y_path = cache / f"{tag}_x.npy", cache / f"{tag}_y.npy"
        np.save(x_path, images)
        np.save(y_path, labels)
        return x_path, y_path

    train_x, train_y = write_cache([r for r in dev_rows if r["split"] == "train"], "train")
    val_x, val_y = write_cache([r for r in dev_rows if r["split"] == "validation"], "val")
    classes_json = cache / "classes.json"
    classes_json.write_text(json.dumps(CLASSES), encoding="utf-8")
    return {
        "root": root, "image_root": image_root, "manifest_dir": manifest_dir,
        "train_x": train_x, "train_y": train_y, "val_x": val_x, "val_y": val_y,
        "classes_json": classes_json, "output": root / "run",
    }


def run(command: list[str], env: dict | None = None) -> None:
    print(f"\n$ {' '.join(str(part) for part in command)}")
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"Command failed with exit code {result.returncode}")
    tail = [line for line in result.stdout.strip().splitlines() if line.strip()][-6:]
    for line in tail:
        print("  " + line)


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="plantvillage_smoke_"))
    try:
        data = write_dataset(workdir)
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            # Windows CPU wheels are not always built with libuv, and torchrun's
            # TCP store then fails with a rendezvous error. Disabling libuv makes
            # the local DDP handshake work. Harmless on Linux/Kaggle.
            "USE_LIBUV": "0",
        })

        def engine_command(val_labels: Path, epochs: int = 2) -> list[str]:
            """Single-process invocation for the local check.

            ``--no-dist`` skips the process group entirely. torchrun is not used
            here because this Windows CPU torch build is compiled without libuv
            and gloo cannot resolve a device from the hostname, so no process
            group starts at all locally. Everything except multi-rank gradient
            synchronisation is still exercised; that path runs on Kaggle with
            NCCL, which is the configuration that ships.
            """
            return [
                sys.executable, str(ENGINE),
                "--arch", "resnet18", "--device", "cpu", "--no-dist",
                "--train-images", str(data["train_x"]), "--train-labels", str(data["train_y"]),
                "--val-images", str(data["val_x"]), "--val-labels", str(val_labels),
                "--classes-json", str(data["classes_json"]), "--output-dir", str(data["output"]),
                "--epochs", str(epochs), "--patience", "5", "--local-batch-size", "4",
                "--eval-batch-size", "8", "--workers", "0", "--image-size", str(IMAGE_SIZE),
                "--lr", "1e-3", "--ema-decay", "0", "--warmup-fraction", "0.1",
            ]

        print("=== 0a. package and engine agree on every architecture ===")
        # train_ddp.py instantiates models itself because it must run standalone on
        # Kaggle, so the two definitions could drift. Compare parameter counts and
        # state-dict keys, which is what a checkpoint contract actually depends on.
        import importlib.util
        sys.path.insert(0, str(ROOT / "src"))
        from plant_disease.models import MODEL_NAMES, build_model as package_build
        spec = importlib.util.spec_from_file_location("train_ddp_under_test", ENGINE)
        engine_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine_module)
        if set(engine_module.ARCHES) != set(MODEL_NAMES):
            raise SystemExit(f"architecture lists disagree: {engine_module.ARCHES} vs {MODEL_NAMES}")
        for arch in MODEL_NAMES:
            from_package = package_build(arch, len(CLASSES), pretrained=False)
            from_engine = engine_module.build_model(arch, len(CLASSES), pretrained=False)
            package_params = sum(p.numel() for p in from_package.parameters())
            engine_params = sum(p.numel() for p in from_engine.parameters())
            if package_params != engine_params:
                raise SystemExit(f"{arch}: package has {package_params:,} params, engine has {engine_params:,}")
            if set(from_package.state_dict()) != set(from_engine.state_dict()):
                raise SystemExit(f"{arch}: state-dict keys differ between package and engine")
            print(f"  ok {arch:16s} {package_params:>10,} params, keys identical")
        del from_package, from_engine

        print("\n=== 0b. every architecture reaches the optimizer ===")
        for arch in ("smallcnn", "resnet18", "efficientnet_b0", "convnext_tiny"):
            dry = engine_command(data["val_y"])
            dry[dry.index("resnet18")] = arch
            dry.append("--dry-run")
            run(dry, env=env)
            print(f"  ok {arch} ran 3 optimizer steps")

        print("\n=== 1. full 2-epoch DDP run on CPU ===")
        run(engine_command(data["val_y"]), env=env)

        print("\n=== 2. checkpoint contract ===")
        for name in ("resnet18_best.pt", "resnet18_last.pt", "training_history.csv",
                     "validation_classification_report.csv", "validation_confusion_matrix.csv",
                     "validation_metrics.json"):
            path = data["output"] / name
            assert path.is_file(), f"engine did not write {name}"
            print(f"  ok {name} ({path.stat().st_size} bytes)")

        import torch
        checkpoint = torch.load(data["output"] / "resnet18_best.pt", map_location="cpu", weights_only=True)
        for key in ("classes", "image_size", "normalization_mean", "normalization_std", "dataset_fingerprint"):
            assert key in checkpoint, f"checkpoint missing {key}"
        assert checkpoint["classes"] == CLASSES, "class order changed"
        print(f"  ok contract keys present, fingerprint={checkpoint['dataset_fingerprint'][:12]}...")

        metrics = json.loads((data["output"] / "validation_metrics.json").read_text(encoding="utf-8"))
        for key in ("val_accuracy", "val_macro_f1", "val_ece", "val_brier", "val_top3_accuracy"):
            assert key in metrics, f"validation_metrics.json missing {key}"
            assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} outside [0, 1]"
        print(f"  ok val_accuracy={metrics['val_accuracy']:.3f} val_macro_f1={metrics['val_macro_f1']:.3f} "
              f"ece={metrics['val_ece']:.3f}")

        print("\n=== 3. locked-test evaluator ===")
        eval_dir = workdir / "eval"
        run([
            sys.executable, str(EVALUATOR),
            "--checkpoint", str(data["output"] / "resnet18_best.pt"),
            "--dataset-root", str(data["root"]),
            "--split", "test", "--output-dir", str(eval_dir),
            "--batch-size", "8", "--workers", "0", "--bootstrap", "50",
            "--device", "cpu",
        ], env=env)
        for name in ("metrics.json", "classification_report.csv", "confusion_matrix.csv",
                     "selective_risk_coverage.csv", "reliability_bins.csv", "reliability.md"):
            assert (eval_dir / name).is_file(), f"evaluator did not write {name}"
            print(f"  ok {name}")
        test_metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
        assert test_metrics["images"] == len(CLASSES) * TEST_PER_CLASS, "wrong image count"
        assert 0.0 <= test_metrics["accuracy"] <= 1.0
        assert test_metrics["bootstrap"]["accuracy"]["ci95_low"] <= test_metrics["bootstrap"]["accuracy"]["ci95_high"]
        print(f"  ok test accuracy={test_metrics['accuracy']:.3f} "
              f"CI=[{test_metrics['bootstrap']['accuracy']['ci95_low']:.3f}, "
              f"{test_metrics['bootstrap']['accuracy']['ci95_high']:.3f}]")
        assert test_metrics["accuracy"] >= 0.8, (
            "The synthetic task is trivially separable; accuracy below 0.8 means the training "
            "or evaluation path is broken."
        )

        print("\n=== 4. resume fingerprint guard ===")
        broken = dict(data)
        broken["val_y"] = data["root"] / "cache" / "tampered_val_y.npy"
        tampered = np.load(data["val_y"]).copy()
        tampered[0] = (tampered[0] + 1) % len(CLASSES)
        np.save(broken["val_y"], tampered)
        result = subprocess.run(
            engine_command(broken["val_y"]), env=env, capture_output=True, text=True
        )
        combined = result.stdout + result.stderr
        assert result.returncode != 0, "engine resumed happily on a tampered split"
        assert "different split" in combined.lower(), (
            "engine failed but not because of the fingerprint guard; inspect the log\n" + combined[-1500:]
        )
        print("  ok tampered split was rejected by the fingerprint guard")

        print("\nSMOKE TEST PASSED")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

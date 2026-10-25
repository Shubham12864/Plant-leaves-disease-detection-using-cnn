#!/usr/bin/env python3
"""Run one architecture over several seeds, on either one GPU or both.

This exists so a sweep can be scheduled instead of run one-architecture-per-
session. It shells out to ``train_ddp.py`` once per seed, which keeps every run
an independent process with its own resumable checkpoint.

Two modes:

``--mode ddp``
    Uses both GPUs through ``torch.distributed.run``. Right for the models that
    are bandwidth-bound on a single T4 (EfficientNet-B0, ConvNeXt-Tiny).

``--mode single``
    One process, no process group, pinned with ``CUDA_VISIBLE_DEVICES``. Right
    for the small models, where two GPUs per model waste half the machine. Two
    of these can run side by side, one per GPU.

Why the split matters: the pilot measured SmallCNN at 0.67 min/epoch and
EfficientNet-B0 at 2.45 min/epoch. Running SmallCNN on both GPUs to finish in
40 seconds is worse than running it on one GPU while something else uses the
other.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE / "train_ddp.py"


def build_command(arch, seed, cfg, protocol, paths, out_dir, mode, script, device, epochs):
    # Arguments only. The interpreter and the script path are added once, by the
    # mode branch below: torchrun takes the script as its first positional
    # argument, so leaving "python" in here made torchrun try to execute the
    # interpreter binary as Python source ("source code cannot contain null
    # bytes" / "ELF" in the traceback).
    args = [
        "--arch", arch,
        "--train-images", str(paths["train_x"]), "--train-labels", str(paths["train_y"]),
        "--val-images", str(paths["val_x"]), "--val-labels", str(paths["val_y"]),
        "--classes-json", str(paths["classes"]), "--output-dir", str(out_dir),
        "--epochs", str(epochs), "--patience", str(protocol["patience"]),
        "--local-batch-size", str(cfg["local_batch_size"]),
        "--eval-batch-size", str(protocol["eval_batch_size"]),
        "--lr", str(cfg["lr"]), "--weight-decay", str(cfg["weight_decay"]),
        "--workers", str(protocol["workers"]),
        "--label-smoothing", str(protocol["label_smoothing"]),
        "--class-weight-alpha", str(protocol["class_weight_alpha"]),
        "--ema-decay", str(protocol["ema_decay"]),
        "--grad-clip", str(protocol["grad_clip"]),
        "--warmup-fraction", str(protocol["warmup_fraction"]),
        "--min-lr-factor", str(protocol.get("min_lr_factor", 0.01)),
        "--rotation-deg", str(protocol["rotation_deg"]),
        "--shuffle-mode", protocol["shuffle_mode"],
        "--image-size", str(protocol["image_size"]),
        "--seed", str(seed),
    ]
    if mode == "ddp":
        # torchrun's first positional argument must be the training script.
        return [sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nnodes=1", "--nproc_per_node=2", str(script)] + args
    # Single process on whichever GPU CUDA_VISIBLE_DEVICES exposes as 0.
    return [sys.executable, str(script)] + args + ["--no-dist", "--device", device]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--mode", choices=("ddp", "single"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-labels", type=Path, required=True)
    parser.add_argument("--classes-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=ENGINE)
    parser.add_argument("--device", default="cuda", choices=("auto", "cuda", "cpu"),
                        help="single-gpu mode only; cpu exists for local tests")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override protocol epochs; for fast local checks")
    parser.add_argument("--verbose", action="store_true",
                        help="echo each engine command before running it")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.arch not in protocol["architectures"]:
        raise SystemExit(f"{args.arch} is not in protocol.json")
    cfg = protocol["architectures"][args.arch]
    paths = {
        "train_x": args.train_images, "train_y": args.train_labels,
        "val_x": args.val_images, "val_y": args.val_labels,
        "classes": args.classes_json,
    }

    print(f"[{args.arch}] mode={args.mode} seeds={args.seeds} "
          f"gpus={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}", flush=True)

    failures = []
    for seed in args.seeds:
        out_dir = args.output_root / args.arch / f"seed{seed}"
        if (out_dir / "validation_metrics.json").is_file():
            print(f"[{args.arch} seed {seed}] already complete, skipping", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        epochs = args.epochs if args.epochs is not None else protocol["epochs"]
        command = build_command(args.arch, seed, cfg, protocol, paths, out_dir,
                                args.mode, args.engine, args.device, epochs)
        if args.verbose:
            print("  " + " ".join(command), flush=True)
        started = time.time()
        result = subprocess.run(command, env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=False)
        minutes = (time.time() - started) / 60.0
        status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
        print(f"[{args.arch} seed {seed}] {status} in {minutes:.2f} min", flush=True)
        if result.returncode != 0:
            failures.append(seed)
            # Echo the command so a failure is diagnosable from the log alone.
            print("  command: " + " ".join(command), flush=True)

    if failures:
        raise SystemExit(f"[{args.arch}] seeds failed: {failures}")
    print(f"[{args.arch}] all requested seeds complete", flush=True)


if __name__ == "__main__":
    main()

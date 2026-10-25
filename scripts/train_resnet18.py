#!/usr/bin/env python3
"""Train ResNet18 through the shared DDP engine (closes the P0.2 gap).

The audit found no ResNet18 training code in the repository: only the saved
weights existed, produced by a Kaggle script that was never committed. That made
two of three models impossible to retrain and turned the comparison into three
ad hoc runs.

Rather than adding a second, divergent training script, this is a thin launcher
over ``scripts/train_ddp.py``. The comparison therefore becomes *per-architecture
tuned on one auditable code path*, which is a defensible way to describe it.

Differences from the EfficientNet-B0 defaults, and why:

  * **LR 3e-4** instead of 3e-4 for B0 -- ResNet18 was trained at 3e-4 in the
    original run (its checkpoint's ``resume_config`` records this), so keeping
    the value preserves comparability with the existing weights.
  * **Batch 64/GPU** matches the engine default. The original ResNet18 run used a
    global batch of 64 via DataParallel; 128 global here is a stated deviation.
  * **No depthwise convolutions**, so ResNet18 is cheaper per step than B0 and
    should finish faster despite having roughly twice the parameters.

Usage:
    python scripts/train_resnet18.py \\
        --train-images /kaggle/temp/train_x.npy --train-labels /kaggle/temp/train_y.npy \\
        --val-images /kaggle/temp/val_x.npy --val-labels /kaggle/temp/val_y.npy \\
        --classes-json /kaggle/temp/classes.json --output-dir /kaggle/working/model_comparison/resnet18

Any extra flag is forwarded verbatim to the engine, so ``--dry-run``,
``--epochs`` and the loss knobs all work.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "scripts" / "train_ddp.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-labels", required=True)
    parser.add_argument("--classes-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nproc", type=int, default=2, help="processes (one per GPU)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="forwarded to train_ddp.py")
    args = parser.parse_args()

    if not ENGINE.is_file():
        raise SystemExit(f"Training engine not found: {ENGINE}")

    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        f"--nproc_per_node={args.nproc}", str(ENGINE),
        "--arch", "resnet18",
        "--train-images", args.train_images, "--train-labels", args.train_labels,
        "--val-images", args.val_images, "--val-labels", args.val_labels,
        "--classes-json", args.classes_json, "--output-dir", args.output_dir,
        "--lr", str(args.lr),
    ]
    command.extend(arg for arg in args.extra if arg != "--")

    print("Launching ResNet18 on the shared engine:")
    print("  " + " ".join(command))
    sys.exit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""How many classes does a training batch actually contain?

The manifest is written in class order. ``block_shuffled_indices`` permutes the
order of 2048-image blocks and shuffles inside each block, so a block of
consecutive indices is still drawn from very few classes. If a batch ends up
single-class, BatchNorm normalises every image by statistics from one class, and
the network learns much less per step.

ConvNeXt uses LayerNorm and was the only architecture that trained properly in
the v1 sweep. EfficientNet-B0, ResNet18 and SmallCNN all use BatchNorm and all
collapsed. This script measures whether that explanation holds.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
dev = pd.read_csv(REPO / "data" / "manifests" / "plantvillage_development.csv")
train = dev[dev.split == "train"].reset_index(drop=True)
labels = train.label.values
n = len(labels)

print(f"train rows: {n}")
runs = int((labels[1:] != labels[:-1]).sum() + 1)
print(f"contiguous label runs: {runs} for {len(set(labels))} classes "
      f"-> manifest is {'sorted by class' if runs == len(set(labels)) else 'interleaved'}")
print(f"first 8 labels: {list(labels[:8])}")


def blocked_order(n_items: int, seed: int, block: int) -> np.ndarray:
    """Copy of block_shuffled_indices from scripts/train_ddp.py."""
    order = np.arange(n_items)
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n_items, block)
    rng.shuffle(starts)
    out = []
    for start in starts:
        chunk = order[start:start + block]
        rng.shuffle(chunk)
        out.append(chunk)
    return np.concatenate(out)


def report(name: str, permutation: np.ndarray, batch_size: int) -> None:
    ordered = labels[permutation]
    n_batches = n // batch_size
    distinct = np.array([
        len(np.unique(ordered[i * batch_size:(i + 1) * batch_size]))
        for i in range(n_batches)
    ])
    pure = int((distinct == 1).sum())
    print(f"\n{name} | batch {batch_size} | {n_batches} batches")
    print(f"  single-class batches : {pure}/{n_batches} ({100 * pure / n_batches:.1f}%)")
    print(f"  classes per batch    : median {np.median(distinct):.1f}, mean {distinct.mean():.2f}, max {distinct.max()}")


for batch_size in (16, 64, 128):
    report(f"blocked shuffle (block=2048)", blocked_order(n, 42, 2048), batch_size)

for batch_size in (16, 64, 128):
    report("random shuffle", np.random.default_rng(42).permutation(n), batch_size)

# What the block size would have to be for batches to mix classes.
print("\nIf blocks were small enough to mix classes:")
for block in (64, 128, 256, 512):
    ordered = labels[blocked_order(n, 42, block)]
    n_batches = n // 16
    distinct = np.array([len(np.unique(ordered[i * 16:(i + 1) * 16])) for i in range(n_batches)])
    pure = int((distinct == 1).sum())
    print(f"  block={block:4d} batch=16: single-class batches {pure}/{n_batches} "
          f"({100 * pure / n_batches:.1f}%), mean classes {distinct.mean():.2f}")

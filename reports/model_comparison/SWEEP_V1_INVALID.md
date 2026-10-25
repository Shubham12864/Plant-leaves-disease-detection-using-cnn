# Sweep v1 — twelve runs, none of them usable

**Verdict: the sweep is invalid as a comparison. Do not quote any number in this
file except as evidence of a defect.**

Run date: 2026-10-25. Source archive: `sweep_all.zip` (3.24 GB, 99 files, 12
run directories). Protocol `configs/protocol.json` version 1, seeds 42/43/44.

The data pipeline is sound: every one of the twelve runs carries the same
`dataset_fingerprint` (`01c1fe7339df2701…`), so all of them saw the same frozen
split. The failure is in the training configuration, not in the data.

## What came back

| Architecture | val accuracy (3 seeds) | macro-F1 (3 seeds) | best epoch | min |
|---|---|---|---|---|
| ConvNeXt-Tiny | 0.9461 / 0.9509 / 0.9363 | 0.9304 / 0.9368 / 0.9184 | 12 / 12 / 12 | 170 |
| EfficientNet-B0 | 0.6815 / 0.7246 / 0.6799 | 0.6602 / 0.7081 / 0.6793 | 12 / 11 / 10 | 90 |
| ResNet18 | 0.0721 / 0.5839 / 0.4867 | 0.0263 / 0.4745 / 0.4750 | 1 / 11 / 12 | 41 |
| SmallCNN | 0.1507 / 0.2311 / 0.1802 | 0.0311 / 0.0813 / 0.0433 | 3 / 11 / 12 | 36 |

Against the architecture's established result on the same split:

| Architecture | Established | Sweep best | Difference |
|---|---|---|---|
| EfficientNet-B0 | 0.9954 val (0.99369 macro-F1) | 0.7246 | **−27.1 pp** |
| ResNet18 | 0.9615 val | 0.5839 | **−37.8 pp** |
| SmallCNN | 0.9601 val | 0.2311 | **−72.9 pp** |

A seed moves these numbers by a few tenths of a point. These gaps are not seed
variance; the protocol is not doing the job.

## Why — the reference converged in one epoch, the protocol does not

EfficientNet-B0, epoch 1, same split, same architecture:

| | train loss | val accuracy |
|---|---|---|
| Original run (`01_Train_EfficientNet_B0.ipynb`, produced the 99.54% checkpoint) | 0.5166 | **0.9826** |
| Protocol v1 sweep, seed 42 | 3.3035 | **0.1226** |

The original was at 98.26% after one epoch. Protocol v1 needed twelve epochs to
reach 68%.

## Measured cause: the backbone never fine-tuned

Whole-model relative L2 distance to pristine torchvision ImageNet weights, 358
tensors:

| Checkpoint | Distance from ImageNet init | val accuracy |
|---|---|---|
| Verified reference | **1.96%** | 0.9954 |
| Sweep seed 42 | **0.49%** | 0.6815 |

The sweep's network is four times closer to ImageNet than the reference, i.e.
almost none of the fine-tuning happened. Its head learned (head cosine similarity
to the reference 0.585) while the backbone stayed frozen in all but name. That is
what 0.68 accuracy looks like: an ImageNet feature extractor with a trained
38-class head.

Four settings in protocol v1 explain it, each verified against the original
script's source:

| Setting | Original (proven) | Protocol v1 | Effect |
|---|---|---|---|
| weight decay | 1e-4 | **0.05** | 500× stronger; pulls weights back toward init every step |
| global batch | 32 | **128** | 4× fewer optimizer steps per epoch (287 vs 1149) |
| LR schedule | constant 1e-4 | 3e-4, 10% warmup, cosine to 0 | mean applied LR far below the nominal 3e-4 |
| label smoothing / class weights | none | 0.05 / α=0.5 | changes gradient scale and floors the loss |

The nominal LR went *up* (1e-4 → 3e-4), which is why the defect is not obvious
from the config alone. It is outweighed by 4× fewer steps, a decaying schedule,
and a 500× stronger weight decay.

The schedule decay is now measured directly. Running the engine on CPU with
three epochs and reading back the learning rate:

| Protocol | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| v2 (warmup 0.0, min_lr_factor 1.0) | 1.000e-03 | 1.000e-03 | 1.000e-03 |
| v1 (warmup 0.10, min_lr_factor 0.01) | 8.033e-04 | 2.844e-04 | **1.000e-05** |

Under v1 the final epoch trained at **1% of the nominal rate**. The nominal
3e-4 that looks healthy in the config was 1e-5 by the end of the run, which is
one to two orders of magnitude too small to move a backbone.

This is also why the defect was invisible in the artifacts: the engine did not
record the applied learning rate. It does now (`lr` column in
`training_history.csv`).

## Secondary defect: everything ran ~4.5× slower than the pilot

| Architecture | Pilot min/epoch | Sweep min/epoch | Ratio |
|---|---|---|---|
| SmallCNN | 0.67 | 3.01 | 4.5× |
| ResNet18 | 0.78 | 3.44 | 4.4× |
| EfficientNet-B0 | 2.45 | 7.52 | 3.1× |
| ConvNeXt-Tiny | 3.15 | 14.16 | 4.5× |

The sweep took 5.6 h against a 4.2 h projection. The cause is not visible in the
artifacts; it needs the session log (data-loader throughput or `/kaggle/temp`
read speed). Recorded rather than explained.

## Early stopping compounded the damage

Patience 3 against a metric that was not improving killed two runs early:
ResNet18 seed 42 stopped after 4 epochs with best epoch 1, SmallCNN seed 42 after
6 epochs with best epoch 3. Under a working configuration those runs would not
have been near their best epoch yet.

## What must change before any sweep is re-run

1. **Do not re-run 12 runs on this protocol.** It cannot reach the reference
   numbers, so the comparison would compare four under-trained models.
2. Restore the proven optimisation settings for the baseline arm: weight decay
   1e-4, global batch 32 (local 16), constant LR 1e-4, no label smoothing, no
   class weights.
3. Keep augmentation and class balancing as a **separate arm** for H3 rather
   than baking them into the baseline. That is also what H3 requires, since the
   claim is that augmentation recovers part of the field gap.
4. Gate any future sweep behind a **two-epoch check**: EfficientNet-B0 must be
   at or above 0.95 validation accuracy by epoch 2. The original was at 0.9826
   after one epoch, so this is a loose gate.
5. Investigate the 4.5× slowdown separately, from a session log.

## What this run is worth

It is a real, measured negative result about the project's own pipeline, obtained
before any field evaluation, and it is the reason the sweep took 5.6 hours
instead of producing four wrong rows in a comparison table. It belongs in the
report as a methodology finding: a frozen protocol was adopted as "one
configuration for every architecture" and was never validated against the result
it was supposed to reproduce. The instrumentation caught it because every run
recorded its fingerprint, its history, and its checkpoint.

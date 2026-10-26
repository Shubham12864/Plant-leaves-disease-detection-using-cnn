# Sweep v2 — fourteen runs, zero failures

**Date:** 2026-10-26 · **Artifact:** `overnight_runs.zip` (2.12 GB, 93 files)
**Protocol:** `configs/protocol_v2.json` · **Notebook:** `notebooks/08_Overnight_Sweep.ipynb`

Every run carries the same `dataset_fingerprint`
(`01c1fe7339df27010e6da92466f3bf8af82f4eaca50f36983328d22b0b669608`), so all of
them saw the identical frozen split. Fourteen runs were planned, fourteen
completed, and `_overnight_log.json` contains no `failed`, `timeout` or `error`
entries.

## The gate released the sweep

| | measured | original run |
|---|---:|---:|
| epoch 1 validation accuracy | **0.9836** | 0.9826 |
| epoch 2 validation accuracy | 0.9869 | — |
| minutes per epoch | 2.15 | 1.70 at epoch 1 |

The recipe now reproduces the behaviour of the run that produced the published
checkpoint. The defect was batch composition, not the learning rate: blocked
shuffling gave a median of 3 classes per batch on a class-sorted manifest, which
broke BatchNorm. Evidence and fix in `docs/KNOWN_ISSUES.md` R14.

## Results — PlantVillage validation, 6,544 images, 38 classes

| Architecture | n | macro-F1 mean | sd | 95% CI | spread |
|---|---:|---:|---:|---|---:|
| ConvNeXt-Tiny | 3 | 0.98951 | 0.00028 | 0.98881 – 0.99021 | 0.06 pp |
| **EfficientNet-B0** (reference) | 5 | **0.98946** | **0.00171** | 0.98734 – 0.99158 | 0.40 pp |
| ResNet18 | 3 | 0.98460 | 0.00166 | 0.98048 – 0.98871 | 0.32 pp |
| SmallCNN | 3 | 0.91599 | 0.00865 | 0.89450 – 0.93749 | 1.60 pp |

Accuracy and calibration on the same runs:

| Architecture | accuracy mean | ECE mean |
|---|---:|---:|
| ConvNeXt-Tiny | 0.99304 | 0.00289 |
| EfficientNet-B0 | 0.99280 | 0.00210 |
| ResNet18 | 0.98839 | 0.00327 |
| SmallCNN | 0.94240 | 0.00900 |

## What the numbers say

**Architecture does not separate the top two.** EfficientNet-B0 and
ConvNeXt-Tiny differ by 0.00005 macro-F1, while the seed spread inside
EfficientNet-B0 alone is 0.0017 — thirty-four times larger. This is not a
"ConvNeXt wins" or "B0 wins" result; it is evidence that on this benchmark the
architecture choice is not what determines the outcome.

**ResNet18 is genuinely behind**, by 0.49 pp (about 2.9 pooled standard
deviations). Small is not nothing, but it is also not the 5–10 pp gaps that
architecture-comparison papers usually report.

**SmallCNN is far behind**, 7.35 pp, and with a spread eight times wider. That is
the expected shape: a network trained from scratch on 36,762 images at 224²
cannot match an ImageNet-pretrained backbone.

**Calibration is excellent in-domain**: ECE between 0.0013 and 0.0110 on
validation. This must not be read as a claim about field images — that is what
Stage E measures.

## Two disclosures that must travel with these numbers

**1. The reference is 0.42 pp below the published checkpoint.** The published
value is 0.99369; our five-seed mean is 0.98946. The runs stop early (best epoch
4, 7, 5, 8, 10 under patience 3), whereas the published run was carried to epoch
12. A single seed here (42) reached 0.9868, which is 0.69 pp below the published
number and would have been reported as "the" reference result if only one run
had been made.

That is the point of running seeds: **the published 99.37% is one draw from a
distribution with a standard deviation near 0.17 pp.** Quoting it to four decimal
places overstates its precision.

**2. SmallCNN replicates at 91.60%, not the previously recorded 96.01%.** The
three seeds agree tightly (0.9061, 0.9221, 0.9197), so this is a stable result
under protocol v2 rather than noise. The older 96.01% figure came from
`reports/phase2_baseline/`, whose configuration differs from v2 in ways not yet
isolated. **The 96.01% figure must not be quoted until that discrepancy is
resolved**, either by re-running the phase-2 configuration under the current
engine or by identifying the difference.

## Use restrictions

- These are validation numbers. The locked test set stays sealed until Stage F.
- No PlantDoc number exists yet. Nothing here says anything about field images.
- Nothing may be quoted in a document until it is registered in
  `results/registry.json`; the four headline comparisons are registered under
  `sweep_v2_overnight`.
- Checkpoints from these runs are in the archive but have not been inspected or
  hashed. `checkpoints/SHA256SUMS` still describes only the pre-sweep artifacts.

## What this unlocks

Stage B is complete: four architectures, one protocol, three to five seeds, with
the spread reported. The comparison is now fair in the sense that every model
faced identical data order, identical augmentation, identical optimisation, and
identical early-stopping rules.

Next is Stage C, the part the whole project exists for: score these models on
PlantDoc and measure how far accuracy falls when the background is real soil
instead of a lab bench.

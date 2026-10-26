# EXPERIMENT LOG — Plant Leaf Disease Detection Using CNN (BT40)

*Timestamped runs, configs, seeds, checkpoints, metrics, and observations. Every number traced to `results/registry.json` with artifact path. Frontmatter uses Obsidian-compatible YAML.*

---
# Experiment Log Index

| Run ID | Date | Architecture | Seed | GPU | Config | Val Metrics | Test Metrics | Checkpoint | Status |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 2026-10-25 | EfficientNet-B0 | 42 | 2× T4 | `configs/protocol.json` v1 | val_macro-F1 0.99369 | 53 errors/10,709, macro-F1 0.99314 | `efficientnet_b0_best.pt` | ✅ Verified |
| R2 | 2026-10-25 | SmallCNN | 42 | 2× T4 | `configs/protocol.json` v1 | val results (per arch) | accuracy 96.95%, balanced acc 96.02%, macro-F1 96.07% | `smallcnn_best.pt` | ✅ CPU-reproducible |
| R3 | 2026-10-25 | ResNet18 | 42 | 2× T4 | `configs/protocol.json` v1 | val results (per arch) | no saved locked-test result | `resnet18_best.pt` | ⚠️ D20: manifest_sha256 mismatch |
| R4 | 2026-10-25 | ConvNeXt-Tiny | 42 | 2× T4 | `configs/protocol.json` v1 | epoch-1 macro-F1 0.060 (warmup) | N/A | generated S5 | 🔄 In progress |

---
# Run Details: R1 — EfficientNet-B0 (primary reference model)

## Configuration

| Field | Value |
|---|---|
| **Run date** | 2026-10-25 |
| **Architecture** | EfficientNet-B0 |
| **Seed** | 42 |
| **GPUs** | 2× Tesla T4 (Kaggle) |
| **Config file** | `configs/protocol.json` v1 |
| **Protocol version** | v1 (later found to cause R12 invalidation; see below) |
| **ONLY_ARCH** | `efficientnet_b0` |
| **Batch size** | 64 (32 per GPU) |
| **Image size** | 224 × 224 pixels |
| **Epochs** | 12 (best checkpoint selected by val macro-F1) |
| **Mixed precision** | FP16 |
| **Optimizer** | As per `train_efficientnet_b0_ddp.py` |
| **Learning rate schedule** | warmup+cosine (per protocol.json) |
| **Weight decay** | 0.05 |
| **Class balancing** | Per protocol.json configuration |
| **Data augmentation** | Protocol-specified training-only augmentations |

## Checkpoint

| Field | Value |
|---|---|
| **Filename** | `checkpoints/efficientnet_b0/efficientnet_b0_best.pt` |
| **SHA-256** | `90e003cf…` (verified via `checkpoints/SHA256SUMS`) |
| **Parameters** | 4,056,226 |
| **Loaded with** | `strict=True`; class order assertion: 38/38 identical order |
| **Provenance** | `efficientnet_b0/provenance.json` — contract verified; no `dataset_fingerprint` (D27) |

## Results — PlantVillage Validation Split

| Metric | Value |
|---|---|
| **Accuracy** | 99.54% |
| **Macro precision** | 99.31% |
| **Macro recall** | 99.46% |
| **Macro-F1** | 99.37% |
| **Balanced accuracy** | 99.46% (= macro recall) |

> **ECE**: 0.0022 (aggregate); **MCE**: 0.53 (from one 15-bin cell with n=1 image). Bin counts in `test/reliability_bins.csv`.

## Results — Locked PlantVillage Test Split (Verified 2026-10-25)

> **Artifact path**: `reports/model_comparison/locked_test_efficientnet_b0/test/`

| Metric | Value | 95% CI / Source |
|---|---|---|
| **Accuracy** | 99.51% | — |
| **95% CI (accuracy)** | 99.37–99.64% | Wilson score interval |
| **Balanced accuracy** | 99.31% | = macro recall (D07) |
| **Macro precision** | 99.33% | — |
| **Macro recall** | 99.31% | = balanced accuracy |
| **Macro-F1** | 99.31% | — |
| **95% CI (F1)** | 99.10–99.51% | Bootstrap over independent groups |
| **Errors** | 53 in 10,709 | — |
| **ECE** | 0.0022 | Aggregate; bin counts in `test/reliability_bins.csv` |
| **MCE** | 0.53 | Max calibration error; from one bin with n=1 image |

### Cross-Crop Healthy Confusions (EfficientNet-B0, Test)

| True → Predicted | Count |
|---|---|
| Raspberry→Cherry healthy | 4 |
| Apple→Blueberry healthy | 1 |
| Apple→Soybean healthy | 1 |
| Potato→Soybean healthy | 1 |
| Soybean→Corn healthy | 1 |
| **Total** | **8** in 10,709 |

> R12: EfficientNet-B0 cross-crop healthy confusion total is 8/10,709 — suspicion from SmallCNN (20/32 errors) does not transfer.

## Results — PlantDoc (Domain-Shift, Planned S11–S13)

> Not yet evaluated. See ROADMAP.md S11–S13 for planned protocol.

| Metric | Planned | Notes |
|---|---|---|
| **Mapped-subset accuracy** | To be reported | 28 of 38 PlantVillage classes have PlantDoc counterparts |
| **Out-of-mapping rate** | To be reported | Predictions on classes PlantDoc does not contain |
| **Per-class recall** | To be reported | Zero-support classes must be listed |
| **ECE** | To be measured | Calibration on field data is unmeasured |

## Run Details: R2 — SmallCNN

| Field | Value |
|---|---|
| **Run date** | 2026-10-25 |
| **Architecture** | SmallCNN (from scratch) |
| **Seed** | 42 |
| **GPUs** | 2× Tesla T4 (Kaggle) |
| **Config file** | `configs/protocol.json` v1 |
| **ONLY_ARCH** | `smallcnn` |
| **Batch size** | 128 |
| **Image size** | 224 × 224 pixels |
| **Epochs** | 12 (best epoch selected by val macro-F1) |
| **Mixed precision** | FP16 |
| **Parameters** | 398,662 |

### Results — PlantVillage Validation Split

| Metric | Value |
|---|---|
| **Accuracy** | 96.01% |
| **Macro precision** | 94.29% |
| **Macro recall** | 95.40% |
| **Macro-F1** | 94.44% |
| **Balanced accuracy** | 96.02% (= macro recall per D07) |

### Results — Locked PlantVillage Test Split

| Metric | Value |
|---|---|
| **Accuracy** | 96.95% |
| **Balanced accuracy** | 96.02% |
| **Macro-F1** | 96.07% |

> SmallCNN and ResNet18 were reproduced on CPU; parity check lives in `scripts/smoke_test_engine.py`.

## Run Details: R3 — ResNet18

| Field | Value |
|---|---|
| **Run date** | 2026-10-25 |
| **Architecture** | ResNet18 (ImageNet-pretrained, fine-tuned) |
| **Seed** | 42 |
| **GPUs** | 2× Tesla T4 (Kaggle) |
| **Config file** | `configs/protocol.json` v1 |
| **ONLY_ARCH** | `resnet18` |
| **Batch size** | 64 |
| **Image size** | 224 × 224 pixels |
| **Epochs** | 8 (best epoch) |
| **Mixed precision** | FP16 |
| **Parameters** | 11,196,006 |

### Results — PlantVillage Validation Split

| Metric | Value |
|---|---|
| **Accuracy** | 96.15% |
| **Macro precision** | 96.06% |
| **Macro recall** | 95.61% |
| **Macro-F1** | 95.46% |
| **Balanced accuracy** | 95.61% (= macro recall per D07) |

### Results — Locked PlantVillage Test Split

| Metric | Status |
|---|---|
| **No saved locked-test result** | ResNet18 has no saved result on this locked test split (README §43) |

> **D20**: `resnet18_last.pt` cannot reproduce its `manifest_sha256` — chain of custody breaks (KNOWN_ISSUES.md D20).

## Run Details: R4 — ConvNeXt-Tiny

| Field | Value |
|---|---|
| **Run date** | 2026-10-25 |
| **Architecture** | ConvNeXt-Tiny |
| **Seed** | 42 |
| **GPUs** | 2× Tesla T4 (Kaggle) |
| **Config file** | `configs/protocol.json` v1 |
| **ONLY_ARCH** | `convnext_tiny` |
| **Batch size** | 32 (smallest batch due to GPU memory) |
| **Image size** | 224 × 224 pixels |
| **Epochs** | 1 (pilot; epoch-1 scores not results — LR still warming up) |
| **Mixed precision** | FP16 |
| **Parameters** | 27,849,350 |

### Pilot Results — Epoch 1

| Metric | Value |
|---|---|
| **train img/s** | 194 |
| **min/epoch** | 3.15 |
| **macro-F1** | 0.060 (learning rate still ramping with 10% warmup over 12 epochs) |

> Epoch-1 scores are not results; the learning rate is still warming up. Artifacts in `reports/model_comparison/pilot/`.

## Protocol Configuration (`configs/protocol.json` v1)

| Parameter | Value | Note |
|---|---|---|
| **image_size** | 224 | — |
| **epochs** | 12 | — |
| **patience** | — | — |
| **seed** | 42 (per-run override possible) | — |
| **eval_batch** | — | — |
| **workers** | — | — |
| **augmentation** | Protocol-specified training-only | — |
| **loss** | — | — |
| **EMA** | — | — |
| **LR** | Per-architecture: EfficientNet-B0 uses warmup+cosine; SmallCNN needs higher LR (from scratch) | Primary cause of R12 invalidation (weight decay 0.05 vs ref 1e-4) |
| **batch** | 128 global (4×32 per GPU) | Reference used 32; 4× fewer optimizer steps contributed to sweep invalidation |
| **weight_decay** | 0.05 | Differs from reference 1e-4 (R12) |
| **scheduler** | warmup+cosine | Reference used constant (R12) |
| **label smoothing** | Present | Reference absent (R12) |
| **class weights** | Present | Reference absent (R12) |

## Sweep v1 Invalidation (R12)

| Detail | Value |
|---|---|
| **Runs** | 12 (4 architectures × 3 seeds) |
| **Date** | 2026-10-25 |
| **Defect** | All architectures landed far below established results; backbone never fine-tuned |
| **Cause** | Four settings differ from reference: weight decay (0.05 vs 1e-4), global batch (128 vs 32), LR schedule (warmup+cosine vs constant), label smoothing + class weights (present vs absent) |
| **Fix** | Protocol v2 uses proven optimisation settings; augmentation/class balancing move to separate arm; two-epoch gate (B0 ≥ 0.95 val accuracy by epoch 2) before full training |
| **Status** | S9 (12-run sweep) **NEXT** — needs Kaggle; gate before starting |

## Gradient-CAM Implementation

| Component | Status | Artifact |
|---|---|---|
| **Grad-CAM code** | Implemented | `src/plant_disease/gradcam.py` |
| **Target layer** | Determined per architecture | `gradcam.py::target_layer_for()` |
| **Overlay** | Tints whole frame; mask below ~6% activation recommended | `gradcam.py::overlay()` |
| **Report script** | Available | `scripts/gradcam_report.py` |
| **Grad-CAM audit** | Not yet run per ROADMAP.md | S14 planned: Grad-CAM on cross-crop confusions, high-confidence errors, correct-but-suspicious cases |

## Calibration Code

| Component | Status | Artifact |
|---|---|---|
| **Calibration code** | Implemented | `src/plant_disease/calibration.py` |
| **ECE computation** | Available | On test: 0.0022 (EfficientNet-B0) |
| **MCE computation** | Available | 0.53 (single-bin extreme) |
| **Reliability bins** | Generated | `test/reliability_bins.csv` |
| **Abstention curve** | Generated | Coverage vs retained accuracy at multiple thresholds |

## Experiment Log Obsidian Links

- `[[WALLET_INDEX.md]]` — central navigation hub
- `[[PROJECT_STATUS.md]]` — current project state
- `[[MODEL_CARD.md]]` — model-specific card for each run
- `[[DATASET_CARD.md]]` — which datasets each run used
- `[[KNOWN_ISSUES.md]]` — D20, D27, D12, D13, D28
- `[[RESEARCH_CLAIM.md]]` — H1–H4 this run subject to
- `[[ROADMAP.md]]` — which step each run supports
- `[[configs/protocol.json]]` — the configuration that produced this run
- `[[results/registry.json]]` — registered metrics with artifact paths
- `[[checkpoints/]]` — the checkpoint(s) from this run
- `[[scripts/smoke_test_engine.py]]` — regression test for this engine
- `[[scripts/evaluate_locked_test.py]]` — evaluator that produced test metrics

---
*This experiment log is the authoritative source for run provenance in the Obsidian vault. Linked from `WALLET_INDEX.md`, `PROJECT_STATUS.md`, `MODEL_CARD.md`, and `DATASET_CARD.md`. Updated whenever new runs are executed, checkpoints re-trained, or evaluation protocols change.*

*Generated: 2026-10-25*

## 2026-10-26 — sweep v2, overnight, 14 runs

**Notebook:** `notebooks/08_Overnight_Sweep.ipynb` · **Protocol:** `configs/protocol_v2.json`
**Archive:** `overnight_runs.zip` (2.12 GB) · **Hardware:** Kaggle 2× Tesla T4

Gate released the sweep: EfficientNet-B0 seed 42 reached 0.9836 validation
accuracy after one epoch (original run: 0.9826), 2.15 min/epoch.

| architecture | n | macro-F1 mean | sd | 95% CI | coupling |
|---|---:|---:|---:|---|---|
| convnext_tiny | 3 | 0.98951 | 0.00028 | 0.98881 – 0.99021 | random shuffle, constant LR |
| efficientnet_b0 | 5 | 0.98946 | 0.00171 | 0.98734 – 0.99158 | random shuffle, constant LR |
| resnet18 | 3 | 0.98460 | 0.00166 | 0.98048 – 0.98871 | random shuffle, constant LR |
| smallcnn | 3 | 0.91599 | 0.00865 | 0.89450 – 0.93749 | random shuffle, constant LR, lr 1e-3 |

- 14/14 runs completed; `_overnight_log.json` has no failed, timeout or error entry.
- All runs share dataset fingerprint `01c1fe7339df2701…`.
- Learning rate flat at 1.0e-04 (SmallCNN 1.0e-03) in every history, as protocol v2 requires.
- Reference and ConvNeXt differ by 0.00005; the B0 seed spread is 0.00171, so the
  two are indistinguishable. ResNet18 is 0.49 pp behind, SmallCNN 7.35 pp behind.
- Disclosures: reference mean is 0.42 pp below the published 0.99369 because runs
  early-stop (best epochs 4–10); SmallCNN replicates at 91.60%, not the 96.01%
  recorded in phase 2 — unresolved.

Full analysis: `reports/model_comparison/SWEEP_V2_RESULTS.md`. Registered in
`results/registry.json` under `sweeps.sweep_v2_overnight`.

## 2026-10-26 — S11/S12/S13, PlantDoc field evaluation, H1 confirmed

**Model:** EfficientNet-B0 published checkpoint (`90e003cf…`, 0.99369 PV val macro-F1)
**Data:** PlantDoc test, 236 images, 27 mapped classes, CC BY 4.0
**Command:** `python scripts/evaluate_plantdoc.py --checkpoint checkpoints/efficientnet_b0/efficientnet_b0_best.pt --arch efficientnet_b0 --split test --device cpu`
**Artifacts:** `reports/field_gap/b0_plantdoc_test/`, `reports/field_gap/FIELD_GAP_RESULTS.md`

| metric | PlantVillage | PlantDoc |
|---|---:|---:|
| accuracy | 0.9928 (5-seed) | 0.1737 (CI 0.1271–0.2246) |
| mean per-class recall | 0.9939 | 0.1746 |
| macro-F1 | 0.9895 (5-seed) | 0.1798 |
| mean confidence | 0.998 | 0.7522 |
| ECE | 0.0026 | 0.5784 |

- All 27 mapped classes fell: mean drop 0.8192, sd 0.1877, median 0.8508.
- Reliability: the 0.95–1.00 confidence band holds 65 images at 0.292 accuracy and
  0.986 mean confidence.
- 25.42% of predictions (60/236) land on PlantVillage classes PlantDoc lacks.
- **Pipeline control:** the same script on 300 PlantVillage validation images scores
  0.9967, so the collapse is domain shift rather than an inference bug.
- One PlantDoc class has 0 test images and is excluded from macro averages; macro
  uses classes with non-zero support only.

H1 verdict: **confirmed**, magnitude far outside the 9.75 pp CI width.
Registered in `results/registry.json` as `s11_efficientnet_b0_plantdoc_test`.

## 2026-10-26 — Stages D, E, F

### D — mechanism (reports/mechanism/MECHANISM_RESULTS.md)

Occlusion probe, 4 treatments, both domains. The first probe was discarded: filling
the border with grey also erased part of the leaf, collapsing both domains and
measuring scale sensitivity instead of background use. The rebuilt probe replaces
only non-green pixels via an HSV leaf mask.

| | PlantVillage (n=400) | PlantDoc (n=236) |
|---|---:|---:|
| baseline accuracy | 0.9975 | 0.1737 |
| background → grey | **0.4325 (−56.5 pp)** | 0.1441 (−3.0 pp) |
| background → noise | 0.1450 (−85.3 pp) | 0.0551 (−11.9 pp) |
| centre crop | 0.8100 (−18.8 pp) | 0.1992 (+2.5 pp) |
| flip rate, background → grey | 56.8% | 55.9% |

Grad-CAM: 18 overlays on PlantDoc (15 high-confidence-wrong, 39 cross-crop, only
6 high-confidence-right available). CAM mass inside the leaf mask 0.735 against a
mask covering 0.613 of the frame — **1.20× enrichment, weak**. H2 therefore
partially supported: the model is background-dependent in the lab, but it is not
simply looking at soil.

S16 human labelling outstanding (`labelling_sheet.csv`). S18 blocked: no
licence-clean background corpus disjoint from PlantDoc.

### E — usefulness (reports/field_gap/USEFULNESS_RESULTS.md)

| model | PV macro-F1 | PlantDoc acc | PlantDoc ECE |
|---|---:|---:|---:|
| convnext_tiny | 0.98951 | **0.2627** | 0.4440 |
| resnet18 | 0.98460 | 0.1822 | 0.4940 |
| efficientnet_b0 | 0.98946 | 0.1737 | 0.5808 |
| smallcnn | 0.91599 | 0.0678 | 0.7942 |

Abstention at 90% coverage: reference gains 0.8 pp (17.37 → 17.45). SmallCNN's
confidence is anti-correlated with correctness (50% coverage lowers accuracy to
5.08%). **H4 confirmed**: field confidence carries almost no information.

**Architecture ranking reverses.** ConvNeXt-Tiny and B0 differ by 0.00005 macro-F1
on PlantVillage and by 8.9 pp accuracy on PlantDoc. Lab accuracy does not rank
candidates correctly for field use.

### F — lock down

- `tests/` 17 tests: split integrity, locked-test disjointness, checkpoint contract
  and parameter counts, metric sanity (balanced_accuracy ≡ macro_recall; the
  zero-support macro trap), evaluator output contract. **17 passed.**
- `scripts/build_tables.py` renders 4 tables from the registry; `--verify` fails on
  an unbacked claim, a missing artifact, or a stale table. Deleting the registry
  exits 1 by design.
- `scripts/run_all.py --check` runs all six local gates: **6/6 passed**.
- `requirements.lock` pinned to the versions the checks were verified against.

Two test failures found real drift: resnet18 had been recorded as 11,689,512
params (the ImageNet 1000-class model) when the 38-class checkpoint is 11,196,006,
and `classification_metrics()` does not take `num_classes`. Both corrected.

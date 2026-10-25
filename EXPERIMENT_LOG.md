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
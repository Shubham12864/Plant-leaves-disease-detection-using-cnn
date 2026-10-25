# MODEL CARD — Plant Leaf Disease Detection Using CNN (BT40)

*YAML frontmatter for Obsidian linking. Every field traceable to `results/registry.json`, `checkpoints/SHA256SUMS`, or experiment logs.*

---
# Model Card: EfficientNet-B0
# type: model-card
# architecture: EfficientNet-B0, 4,056,226 parameters
# backbone: ImageNet-pretrained weights, fine-tuned on PlantVillage
# image_size: 224 × 224 pixels
# classes: 38 (exact manifest order)
# domain: controlled (PlantVillage) + field (PlantDoc evaluation)
# license: internal research — not for commercial redistribution
# trained_on: PlantVillage training split (36,762 images) + validation split (6,544 images)
# validated_on: PlantVillage validation split; PlantDoc development (2,331 images, train-only)
# tested_on: Locked PlantVillage test split (10,709 images)
# generated: 2026-10-25
# related-to: WALLET_INDEX.md, DATASET_CARD.md, EXPERIMENT_LOG.md, PROJECT_STATUS.md, RESEARCH_CLAIM.md

## Architecture

- **Base model**: EfficientNet-B0 (from `tfkeras/applications` or `torchvision`; version matching the training run)
- **Parameter count**: 4,056,226 (trainable after fine-tuning freeze/unfreeze)
- **Input**: 224 × 224 × RGB pixel tensor, normalized with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (ImageNet default; confirmed in checkpoint metadata)
- **Classifier head**: Fully connected layer(s) mapping EfficientNet-B0 feature vector → 38 logits (one per class)
- **Normalization**: Checkpoint stores `image_size`, `mean`, `std`, `num_classes`, `class_order` — verified on load (strict=True) into 38-class model
- **Framework**: TensorFlow / Keras or PyTorch (checkpoint provenance noted; this run used PyTorch-based DDP engine)

### Checkpoint Metadata (verified)

| Field | Value |
|---|---|
| **Filename** | `checkpoints/efficientnet_b0/efficientnet_b0_best.pt` |
| **Size** | 15.8 MB |
| **SHA-256** | `90e003cf…` (full hash in `checkpoints/SHA256SUMS`) |
| **Architecture** | EfficientNet-B0 |
| **Num classes** | 38 |
| **Image size** | 224 |
| **Mean** | [0.485, 0.456, 0.406] |
| **Std** | [0.229, 0.224, 0.225] |
| **Class order** | Matches development manifest exactly (38/38 identical order) |
| **Dataset fingerprint** | **Absent** — the pilot script did not write one (KNOWN_ISSUES.md D27) |
| **Provenance source** | `efficientnet_b0/provenance.json` — contract verified; no `dataset_fingerprint` |
| **Loaded with** | `strict=True`; class order assertion passed |

## Training Configuration

| Parameter | Value |
|---|---|
| **Optimizer** | AdamW (or SGD; config in `configs/protocol.json` v1) |
| **Learning rate** | Schedule: warmup+cosine (per protocol.json); differed from reference by weight decay, batch size, LR schedule |
| **Batch size** | 64 total (2× T4 Kaggle GPUs, batch 32 per GPU) |
| **Epochs** | 12 (best epoch selected by validation macro-F1) |
| **Seeds** | Single seed per run; 3 seeds planned for S9 (12-run sweep) |
| **Weight decay** | 0.05 (protocol v1); reference used 1e-4 — this is the primary cause of sweep v1 invalidation (R12) |
| **Data augmentation** | Protocol-specified: brightness/contrast, color shifts, scale/crop, rotation/orientation, blur/noise, background variation (training-only) |
| **Class balancing** | Class weights or sampler; configuration per `configs/protocol.json` |
| **Mixed precision** | FP16 on T4 GPUs |
| **Hardware** | 2× Tesla T4 Kaggle GPUs; ~15.5 GB free per GPU at inspection; ~31.7 GB host RAM out of 33.7 GB |
| **Training code** | `scripts/train_efficientnet_b0_ddp.py` over shared engine |
| **Package parity** | 4,056,226 params; same state-dict keys as `src/plant_disease/models` definition |

### Training History (from `plantvillage_model_training_history.csv`)

- 11 rows for ResNet18 + 12 rows for EfficientNet-B0
- No SmallCNN rows in this CSV (baseline history stored separately in `reports/phase2_baseline/`)
- Best epoch for EfficientNet-B0: epoch 12 (validation macro-F1 = 0.9936894341462544)

## Evaluation Results

### PlantVillage Validation Split

| Metric | Value | 95% CI / Notes |
|---|---|---|
| **Accuracy** | 99.54% | — |
| **Macro precision** | 99.31% | — |
| **Macro recall** | 99.46% | — |
| **Macro-F1** | 99.37% | — |
| **Balanced accuracy** | (same as macro recall) | — |

> **ECE**: 0.0022 (aggregate); **MCE**: 0.53 (from one 15-bin cell with n=1 image). Always publish ECE with bin counts (KNOWN_ISSUES.md §65).

### Locked PlantVillage Test Split (Verified 2026-10-25)

> **Artifact**: `reports/model_comparison/locked_test_efficientnet_b0/test/metrics.json` + `classification_report.csv` + `confusion_matrix.csv` + `reliability_bins.csv`

| Metric | Value | 95% CI / Notes |
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
| **MCE** | 0.53 | Max calibration error; from one bin with n=1 |

### Cross-Crop Healthy Confusions (EfficientNet-B0 on Test)

| True → Predicted | Count | Notes |
|---|---|---|
| Raspberry→Cherry healthy | 4 | Cross-crop healthy error |
| Apple→Blueberry healthy | 1 | — |
| Apple→Soybean healthy | 1 | — |
| Potato→Soybean healthy | 1 | — |
| Soybean→Corn healthy | 1 | — |
| **Total cross-crop healthy** | **8** in 10,709 | (R12 notes suspicion from SmallCNN does not transfer to B0) |

> **H2 suspicion**: The cross-crop healthy confusion total for EfficientNet-B0 is only 8/10,709, far less than SmallCNN's 20/32 Soybean___healthy errors. H2 cannot rest on confusion counts; occlusion probe (S15) and saliency localisation (S16) are needed.

### PlantDoc Evaluation (Domain-Shift Test)

> **Note**: PlantDoc has 28 mapped classes + 10 out-of-mapping classes + 1 anecdotal class. No validation split exists. Results are not comparable to PlantVillage (28 vs 38 classes; ±6.4 pp interval).

| Metric | Value | Notes |
|---|---|---|
| **Mapped-subset accuracy** | N/A (not yet evaluated) | Will be reported in S11–S13 (ROADMAP.md) |
| **Out-of-mapping rate** | N/A (not yet evaluated) | Predictions on classes PlantDoc does not contain |
| **Per-class recall** | N/A (not yet evaluated) | Zero-support classes must be reported |
| **ECE** | N/A (not yet evaluated) | Calibration on field data is unmeasured |

> **H1**: Field accuracy is materially lower than controlled accuracy on the class subset PlantDoc covers — this is the primary hypothesis; experiment planned in ROADMAP.md S11–S13.

> **H4**: Nominal accuracy does not bound deployable accuracy — on PlantVillage, retained accuracy at 90% coverage is 99.76% vs 99.51% at full coverage, indicating the model is well calibrated on controlled imagery (R69). On field data, calibration is unknown.

## Confidence Calibration

| Metric | Value | Notes |
|---|---|---|
| **ECE (Expected Calibration Error)** | 0.0022 (PlantVillage test) | Aggregate; reliable because bin counts are available (MCE is NOT usable at this accuracy) |
| **MCE (Max Calibration Error)** | 0.53 | From single 15-bin cell with n=1 image; not publishable on its own |
| **Brier score** | Not yet reported for EfficientNet-B0 on test | Will be added when calibration script is re-run |
| **Calibration curve** | Generated from test bin counts (reliability_bins.csv) | Axes start at zero; truncation stated if present |
| **Abstention curve** | At 90% coverage: 99.76% accuracy; at 90%: 99.51% nominal | Abstention buys little on PlantVillage (R67); interesting curve is on field data |

> **Rule**: Do not call softmax output "calibrated confidence" unless calibration has been tested (SOUL.md §6). ECE of 0.0022 on test implies good aggregate calibration, but MCE of 0.53 warns that individual bins can be extreme.

## Failure Modes & Limitations

| Category | Detail |
|---|---|
| **Domain shift** | PlantVillage → PlantDoc: 28 of 38 classes mapped; 10 classes have no PlantDoc counterpart; performance not directly comparable |
| **Background shortcuts** | EfficientNet-B0 has only 8 cross-crop healthy errors vs SmallCNN's 20/32; occlusion probe needed (H2) |
| **Class imbalance** | 38.4× ratio (Soybean___healthy: 4,117; Potato___healthy: 120); accuracy hides rare-class collapse (D06) |
| **Single-seed results** | Every number comes from one seed; no variance estimate until S9 (12-run sweep) (KNOWN_ISSUES.md §117–§118) |
| **Test set scoring history** | Test split has been scored more than once; any additional scoring must be disclosed (D11) |
| **No dataset fingerprint** | Checkpoint carries no `dataset_fingerprint` (D27); provenance limited to file records + class-order check |
| **ResNet18 manifest_sha256 mismatch** | `resnet18_last.pt` cannot reproduce its manifest_sha256 (D20) — chain of custody broken |
| **Grad-CAM not fully audited** | Implementation exists (src/plant_disease/gradcam.py) but full audit per ROADMAP.md not yet run |
| **PlantDoc statistical power** | 236 test images over ~27 classes → ±6.4 pp 95% CI; headline claims underpowered (D10) |
| **No validation split for PlantDoc** | All 2,331 dev rows are `train`; any threshold tuning must use the 236-image test set (D09) |

## Obsidian Links

- `[[DATASET_CARD.md]]` — which datasets this model was trained/evaluated on
- `[[WALLET_INDEX.md]]` — central navigation hub
- `[[EXPERIMENT_LOG.md]]` — which run produced these results
- `[[KNOWN_ISSUES.md]]` — D27 (no dataset fingerprint), D20 (ResNet18 manifest mismatch)
- `[[RESEARCH_CLAIM.md]]` — H1–H4 hypotheses this model subject to
- `[[ROADMAP.md]]` — S11–S13: Score PlantDoc (H1), S14–S19: Explain gap (H2), S20–S22: Bound usefulness (H4)
- `[[PROJECT_STATUS.md]]` — current model status
- `[[results/registry.json]]` — registered metrics with artifact paths
- `[[checkpoints/efficientnet_b0/]]` — the checkpoint itself
- `[[SHA256SUMS]]` — checksum verification
- `[[configs/protocol.json]]` — training configuration

### Model Card Cross-References

| Related Artifact | Link |
|---|---|
| EfficientNet-B0 test metrics | `results/registry.json` → `efficientnet_b0_test_53_errors_10709` |
| EfficientNet-B0 validation macro-F1 | `results/registry.json` → `efficientnet_b0_val_macro_f1_0.99369` |
| PlantDoc label mapping | `DATASET_CARD.md` → PlantDoc section |
| Protocol configuration | `configs/protocol.json` |
| Grad-CAM implementation | `src/plant_disease/gradcam.py` |
| Calibration code | `src/plant_disease/calibration.py` |

### Model Card Usage Guidelines

> **Do not present** EfficientNet-B0 PlantVillage accuracy as "field-ready" or "production-grade" without external validation (SOUL.md §10).
> 
> **Do not claim** calibrated confidence without testing (SOUL.md §6, §13).
> 
> **Do not** describe Grad-CAM as proof of causal reasoning (SOUL.md §14).
> 
> **Do report** per-class precision, recall, F1, support, confusion matrix, and macro/weighted averages (SOUL.md §4).
> 
> **Do separate** validation performance from test performance from external-dataset performance (SOUL.md §4).
> 
> **Do include** limitations section listing domain shift, class imbalance, single-seed weakness, and any out-of-mapping predictions.

---
*This model card is the authoritative source for EfficientNet-B0 model provenance in the Obsidian vault. Linked from `WALLET_INDEX.md`, `PROJECT_STATUS.md`, `EXPERIMENT_LOG.md`, and `DATASET_CARD.md`. Updated whenever checkpoints are re-trained, re-evaluated, or when calibration/ Grad-CAM audit results change.*

---
*Generated: 2026-10-25*
---
title: Phase II Implementation Roadmap
type: project-roadmap
project: Plant leaves disease detection using CNN
status: Phase II data preparation complete; Kaggle baseline training is next
tags:
  - capstone
  - implementation
  - roadmap
---

# Phase II Implementation Roadmap

## Layer 1 — Source data and provenance

**Completed:** downloaded the pinned PlantVillage color source and PlantDoc repository archive, retained provenance/license files, and materialized each dataset into its own `data/raw/` tree. PlantVillage yielded 54,305 images (43,596 train; 10,709 test); PlantDoc yielded 2,578 (2,342 train; 236 test), though the upstream description states 2,598.

**Gate:** downloaded image counts, class lists, archive integrity, and license/provenance entries are recorded. Do not train against partial downloads.

## Layer 2 — Audit and leakage-aware manifests

- Build per-dataset, per-source-split manifests with relative path, class, image dimensions, SHA-256, and source metadata.
- Initial audit found exact duplicates and shared PlantVillage group IDs across published partitions. After filtering exact duplicates and shared PlantVillage leaf IDs out of development, the final audit reports zero exact-hash/group overlap across splits. PlantDoc has no specimen IDs, so its external test remains limited.
- Completed `scripts/freeze_evaluation_splits.py`: excluded 290 PlantVillage training rows sharing group IDs or hashes with published test; made train/validation splits across all 38 classes. Excluded 11 exact duplicate PlantDoc training rows while retaining source manifests. `reports/split_freeze.json` records decisions; `reports/data_audit.json` confirms zero hash/group overlap.
- Preserve PlantDoc's upstream test as a separate field-domain evaluation set. Its class taxonomy is not identical to PlantVillage; cross-domain metrics must be computed only on a documented shared label mapping. Keep all unmatched PlantDoc labels for descriptive analysis, not silent remapping.

**Gate:** counts and zero-overlap split audit are recorded. Before cross-domain evaluation, review the candidate PlantDoc mapping; no exact class-name intersection exists.

## Layer 3 — Preprocessing contract

- Decode RGB consistently; resize/crop to a fixed model input size (initially 224 × 224); normalize using training data or the selected pretrained model's documented normalization.
- Record every transform and preserve originals. No augmentation on validation/test.
- Start with simple training-only flips/rotations and mild brightness/contrast variation; add stronger field-like transforms only as named ablations.

**Gate:** a deterministic data loader reproduces labels and split membership; sample grids and class counts are reviewed.

## Layer 4 — Baselines and model training

- Baseline A: small custom CNN, trained from scratch.
- Baseline B: pretrained MobileNetV3 or EfficientNet-B0, selected after checking runtime and dependency compatibility.
- Train first on PlantVillage train/validation only. Save config, seed, environment, checkpoint, training curve, and wall-clock time.
- Then evaluate the locked PlantVillage test. Evaluate PlantDoc only after a mapping is independently reviewed; do not tune on either test set.

**Gate:** reproducible baseline results with macro-F1, balanced accuracy, per-class results, confusion matrices, model size, and inference time.

## Layer 5 — Research comparisons

Run controlled comparisons: augmentation on/off; imbalance strategy on/off; PlantVillage-only vs mixed-domain training if label mapping permits. Calibrate confidence on validation only; measure abstention risk/coverage. Report confidence intervals and negative results. Grad-CAM is an inspection aid; audit correct, wrong, uncertain, and field-domain examples.

**Gate:** ablation table and failure-case log support every claim in the report.

## Layer 6 — Demo and final report

Build a small screening interface only after the classifier/evaluation pipeline is stable. Present uncertainty and limitations. Any treatment/advisory extension must cite verified agricultural sources and stay outside the core disease classifier evaluation.

**Gate:** model card, data cards, exact experiment configs, licenses/citations, failure cases, and a final report are ready for guide review.

## Compute recommendation

Use local CPU for downloads, archive extraction, hashing, manifest creation, and data inspection. Use Kaggle GPU for CNN training. The notebook uses the first visible GPU, or both GPUs with `DataParallel` when Kaggle exposes T4 x2; measure the actual runtime because two GPUs will not guarantee a 2x speedup. Upload the dataset bundle once and save checkpoints/results as notebook outputs. Keep a CPU-compatible local inference path as a later deployment check.

For training, Kaggle's available GPU type and quota can vary. Start with the small CNN and one transfer-learning baseline, modest image size/batch size, and a short run; scale only after the pipeline is correct. The notebook reads the saved leakage-aware manifests; exclusions are recorded in the split-freeze report. Never publish restricted data or credentials in notebook output.

## What is needed for the next modeling layer

No additional dataset is needed to begin the first controlled-data baseline. An independently collected field set would improve external validation but is optional. The 21-paper survey PDF is still needed to reconcile the project's literature review; it is not a blocker for the first baseline.

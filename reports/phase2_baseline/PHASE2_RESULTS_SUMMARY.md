---
title: Phase 2 Baseline Results
project: Plant leaves disease detection using CNN
stage: Phase 2
tags: [capstone, plant-disease, results, kaggle]
---

# Phase 2 Baseline Results

## Run summary

- **Dataset:** PlantVillage
- **Classes:** 38
- **Images:** 36,762 train / 6,544 validation / 10,709 test
- **Model:** SmallCNN with channels 32, 64, 128, and 256
- **Training:** 12 epochs, batch size 64, AdamW, mixed precision
- **Hardware:** 2 Tesla T4 GPUs using DataParallel
- **Best checkpoint:** epoch 12, selected using validation macro-F1

## Held-out test results

| Metric | Score |
|---|---:|
| Accuracy | 96.95% |
| Balanced accuracy | 96.02% |
| Macro-F1 | 96.07% |

The test set was evaluated after selecting the checkpoint with the validation set. Validation images were not used to update model weights. These scores describe performance on the PlantVillage test split; they do not establish performance on field photographs or another dataset.

## Files

- `training_curves.png` — training loss and validation macro-F1 by epoch
- `training_history.csv` — per-epoch loss and validation metrics
- `locked_test_metrics.json` — overall held-out test scores
- `locked_test_classification_report.csv` — per-class precision, recall, F1, and support
- `locked_test_confusion_matrix.csv` — class-by-class prediction counts
- `experiment_config.json` — data counts, model, training settings, and runtime details
- `plantvillage_smallcnn_best.pt` — selected model checkpoint
- `phase2_baseline_artifacts.zip` — original Kaggle results bundle

## Suggested next work

1. Use the training curve and test metrics in the Phase 2 guide presentation.
2. Inspect the confusion matrix and per-class report; discuss classes with the weakest F1 or recall.
3. Evaluate on a larger independent field-photo collection after defining and documenting a compatible label map.
4. Keep this PlantVillage test score fixed as the baseline; do not tune model choices against this test set.

## Source of these values

Scores and settings above are transcribed from `locked_test_metrics.json` and `experiment_config.json` in this folder. The original files and checkpoint are retained alongside this summary.

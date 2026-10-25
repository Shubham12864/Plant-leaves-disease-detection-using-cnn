# PlantVillage model comparison

This folder reports PlantVillage results only. The scores do not show performance on field images.

## Same-split validation comparison

`plantvillage_validation_comparison.csv` and `plantvillage_validation_comparison.png` compare SmallCNN, ResNet18, and EfficientNet-B0 on the same 6,544-image validation split. The team selected checkpoints by validation macro-F1. The chart shows accuracy, macro precision, macro recall, and macro-F1.

EfficientNet-B0 has the highest validation macro-F1 (99.37%). ResNet18 reaches 95.46%. SmallCNN reaches 94.44%. SmallCNN is the trained-from-scratch baseline. ResNet18 and EfficientNet-B0 use ImageNet-pretrained weights and are fine-tuned for the 38 PlantVillage classes.

## Locked test comparison

`plantvillage_locked_test_comparison.csv` compares the saved locked-test results for SmallCNN and EfficientNet-B0 on the same 10,709-image test split. ResNet18 has no locked-test result in the available files. Do not infer one. EfficientNet-B0 reaches 99.51% accuracy and 99.31% macro-F1. SmallCNN reaches 96.95% accuracy and 96.07% macro-F1.

Do not compare a validation score with a test score. Use the three-model validation comparison for architecture selection and the locked-test results for final test reporting.

## Training curves

`plantvillage_training_curves.png` shows per-epoch training loss and validation macro-F1. SmallCNN uses the Phase 2 baseline history. ResNet18 and EfficientNet-B0 use the architecture-comparison histories. The CSV source is `plantvillage_model_training_history.csv`.

## Source files

- Validation scores, histories, and run details came from the Kaggle model-comparison output bundle.
- SmallCNN test scores and training history came from `reports/phase2_baseline/`.
- EfficientNet-B0 test scores were recorded from the Kaggle final-evaluation run. **Correction (2026-10-25):** the earlier wording said those scores had been "checked against the PlantVillage classification report and confusion matrix". Those files were never saved for EfficientNet-B0, so that check could not have happened. It is now reproducible instead: `notebooks/04_Locked_Test_Evaluation.ipynb` scores the locked test split with the restored checkpoint and writes the report and matrix to `reports/model_comparison/locked_test_efficientnet_b0/`.

## Known gaps in this folder

- `plantvillage_model_training_history.csv` contains ResNet18 and EfficientNet-B0 rows only. The SmallCNN series drawn in `plantvillage_training_curves.png` comes from `reports/phase2_baseline/training_history.csv`, not from this CSV.
- `plantvillage_validation_comparison.png` starts its y-axis at 90, which visually exaggerates the SmallCNN↔ResNet18 difference. Read the CSV, not the chart, for magnitudes.
- `balanced_accuracy` and `macro_recall` are the same quantity (mean per-class recall). They are not independent evidence.

Current plan and defect register: [`docs/ROADMAP.md`](../../docs/ROADMAP.md), [`docs/KNOWN_ISSUES.md`](../../docs/KNOWN_ISSUES.md).

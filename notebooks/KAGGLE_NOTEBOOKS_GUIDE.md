# Kaggle notebook guide

Use these notebooks in order. All model training and image evaluation run in Kaggle.

## Notebook 1: Train EfficientNet-B0

File: `01_Train_EfficientNet_B0.ipynb`

1. Add the prepared PlantVillage dataset.
2. Select **GPU T4 x2**. Enable Internet only if Kaggle must download ImageNet weights.
3. Run the notebook cells in order.
4. The notebook trains EfficientNet-B0 on the PlantVillage train split. It uses the validation split to select the best checkpoint. It does not open the locked test split.
5. Download `model_comparison_artifacts.zip` from the Kaggle Output panel.

The best model file is saved as `/kaggle/working/model_comparison/efficientnet_b0/efficientnet_b0_best.pt`. The result bundle contains the checkpoint, validation scores, training history, and chart. Add the checkpoint as a Kaggle Dataset for later notebooks.

The saved SmallCNN checkpoint is the baseline. Do not train SmallCNN again for this comparison. ResNet18 was trained in the earlier model-comparison run. Its validation history and score are in `reports/model_comparison/`.

## Notebook 2: Final PlantVillage test

File: `02_Final_Evaluation_PlantVillage.ipynb`

1. Add the PlantVillage dataset.
2. Add the checkpoint dataset for the model you selected.
3. Check `SELECTED_ARCHITECTURE` and `CHECKPOINT_PATH`. The default input path is `/kaggle/input/datasets/shubhamkumar12864/efficientnet/efficientnet_b0_best.pt`.
4. Confirm that you selected the model with validation results. Then set `CONFIRM_FINAL_MODEL_SELECTION = True`.
5. Run the cells.

The notebook scores the locked PlantVillage test split once. It saves metrics, a class report, and a confusion matrix under `/kaggle/working/final_evaluation_plantvillage/`.

## Notebook 3: Predict an image

File: `03_Use_Existing_Model_for_Images.ipynb`

1. Add the saved checkpoint as a Kaggle input.
2. Set the image path in the notebook to an image you want to classify.
3. Run the notebook.

This notebook only predicts a class. It does not train or evaluate a model.

## SmallCNN baseline reference

`Plant_leaves_disease_detection_Kaggle_Baseline.ipynb` records the earlier baseline training. Do not run it when you only want to compare the saved models. It trains SmallCNN again.

## Fair comparison

The three-model comparison uses the same 6,544-image PlantVillage validation split. It reports accuracy, macro precision, macro recall, macro-F1, model size, and available runtime data. The final test comparison includes only models with saved scores on the same 10,709-image PlantVillage test split. There is no ResNet18 locked-test score in the current results.

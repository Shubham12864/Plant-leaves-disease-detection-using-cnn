# Kaggle notebook guide

All model training and full-split inference run in Kaggle. Local machines are used for data preparation, manifests, documents and CPU-only code checks.

| Notebook | Purpose | Status |
|---|---|---|
| `01_Train_EfficientNet_B0_v2.ipynb` | Train a model on the PlantVillage development split (2× T4, DDP) | **current** |
| `04_Locked_Test_Evaluation.ipynb` | Verify a checkpoint and score the locked test split | **current** |
| `03_Use_Existing_Model_for_Images.ipynb` | Predict one image with a saved checkpoint | current, lightweight |
| `01_Train_EfficientNet_B0.ipynb` | Pilot training notebook (superseded) | retired — kept for provenance |
| `02_Final_Evaluation_PlantVillage.ipynb` | Pilot evaluation notebook (superseded) | retired — use 04 instead |
| `Plant_leaves_disease_detection_Kaggle_Baseline.ipynb` | SmallCNN baseline run | retired — reference only |

The two `_v2` / `04` notebooks are generated from repository scripts by `scripts/build_training_notebook.py` and `scripts/build_eval_notebook.py`. Both builders verify that the copy embedded in the notebook is byte-identical to the file in `scripts/`, so the notebook and the repository cannot drift apart. Edit the script, then regenerate; do not hand-edit the embedded cells.

---

## 1. `01_Train_EfficientNet_B0_v2.ipynb` — training

**Attach:** the PlantVillage dataset. **Accelerator:** GPU T4 x2.

Cells to edit: only the tunables block in section 1 (`ARCH`, `EPOCHS`, `LOCAL_BATCH_SIZE`, `WORKERS`, `LEARNING_RATE`, …).

Run order:

1. **Setup and preflight** — checks two GPUs, the manifest, the image paths and the cache disk budget.
2. **Build or reuse the decoded cache** — decodes each image once to `/kaggle/temp`. Keyed by a hash of the path list, so a changed split invalidates it automatically.
3. **Write the training program** — writes `train_ddp.py` to `/kaggle/working`.
4. **Smoke test** — three optimizer steps. Always run this first.
5. **Full training** — resumes from the last completed epoch if the session restarts.
6. **Results and bundle** — comparison table, curves, and the artifact ZIP.

**Produces**, under `/kaggle/working/model_comparison/`:

```
<arch>/<arch>_best.pt                     best weights (EMA if it won on validation)
<arch>/<arch>_last.pt                     full resume checkpoint
<arch>/training_history.csv               one row per epoch
<arch>/validation_metrics.json            full metric set incl. ECE, Brier, top-3
<arch>/validation_classification_report.csv
<arch>/validation_confusion_matrix.csv
validation_comparison.csv                 three-model comparison
training_curves.png
experiment_metadata.json
../model_comparison_artifacts_v2.zip      download this
```

Design notes worth knowing when reading results:

- The checkpoint stores `dataset_fingerprint`, so a resumed run refuses to continue on a different split.
- Class-balanced cross-entropy is on by default (`CLASS_WEIGHT_ALPHA = 0.5`) because the training set is 38.4× imbalanced and macro-F1 is the selection metric.
- The best checkpoint is selected on validation macro-F1 only. The test split is never opened here.
- `ARCH = "resnet18"` trains ResNet18 through the same engine, which is how the comparison stays on one code path.

---

## 2. `04_Locked_Test_Evaluation.ipynb` — verification and locked-test scoring

**Attach:** the PlantVillage dataset **and** the checkpoint dataset. No GPU strictly required, but attach one.

Cells to edit: `DATASET_ROOT` and `CHECKPOINT_PATH` in section 1.

What it does:

1. **Contract check** — loads with `weights_only=True`, loads the state dict with `strict=True`, compares the checkpoint's class order against the development manifest and raises if they differ, prints parameter count and SHA-256, and shows whether the file records a `dataset_fingerprint`.
2. **Evaluation** — scores validation (to reproduce the recorded score) and test (reporting pass), with bootstrap 95% confidence intervals, calibration, and selective risk/coverage.
3. **Error analysis** — top off-diagonal confusions, weakest classes, thin-support classes, and cross-crop healthy-class confusions.
4. **Evidence record** — writes `P0.1_VERIFICATION.md` containing every number plus the checkpoint hash, then zips the folder.

**Produces**, under `/kaggle/working/p01_verification/`:

```
validation/{metrics.json,classification_report.csv,confusion_matrix.csv,
            selective_risk_coverage.csv,reliability_bins.csv,reliability.md}
test/{same file set}
P0.1_VERIFICATION.md
```

**Reading rule:** if validation macro-F1 does not reproduce the value recorded in the checkpoint within about 0.002, stop and investigate before quoting any test number. A small gap is expected between GPU FP16 training-time validation and CPU FP32 re-evaluation; a large gap means the checkpoint and the recorded score disagree.

---

## 3. `03_Use_Existing_Model_for_Images.ipynb` — single-image prediction

Attach the checkpoint, set the image path, run. Predicts one class. Does not train or evaluate.

---

## Compute and cost notes

- Pilot EfficientNet-B0 run: 104 minutes for 12 epochs, with per-epoch time climbing from 1.7 to 15.65 minutes because the data loader ran single-threaded against a 5.5 GB memory map. The v2 engine fixes the data path; expect roughly 1–1.5 minutes per epoch on 2× T4.
- Training writes about 45 MB (best) and 134 MB (resume) per architecture. Download the ZIP before the session ends; `/kaggle/working` is not durable.

---

## What is deliberately not in Kaggle

- Grad-CAM (`scripts/gradcam_report.py`) and PlantDoc evaluation are separate higher-phase artifacts; they consume the same checkpoints.
- The locked test split is not used for model selection anywhere. Any additional scoring of it must be disclosed in the report as a reporting pass, not a blind holdout.

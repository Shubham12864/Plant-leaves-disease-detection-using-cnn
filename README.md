# Plant leaves disease detection using CNN

## Project status

This repository contains the Phase 2 work. It does not contain the final project code.

Phase 2 includes data preparation, a trained SmallCNN baseline, and a held-out test report. The baseline uses the PlantVillage dataset. It classifies 38 classes.

## Baseline results

| Measure | Result |
|---|---:|
| Test accuracy | 96.95% |
| Test balanced accuracy | 96.02% |
| Test macro-F1 | 96.07% |
| Test images | 10,709 |
| Best checkpoint | Epoch 12 |
| Validation macro-F1 | 94.44% |

The model uses a four-stage SmallCNN. The channel sizes are 32, 64, 128, and 256. The training run used two Tesla T4 GPUs. It used a batch size of 64 and mixed precision.

The validation set selected the checkpoint. The test set measured the selected checkpoint once. The validation and test images did not update the model weights.

These results apply to the PlantVillage test split. They do not show performance on field images or on PlantDoc.

See [the Phase 2 results summary](reports/phase2_baseline/PHASE2_RESULTS_SUMMARY.md) for the result files and next steps.

## Project files

| Folder or file | Purpose |
|---|---|
| `notebooks/` | Kaggle training notebook |
| `scripts/` | Dataset download, audit, and manifest scripts |
| `src/` | Project source code |
| `configs/` | Data contract and run settings |
| `data/manifests/` | Image paths, labels, and split records |
| `reports/` | Data audits and Phase 2 results |
| `MEMORY.md` | Project memory and decisions |
| `ARCHITECTURE.md` | System design notes |
| `PHASE-II-ROADMAP.md` | Phase 2 work plan |
| `Literature-Survey-Evidence-Register.md` | Evidence register for the literature survey |

## Data

Git does not contain image files. The image files are large and have separate source terms. Keep the files in `data/raw/` on the local computer. The `.gitignore` file excludes them from Git.

The prepared Kaggle input is also excluded from Git. To build it again, run `scripts/prepare_kaggle_upload.ps1`. Then upload the generated dataset to Kaggle.

Read [data/README.md](data/README.md) before you use the datasets. It records the sources, split rules, and known count differences. Do not redistribute dataset images until you check the source terms.

## Run the training notebook

1. Upload the prepared PlantVillage dataset to Kaggle.
2. Add that dataset to the Kaggle notebook.
3. Turn on the GPU accelerator.
4. Open `notebooks/Plant_leaves_disease_detection_Kaggle_Baseline.ipynb`.
5. Run the notebook cells in order.

The notebook saves the model, metrics, reports, and charts under `/kaggle/working/phase2_baseline/`. It also creates `phase2_baseline_artifacts.zip`.

## Limits and next work

The PlantVillage images have controlled backgrounds. High test scores on this dataset do not prove field performance.

Phase 2 does not include a public web demo, a final model, Grad-CAM, or a completed PlantDoc evaluation. The next work is to review the class errors and evaluate compatible PlantDoc classes. Record all class mappings and exclusions before you report those results.

## Project documents

- [Project memory](MEMORY.md)
- [Architecture notes](ARCHITECTURE.md)
- [Phase 2 roadmap](PHASE-II-ROADMAP.md)
- [Literature survey evidence register](Literature-Survey-Evidence-Register.md)

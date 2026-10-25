# Known issues — defect register

Every entry is a real defect with evidence in the working tree, not a wish-list.
Severity: **S1** = an evaluator or reviewer can catch a false or unsupported claim · **S2** = weakens the science · **S3** = hygiene/engineering.

"Phase" points at the work item in `docs/ROADMAP.md` that closes it.

| ID | Sev | Issue | Evidence | Phase |
|---|---|---|---|---|
| D01 | S1 | **Checkpoints are git-ignored, so a fresh clone cannot reproduce anything.** The restored EfficientNet-B0 artifact is excluded by the same rule that excludes the others. | `.gitignore:7` `checkpoints/**`; `git check-ignore` confirms all three checkpoints are ignored | 0.7 |
| D02 | S1 | **`plantvillage_development.summary.json` describes a superseded split.** It reports 43,596 rows / 37,057 train / 6,539 validation; the manifest holds 43,306 / 36,762 / 6,544. Nothing regenerates it, so it can drift again. | `data/manifests/plantvillage_development.summary.json` `"rows": 43596` vs `pandas` count on `plantvillage_development.csv` | 0.5 |
| D03 | — | ~~A report claimed a check that never happened.~~ **Resolved 2026-10-25** — the check now exists: the locked test split was scored, producing `classification_report.csv` and `confusion_matrix.csv` under `reports/model_comparison/locked_test_efficientnet_b0/test/`. | `reports/model_comparison/README.md` corrected | done |
| D04 | S1 | **Four planning documents contradict the code.** They still say no model was trained and no result exists. | `ARCHITECTURE.md:9`, `ARCHITECTURE.md:184`, `MEMORY.md:166`, `PHASE-II-ROADMAP.md:5`, `Literature-Survey-Evidence-Register.md:7` | 0.6 |
| D05 | S1 | **Damaged front-matter title in an uncommitted working-tree change.** | `git diff` on `reports/guide_brief/UPDATED_LITERATURE_SURVEY.md`: `-title: Updated Literature Survey — ...` → `+title:  — ...` | 0.5 |
| D06 | S2 | **Macro-F1 over a constant 38-class label list collapses on subsets.** Scoring 256 validation rows gave accuracy 0.996 but macro-F1 **0.105**, because 35 classes had zero support. Any partial-split table computed this way is meaningless unless zero-support classes are reported. | Observed while testing `scripts/evaluate_locked_test.py --limit 256`; `classes_with_zero_support` field added to `metrics.json` in response | 1.5 |
| D07 | S2 | **`balanced_accuracy` and `macro_recall` are the same number** presented as two independent metrics. | Both are mean per-class recall; identical to 8 decimals in every table in `reports/model_comparison/` | 1.4 |
| D08 | S2 | **PlantDoc accuracy is not comparable to PlantVillage accuracy**, yet the project language implies it is. Only a subset of PlantVillage classes has a PlantDoc counterpart, and PlantDoc has 28 classes to PlantVillage's 38. | 28 PlantDoc labels in `data/manifests/plantdoc_train.summary.json`; 38 PlantVillage classes | 2.1 |
| D09 | S2 | **PlantDoc has no validation split.** All 2,331 development rows are `train`, so any confidence threshold tuned on PlantDoc would have to use the 236-image test set. | `plantdoc_development.csv` split counts: `{'train': 2331}` | 2.5, 1.3 |
| D10 | S2 | **PlantDoc's statistical power is too low for headline claims.** 236 test images over 27 classes is 4–12 per class; a 95% interval on accuracy is roughly ±6.4 pp. | Manifest class counts; interval arithmetic | 2.1, 2.2 |
| D11 | S2 | **The test split is no longer pristine.** It has been scored more than once historically; any additional scoring must be disclosed as a reporting pass, not a blind holdout. | README reports SmallCNN then EfficientNet test results from separate runs | 0.2, 6.3 |
| D12 | S2 | **Two training engines coexist.** The pilot script still ships beside the v2 engine, so "which script trained this model" is ambiguous. | `scripts/train_efficientnet_b0_ddp.py` and `scripts/train_ddp.py` | 4.1 |
| D13 | S2 | **Two evaluation paths coexist and disagree.** `src/plant_disease/evaluate.py` refuses any split other than validation; the new evaluator accepts any manifest. The hard-coded image root is fixed (`resolve_image_root`), but the old entry point remains. | `evaluate.py:22-23` | 4.1, 5.1 |
| D14 | S2 | **Duplicate metric implementations will drift.** Metrics live in `src/plant_disease/metrics.py`, `calibration.py`, and again inline in `train_ddp.py` and `evaluate_locked_test.py`. | Four copies of ECE/Brier/macro-F1 logic | 1.6 |
| D15 | S2 | **`run_accuracy_pipeline.py` writes a second, divergent source of truth** with different columns from the tracked comparison CSV. | `run_accuracy_pipeline.py:28`, output `reports/model_comparison/accuracy_pipeline/validation_comparison.csv` | 1.4 |
| D16 | S3 | **Dead build script.** `build_efficientnet_notebook.py` reads `reports/model_comparison/previous/validation_comparison.csv`, which does not exist, so it cannot regenerate notebook 01. | `build_efficientnet_notebook.py:10` | 0.6 |
| D17 | S3 | **Orphan bytecode with no source.** `scripts/__pycache__/package_plantdoc_external_test.cpython-314.pyc` has no matching `.py`. | Directory listing; `test -f` returns false | 5.5 |
| D18 | S3 | **Old notebooks still redeclare model classes** instead of importing `src/plant_disease/models`, so they can drift from the package. | `notebooks/02_Final_Evaluation_PlantVillage.ipynb`, `03_Use_Existing_Model_for_Images.ipynb`, baseline notebook | 4.1 |
| D19 | S3 | **The Kaggle guide describes the retired notebook** and does not mention the v2 training notebook or the evaluation notebook. | `notebooks/KAGGLE_NOTEBOOKS_GUIDE.md` references only 01/02/03 | 0.6 |
| D20 | S3 | **The run recorded in `resnet18_last.pt` cannot reproduce its `manifest_sha256`**, so the chain of custody breaks exactly where it is asserted. | Audit tried 24 serializations of the shipped manifest, none matched | 0.7 |
| D21 | S3 | **Training-curve provenance is incomplete.** The README says the curves use the Phase 2 SmallCNN history, but `plantvillage_model_training_history.csv` contains only ResNet18 and EfficientNet-B0 rows. | CSV has 11 resnet18 + 12 efficientnet_b0 rows, no smallcnn | 0.6 |
| D22 | S3 | **Chart y-axis is truncated at 90**, visually tripling the SmallCNN↔ResNet18 difference. | `plantvillage_validation_comparison.png` | 6.2 |
| D23 | S3 | **No test suite, no pinned dependencies.** `pyproject.toml` uses `torch>=2.1` while the runs used a specific CUDA build. | `pyproject.toml:16-17` | 5.1, 5.2 |
| D24 | S3 | **Storage is redundant.** PlantVillage is stored in `data/raw` and again under `kaggle_upload/`; PlantDoc raw (1.99 GB) is unused by any current script; a local `.venv-smoke` adds ~2.5 GB. | Directory sizes | 5.5 |
| D25 | S3 | **Required project documents do not exist.** No `PROJECT_STATUS.md`, `DATASET_CARD.md`, `MODEL_CARD.md`, `EXPERIMENT_LOG.md`. | Absent at repo root | 5.3 |
| D26 | S3 | **No results registry**, so tables are hand-typed and can silently diverge from artifacts. | No `results/` directory | 1.4 |
| D28 | S2 | **Leaf-level leakage protection covers 75% of the test split, not 100%.** 2,676 of 10,709 test images (25.0%) carry `fallback_<id>` group IDs because the upstream leaf map did not match them, so each got a per-image ID and the "zero group overlap" audit does not constrain them. In principle one could be a second photograph of a leaf that also appears in training. Exact SHA-256 and path overlap are both verified zero, and the audit's perceptual-hash probe found one credible cross-split near-duplicate group in 54,015 images, so the practical risk is small. The correct claim is "no exact duplication and no identifiable leaf overlap", never "no leakage". |
verified 2026-10-25 from the manifests: `sha256` overlap 0, `relative_path` overlap 0, `group_id` overlap 0, test fallback 2,676 / 10,709 | 1.5, 6.3 |
| D27 | S3 | **EfficientNet-B0 checkpoint carries no `dataset_fingerprint`**, because the pilot script did not write one. Provenance is limited to what the file records plus the class-order check. | `checkpoints/efficientnet_b0/provenance.json` `metadata_absent` | 0.7 |

---

## Resolved during the 2026-10-25 session

| ID | Was | Now |
|---|---|---|
| R01 | EfficientNet-B0 checkpoint missing entirely; 99.51% unverifiable | Checkpoint restored, contract verified, provenance recorded |
| R02 | No local/scripted locked-test evaluation; test scoring existed only inside a notebook | `scripts/evaluate_locked_test.py` scores any split, emits per-class table, confusion matrix, bootstrap CIs, calibration, selective coverage |
| R03 | No Grad-CAM implementation | `src/plant_disease/gradcam.py` + `scripts/gradcam_report.py` |
| R04 | No calibration code despite `configs/data_contract.yaml` requiring it | `src/plant_disease/calibration.py` |
| R05 | Training notebook embedded a hand-maintained copy of the training script | `scripts/build_training_notebook.py` regenerates it and asserts byte-identity |
| R06 | ResNet18 had no training entry point | `scripts/train_resnet18.py` over the shared engine |
| R07 | Pilot training was I/O-bound (~8.7 min/epoch, growing per epoch) | v2 engine: worker processes, prefetch, larger batch, cuDNN benchmark |
| R08 | No end-to-end test of training + evaluation | `scripts/smoke_test_engine.py`, passing |
| R09 | The P0.1 notebook imported `plant_disease.models` in its setup cell, but the cells that create the package ran afterwards, so the first cell always raised `ImportError` | Cells reordered: paths → writefile → checkpoint check → evaluate; the import now fails with an instruction instead of a bare traceback |
| R10 | The evaluator hard-coded `data/raw/PlantVillage/train` as the image root, but published test rows are materialised under `.../test`, so every test evaluation would have failed with "images missing" | `plant_disease.data.resolve_image_root()` picks the root from the manifest; verified to return `train` for train/validation and `test` for test |
| R11 | P0.1 could not be closed: no locked-test artifact existed for EfficientNet-B0 | Scored both splits on Kaggle. Validation macro-F1 reproduced **bit-identically** (0.9936894341462544); test = 53 errors in 10,709. All five README claims verified. Registered in `results/registry.json` |
| R12 | The audit suspected background shortcuts from 20 of 32 `Soybean___healthy` errors landing on another crop's healthy class | For EfficientNet-B0 the cross-crop healthy total is **8 errors in 10,709** (Raspberry→Cherry 4, Apple→Blueberry 1, Apple→Soybean 1, Potato→Soybean 1, Soybean→Corn 1). The suspicion was drawn from SmallCNN and does **not** transfer to the stronger model. H2 now needs the occlusion probe, not this static count |

---

## Reporting caveats learned from the first real run

These are not defects in the code; they are ways a report can be wrong while every number is correct.

- **MCE is not usable at this accuracy.** 10,709 test images fill 15 confidence bins so unevenly that the middle bins hold 1–36 images. The maximum calibration error (0.53) comes from one bin containing a single image. ECE (0.0022) is meaningful; MCE is not. Always publish ECE with the bin counts.
- **Pipeline latency is not model latency.** The evaluator measured 37.6 ms/image end to end on a T4, which implies 26 images/second — far too slow for EfficientNet-B0 at batch 128. That figure describes the DataLoader and JPEG decode path in this notebook, not the network. Do not quote it as a model benchmark; a real latency measurement needs a warm cache and a batched loop.
- **Abstention is nearly free on PlantVillage and says nothing about the field.** Accuracy at 90% coverage (99.76%) barely exceeds full coverage (99.51%) because the model is already well calibrated on controlled imagery. The interesting abstention curve is the one measured on field data, which does not exist yet.

## Rules that stop these from returning

1. A number enters a document only after it exists in `results/registry.json` with its artifact path (D26, D03, D07).
2. Generated files are written by the script that owns them, never hand-edited (D02).
3. Any checkpoint or table in a document must have a tracked checksum and a stated provenance (D01, D20, D27).
4. Partial-split metrics always report zero-support classes alongside the headline number (D06).
5. Cross-dataset comparisons state the mapped-class subset and the interval width (D08, D10).

## R12 — Protocol v1 cannot reproduce the reference result (BLOCKER)

**Status:** open, blocks S9/S10. Evidence: `reports/model_comparison/SWEEP_V1_INVALID.md`.

Twelve runs (4 architectures x 3 seeds) completed on 2026-10-25. All twelve
carried the same dataset fingerprint, so the split was correct, but every
architecture landed far below its established result: EfficientNet-B0 0.684
against 0.9954, ResNet18 0.584 against 0.9615, SmallCNN 0.231 against 0.9601.

Measured cause: the backbone never fine-tuned. Whole-model distance to pristine
ImageNet weights is 0.49% for the sweep's EfficientNet-B0 against 1.96% for the
verified reference. Four settings do it: weight decay 0.05 instead of 1e-4,
global batch 128 instead of 32 (4x fewer optimizer steps), a warmup+cosine
schedule instead of constant LR, and label smoothing plus class weights.

**Fix before re-running:** baseline arm uses the proven optimisation settings;
augmentation and class balancing move to a separate arm (which H3 needs anyway);
gate any sweep behind a two-epoch check requiring B0 >= 0.95 val accuracy by
epoch 2.

## R13 — Sweep ran 4.5x slower per epoch than the pilot (UNEXPLAINED)

Pilot measured 2.45 min/epoch for EfficientNet-B0; the sweep measured 7.52.
SmallCNN 0.67 -> 3.01, ConvNeXt 3.15 -> 14.16. Total 5.6 h against a 4.2 h
projection. Not visible in the artifacts; needs a session log to decide between
data-loader throughput and `/kaggle/temp` read speed.

## R14 — Blocked shuffle made batches near single-class (ROOT CAUSE of R12)

**Status:** fixed in protocol v2 (`shuffle_mode: random`), pending gate confirmation.

`block_shuffled_indices` permutes 2048-image blocks and shuffles inside each
block. It was added to stop the pilot's page-fault pathology, but the train
manifest is written in class order, so a block of 2048 consecutive rows spans
only a handful of classes. Measured with `scripts/check_batch_purity.py` on the
36,762-row development split:

| shuffle | batch | single-class batches | classes per batch (median) |
|---|---|---|---|
| blocked (2048) | 16 | 12.1% | **3** |
| blocked (2048) | 64 | 10.8% | 3 |
| random | 16 | 0% | **12** |
| random | 64 | 0% | 26 |

Three classes per batch is a BatchNorm problem: normalisation statistics come
from three classes instead of the whole distribution. It explains the exact v1
sweep pattern, which nothing else did:

| architecture | norm | v1 result |
|---|---|---|
| ConvNeXt-Tiny | LayerNorm (batch-independent) | **0.95** |
| EfficientNet-B0 | BatchNorm | 0.68 |
| ResNet18 | BatchNorm | 0.58 |
| SmallCNN | BatchNorm | 0.23 |

It also explains why the 380-image subset check on Kaggle trained normally
(0.54 by epoch 3): 380 images fit inside a single 2048 block, so the subset was
shuffled effectively at random. The pipeline was never broken — the full-dataset
ordering was.

The original run that reached 0.9826 in epoch 1 used an ordinary shuffled
DataLoader, which is what v2 now does.

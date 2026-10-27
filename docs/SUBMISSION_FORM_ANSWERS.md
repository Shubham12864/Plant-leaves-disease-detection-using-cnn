# B.Tech Project – Mid-Term & Phase III Evaluation 2026–27: paste-ready answers

Project ID **BT40**. Title: *Plant Leaf Disease Detection Using CNN*. Guide: Dr. Harendra Singh.
Form deadline: 29/10/2026. One submission covers both stages (max 20 marks each).

Each heading below is the exact field name on the Google Form. Paste the text under it.
Every number is backed by `results/registry.json` (23 registered claims) and was produced by
this project's own runs. Nothing here is a published or expected result.

**Fill before submitting (not recorded in the repository, do not guess):**
enrolment numbers for all four members (two members are both named Shubham Kumar); the
guide's official email; confirmation of the approved title; the week dates in `docs/WPR_1.md`
to `WPR_3.md`. The coordinator's mail requires at least three WPRs to have been sent to the
guide before the form is filled.

**Before pasting links:** the repository must be pushed and public (or shared with the
evaluators), otherwise the links in the two "Supporting Evidence" fields will 404.

---

## Section 2: Evaluation Stage

Tick both: **Mid-Term Evaluation** and **Phase III – Implementation Evaluation**.

---

# PART A: MID-TERM

## Current Implementation / Prototype Status

Implemented and working at mid-term:
- A frozen data split for PlantVillage (54,305 images, 38 classes): 36,762 train, 6,544 validation, 10,709 test, checked for disjointness by a gate.
- One distributed training engine (`scripts/train_ddp.py`) driven by a frozen protocol (`configs/protocol_v2.json`), used for all four architectures: SmallCNN, ResNet18, EfficientNet-B0, ConvNeXt-Tiny.
- A fourteen-run comparison sweep (3 to 5 seeds per architecture), zero failed runs.
- A field-image evaluation on PlantDoc (236 test images) with bootstrap intervals.
- 17 automated tests and a results registry that blocks any number without an artifact.
- A static web front end (`app/static/`, four pages) that shows recorded model output.

Not done at mid-term: the leaf-versus-background mechanism experiment and the background-augmentation experiment (both completed afterwards, see Phase III).

Working now: the training and evaluation pipeline end to end, and the front end in demo mode. Not working: there is no live inference service, so the front end does not classify uploaded images.

## Design / Architecture & Validation

Pipeline: frozen manifests → cached uint8 image store (keyed by a hash of the path list, so a changed split invalidates it) → one DistributedDataParallel engine on two GPUs → per-run checkpoint plus a metrics JSON carrying the split fingerprint and the learning rate applied each epoch → separate evaluation scripts for PlantVillage and PlantDoc → registry → generated tables.

Validation: all fourteen runs report the same dataset fingerprint, so they provably saw the same split. A parity check confirms each model built through the package and through the engine has the same parameter count (398,662 / 11,196,006 / 4,056,226 / 27,849,350) and state-dict keys. The reference EfficientNet-B0 validation score reproduced bit-identically (0.9936894, difference 0.0).

Important design change: the first sweep could not learn (twelve runs between 0.07 and 0.95 against a 0.99 reference). The cause was blocked shuffling on a class-sorted manifest: the median batch held 3 classes, which broke BatchNorm (`reports/model_comparison/SWEEP_V1_INVALID.md`, `SWEEP_V2_RESULTS.md`). The shuffle mode was changed and a two-epoch gate (0.95 validation accuracy by epoch 2) now precedes any long run.

## Testing / Experimental Results So Far

PlantVillage validation macro-F1 under the frozen protocol: ConvNeXt-Tiny 0.98951 (3 seeds), EfficientNet-B0 0.98946 (5 seeds), ResNet18 0.98460, SmallCNN 0.91599. The top two differ by 0.00005, 34 times smaller than the reference model's own seed spread.

Locked test (10,709 images), reference model: macro-F1 0.99314; 53 errors.

Field test (PlantDoc, 236 images, 27 classes with test images): EfficientNet-B0 accuracy falls from 0.9928 to 0.2085 (five-seed mean; the published checkpoint scores 0.1737). Expected calibration error rises from 0.0026 to 0.5808. In the 0.95–1.00 confidence band (65 images) the published checkpoint is right 29.2% of the time. A confidence threshold gains only 0.8 points at 90% coverage.

Across all fourteen checkpoints the field accuracies are ConvNeXt-Tiny 0.3008, EfficientNet-B0 0.2085, ResNet18 0.1610, SmallCNN 0.0438. Caveat: 236 images give about ±6 points of uncertainty on a single estimate.

## Supporting Evidence / Repository / Demonstration Links – Mid-Term

https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/blob/main/PROJECT_STATUS.md
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/blob/main/reports/model_comparison/SWEEP_V2_RESULTS.md
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/tree/main/reports/field_gap
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/blob/main/results/registry.json
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/tree/main/app/static

## Major Challenges / Pending Work

- The first sweep collapsed because of batch composition (see above), which made the twelve runs of that sweep unusable. Documented in `reports/model_comparison/SWEEP_V1_INVALID.md`.
- The first occlusion probe filled the image border grey, which also erased leaf, and would have been reported as strong background dependence. A rendered check caught it and the probe was rebuilt around a leaf mask.
- Pending at mid-term: the mechanism measurement that separates "depends on the background" from "disrupted by the edit", the background-augmentation experiment, and a live inference service.
- The approved abstract promises a confidence threshold for uncertain cases. The measured result (0.8 points gained) does not support it, so the abstract wording needs to be agreed with the guide.

---

# PART B: PHASE III

## Implementation Completed

Completed since mid-term:
- **Mechanism experiment** (`scripts/swap_localisation_test.py`): replaces an image's leaf or its background with a donor of another class and measures whether the prediction follows the donor. On 298 validation images, leaf swap moved the prediction 50.3% of the time (95% CI 44.6–55.7), background swap 21.8% (17.1–26.8), against 2.6% chance and a 4.4% same-class control.
- **Background-replacement augmentation** (`scripts/make_leaf_masks.py`, `scripts/build_composite_cache.py`).
- **Paired augmentation sweep**: six runs, three seeds per condition (`reports/mechanism/S19_BACKGROUND_AUGMENTATION_RESULTS.md`).
- **Calibration and abstention** analysis for all four models; Grad-CAM overlays for 18 field cases.
- **Engineering hardening**: shared safe checkpoint loader, 39 automated tests (up from 17), CI workflow (ruff and pytest), a repository-cleanliness check, MIT licence.
- **Dissertation** generated by script from the registered artifacts (`scripts/build_thesis.py`).

Still in progress / not built: a live inference service (no API, no server-side model loading, no deployment), and the hand-labelled saliency sheet (50 maps), which is now a visual cross-check rather than a dependency.

## Code / Hardware / System Integration Status

Software only; no hardware or sensors. Code is in `src/plant_disease/` (package: data, models, metrics, calibration, Grad-CAM, checkpoint loader) and `scripts/` (training, evaluation, experiments, builders). Versions are pinned in `requirements.lock`; settings in `configs/`.

Integration is verified at interfaces: model parity between package and engine (all four match); `scripts/run_all.py --check` runs six reproducibility gates; `scripts/build_tables.py --verify` compares four generated tables with the registry and fails on a missing run, a missing artifact or a stale table (currently passes, 23 claims backed).

The static front end (`app/static/`) is not connected to a model: with no backend configured, `api.js` fails fast with code `demo` and the pages show recorded checkpoint output, never an invented prediction. The serving API is specified in `docs/FRONTEND_PLAN.md` but not implemented. Documentation: README, `DATASET_CARD.md`, `MODEL_CARD.md`, `docs/KNOWN_ISSUES.md` (defect register).

## Testing, Validation & Performance Results

**Automated:** 39 tests pass (split integrity and disjointness, checkpoint contract with strict loading, metric sanity, calibration, data loading, Grad-CAM shape, checkpoint loader fallback) plus the six reproducibility gates. The parity check caught one real error: ResNet18 had been recorded with the 1000-class ImageNet parameter count although the 38-class checkpoint has 11,196,006 (`EXPERIMENT_LOG.md`).

**Performance** (PlantVillage locked test, 10,709 images; PlantDoc field test, 236 images; evaluation on CPU, training on Kaggle 2× Tesla T4):

| Measure | PlantVillage | PlantDoc |
|---|---|---|
| EfficientNet-B0 accuracy | 0.9928 | 0.2085 (5-seed mean) |
| Expected calibration error | 0.0026 | 0.5808 |
| Best field model | n/a | ConvNeXt-Tiny 0.3008 |

**Augmentation experiment** (3 seeds per condition): field macro-F1 0.2232 baseline vs 0.2356 with background replacement. Paired difference +1.24 points, 95% CI [−3.32, +5.80], two of three seeds improved, 1.6% of the gap recovered. In-domain macro-F1 fell 0.87 points. Conclusion: no detectable effect. Abstention gains 0.8 points at 90% coverage.

**Defects found and fixed this phase:** a notebook ran a check before the cell that wrote its script; a field evaluation failed remotely because the project root resolved one level too high; a reliability figure gave a one-image bin the same weight as the bin holding 10,590 of 10,709 images. Latency was not benchmarked and is not reported.

## Current Milestone Completion Status

Select **Most planned milestones completed**.

## Pending Milestones / Remaining Work

- A second independent field dataset with expert labels (not started); this is what would turn a single-dataset study into a publishable claim, and a held-out validation split for field data (PlantDoc has none).
- Out-of-distribution detection to replace the confidence threshold, which was measured and found ineffective (not started).
- Leaf-level (rather than background-level) field augmentation, which the mechanism result points to.
- A live inference service so the front end can classify uploaded images.
- Agree the abstract wording with the guide: the confidence-threshold and targeted-augmentation expectations were both implemented and measured as negative results (`docs/ABSTRACT_VS_EVIDENCE.md`).
- Complete the 50 hand-labelled saliency maps; fix open defects R13 and R15 in `docs/KNOWN_ISSUES.md`.

## Supporting Evidence / Repository / Demonstration Links – Phase III

https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/tree/main/reports/mechanism
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/tree/main/reports/field_gap
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/tree/main/tests
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/blob/main/results/registry.json
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/blob/main/docs/KNOWN_ISSUES.md
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/blob/main/docs/ABSTRACT_VS_EVIDENCE.md
https://github.com/Shubham12864/Plant-leaves-disease-detection-using-cnn/blob/main/MODEL_CARD.md

---
title: Project Status — Plant Leaf Disease Detection Using CNN (BT40)
type: project-status
status: Phase II baseline trained; later project work remains
tags:
  - capstone
  - plant-disease
  - computer-vision
  - research
---

# Project status — start here

Read this file first. It is the handoff for any new session, and it says what is
verified, what is in progress, and where the plan lives.

Last updated: 2026-10-25.

---
## Documents that own each question

| Question | File |
|---|---|
| What are we doing and in what order? | `docs/ROADMAP.md` — 28 steps in 7 stages, each with a gate |
| What is the scientific claim? | `docs/RESEARCH_CLAIM.md` — H1–H4, falsifiers, pre-registration rules |
| What is broken? | `docs/KNOWN_ISSUES.md` — 27 defects with evidence, plus resolved items |
| Where do model weights live? | `docs/CHECKPOINTS.md` + `checkpoints/SHA256SUMS` |
| How do I run something on Kaggle? | `notebooks/KAGGLE_NOTEBOOKS_GUIDE.md` |
| What did the old audit find? | `WORKSPACE-ANALYSIS.md` (historical, superseded by KNOWN_ISSUES) |

---
## Current position

Stage A closed, Stage B closed, **Stage C closed**. The field gap is measured.
Next is Stage D: explain it (Grad-CAM + occlusion probe), then Stage E.

| Step | State |
|---|---|
| S1 locked-test evaluation | **done** — bit-identical, 53 errors in 10,709 |
| S2 PlantDoc label map | **done** — 28/28 decided, 3 caveated |
| S3 stale documents | **done** |
| S4 baseline commit | **done** — tag `baseline-p0.1` |
| S5–S8 ConvNeXt, SmallCNN, protocol | **done** — v1 later found invalid (R12/R14) |
| S9 pilot | **done** |
| S9b gate | **passed** — 0.9836 at epoch 1 |
| S9c sweep v2 | **done** — 14 runs, 0 failures |
| S10 comparison table | **done** — `reports/model_comparison/SWEEP_V2_RESULTS.md` |
| S11 score PlantDoc | **done** — 236 images, 27 classes |
| S12 quantify the gap | **done** — −81.9 pp, 27/27 classes |
| S13 answer H1 | **done — H1 confirmed** |
| **S14 Grad-CAM on the failures** | **NEXT** |

### Stage C result — the field gap (H1 confirmed)

| | PlantVillage (lab) | PlantDoc (field) |
|---|---:|---:|
| accuracy | 0.9928 | **0.1737** (95% CI 0.127 – 0.225) |
| mean per-class recall | 0.9939 | **0.1746** |
| mean confidence | 0.998 | 0.7522 |
| ECE | 0.0026 | **0.5784** |

**All 27 mapped classes got worse**, median drop 0.851. In the 0.95–1.00
confidence band the model is right **29.2% of the time at 98.6% confidence**.
25.4% of field predictions land on classes PlantDoc does not contain.

Pipeline verified by control: the same script scores 0.9967 on 300 PlantVillage
validation images, so this is domain shift, not a broken inference path.

Full analysis: `reports/field_gap/FIELD_GAP_RESULTS.md` · figure `field_gap.png`.

### Sweep v2 (PlantVillage validation, 6,544 images)

| Architecture | n | macro-F1 mean | sd |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 3 | 0.98951 | 0.00028 |
| EfficientNet-B0 | 5 | 0.98946 | 0.00171 |
| ResNet18 | 3 | 0.98460 | 0.00166 |
| SmallCNN | 3 | 0.91599 | 0.00865 |

The top two are statistically indistinguishable (gap 0.00005, thirty-four times
smaller than the B0 seed spread): **architecture is not what decides the outcome
on this benchmark**, which is exactly why Stage C mattered more than another
model.

### Measured pilot (2x T4, one epoch each)

| Architecture | batch/GPU | min/epoch | train img/s |
|---|---|---|---|
| SmallCNN | 128 | 0.67 | 914 |
| ResNet18 | 64 | 0.78 | 786 |
| EfficientNet-B0 | 64 | 2.45 | 250 |
| ConvNeXt-Tiny | 32 | 3.15 | 194 |

Artifacts: `reports/model_comparison/pilot/`. Epoch-1 scores there are not
results — the learning rate is still warming up.

### S9 — what to run

`notebooks/01_Train_EfficientNet_B0_v2.ipynb`, `RUN_MODE = "sweep"`, and set
`ONLY_ARCH` to one architecture per session:

```
ONLY_ARCH = "smallcnn"          # ~0.4 h
ONLY_ARCH = "resnet18"          # ~0.5 h
ONLY_ARCH = "efficientnet_b0"   # ~1.5 h
ONLY_ARCH = "convnext_tiny"     # ~1.9 h
```

One architecture per session because a restart then costs one architecture
rather than the whole sweep. Each run resumes from its own `_last.pt`.

*Gate:* 12 checkpoints with `dataset_fingerprint` and seed, and a per-architecture
spread across seeds instead of a single number.

---
## Verified facts (checked, not assumed)

| Fact | Evidence |
|---|---|
| EfficientNet-B0 checkpoint restored | `checkpoints/efficientnet_b0/efficientnet_b0_best.pt`, SHA-256 `90e003cf…` |
| It loads `strict=True` into a 38-class B0 | run on 2026-10-25, 4,056,226 params |
| Class order matches the development manifest exactly | 38/38 identical order |
| Its recorded `val_macro_f1` = 0.99369 | matches `plantvillage_model_training_history.csv` epoch 12 |
| Development split is 43,306 rows: 36,762 train / 6,544 validation | regenerated summaries now agree |
| Published test split is 10,709 rows | `plantvillage_test.summary.json` |
| PlantDoc development is 2,331 rows, all `train` — **no validation split** | `plantdoc_development.summary.json` |
| Test images live under `data/raw/PlantVillage/test`, not `train` | checked against both manifests |
| SmallCNN and ResNet18 checkpoints exist | `checkpoints/smallcnn`, `checkpoints/resnet18` |
| EfficientNet-B0 validation score reproduces bit-identically | measured 0.9936894341462544 = recorded value, delta 0.0 |
| EfficientNet-B0 test: 53 errors in 10,709 | `.../test/metrics.json`, registered in `results/registry.json` |
| Cross-crop healthy confusions for B0 total 8 in 10,709 | `.../test/confusion_matrix.csv` — the audit's shortcut suspicion was drawn from SmallCNN and does not transfer |
| Four architectures train through one engine | parameter counts agree between `src/plant_disease/models` and `scripts/train_ddp.py`: 398,662 / 11,196,006 / 4,056,226 / 27,849,350 |
| `smallcnn` inline definition cannot drift | parity between the engine's inline class and the package class is asserted in `scripts/smoke_test_engine.py` |

## Known-weak areas (do not overclaim)

- Every number comes from **one seed**. No variance estimate exists yet. This is
  the single biggest weakness; `docs/ROADMAP.md` S9 fixes it.
- The three original models were trained with different LR, batch, augmentation
  and parallelism. Only the split was shared. Say "same split", never
  "controlled comparison", until S9.
- PlantDoc accuracy is **not** comparable to PlantVillage accuracy: 28 classes vs
  38, and 236 test images ⇒ roughly ±6.4 pp.
- `balanced_accuracy` and `macro_recall` are the same quantity, not two metrics.
- Macro-F1 on a class subset is meaningless without listing zero-support classes.

## Compute policy

All training and full-split inference runs on **Kaggle GPUs**. Local work is
limited to data preparation, manifests, documents and CPU-only code checks.

There is a `.venv-smoke/` in the repo used only for `py_compile` and
`scripts/smoke_test_engine.py`. It is not for producing results and should be
deleted before submission (roadmap S25).

## Tooling notes

- `computer_use` (cua-driver) is installed and enabled.
- `scripts/smoke_test_engine.py` is the regression test for training +
  evaluation. It passes. Run it after touching `train_ddp.py` or
  `evaluate_locked_test.py`.
- `notebooks/04_Locked_Test_Evaluation.ipynb` and
  `notebooks/01_Train_EfficientNet_B0_v2.ipynb` are **generated**. Edit
  `scripts/build_eval_notebook.py` / `scripts/build_training_notebook.py` and
  regenerate; never hand-edit the notebook's embedded cells.

## Immediate queue

| Step | Action | Needs |
|---|---|---|
| S1–S6, S8 | **done** | — |
| **S7** | `RUN_MODE="pilot"` in notebook 01_v2, one epoch x four architectures | Kaggle GPU, ~15 min |
| S9 | `RUN_MODE="sweep"`, 12 runs | Kaggle GPU, hours |
| S10 | comparison table from artifacts | local |

Stage B order was adjusted: S8 (freeze the protocol) runs before S7 (pilot) so
the pilot measures the settings the sweep will actually use, instead of measuring
a different configuration.

## Hard rules

1. No number in any document unless `results/registry.json` has it with an
   artifact path.
2. Test split is scored for reporting only; re-scoring is disclosed.
3. A gate that fails stops the queue. Do not proceed on a failed gate.
4. Negative results are reported, not buried. H1–H4 can all come back null and
   that is still the answer.
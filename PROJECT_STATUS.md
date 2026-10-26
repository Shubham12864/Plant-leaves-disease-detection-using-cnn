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

Stages A, B, C, D, E and F are closed. **Stage G (writing) is what remains**, plus
one blocked experiment (S19) and one outstanding manual step (S16 labels).

| Stage | State |
|---|---|
| A Evidence (S1–S4) | **closed** |
| B Fair comparison (S5–S10) | **closed** — 14 runs, 0 failures |
| C Field gap (S11–S13) | **closed — H1 confirmed (−81.9 pp)** |
| D Mechanism (S14–S19) | **closed except S19** — H2 partial, S18 blocked |
| E Usefulness (S20–S22) | **closed — H4 confirmed** |
| F Lock down (S23–S25) | **closed** — 17 tests, registry check, run_all |
| **G Write (S26–S28)** | **NEXT** |

### Hypothesis verdicts (registered in results/registry.json)

| H | Verdict | Core evidence |
|---|---|---|
| **H1** field accuracy is far below nominal | **confirmed** | 0.9928 → 0.2085 (5 seeds); 27/27 classes worse; lab gaps understate field gaps 19 to 28× |
| **H2** gap is partly background dependence | **partially supported, now precise** | leaf swap moves the prediction 50.3%, background swap 21.8% vs 2.6% chance: joint use of leaf and scene, leaf dominant |
| **H3** augmentation recovers part | **not supported** | 6 runs, 3 seeds/condition: field macro-F1 +1.24 pp, CI [-3.32, +5.80], 2/3 seeds positive, sign test p=0.50; in-domain cost -0.87 pp; 1.6% of the gap recovered |
| **H4** nominal accuracy doesn't bound deployable | **confirmed** | ECE 0.0026 → 0.5808; abstention gains 0.8 pp at 90% coverage |

### The four numbers that carry the project

| | PlantVillage | PlantDoc |
|---|---:|---:|
| reference accuracy | 0.9928 | 0.2085 (5-seed mean; published checkpoint 0.1737) |
| reference ECE | 0.0026 | 0.5808 |
| best field model | — | ConvNeXt-Tiny 0.3008 |
| best field ECE | — | ConvNeXt-Tiny 0.4440 |

Field numbers across **all fourteen checkpoints** (closes the one-checkpoint
caveat): ConvNeXt-Tiny 0.3008 (3 seeds, sd 0.038), EfficientNet-B0 0.2085 (5, sd
0.030), ResNet18 0.1610 (3, sd 0.027), SmallCNN 0.0438 (3, sd 0.011).

Lab order is preserved in the field; lab gaps are not. Models tied in the lab
(0.00005) are 9.2 pp apart in the field, and SmallCNN's 7.4 pp lab deficit becomes
25.7 pp. An earlier claim that the ranking *reverses* was wrong: it compared
checkpoints from different protocols. Corrected in `USEFULNESS_RESULTS.md`.

### Open items

- **S16** 50 hand-labelled saliency maps: instrument ready
  (`reports/mechanism/gradcam_b0_plantdoc/labelling_sheet.csv`), labels not done.
- **S18/S19** background corpus + augmentation sweep: blocked, needs a licensing
  decision.
- **Abstract wording**: the promised confidence threshold does not survive on
  field data (S22). Needs a decision, not a silent edit.

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
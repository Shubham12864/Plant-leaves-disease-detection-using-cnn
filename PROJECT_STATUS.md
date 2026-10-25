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

Do not start work without reading `docs/ROADMAP.md`. The sequence matters: each
step produces the artifact the next one depends on.

---

## Current position

Stage A is closed. Stage B is **blocked**: the sweep ran and came back invalid,
so S9 must be re-run on a corrected protocol before S10 can build any table.

| Step | State |
|---|---|
| S1 locked-test evaluation | **done** — bit-identical reproduction, 53 errors in 10,709 |
| S2 PlantDoc label map | **done** — 28/28 decided, 3 caveated |
| S3 stale documents | **done** |
| S4 baseline commit | **done** — tag `baseline-p0.1` |
| S5 ConvNeXt-Tiny | **done** |
| S6 SmallCNN on the engine | **done** |
| S8 protocol frozen | **done** — `configs/protocol.json` v1 |
| S7 protocol pilot | **done** — no OOM, 4.2 h projected |
| S9 12-run sweep | **ran, INVALID** — see R12 |
| **S9b protocol fix + 2-epoch gate** | **NEXT** |

### Sweep v1 result: invalid

Twelve runs, seeds 42/43/44, identical dataset fingerprint across all of them.
Every architecture landed far below its established result:

| Architecture | Established | Sweep best | Gap |
|---|---|---|---|
| EfficientNet-B0 | 0.9954 val | 0.7246 | −27.1 pp |
| ResNet18 | 0.9615 val | 0.5839 | −37.8 pp |
| SmallCNN | 0.9601 val | 0.2311 | −72.9 pp |
| ConvNeXt-Tiny | none | 0.9509 | — |

Measured cause: the backbone never fine-tuned. Distance to pristine ImageNet
weights is 0.49% for the sweep's B0 against 1.96% for the verified reference.
Protocol v1 differs from the script that produced the reference in weight decay
(0.05 vs 1e-4), global batch (128 vs 32), LR schedule (warmup+cosine vs
constant) and label smoothing / class weights (present vs absent).

**No number from sweep v1 may be quoted except as evidence of the defect.**
Full evidence: `reports/model_comparison/SWEEP_V1_INVALID.md`. Defects R12/R13 in
`docs/KNOWN_ISSUES.md`.

### Measured pilot (2x T4, one epoch each)

| Architecture | batch/GPU | min/epoch | train img/s |
|---|---:|---:|---:|
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
- Macro-F1 computed on a class subset collapses (measured: 99.6% accuracy
  alongside 10.5% macro-F1 on 256 rows). Always report zero-support classes.

---

## Compute policy

All training and full-split inference runs on **Kaggle GPUs**. Local work is
limited to data preparation, manifests, documents and CPU-only code checks.

There is a `.venv-smoke/` in the repo used only for `py_compile` and
`scripts/smoke_test_engine.py`. It is not for producing results and should be
deleted before submission (roadmap S25).

---

## Tooling notes

- `computer_use` (cua-driver) is installed and enabled. It is only loaded in a
  session that started *after* it was enabled.
- `scripts/smoke_test_engine.py` is the regression test for training +
  evaluation. It passes. Run it after touching `train_ddp.py` or
  `evaluate_locked_test.py`.
- `notebooks/04_Locked_Test_Evaluation.ipynb` and
  `notebooks/01_Train_EfficientNet_B0_v2.ipynb` are **generated**. Edit
  `scripts/build_eval_notebook.py` / `scripts/build_training_notebook.py` and
  regenerate; never hand-edit the notebook's embedded cells.

---

## Immediate queue

| Step | Action | Needs |
|---|---|---|
| S1–S6, S8 | **done** | — |
| **S7** | `RUN_MODE="pilot"` in notebook 01_v2, one epoch x four architectures | Kaggle GPU, ~15 min |
| S9 | `RUN_MODE="sweep"`, 12 runs | Kaggle GPU, hours |
| S10 | comparison table from artifacts | local |

Stage B order was adjusted: S8 (freeze the protocol) runs before S7 (pilot) so
the pilot measures the settings the sweep will actually use.

---

## Hard rules

1. No number in any document unless `results/registry.json` has it with an
   artifact path.
2. Test split is scored for reporting only; re-scoring is disclosed.
3. A gate that fails stops the queue. Do not proceed on a failed gate.
4. Negative results are reported, not buried. H1–H4 can all come back null and
   that is still the answer.

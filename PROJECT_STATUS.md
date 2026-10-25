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

**S1 — run the locked-test notebook.** Everything else waits on it.

```
notebooks/04_Locked_Test_Evaluation.ipynb   (16 cells)
```

Attach two Kaggle inputs: the PlantVillage dataset, and the EfficientNet-B0
checkpoint. Cell 2 auto-discovers both mount paths.

Expected output after cell 4: `package import OK`
Expected output after cell 6: `CHECKPOINT OK`, 4,056,226 parameters, 38 classes

**Gate:** validation macro-F1 reproduces 0.99369 within 0.002, and
`test/` contains `metrics.json`, `classification_report.csv`,
`confusion_matrix.csv`. If validation differs by more than that, stop — the
checkpoint and the recorded score disagree and no test number is quotable.

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
| S1 | Run notebook 04, send `p01_verification.zip` | Kaggle |
| S2 | `python scripts/complete_label_mapping.py --check-output` | local, 2 min |
| S3 | Fix `ARCHITECTURE.md:9,184`, `MEMORY.md:166`, `Literature-Survey-Evidence-Register.md:7` | local |
| S4 | Commit and tag the baseline | local |

After S1 passes, Stage B begins with S5 (add ConvNeXt-Tiny as the fourth
architecture), then S6, S7, S8, S9 (12 runs), S10.

---

## Hard rules

1. No number in any document unless `results/registry.json` has it with an
   artifact path.
2. Test split is scored for reporting only; re-scoring is disclosed.
3. A gate that fails stops the queue. Do not proceed on a failed gate.
4. Negative results are reported, not buried. H1–H4 can all come back null and
   that is still the answer.

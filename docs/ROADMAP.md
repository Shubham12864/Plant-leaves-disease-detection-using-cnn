# Workplan — one step at a time

A linear sequence. Do the next step, pass its gate, then move on. No parallel branches, no deadline. If a gate fails, the step is not finished and the next one is not started.

The point of the order is that every later claim rests on an artifact produced earlier. Skip a step and the final report contains a number nobody can check — which is exactly the failure mode this whole plan exists to avoid.

The science behind these steps is in [`RESEARCH_CLAIM.md`](RESEARCH_CLAIM.md). Defects being closed are in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

---

## Fixed decisions

Set once, not revisited when a result looks inconvenient.

| Decision | Value | Reason |
|---|---|---|
| Reference model | EfficientNet-B0 | best existing validation macro-F1 |
| Models compared | SmallCNN, ResNet18, EfficientNet-B0, ConvNeXt-Tiny | one per design family: scratch, classic residual, efficient NAS-style, modern hybrid |
| Protocol | identical for all four; one documented deviation | SmallCNN needs a higher LR because it trains from scratch. Nothing else differs. |
| Seeds | 3 per model | one run cannot distinguish 99.3 from 99.6 |
| Primary metric | macro-F1 and per-class recall | 38.4× class imbalance makes accuracy flattering |
| Independence unit | leaf (`group_id`) | near-duplicate leaves are not independent samples |
| Test split | reporting only, once per model | already scored historically; disclosed, not repeated |
| Rows in every table | accuracy, macro-F1, field drop, ECE, accuracy at 90% coverage | one number cannot rank models |

---

## Stage A — Close the evidence gap

Nothing downstream is trustworthy while artifacts are missing.

**S1. Run the locked-test notebook**
Run `notebooks/04_Locked_Test_Evaluation.ipynb` on Kaggle with the PlantVillage dataset and the EfficientNet checkpoint.
*Produces:* `validation/` and `test/` metric sets, per-class report, confusion matrix, CIs, calibration, `P0.1_VERIFICATION.md`.
*Gate:* validation macro-F1 reproduces 0.99369 within 0.002 **and** the test report files exist.
*If it fails:* stop. The checkpoint and the recorded score disagree; every test claim is void until resolved.

**S2. Decide the PlantDoc label map**
Run `python scripts/complete_label_mapping.py --check-output`.
*Produces:* `data/label_mapping.csv`, 28 rows, each with relationship and caveat.
*Gate:* every PlantVillage target exists in the manifest; no class silently dropped; the three ambiguous rows carry their caveat in text.

**S3. Fix the three remaining stale documents**
`ARCHITECTURE.md` (lines 9 and 184 say no model was trained), `MEMORY.md` (line 166 says the same), `Literature-Survey-Evidence-Register.md` (line 7 says the source PDF is missing).
*Gate:* `grep -rE "not trained|no performance result|source PDF missing"` returns nothing current.

**S4. Commit a clean baseline**
Commit everything, tag it. The working tree becomes the reference point for every later change.
*Gate:* clean tree, `checkpoints/SHA256SUMS` tracked, weights still ignored.

---

## Stage B — Make the comparison fair

The current comparison varies learning rate, batch size, augmentation and parallelism between models. Any of those can move PlantVillage accuracy by more than a point, so "same-split comparison" is the most it can honestly claim. This stage fixes that.

**S5. Add ConvNeXt-Tiny as a fourth architecture** — **DONE**
Added `src/plant_disease/models/convnext.py`, a branch in `models/__init__.py`, entries in `train_ddp.py`'s `ARCHES` and `build_model`, and the evaluator allow-list now derives from `MODEL_NAMES` instead of a hardcoded tuple.
*Gate:* package and engine agree on 27,849,350 parameters and identical state-dict keys; forward pass returns `(2, 38)`; both notebooks rebuild byte-identical. Verified.

**S6. Put SmallCNN on the engine** — **DONE**
`SmallCNN` is now inlined in `train_ddp.py` (needed because the engine runs standalone on Kaggle) with a comment pointing at the package copy, and the mean/std switch already keys off the architecture name.
*Gate:* package and engine produce parameter-identical modules (398,662 params, same state-dict keys) so the duplicated definition cannot drift; 3 optimizer steps run on CPU. Verified — the parity check lives in `scripts/smoke_test_engine.py`.

**S7. Pilot run — one epoch, four models** — **DONE (2026-10-25)**
One epoch per architecture at seed 42, 2x T4. No OOM at any batch size in `protocol.json`, so the protocol stands unchanged.

| Architecture | batch/GPU | params | min/epoch | train img/s | epoch-1 macro-F1 |
|---|---:|---:|---:|---:|---:|
| SmallCNN | 128 | 398,662 | 0.67 | 914 | 0.008 |
| ResNet18 | 64 | 11,196,006 | 0.78 | 786 | 0.028 |
| EfficientNet-B0 | 64 | 4,056,226 | 2.45 | 250 | 0.084 |
| ConvNeXt-Tiny | 32 | 27,849,350 | 3.15 | 194 | 0.060 |

Epoch-1 scores are not results: with 10% warmup over 12 epochs the learning rate is still ramping. The pilot measures time and memory.

*Gate:* passed — four architectures completed, no OOM. Artifacts in `reports/model_comparison/pilot/`.
*Aberration worth noting:* EfficientNet-B0 runs at 250 img/s while the larger ResNet18 runs at 786. Depthwise-separable convolutions are bandwidth-bound on T4 and do not scale with parameter count, so this is expected rather than a bug — but it means "fewer parameters" does not predict wall-clock here.

**S8. Freeze the protocol** — **DONE (run before S7, deliberately)**
`configs/protocol.json`: image size, epochs, patience, seeds, eval batch, workers, augmentation, loss, EMA, scheduler, and per-architecture LR / batch / weight decay with a written justification for each deviation. The notebook parses it and builds every command from it.
*Gate:* changing a value in one place changes the run; there are no per-run edits. Moved ahead of S7 so the pilot measures the settings the sweep will actually use, instead of measuring a different configuration.

**S9. Twelve runs** — **NEXT, needs Kaggle**
3 seeds × 4 models through the frozen protocol, about 4.2 h as an upper bound (ignores early stopping). Run one architecture per session via `ONLY_ARCH` so a restart costs one architecture, not the sweep: `smallcnn` ~0.4 h, `resnet18` ~0.5 h, `efficientnet_b0` ~1.5 h, `convnext_tiny` ~1.9 h.
*Gate:* 12 checkpoints, each carrying `dataset_fingerprint` and a seed, plus a per-architecture spread across seeds rather than a single number.

**S10. Generate the comparison table**
Four models × the five rows fixed above, with group-aware CIs.
*Gate:* table produced by script from artifacts; no hand-typed digits.

---

## Stage C — Measure the field gap (H1)

**S11. Score PlantDoc**
Apply the label map, count predictions that land on classes PlantDoc does not contain.
*Gate:* mapped-subset accuracy, out-of-mapping rate, per-class support, interval width all present.

**S12. Quantify the gap**
Per-class and per-crop recall delta, PlantVillage → PlantDoc, with CIs on the delta.
*Gate:* one sentence of the form "recall falls from X to Y on N mapped classes, ±Z" is supportable.

**S13. Answer H1**
Record the verdict in `results/registry.json` and write the paragraph. Either answer is acceptable; a null result is a result.

---

## Stage D — Explain the gap (H2), then test the fix (H3)

**S14. Grad-CAM on the cases that matter**
Not random images: the taxonomy's cross-crop confusions, high-confidence errors, and correct-but-suspicious cases.
*Gate:* maps saved per case with true/predicted/confidence in an index CSV.

**S15. Occlusion probe**
Three perturbations — grey border replacement, centre-crop only, noise border — measuring prediction-flip rate per crop.
*Gate:* a flip rate per crop with the perturbation named.

**S16. Saliency localisation rate**
Hand-label 50 maps as leaf-peak or background-peak; state the labelling rule.
*Gate:* a rate, not an impression.

**S17. Answer H2**
Combine S15 and S16. If the shortcut is real, H3 becomes worth running; if not, say so and skip S18–S19.

**S18. Background source, checked for leakage**
Licence-clean backgrounds, **disjoint from PlantDoc**. Using PlantDoc imagery as augmentation material would contaminate the field evaluation in S11 and invalidate the whole claim.
*Gate:* source, licence and disjointness documented before any training starts.

**S19. Background augmentation, 3 seeds, paired comparison**
Retrain the reference model with background compositing; everything else frozen.
*Gate:* paired bootstrap on the PlantVillage→PlantDoc delta. Answer H3.

---

## Stage E — Bound usefulness (H4)

**S20. Calibration for all four models**
Reliability bins, ECE, MCE, Brier on validation.
*Gate:* a measured ECE per model; the word "calibrated" appears only beside it.

**S21. Abstention curve**
Threshold chosen on validation only; coverage, retained accuracy and retained error rate at 0.5/0.8/0.9/0.95.
*Gate:* accuracy at 90% and 80% coverage reported for every model.

**S22. Answer H4**
If retained accuracy at 90% coverage is essentially nominal, the model is well calibrated and abstention buys nothing — report that. If it rises sharply, the operating point is a result.

---

## Stage F — Lock it down

**S23. Tests**
`tests/` with pytest: split integrity, checkpoint contract, metric sanity (balanced accuracy equals macro recall by construction), evaluator output contract. Plus the existing engine smoke test.
*Gate:* green from a fresh environment.

**S24. Registry and generated tables**
`results/registry.json` holds every reported number with artifact path and command; `scripts/build_tables.py` renders the tables; a check fails if a document quotes a number absent from the registry.
*Gate:* deleting the registry breaks the table build, by design.

**S25. Packaging**
Pinned `requirements.lock`, `PROJECT_STATUS.md`, `DATASET_CARD.md`, `MODEL_CARD.md`, `EXPERIMENT_LOG.md`, and `scripts/run_all.py`.
*Gate:* a fresh clone plus documented commands reproduces the headline table.

---

## Stage G — Write it as four answers

**S26. Results organised by hypothesis**
H1, H2, H3, H4 — not by phase, not by model. Each section answers one question and ends with a verdict.

**S27. Figures**
Reliability diagram, confusion matrix, field-gap bars with CIs, saliency grid, coverage curve. Axes start at zero or state the truncation.

**S28. Limitations and defence**
Scope statement (one controlled source, one 236-image field set, no expert re-labelling), the disclosure that test was scored more than once, and a viva pack where every answer points at an artifact.

---

## Status

| Stage | Steps | State |
|---|---|---|
| A — evidence | S1–S4 | **complete** |
| B — fair comparison | S5–S10 | **complete** — 14 runs, 0 failures, spread reported |
| C — field gap | S11–S13 | **complete — H1 confirmed (−78.4 pp over 5 seeds)** |
| D — mechanism | S14–S19 | S14, S15 complete; S16 instrument ready, labels outstanding; S17 H2 **partial**; **S18 blocked**; S19 not started |
| E — usefulness | S20–S22 | **complete — H4 confirmed** |
| F — lock down | S23–S25 | **complete** — 17 tests, registry check, run_all |
| G — write | S26–S28 | **complete as drafts** — `reports/paper/` |

Remaining work is not computational:

1. Fill `reports/mechanism/gradcam_b0_plantdoc/labelling_sheet.csv` (S16).
2. Decide the H3 route: obtain a licence-clean background corpus (S18) or drop H3
   from the abstract with a stated reason.
3. Decide the confidence-threshold wording in the approved abstract.
4. Render the dissertation against the university template, which has not been
   supplied to this repository.

## Gates that must not be crossed

- No test number quoted before S1 passes.
- No PlantDoc claim before S2 and S11.
- No "the model uses background cues" before S15 and S16.
- No augmentation result before S18 documents a disjoint background source.
- No table typed by hand after S24.

## Compute ledger

| Stage | Runs | Approx. GPU time |
|---|---|---|
| A | 1 evaluation | 10 min |
| B | 4 pilots + 12 training | 3–5 h |
| D | 3 augmentation runs | 1–1.5 h |
| E | inference only | 20 min |

All on Kaggle. Local work is data preparation, documents and CPU-only code checks.

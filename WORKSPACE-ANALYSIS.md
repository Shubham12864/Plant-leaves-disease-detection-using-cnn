# Workspace analysis — Plant leaf disease detection using CNN

> **Superseded.** This is a snapshot from 2026-10-18. Several statements below have since changed (for example, the EfficientNet-B0 checkpoint was restored and the locked test split was re-scored on 2026-10-25). For the current state, read [`PROJECT_STATUS.md`](PROJECT_STATUS.md) and [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md). Kept for history only.

Analysis date: 2026-10-18. Scope: the whole working tree (`C:\Users\Shubham\Downloads\Capstone project`).
Method: read every source, script, config, notebook, manifest and report; then independently
re-evaluated the saved checkpoints on CPU and re-audited the split manifests.

---

## 1. What the workspace is

| Property | Value |
|---|---|
| Project | Capstone, IILM — "Plant leaves disease detection using CNN", group BT40 |
| Code | Python 3.10+ package `src/plant_disease` (8 files, ~250 LOC) + 17 scripts |
| Compute model | Training/evaluation runs in Kaggle notebooks; repo holds code, manifests, results |
| Data on disk | 54,305 PlantVillage images + 2,578 PlantDoc images (2.8 GB) |
| Workspace size | 4.97 GB, 111,458 files |
| Git | 5 commits on `main`, 1 uncommitted edit, 68 tracked files, 13 MB of objects |
| Tests / CI / lockfile / LICENSE | none |

The pipeline is: pinned download → materialize → SHA-256 manifest → leakage-aware freeze →
Kaggle training → results transcribed into `reports/`.

---

## 2. Headline verdict

The **data and reporting layer is genuinely good** — far above typical capstone standard:
pinned dataset revisions, per-image SHA-256, group-aware splitting, frozen split reports,
audit JSON, explicit "do not infer" statements, and a literature survey that refuses to invent
17 missing citations.

The **model layer is thin and partly unverifiable**, and the **research contribution described in
the project's own plan (domain shift, uncertainty, explainability) is 0% implemented**.

Verified strengths, verified defects, and a prioritized fix list follow.

---

## 3. Claims that I verified as correct

### 3.1 Independent re-evaluation of the saved checkpoints

I rebuilt each model from `src/plant_disease/models`, loaded the shipped checkpoints, and ran
them over the shipped manifests on CPU in FP32 (17,253 image predictions, no GPU).

| Model | Split | Reported accuracy | Reproduced accuracy | Verdict |
|---|---|---:|---:|---|
| SmallCNN | validation (6,544) | 0.96011614 | 0.9601161369193154 | **exact** |
| ResNet18 | validation (6,544) | 0.96149144 | 0.9614914425427873 | **exact** |
| SmallCNN | locked test (10,709) | 0.96946494 | 0.9693715566346064 | 1 image of 10,709 differs |
| EfficientNet-B0 | validation / test | 0.99541565 / 0.99505089 | — | **checkpoint absent, cannot verify** |

Macro-F1 differed by 1e-5 … 3e-4 while accuracy was bit-identical, which is the expected
signature of GPU FP16 autocast vs CPU FP32 changing a couple of near-tie argmaxes — not of a
wrong number. **The reported SmallCNN and ResNet18 results are real and reproducible.**

### 3.2 Documentation vs data

| Claim | Source | Status |
|---|---|---|
| 38 classes / 14 crops / 54,305 images | README | correct |
| 36,762 train / 6,544 val / 10,709 test | README, MEMORY | correct |
| Zero exact-hash and zero group-ID overlap across partitions | `data_audit.json` | **independently confirmed** (0/0/0 for sha256, group_id and path on all three partition pairs) |
| Test split not used for model selection | README, Kaggle metadata | correct (`test_manifest_opened: false`, `locked_test_evaluated: false`, notebook 01 reads only the development manifest) |
| Every headline number in README | vs report CSVs | **all 15 values match to the rounding** |
| 290 PlantVillage + 11 PlantDoc rows excluded | `split_freeze.json` | consistent with manifest arithmetic |
| ResNet18 has no locked-test score | README | correct, and correctly left blank |

---

## 4. Defects, ordered by impact

### P0 — blocks a defensible submission

**1. The best model is missing from the workspace.**
There is no EfficientNet-B0 checkpoint anywhere (`*.pt`, `*.pth`, `*.onnx`, `*.ckpt`, zips).
`checkpoints/source/model_comparison_artifacts.zip` contains only `resnet18/` and `smallcnn/`
and its own metadata says `completed_models: [smallcnn, resnet18]`.
Consequences: the 99.51 % test / 99.37 % validation numbers cannot be re-run, `scripts/
run_accuracy_pipeline.py` fails on its default EfficientNet path, and Notebook 2 depends on a
Kaggle dataset that is not in the repo. The EfficientNet **locked-test classification report and
confusion matrix are also absent**, so `reports/model_comparison/README.md`'s statement that the
scores were "checked against the PlantVillage classification report and confusion matrix" is
not checkable from this workspace.

**2. ResNet18 training code does not exist.**
Only `train_efficientnet_b0_ddp.py` is in `scripts/`. ResNet18 was trained by a Kaggle script
that was never saved — only its outputs were (`resnet18_last.pt` carries `resume_config`,
`history`, `rng_state`). Two of three models therefore cannot be retrained from the repo.

**3. The project's actual research question has no result.**
`MEMORY.md` / `ARCHITECTURE.md` define the contribution as robustness beyond clean images,
uncertainty/abstention, and interpretability. Grep across `src/` and `scripts/`:

| Feature | Matches |
|---|---|
| `grad.?cam` | 0 |
| `calibrat` | 0 |
| `abstain` | 0 |
| `uncertain` | 0 |
| `yolo` | 0 |

`configs/data_contract.yaml` already lists `calibration` and `selective_risk_coverage` as
required metrics; nothing computes them. What exists is an accuracy comparison on a clean
benchmark — the part the project itself says is *not* the contribution.

**4. The field-domain evaluation is blocked by an incomplete label map.**
`data/label_mapping_review.csv` maps 16 of 28 PlantDoc classes, and **all 16 are
`review_required`** — none is confirmed. The 12 unmapped classes include:

- all 10 healthy classes (`Apple leaf`, `Tomato leaf`, `grape leaf`, …)
- `Corn leaf blight` (disease, 179 train / 12 test)
- `grape leaf black rot` (disease, 56 train / 8 test) — despite `Grape___Black_rot` existing

So PlantDoc cannot be scored today, and PlantDoc is the only field-domain data in the project.

**5. A tracked manifest summary contradicts the tracked manifest.**
`data/manifests/plantvillage_development.summary.json` (committed) describes a **superseded
split**: 43,596 rows, 37,057 train / 6,539 validation, from `assign_group_splits.py`, and even
warns "Greedy group-size assignment is not class-stratified". The real
`plantvillage_development.csv` has 43,306 rows / 36,762 / 6,544 from
`freeze_evaluation_splits.py`. The summary is stale and its warning does not apply.
(PlantDoc development has no summary at all.)

**6. Uncommitted corruption in a tracked document.**
`git diff` shows `reports/guide_brief/UPDATED_LITERATURE_SURVEY.md` front matter was damaged:

```
-title: Updated Literature Survey — Plant Leaves Disease Detection Using CNN
+title:  — Plant Leaves Disease Detection Using CNN
```

### P1 — weakens the science

**7. The three-model comparison is not controlled.** Only the split is held constant; everything
else differs:

| | SmallCNN | ResNet18 | EfficientNet-B0 |
|---|---|---|---|
| LR | 1e-3 | 3e-4 (from `resume_config`) | 1e-4 |
| Global batch | 64 | 64 | 32 (16/GPU) |
| Parallel | DataParallel | DataParallel | DDP |
| Data path | PIL decode per batch | memmap cache + producer thread | memmap cache, `num_workers=0` |
| Augmentation | torchvision flip/rotation/jitter | not in repo | custom GPU flip/rotation/brightness |
| Train minutes | 70.2 | 9.45 | 104.08 |

Any of these can move PlantVillage accuracy by >1 pp. Calling it a "same-split comparison" is
accurate; calling it a controlled ablation is not.

**8. The EfficientNet run was ~95 % idle.** 8.7 min/epoch for a *smaller* architecture than
ResNet18 (0.86 min/epoch) is not credible. Per-epoch time grew linearly (1.7 → 15.65 min),
the signature of page-cache eviction on the 5.5 GB mmap cache read with `num_workers=0`.
At ~450 ms per global batch of 32 the GPUs were starved. A decoded cache + workers should
finish this in ~1 min/epoch.

**9. No confidence intervals anywhere**, although the project plan requires them. I computed
image-level bootstrap CIs (4,000 resamples) from the shipped confusion matrix:

| Metric | Value | 95 % CI |
|---|---:|---|
| SmallCNN test accuracy | 96.95 % | [96.63, 97.26] |
| SmallCNN test balanced accuracy | 96.02 % | [95.48, 96.53] |
| SmallCNN test macro-F1 | 96.07 % | [95.59, 96.53] |

The SmallCNN↔EfficientNet gap is far outside these intervals, so the ranking is safe — but the
claim "ResNet18 95.46 vs SmallCNN 94.44" on validation is being read without any interval.

**10. `balanced_accuracy` and `macro_recall` are the same number.** In every result table they
are identical to 8 decimals (mean per-class recall). They are presented as independent
evidence; they are not.

**11. The locked test set is no longer locked.** It has been evaluated twice (SmallCNN, then
EfficientNet) after selection, and both scores are now known. It can no longer act as an
untouched holdout for any third model.

**12. PlantDoc's statistical power is very low.** 236 test images over 27 classes (4–12 images
per class). A 95 % CI on accuracy is ±6.4 pp — versus ±0.95 pp for the PlantVillage test set.
Any "field accuracy" number from this set will be too noisy to support a conclusion without
per-class support reporting.

**13. Leakage protection covers only 76 % of images.** 13,194 of 54,305 PlantVillage images
(24.3 %) got a `fallback_<image_id>` group ID because the upstream leaf map did not match them;
fallback IDs are per-image, so those images have *no* group protection. 9,328 are in train, 900
in validation, 2,676 in the published test. The "zero group overlap" audit is true but weaker
than it reads. The matched 7,593 groups are large (7,525 have ≥4 images), so the mechanism
works where it applies.

**14. A perceptual-hash probe finds a small residual duplicate across partitions.** The audit
checks exact SHA-256 only. A 64-bit dHash over all 54,015 images found one credible cross-split
group: `Corn_(maize)___Common_rust_/RS_Rust 2439 / 2464 / 2527 / 2532` — identical perceptual
hash, spanning train, validation **and test**. Also 2 train↔test, 1 validation↔test and
2 train↔validation pairs at distance 0. The ≤4-bit matches are mostly cross-class pairs and
should be treated as false positives of a coarse hash on uniform backgrounds.
Impact on the metric is negligible (1 test image = 0.01 pp), but the "no duplicates" statement
needs the word *exact*.

**15. The recorded integrity anchor cannot be reproduced.** `resnet18_last.pt` stores
`manifest_sha256 = f2d892b3…`. I tested 24 representations of the shipped manifest (raw CRLF,
LF-normalized, git blob, pandas re-serialization, sorted rows, train-only, path lists, sha256
column joins, JSON split map) — **none matches**. Content appears unchanged (the row multiset by
`sha256`+`label`+`split` is identical between the Kaggle upload and the repo, and both
SmallCNN/ResNet18 re-evaluations match), so this is most likely a different serialization rather
than a different split — but the chain of custody is broken at the one place it is asserted.

**16. The Kaggle upload snapshot is stale.** `kaggle_upload/PlantVillage.zip` has
`manifest_sha256 = 7db5f0c7…` and differs from the repo manifest in exactly 3 filenames, where
the upload has `… R.S_HL 8101 copy.jpg` and both the repo manifest and the files on disk have
`… R.S_HL 8101 copy .jpg` (space before the extension). The zip is internally consistent, so past
runs were fine, but it is not the current dataset.

### P2 — engineering and hygiene

17. **No local test-evaluation path.** `src/plant_disease/data.py` hard-codes
    `data/raw/PlantVillage/train` as the image root and `evaluate.py` refuses any split other
    than validation, so test scoring exists only inside Notebook 2 (which is why I had to write
    my own loader to check the headline number).
18. **Duplicated model definitions.** `SmallCNN`/`resnet18`/`efficientnet_b0` are re-declared in
    notebooks 02, 03 and the baseline notebook instead of importing `src/plant_disease/models`.
    They currently agree; they will not stay in agreement.
19. **`torch.load(..., weights_only=False)`** on every checkpoint and on downloaded Kaggle zips —
    arbitrary pickle execution. Prefer `weights_only=True` with the metadata you need.
20. **Dead build script.** `scripts/build_efficientnet_notebook.py` reads
    `reports/model_comparison/previous/validation_comparison.csv`, which does not exist, so it
    cannot regenerate notebook 01.
21. **Pipeline output drift.** `run_accuracy_pipeline.py` writes
    `reports/model_comparison/accuracy_pipeline/validation_comparison.csv` with different columns
    from the tracked `plantvillage_validation_comparison.csv`, so re-running produces a second,
    divergent source of truth.
22. **Chart with a truncated axis.** `plantvillage_validation_comparison.png` starts at y = 90,
    which visually triples the SmallCNN↔ResNet18 difference.
23. **Training-curve provenance.** `reports/model_comparison/README.md` says the curves use the
    Phase 2 SmallCNN history and that the CSV is the source, but
    `plantvillage_model_training_history.csv` contains **only** resnet18 + efficientnet_b0 rows —
    the SmallCNN series in the PNG has no shipped CSV source.
24. **Missing provenance file.** `download_datasets.ps1` fetches
    `data/raw/PlantVillage/source/DATASET_CARD.md`; it is not on disk. The PlantDoc license and
    README are present and the registry's CC BY 4.0 claim is correct.
25. **`data.zip` is deleted** after extraction, so `materialize_plantvillage.py` cannot be re-run
    without re-downloading (documented, but worth pinning a checksum of the archive).
26. **No tests, no requirements lock, no LICENSE, no CI.** `pyproject.toml` has loose ranges
    (`torch>=2.1`); the runs used torch 2.11.0+cu128 and torchvision 0.26.0+cu128.
27. **Storage:** 4.97 GB for a repo whose git history is 13 MB.
    PlantVillage exists three times — `data/raw` 831 MB, `kaggle_upload/PlantVillage` 840 MB,
    `kaggle_upload/PlantVillage.zip` 855 MB — and PlantDoc raw is 1.99 GB of unused field photos.
28. **`checkpoints/` and `kaggle_upload/` are gitignored**, and `MEMORY.md`, `ARCHITECTURE.md`,
    `PHASE-II-ROADMAP.md`, `Literature-Survey-Evidence-Register.md` are excluded through
    `.git/info/exclude`. The "source of truth" document is therefore the only one with no
    backup, no history and no review.

### Documentation is stale in four places

| File | Says | Reality |
|---|---|---|
| `MEMORY.md:166` | "Kaggle GPU baseline notebook is prepared but untrained; no metric/result exists" | Contradicts its own line 5 and all results |
| `ARCHITECTURE.md:9,184` | "prepared but not trained", "No model has been trained and no performance result exists" | Two models trained and scored |
| `PHASE-II-ROADMAP.md:5` | "Kaggle baseline training is next" | Training and comparison are done |
| `Literature-Survey-Evidence-Register.md:7` | "source PDF missing; bibliography incomplete" | `reports/guide_brief/UPDATED_LITERATURE_SURVEY.md` records all 21 entries and marks ~11 verified |

`README.md` and everything under `reports/` are current and accurate.

---

## 5. Notable analytical findings (from the shipped results)

Top off-diagonal confusions on the SmallCNN test split (327 errors in 10,709):

| Count | True | Predicted |
|---:|---|---|
| 23 | Corn Cercospora / Gray leaf spot | Corn Northern Leaf Blight |
| 21 | Potato Late blight | **Tomato** Late blight |
| 20 | Soybean healthy | **Pepper, bell** healthy |
| 15 | Tomato Target Spot | Tomato Spider mites |
| 13 | Tomato Early blight | Tomato Target Spot |

Weakest classes by F1: Corn Cercospora 0.851 (recall 0.776), Potato healthy 0.875,
Tomato Early blight 0.878. Strongest: Squash Powdery mildew 1.000, Orange HLB 0.996.

Two of these are worth a paragraph in the final report:

- the confusions are **semantically plausible within a crop** (Early/Late blight, Cercospora/
  Northern blight), which supports "the model learned lesions";
- but 20 of the 32 `Soybean___healthy` errors land on a **different crop's** healthy class, which
  suggests background/leaf-appearance cues. That is exactly the spurious-cue risk the project
  plan warns about, and it is direct evidence for the Grad-CAM audit that has not been run.

Class imbalance is 38.4× (Orange HLB 3,842 vs Potato healthy 100 train images), and the loss is
unweighted. Validation support drops to 20 images for `Potato___healthy`, so that class
contributes 1/38 of every macro metric on very thin evidence.

---

## 6. Prioritized actions

**P0 — do before the next guide review**

1. Commit the EfficientNet-B0 best checkpoint and its locked-test
   `classification_report` + `confusion_matrix` (re-run Notebook 2 if they are gone), or
   explicitly downgrade every EfficientNet claim to "reported, artifact not retained".
2. Save the ResNet18 training script into `scripts/` and re-state the comparison as
   "per-architecture tuned", listing LR / batch / augmentation / parallelism per model.
3. Finish `data/label_mapping_review.csv`: add the 10 healthy classes plus `Corn leaf blight`
   and `grape leaf black rot`, and move each row from `review_required` to a decision.
   Then run the PlantDoc evaluation and report per-class support with it.
4. Delete or regenerate `plantvillage_development.summary.json`; have
   `freeze_evaluation_splits.py` write it so it can never go stale again.
5. Revert the damaged title in `UPDATED_LITERATURE_SURVEY.md` and commit.

**P1 — makes the result publishable**

6. Add bootstrap CIs (block-bootstrap over `group_id`) to every table; the numbers in §4.9 are a
   starting point.
7. Implement calibration (reliability plot + ECE/Brier) and selective risk/coverage on
   validation, and report coverage + retained accuracy. This is the stated contribution.
8. Run the Grad-CAM audit on correct / wrong / high-confidence-wrong cases, starting with the
   Soybean→Pepper healthy errors.
9. Fix the EfficientNet data path (decode cache + workers) and re-time it; publish minutes/epoch.
10. Resolve `manifest_sha256`: record the hash of the exact manifest bytes used, and check it in
    the training script.
11. Add three tests: split integrity (0 hash/group overlap, 38 classes everywhere), checkpoint
    contract (classes + normalization present), metric sanity (balanced accuracy == macro recall).

**P2 — hygiene**

12. Delete `kaggle_upload/PlantVillage/` (840 MB) and keep only the zip, or vice versa; move
    PlantDoc raw (1.99 GB) out of the tree if it is not needed for the next run.
13. `weights_only=True` in all `torch.load` calls.
14. Import models from `src/` in the notebooks instead of redefining them.
15. Fix or delete `build_efficientnet_notebook.py`; align `run_accuracy_pipeline.py` output with
    the tracked comparison CSV.
16. Add a lockfile, a LICENSE, and a CI job that runs the manifest audit.
17. Start the chart y-axis at 0 (or annotate the truncation), and ship the SmallCNN training
    history inside `plantvillage_model_training_history.csv`.
18. Refresh the four stale planning documents, or mark them "historical".

---

## 7. Bottom line

- Data engineering, provenance and honesty: **strong**. Verified.
- Reported SmallCNN / ResNet18 numbers: **reproduced exactly**.
- EfficientNet-B0 (the headline model): **unverifiable from this workspace**.
- Field-domain evaluation, uncertainty, explainability: **not started** — and those are the
  project's declared contribution.
- Documentation: accurate where it matters (README, reports), stale in the four planning files.

The shortest path from here to a defensible capstone is: restore the missing EfficientNet
artifacts, finish the PlantDoc label map, add CIs + calibration + abstention, and run one
Grad-CAM audit. None of that requires new data or new compute beyond a single Kaggle session.

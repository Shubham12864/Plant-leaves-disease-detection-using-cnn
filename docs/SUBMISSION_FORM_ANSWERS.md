# Capstone Phase III and Mid-Term submission: form answers

Project ID **BT40**.

**Team:** Shubham Kumar `<enrolment>`, Ujjwal Pratap Singh `<enrolment>`, Vivek Kumar Prajapati `<enrolment>`, Shubham Kumar `<enrolment>`

Paste-ready text for the nine sections of the Google Form. Every number here is
registered in `results/registry.json` with the artifact that produced it, so any
claim can be traced back to a file.

> **Two members are named Shubham Kumar.** Both are listed above, in the order
> given. The form will need enrolment numbers to tell them apart, and those are
> not recorded anywhere in this repository, so they are left as `<enrolment>`
> rather than guessed. Do not merge the two entries when typing the form.


**Verify before submitting, because these are not mine to know:**

- Supervisor name and official email. The approved guide is **Dr. Harendra
  Singh**; the email address is not recorded anywhere in this repository and must
  be confirmed by the student.
- Project title. The working title used throughout is *Plant Leaf Disease
  Detection Using CNN*. If the university document carries a different approved
  title, that one wins.
- Three WPRs must already be with the supervisor. Drafts are in
  `docs/WPR_1.md`, `docs/WPR_2.md`, `docs/WPR_3.md`.

---

## 1. System architecture and design

Data preparation reads the frozen manifests and decodes every image once into a
`uint8` memory-mapped cache. Train and validation are separate files keyed by a
hash of the path list, so changing the split invalidates the cache instead of
silently reusing it.

Training is a single engine, `scripts/train_ddp.py`, parameterised by
`configs/protocol_v2.json`. It runs on two GPUs through DistributedDataParallel,
applies augmentation on the accelerator, and writes three artifacts per run: a
best checkpoint, a resumable last checkpoint, and a metrics JSON carrying the
split fingerprint, the applied learning rate per epoch, and the full
hyperparameter set.

Evaluation is separate from training on purpose. `scripts/evaluate_locked_test.py`
scores PlantVillage splits with bootstrap confidence intervals, and
`scripts/evaluate_plantdoc.py` scores the field dataset through the label map,
reporting out-of-mapping predictions rather than hiding them. Explainability and
calibration sit on top: `scripts/occlusion_probe.py` for background sensitivity,
`scripts/explain_field_failures.py` for Grad-CAM, and a calibration and
abstention analysis.

Everything reported passes through `results/registry.json`, which
`scripts/build_tables.py` renders into tables. The registry check fails the build
if a document quotes a number that is not registered against an artifact.

Stack: Python 3.11, PyTorch 2.9.1 with torchvision, NumPy, pandas, scikit-learn,
Matplotlib. Training on Kaggle 2× Tesla T4; all evaluation local on CPU so the
numbers can be re-derived without a GPU.

**Evidence:** `ARCHITECTURE.md`, `src/plant_disease/`, `scripts/`,
`configs/protocol_v2.json`, `results/registry.json`.

---

## 2. Methodology and implementation

Four architectures are compared under one frozen protocol: SmallCNN (398,662
parameters, trained from scratch), ResNet18 (11,196,006 with a 38-class head),
EfficientNet-B0 (4,056,226) and ConvNeXt-Tiny (27,849,350).

Protocol v2 is AdamW at learning rate 1e-4 with weight decay 1e-4, a constant
schedule, global batch 32, 12 epochs, early stopping with patience 3 on
validation macro-F1. SmallCNN is the single documented deviation: learning rate
1e-3 and global batch 64, taken from the configuration that produced its
earlier 96.01% result. Augmentation is horizontal flip, rotation up to 12
degrees, brightness and per-channel gain, applied on the GPU.

The reference model is trained with three to five seeds so variance is reported
rather than a single lucky run. Evaluation uses 2,000-round bootstrap intervals
over images, and macro averages are computed only over classes with non-zero
support, which matters because an earlier 96-image check produced a meaningless
macro-F1 of 0.026 by averaging over absent classes.

Field transfer is measured by mapping all 28 PlantDoc classes to PlantVillage
classes with an explicit relationship and caveat per row, then scoring only the
mapped subset. Nothing in PlantDoc was used for training, augmentation, early
stopping or threshold selection.

Interpretability uses Grad-CAM on the target convolution block, plus an occlusion
probe that replaces background pixels identified by an HSV leaf mask. No
database is used; the artifacts are files plus a JSON registry. No cloud API is
called at any point.

**Evidence:** `configs/protocol_v2.json`, `scripts/train_ddp.py`,
`scripts/complete_label_mapping.py`, `data/manifests/plantdoc_to_plantvillage_map.csv`.

---

## 3. Prototype development

Two things run today.

The research pipeline is complete and reproducible. `scripts/run_all.py --check`
executes six gates in under a minute: 17 pytest cases, the table build, the
registry check, a batch-composition check, a command-contract test, and 13
failure-path tests for the unattended training notebook. All six pass. The
training engine resumes from a checkpoint, refuses to resume when the split
fingerprint differs, and refuses to run when hyperparameters have changed.

The second is a static front end in `app/static/`, four pages: a landing page,
single-image scan with a Grad-CAM slider, batch upload with per-image progress
and CSV export, and a full inference view showing all 38 class scores, per-stage
timings, the checkpoint identity and the raw API JSON. It has no build step and
opens in a browser.

The front end is not wired to a live model. When no backend
URL is configured, `app/static/js/api.js` fails fast with code `demo` and each
page shows recorded checkpoint output instead of inventing a prediction. The
FastAPI service that would serve live inference is specified in
`docs/FRONTEND_PLAN.md` and is not implemented. So the prototype demonstrates
the interface and the recorded results, not a deployed classifier.

The interface is described in `app/README.md` and `docs/FRONTEND_PLAN.md`; the
demo fallback is in `app/static/js/api.js`.

---

## 4. Dataset and experimental setup

PlantVillage: 54,305 images across 38 classes and 14 crops. The frozen split is
36,762 train, 6,544 validation and 10,709 test. Train and validation are
disjoint by path and every class appears in both; the test split is disjoint from
both and was opened once for the reference evaluation, reproducing bit-identical
results (53 errors in 10,709).

PlantDoc: 2,578 field images across 28 classes, CC BY 4.0. The test split of 236
images is the entire field evaluation and has never been trained on.
Preprocessing is resize to 224×224, conversion to RGB, scaling to [0,1] and
ImageNet normalisation, identical for every model and both datasets.

Augmentation and optimisation are as described in section 2. Training hardware
was two Tesla T4 GPUs on Kaggle with mixed precision; the fourteen-run sweep
completed in about 2.6 hours of GPU time with no run failures. Evaluation ran on
CPU. Versions are pinned in `requirements.lock`, and each checkpoint stores the
torch and torchvision versions used to produce it.

Every run records a dataset fingerprint. All fourteen sweep runs share the
fingerprint `01c1fe7339df2701`, which is how the project proves they saw the same
split rather than assuming it.

Counts and split definitions are in `DATASET_CARD.md` and the manifest summary
files; pinned versions are in `requirements.lock`.

---

## 5. Testing and validation

Seventeen automated tests run in `tests/`: split integrity and disjointness,
checkpoint contract including strict state-dict loading and parameter counts,
metric sanity, and evaluator output contract. Two of them caught real errors
during this work: ResNet18 had been recorded with the parameter count of the
1000-class ImageNet model rather than the 38-class one, and a metrics function
signature was assumed incorrectly.

Validation beyond unit tests:

The training engine has a smoke test covering a full two-epoch distributed run,
the checkpoint contract, the locked-test evaluator, and the resume fingerprint
guard across all four architectures.

The unattended training notebook has 13 failure-path tests that execute the real
notebook cells against stubbed worker processes: a failed gate ships its evidence
without starting the sweep, a crashed run does not abort the loop, a hung run is
killed by timeout, finished seeds are skipped on restart, an exhausted time
budget starts nothing, and partial results still produce a downloadable archive.

Defects found and corrected during validation: a frozen protocol that could not
reproduce the reference result (root cause: blocked shuffling reduced batches to
a median of three classes and broke BatchNorm), a learning-rate schedule that
decayed to 1% of nominal, an interpretation probe that erased leaf along with
background, and a torchrun invocation that passed the Python interpreter as the
training script.

**Evidence:** `tests/`, `scripts/smoke_test_engine.py`,
`scripts/test_overnight_logic.py`, `docs/KNOWN_ISSUES.md` (defects R12 to R15).

---

## 6. Quantitative results and performance analysis

All figures below are PlantVillage validation (6,544 images) unless marked
PlantDoc. Full analysis and figures are in `reports/`.

Architecture comparison under protocol v2, mean macro-F1 with the spread across
seeds: ConvNeXt-Tiny 0.98951 (3 seeds, sd 0.00028), EfficientNet-B0 0.98946
(5 seeds, sd 0.00171), ResNet18 0.98460 (3 seeds, sd 0.00166), SmallCNN 0.91599
(3 seeds, sd 0.00865).

The reference model and ConvNeXt-Tiny differ by 0.00005, which is thirty-four
times smaller than the reference model's own seed spread. On this benchmark the
architecture does not decide the outcome.

Field transfer on PlantDoc test (236 images, 27 mapped classes): accuracy falls
from 0.9928 to 0.2085 averaged over five protocol-v2 seeds, a drop of 78.4
percentage points. The published single checkpoint scores 0.1737 with a 95%
interval of 0.1271 to 0.2246; it sits 3.5 points below the mean of its own
architecture and is reported separately rather than as the headline. For the
published checkpoint all 27 mapped classes fell, median drop 0.851, and expected
calibration error rose from 0.0026 to 0.5808. In the top confidence band, 65
images above 0.95 confidence, accuracy is 29.2%. A confidence threshold gains 0.8
points at 90% coverage, so abstention does not recover a usable subset.

Scoring all fourteen checkpoints on the field set shows the lab ranking order is
preserved but the lab gaps are not: ConvNeXt-Tiny 0.3008 (3 seeds, sd 0.038),
EfficientNet-B0 0.2085 (5 seeds, sd 0.030), ResNet18 0.1610 (3 seeds, sd 0.027),
SmallCNN 0.0438 (3 seeds, sd 0.011). The two models tied on the lab benchmark are
9.2 percentage points apart in the field, and the model that trails by 7.4 points
in the lab trails by 25.7 in the field.

Mechanism: replacing the background while keeping the leaf costs 56.5 points on
PlantVillage (0.9975 to 0.4325) and changes 56.8% of predictions, which is direct
evidence that lab accuracy is substantially background-dependent.

Taken together these say that the lab benchmark ranks the four models almost
arbitrarily once they are close, and that neither of the two things a practitioner
would do about the field gap, choosing the best lab model or thresholding
confidence, recovers a usable system. Latency was not benchmarked and is not
reported.

Files behind every figure: `reports/model_comparison/SWEEP_V2_RESULTS.md`,
`reports/field_gap/FIELD_GAP_RESULTS.md`,
`reports/field_gap/USEFULNESS_RESULTS.md`,
`reports/mechanism/MECHANISM_RESULTS.md`, `results/registry.json`.

---

## 7. Integration and verification

Modules are verified at their interfaces rather than assumed to fit.

The training engine and the sweep worker agree on every architecture: a parity
check builds each model through the package and through the engine and compares
parameter counts and state-dict keys, matching for all four.

The notebooks are generated by builder scripts that embed the engine, the worker
and the protocol byte for byte, and the build fails if an embedded file differs
from the repository version. This exists because a hand-edited notebook cell
silently diverged once already.

The registry is integrated with the documents. `scripts/build_tables.py` renders
the reported tables from `results/registry.json` and `--verify` fails when a
claim has no registered run, when an artifact directory is missing, or when a
generated table is stale. Deleting the registry exits non-zero by design.

Cross-run consistency is checked by fingerprint: all fourteen sweep runs report
the same split hash, and the metrics JSON records the applied learning rate per
epoch so a schedule defect is visible in the artifact rather than inferred.

**Evidence:** `scripts/smoke_test_engine.py`,
`scripts/build_training_notebook.py`, `scripts/build_tables.py`,
`scripts/run_all.py`.

---

## 8. Technical challenges and limitations

The most expensive problem was a frozen training protocol that looked reasonable
and could not learn. Twelve runs across four architectures came back between
0.07 and 0.95 where the reference model was at 0.99. The cause was not the
learning rate, which is what the configuration suggested. Blocked shuffling was
permuting 2,048-image blocks on a manifest sorted by class, so the median batch
contained three distinct classes instead of twelve, and BatchNorm normalised
using statistics from three classes. The tell was that the only architecture that
trained, ConvNeXt-Tiny, uses LayerNorm and ignores batch composition. Fixing the
shuffle mode restored the reference behaviour and the next sweep reproduced
0.9836 validation accuracy in the first epoch.

A smaller but instructive failure: the first occlusion probe filled the image
border with flat grey, which also erased part of the leaf because the leaf
extends into the outer 20%. Both domains collapsed to near zero, which would have
been reported as strong background dependence. A rendered check of the perturbed
images caught it, and the probe was rebuilt around a leaf mask.

Open limitations. The field evaluation rests on 236 images from a single source,
with four to twelve images per class, so per-class field numbers are indicative.
Part of the 78-point gap is likely PlantDoc annotation noise, which this work
does not estimate. The field results use one checkpoint per architecture, so the
8.9-point ConvNeXt lead carries no seed variance and should be replicated before
it is treated as an architectural property. The background augmentation
experiment is blocked: it needs a licence-clean background corpus disjoint from
PlantDoc, and using PlantDoc backgrounds would contaminate the field evaluation
that produced the headline result.

**Evidence:** `reports/model_comparison/SWEEP_V1_INVALID.md`,
`reports/mechanism/perturbation_check.png`, `docs/KNOWN_ISSUES.md`.

---

## 9. Milestone achievement and future work

Completed. Frozen split with a proven disjointness check. Reference checkpoint
restored and structurally verified. Four architectures on one engine with a
single protocol. A fourteen-run sweep with zero failures and seed variance
reported. A complete PlantDoc label map with three caveats recorded. Field
evaluation with bootstrap intervals and out-of-mapping accounting. Calibration,
abstention and an occlusion probe for all four models. Seventeen tests, a
registry that gates the documents, and a six-gate reproducibility script.

Partially done. Explainability has Grad-CAM overlays for 18 field cases and an
automated saliency statistic, but the 50 hand-labelled maps the plan requires are
not finished; the labelling sheet is written and empty. The front end is built
and honest about being demo-only, with no live inference service.

Planned and not started. Background augmentation with paired bootstrap
comparison, blocked on obtaining a licence-clean background corpus. A second
independent field dataset with expert labels, which is what would move this from
a single-dataset validity study to a publishable claim. Out-of-distribution
detection as a replacement for the confidence threshold that was measured and
found ineffective.

One point of consistency with the approved abstract is worth stating here rather
than leaving for a reader to find. The abstract commits to "a confidence threshold
will flag uncertain cases". That mechanism was implemented and measured. On field
data it gained 0.8 percentage points at 90% coverage, so the expectation was not
confirmed, and it is reported as a negative finding rather than as a delivered
feature.

Remaining work is decision work and writing, not compute: fill the labelling
sheet, settle the background corpus question, and agree the abstract wording with
the guide where the field evidence no longer supports it.

**Files:** `PROJECT_STATUS.md`, `docs/ROADMAP.md`,
`reports/paper/RESULTS_BY_HYPOTHESIS.md`, `docs/ABSTRACT_VS_EVIDENCE.md`.

**Evidence:** `PROJECT_STATUS.md`, `docs/ROADMAP.md`,
`reports/paper/RESULTS_BY_HYPOTHESIS.md`.

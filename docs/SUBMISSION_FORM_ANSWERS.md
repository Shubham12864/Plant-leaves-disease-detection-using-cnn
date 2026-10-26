# Capstone Phase III and Mid-Term submission: paste-ready answers

Project ID **BT40**. Title: *Plant Leaf Disease Detection Using CNN*.

Full evidence, tables and the mapping to artifacts are in
`reports/submission/BT40_PhaseIII_MidTerm_Submission.pdf`. The text below is the
short version for the Google Form fields.

**Fill before submitting (not recorded in the repository, do not guess):**
enrolment numbers for all four members; the guide's official email; confirmation of
the approved title. Two members are named Shubham Kumar, so enrolment numbers are the
only reliable way to keep the entries separate.

---

## 1. System architecture and design

Cached decoding reads the frozen manifests once into a `uint8` memory-mapped store,
keyed by a hash of the path list, so a changed split invalidates the cache rather than
being silently reused. Training is one engine, `scripts/train_ddp.py`, driven by
`configs/protocol_v2.json`, running DistributedDataParallel on two GPUs with
augmentation applied on the accelerator. Each run writes a best checkpoint, a
resumable checkpoint, and a metrics JSON carrying the split fingerprint, the learning
rate applied per epoch, and the full hyperparameter set.

Evaluation is deliberately separate from training: `evaluate_locked_test.py` scores
PlantVillage with bootstrap intervals, `evaluate_plantdoc.py` scores the field set
through the label map and reports out-of-mapping predictions instead of hiding them.

Every reported number passes through `results/registry.json`, which
`scripts/build_tables.py` renders into tables. The build fails if a document quotes a
number not registered against an artifact. The registry now backs 23 claims.

**Evidence:** `src/plant_disease/`, `scripts/`, `configs/protocol_v2.json`,
`results/registry.json`.

## 2. Methodology and implementation

Four architectures under one frozen protocol: SmallCNN (398,662 parameters, from
scratch), ResNet18 (11,196,006), EfficientNet-B0 (4,056,226, reference) and
ConvNeXt-Tiny (27,849,350). Protocol v2 is AdamW at 1e-4 with weight decay 1e-4,
global batch 32, 12 epochs, early stopping on validation macro-F1. SmallCNN is the one
documented deviation.

Beyond mid-term, three methodological additions: an intervention design that replaces
an image's leaf or background with a donor's and measures whether the prediction
follows the donor, which answers "what does this decision depend on" without human
labelling of saliency maps; a pre-registration of the expected augmentation direction
recorded before results returned; and a paired augmentation design of three seeds per
condition with one protocol.

Macro averages are computed only over classes with non-zero support, after an earlier
96-image check produced a meaningless macro-F1 of 0.026 by averaging over absent
classes. All 28 PlantDoc classes are mapped with a recorded relationship and caveat.

**Evidence:** `configs/protocol_v2.json`, `scripts/swap_localisation_test.py`,
`data/manifests/plantdoc_to_plantvillage_map.csv`.

## 3. Prototype development

Two working components. The research pipeline is complete: `scripts/run_all.py
--check` runs six gates in under a minute and all pass, and the training engine
resumes from checkpoint but refuses when the split fingerprint or hyperparameters have
changed.

The second is a static front end in `app/static/` with four pages: landing, single
image scan with a Grad-CAM slider, batch upload with CSV export, and a full inference
view showing all 38 class scores and timings. It has no build step and opens in a
browser.

It is not wired to a live model. With no backend configured, `app/static/js/api.js`
fails fast with code `demo` and each page shows recorded checkpoint output rather than
inventing a prediction. Demo assets are generated from a real checkpoint by
`scripts/make_demo_assets.py`. The serving layer is specified but not implemented, so
the prototype demonstrates the interface and the recorded results, not a deployed
classifier.

**Evidence:** `app/README.md`, `docs/FRONTEND_PLAN.md`, `app/static/js/api.js`.

## 4. Dataset and experimental setup

PlantVillage: 54,305 images, 38 classes, 14 crops. Frozen split of 36,762 train, 6,544
validation, 10,709 test. Train and validation are disjoint by path with every class in
both; test is disjoint from both and was opened once for the reference evaluation,
reproducing bit-identical results.

PlantDoc: 2,578 field images, 28 classes, CC BY 4.0. The 236-image test split is the
entire field evaluation and was never trained on. All 28 classes are mapped to
PlantVillage targets: 25 confirmed, 3 confirmed with a recorded caveat, none excluded.
27 classes have test images.

Preprocessing is resize to 224×224, RGB, scale to [0,1], ImageNet normalisation,
identical across models and datasets. Training ran on two Tesla T4 GPUs with mixed
precision; the fourteen-run sweep took about 2.6 hours with zero failures. Evaluation
runs on CPU so numbers can be re-derived without a GPU. Versions are pinned in
`requirements.lock`.

Every run records a dataset fingerprint; all sweep runs share one, which is how the
project proves they saw the same split rather than assuming it. The augmentation
experiment additionally requires a background corpus, admitted only through a
hash-disjointness gate that refuses any byte-identical overlap with either evaluation
set.

**Evidence:** `DATASET_CARD.md`, manifest summaries, `requirements.lock`.

## 5. Testing and validation

39 automated tests, plus six reproducibility gates run by `scripts/run_all.py
--check`. Coverage: split integrity and disjointness, the checkpoint contract with
strict state-dict loading and parameter counts, metric sanity, evaluator output
contract, calibration path, and the intervention experiment's output contract.

Two tests caught real errors during the work: ResNet18 had been recorded with the
parameter count of the 1000-class ImageNet model rather than the 38-class one, and a
metrics function signature had been assumed incorrectly.

The unattended training notebook has failure-path tests that execute the real cells
against stubbed workers: a failed gate ships evidence without starting the sweep, a
crashed run does not abort the loop, a hung run is killed by timeout, finished seeds
are skipped on restart, and partial results still produce a downloadable archive.

Three defects were found in this phase and are documented rather than hidden: a
generated notebook ran a check before the cell that wrote the script it needed; a
field evaluation failed remotely because the script resolved its project root one
level too high; and a reliability figure connected bins with a line, giving a
one-image bin the same visual weight as the bin holding 10,590 of 10,709.

**Evidence:** `tests/`, `scripts/smoke_test_engine.py`, `docs/KNOWN_ISSUES.md`.

## 6. Quantitative results and performance analysis

Validation macro-F1 under protocol v2: ConvNeXt-Tiny 0.98951 (3 seeds, sd 0.00028),
EfficientNet-B0 0.98946 (5 seeds, sd 0.00171), ResNet18 0.98460, SmallCNN 0.91599. The
top two differ by 0.00005, thirty-four times smaller than the reference model's own
seed spread, so on this benchmark architecture does not decide the outcome.

Locked test: macro-F1 0.9931, 95% interval [0.9910, 0.9951], calibration error 0.0022.

Domain gap: accuracy falls from 0.9928 to 0.1737 (five-seed mean 0.2085), a drop of
about 81 points on macro-F1.

Intervention on 298 validation images: replacing the leaf moved the prediction to the
donor's class 50.3% of the time (95% CI 44.6–55.7), replacing the background 21.8%
(17.1–26.8), against 2.6% chance and a 4.4% same-class control that measures the edit
alone. The leaf dominates, but the background is still eight times chance.

Augmentation experiment, three seeds per condition: in-domain macro-F1 0.9883 baseline
against 0.9796 composite; field macro-F1 0.2232 against 0.2356. Paired difference
+1.24 points, 95% CI −3.32 to +5.80, two of three seeds improved, 1.6% of the gap
recovered. No detectable effect.

Calibration error on field data rises to 0.5808. A confidence threshold gains 0.8
points at 90% coverage, so abstention does not recover a usable subset. Latency was
not benchmarked and is not reported.

**Evidence:** `reports/model_comparison/SWEEP_V2_RESULTS.md`,
`reports/field_gap/`, `reports/mechanism/`, `results/registry.json`.

## 7. Integration and verification

Modules are verified at their interfaces. A parity check builds each model through the
package and through the engine and compares parameter counts and state-dict keys;
all four match.

Notebooks are generated by builder scripts that embed the engine, worker and protocol
byte for byte, and the build refuses if an embedded file differs from the repository
version. This exists because a hand-edited notebook cell silently diverged once
already.

The registry is integrated with the documents: `build_tables.py --verify` compares
four generated tables against the registry and fails on a missing run, a missing
artifact directory, or a stale table. It runs on every build and currently passes with
23 claims backed.

Cross-run consistency is checked by fingerprint, and each metrics file records the
learning rate applied per epoch so a schedule defect is visible in the artifact rather
than inferred.

**Evidence:** `scripts/build_training_notebook.py`, `scripts/build_tables.py`,
`scripts/run_all.py`.

## 8. Technical challenges and limitations

The most expensive problem was a frozen protocol that looked reasonable and could not
learn: twelve runs between 0.07 and 0.95 where the reference was at 0.99. The cause
was not the learning rate. Blocked shuffling permuted 2,048-image blocks on a manifest
sorted by class, so the median batch held three classes instead of twelve and
BatchNorm normalised on statistics from three classes. The tell was that the only
architecture which trained, ConvNeXt-Tiny, uses LayerNorm and ignores batch
composition. Fixing the shuffle mode restored reference behaviour.

A second instructive failure: the first occlusion probe filled the image border with
grey, which also erased part of the leaf, and both domains collapsed to near zero.
That would have been reported as strong background dependence. A rendered check caught
it and the probe was rebuilt around a leaf mask.

In this phase, the notable failure was organisational rather than numerical: a
notebook passed its compile check and failed on first execution because of cell
ordering, and the error handler reported it as dataset contamination. A check that
cannot distinguish a missing input from a failed test will eventually report the wrong
one; the builder now refuses to emit such a notebook.

Open limitations. The field evaluation rests on 236 images from one source with four
to twelve images per class, so a single accuracy estimate carries a roughly ten-point
interval. Part of the gap is likely annotation noise, which this work does not
estimate. Three mappings are caveated rather than confirmed. The augmentation result
has three seeds, so its paired interval has two degrees of freedom and is wide.

**Evidence:** `reports/model_comparison/SWEEP_V1_INVALID.md`,
`docs/KNOWN_ISSUES.md`.

## 9. Milestone achievement and future work

Completed: frozen split with a disjointness gate; reference checkpoint restored and
structurally verified; four architectures on one engine; fourteen-run sweep with zero
failures and seed variance reported; complete PlantDoc mapping with caveats recorded;
field evaluation with bootstrap intervals and out-of-mapping accounting; calibration
and abstention for all four models; mechanism measured by intervention; augmentation
tested at three seeds per condition with a null result; 39 tests and six
reproducibility gates; a registry backing 23 claims; a 47-page report generated by
script from the same artifacts.

Partial: the front end is built and honest about being demo-only, with no live
inference service. Grad-CAM overlays exist for 18 field cases with an automated
saliency statistic, and the intervention experiment has since answered the question
the hand-labelled maps were intended to answer, so the labelling sheet is now a
visual cross-check rather than a dependency.

Planned and not started: a second independent field dataset with expert labels, which
is what would move this from a single-dataset validity study to a publishable claim;
and out-of-distribution detection as a replacement for the confidence threshold that
was measured and found ineffective; field-domain augmentation applied to the leaf
rather than the background, which the mechanism result points to.

Consistency note carried forward rather than left for a reader to find. The approved
abstract commits to a confidence threshold flagging uncertain cases and to targeted
augmentation for realistic variation. Both were implemented and measured. The
threshold gained 0.8 points at 90% coverage and the augmentation recovered 1.6% of the
gap, so neither expectation was confirmed, and both are reported as negative findings
rather than as delivered features.

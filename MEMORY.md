---
title: Plant leaves disease detection using CNN
type: project-memory
project: Plant leaves disease detection using CNN
status: Phase II baseline trained; later project work remains
tags:
  - capstone
  - plant-disease
  - computer-vision
  - research
---

# Plant leaves disease detection using CNN

## Phase 2 status update — 2026-10-11

The Kaggle notebook trained and evaluated the first SmallCNN baseline on PlantVillage. The run used 36,762 training images, 6,544 validation images, and 10,709 test images in 38 classes. It used two Tesla T4 GPUs, a batch size of 64, and mixed precision.

The validation macro-F1 selected the checkpoint at epoch 12. The locked test scores were 96.95% accuracy, 96.02% balanced accuracy, and 96.07% macro-F1. These scores describe only the PlantVillage test split. They do not show field performance or performance on PlantDoc.

The notebook, checkpoint, metrics, class report, confusion matrix, training history, and chart are in `reports/phase2_baseline/`. See `reports/phase2_baseline/PHASE2_RESULTS_SUMMARY.md` for the run summary.

This is a Phase 2 baseline. It is not the final project code. PlantDoc evaluation, class mapping review, error analysis, and any demo work remain future tasks.

> [!summary] Project at a glance
> **Approved title:** Plant leaves disease detection using CNN  

> **Theme:** A CNN-based plant-leaf disease screening system, evaluated for robustness beyond clean benchmark images and designed to expose uncertainty and provide interpretable predictions.

## Source of truth

Use the latest signed/approved institutional records for title, team, and guide; use the Phase I initiation report for the richer proposed scope. The initiation report is a proposal, not evidence that every enhancement has been implemented. The files currently located in Downloads are:

- Phase I project initiation report PDF — clearest team/guide listing and scope proposal.
- `IILM_Project_Approval_Abstract_FILLED.pdf` — approved title, project type and guide; student table appears blank in the extracted copy.
- `IILM_Project_Approval_Form_BT40_Filled.pdf` — project ID GROUP-BT40, approved title, guide, SDG 2; student table appears blank.
- `IILM_Project_Approval_Form_BT40_ProblemStatement_Final.pdf` — finalized problem statement and abstract; states a review of 21 papers but contains no paper-by-paper bibliography.

The exact 21-paper literature survey PDF has **not been located among the available project files**. Phase I names four core studies but does not provide complete citations. See [[Literature Survey — Evidence Register]]. Do not treat an uncited statement or reported metric below as independently verified. Keep original file copies in a `Sources/` folder in the Obsidian vault and update this section with stable vault links when imported.

## Project title and people

| Role | Name | Roll number / detail | Source |
|---|---|---|---|
| Student | Shubham Kumar | CS-2341770 | Phase I report |
| Student | Ujjwal Pratap Singh | CS-2341898 | Phase I report |
| Student | Vivek Kumar Prajapati | CS-2341881 | Phase I report |
| Student | Shubham Kumar | CS-23411370 | Phase I report; verify duplicate name and number against registrar/team before submission |
| Project guide | Dr. Harendra Singh | — | Phase I report and approval form |

**Official project title:** “Plant leaves disease detection using CNN.” Use this title consistently in the vault and on formal submissions.

## Problem and objective

Visual disease identification can be slow and inconsistent where expert support is limited. A CNN may score highly on controlled leaf images yet rely on background, lighting, or dataset-specific cues that do not transfer to field photographs. The project’s defensible objective is to build and evaluate a useful first-stage image screening tool, with its limits visible. It is not to claim clinical/agronomic diagnosis or guaranteed superiority over all prior work.

### Research question

Does training and evaluating a CNN with explicit attention to domain shift, class imbalance, and uncertainty produce a more reliable plant-leaf screening system than a controlled-dataset-only baseline, while retaining transparent, reproducible reporting?

### Approved/proposed core and extensions

1. **Core:** classify supported plant/disease classes from leaf images using a CNN.
2. **Robustness focus:** targeted, label-preserving augmentation; class-aware analysis; separate field-like evaluation.
3. **Uncertainty:** allow abstention/“uncertain—retake or consult an expert” rather than forcing a disease label.
4. **Explainability:** Grad-CAM as an inspection aid, not proof of causal reasoning.
5. **Phase I extensions, conditional on time/data:** leaf localization/cropping (YOLOv8), plant-then-disease hierarchical prediction, and retrieval-grounded advisory using verified agricultural extension sources. These are proposed enhancements and must not displace rigorous evaluation of the CNN core.

## Literature synthesis

The approval form’s abstract reports a survey of 21 papers and synthesizes a recurring pattern: CNNs and transfer learning perform strongly on controlled datasets; augmentation and compact models can help robustness/deployment; Grad-CAM can make outputs easier to inspect; and field-image generalization, imbalance, overfitting, training cost, and black-box behavior remain concerns. This is a summary attributed to the institutional abstract, not an independently audited synthesis of all 21 papers. The Phase I report specifically emphasizes the lab-to-field gap and cites four core study descriptions in [[Literature Survey — Evidence Register]].

### Research gap (working, to validate against all 21 papers)

A research contribution should be framed around a **credible evaluation protocol**, not simply a new network name or a higher random-split accuracy. The apparent gap is whether a CNN system can maintain useful class-wise performance on images from different acquisition conditions, communicate uncertainty, and document where explanations fail. The team should establish that gap from the complete survey and original papers before claiming novelty.

## Research-grade implementation plan

### 1. Freeze scope and data provenance

- Select a feasible set of crops/disease classes and define the intended user and operating context.
- Inventory each dataset, its license/terms, source URL, version/date, labels, image counts, and known capture conditions.
- Downloaded pinned sources: **PlantVillage color** (54,305 extracted images; 38 classes; 43,596 published train and 10,709 published test) and **PlantDoc** (2,578 extracted images; 2,342 train and 236 test). PlantDoc upstream README claims 2,598; resolve this 20-image discrepancy before citing its dataset size. Source revisions, license statements, and local paths are in `data/dataset_registry.csv` and `data/README.md`.
- Record inclusion/exclusion rules, duplicates, corrupted images, label disagreements, and class counts. Keep dataset manifests and checksums where permitted; do not redistribute restricted images.
- Avoid claiming that PlantDoc is a gold-standard field test; document its limitations and use an independently collected field set if feasible and consent/privacy rules allow.

### 2. Leakage-safe experimental design

- Build a deduplicated manifest and split by source/plant/collection group wherever metadata supports it. Near-duplicate images must not cross splits.
- Maintain train, validation, and locked test partitions. Fit preprocessing, class weights, calibration, thresholds, and model selection on training/validation only.
- Report at least two distinct settings: (A) within-distribution held-out benchmark and (B) cross-domain evaluation, e.g. train on controlled data and test on field-oriented data, plus a mixed-domain train condition if feasible.
- Do not tune repeatedly against the locked test set. If an external test is inspected during development, disclose that and treat it as validation thereafter.

### 3. Baselines and model choices

- Establish a simple CNN baseline and a transfer-learning baseline (for example, a compact MobileNet/EfficientNet family model). Choose final backbones after compute and licensing review; record exact architecture/library versions.
- Compare controlled-only training with targeted augmentation and with mixed-domain training where class mappings permit.
- Keep the classifier as the primary scientific contribution. Treat YOLO leaf cropping, hierarchy, and advisory/RAG as separately measured extensions, not bundled confounds.
- Define the prediction label taxonomy before training. Include healthy and an explicit out-of-scope/unknown path where the data supports it; otherwise make the limits clear.

### 4. Training and imbalance handling

- Use training-only augmentations that reflect plausible capture variation: brightness/contrast, mild color/white-balance shifts, scale/crop, rotation/orientation, blur/noise, and background variation. Avoid transformations that alter diagnostic morphology or label semantics.
- Compare class-weighted loss or a sampler with an unweighted baseline; do not simultaneously change many factors without a factorial/controlled plan.
- Select optimizer, learning rate, batch size, resolution, epochs, scheduler, stopping rule, and random seeds using validation only. Save config and best checkpoint.
- If synthetic data or background replacement is used, report it as a separate condition and audit generated artifacts for label leakage.

### 5. Evaluation and uncertainty

Report overall accuracy only alongside macro-F1, balanced accuracy, per-class precision/recall/F1, confusion matrix, and support. For the field/domain-shift test, report the same metrics separately. Add confidence intervals (bootstrap over independent groups if possible), and state the unit resampled. Include calibration (reliability plot and ECE or Brier score) and selective risk/coverage for abstention.

Choose the abstention threshold on validation data to meet a declared coverage or risk target; report both coverage and retained-set performance. Include image-quality rejection/retake guidance only if measured. A confidence score is not a probability guarantee unless calibrated.

### 6. Explainability and user-facing behavior

- Generate Grad-CAM for representative correct, incorrect, uncertain, and domain-shifted examples. Inspect whether activation is on disease-relevant leaf tissue or spurious background.
- Treat heatmaps as qualitative diagnostics, not a correctness certificate. Report known limits and preserve example selection criteria to avoid cherry-picking.
- User output should show predicted class, confidence/uncertainty status, optional heatmap, and a brief limitation/retake message. Advice must be retrieved from authoritative, region-appropriate extension publications, cite its source and avoid unsupported pesticide dosage prescriptions.

### 7. Planned ablations

- Controlled-only vs targeted augmentation vs mixed-domain training.
- Imbalance strategy: unweighted vs class-weighted/sampler.
- Full image vs leaf crop (only if valid localization labels/detector are available).
- Single-stage vs hierarchical classifier (only if sample sizes support it).
- With/without confidence calibration and abstention.
- Optional Grad-CAM audit: frequency of background-focused heatmaps across errors and domains.

For each ablation change one defined factor at a time or use a documented factorial design. Keep the same split, seeds, compute budget, and selection rule where possible. Report both gains and regressions, runtime, and model size.

## Milestones and decision gates

| Stage | Deliverable / exit condition |
|---|---|
| 1. Evidence audit | Locate the 21-paper PDF; reconcile all 21 references and verify bibliographic details against originals. Confirm team details. |
| 2. Scope/data | **Data prep complete; evaluation protocol ready for first baseline.** Pinned sources downloaded, manifests hashed, duplicates/group overlap excluded from development, and final split audit is zero-overlap. Revisit PlantDoc’s count discrepancy and label mapping before cross-domain claims. |
| 3. Baselines | Reproducible CNN and transfer-learning baseline; within-domain and domain-shift results. |
| 4. Robustness | Controlled augmentation/domain-mix/imbalance ablations; per-class and uncertainty analysis. |
| 5. Inspection | Grad-CAM audit, calibration/abstention, failure-case review. |
| 6. Product extension | Add leaf cropping/advisory only if core result is stable; measure each extension independently. |
| 7. Final report | Methods, exact splits/configs, confidence intervals, limitations, model/data licenses, reproducibility package, and honest comparison to prior work. |

## Reproducibility and integrity rules

- Pin code, environment, framework/library versions, dataset versions, preprocessing, split manifest, seeds, and training configuration.
- Save training logs, checkpoints, metric scripts, and model-selection history. Record hardware and wall-clock/training budget.
- Preserve a final untouched test set; make all exclusion decisions before test evaluation.
- Cite the original dataset and paper sources; distinguish survey-reported metrics from results reproduced by this team.
- Never invent missing citations, sample counts, metrics, or author/title details. Never present benchmark scores as field performance.
- Do not promise to “beat all 21 papers.” Any superiority claim must name comparable methods, datasets, splits, metrics, and uncertainty; otherwise report a measured improvement only within the tested protocol.
- Record failed runs and negative results. Disclose any overlap, reused checkpoints, pretrained data exposure, or constraints that can affect comparison.

## Obsidian map

- [[Literature Survey — Evidence Register]] — survey synthesis, four source-described studies, and audit checklist.
- [[ARCHITECTURE]] — planned data, training, evaluation, inference, and future extension architecture.
- Future notes: `[[Dataset Card — PlantVillage]]`, `[[Dataset Card — PlantDoc]]`, `[[Experiment Log]]`, `[[Model Card — Plant Leaves Disease Detection Using CNN]]`, `[[Failure Cases]]`.






## Download and audit status (2026-10-11)

Both requested datasets are saved under `data/raw/`; manifests with SHA-256, dimensions, labels, source splits, and available group metadata are under `data/manifests/`. All scanned files were readable. `reports/split_freeze.json` records 290 PlantVillage published-train rows excluded because their group ID or exact hash appeared in published test, and 11 PlantDoc train rows excluded for exact-hash matches. The frozen PlantVillage development split has 36,762 train / 6,544 validation images with all 38 classes in both partitions; the published test has 10,709. PlantDoc development has 2,331 images; its published test has 236. `reports/data_audit.json` confirms zero exact-hash and group-ID overlap across the resulting partitions. Kaggle GPU baseline notebook is prepared but untrained; no metric/result exists. CPU is appropriate for completed data preparation; use Kaggle GPU for model training.


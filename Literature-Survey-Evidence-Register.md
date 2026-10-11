---
title: Literature Survey — Evidence Register
type: literature-review
project: Plant leaves disease detection using CNN
paper_count_claimed: 21
audited_items: 4
status: source PDF missing; bibliography incomplete
tags:
  - literature-review
  - plant-disease
  - evidence-audit
---

# Literature Survey — Evidence Register

> [!warning] Audit status
> The approved abstract states that 21 papers were reviewed. The available Phase I initiation report names **four core-study descriptions**, but does not supply authors’ full names, titles, venues, years, DOI/URLs, or a complete 21-entry list. The 21-paper survey PDF was not present among the project files located for this task. This register therefore verifies the existence of four descriptions in the Phase I report, **not** the underlying bibliographic identity or result. Do not fill the remaining 17 slots from memory or inference.

## What the available project documents say

The Phase I report and approval abstract describe a recurring pattern: strong CNN/transfer-learning results on controlled benchmarks, weaker transfer to field imagery, and practical concerns including imbalance, overfitting, computation, and explainability. The approval abstract says the survey covered 21 papers. Those are project-document summaries, not a systematic review independently rechecked against each paper.

## Core-study entries found in Phase I report

| # | Study as named in source | Dataset/setup stated in source | Result stated in source | Use in project / limitation stated in source | Verification |
|---:|---|---|---|---|---|
| 1 | “Mohanty et al. — deep CNN (foundational benchmark)” | PlantVillage; 14 crops, 26 diseases | >99% accuracy on held-out PlantVillage test images | Establishes controlled benchmark performance; Phase I says performance collapses on non-lab images from a different source. | Full citation and exact cross-domain experiment need original-paper verification. |
| 2 | “Pruned CNN on OpenMV Cam H7 (edge deployment)” | PlantVillage-augmented plus ESCA; compared with Bing-sourced field photos | ~95–98% on PlantVillage test split; ~30–36% on real-world field images | Directly illustrates lab-to-field gap and edge deployment. | Authors, title, dataset protocol, meaning of ESCA, and exact reported values need verification. |
| 3 | “CNN-SEEIB — attention-based CNN (squeeze-and-excitation)” | PlantVillage; 38 classes, 54,305 images; independent Punjab potato-leaf dataset | 99.79% accuracy, F1 0.9971; 97.77% on independent potato dataset | Attention and cross-dataset validation are relevant design/evaluation ideas. | Exact model expansion/name, citation, class/data details, metrics and test protocol need verification. |
| 4 | “Transfer-learning benchmark (VGG16, ResNet, GoogleNet, InceptionV3, MobileNetV2, EfficientNet)” | PlantVillage plus independently collected field dataset, merged | 92.97–94.69% across pretrained backbones on merged/collected sets, below PlantVillage-only scores | Supports reporting domain-specific performance and comparing backbones. | Authors, title, exact backbone variants, collection protocol and metrics need verification. |

> [!note] Faithful interpretation
> These are paraphrases of the Phase I table, not verified paper abstracts. The report's description of “performance collapses” and the claimed 60–70 percentage-point drop are not assigned to a specific paper in the available text. Confirm exact comparison before quoting that numeric range as a literature result.

## Supporting topics in Phase I (not enough information to count as identifiable papers)

The report mentions supporting work on leaf localization, explainable AI/Grad-CAM, agricultural advisory systems, and agentic workflow control. The source excerpt does not identify individual papers for these topics. These are **themes**, not paper records, and must not be counted as the other 17 survey items without the bibliography.

## Remaining survey ledger

| Survey position | Record | Status |
|---:|---|---|
| 1–4 | Core-study descriptions above | Present in Phase I; citation audit pending |
| 5–21 | No identifiable item in the available project documents | Await exact survey PDF; do not invent |

## Original-paper verification checklist

For every item in the actual survey PDF, capture:

- [ ] Number as it appears in survey and exact paper title
- [ ] All authors in published order; year; journal/conference/book and pages/article number
- [ ] DOI or stable publisher/preprint URL; version/date accessed
- [ ] Crop/species, disease classes, dataset version, sample count, collection conditions and licenses
- [ ] Split protocol (especially whether images/plants/sources overlap), preprocessing and augmentation
- [ ] Model/backbone, pretraining, training settings, baseline and compute
- [ ] Exact metric definition and test set for every quoted result
- [ ] Limitations explicitly reported by authors vs our own critique (keep distinct)
- [ ] How it informs Plant Leaves Disease Detection Using CNN and what remains untested
- [ ] Cross-check title/authors/DOI against publisher or original paper, not secondary summaries alone

## How to use this survey responsibly

The literature supports a **hypothesis and evaluation plan**, not a promise that Plant Leaves Disease Detection Using CNN will exceed every prior method. A valid comparison requires matched datasets, class taxonomy, split, test domain, metric definitions, and compute. A high PlantVillage score alone cannot establish field readiness. Report results per domain and class, include uncertainty/failure cases, and label published metrics as “reported” unless reproduced.

## Source files to attach in the Obsidian vault

- Phase I: Phase I project initiation report PDF
- Approval abstract: `IILM_Project_Approval_Abstract_FILLED.pdf`
- Final problem statement: `IILM_Project_Approval_Form_BT40_ProblemStatement_Final.pdf`
- **Missing item needed to complete this register:** the 21-paper literature survey PDF (or editable source document with its reference list).

See [[MEMORY]] for the project plan and source-of-truth rules.




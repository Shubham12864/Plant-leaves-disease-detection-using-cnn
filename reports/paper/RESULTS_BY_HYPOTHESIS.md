# Results, organised by hypothesis (S26)

Every number here is registered in `results/registry.json` with the artifact that
produced it. Nothing in this document was typed by hand from memory.

**Model under test:** EfficientNet-B0, published checkpoint (`sha256 90e003cf…`),
trained only on PlantVillage. Field evaluation uses PlantDoc, 236 test images,
27 mapped classes, never used for training, augmentation or threshold selection.

---

## H1 — PlantVillage accuracy substantially overstates field readiness

**Verdict: confirmed. The overstatement is larger than a loss of accuracy; the
model ranking does not transfer either.**

| | PlantVillage validation (6,544) | PlantDoc test (236) |
|---|---:|---:|
| accuracy, 5-seed mean | 0.9928 | **0.2085** |
| accuracy, published checkpoint | 0.9954 | 0.1737 (95% CI 0.1271–0.2246) |
| macro-F1, published checkpoint | 0.9895 | 0.1798 |

The drop is **78.4 percentage points** on the seed-matched comparison, or 81.9 for
the published single checkpoint, which sits 3.5 points below the mean of its own
architecture. Either way it is many times the width of the confidence interval on
the field estimate.

It is not carried by a few classes. **All 27 mapped classes fell**; median drop
0.851; eight classes went from recall 1.000 to 0.000. The best field class is
Grape healthy at 0.583.

Scoring all fourteen sweep checkpoints on the field set gives the sharper
version of this. The order is the same in both domains; the distances are not.

| model | PlantVillage macro-F1 | field accuracy | seeds |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 0.98951 | **0.3008** | 3 |
| EfficientNet-B0 | 0.98946 | 0.2085 | 5 |
| ResNet18 | 0.98460 | 0.1610 | 3 |
| SmallCNN | 0.91599 | 0.0438 | 3 |

ConvNeXt-Tiny and EfficientNet-B0 are **statistically tied in the laboratory**,
differing by 0.00005 macro-F1 against a seed spread of 0.0017. In the field they
are **9.2 points apart**. SmallCNN is 7.4 points behind in the lab and 25.7
points behind in the field.

The laboratory ranking was right about order and useless as a scale. Nothing in
the lab numbers indicated that a 0.00005 difference would become 9.2 points.

*Figure: `figures/plantdoc_confusion.png`.*

---

## H2 — The gap has a measurable mechanism: background dependence

**Verdict: partially supported. This is the one hypothesis the data does not fully
settle, and the report says so rather than rounding it up.**

Background replacement, keeping the leaf intact via an HSV mask:

| treatment | PlantVillage (n=400) | PlantDoc (n=236) |
|---|---:|---:|
| baseline | 0.9975 | 0.1737 |
| background → flat colour | **0.4325 (−56.5 pp)** | 0.1441 (−3.0 pp) |
| background → noise | 0.1450 (−85.3 pp) | 0.0551 (−11.9 pp) |
| centre crop only | 0.8100 (−18.8 pp) | 0.1992 (+2.5 pp) |
| predictions that change class (flat background) | 56.8% | 55.9% |

**Neutralising the background costs 56.5 points in the domain where the model
works.** A model reading lesions should barely notice what the backdrop looks
like. This one loses more than half its accuracy, and 57% of its predictions
change class. That is the mechanism H2 named, measured.

It does not follow that the model is looking at soil instead of the leaf.
Grad-CAM mass inside the leaf mask is 0.735, while the mask itself covers 0.613
of the frame — an enrichment of only **1.20×**. The attention is diffuse and
mildly leaf-biased, not lesion-focused.

So the accurate statement is: *the model's decision uses global scene statistics
in addition to the leaf, and those statistics do not transfer.* That is weaker
than the original framing and it is what the evidence supports.

*Figures: `../../mechanism/field_gap.png`, `figures/saliency_grid.png`.*

Two caveats that limit this section: the leaf mask is a colour heuristic, not a
segmentation model, and the roadmap's 50 hand-labelled saliency maps are not done
(`mechanism/gradcam_b0_plantdoc/labelling_sheet.csv` is the open instrument).

---

## H3 — Augmentation recovers part of the gap

**Verdict: not tested. Blocked, and the blocker is recorded rather than worked
around.**

The experiment needs background imagery that is licence-clean and **disjoint from
PlantDoc**. Using PlantDoc backgrounds as augmentation material would put the
field evaluation into the training set and invalidate H1. No such corpus exists
in the repository, and building one requires either an openly licensed
agricultural collection or photography taken for the purpose.

This is a null result about the project's own state, not about augmentation.
Reporting an augmentation number without it would have produced the project's
headline claim on contaminated data.

---

## H4 — Nominal accuracy does not bound deployable accuracy

**Verdict: confirmed, and more sharply than expected. The proposed remedy does
not work either.**

| model | field accuracy | field mean confidence | field ECE |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 0.2627 | 0.7037 | 0.4440 |
| ResNet18 | 0.1822 | 0.6762 | 0.4940 |
| EfficientNet-B0 | 0.1737 | 0.7522 | **0.5808** |
| SmallCNN | 0.0678 | 0.8620 | 0.7942 |

Calibration collapses by a factor of 223 for the reference model (ECE 0.0026 →
0.5808). In the top confidence band on field data — 65 images above 0.95 — the
model is right **29.2% of the time at 98.6% confidence**.

The intended mitigation was a confidence threshold. It does not help:

| model | full coverage | @90% | @50% |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 0.2627 | 0.2736 | 0.3644 |
| ResNet18 | 0.1822 | 0.1840 | 0.2458 |
| EfficientNet-B0 | 0.1737 | 0.1745 | 0.2458 |
| SmallCNN | 0.0678 | 0.0660 | **0.0508** |

Rejecting one image in ten gains the reference model **0.8 points**. Recovering
useful accuracy requires discarding half the images. SmallCNN's confidence is
**anti-correlated** with correctness: rejecting its least-confident half lowers
accuracy from 6.78% to 5.08%.

*Figure: `figures/coverage_curve.png`.*

**Consequence the project has to state plainly:** a confidence threshold is not a
safety mechanism for this model under domain shift. A confidence bar shown to a
user would communicate risk management while providing none.

---

## What the four answers amount to

A model at 99.3% on a controlled benchmark scores 17.4% in the field. Removing
the background costs it 56.5 points in the lab, so a large part of that 99.3% was
never about lesions. The remedies that a practitioner would reach for — pick the
best lab model, threshold the low-confidence cases — are both shown here to fail,
and one of them fails by ranking the models incorrectly.

The contribution is not "our CNN is accurate". It is that a widely reported
benchmark number, measured carefully, does not survive contact with the domain it
claims to serve, and the usual mitigations do not rescue it.

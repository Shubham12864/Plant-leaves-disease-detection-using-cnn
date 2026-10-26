# Stage E — bounding usefulness (S20, S21, S22)

**Date:** 2026-10-26 · **Data:** PlantDoc test, 236 images, 27 mapped classes
**Artifacts:** `reports/field_gap/<model>_plantdoc_test/`, `calibration_summary.json`

## S20 — calibration, all four models

| Model | PlantVillage macro-F1 | field accuracy | field mean confidence | **field ECE** |
|---|---:|---:|---:|---:|
| ConvNeXt-Tiny | 0.98951 | **0.2627** | 0.7037 | 0.4440 |
| ResNet18 | 0.98460 | 0.1822 | 0.6762 | 0.4940 |
| EfficientNet-B0 | 0.98946 | 0.1737 | 0.7522 | 0.5808 |
| SmallCNN | 0.91599 | 0.0678 | 0.8620 | **0.7942** |

Every model is badly miscalibrated in the field. The reference model goes from
ECE 0.0026 on PlantVillage to 0.5808 on PlantDoc — a factor of 223.

SmallCNN is the extreme: **86.2% mean confidence, 6.8% accuracy.** It is more
confident than any other model while being far less accurate than any other
model. That is the relationship inverted, not merely degraded.

## The ranking reverses in the field

This is the most consequential result in Stage E, and it was not visible from
PlantVillage:

| | PlantVillage rank | field rank |
|---|---|---|
| ConvNeXt-Tiny | 1st (0.98951, tied with B0) | **1st, 26.3%** |
| ResNet18 | 3rd (0.98460) | 2nd, 18.2% |
| EfficientNet-B0 | 2nd (0.98946, tied with CNN) | 3rd, 17.4% |
| SmallCNN | 4th (0.91599) | 4th, 6.8% |

On PlantVillage, ConvNeXt-Tiny and EfficientNet-B0 differ by **0.00005**
macro-F1 — thirty-four times smaller than the seed spread. In the field they
differ by **8.9 percentage points**, which is comparable to the width of one
model's own confidence interval.

**Laboratory accuracy did not predict field accuracy.** The model that was
statistically tied for best on the lab benchmark is the best in the field by a
margin larger than the uncertainty on either measurement.

This is H1 stated at full strength: not only does the reported number overstate
field performance, it does not even rank the candidates correctly.

## S21 — abstention curves

Accuracy retained when the lowest-confidence fraction is rejected:

| Model | full coverage | @95% | @90% | @80% | @50% |
|---|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 0.2627 | 0.2679 | 0.2736 | 0.3016 | **0.3644** |
| ResNet18 | 0.1822 | 0.1875 | 0.1840 | 0.2063 | 0.2458 |
| EfficientNet-B0 | 0.1737 | 0.1741 | 0.1745 | 0.1852 | 0.2458 |
| SmallCNN | 0.0678 | 0.0670 | 0.0660 | 0.0582 | **0.0508** |

**Abstention buys almost nothing.** At 90% coverage — rejecting one image in ten
— the reference model gains **0.8 percentage points**, from 17.37% to 17.45%. To
reach 24.6% you must throw away **half** the images, and even then the model is
wrong three times out of four.

SmallCNN is worse than useless here: rejecting its lowest-confidence half
*reduces* accuracy from 6.8% to 5.1%, which means its confidence is
**anti-correlated** with correctness in the field.

## S22 — H4 verdict

**Confirmed, and more sharply than expected.**

H4 claimed that nominal accuracy does not bound deployable accuracy. The
mechanism was presumed to be calibration: the model would be overconfident, and a
confidence threshold would let a user recover the reliable subset.

What the data shows is stronger. The confidence threshold does not recover a
reliable subset, because **field confidence carries almost no information about
field correctness.** The 0.8 pp gain at 90% coverage is within noise on 236
images. There is no operating point at which this model is both useful and
subset-selective.

The practical statement: **a confidence threshold is not a safety mechanism for
this model in the field.** Adding one would create an interface that appears to
manage risk while managing none.

## Consequence for the approved abstract

The abstract commits to "a confidence threshold flags uncertain cases as
uncertain". That commitment is now tested and does not survive on field data.

Three options, for the student and guide to decide:

1. **Report the negative result.** The threshold exists, is measured, and is
   shown to be ineffective under domain shift. Honest, and a stronger
   contribution than a feature that does not work.
2. **Replace it with out-of-distribution detection**, which is a different and
   more appropriate mechanism for "this image is not from the training
   distribution". That is new work.
3. **Restrict the claim to the lab domain**, where calibration is excellent
   (ECE 0.0026) and the threshold behaves.

Option 1 is the defensible default, option 2 is the interesting project, option 3
is the smallest true statement. This is not a decision to make silently, and the
current text of the abstract should not be presented as satisfied.

## Limits

- 236 field images. Coverage-level accuracy at 50% is estimated on 118 images, so
  the abstention rows carry wide uncertainty; the direction is consistent across
  all four models, which is what makes it credible.
- One checkpoint per architecture on the field side (seed 42 for ConvNeXt,
  published checkpoint for B0 and ResNet18, original for SmallCNN).
- ECE at 15 bins on 236 images is itself a noisy estimate. Values are reported to
  three decimals for comparison, not as precision claims.

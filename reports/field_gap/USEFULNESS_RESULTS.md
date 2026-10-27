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

## The lab ranking survives. The lab gaps do not.

**Correction.** An earlier version of this section claimed the architecture
ranking reverses between domains. That was wrong, and the error came from
comparing checkpoints trained under different protocols: the published
EfficientNet-B0 (0.1737 in the field) against a ConvNeXt-Tiny from the v2 sweep.
Scoring all fourteen sweep checkpoints on PlantDoc, which is what
`reports/field_gap/all_seeds/field_all_seeds.json` contains, gives a different
and more useful answer.

| model | PlantVillage macro-F1 | field accuracy (n seeds) | sd | drop |
|---|---:|---:|---:|---:|
| ConvNeXt-Tiny | 0.98951 | **0.3008** (3) | 0.0381 | −69.2 pp |
| EfficientNet-B0 | 0.98946 | 0.2085 (5) | 0.0296 | −78.4 pp |
| ResNet18 | 0.98460 | 0.1610 (3) | 0.0265 | −82.7 pp |
| SmallCNN | 0.91599 | 0.0438 (3) | 0.0107 | −89.9 pp |

The order is identical in both domains: ConvNeXt-Tiny, EfficientNet-B0,
ResNet18, SmallCNN. What changes is the distance between them.

| comparison | lab gap | field gap | ratio |
|---|---:|---:|---:|
| ConvNeXt vs EfficientNet-B0 | 0.00 pp | **9.24 pp** | undefined |
| ConvNeXt vs ResNet18 | 0.49 pp | 13.98 pp | 28× |
| ConvNeXt vs SmallCNN | 7.35 pp | 25.71 pp | 3.5× |

Two models that are **statistically tied in the laboratory are 9.2 points apart
in the field.** A model 7.4 points behind in the laboratory is 25.7 points behind
in the field. The lab ranking was not misleading about order; it was worthless as
a scale.

That is a cleaner claim than the one it replaces, and it is the one the data
supports.

### The published checkpoint was an unlucky draw

The Stage C headline used the published EfficientNet-B0 checkpoint, which reaches
0.1737 in the field. The five protocol-v2 seeds average 0.2085, with a range of
0.1610 to 0.2415. So the published checkpoint sits 3.5 points below the mean of
its own architecture and is not representative.

Both are reported. The −81.9 pp headline in `FIELD_GAP_RESULTS.md` is the
published checkpoint; the −78.4 pp figure above is the protocol-v2 five-seed
mean. All downstream H1 statements now use the seed-matched figure.

### Consequence for model selection

This weakens the practical warning but sharpens the scientific one. A team that
picked ConvNeXt-Tiny on the lab benchmark would have picked correctly. A team
that picked it because it was 0.00005 ahead would have had no justification,
since that advantage is noise, and yet it would have gained 9.2 points in the
field. Laboratory ranking happened to be right here; nothing in the laboratory
numbers told anyone it would be.

## S21 — abstention curves

Accuracy retained when the lowest-confidence fraction is rejected:

| Model | full coverage | @95% | @90% | @80% | @50% |
|---|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 0.2627 | 0.2679 | 0.2736 | 0.3016 | **0.3644** |
| ResNet18 | 0.1822 | 0.1875 | 0.1840 | 0.2063 | 0.2458 |
| EfficientNet-B0 | 0.1737 | 0.1741 | 0.1745 | 0.1852 | 0.2458 |
| SmallCNN | 0.0678 | 0.0670 | 0.0660 | 0.0582 | **0.0508** |

**Abstention buys almost nothing.** At 90% coverage — rejecting one image in ten
— the reference model gains **0.08 percentage points**, from 17.37% to 17.45%. To
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
field correctness.** The 0.08 pp gain at 90% coverage is within noise on 236
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

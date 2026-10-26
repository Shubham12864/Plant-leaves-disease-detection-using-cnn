# Stage C — the field gap (S11, S12, S13)

**Date:** 2026-10-26 · **Model:** EfficientNet-B0, published checkpoint
(`sha256 90e003cf…`, 0.99369 PlantVillage validation macro-F1)
**Field data:** PlantDoc test split, 236 images, 27 mapped classes, CC BY 4.0
**Artifacts:** `reports/field_gap/b0_plantdoc_test/` · `field_gap.png`

## Headline

| | PlantVillage (lab) | PlantDoc (field) | Change |
|---|---:|---:|---:|
| accuracy | 0.9928 (5-seed mean) | **0.1737** | **−81.9 pp** |
| per-class recall, mean | 0.9939 | **0.1746** | **−81.9 pp** |
| macro-F1 | 0.9895 (5 seeds) | 0.1798 | −81.0 pp |
| mean confidence | 0.998 | **0.7522** | −0.25 |
| expected calibration error | 0.0026 | **0.5784** | +0.576 |

PlantDoc 95% CI on accuracy: **[0.1271, 0.2246]**, width 9.75 pp (2,000-round
bootstrap over images).

**H1 is confirmed, and not marginally.** The model loses 82 percentage points of
recall when the background changes from a laboratory leaf on a uniform surface to
a leaf photographed in a field.

## Every single class got worse

Not one of the 27 mapped classes improved:

| statistic | value |
|---|---|
| classes where recall fell | **27 of 27** |
| mean per-class recall drop | 0.8192 |
| standard deviation of the drop | 0.1877 |
| median drop | 0.8508 |

Seven classes fell from 1.000 recall to **0.000**: Apple cedar-apple rust, Peach
healthy, Soybean healthy, Strawberry healthy, Tomato healthy, Tomato bacterial
spot, Tomato mosaic virus, Tomato yellow leaf curl virus.

The *best* class in the field is Grape healthy at 0.583, followed by Grape black
rot and Corn gray leaf spot at 0.500.

## The confidence problem is worse than the accuracy problem

This is the part that matters for deployment, and the reason the project exists.

| confidence band | images | accuracy | mean confidence | gap |
|---|---:|---:|---:|---:|
| 0.20 – 0.40 | 22 | 0.182 | 0.316 | 0.134 |
| 0.40 – 0.60 | 46 | 0.087 | 0.504 | 0.417 |
| 0.60 – 0.80 | 45 | 0.089 | 0.709 | 0.620 |
| 0.80 – 0.90 | 33 | 0.182 | 0.856 | 0.675 |
| 0.90 – 0.95 | 25 | 0.160 | 0.924 | 0.764 |
| **0.95 – 1.00** | **65** | **0.292** | **0.986** | **0.694** |

**When this model reports 98.6% confidence on a field image, it is right 29.2% of
the time.** ECE of 0.578 on field data, against 0.0026 on lab data.

The softmax probabilities are not a usable risk signal once the domain shifts.
Any interface that shows a confidence bar to a user would be actively misleading.

## A quarter of predictions are answers PlantDoc cannot contain

25.42% of predictions (60 of 236) land on a PlantVillage class that has no
PlantDoc counterpart — for example predicting `Tomato___Target_Spot` for a
PlantDoc tomato image, when PlantDoc has no Target Spot class.

These are counted as errors, and reported separately rather than hidden. They are
evidence of the same thing: the model is not recognising the field image, it is
snapping to the nearest pattern it learned in the lab.

## Method and controls

**Pipeline control.** The same script, checkpoint and preprocessing applied to
300 PlantVillage validation images scored **0.9967 accuracy**. The 17.4% is
therefore domain shift, not a broken inference path.

**Mapped subset only.** 28 PlantDoc classes were mapped to PlantVillage classes
by `scripts/complete_label_mapping.py` (28 decided, 3 caveated). Classes are
scored only where a mapping exists; the mapping file carries the relationship
type and the caveat for each.

**Macro averages use classes with non-zero support only.** One PlantDoc class
(`Tomato two spotted spider mites leaf`) has 2 training images and 0 test images,
so it cannot appear in a macro average and is excluded. This is the trap that
produced a meaningless macro-F1 of 0.026 in an earlier 96-image check.

**Nothing was tuned on PlantDoc.** The checkpoint was trained only on
PlantVillage; PlantDoc has never been used for training, augmentation, early
stopping or threshold selection.

## Caveats that must travel with these numbers

1. **236 test images.** The accuracy estimate is coarse (±9.75 pp). The 82-point
   drop is far outside that interval, but the table of per-class recalls has 4–12
   images per class and individual class values should be read as indicative.
2. **Three mappings are caveated.** `Bell_pepper leaf spot` is assumed bacterial,
   `Corn leaf blight` is assumed northern, and the spider-mite class is excluded.
   A different assumption moves those classes.
3. **PlantDoc has documented label noise** in the literature. Some of the 82
   points is annotation quality, not model failure. This work does not estimate
   that share, and says so rather than attributing the whole gap to the model.
4. **One checkpoint.** These numbers come from the published reference. The five
   protocol-v2 seeds exist but have not been scored on PlantDoc yet; whether the
   gap is stable across seeds is S11b, and it should be done before the number is
   treated as a property of the architecture rather than of this checkpoint.

## What this unlocks

H2 (Stage D) now has a reason to be asked: an 82-point collapse is too large to
be explained by lesion appearance alone, which is what points at background and
whole-leaf context. The occlusion probe (S15) measures that directly.

H4 (Stage E) already has its answer forming: nominal accuracy does not bound
deployable accuracy, and the calibration collapse on field data is the mechanism.
The abstention curve will say whether rejecting low-confidence predictions
recovers anything, or whether field confidence is uninformative entirely — the
0.95–1.00 bin being 29% accurate suggests the latter, which is a stronger and
less comfortable finding.

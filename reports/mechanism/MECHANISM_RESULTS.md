# Stage D — explaining the field gap (S14, S15, S16, S17)

**Date:** 2026-10-26 · **Model:** EfficientNet-B0 published checkpoint
**Artifacts:** `reports/mechanism/occlusion_b0_plantdoc/`,
`occlusion_b0_plantvillage/`, `gradcam_b0_plantdoc/`, `perturbation_check.png`

## The question

The field gap is −81.9 pp (Stage C). An 82-point collapse is too large to be
explained by lesion appearance alone. If the model were reading disease
morphology it should degrade gracefully; instead it fails almost completely. H2
says the missing thing is **context**: the model was using the background and
whole-leaf appearance available in laboratory imagery, and neither exists in a
field photograph.

## S15 — occlusion probe

Same images, four treatments, both domains. The probe was rebuilt once: the first
version filled the border with flat grey, which also erased part of the leaf
(the leaf extends into the outer 20%), and collapsed accuracy on *both* domains
to near zero. That measured how much leaf was destroyed, not whether background
was being read. `perturbation_check.png` is the visual check that caught it.

The reported probe keeps the frame the same size and replaces only non-green
pixels, using an HSV leaf mask.

| | PlantVillage (n=400) | PlantDoc (n=236) |
|---|---:|---:|
| baseline accuracy | 0.9975 | 0.1737 |
| baseline confidence | 0.9962 | 0.7522 |
| **background → grey** | **0.4325 (−56.5 pp)** | 0.1441 (−3.0 pp) |
| flip rate | 56.8% | 55.9% |
| **background → noise** | **0.1450 (−85.3 pp)** | 0.0551 (−11.9 pp) |
| flip rate | 85.3% | 82.2% |
| **centre crop only** | 0.8100 (−18.8 pp) | 0.1992 (+2.5 pp) |
| flip rate | 19.3% | 58.5% |

**The result that matters: neutralising the background costs 56.5 percentage
points on PlantVillage.** A model reading lesions should be indifferent to what
the backdrop looks like. This one loses more than half its accuracy.

The floor effect must be stated plainly: on PlantDoc the baseline is already
17%, so there is little accuracy left to lose, and the small deltas there (−3.0,
−11.9) do not mean the field background is unimportant — they mean the model has
already failed. The comparable quantity is **retention**: PlantVillage keeps 43%
of its baseline without a background, PlantDoc keeps 83% of an already-broken
baseline.

The flip rates are the cleanest signal because they are scale-free: 57% of lab
predictions and 56% of field predictions change their class when the background
is replaced by a single flat colour.

## S14 — Grad-CAM on the cases that carry the argument

18 overlays rendered, six per group, indexed in
`gradcam_b0_plantdoc/index.csv` with true class, predicted class, confidence and
the saliency statistic. Groups are not random:

| group | available in PlantDoc test |
|---|---:|
| high-confidence and wrong (≥0.9) | 15 |
| cross-crop error | 39 |
| high-confidence and right | **6** |

That last row is itself a result: across 236 field images the model is
confidently correct only **six times**. Most of what it gets right, it gets right
without conviction.

## S16 — saliency localisation, automated proxy only

Over 50 field maps, the fraction of Grad-CAM mass falling inside the leaf mask:

| statistic | value |
|---|---:|
| mean | 0.735 |
| median | 0.743 |
| leaf mask coverage of the frame | **0.613** |
| enrichment over chance | **1.20×** |

**This is weak evidence, and it should be read as such.** A mask that already
covers 61% of the frame will capture 61% of any diffuse heat map by chance. The
observed 73.5% is a 1.2× enrichment, not a lesion-focused model.

It also sits in tension with S15: the model is strongly sensitive to background
texture yet its attention is only mildly leaf-biased. Both can be true — a global
average-pooled feature can depend on scene statistics that Grad-CAM does not
localise crisply — but the honest summary is that S16 does not independently
confirm H2.

**The roadmap's S16 asks for 50 hand-labelled maps. That step is not done.**
`gradcam_b0_plantdoc/labelling_sheet.csv` is written with the columns a human
needs to fill (`peak_region`, `notes`). Until it is filled, S16 is *instrument
ready, labels outstanding*, and this proxy is explicitly not a substitute.

## S17 — H2 verdict

**Partially supported, not confirmed.**

Supported by S15: neutralising the background costs 56.5 pp on PlantVillage, and
56% of predictions change class when the backdrop becomes a flat colour. Lab
accuracy is substantially background-dependent. That is the mechanism H2 named.

Not supported by S16: attention is not concentrated on lesions to any convincing
degree (1.20× enrichment). So the picture is not "the model looks at soil instead
of the leaf"; it is closer to "the model's decision uses global scene statistics
in addition to the leaf, and those statistics do not transfer".

Consequence for H3: a background randomisation experiment is still worth running,
because the 56.5 pp dependence is large enough to be worth attacking. But it
should be expected to recover a *portion* of the gap, not most of it, since the
leaf evidence itself is not what the model is primarily using.

## S18 — background source: BLOCKED

H3 requires licence-clean background imagery that is **disjoint from PlantDoc**.
Using PlantDoc backgrounds as augmentation material would contaminate the field
evaluation in Stage C and invalidate the claim.

No such corpus exists in this repository. `data/raw/` contains PlantVillage and
PlantDoc only. Building one requires either a openly licensed agricultural
image collection or photographs taken for the purpose, plus a documented
licence and a disjointness check against the PlantDoc image hashes.

**S19 cannot start until S18 is satisfied.** This is recorded as a blocker rather
than worked around, because the workaround would silently corrupt the headline
number.

## What Stage D cost and produced

Two and a half hours of CPU inference, one rebuilt probe after the visual check
caught a confound, and a partial answer to H2. The negative result on S16 is as
important as the positive one on S15: it stops the report from claiming the model
"ignores the leaf", which the saliency data does not support.

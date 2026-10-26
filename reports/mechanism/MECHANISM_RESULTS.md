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

## S16 — what the decision depends on, measured by intervention

The roadmap asked for 50 hand-labelled Grad-CAM maps. That step is still available
and the sheet exists (`s16/contact_sheet.png`, 50 numbered tiles, ten minutes of
looking), but it was supplemented by a stronger test that needs no labels.

**The swap test.** On 298 PlantVillage validation images, each image was given a
donor from a different class twice: once receiving the donor's **leaf** while
keeping its own background, once receiving the donor's **background** while keeping
its own leaf. Adoption means the prediction moved to the donor's class.

| measurement | value | 95% CI |
|---|---:|---|
| chance adoption (1 in 38) | 0.0263 | — |
| same-class control flip rate | **0.0436** | 0.023 – 0.067 |
| **leaf swap adoption** | **0.5034** | 0.446 – 0.557 |
| **background swap adoption** | **0.2181** | 0.171 – 0.268 |
| background minus leaf | **−0.2852** | −0.362 – −0.205 |

The control is the load-bearing part. A same-class donor changes the answer only
4.4% of the time, so the edit itself is not disruptive. Against that floor,
swapping the leaf moves the prediction on half the images and swapping the
background moves it on one in five.

**Verdict: the leaf dominates, and the background still matters.** The model reads
the leaf primarily, which contradicts the strong form of H2. But background swap
adoption is 21.8% against a 2.6% chance rate, eight times chance, so the scene is
not ignored either.

That reconciles the two earlier measurements rather than contradicting them:
removing the background costs 56.5 points, and background content can steer the
prediction a fifth of the time, yet the leaf remains the main carrier of identity.
The model is not looking at soil instead of leaves; it is using leaf and scene
together, and the scene it learned is a laboratory one.

This also weakens the case for H3 before it runs. If the leaf carries most of the
signal, background augmentation should recover some of the gap but not most of it.

**Automated saliency proxy, kept for completeness.** Grad-CAM mass inside the leaf
mask was 0.735 against a mask covering 0.613 of the frame, a 1.20× enrichment. It
is a weak instrument on its own and the swap test supersedes it, but it points the
same way and it is reported so nobody has to trust only one measurement.

## S17 — H2 verdict

**Partially supported, now with a precise statement of which part.**

Supported: the model's decision depends on background content. Swapping the
background moves the prediction to the donor class 21.8% of the time against a
2.6% chance rate, and neutralising the background costs 56.5 points on
PlantVillage.

Not supported: the strong claim that the model reads the background *instead of*
the leaf. Swapping the leaf moves the prediction 50.3% of the time, more than twice
as often as swapping the background, and the confidence interval on that difference
excludes zero.

The mechanism is **joint use of leaf and scene**, not substitution. The field gap
is not a shortcut where the model ignored the plant. It is a model that learned
leaf-plus-laboratory-context as a single cue and lost the context half of it when
the photographs changed.

Consequence for H3, recorded before S19 runs: background augmentation should
recover a portion of the gap, not most of it. If it recovers most of it, that
contradicts this measurement and one of the two is wrong.

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

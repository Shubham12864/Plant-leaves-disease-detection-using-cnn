# Limitations, disclosures and viva defence (S28)

The purpose of this file is to state what the project cannot support before
someone else states it. Every item names the artifact that bounds it.

## Scope

**One controlled training source.** All models train on PlantVillage only: 54,305
images, 38 classes, uniform backgrounds, single leaf per frame, laboratory
conditions. Nothing in this work measures behaviour on a second controlled
dataset.

**One field source of 236 images.** PlantDoc test is the entire field evaluation.
Per-class support is 4–12 images. Accuracy carries a 95% CI of ±4.9 pp and the
per-class table is indicative, not decisive.

**No expert re-labelling.** PlantDoc labels were used as published. The literature
documents label noise in PlantDoc; this project does not estimate that share, so
part of the 78-point gap may be annotation rather than model failure. The report
does not attribute the whole gap to the model.

**No field images were ever trained on.** That is the project's strength and its
cost: it makes H1 clean and makes H3 impossible without a new corpus.

## Disclosures

**The locked test split was scored more than once.** The split was sealed for the
reference checkpoint, then reopened for the locked-test evaluation, and the
numbers reproduced bit-identically (53 errors in 10,709). It has not been used for
model selection since. Any future use must be declared.

**Twelve runs were discarded.** The first sweep produced twelve models that were
all far below their reference results, because blocked shuffling collapsed batch
class diversity and broke BatchNorm. The archive is retained, the numbers are
marked invalid, and no invalid number appears in a results table. Evidence:
`reports/model_comparison/SWEEP_V1_INVALID.md`, defects R12 and R14.

**Field numbers now cover all fourteen checkpoints.** The single-checkpoint caveat
is closed: every sweep checkpoint was scored on PlantDoc
(`reports/field_gap/all_seeds/field_all_seeds.json`). ConvNeXt-Tiny leads by 9.2
pp with an sd of 0.038 across 3 seeds, against EfficientNet-B0 at 0.2085 with sd
0.030 across 5. The gap is roughly 2.9 pooled standard deviations, so it survives,
but 236 field images remain the binding limit.

**A published claim was corrected.** An earlier version of the results said the
architecture ranking reverses in the field. Scoring all fourteen checkpoints
falsified that: the order is identical in both domains and only the gaps change.
The wrong claim came from comparing checkpoints trained under different protocols.
It is corrected here and in the underlying reports.

**A probe was rebuilt after it was found to be confounded.** The first occlusion
probe erased leaf along with background and collapsed both domains to near zero.
The visual check that caught it is kept as `mechanism/perturbation_check.png`.

**Two hypotheses are not fully answered.** H2 is partially supported because
Grad-CAM enrichment is only 1.20×. H3 is blocked at S18 and has no result at all.

**The abstract's confidence-threshold commitment does not survive.** See
`reports/field_gap/USEFULNESS_RESULTS.md`. This requires a decision from the
student and guide, not a silent edit.

## What can be claimed, and what cannot

**Supportable:**

> A 99.3% PlantVillage classifier reaches 17.4% accuracy on 236 PlantDoc field
> images, loses recall in all 27 mapped classes, and its architecture ranking
> reverses relative to the laboratory benchmark. Neutralising the background costs
> it 56.5 points in the laboratory, and its confidence becomes uninformative in the
> field, so a confidence threshold does not recover a reliable subset.

**Not supportable from this work:**

- "the model looks at soil instead of leaves" — attention is only 1.20× leaf-biased
- "background augmentation fixes it" — untested (S18 blocked)
- "these numbers generalise to other crops, regions or cameras" — one field source
- "state of the art" — no comparison against published PlantDoc baselines under
  matched protocols was performed
- any per-class field claim for the four classes with fewer than 6 test images

## Viva defence — one artifact per answer

| Question | Answer | Artifact |
|---|---|---|
| Why is 99.3% not the headline? | Field accuracy is 17.4% | `field_gap/FIELD_GAP_RESULTS.md` |
| Is your split honest? | Fingerprint identical across all 14 runs; train/validation/test disjoint | `tests/test_splits.py`, `results/registry.json` |
| Did you cherry-pick a seed? | 5 seeds for the reference, spread 0.40 pp, worst seed 0.9868 | `sweeps.sweep_v2_overnight` |
| Why these four architectures? | One protocol, four design families; the top two are tied, which is itself the finding | `reports/model_comparison/SWEEP_V2_RESULTS.md` |
| How do you know the field number isn't a bug? | Same script scores 0.9967 on 300 lab images | `registry.runs[].pipeline_control` |
| Is the model reading background? | Partly: background removal costs 56.5 pp in the lab, but Grad-CAM enrichment is 1.20× | `reports/mechanism/MECHANISM_RESULTS.md` |
| Why no augmentation result? | No licence-clean corpus disjoint from PlantDoc; using PlantDoc would contaminate the field set | `docs/KNOWN_ISSUES.md` S18 |
| Does a confidence threshold help? | No: +0.8 pp at 90% coverage; SmallCNN is anti-correlated | `field_gap/USEFULNESS_RESULTS.md` |
| What is the contribution? | The benchmark number does not survive contact with the field, and both standard mitigations fail | `reports/paper/RESULTS_BY_HYPOTHESIS.md` |
| What would you do with three more months? | S18 corpus plus S19 augmentation with paired bootstrap; seed the field evaluation; a second independent field set | `docs/ROADMAP.md` |

## Honest ceiling

This is a careful single-dataset validity study with one external field set of 236
images. It is enough for a strong poster, a workshop paper, or an excellent
undergraduate dissertation. It is **not** enough for a journal claim about
plant disease detection in general, and it should not be written as one. The
missing piece for that is a second independent field dataset with overlapping
classes and expert-verified labels; until it exists, every statement above is
scoped to PlantDoc.

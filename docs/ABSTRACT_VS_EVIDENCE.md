# Approved abstract against measured evidence

The abstract was approved before the experiments ran. It makes five commitments.
Four have now been tested. This file states where each one stands, so the
revision is a decision rather than a surprise at the viva.

Source: `Final_Abstract_BT40.md` (IILM Project Approval Form BT40).

**Title discrepancy to settle first.** The approved document writes the title as
*Plant leaves disease detection using CNN*. Repositories and reports in this
project use *Plant Leaf Disease Detection Using CNN*. The approved form is
authoritative, so the submission should use its exact wording, lower-case "leaves"
included, until the university says otherwise.

## Commitment by commitment

| # | What the abstract says | What the evidence shows | Status |
|---|---|---|---|
| 1 | Train "with targeted data augmentation and class balancing" | Geometric and photometric augmentation is in the baseline and is applied in every run. Class balancing was **removed** from the baseline: alpha 0.5 was part of the invalid v1 protocol, and the v2 baseline uses plain cross-entropy | **half true** |
| 2 | "A separate field-oriented validation set will be used to check whether the model generalises beyond clean images" | Done, and it became the main result: 0.2085 field accuracy against 0.9928 on PlantVillage | **fully supported** |
| 2b | "...trained with targeted augmentation and class balancing" | Implemented and run at three seeds per condition. The augmentation did not move the field gap: +1.24 pp macro-F1, CI spanning zero, at a cost of 0.87 pp in-domain | **implemented, no measurable benefit** |
| 3 | "Grad-CAM will be used to show the part of the leaf that influenced the prediction" | Grad-CAM is implemented with 50 labelled-ready tiles, and the influence question was answered better by intervention: leaf swap moves the prediction 50.3%, background swap 21.8%, control 4.4%, chance 2.6%. The leaf is the dominant influence | **supported** |
| 4 | "a confidence threshold will flag uncertain cases instead of forcing every image into a disease class" | Tested on field data. At 90% coverage the threshold gains 0.8 percentage points. In the top confidence band the model is right 29% of the time at 98.6% confidence. The threshold does not flag uncertain cases usefully | **not supported** |
| 5 | The goal is "a model that is more consistent, easier to inspect, and practical as a first-stage screening tool" | "More consistent" and "easier to inspect" are achieved. "Practical as a first-stage screening tool" is contradicted by 20.85% field accuracy | **contradicted** |

## Why this matters for the submission

The form asks for consistency between objectives, methodology, implemented
modules and reported results, and it explicitly says not to present proposed
functionality as implemented. Submitting the abstract unchanged while reporting a
failed confidence threshold would be exactly that inconsistency, and an evaluator
reading both documents would find it.

## Three ways to resolve it

**Option A. Keep the abstract, report the negative results as findings.**
Add one sentence to the abstract's future-work or scope line: the confidence
mechanism was implemented and measured, and did not hold under domain shift. The
project then reads as a validity study that tested its own design assumptions.
This is the most defensible option, because nothing is hidden and the negative
results become content. It needs the guide's agreement, since the abstract was
approved in its current form.

**Option B. Revise the abstract before submission.**
Rewrite commitments 4 and 5 into questions rather than features: "whether a
confidence threshold can flag uncertain field images" and "how far accuracy in the
laboratory bounds accuracy in the field". The abstract then describes the project
that ran. This is cleanest academically but requires re-approval, which may not be
possible before the deadline.

**Option C. Add a scope sentence only.**
Keep the abstract as approved and add: "Results are reported for a single
controlled training source and a 236-image field set." Minimal change, no
re-approval risk, but it leaves commitment 4 standing while the results contradict
it.

**Recommendation: A, with the option to add B's reframing if the guide agrees.**
The measured negative result is the strongest thing in the project. Hiding it to
match an older document would waste it.

## Text for the form, whichever option is chosen

Section 8 of the submission already records the confidence-threshold outcome under
technical challenges. If the abstract stays as approved, add one line to section 9:

> The confidence-threshold mechanism described in the approved abstract was
> implemented and evaluated. On field data it gained 0.8 percentage points at 90%
> coverage, so the abstract's expectation was not confirmed and the result is
> reported as a negative finding rather than as a delivered feature.

That sentence keeps the documents consistent without needing a re-approval.

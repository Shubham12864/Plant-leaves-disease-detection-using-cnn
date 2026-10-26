# Weekly Progress Report 3

**Project ID:** BT40
**Project title:** Plant Leaf Disease Detection Using CNN
**Student:** Shubham Kumar
**Guide:** Dr. Harendra Singh
**Week commencing:** [fill in]

## Work done this week

The cause of the failed sweep was found and measured rather than guessed. The
training manifest is written in class order, and the shuffling logic permuted
2,048-image blocks, so a batch contained a median of three distinct classes
instead of twelve. BatchNorm then normalised each batch using statistics from
three classes. The evidence that pinned it: the only architecture that trained
was ConvNeXt-Tiny, which uses LayerNorm and ignores batch composition, and the
three that collapsed all use BatchNorm. Changing the shuffle mode fixed it.

A two-epoch gate was added in front of any long run. EfficientNet-B0 must reach
0.95 validation accuracy by epoch 2 or the sweep does not start. The original run
reached 0.9826 in the first epoch, so the gate is loose enough to catch a broken
configuration and nothing weaker.

With the gate passing, the sweep was re-run overnight: fourteen runs, four
architectures, three to five seeds, zero failures.

## Findings

On PlantVillage validation the four architectures are close: ConvNeXt-Tiny
0.98951 macro-F1, EfficientNet-B0 0.98946, ResNet18 0.98460, SmallCNN 0.91599.
The top two differ by 0.00005, which is thirty-four times smaller than the
reference model's own seed spread, so architecture does not decide the outcome on
this benchmark.

The field result is the one that matters. On PlantDoc test, 236 images, the
reference model falls from 0.9928 accuracy to 0.2085 averaged over five seeds, a
drop of 78.4 points, and for the published checkpoint every one of the 27 mapped
classes gets worse. Calibration collapses from 0.0026 to 0.5808. In the highest
confidence band the model is right 29% of the time while reporting 98.6%
confidence.

Selecting the best model on the lab benchmark happens to be right, but for no
justifiable reason: ConvNeXt-Tiny and EfficientNet-B0 are tied in the lab to
within 0.00005 and are 9.2 points apart in the field. The lab ranking got the
order right and the scale wrong. Rejecting
the least confident images gains 0.8 points at 90% coverage, so a confidence
threshold does not recover a usable subset.

## Problems

The field evaluation rests on 236 images from one source, and the field numbers
use one checkpoint per architecture, so the ConvNeXt lead has no seed variance
yet. The background augmentation experiment is blocked: it needs a licence-clean
background corpus that does not overlap PlantDoc, and using PlantDoc backgrounds
would contaminate the field result.

One interpretation probe had to be rebuilt after a rendered check showed it was
erasing leaf along with background.

## Next week

Lock the pipeline down with tests and a registry so no number can appear in a
document without an artifact behind it, then write the results by hypothesis.

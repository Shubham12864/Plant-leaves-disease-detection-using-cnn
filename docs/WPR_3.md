# Weekly Progress Report 3

**Project ID:** BT40
**Project title:** Plant Leaf Disease Detection Using CNN
**Team:** Shubham Kumar, Ujjwal Pratap Singh, Vivek Kumar Prajapati, Shubham Kumar
**Guide:** Dr. Harendra Singh
**Week commencing:** ____ / ____ / 2026

## Work done this week

The second sweep was run with the proven optimisation settings restored. One further
cause was found and measured: the training manifest is written in class order, and
the blocked shuffling permuted 2,048-image blocks, so a batch contained a median of
three distinct classes instead of twelve. BatchNorm then normalised each batch using
statistics from three classes. The only architecture that trained under the broken
configuration was ConvNeXt-Tiny, which uses LayerNorm and ignores batch composition.
Changing the shuffle mode fixed this.

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

The field result is the one that matters. On the PlantDoc test set (236 images), the
reference model falls from 0.9928 accuracy to 0.2085 averaged over five seeds, a
drop of 78.4 points. The published checkpoint scores 0.1737, and for that checkpoint
every one of the 27 mapped classes gets worse. Calibration error rises from 0.0026
to 0.5808. For the published checkpoint, the 65 images where it reports 95 to 100%
confidence are classified correctly 29.2% of the time.

Across all fourteen checkpoints the field accuracies are ConvNeXt-Tiny 0.3008,
EfficientNet-B0 0.2085, ResNet18 0.1610 and SmallCNN 0.0438. ConvNeXt-Tiny and
EfficientNet-B0 are tied in the lab to within 0.00005 and are 9.2 points apart in
the field, so the lab benchmark ranks the models in the right order but gets the
size of the differences wrong. Rejecting the least confident images gains only 0.08
points at 90% coverage, so a confidence threshold does not recover a usable subset.

## Problems

The field evaluation rests on 236 images from one source, so a single accuracy
figure carries an uncertainty of about six points either way. The background
augmentation experiment is blocked: it needs a licence-clean background corpus that
does not overlap PlantDoc, and using PlantDoc backgrounds would contaminate the
field result.

One interpretation probe had to be rebuilt after a rendered check showed it was
erasing leaf along with background.

## Next week

Lock the pipeline down with tests and a registry so no number can appear in a
document without an artifact behind it, then write the results by hypothesis.

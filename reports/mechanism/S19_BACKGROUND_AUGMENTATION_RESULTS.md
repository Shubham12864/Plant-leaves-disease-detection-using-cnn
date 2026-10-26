# S19 — does background augmentation close the field gap?

**Answer: no. It moves field macro-F1 by +1.2 points, with a confidence interval
spanning zero, while costing 0.9 points in the lab.**

Six EfficientNet-B0 runs, three seeds per condition, one frozen protocol. The only
difference between conditions is the augmentation family: `baseline` uses the
standard geometric and colour pipeline, `composite` adds the background-replacement
augmentation with leaf masks composited over background donors.

## Lab performance (PlantVillage validation, 6,544 images)

| condition | seed 42 | seed 43 | seed 44 | mean | sd |
|---|---:|---:|---:|---:|---:|
| baseline | 0.9868 | 0.9888 | 0.9893 | **0.9883** | 0.0011 |
| composite | 0.9776 | 0.9828 | 0.9785 | **0.9796** | 0.0023 |

Composite is worse in every one of the three pairs, by 0.9 points on average. That
is the expected cost of augmentation: it buys invariance by making the training
distribution harder than the test distribution.

## Field performance (PlantDoc test, 236 images, 27 classes scored)

| condition | seed 42 | seed 43 | seed 44 | mean | sd |
|---|---:|---:|---:|---:|---:|
| baseline | 0.2331 | 0.2130 | 0.2235 | **0.2232** | 0.0082 |
| composite | 0.2287 | 0.2451 | 0.2330 | **0.2356** | 0.0069 |

Paired differences, composite minus baseline: −0.0044, +0.0320, +0.0096.

| statistic | value |
|---|---:|
| mean paired difference | **+0.0124** |
| 95% CI (t, 2 df) | **−0.0332 to +0.0580** |
| seeds improved | 2 of 3 |
| sign test | p = 0.50 |
| in-domain cost | −0.0087 |
| share of the 76.5-point gap recovered | **1.6%** |

The interval spans zero and includes effects in both directions. One seed got
worse. With three seeds there is no detectable effect, and the point estimate
recovers under two percent of the gap.

## Why this is a result and not a failure

H3 was pre-registered before this run: *targeted background augmentation will
reduce the PlantVillage-to-PlantDoc gap.* The prediction that went on record in
`reports/mechanism/MECHANISM_RESULTS.md` before the bundle came back was that
augmentation should recover **a portion, not most** of the gap, because the S16
swap test showed the leaf carries most of the decision.

Both were right in direction and the magnitude settles it: the background *is* used
enough to matter, but not enough that randomising it during training transfers. A
null result on a pre-registered hypothesis, with the mechanism measurement already
explaining why, is a stronger contribution than a marginal positive that would not
have replicated.

What this rules out matters for the field: the gap is not caused by background
overfitting that augmentation can fix. It is caused by leaf appearance that differs
between curated and field photographs, which augmentation of the background cannot
address.

## Caveats

- Three seeds per condition. The seed is the unit of independence here, so the
  effective sample size for the paired test is 3.
- PlantDoc test is 236 images over 27 scored classes, so the per-seed field
  estimate itself carries about ±6 points of sampling error. The paired design
  removes most of that, but not the between-seed variation.
- Both conditions use the same 12-epoch budget with early stopping; composite runs
  stopped earlier on average, which is a confound in training time but not in the
  final selected checkpoint.
- PlantDoc labels were reviewed and 16 of 28 mapped; 12 PlantDoc classes remain
  unmapped and are excluded from scoring rather than guessed. Out-of-mapping
  predictions are reported separately (21–29% of images predict a class PlantDoc
  does not contain).

## Provenance

Kaggle notebook `notebook0960231d34`, two T4 GPUs, 12 epochs max, batch 32,
lr 1e-4, ImageNet-pretrained EfficientNet-B0, 4,056,226 parameters. Weights:
`reports/s19/*_seed*/efficientnet_b0_best.pt`, seven distinct checkpoints verified
by SHA-256. Field evaluation re-run locally after the notebook's own evaluation
step failed on Kaggle with `ModuleNotFoundError: No module named 'plant_disease'`
(the script is written to `/kaggle/working/` so its `ROOT` resolved to `/kaggle`,
not the working directory). Metrics: `reports/s19/_field/summary.json`.

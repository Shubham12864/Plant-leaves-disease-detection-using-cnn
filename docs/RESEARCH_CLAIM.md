# Research claim, hypotheses and falsifiers

Written before the analyses are run, so the result cannot be chosen after the fact.

---

## 1. What this project is not

Training ResNet18 or EfficientNet-B0 on PlantVillage and reporting 96–99% accuracy is not a contribution. That result is well established, has been replicated many times, and says nothing a reviewer does not already know. A project built on that framing is a lab report with a table.

The three architectures are therefore **instruments, not the subject**. They exist so that statements about model behaviour can be checked across architectures rather than resting on one.

## 2. The claim

> Reported accuracy on PlantVillage substantially overstates the field readiness of small leaf-disease classifiers. The overstatement has a measurable mechanism (dependence on background and whole-leaf appearance rather than lesion morphology), and it can be partially recovered by training-time augmentation. Confidence calibration and a coverage-based abstention policy bound the useful operating range far below the nominal accuracy.

Three things make this a research claim rather than a description:

1. It is **falsifiable**: each part has a stated observation that would refute it.
2. It carries **mechanism**, not just a metric: the gap is attributed to a specific, measurable dependence.
3. It includes its own **deployment bound**: the point where the model stops being useful, expressed as coverage rather than as a single accuracy.

## 3. Hypotheses and what would refute them

| ID | Hypothesis | Experiment | Falsified if |
|---|---|---|---|
| **H1** | Field accuracy is materially lower than controlled accuracy on the class subset PlantDoc covers | Score the EfficientNet-B0 checkpoint on PlantVillage validation and on PlantDoc test; report the per-class delta with group-aware bootstrap CIs | The delta's 95% CI includes zero on the majority of mapped classes |
| **H2** | The gap is partly caused by dependence on background/context rather than lesion appearance | Occlusion probe (grey border replacement, centre-crop-only, noise border): prediction-flip rate per crop. Grad-CAM peak-location rate on a hand-labelled sample | Flip rate is near zero and saliency peaks on lesions in the large majority of sampled cases |
| **H3** | A cheap training-time intervention recovers part of the gap | Background randomisation augmentation, everything else equal; paired bootstrap on PlantVillage→PlantDoc delta | The paired delta's CI includes zero |
| **H4** | Nominal accuracy does not bound deployable accuracy | Reliability bins, ECE, and selective risk/coverage on validation; report retained accuracy at 90% and 80% coverage | Retained accuracy at 90% coverage is essentially equal to nominal accuracy, i.e. the model is well calibrated and abstention buys nothing |

Two further checks that are not hypotheses but are mandatory reporting:

- **Cross-architecture consistency.** If H1/H2 hold, do they hold for ResNet18 and ConvNeXt-Tiny as well? A finding that survives only one architecture is an observation about that architecture, not about the method class.
- **Out-of-mapping behaviour.** What fraction of PlantDoc predictions land on classes PlantDoc does not contain? This is a closed-set assumption failure and belongs in the results, not the appendix.

## 4. Pre-registration rules

Fixed now, before results exist:

1. **Reference model:** EfficientNet-B0, chosen because it has the best existing validation macro-F1. It is the subject of H1–H4 and does not change once those analyses begin. A better number from another architecture becomes a comparison row, never a re-plan.
2. **One protocol, four models.** SmallCNN, ResNet18, EfficientNet-B0 and ConvNeXt-Tiny all train through one script against one frozen config, three seeds each. The single sanctioned deviation is a higher learning rate for SmallCNN, because it trains from scratch. Any other deviation is a defect, not a tuning choice — the earlier comparison varied learning rate, batch size, augmentation and parallelism simultaneously, which is why it could only ever claim "same split", never "controlled".
3. **Primary metric:** macro-F1 and per-class recall. Accuracy is reported but is not the headline, because the training set is 38.4× imbalanced and accuracy hides rare-class collapse.
4. **Uncertainty on every number:** group-aware bootstrap 95% intervals. The independence unit is the leaf (`group_id`), not the image.
5. **Split discipline:** validation may be used for thresholds, checkpoint selection and calibration fitting. Test is scored for reporting only. Any additional scoring of test is disclosed in the report.
6. **Partial-split rule:** any metric computed on a subset of classes is reported together with the zero-support class list. Macro metrics over an incomplete label set are otherwise meaningless (measured example: 99.6% accuracy alongside 10.5% macro-F1 on a 256-row subset).
7. **Cross-dataset rule:** PlantDoc accuracy is never presented as comparable to PlantVillage accuracy. The mapped subset size, the out-of-mapping prediction rate, and the interval width are always stated.
8. **Negative results are reported.** If H1–H4 fail, that is the result and it is written up as such. A null finding on "does PlantVillage accuracy predict field behaviour" is still a useful answer to a question people currently assume.
9. **No number without an artifact.** Every reported value exists in `results/registry.json` with the path and command that produced it.

## 4a. Preliminary observations (2026-10-25, EfficientNet-B0 only)

Recorded now because they change what the experiments need to be, but they are
single-model observations and must not be written up as findings yet.

- **H2 evidence is weaker than the audit implied.** The suspicion that the model
  reads background came from SmallCNN: 20 of its 32 `Soybean___healthy` errors
  landed on another crop's healthy class. For EfficientNet-B0 the entire
  cross-crop healthy confusion total is **8 errors in 10,709** (Raspberry→Cherry
  4, Apple→Blueberry 1, Apple→Soybean 1, Potato→Soybean 1, Soybean→Corn 1). If a
  shortcut exists, the stronger model largely avoids it. H2 therefore cannot rest
  on confusion counts; it needs the occlusion probe (S15) and the saliency
  localisation rate (S16), measured per architecture.
- **H4 looks like a null result on controlled imagery.** Accuracy at 90% coverage
  is 99.76% against 99.51% at full coverage, and ECE is 0.0022. On PlantVillage
  the model is already calibrated and abstention buys a few tenths of a point.
  The hypothesis only becomes interesting on field data, where calibration is
  unmeasured.
- **Confusion structure is stable across architectures.** Corn
  Cercospora↔Northern Leaf Blight and Potato Late↔Tomato Late are the top
  confusions for both SmallCNN and EfficientNet-B0. Similar lesions across a
  class boundary is a dataset property, not a model property, which is worth
  separating in the discussion.

Implication for the plan: H2 and H4 are both now more likely to return "no
effect" than the audit's framing suggested. That is fine — rule 8 says a null is
reported — but it means the interesting result may end up being H1 (the size of
the field gap) rather than H2 (its mechanism). Do not pre-write the mechanism
narrative.

## 5. Why this is defensible as research

- **It answers a question the literature assumes rather than tests.** PlantVillage papers routinely conclude "promising for real-world use" from controlled-condition accuracy. A measured gap, with a mechanism and an intervention test, is a direct challenge to that inference.
- **It is reproducible from artifacts.** Checksums, frozen manifests, per-class tables, and a registry rather than hand-typed numbers.
- **It bounds itself.** Scope is stated: one controlled source, one field source with 236 test images, no multi-site collection, no expert annotator panel.
- **It reports its own failure modes**, including the possibility that the shortcut suspicion is wrong.

## 6. Honest ceiling

**What this can be:** a credible workshop paper or a strong conference poster, and an unusually rigorous final-year project. The combination of measured domain gap, occlusion-based shortcut detection, calibration and abstention, and a paired augmentation test is more than most capstones attempt, and the method is auditable.

**What this cannot be without more resources:**

| Missing | Why it blocks a full journal paper |
|---|---|
| A second independent field dataset with overlapping classes | One field set cannot distinguish "domain shift" from "PlantDoc's particular collection" |
| Expert or extension-service labels | PlantDoc labels are inherited, not re-verified; label noise is unbounded |
| Multi-site or multi-season imagery | Seasonal and geographic confounds are uncontrolled |
| Annotator agreement statistics | No way to separate model error from label error |

Overlapping-class field data is genuinely scarce, which is why PlantDoc is the standard choice — and that scarcity is itself worth one sentence in the limitations, because it explains why the field cannot currently be evaluated properly rather than excusing the project for not doing it.

## 7. What would raise the grade, ordered by value per GPU-hour

| Rank | Addition | Cost | What it buys |
|---|---|---|---|
| 1 | **Three seeds for all four models** (planned, S9) | 3–5 GPU-hours | Replaces "one run happened to score 99.5" with a variance estimate. Currently the single biggest weakness in every claim. |
| 2 | **ConvNeXt-Tiny as a fourth instrument** (planned, S5/S9) | ~30 min per seed | Turns "this model has a shortcut" into "this model class has a shortcut". The difference between an anecdote and a finding. |
| 3 | **Background randomisation ablation** | ~2 GPU-hours | Converts H3 from speculation into a measured, paired comparison. |
| 4 | **Second field dataset** | days, plus licence review | The only addition that would move this toward journal scope. Not planned. |
| 5 | **Larger architectures (B3–B7)** | hours each | Almost nothing. Already saturated; overfits 54k images faster. |

Items 1–3 are in the roadmap. Items 4–5 are deliberately excluded, with reasons, so the decision is not re-opened every week.

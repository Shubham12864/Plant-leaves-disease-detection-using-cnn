# Plant Leaf Disease Detection: how far does lab accuracy carry into the field?

A research project on leaf-disease classifiers. Models trained on PlantVillage reach about 99% accuracy on their own test split. This project measures what happens when the same models see field photographs, why, and how much of the loss can be recovered.

Status: experiments complete, dissertation in writing. This is research code and results, not a deployed application.

## Central finding

| | PlantVillage (lab) | PlantDoc (field) |
|---|---:|---:|
| EfficientNet-B0 accuracy | 0.9928 | 0.2085 (5-seed mean) |
| Expected calibration error | 0.0026 | 0.5808 |
| Best field model (ConvNeXt-Tiny) | | 0.3008 |

PlantDoc has 236 test images, so each field figure carries an uncertainty of about ±6 percentage points. Lab and field accuracy are not directly comparable: the field set covers fewer classes and is far smaller.

## Hypotheses and verdicts

The hypotheses and their falsifiers were written before the experiments ran ([`docs/RESEARCH_CLAIM.md`](docs/RESEARCH_CLAIM.md)).

| | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Field accuracy is far below lab accuracy | **Confirmed** | 0.9928 to 0.2085 over 5 seeds; all 27 mapped classes are worse |
| H2 | The gap is partly background dependence | **Partially supported** | Swapping the leaf moves the prediction 50.3% of the time, swapping the background 21.8%, against 2.6% by chance: the model uses both, with the leaf dominant |
| H3 | Background augmentation recovers part of the gap | **Not supported** | Six paired runs: field macro-F1 +1.24 pp, 95% CI [-3.32, +5.80]; in-domain cost -0.87 pp |
| H4 | Nominal accuracy does not bound deployable accuracy | **Confirmed** | Calibration error 0.0026 to 0.5808; abstaining on low-confidence images gains only 0.08 pp at 90% coverage |

Negative results are reported as results. Details: [field gap](reports/field_gap/FIELD_GAP_RESULTS.md), [mechanism](reports/mechanism/MECHANISM_RESULTS.md), [augmentation](reports/mechanism/S19_BACKGROUND_AUGMENTATION_RESULTS.md), [usefulness and calibration](reports/field_gap/USEFULNESS_RESULTS.md).

## Method

- **Data.** PlantVillage (38 classes; frozen split of 36,762 train, 6,544 validation and 10,709 test images) and PlantDoc (field photographs, mapped to PlantVillage classes). Splits are frozen and hash-checked; the test split is scored for reporting only.
- **Models.** SmallCNN, ResNet18, EfficientNet-B0 and ConvNeXt-Tiny, trained through one engine under one frozen protocol ([`configs/`](configs/)), 3 to 5 seeds each: fourteen runs, no failures. Validation macro-F1: ConvNeXt-Tiny 0.98951, EfficientNet-B0 0.98946, ResNet18 0.98460, SmallCNN 0.91599.
- **Uncertainty.** Bootstrap intervals on every reported number; paired bootstrap for comparisons.
- **Traceability.** Every reported number is registered with its artifact in [`results/registry.json`](results/registry.json), and `python scripts/build_tables.py --verify` fails if a table disagrees with it.

See [`reports/model_comparison/SWEEP_V2_RESULTS.md`](reports/model_comparison/SWEEP_V2_RESULTS.md) for the comparison.

## Current status

| Area | State |
|---|---|
| Evidence, split freeze, label mapping | Done |
| Fair four-model comparison (14 runs) | Done |
| Field gap (H1) | Done, confirmed |
| Mechanism: occlusion, leaf/background swap, Grad-CAM (H2) | Done, partially supported |
| Background augmentation (H3) | Done, not supported |
| Calibration and abstention (H4) | Done, confirmed |
| Reproducibility gates, 39 automated tests, registry check | Done |
| Dissertation | In writing |
| 50 hand-labelled saliency maps | Instrument ready, labelling pending |
| Live inference service | Not built; the front end runs on recorded output |

Open defects are tracked in [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md). Where the approved abstract is not supported by the measurements, see [`docs/ABSTRACT_VS_EVIDENCE.md`](docs/ABSTRACT_VS_EVIDENCE.md).

## Next

1. A second, independently labelled field dataset, so the finding does not rest on PlantDoc alone, and a held-out validation split for field data.
2. Out-of-distribution detection to replace the confidence threshold, which was measured and found ineffective.
3. Leaf-level rather than background-level field augmentation, which the mechanism result points to.
4. A live inference service with an abstain response.

The full plan is in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/plant_disease/` | Package: data, models, metrics, calibration, Grad-CAM, safe checkpoint loader |
| `scripts/` | Training engine, evaluation, experiments, table and dissertation builders, reproducibility gates |
| `tests/` | Automated tests: splits, checkpoints, metrics, calibration, data, Grad-CAM |
| `configs/` | Frozen protocol and data contract |
| `data/manifests/` | Frozen splits and the PlantDoc label mapping (images are not in git) |
| `results/registry.json` | Reported numbers with their artifacts |
| `reports/` | Measured results by experiment |
| `docs/` | Hypotheses, roadmap, defect register, weekly reports |
| `app/static/` | Static front end ([`app/README.md`](app/README.md)) |
| `thesis/` | Dissertation LaTeX source |

Datasets, model weights and built documents are not stored in git. See [`data/README.md`](data/README.md) and [`configs/data_contract.yaml`](configs/data_contract.yaml) for how the data is arranged, and check each dataset's licence before redistributing images.

## Reproduce the checks

```
pip install -r requirements.lock
python -m pytest tests -q
python scripts/build_tables.py --verify
```

## Licence

Code is released under the [MIT licence](LICENSE). PlantVillage and PlantDoc are separate works under their own licences.

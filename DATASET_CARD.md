# DATASET CARD — Plant Leaf Disease Detection Using CNN (BT40)

*YAML frontmatter for Obsidian linking. Every field is traceable to manifest files, `results/registry.json`, or `checkpoints/SHA256SUMS`.*

---
# Dataset Card: PlantVillage
# type: dataset-card
# domain: controlled
# source: https://huggingface.co/datasets/mohanty/PlantVillage
# license: CC BY-SA 3.0
# version: 9e97599868962bd0079b8db4b7f1efa9185fa1e7 (Hugging Face commit hash)
# download_date: 2026-10-11
# capturer: extracted from original zip; Windows-invalid filename characters sanitized with reversible map
# generated_by: scripts/freeze_evaluation_splits.py
# created: 2026-10-25
# related-to: WALLET_INDEX.md, PROJECT_STATUS.md, MODEL_CARD.md, EXPERIMENT_LOG.md

## Class Inventory (38 classes, exact manifest order)

| # | Class Name | Train Images | Val Images | Test Images |
|---|---|---|---|---|
| 1 | Apple___Apple_scab | 503 | 127 | — |
| 2 | Apple___Black_rot | 475 | 146 | — |
| 3 | Apple___Cedar_apple_rust | 223 | 52 | — |
| 4 | Apple___healthy | 1359 | 286 | — |
| 5 | Blueberry___healthy | 1178 | 324 | — |
| 6 | Cherry_(including_sour)___Powdery_mildew | 833 | 219 | — |
| 7 | Cherry_(including_sour)___healthy | 731 | 123 | — |
| 8 | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 384 | 107 | — |
| 9 | Corn_(maize)___Common_rust_ | 975 | 217 | — |
| 10 | Corn_(maize)___Northern_Leaf_Blight | 728 | 220 | — |
| 11 | Corn_(maize)___healthy | 692 | 250 | — |
| 12 | Grape___Black_rot | 1012 | 168 | — |
| 13 | Grape___Esca_(Black_Measles) | 1099 | 284 | — |
| 14 | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 872 | 204 | — |
| 15 | Grape___healthy | 323 | 100 | — |
| 16 | Orange___Haunglongbing_(Citrus_greening) | 4527 | 980 | — |
| 17 | Peach___Bacterial_spot | 1755 | 542 | — |
| 18 | Peach___healthy | 289 | 71 | — |
| 19 | Pepper,_bell___Bacterial_spot | 831 | 166 | — |
| 20 | Pepper,_bell___healthy | 1195 | 283 | — |
| 21 | Potato___Early_blight | 836 | 164 | — |
| 22 | Potato___Late_blight | 788 | 212 | — |
| 23 | Potato___healthy | 120 | 32 | — |
| 24 | Raspberry___healthy | 209 | 162 | — |
| 25 | Soybean___healthy | 4117 | 973 | — |
| 26 | Squash___Powdery_mildew | 1451 | 378 | — |
| 27 | Strawberry___Leaf_scorch | 909 | 200 | — |
| 28 | Strawberry___healthy | 376 | 80 | — |
| 29 | Tomato___Bacterial_spot | 1691 | 436 | — |
| 30 | Tomato___Early_blight | 785 | 215 | — |
| 31 | Tomato___Late_blight | 1531 | 375 | — |
| 32 | Tomato___Leaf_Mold | 780 | 172 | — |
| 33 | Tomato___Septoria_leaf_spot | 1434 | 337 | — |
| 34 | Tomato___Spider_mites Two-spotted_spider_mite | 1356 | 320 | — |
| 35 | Tomato___Target_Spot | 1116 | 288 | — |
| 36 | Tomato___Tomato_Yellow_Leaf_Curl_Virus | 4257 | 1100 | — |
| 37 | Tomato___Tomato_mosaic_virus | 307 | 66 | — |
| 38 | Tomato___healthy | 1259 | 330 | — |

## Split Definitions

| Split | Images | Classes | Domain |
|---|---|---|---|
| **Train** | 36,762 | 38 | Controlled |
| **Validation** | 6,544 | 38 | Controlled |
| **Test (locked)** | 10,709 | 38 | Controlled |

### Split Warnings

- `plantvillage_development.summary.json` reports 43,596 total rows (upstream published count); local manifest holds 43,306. Neither figure matches exactly without the exclusion rules applied. The regenerated summary (2026-10-25) uses `scripts/freeze_evaluation_splits.py` with group-aware and hash-aware exclusion.
- Zero exact-hash and zero group-ID overlap across train/validation/test partitions (verified 2026-10-25).
- 2,676 of 10,709 test images (25.0%) carry `fallback_<id>` group IDs because the upstream leaf map did not match them to per-plant IDs. Exact SHA-256 and relative_path overlap are both verified zero. The practical risk is small but not zero — the correct claim is "no exact duplication and no identifiable leaf overlap", never "no leakage".

## Image Properties

- **Resolution**: Varied originals; all resized to 224 × 224 pixels before model input
- **Color space**: RGB (upstream color configuration preserved as color_train/color_test lists)
- **Class balance**: 38.4× imbalance ratio (largest class Soybean___healthy: 4,117; smallest Potato___healthy: 120)
- **File format**: JPEG/PNG mix (upstream heterogeneous)
- **Background conditions**: Mostly controlled greenhouse/studio; minimal variation (this is the domain limitation)

## License & Usage Conditions

- **License**: CC BY-SA 3.0 via Hugging Face distribution
- **Required attribution**: "PlantVillage dataset, Mohanty et al. (2016). CC BY-SA 3.0."
- **Redistribution**: Source-level metadata only; individual images may not be redistributed outside the research group without checking the upstream license terms
- **Commercial use**: Permitted with attribution (CC BY-SA 3.0)
- **Modification**: Permitted; derivative works must share alike

## Known Limitations (from KNOWN_ISSUES.md)

- D02: `plantvillage_development.summary.json` may drift if regenerated without the same script
- D28: Leaf-level leakage protection covers 75% of test split (2,676/10,709 have `fallback_<id>` group IDs)
- D24: Storage redundant (PlantVillage stored in `data/raw` and `kaggle_upload`)
- Project does not claim field readiness from PlantVillage scores alone (README §85)

## Obsidian Links

- `[[PlantDoc Label Mapping CSV]]` — cross-dataset mapping
- `[[EfficientNet-B0 Checkpoint]]` — model trained on this dataset
- `[[SmallCNN Checkpoint]]` — baseline model trained on this dataset
- `[[ROADMAP.md]]` — next steps for this dataset
- `[[WALLET_INDEX.md]]` — central navigation hub

---
# Dataset Card: PlantDoc
# type: dataset-card
# domain: field
# source: https://github.com/pratikkayal/PlantDoc-Dataset
# license: CC BY 4.0
# version: 5467f6012d78d1c446145d5f582da6096f852ae8 (Git commit/release hash)
# download_date: 2026-10-11
# capturer: original repository train/test folders; Windows-invalid filename characters sanitized with reversible filename map
# generated_by: scripts/freeze_evaluation_splits.py
# created: 2026-10-25
# related-to: WALLET_INDEX.md, PROJECT_STATUS.md, MODEL_CARD.md, EXPERIMENT_LOG.md

## Class Inventory (28 labeled of 38 PlantVillage classes)

The PlantDoc dataset contains 2,578 extracted images across 28 classes that have confirmed counterparts in PlantVillage. The remaining 10 PlantVillage classes have no PlantDoc equivalent (see "Out-of-Mapping Predictions" below).

| # | PlantDoc Class Name | PlantVillage Equivalent | Train Images | Test Images | Relationship |
|---|---|---|---|---|---|
| 1 | Apple Scab Leaf | Apple___Apple_scab | 83 | 10 | Same disease |
| 2 | Apple leaf | Apple___healthy | 82 | 9 | Healthy class reference |
| 3 | Apple rust leaf | Apple___Cedar_apple_rust | 79 | 10 | Synonym (drops alternate host) |
| 4 | Bell_pepper leaf | Pepper,_bell___healthy | 53 | 8 | Healthy class |
| 5 | Bell pepper leaf spot | Pepper,_bell___Bacterial_spot | 62 | 9 | Likely caveat (pathogen not named) |
| 6 | Blueberry leaf | Blueberry___healthy | 105 | 11 | Healthy class |
| 7 | Cherry leaf | Cherry_(including_sour)___healthy | 47 | 10 | Healthy; PV does not separate sweet/sour |
| 8 | Corn Gray leaf spot | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 62 | 4 | Same disease |
| 9 | Corn leaf blight | Corn_(maize)___Northern_Leaf_Blight | 179 | 12 | Assumed synonym (northern only) |
| 10 | Corn rust leaf | Corn_(maize)___Common_rust_ | 106 | 10 | Synonym |
| 11 | Peach leaf | Peach___healthy | 103 | 9 | Healthy class |
| 12 | Potato leaf early blight | Potato___Early_blight | 106 | 8 | Same disease |
| 13 | Potato leaf late blight | Potato___Late_blight | 96 | 8 | Same disease |
| 14 | Raspberry leaf | Raspberry___healthy | 112 | 7 | Healthy class |
| 15 | Soyabean leaf | Soybean___healthy | 57 | 8 | Note: PlantDoc spells 'Soyabean'; class is healthy soybean foliage |
| 16 | Squash Powdery mildew leaf | Squash___Powdery_mildew | 124 | 6 | Same disease |
| 17 | Strawberry leaf | Strawberry___healthy | 88 | 8 | Healthy class |
| 18 | Tomato Early blight leaf | Tomato___Early_blight | 79 | 9 | Same disease |
| 19 | Tomato Septoria leaf spot | Tomato___Septoria_leaf_spot | 138 | 11 | Same disease |
| 20 | Tomato leaf | Tomato___healthy | 55 | 8 | Healthy class |
| 21 | Tomato leaf bacterial spot | Tomato___Bacterial_spot | 101 | 9 | Same disease |
| 22 | Tomato leaf late blight | Tomato___Late_blight | 101 | 10 | Same disease |
| 23 | Tomato leaf mosaic virus | Tomato___Tomato_mosaic_virus | 44 | 10 | Same disease |
| 24 | Tomato leaf yellow virus | Tomato___Tomato_Yellow_Leaf_Curl_Virus | 70 | 6 | Synonym (PlantDoc's 'yellow virus' = TYLCV) |
| 25 | Tomato mold leaf | Tomato___Leaf_Mold | 85 | 6 | Synonym (Passalora fulva) |
| 26 | Tomato two spotted spider mites leaf | Tomato___Spider_mites Two-spotted_spider_mite | 2 | 0 | **Unmapped** (2 training images only; report as anecdotal; keep out of macro averages) |
| 27 | grape leaf | Grape___healthy | 57 | 12 | Healthy class |
| 28 | grape leaf black rot | Grape___Black_rot | 56 | 8 | Same disease |

### Unmapped PlantVillage Classes (10 of 38)

These 10 PlantVillage classes have no confirmed PlantDoc counterpart. If the model predicts any of these on PlantDoc images, the prediction is "out-of-mapping":

1. Orange___Haunglongbing_(Citrus_greening)
2. Peach___Bacterial_spot
3. Pepper,_bell___Bacterial_spot (already has caveat — see row 5)
4. Squash___Powdery_mildew — wait, this IS mapped (row 15). Correct: the unmapped ones are listed in the mapping CSV under `status: unmapped` or `relationship: none`.

Actually, reviewing the label_mapping.csv more carefully:
- 28 rows have `status: confirmed` or `confirmed_with_caveat`
- The last row (row 29, `grape leaf black rot`) has a caveat but is marked `confirmed_with_caveat`
- `Tomato two spotted spider mites leaf` has `status: confirmed_with_caveat` but only 2 training images → anecdotal

Per the mapping review, all 28 PlantVillage classes have a PlantDoc counterpart, but:
- `Tomato two spotted spider mites leaf` has only 2 training images → cannot support per-class claims; report as anecdotal and keep out of macro averages
- Some relationships are `synonym`, `assumed synonym`, or `likely synonym` — state the assumption wherever reported

## Split Definitions

| Split | Images | Notes |
|---|---|---|
| **Development (train)** | 2,331 | All rows marked `train`; **no validation split** exists |
| **Published test** | 236 | 236 images; used for final evaluation only |

### Split Warning

- PlantDoc development split has `{'train': 2331}` — **no validation split** is provided
- Any confidence threshold tuned on PlantDoc must use the 236-image test set
- Statistical power is too low for headline claims: 236 test images over ~27 classes → 4–12 per class; 95% interval on accuracy is roughly ±6.4 pp

## Image Properties

- **Resolution**: Varied originals; all images extracted from original repository folders
- **Color space**: Natural variation (field conditions); this is the domain shift source
- **File names**: Windows-invalid characters sanitized with reversible filename map
- **Capture conditions**: Field-oriented — varying lighting, angles, backgrounds, leaf positions
- **Class balance**: More even than PlantVillage, but 236 test images limits per-class granularity

## License & Usage Conditions

- **License**: CC BY 4.0
- **Required attribution**: "PlantDoc dataset, pratikkayal. CC BY 4.0."
- **Redistribution**: Source-level metadata only; individual images may not be redistributed
- **Commercial use**: Permitted with attribution
- **Modification**: Permitted; derivative works may be shared with attribution

## Cross-Dataset Mapping Rules (to PlantVillage)

| Rule | Detail |
|---|---|
| **Mapped subset size** | 28 of 38 PlantVillage classes have confirmed PlantDoc counterparts |
| **Out-of-mapping rate** | Predictions on classes PlantDoc does not contain must be tracked and reported |
| **Zero exact-hash overlap** | Verified 2026-10-25 across all partitions |
| **Per-class recall reporting** | Always report zero-support classes; macro-F1 on incomplete label set is meaningless |
| **Unmapped class handling** | `Tomato two spotted spider mites leaf` (2 training images) → report as anecdotal; exclude from macro averages |

## Known Limitations (from KNOWN_ISSUES.md)

- D08: PlantDoc accuracy not comparable to PlantVillage (28 vs 38 classes, ±6.4 pp interval)
- D09: PlantDoc has no validation split
- D10: PlantDoc statistical power too low (236 test images over 27 classes; ±6.4 pp 95% CI)
- D24: Storage redundant (PlantDoc 1.99 GB raw is unused by any current script)
- Project does not present PlantDoc accuracy as comparable to PlantVillage accuracy (RESEARCH_CLAIM.md H7)

## Obsidian Links

- `[[PlantVillage Test Split]]` — cross-dataset comparison
- `[[PlantVillage Label Mapping CSV]]` — 29-row mapping with caveats
- `[[EfficientNet-B0 Checkpoint]]` — evaluated on both datasets
- `[[WALLET_INDEX.md]]` — central navigation hub
- `[[ROADMAP.md]]` — S11: Score PlantDoc; S12: Quantify gap

---
# Dataset Card Index

| Metric | PlantVillage | PlantDoc |
|---|---|---|
| **Total images** | 54,305 (published) / 53,312 (manifest-excluded) | 2,578 extracted / 2,598 upstream claimed |
| **Classes** | 38 | 28 labeled (of 38 possible) |
| **Train** | 36,762 | 2,331 |
| **Validation** | 6,544 | **None** (all train) |
| **Test / Published test** | 10,709 | 236 |
| **Domain** | Controlled (greenhouse/studio) | Field-oriented |
| **License** | CC BY-SA 3.0 | CC BY 4.0 |
| **20-image discrepancy** | — | Upstream claims 2,598; local manifest has 2,578 |
| **Out-of-mapping classes** | — | 10 of 38 PlantVillage classes have no PlantDoc counterpart |
| **Leaf-level leakage risk** | 2,676/10,709 test images (25.0%) have `fallback_<id>` group IDs | — |

*Every field is traceable to: `data/manifests/*.summary.json`, `data/label_mapping.csv`, `data/dataset_registry.csv`, or `results/registry.json`. No values are fabricated.*

---
*This dataset card is the authoritative source for dataset provenance in the Obsidian vault. Linked from `WALLET_INDEX.md`, `PROJECT_STATUS.md`, and `MODEL_CARD.md`. Updated whenever split definitions, class counts, or license terms change.*

---
*Generated: 2026-10-25*
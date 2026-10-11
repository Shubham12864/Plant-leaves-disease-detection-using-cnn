# Dataset sources and handling

## Downloaded datasets

- PlantVillage color configuration: owner-provided color train/test lists and leaf map are in `data/raw/PlantVillage/source/`; materialized class folders are in `data/raw/PlantVillage/{train,test}/`. The ZIP archive was removed after extraction to save space and can be downloaded again from the pinned source.
- PlantDoc: upstream license/readme and filename map are in `data/raw/PlantDoc/source/`; materialized upstream `train/` and `test/` class folders are retained. The ZIP archive was removed after extraction and can be downloaded again from the pinned source.

`data/dataset_registry.csv` records source revisions and license statements. Raw datasets are excluded from Git. Preserve attribution, do not redistribute without checking the original terms, and do not merge labels or image folders across datasets.

## Provenance and split policy

PlantVillage upstream train/test manifests are retained. `plantvillage_development.csv` excludes published-train rows sharing an upstream leaf-map group ID or exact SHA-256 with the published test; remaining groups are partitioned into train/validation. The frozen split has zero exact-hash or group-ID overlap and all 38 classes in validation. The exclusion and seed are recorded in `reports/split_freeze.json`.

PlantDoc official train/test folders are retained. The upstream description says 2,598 images; the retrieved archive currently contains 2,578 image files (train/test counts are in the generated manifests). Keep this discrepancy visible and verify it before citing dataset size. The repository contains filenames with characters Windows cannot represent; extraction replaces those filename characters and writes `data/raw/PlantDoc/source/filename_map.csv` to map original names to saved paths.

Eleven PlantDoc train images exactly matched published-test hashes and were excluded from `plantdoc_development.csv`; the original `plantdoc_train.csv` remains intact. The resulting development/test manifests have zero exact-hash overlap. No specimen group metadata is available, so PlantDoc test remains a limited external evaluation set. PlantVillage and PlantDoc class names are different taxonomies; cross-domain reporting requires a reviewed mapping and should use only confirmed shared classes.

## Rebuild steps

Run `scripts/download_datasets.ps1` to fetch the pinned source versions and materialize them. Then run `scripts/build_all_manifests.ps1` to hash images, retain source manifests, freeze leakage-aware development manifests, and write the audit reports. The dataset archive files are large; do not redownload unless needed.

Images are not modified for preprocessing. Model-time transforms should be training-only; validation and test images must not be augmented.

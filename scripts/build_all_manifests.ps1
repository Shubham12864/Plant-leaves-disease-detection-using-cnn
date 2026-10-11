$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

python scripts\build_manifest.py --image-root data\raw\PlantVillage\train --dataset PlantVillage --domain controlled --metadata data\raw\PlantVillage\metadata.csv --source-split train --output data\manifests\plantvillage_train.csv
if ($LASTEXITCODE -ne 0) { throw 'PlantVillage train manifest failed' }
python scripts\build_manifest.py --image-root data\raw\PlantVillage\test --dataset PlantVillage --domain controlled --metadata data\raw\PlantVillage\metadata.csv --source-split test --output data\manifests\plantvillage_test.csv
if ($LASTEXITCODE -ne 0) { throw 'PlantVillage test manifest failed' }
python scripts\build_manifest.py --image-root data\raw\PlantDoc\train --dataset PlantDoc --domain field --source-split train --output data\manifests\plantdoc_train.csv
if ($LASTEXITCODE -ne 0) { throw 'PlantDoc train manifest failed' }
python scripts\build_manifest.py --image-root data\raw\PlantDoc\test --dataset PlantDoc --domain field --source-split test --output data\manifests\plantdoc_test.csv
if ($LASTEXITCODE -ne 0) { throw 'PlantDoc test manifest failed' }
python scripts\freeze_evaluation_splits.py
if ($LASTEXITCODE -ne 0) { throw 'Leakage-aware split freeze failed' }
python scripts\audit_manifests.py data\manifests\plantvillage_development.csv data\manifests\plantvillage_test.csv data\manifests\plantdoc_development.csv data\manifests\plantdoc_test.csv --output reports\data_audit.json
if ($LASTEXITCODE -ne 0) { throw 'Manifest overlap audit failed' }

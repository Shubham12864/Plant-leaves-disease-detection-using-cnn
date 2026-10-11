$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$pvRevision = '9e97599868962bd0079b8db4b7f1efa9185fa1e7'
$pdRevision = '5467f6012d78d1c446145d5f582da6096f852ae8'
$pvSource = Join-Path $projectRoot 'data\raw\PlantVillage\source'
$pdSource = Join-Path $projectRoot 'data\raw\PlantDoc\source'
New-Item -ItemType Directory -Force -Path $pvSource, $pdSource | Out-Null

curl.exe -L --fail --retry 5 "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/$pvRevision/data.zip" -o (Join-Path $pvSource 'data.zip')
if ($LASTEXITCODE -ne 0) { throw 'PlantVillage archive download failed' }
foreach ($split in @('train','test')) {
    curl.exe -L --fail --retry 5 "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/$pvRevision/splits/color_$split.txt" -o (Join-Path $pvSource "color_$split.txt")
    if ($LASTEXITCODE -ne 0) { throw "PlantVillage $split split download failed" }
}
curl.exe -L --fail --retry 5 "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/$pvRevision/leaf_grouping/leaf-map.json" -o (Join-Path $pvSource 'leaf-map.json')
if ($LASTEXITCODE -ne 0) { throw 'PlantVillage leaf map download failed' }
curl.exe -L --fail --retry 5 "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/$pvRevision/README.md" -o (Join-Path $pvSource 'DATASET_CARD.md')
if ($LASTEXITCODE -ne 0) { throw 'PlantVillage dataset card download failed' }

curl.exe -L --fail --retry 5 "https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/$pdRevision" -o (Join-Path $pdSource 'PlantDoc-Dataset.zip')
if ($LASTEXITCODE -ne 0) { throw 'PlantDoc archive download failed' }
python scripts\materialize_plantvillage.py --source-dir data\raw\PlantVillage\source --output-dir data\raw\PlantVillage
if ($LASTEXITCODE -ne 0) { throw 'PlantVillage extraction failed' }
python scripts\materialize_plantdoc.py --archive data\raw\PlantDoc\source\PlantDoc-Dataset.zip --output-dir data\raw\PlantDoc
if ($LASTEXITCODE -ne 0) { throw 'PlantDoc extraction failed' }

param([switch]$IncludePlantDoc)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$destination = Join-Path $projectRoot 'kaggle_upload'
$pv = Join-Path $destination 'PlantVillage'
New-Item -ItemType Directory -Force -Path $pv | Out-Null
if ($IncludePlantDoc) {
    $pd = Join-Path $destination 'PlantDoc'
    New-Item -ItemType Directory -Force -Path $pd | Out-Null
}

function Copy-File($source, $target) {
    if (Test-Path -LiteralPath $source) {
        $parent = Split-Path -Parent $target
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        Copy-Item -LiteralPath $source -Destination $target -Force
    }
}
function Copy-Tree($source, $target) {
    if (Test-Path -LiteralPath $source) {
        New-Item -ItemType Directory -Force -Path $target | Out-Null
        Copy-Item -Path (Join-Path $source '*') -Destination $target -Recurse -Force
    }
}

# Package only color images, provenance, and manifests; omit large upstream archives and .git.
Copy-Tree 'data\raw\PlantVillage\train' (Join-Path $pv 'data\raw\PlantVillage\train')
Copy-Tree 'data\raw\PlantVillage\test' (Join-Path $pv 'data\raw\PlantVillage\test')
Copy-File 'data\raw\PlantVillage\metadata.csv' (Join-Path $pv 'data\raw\PlantVillage\metadata.csv')
Copy-File 'data\raw\PlantVillage\source\DATASET_CARD.md' (Join-Path $pv 'data\raw\PlantVillage\source\DATASET_CARD.md')
Copy-File 'data\manifests\plantvillage_development.csv' (Join-Path $pv 'data\manifests\plantvillage_development.csv')
Copy-File 'data\manifests\plantvillage_test.csv' (Join-Path $pv 'data\manifests\plantvillage_test.csv')
Copy-File 'data\dataset_registry.csv' (Join-Path $pv 'data\dataset_registry.csv')

if ($IncludePlantDoc) {
    Copy-Tree 'data\raw\PlantDoc\train' (Join-Path $pd 'data\raw\PlantDoc\train')
    Copy-Tree 'data\raw\PlantDoc\test' (Join-Path $pd 'data\raw\PlantDoc\test')
    Copy-File 'data\raw\PlantDoc\source\filename_map.csv' (Join-Path $pd 'data\raw\PlantDoc\source\filename_map.csv')
    Copy-File 'data\raw\PlantDoc\source\UPSTREAM_LICENSE.txt' (Join-Path $pd 'data\raw\PlantDoc\source\UPSTREAM_LICENSE.txt')
    Copy-File 'data\raw\PlantDoc\source\UPSTREAM_README.md' (Join-Path $pd 'data\raw\PlantDoc\source\UPSTREAM_README.md')
    Copy-File 'data\manifests\plantdoc_development.csv' (Join-Path $pd 'data\manifests\plantdoc_development.csv')
    Copy-File 'data\manifests\plantdoc_test.csv' (Join-Path $pd 'data\manifests\plantdoc_test.csv')
    Copy-File 'data\dataset_registry.csv' (Join-Path $pd 'data\dataset_registry.csv')
}
Write-Output "PlantVillage upload folder prepared under $pv. Use -IncludePlantDoc only when preparing the later field-evaluation input. Keep uploads private unless redistribution terms have been reviewed."

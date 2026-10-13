param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,

    [Parameter(Mandatory = $true)]
    [string]$KaggleResultsZip,

    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Importer = Join-Path $PSScriptRoot "import_kaggle_results.py"
$AccuracyPipeline = Join-Path $PSScriptRoot "run_accuracy_pipeline.py"

if (-not (Test-Path -LiteralPath $DatasetRoot -PathType Container)) {
    throw "Dataset root not found: $DatasetRoot"
}
if (-not (Test-Path -LiteralPath $KaggleResultsZip -PathType Leaf)) {
    throw "Kaggle results ZIP not found: $KaggleResultsZip"
}

Write-Host "[1/2] Importing the EfficientNet-B0 checkpoint and Kaggle reports..."
& $Python $Importer $KaggleResultsZip
if ($LASTEXITCODE -ne 0) { throw "Kaggle result import failed with exit code $LASTEXITCODE" }

Write-Host "[2/2] Evaluating SmallCNN, ResNet18, and EfficientNet-B0 on validation..."
& $Python $AccuracyPipeline --dataset-root $DatasetRoot
if ($LASTEXITCODE -ne 0) { throw "Model accuracy pipeline failed with exit code $LASTEXITCODE" }

Write-Host "Pipeline complete. Results are under reports/model_comparison/accuracy_pipeline."

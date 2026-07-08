param(
    [string]$ProjectRoot = (Resolve-Path ".").Path,
    [string]$PriorityCsv = "outputs\priority_sensitivity_20260706_152020\penalty_0\priority_scores_pseudo.csv",
    [string]$Strategies = "Random,RandomClassDatasetBalanced,ConsistencyOnly,ConsistencyOnlyClassDatasetBalanced",
    [string]$Seeds = "42,43,44,45,46,47,48,49",
    [string]$Device = "0",
    [int]$BatchSize = 8,
    [int]$Workers = 4,
    [int]$Epochs = 30,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot

$env:AL_PROJECT_ROOT = $ProjectRoot
$env:AL_PRIORITY_CSV = $PriorityCsv
$env:AL_STRATEGIES = $Strategies
$env:AL_SEEDS = $Seeds
$env:AL_YOLO_DEVICE = $Device
$env:AL_BATCH_SIZE = "$BatchSize"
$env:AL_WORKERS = "$Workers"
$env:AL_EPOCHS_PER_ROUND = "$Epochs"
$env:AL_SUPPRESS_NO_PSEUDO_GAMMA = "0.1"

if ($DryRun) {
    $env:AL_DRY_RUN_ONLY = "1"
} else {
    Remove-Item Env:\AL_DRY_RUN_ONLY -ErrorAction SilentlyContinue
}

Write-Host "[AL Windows CUDA]"
Write-Host "ProjectRoot : $env:AL_PROJECT_ROOT"
Write-Host "PriorityCsv : $env:AL_PRIORITY_CSV"
Write-Host "Strategies  : $env:AL_STRATEGIES"
Write-Host "Seeds       : $env:AL_SEEDS"
Write-Host "Device      : $env:AL_YOLO_DEVICE"
Write-Host "BatchSize   : $env:AL_BATCH_SIZE"
Write-Host "Workers     : $env:AL_WORKERS"
Write-Host "Epochs      : $env:AL_EPOCHS_PER_ROUND"
Write-Host "DryRun      : $DryRun"

.\.venv\Scripts\python.exe scripts\02_active_learning\run_al_yolo_ablation_v3_windows_cuda.py

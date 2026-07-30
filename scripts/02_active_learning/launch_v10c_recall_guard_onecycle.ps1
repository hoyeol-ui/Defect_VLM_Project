param(
    [string]$Seeds = "47,48,49,50",
    [string]$Device = "0",
    [string]$DryRun = "0"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\user\Desktop\vlm\Defect_VLM_Project"
Set-Location $ProjectRoot

$env:AL_DRY_RUN_ONLY = $DryRun
$env:AL_ACQUISITION_SEEDS = $Seeds

$env:AL_YOLO_DEVICE = $Device
$env:AL_BATCH_SIZE = "8"
$env:AL_WORKERS = "4"
$env:AL_YOLO_CACHE = "false"
$env:AL_YOLO_PLOTS = "false"

$env:AL_INITIAL_SEED_SIZE = "60"
$env:AL_QUERY_SIZE = "30"

# V10c core: keep the V10b useful part, but reduce instance over-bias.
$env:AL_V10C_W_UNCERTAINTY = "0.30"
$env:AL_V10C_W_DINO = "0.35"
$env:AL_V10C_W_BALANCE = "0.20"
$env:AL_V10C_W_INSTANCE = "0.15"

# V10c recall guard: reserve a small quota for low-coverage candidates.
$env:AL_V10C_CORE_SIZE = "24"
$env:AL_V10C_RECALL_GUARD_SIZE = "6"
$env:AL_V10C_CORE_MAX_NO_BOX = "0"
$env:AL_V10C_CORE_MIN_PSEUDO_BOXES = "2"
$env:AL_V10C_CORE_MAX_PER_PRED_CLASS = "2"
$env:AL_V10C_GUARD_MAX_NO_BOX = "2"
$env:AL_V10C_GUARD_MAX_PER_PRED_CLASS = "3"

$env:AL_V10C_GUARD_W_UNCERTAINTY = "0.25"
$env:AL_V10C_GUARD_W_DINO = "0.30"
$env:AL_V10C_GUARD_W_BALANCE = "0.20"
$env:AL_V10C_GUARD_W_LOW_COVERAGE = "0.25"

# Detector scoring settings inherited from V9/V10b.
$env:AL_V9_CANDIDATE_FRACTION = "1.00"
$env:AL_V9_PREDICT_CONF = "0.05"
$env:AL_V9_PREDICT_IOU = "0.70"

Write-Host "===================================================================================================="
Write-Host "[V10c recall-guard one-cycle launcher]"
Write-Host "Project: $ProjectRoot"
Write-Host "Seeds  : $Seeds"
Write-Host "Device : $Device"
Write-Host "DryRun : $DryRun"
Write-Host "Final test remains locked / unused"
Write-Host "===================================================================================================="

& .\.python311\python.exe scripts\02_active_learning\run_v10c_recall_guard_onecycle.py

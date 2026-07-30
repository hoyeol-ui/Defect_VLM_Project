param(
    [string]$Seeds = "47,48",
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

# PDF proposal: V10b-like core + explicit recall guard.
$env:AL_V10C_CORE_SIZE = "21"
$env:AL_V10C_RECALL_GUARD_SIZE = "9"
$env:AL_V10C_CORE_MAX_NO_BOX = "0"
$env:AL_V10C_CORE_MIN_PSEUDO_BOXES = "2"
$env:AL_V10C_CORE_MAX_PER_PRED_CLASS = "2"

$env:AL_V10C_PDF_NO_BOX_QUOTA = "3"
$env:AL_V10C_PDF_ONE_BOX_QUOTA = "6"
$env:AL_V10C_GUARD_MAX_NO_BOX = "3"
$env:AL_V10C_PDF_GUARD_UNCERTAINTY_Q = "0.70"
$env:AL_V10C_PDF_GUARD_DINO_Q = "0.50"

# Keep V10b-like core weights.
$env:AL_V10C_PDF_CORE_W_UNCERTAINTY = "0.25"
$env:AL_V10C_PDF_CORE_W_DINO = "0.35"
$env:AL_V10C_PDF_CORE_W_BALANCE = "0.15"
$env:AL_V10C_PDF_CORE_W_INSTANCE = "0.25"

# Proposal guard score:
# 0.45 uncertainty + 0.30 DINO distance + 0.15 class deficit + 0.10 low confidence.
$env:AL_V10C_PDF_GUARD_W_UNCERTAINTY = "0.45"
$env:AL_V10C_PDF_GUARD_W_DINO = "0.30"
$env:AL_V10C_PDF_GUARD_W_BALANCE = "0.15"
$env:AL_V10C_PDF_GUARD_W_LOW_CONF = "0.10"

$env:AL_V9_CANDIDATE_FRACTION = "1.00"
$env:AL_V9_PREDICT_CONF = "0.05"
$env:AL_V9_PREDICT_IOU = "0.70"

Write-Host "===================================================================================================="
Write-Host "[V10c-PDF recall-guard one-cycle launcher]"
Write-Host "Project: $ProjectRoot"
Write-Host "Seeds  : $Seeds"
Write-Host "Device : $Device"
Write-Host "DryRun : $DryRun"
Write-Host "Core/Guard: 21/9; no-box <= 3; one-box <= 6"
Write-Host "Final test remains locked / unused"
Write-Host "===================================================================================================="

& .\.python311\python.exe scripts\02_active_learning\run_v10c_pdf_recall_guard_onecycle.py

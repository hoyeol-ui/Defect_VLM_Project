param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$GpuMaxMemory = "6.5GiB",
    [string]$CpuMaxMemory = "24GiB"
)

$ErrorActionPreference = "Stop"
$Manifest = "runs\vlm_consistency_groundedness_validity\paired_compliance_probe20_gc10_20260715\paired_probe_manifest.csv"
$GtAudit = "runs\vlm_consistency_groundedness_validity\paired_compliance_probe20_gc10_20260715\paired_probe_gt_audit.csv"
$OriginalAudit = "runs\vlm_consistency_groundedness_validity\paired_compliance_probe20_gc10_20260715\audit"
$Root = "runs\vlm_consistency_groundedness_validity\paired_model_comparison_gc10_20260715"

$Models = @(
    @{
        Name = "qwen3_vl_2b"
        ModelId = "Qwen/Qwen3-VL-2B-Instruct"
        Revision = "89644892e4d85e24eaac8bacfd4f463576704203"
    },
    @{
        Name = "qwen25_vl_3b"
        ModelId = "Qwen/Qwen2.5-VL-3B-Instruct"
        Revision = "66285546d2b821cf421d4f5eb2576359d3770cd3"
    }
)

foreach ($Model in $Models) {
    $ModelRoot = Join-Path $Root $Model.Name
    $Responses = Join-Path $ModelRoot "responses"
    $Audit = Join-Path $ModelRoot "audit"

    & $Python scripts\01_score_generation\generate_forced_vlm_compliance_probe.py `
        --manifest $Manifest `
        --output-dir $Responses `
        --model-id $Model.ModelId `
        --revision $Model.Revision `
        --gpu-max-memory $GpuMaxMemory `
        --cpu-max-memory $CpuMaxMemory
    if ($LASTEXITCODE -ne 0) {
        throw "Inference failed for $($Model.Name)"
    }

    & $Python scripts\03_analysis\audit_paired_vlm_compliance_probe.py `
        --responses (Join-Path $Responses "forced_compliance_responses.jsonl") `
        --gt-audit $GtAudit `
        --output-dir $Audit
    if ($LASTEXITCODE -ne 0) {
        throw "Audit failed for $($Model.Name)"
    }
}

& $Python scripts\03_analysis\compare_paired_vlm_model_gates.py `
    --model "qwen2_vl_2b=$OriginalAudit" `
    --model "qwen3_vl_2b=$(Join-Path $Root 'qwen3_vl_2b\audit')" `
    --model "qwen25_vl_3b=$(Join-Path $Root 'qwen25_vl_3b\audit')" `
    --output-dir (Join-Path $Root "comparison")
if ($LASTEXITCODE -ne 0) {
    throw "Cross-model comparison failed"
}

Write-Host "[DONE] Frozen paired VLM model comparison finished"
Write-Host "[SUMMARY] $(Join-Path $Root 'comparison\paired_model_gate_comparison_summary.md')"

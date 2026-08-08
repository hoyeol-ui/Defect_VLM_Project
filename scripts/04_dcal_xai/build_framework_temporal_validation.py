from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
OUT = ROOT / "runs" / "framework_temporal_validation_20260718"


def row(**kwargs: Any) -> dict[str, Any]:
    return kwargs


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def file_time(path: str, fallback: str) -> str:
    p = ROOT / path.replace("/", "\\")
    if p.exists():
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return fallback


BRANCHES = [
    row(branch_id="B01", branch="V10c24 scale extension", stage="downstream_learning_utility", protocol_time=file_time("scripts/02_active_learning/run_v10c24_round2_scale_smoke.py", "2026-07-13 23:25:14"), outcome_time=file_time("runs/active_learning_v10c24_round2_scale_smoke/v10c24_round2_scale_smoke_20260713_225326/recovered_v10c24_round2_scale_smoke_summary.md", "2026-07-13 23:27:20"), hypothesis="Detector-aware selector scales at round2 budget120", primary_endpoint="mAP50-95 and recall/class safety", threshold="six frozen scale checks", upstream_result="mAP delta -0.004863", local_gate="FAIL", local_gate_precommitted=False, generic_policy_precommitted=False, workflow_action="STOP", next_expensive_step="additional V10c/V10d scale runs", actual_later_outcome="not observed because stopped", actual_outcome_class="COUNTERFACTUAL_UNOBSERVED", model_runs_consumed="existing run only", model_runs_prevented="not quantified", final_test_consumed=0, source_file="runs/active_learning_v10c24_round2_scale_smoke/v10c24_round2_scale_smoke_20260713_225326/recovered_v10c24_round2_scale_smoke_summary.md"),
    row(branch_id="B02", branch="NEU seed45 fixed-set stability", stage="downstream_learning_utility", protocol_time=file_time("scripts/02_active_learning/run_seed45_fixed_set_stability.py", "2026-07-14 21:16:53"), outcome_time=file_time("runs/seed45_fixed_set_stability/seed45_fixed_set_stability_main/seed45_fixed_set_stability_summary.md", "2026-07-14 21:17:03"), hypothesis="One Visual20 selected set is stable across training seeds", primary_endpoint="mAP50-95 stability", threshold="gain>=0.01; >=4/5 wins; safety", upstream_result="+0.016236; 5/5", local_gate="PASS_DIAGNOSTIC", local_gate_precommitted=False, generic_policy_precommitted=False, workflow_action="ADVANCE_TO_ACQUISITION_CONFIRMATION", next_expensive_step="10 new acquisition-set confirmation runs", actual_later_outcome="independent confirmation FAIL (+0.007019; p=0.322266)", actual_outcome_class="LATER_FAILURE", model_runs_consumed=20, model_runs_prevented=0, final_test_consumed=0, source_file="runs/seed45_fixed_set_stability/seed45_fixed_set_stability_main/screen_gate.csv"),
    row(branch_id="B03", branch="NEU independent acquisition confirmation", stage="acquisition_set_confirmation", protocol_time=file_time("docs/v8_cold_start_visual_confirmation_protocol_20260714.md", "2026-07-14 21:21:51"), outcome_time=file_time("runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main/v8_cold_start_visual_confirmation_analysis.md", "2026-07-14 23:10:58"), hypothesis="Seed45 benefit generalizes to new selected sets", primary_endpoint="paired mAP50-95 across 10 acquisition seeds", threshold="gain>=0.01; 7/10; p<=0.05; recall safety", upstream_result="+0.007019; CI crosses 0; p=0.322266", local_gate="FAIL", local_gate_precommitted=True, generic_policy_precommitted=False, workflow_action="STOP", next_expensive_step="larger cold-start selector study", actual_later_outcome="not observed because stopped", actual_outcome_class="COUNTERFACTUAL_UNOBSERVED", model_runs_consumed=30, model_runs_prevented="not quantified", final_test_consumed=0, source_file="runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main/confirmatory_gate.csv"),
    row(branch_id="B04", branch="VisA global DINO selection", stage="composition_safety", protocol_time=file_time("docs/visa_gt_free_annotation_triage_protocol_20260715.md", "2026-07-15 19:24:35"), outcome_time=file_time("docs/visa_selection_only_decision_20260715.md", "2026-07-15 19:51:13"), hypothesis="Anomaly discovery improves without category collapse", primary_endpoint="anomaly yield plus 12-category safety", threshold="category coverage not worse", upstream_result="anomaly +14.480; category -4.110", local_gate="FAIL", local_gate_precommitted=True, generic_policy_precommitted=False, workflow_action="STOP", next_expensive_step="downstream anomaly-model experiment", actual_later_outcome="not observed because stopped", actual_outcome_class="COUNTERFACTUAL_UNOBSERVED", model_runs_consumed=0, model_runs_prevented="not quantified", final_test_consumed=0, source_file="docs/visa_selection_only_decision_20260715.md"),
    row(branch_id="B05", branch="MPDD global DINO selection", stage="composition_safety", protocol_time=file_time("docs/mpdd_hierarchical_dino_selection_protocol_20260715.md", "2026-07-15 20:03:16"), outcome_time=file_time("docs/mpdd_selection_only_decision_20260715.md", "2026-07-15 20:11:17"), hypothesis="Anomaly discovery improves with category safety", primary_endpoint="anomaly yield plus product-category safety", threshold="category delta>=-0.25", upstream_result="anomaly +6.245; category -0.325", local_gate="FAIL", local_gate_precommitted=True, generic_policy_precommitted=False, workflow_action="STOP", next_expensive_step="downstream anomaly-model experiment", actual_later_outcome="not observed because stopped", actual_outcome_class="COUNTERFACTUAL_UNOBSERVED", model_runs_consumed=0, model_runs_prevented="not quantified", final_test_consumed=0, source_file="docs/mpdd_selection_only_decision_20260715.md"),
    row(branch_id="B06", branch="GC10 global DINO selection", stage="selection_validity", protocol_time=file_time("docs/gc10_taxonomy_selection_protocol_20260715.md", "2026-07-15 20:22:46"), outcome_time=file_time("runs/gc10_taxonomy_selection_audit/gc10_random_vs_dino_200seed_20260715/gc10_selection_only_gate.csv", "2026-07-15 20:24:24"), hypothesis="Rare/taxonomy discovery improves safely", primary_endpoint="11 discovery/coverage/safety checks", threshold="rare gain>=0.75; combined-class gain>=0.25; safety", upstream_result="rare +2.720; combined classes +0.825", local_gate="PASS", local_gate_precommitted=True, generic_policy_precommitted=False, workflow_action="ADVANCE_TO_BOUNDED_SCREEN", next_expensive_step="one 5-acquisition x 3-training-seed detector screen", actual_later_outcome="overall mAP +0.017378 but rare AP -0.019877; downstream gate FAIL", actual_outcome_class="LATER_FAILURE", model_runs_consumed=30, model_runs_prevented=0, final_test_consumed=0, source_file="runs/gc10_taxonomy_selection_audit/gc10_random_vs_dino_200seed_20260715/gc10_selection_only_gate.csv"),
    row(branch_id="B07", branch="GC10 first detector translation", stage="downstream_learning_utility", protocol_time=file_time("docs/gc10_development_detector_confirmation_protocol_20260715.md", "2026-07-15 20:29:17"), outcome_time=file_time("runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/gc10_detector_confirmation_summary.md", "2026-07-15 21:10:17"), hypothesis="Discovery gain translates to overall and rare detector utility", primary_endpoint="mAP50-95 and rare macro AP", threshold="mAP gain>=0.010; rare gain>=0.020; safety", upstream_result="mAP +0.017378; rare AP -0.019877", local_gate="FAIL", local_gate_precommitted=True, generic_policy_precommitted=False, workflow_action="STOP", next_expensive_step="expanded detector study", actual_later_outcome="not observed because stopped", actual_outcome_class="COUNTERFACTUAL_UNOBSERVED", model_runs_consumed=30, model_runs_prevented="not quantified", final_test_consumed=0, source_file="runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_development_confirmation_gate.csv"),
    row(branch_id="B08", branch="VLM consistency validity", stage="signal_validity", protocol_time=file_time("docs/vlm_consistency_groundedness_validity_protocol_20260715.md", "2026-07-15 21:33:30"), outcome_time=file_time("runs/vlm_consistency_groundedness_validity/legacy_pilot_audit_20260715_main/legacy_validity_summary.md", "2026-07-15 21:40:32"), hypothesis="Explanation consistency tracks grounding/error", primary_endpoint="Spearman/AUC/quartile gap", threshold="six frozen signal-validity checks", upstream_result="Spearman -0.181199; AUC 0.373433", local_gate="FAIL", local_gate_precommitted=True, generic_policy_precommitted=False, workflow_action="STOP_EXPENSIVE_PATH_ALLOW_CHEAP_DIAGNOSTIC", next_expensive_step="VLM-scored acquisition plus detector training", actual_later_outcome="oracle/model diagnostics also failed; expensive acquisition not run", actual_outcome_class="DIAGNOSTIC_CORROBORATION", model_runs_consumed=0, model_runs_prevented="not quantified", final_test_consumed=0, source_file="runs/vlm_consistency_groundedness_validity/legacy_pilot_audit_20260715_main/legacy_validity_gate.csv"),
    row(branch_id="B09", branch="VLM oracle crop groundedness", stage="signal_validity", protocol_time=file_time("scripts/03_analysis/audit_oracle_crop_vlm_diagnostic.py", "2026-07-15 22:33:28"), outcome_time=file_time("runs/vlm_consistency_groundedness_validity/oracle_crop_diagnostic20_gc10_20260715/oracle_crop_audit/oracle_crop_diagnostic_summary.md", "2026-07-15 22:33:36"), hypothesis="Oracle localization restores grounded evidence", primary_endpoint="presence/evidence/IoU", threshold="presence>=0.50; evidence>=0.70", upstream_result="presence 0; evidence 0; median IoU 0", local_gate="FAIL", local_gate_precommitted=False, generic_policy_precommitted=False, workflow_action="STOP_EXPENSIVE_PATH_ALLOW_CHEAP_DIAGNOSTIC", next_expensive_step="prompt/selector expansion", actual_later_outcome="paired model probe also 0/3 pass", actual_outcome_class="DIAGNOSTIC_CORROBORATION", model_runs_consumed=0, model_runs_prevented="not quantified", final_test_consumed=0, source_file="runs/vlm_consistency_groundedness_validity/oracle_crop_diagnostic20_gc10_20260715/oracle_crop_audit/oracle_crop_diagnostic_gate.csv"),
    row(branch_id="B10", branch="Paired VLM model comparison", stage="signal_validity", protocol_time=file_time("scripts/03_analysis/run_paired_vlm_model_comparison.ps1", "2026-07-15 22:51:04"), outcome_time=file_time("runs/vlm_consistency_groundedness_validity/paired_model_comparison_gc10_20260715/comparison/paired_model_gate_comparison_summary.md", "2026-07-15 23:15:53"), hypothesis="Another small VLM restores compliance and grounding", primary_endpoint="balanced accuracy plus bbox grounding", threshold="all frozen paired checks", upstream_result="0/3 pass; best BA 0.70 with IoU 0", local_gate="FAIL", local_gate_precommitted=False, generic_policy_precommitted=False, workflow_action="STOP", next_expensive_step="further prompt/model search", actual_later_outcome="not observed because stopped", actual_outcome_class="COUNTERFACTUAL_UNOBSERVED", model_runs_consumed=0, model_runs_prevented="not quantified", final_test_consumed=0, source_file="runs/vlm_consistency_groundedness_validity/paired_model_comparison_gc10_20260715/comparison/paired_model_gate_comparison.csv"),
    row(branch_id="B11", branch="GC10 D2R label-aware retrieval", stage="signal_validity", protocol_time=file_time("docs/gc10_d2r_label_aware_retrieval_diagnostic_protocol_20260715.md", "2026-07-15 23:52:05"), outcome_time=file_time("runs/gc10_discovery_representation_audit/gc10_d2r_200seed_20260715/label_aware_retrieval_audit/label_aware_class8_recovery_gate.csv", "2026-07-15 23:53:24"), hypothesis="Label-aware global retrieval repairs class8 representation", primary_endpoint="top1/P@5/top32 recovery", threshold="top1>=0.50 plus six checks", upstream_result="best top1 0.153846; 0 eligible methods", local_gate="FAIL", local_gate_precommitted=True, generic_policy_precommitted=False, workflow_action="STOP", next_expensive_step="15-model detector confirmation", actual_later_outcome="not observed because stopped", actual_outcome_class="COUNTERFACTUAL_UNOBSERVED", model_runs_consumed=0, model_runs_prevented=15, final_test_consumed=0, source_file="runs/gc10_discovery_representation_audit/gc10_d2r_200seed_20260715/label_aware_retrieval_audit/label_aware_class8_recovery_gate.csv"),
    row(branch_id="B12", branch="DCAL-XAI flip disagreement", stage="selection_validity", protocol_time=file_time("scripts/04_dcal_xai/protocol.md", "2026-07-18 12:15:25"), outcome_time=file_time("runs/dcal_xai/gc10_r1/selection_summary.md", "2026-07-18 12:47:51"), hypothesis="Flip disagreement is useful detector uncertainty", primary_endpoint="coverage plus instance noninferiority", threshold="all seven R1 checks", upstream_result="instances -2.2; 94/100 one-view dominated", local_gate="FAIL", local_gate_precommitted=True, generic_policy_precommitted=False, workflow_action="STOP", next_expensive_step="hybrid detector confirmation", actual_later_outcome="not observed because stopped", actual_outcome_class="COUNTERFACTUAL_UNOBSERVED", model_runs_consumed="acquisition warm-start training only", model_runs_prevented="not quantified", final_test_consumed=0, source_file="runs/dcal_xai/gc10_r1/selection_summary.md"),
    row(branch_id="B13", branch="K40 budget140 selection holdout", stage="selection_validity", protocol_time=file_time("scripts/04_dcal_xai/v2_budget_extension_protocol.md", "2026-07-18 12:57:32"), outcome_time=file_time("runs/dcal_xai/v2_budget_extended/budget_decision.md", "2026-07-18 12:58:10"), hypothesis="Larger budget and K40 reduce taxonomy omission", primary_endpoint="all-class/all-rare/min-two coverage and safety", threshold="design thresholds reused on independent holdout", upstream_result="all-class 0.955 vs 0.940; selection PASS", local_gate="PASS", local_gate_precommitted=True, generic_policy_precommitted=False, workflow_action="ADVANCE_TO_BOUNDED_SCREEN", next_expensive_step="30-model YOLOv8n screen", actual_later_outcome="mAP -0.001678; rare -0.018290; recall -0.021871; FAIL", actual_outcome_class="LATER_FAILURE", model_runs_consumed=0, model_runs_prevented=0, final_test_consumed=0, source_file="runs/dcal_xai/v2_budget_extended/decision.json"),
    row(branch_id="B14", branch="K40 YOLOv8n downstream", stage="downstream_learning_utility", protocol_time=file_time("scripts/04_dcal_xai/v2_backbone_protocol.md", "2026-07-18 13:31:30"), outcome_time=file_time("runs/dcal_xai/v2_backbone_main/v8n_screen_decision.md", "2026-07-18 15:10:25"), hypothesis="K40 coverage translates to detector and rare utility", primary_endpoint="mAP, recall, rare/worst-class safety", threshold="five frozen downstream checks", upstream_result="mAP -0.001678; rare -0.018290; recall -0.021871", local_gate="FAIL", local_gate_precommitted=True, generic_policy_precommitted=False, workflow_action="STOP", next_expensive_step="30 YOLOv8s expansion models", actual_later_outcome="not observed because stopped", actual_outcome_class="COUNTERFACTUAL_UNOBSERVED", model_runs_consumed=30, model_runs_prevented=30, final_test_consumed=0, source_file="runs/dcal_xai/v2_backbone_main/v8n_screen_decision.json"),
    row(branch_id="B15", branch="V2.3 detector uncertainty", stage="operational_validity", protocol_time=file_time("scripts/04_dcal_xai/v2_detector_signal_validity_protocol.md", "2026-07-18 15:32:30"), outcome_time=file_time("runs/dcal_xai/v2_detector_signal_validity/detector_signal_validity_decision.md", "2026-07-18 15:35:37"), hypothesis="Top20% total-error enrichment is operationally sufficient", primary_endpoint="total FP+FN enrichment", threshold=">=1.50 plus CI/AUROC/rho/stability", upstream_result="1.115901; AUROC 0.766432; primary effect-size FAIL", local_gate="FAIL", local_gate_precommitted=True, generic_policy_precommitted=False, workflow_action="STOP", next_expensive_step="selection/retraining/FN implementation", actual_later_outcome="not observed because stopped; FN endpoints exploratory only", actual_outcome_class="COUNTERFACTUAL_UNOBSERVED", model_runs_consumed=0, model_runs_prevented=0, final_test_consumed=0, source_file="runs/dcal_xai/v2_detector_signal_validity/decision.json"),
]


PROTOCOL_SOURCES = {
    "B01": "scripts/02_active_learning/run_v10c24_round2_scale_smoke.py",
    "B02": "scripts/02_active_learning/run_seed45_fixed_set_stability.py",
    "B03": "docs/v8_cold_start_visual_confirmation_protocol_20260714.md",
    "B04": "docs/visa_gt_free_annotation_triage_protocol_20260715.md",
    "B05": "docs/mpdd_hierarchical_dino_selection_protocol_20260715.md",
    "B06": "docs/gc10_taxonomy_selection_protocol_20260715.md",
    "B07": "docs/gc10_development_detector_confirmation_protocol_20260715.md",
    "B08": "docs/vlm_consistency_groundedness_validity_protocol_20260715.md",
    "B09": "scripts/03_analysis/audit_oracle_crop_vlm_diagnostic.py",
    "B10": "scripts/03_analysis/run_paired_vlm_model_comparison.ps1",
    "B11": "docs/gc10_d2r_label_aware_retrieval_diagnostic_protocol_20260715.md",
    "B12": "scripts/04_dcal_xai/protocol.md",
    "B13": "scripts/04_dcal_xai/v2_budget_extension_protocol.md",
    "B14": "scripts/04_dcal_xai/v2_backbone_protocol.md",
    "B15": "scripts/04_dcal_xai/v2_detector_signal_validity_protocol.md",
}


FIELDS = ["branch_id", "chronology_partition", "branch", "stage", "protocol_time", "outcome_time", "protocol_source_file", "protocol_evidence_status", "hypothesis", "primary_endpoint", "threshold", "upstream_result", "local_gate", "local_gate_precommitted", "generic_policy_precommitted", "workflow_action", "next_expensive_step", "actual_later_outcome", "actual_outcome_class", "model_runs_consumed", "model_runs_prevented", "final_test_consumed", "source_file"]


def assign_partitions() -> None:
    ordered = sorted(BRANCHES, key=lambda x: x["outcome_time"])
    n_dev = math.floor(len(ordered) * 0.65)
    for index, branch in enumerate(ordered):
        branch["chronology_partition"] = "development_chronology" if index < n_dev else "temporal_holdout"
        branch["protocol_source_file"] = PROTOCOL_SOURCES[branch["branch_id"]]
        branch["protocol_evidence_status"] = "EXPLICIT_PROTOCOL_BEFORE_OUTCOME" if branch["local_gate_precommitted"] else "LOCAL_SCRIPT_BEFORE_SUMMARY_ONLY"


def holdout_rows() -> list[dict[str, Any]]:
    rows = []
    for branch in BRANCHES:
        if branch["chronology_partition"] != "temporal_holdout":
            continue
        rows.append(row(
            branch_id=branch["branch_id"], branch=branch["branch"], stage=branch["stage"],
            requested_predicted_decision="NOT_IDENTIFIABLE",
            actual_outcome=branch["actual_outcome_class"],
            correct_early_stop="NA", false_advance="NA", possible_false_stop="NA",
            reason="Generic six-stage policy was not frozen before holdout; stopped branches lack counterfactual downstream outcomes.",
            local_gate_action=branch["workflow_action"],
            local_process_conformance="PASS" if branch["local_gate_precommitted"] else "UNKNOWN",
            prevented_model_runs=branch["model_runs_prevented"], prevented_final_test=0,
            source_file=branch["source_file"],
        ))
    return rows


def loo_rows() -> list[dict[str, Any]]:
    claims = {
        "discovery_safety_separation": {"B04", "B05", "B06"},
        "selection_learning_separation": {"B06", "B07", "B13", "B14"},
        "training_stability_not_acquisition_generalization": {"B02", "B03"},
        "signal_ranking_not_operational_utility": {"B15"},
    }
    rows = []
    for branch in BRANCHES:
        for claim, support in claims.items():
            remaining = support - {branch["branch_id"]}
            if claim == "discovery_safety_separation":
                stable = len(remaining) >= 2
            elif claim == "selection_learning_separation":
                stable = bool(remaining & {"B06", "B13"}) and bool(remaining & {"B07", "B14"})
            elif claim == "training_stability_not_acquisition_generalization":
                stable = {"B02", "B03"}.issubset(remaining)
            else:
                stable = "B15" in remaining
            rows.append(row(removed_branch_id=branch["branch_id"], removed_branch=branch["branch"], claim=claim,
                            remaining_support_count=len(remaining), stable_after_removal=stable,
                            note="Claim-specific minimum support is lost." if not stable else "Claim retains its required cross-branch support."))
    return rows


def decision_doc() -> str:
    dev = [b["branch"] for b in BRANCHES if b["chronology_partition"] == "development_chronology"]
    hold = [b["branch"] for b in BRANCHES if b["chronology_partition"] == "temporal_holdout"]
    return f"""# Framework temporal validation decision

## Final decision

**C. RETROSPECTIVE_AUDIT_ONLY**

이 판정은 Evidence Freeze v2의 학위논문 재구성 판정을 뒤집지 않는다. 다만 현재 workflow를 **시간적으로 독립 검증된 predictive screening policy**라고 부를 수 없다는 뜻이다. 검증 가능한 기여는 branch-specific frozen gate의 process conformance, cost containment, claim-boundary enforcement다.

## 왜 제안된 temporal validation이 식별되지 않는가

1. 여섯 단계 generic policy는 모든 주요 branch가 종료된 뒤 Evidence Freeze v2에서 정식화됐다. 모든 holdout branch의 `generic_policy_precommitted`는 False다.
2. 조기 STOP branch에는 downstream을 실제 실행하지 않았으므로 SUCCESS/FAILURE 참값이 없다. 이것은 selective-label/verification-bias 문제다.
3. 따라서 downstream failure early-stop recall, false-advance rate, correct-stop precision, branch accuracy는 계산 불가다.
4. GC10 q20과 K40 selection PASS 뒤의 bounded detector screen FAIL을 `false advance`로 세면 안 된다. Selection gate의 역할은 learner success를 예측하는 것이 아니라 **한 번의 제한된 translation screen을 허가**하는 것이다.
5. File creation/modification time은 chronology 보조자료이지 policy preregistration의 독립 증거가 아니다. 11개 branch에는 명시적 protocol artifact가 outcome보다 앞서지만, B01/B02/B09/B10은 local script가 최종 summary보다 앞선 것만 확인되어 사전동결로 세지 않는다.

## Chronology partition

- Development chronology ({len(dev)}): {', '.join(dev)}
- Temporal holdout ({len(hold)}): {', '.join(hold)}

65/35 분할은 실제 outcome time 순서로 만들었지만, generic policy가 당시 동결되지 않았으므로 predictive confusion matrix는 `NOT_IDENTIFIABLE`로 남긴다.

## Frozen policy that is actually supportable

현재 자료가 지지하는 것은 예측 classifier가 아니라 다음 **authorization policy**다.

1. 각 branch의 primary endpoint와 threshold는 결과 전에 문서화한다.
2. FAIL/NOT IDENTIFIABLE은 다음 **고비용** 단계로 진행하지 않는다.
3. 실패 원인 규명을 위한 bounded cheap diagnostic은 별도 diagnostic으로 표시할 수 있다.
4. Selection PASS는 downstream success를 선언하지 않고, 정확히 한 번의 bounded translation screen만 허가한다.
5. Downstream 또는 operational FAIL은 확장 학습과 final-test 접근을 금지한다.

## Identifiable audit metrics

| Metric | Result | Interpretation |
|---|---:|---|
| Explicit branch protocol artifact before recorded outcome | 11/15 | local timestamp 기반 process evidence; 독립 preregistration 증명은 아님 |
| Local script before final summary only | 4/15 | B01/B02/B09/B10; 사전동결 수치에서 제외 |
| Generic policy documented before temporal holdout | 0/6 holdout branches | predictive temporal validation 불가 |
| Known fail-after-selection translations | 2 | GC10 q20, K40/140; bounded screen은 false advance가 아님 |
| Documented prevented detector model runs | at least 45 | D2R 15 + YOLOv8s 30 only |
| Locked final-test violations | 0 | 실제 사용 0; counterfactual ‘uses avoided’는 추정하지 않음 |

## Requested predictive metrics

| Requested metric | Value | Reason |
|---|---|---|
| downstream failure early-stop recall | NA | stopped branches have no downstream truth |
| false-advance rate | NA | generic policy was not pre-frozen; bounded screens are not errors |
| correct-stop precision | NA | stopped branches lack verification |
| branch-level predictive accuracy | NA | no valid confusion matrix |
| false-stop performance | NA | robust positive selector branch absent and counterfactual unobserved |

## Leave-one-branch-out claim stability

- Discovery-safety separation: stable across GC10/MPDD/VisA.
- Selection-learning separation: stable across GC10 q20 and K40 protocols.
- Training stability vs acquisition generalization: requires the paired seed45/follow-up sequence; not stable if either member is removed.
- Ranking vs operational enrichment: currently rests on V2.3 only and is not leave-one-branch-out stable.

## Allowed claims

- Eleven explicitly protocolized branches show gate-based authorization or stopping before their recorded outcome; four script-only branches remain retrospective support only.
- The process prevented at least 45 explicitly documented detector model runs and kept final-test use at zero.
- The workflow is a defensible retrospective empirical audit and cost-containment framework.
- Selection PASS authorizes measurement of learning utility; it does not predict or guarantee success.

## Prohibited claims

- The framework prospectively predicts downstream failure with >=75% recall.
- False-advance rate is <=25%.
- Temporal holdout independently validated the framework.
- Final-test uses avoided can be counted beyond the observed zero-use record.
- Good selectors are retained with known sensitivity.

## Effect on thesis direction

학위 방향은 유지 가능하다. 다만 중심 기여의 용어를 `predictive screening framework`가 아니라 **validity-gated empirical evaluation and cost-containment workflow**로 제한해야 한다. 다음 단계는 GPU 실험이 아니라 6-8페이지 미니 논문을 지도교수에게 보여 학위 기여로 인정받는지 확인하는 것이다.
"""


def cost_rows() -> list[dict[str, Any]]:
    return [
        row(metric="prevented_detector_model_runs", scope="D2R class8 representation branch", value=15, unit="model_runs", identifiability="IDENTIFIED", counting_rule="Frozen D2R gate explicitly blocked the planned 15-model detector confirmation.", source_file="runs/gc10_discovery_representation_audit/gc10_d2r_200seed_20260715/label_aware_retrieval_audit/label_aware_class8_recovery_gate.csv"),
        row(metric="prevented_detector_model_runs", scope="K40 YOLOv8s expansion", value=30, unit="model_runs", identifiability="IDENTIFIED", counting_rule="Frozen YOLOv8n screen explicitly blocked the planned 30-model YOLOv8s expansion.", source_file="runs/dcal_xai/v2_backbone_main/v8n_screen_decision.json"),
        row(metric="prevented_detector_model_runs_total", scope="documented non-overlapping branches only", value=45, unit="model_runs", identifiability="IDENTIFIED_LOWER_BOUND", counting_rule="15 D2R + 30 K40 YOLOv8s; unquantified stopped branches excluded.", source_file="B11+B14"),
        row(metric="actual_locked_final_test_uses", scope="all 15 audited branches", value=0, unit="uses", identifiability="IDENTIFIED", counting_rule="Observed use only; no counterfactual avoided-use count.", source_file="docs/research_evolution_and_evidence_freeze_v2_20260718.md"),
        row(metric="counterfactual_final_test_uses_avoided", scope="all stopped branches", value="NA", unit="uses", identifiability="NOT_IDENTIFIABLE", counting_rule="A stopped branch does not reveal whether final test would otherwise have been consumed.", source_file="docs/thesis_claim_boundary_20260718.md"),
        row(metric="prevented_vlm_inference", scope="stopped VLM branches", value="NA", unit="calls", identifiability="NOT_QUANTIFIED", counting_rule="No frozen counterfactual call plan with non-overlapping accounting.", source_file="docs/research_evolution_and_evidence_freeze_v2_20260718.md"),
        row(metric="prevented_gpu_runs", scope="all stopped branches", value="NA", unit="runs", identifiability="NOT_QUANTIFIED", counting_rule="Only explicitly planned detector model counts are reported.", source_file="docs/research_evolution_and_evidence_freeze_v2_20260718.md"),
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_registry_rows() -> list[dict[str, Any]]:
    paths = {branch["source_file"] for branch in BRANCHES}
    paths.update(PROTOCOL_SOURCES.values())
    paths.update({
        "docs/research_evolution_and_evidence_freeze_v2_20260718.md",
        "docs/thesis_claim_boundary_20260718.md",
        "scripts/04_dcal_xai/build_framework_temporal_validation.py",
        "scripts/04_dcal_xai/test_framework_temporal_validation.py",
    })
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(sorted(paths), start=1):
        path = ROOT / raw.replace("/", "\\")
        exists = path.exists()
        rows.append(row(
            source_id=f"S{index:03d}", path=raw, exists=exists,
            last_write_time=datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if exists else "",
            sha256=sha256(path) if exists and path.is_file() else "",
            role="protocol_or_gate_definition" if raw in PROTOCOL_SOURCES.values() else "outcome_or_audit_evidence",
            note="Local filesystem timestamps establish chronology only; they are not independent preregistration evidence.",
        ))
    return rows


def configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for candidate in (Path(r"C:\Windows\Fonts\malgun.ttf"), Path(r"C:\Windows\Fonts\malgunsl.ttf")):
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(candidate)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    return plt


def save_figures() -> None:
    plt = configure_matplotlib()
    from matplotlib.dates import DateFormatter, DayLocator
    from matplotlib.patches import FancyBboxPatch

    FIGURES.mkdir(parents=True, exist_ok=True)
    navy, blue, green, amber, red, gray, pale = "#16324F", "#2F6B9A", "#3F7D57", "#C58A2B", "#B84A4A", "#6B7280", "#F4F7FA"

    # 1. What this audit can and cannot validate.
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.6); ax.axis("off")
    ax.text(0.45, 7.08, "Temporal framework audit: 예측 타당성은 NA, 공정·비용 증거는 확인", fontsize=18.5, weight=400, color=navy)
    panels = [
        (0.5, "요청된 예측 검증", ["Early-stop recall ≥ 0.75", "False advance ≤ 0.25", "Good-selector sensitivity"], red, "NOT IDENTIFIABLE"),
        (5.0, "식별 가능한 감사", ["명시적 protocol 11/15", "명시적 고비용 단계 차단", "Locked final 사용 여부"], green, "SUPPORTED IN SCOPE"),
        (9.5, "최종 판정", ["Generic policy는 사후 정식화", "STOP branch의 downstream truth 없음", "예측 정책 주장은 금지"], amber, "C. RETROSPECTIVE ONLY"),
    ]
    for x, title, lines, color, verdict in panels:
        ax.add_patch(FancyBboxPatch((x, 1.35), 4.0, 4.85, boxstyle="round,pad=0.16", facecolor=pale, edgecolor=color, linewidth=2.0))
        ax.text(x + 0.25, 5.72, title, fontsize=14.5, weight=400, color=color)
        for idx, line in enumerate(lines):
            ax.text(x + 0.28, 4.93 - idx * 0.72, "• " + line, fontsize=11.5, color=navy)
        ax.add_patch(FancyBboxPatch((x + 0.25, 1.72), 3.5, 0.72, boxstyle="round,pad=0.10", facecolor=color, alpha=0.15, edgecolor="none"))
        ax.text(x + 2.0, 2.08, verdict, ha="center", va="center", fontsize=11.3, weight=400, color=navy)
    ax.annotate("", xy=(4.82, 3.78), xytext=(4.18, 3.78), arrowprops=dict(arrowstyle="->", color=gray, lw=2))
    ax.annotate("", xy=(9.32, 3.78), xytext=(8.68, 3.78), arrowprops=dict(arrowstyle="->", color=gray, lw=2))
    ax.text(0.5, 0.55, "해석: 수치 gate를 ‘실패’한 것이 아니라, 필요한 참값과 사전 동결 정책이 없어 계산 자체가 성립하지 않는다.", fontsize=11.5, color=gray)
    fig.tight_layout(); fig.savefig(FIGURES / "framework_temporal_validation.png", bbox_inches="tight"); plt.close(fig)

    # 2. Chronological branch actions on a single time axis.
    ordered = sorted(BRANCHES, key=lambda item: item["outcome_time"])
    times = [datetime.strptime(item["outcome_time"], "%Y-%m-%d %H:%M:%S") for item in ordered]
    fig, ax = plt.subplots(figsize=(15, 9.4))
    y = list(range(len(ordered)))
    colors = [green if "PASS" in item["local_gate"] else red for item in ordered]
    markers = ["o" if item["workflow_action"].startswith("STOP") else "D" for item in ordered]
    for yi, time, item, color, marker in zip(y, times, ordered, colors, markers):
        ax.hlines(yi, min(times), time, color="#D9E0E7", linewidth=1.2, zorder=1)
        face = color if item["local_gate_precommitted"] else "white"
        ax.scatter(time, yi, s=75, marker=marker, facecolor=face, edgecolor=color, linewidth=1.8 if not item["local_gate_precommitted"] else 0.8, zorder=3)
        ax.text(time, yi + 0.28, item["local_gate"], fontsize=8.4, color=color, ha="center")
    ax.axhspan(8.5, 14.5, color=blue, alpha=0.055)
    ax.text(max(times), 8.6, "Temporal holdout (6) — generic policy not pre-frozen", ha="right", va="bottom", fontsize=10.5, color=blue)
    ax.set_yticks(y, [f'{item["branch_id"]}  {item["branch"]}' for item in ordered], fontsize=9.4)
    ax.invert_yaxis(); ax.set_xlabel("Recorded outcome time (local filesystem chronology)", color=navy)
    ax.xaxis.set_major_locator(DayLocator()); ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
    ax.grid(axis="x", color="#D9E0E7", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("Branch chronology: 개별 gate의 STOP/ADVANCE 기록", loc="left", fontsize=18, weight=400, color=navy, pad=18)
    ax.text(0.0, -0.095, "● FAIL/STOP 계열   ◆ PASS 후 bounded screen 허가   빈 표식: local script only   음영: 후기 35%", transform=ax.transAxes, fontsize=10.5, color=gray)
    fig.tight_layout(); fig.savefig(FIGURES / "framework_advance_stop_timeline.png", bbox_inches="tight"); plt.close(fig)

    # 3. Only explicitly documented, non-overlapping prevented runs.
    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    labels, values = ["D2R detector\nconfirmation", "K40 YOLOv8s\nexpansion"], [15, 30]
    bars = ax.barh(labels, values, color=[blue, green], height=0.52)
    for bar, value in zip(bars, values):
        ax.text(value + 0.8, bar.get_y() + bar.get_height()/2, f"{value} models", va="center", fontsize=12.5, color=navy, weight=400)
    ax.set_xlim(0, 35); ax.set_xlabel("Explicitly planned detector model runs not executed", color=navy)
    ax.set_title("Documented cost containment: lower bound = 45 model runs", loc="left", fontsize=18, weight=400, color=navy, pad=16)
    ax.text(0.0, -0.17, "Locked final test: actual uses = 0  |  Counterfactual uses avoided = NA", transform=ax.transAxes, fontsize=11.5, color=gray)
    ax.grid(axis="x", color="#D9E0E7", linewidth=0.8); ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGURES / "framework_cost_avoidance.png", bbox_inches="tight"); plt.close(fig)


def validate_internal() -> None:
    assign_partitions()
    ids = [branch["branch_id"] for branch in BRANCHES]
    if len(BRANCHES) != 15 or len(ids) != len(set(ids)):
        raise RuntimeError("Expected 15 unique research branches")
    if sum(branch["chronology_partition"] == "development_chronology" for branch in BRANCHES) != 9:
        raise RuntimeError("65% chronological development partition must contain 9/15 branches")
    if sum(branch["chronology_partition"] == "temporal_holdout" for branch in BRANCHES) != 6:
        raise RuntimeError("Temporal holdout must contain 6/15 branches")
    if any(datetime.strptime(branch["protocol_time"], "%Y-%m-%d %H:%M:%S") >= datetime.strptime(branch["outcome_time"], "%Y-%m-%d %H:%M:%S") for branch in BRANCHES):
        raise RuntimeError("Each local gate artifact must precede its recorded summary/outcome artifact")
    if any(branch["generic_policy_precommitted"] for branch in BRANCHES):
        raise RuntimeError("Generic six-stage policy must not be retroactively marked precommitted")
    if any(not branch["source_file"] or not branch["protocol_source_file"] for branch in BRANCHES):
        raise RuntimeError("Every branch requires protocol and outcome source locators")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit temporal framework identifiability without training or inference.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_internal()

    config = {
        "date": "2026-07-18",
        "requested_design": "65/35 chronological temporal holdout",
        "decision": "C_RETROSPECTIVE_AUDIT_ONLY",
        "evidence_freeze_v2_decision_unchanged": "A_THESIS_REFRAME_STRONGLY_SUPPORTED",
        "predictive_metrics_identifiable": False,
        "reason": "generic policy was not pre-frozen and stopped branches lack counterfactual downstream truth",
        "inferential_unit": "research_branch",
        "explicit_protocol_branches": 11,
        "local_script_only_branches": 4,
        "training_performed": False,
        "inference_performed": False,
        "vlm_calls_performed": False,
        "embedding_extraction_performed": False,
        "selector_implementation_performed": False,
        "fn_screen_performed": False,
        "final_test_used": False,
    }
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_OK", "branches": len(BRANCHES), "development": 9, "holdout": 6, **config}, ensure_ascii=False, indent=2))
        return

    DOCS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    write_csv(DOCS / "framework_branch_timeline_20260718.csv", sorted(BRANCHES, key=lambda item: item["outcome_time"]), FIELDS)
    holdout = holdout_rows()
    write_csv(DOCS / "framework_holdout_confusion_matrix_20260718.csv", holdout, ["branch_id", "branch", "stage", "requested_predicted_decision", "actual_outcome", "correct_early_stop", "false_advance", "possible_false_stop", "reason", "local_gate_action", "local_process_conformance", "prevented_model_runs", "prevented_final_test", "source_file"])
    costs = cost_rows()
    write_csv(DOCS / "framework_cost_avoidance_summary_20260718.csv", costs, ["metric", "scope", "value", "unit", "identifiability", "counting_rule", "source_file"])
    leave_one_out = loo_rows()
    write_csv(DOCS / "framework_leave_one_branch_out_20260718.csv", leave_one_out, ["removed_branch_id", "removed_branch", "claim", "remaining_support_count", "stable_after_removal", "note"])
    write_text(DOCS / "framework_temporal_validation_decision_20260718.md", decision_doc())
    sources = source_registry_rows()
    write_csv(OUT / "source_registry.csv", sources, ["source_id", "path", "exists", "last_write_time", "sha256", "role", "note"])
    save_figures()
    (OUT / "audit_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    commands = [
        r".\.venv\Scripts\python.exe scripts\04_dcal_xai\build_framework_temporal_validation.py --dry-run",
        r".\.venv\Scripts\python.exe scripts\04_dcal_xai\build_framework_temporal_validation.py",
        r".\.venv\Scripts\python.exe scripts\04_dcal_xai\test_framework_temporal_validation.py",
    ]
    write_text(OUT / "executed_commands.txt", "\n".join(commands))
    expected = [
        DOCS / "framework_temporal_validation_decision_20260718.md",
        DOCS / "framework_branch_timeline_20260718.csv",
        DOCS / "framework_holdout_confusion_matrix_20260718.csv",
        DOCS / "framework_cost_avoidance_summary_20260718.csv",
        DOCS / "framework_leave_one_branch_out_20260718.csv",
        FIGURES / "framework_temporal_validation.png",
        FIGURES / "framework_advance_stop_timeline.png",
        FIGURES / "framework_cost_avoidance.png",
        OUT / "source_registry.csv", OUT / "executed_commands.txt", OUT / "generated_file_manifest.txt",
        OUT / "audit_config.json", OUT / "test_results.txt",
        ROOT / "scripts" / "04_dcal_xai" / "build_framework_temporal_validation.py",
        ROOT / "scripts" / "04_dcal_xai" / "test_framework_temporal_validation.py",
    ]
    write_text(OUT / "generated_file_manifest.txt", "\n".join(str(path.relative_to(ROOT)).replace("\\", "/") for path in expected))
    print(json.dumps({"status": "DONE", "decision": config["decision"], "branches": len(BRANCHES), "development": 9, "holdout": 6, "predictive_metrics": "NA", "documented_prevented_model_runs": 45, "final_test_used": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

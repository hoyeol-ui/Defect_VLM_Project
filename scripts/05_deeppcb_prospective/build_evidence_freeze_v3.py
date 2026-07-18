#!/usr/bin/env python3
"""Freeze DeepPCB closure evidence into the thesis package without experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
OUT = ROOT / "runs" / "evidence_freeze_v3_20260718"
ORIGINAL = ROOT / "runs" / "deeppcb_reference_residual_gate" / "prospective_main_20260718"
PHASE_A = ROOT / "runs" / "deeppcb_small_defect_mechanism_audit"
PHASE_B = ROOT / "runs" / "deeppcb_reference_residual_development"
DECISION = "A_THESIS_CORE_STRENGTHENED_BY_PROSPECTIVE_STOP"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def verify_frozen_inputs() -> dict[str, Any]:
    original = read_json(ORIGINAL / "gate_decision.json")
    phase_a = read_json(PHASE_A / "phase_a_decision.json")
    candidate = read_json(PHASE_B / "frozen_candidate.json")
    methods = pd.read_csv(PHASE_B / "method_comparison.csv", encoding="utf-8-sig")
    bins = pd.read_csv(PHASE_A / "size_bin_enrichment.csv", encoding="utf-8-sig")
    pooled = bins.groupby("size_bin", sort=False)[["pool_boxes", "query_boxes", "expected_random_boxes"]].sum()
    main_bin = pooled.loc["577-1024"]
    main_enrichment = float(main_bin["query_boxes"] / main_bin["expected_random_boxes"])
    m2 = methods.loc[methods["method"] == "M2_SSIM"].iloc[0]
    checks = {
        "original_fail_stop": original["decision"] == "FAIL_STOP" and original["authorization"] == "STOP",
        "total_enrichment_exact": abs(original["aggregate"]["instance_enrichment_vs_random"]["mean"] - 1.107693434196111) < 1e-12,
        "small_enrichment_exact": abs(original["aggregate"]["small_instance_enrichment_vs_random"]["mean"] - 1.3075905008139574) < 1e-12,
        "phase_a_a2": phase_a["phase_a_decision"] == "A2_MECHANISM_AMBIGUOUS",
        "dominant_group_45_79": abs(phase_a["dominance"]["max_group_positive_small_excess_share"] - 0.4579125029068911) < 1e-12,
        "phase_b_no_candidate": candidate["result"] == "NO_CANDIDATE",
        "m2_values": abs(float(m2["small_recall"]) - 0.17857142857142858) < 1e-12 and abs(float(m2["false_positives_per_image"]) - 0.972972972972973) < 1e-12 and abs(float(m2["brightness_recall_retention"]) - 0.2) < 1e-12,
        "no_training": original["training_performed"] is False and phase_a["training_performed"] is False,
        "no_inference": original["detector_inference_performed"] is False and phase_a["detector_inference_performed"] is False,
        "official_final_zero": original["official_test_used"] is False and original["final_test_used"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen DeepPCB evidence mismatch: {checks}")
    return {
        "original": original,
        "phase_a": phase_a,
        "candidate": candidate,
        "methods": methods,
        "main_bin": main_bin,
        "main_enrichment": main_enrichment,
        "checks": checks,
    }


def build_hypothesis_matrix() -> None:
    rows, fields = read_csv(DOCS / "hypothesis_transition_matrix_20260718.csv")
    additions = [
        {
            "hypothesis_id": "H28", "research_phase": "prospective_external_authorization", "original_hypothesis": "Aligned reference residual은 DeepPCB top20%에서 total bbox richness를 1.20 이상 농축한다.",
            "original_claim_strength": "prospective primary", "original_source": "DeepPCB frozen prospective protocol", "dataset": "DeepPCB trainval eligible six groups", "method": "frozen absolute-difference residual richness", "primary_metric": "total bbox instance enrichment", "frozen_gate": "mean>=1.20; CI low>1; >=5/6 groups>1", "observed_result": "1.107693; CI [1.048980,1.171691]; primary threshold 1.20 FAIL", "evidence_status": "confirmatory_negative", "failure_or_support_mechanism": "positive relation without prespecified operational effect size", "revised_research_question": "Secondary small-box signal이 spatially grounded하고 group-independent한가?", "allowed_claim": "A prospectively frozen external branch failed its primary enrichment gate and stopped before detector training.", "prohibited_claim": "DeepPCB reference-residual selector succeeded.", "manuscript_section": "6.4 Prospective External Authorization Case: DeepPCB", "source_file": "runs/deeppcb_reference_residual_gate/prospective_main_20260718/gate_decision.json",
        },
        {
            "hypothesis_id": "H29", "research_phase": "prospective_secondary_audit", "original_hypothesis": "Primary FAIL branch의 small bbox enrichment 1.307591은 일반 small-defect triage utility를 뜻한다.",
            "original_claim_strength": "exploratory secondary", "original_source": "DeepPCB parent gate secondary endpoint", "dataset": "DeepPCB eligible six groups", "method": "frozen selected set post-hoc GT audit", "primary_metric": "small bbox enrichment and small-share uplift", "frozen_gate": "no parent PASS; mechanism audit only", "observed_result": "small enrichment 1.307591; small-share uplift 1.182816 [1.081407,1.371080]; main enriched area 577-1024 px²", "evidence_status": "exploratory_only", "failure_or_support_mechanism": "size-specific secondary signal, not a primary authorization endpoint", "revised_research_question": "Spatial mechanism and development robustness survive frozen audits?", "allowed_claim": "The exploratory signal was concentrated in bbox area 577-1024 px² and was audited separately.", "prohibited_claim": "Tiny defects generally improved or the parent gate passed.", "manuscript_section": "6.4 Prospective External Authorization Case: DeepPCB", "source_file": "runs/deeppcb_small_defect_mechanism_audit/size_bin_enrichment.csv",
        },
        {
            "hypothesis_id": "H30", "research_phase": "mechanism_audit", "original_hypothesis": "Small-box signal is group-independent and mechanism-supported.",
            "original_claim_strength": "post-hoc mechanism gate", "original_source": "frozen mechanism audit policy", "dataset": "DeepPCB eligible six groups", "method": "read-only spatial/confound/group audit", "primary_metric": "small-share CI, spatial ratio, group/class dominance", "frozen_gate": "all seven mechanism checks", "observed_result": "A2_MECHANISM_AMBIGUOUS; dominant group share 45.79% exceeded <40% rule", "evidence_status": "exploratory_only", "failure_or_support_mechanism": "one-group concentration prevented mechanism-supported decision", "revised_research_question": "Does a development-only literature residual yield a robust external candidate?", "allowed_claim": "Several spatial checks were positive, but the complete mechanism gate did not pass.", "prohibited_claim": "Local residual to small-defect mechanism was confirmed.", "manuscript_section": "6.4 Prospective External Authorization Case: DeepPCB", "source_file": "runs/deeppcb_small_defect_mechanism_audit/phase_a_decision.json",
        },
        {
            "hypothesis_id": "H31", "research_phase": "development_candidate", "original_hypothesis": "AbsDiff, SSIM, or fused residual yields a frozen candidate for independent confirmation.",
            "original_claim_strength": "development-only", "original_source": "group92000 development policy v2", "dataset": "DeepPCB group92000 trainval 111", "method": "M1 AbsDiff; M2 SSIM; M3 fused residual", "primary_metric": "small recall at <=1 FP/image plus robustness and class safety", "frozen_gate": "small recall>=0.50; brightness/translation retention>=0.80; coverage>=5", "observed_result": "NO_CANDIDATE; M2 small recall 0.179, FP/image 0.973, brightness retention 0.20; M1/M3 no nonzero <=1-FPPI point", "evidence_status": "confirmatory_negative", "failure_or_support_mechanism": "localization/false-alarm/robustness trade-off", "revised_research_question": "Branch closed; no external confirmation or detector screen authorized.", "allowed_claim": "No tested development candidate met the frozen external-confirmation adequacy criteria.", "prohibited_claim": "Literature reference methods are generally invalid.", "manuscript_section": "6.4 Prospective External Authorization Case: DeepPCB", "source_file": "runs/deeppcb_reference_residual_development/method_comparison.csv",
        },
    ]
    existing = {row["hypothesis_id"] for row in rows}
    rows.extend(item for item in additions if item["hypothesis_id"] not in existing)
    write_csv(DOCS / "hypothesis_transition_matrix_v3_20260718.csv", rows, fields)


def build_mechanism_matrix() -> None:
    rows, fields = read_csv(DOCS / "acquisition_mechanism_matrix_20260718.csv")
    row = {field: "" for field in fields}
    row.update({
        "dataset": "DeepPCB", "strategy_or_signal": "Frozen aligned reference-residual richness", "signal_validity": "G1 score-instance Spearman mean 0.412174; positive 6/6", "discovery_gain": "total bbox enrichment 1.107693 < frozen 1.20; small secondary 1.307591", "composition_change": "class coverage delta 0; max-class-share delta -0.002072", "diversification_or_concentration": "small-box excess group concentration 45.79%; main area 577-1024 px²", "category_safety": "class coverage preserved in parent audit", "source_or_session_safety": "six design groups; one-group dominance rule failed", "acquisition_reproducibility": "prospective six-group parent selection; mechanism LOO positive but A2", "downstream_overall_utility": "not authorized; detector training 0", "downstream_rare_utility": "not tested", "operational_utility": "primary G2 FAIL; Phase B NO_CANDIDATE", "dominant_mechanism": "positive_secondary_signal_without_robust_candidate", "final_decision": "PROSPECTIVE_STOP / BRANCH_CLOSED", "evidence_level": "prospective primary fail + post-hoc mechanism audit + development-only negative", "source_file": "runs/deeppcb_reference_residual_gate/prospective_main_20260718/gate_decision.json; runs/deeppcb_small_defect_mechanism_audit/phase_a_decision.json; runs/deeppcb_reference_residual_development/frozen_candidate.json",
    })
    rows.append(row)
    write_csv(DOCS / "acquisition_mechanism_matrix_v3_20260718.csv", rows, fields)


def ledger_row(evidence_id: str, **values: Any) -> dict[str, Any]:
    return {"evidence_id": evidence_id, **values}


def build_ledger(data: dict[str, Any]) -> None:
    rows, fields = read_csv(ROOT / "runs" / "evidence_freeze_v2_20260718" / "research_evidence_ledger.csv")
    original = data["original"]
    phase_a = data["phase_a"]
    m2 = data["methods"].loc[data["methods"]["method"] == "M2_SSIM"].iloc[0]
    additions = [
        ledger_row("E035", research_question="Was the external protocol and selection frozen before GT audit?", dataset="DeepPCB", protocol="prospective aligned-reference selection-only", strategy="frozen residual richness", metric="selection SHA-256", value=original["selection_sha256"], uncertainty="exact cryptographic hash", gate="hash before annotation join", result="PASS", evidence_status="process_supported", inferential_unit="one prospective branch", allowed_claim="Protocol and selection were frozen before post-hoc GT audit.", prohibited_claim="The hash predicts downstream success.", source_file="runs/deeppcb_reference_residual_gate/prospective_main_20260718/gate_decision.json", source_row_or_key="selection_sha256; protocol_sha256; script_sha256", verified=True),
        ledger_row("E036", research_question="Does residual score relate to bbox count?", dataset="DeepPCB", protocol="eligible six-group parent gate", strategy="frozen residual richness", metric="mean group Spearman score vs instances", value=f"{original['aggregate']['spearman_score_vs_instances']['mean']:.12f}", uncertainty=f"group bootstrap 95% CI [{original['aggregate']['spearman_score_vs_instances']['ci95_low']:.6f},{original['aggregate']['spearman_score_vs_instances']['ci95_high']:.6f}]", gate="mean>=0.30; positive in >=5/6", result="PASS", evidence_status="prospective_secondary_supported", inferential_unit="six predefined DeepPCB design groups", allowed_claim="The frozen score had a positive relation with bbox count in this pool.", prohibited_claim="The relation establishes annotation or detector utility.", source_file="runs/deeppcb_reference_residual_gate/prospective_main_20260718/gate_decision.json", source_row_or_key="aggregate.spearman_score_vs_instances; checks.g1_*", verified=True),
        ledger_row("E037", research_question="Does the top20% enrich total bbox instances enough?", dataset="DeepPCB", protocol="eligible six-group parent gate", strategy="frozen residual richness", metric="total bbox instance enrichment", value=f"{original['aggregate']['instance_enrichment_vs_random']['mean']:.12f}", uncertainty=f"group bootstrap 95% CI [{original['aggregate']['instance_enrichment_vs_random']['ci95_low']:.6f},{original['aggregate']['instance_enrichment_vs_random']['ci95_high']:.6f}]", gate="mean>=1.20", result="FAIL_STOP", evidence_status="confirmatory_negative", inferential_unit="six predefined DeepPCB design groups", allowed_claim="The primary effect-size gate failed and detector authorization stopped.", prohibited_claim="DeepPCB selector succeeded.", source_file="runs/deeppcb_reference_residual_gate/prospective_main_20260718/gate_decision.json", source_row_or_key="aggregate.instance_enrichment_vs_random; checks.g2_instance_enrichment_mean_ge_1_20", verified=True),
        ledger_row("E038", research_question="Was there an exploratory small-box enrichment?", dataset="DeepPCB", protocol="parent post-hoc GT audit", strategy="frozen residual richness", metric="bbox area<=1024 enrichment", value=f"{original['aggregate']['small_instance_enrichment_vs_random']['mean']:.12f}", uncertainty=f"group bootstrap 95% CI [{original['aggregate']['small_instance_enrichment_vs_random']['ci95_low']:.6f},{original['aggregate']['small_instance_enrichment_vs_random']['ci95_high']:.6f}]", gate="secondary only; cannot pass parent", result="EXPLORATORY", evidence_status="exploratory_only", inferential_unit="six predefined groups", allowed_claim="An exploratory secondary small-box enrichment was observed.", prohibited_claim="1.307591 is a confirmatory PASS.", source_file="runs/deeppcb_reference_residual_gate/prospective_main_20260718/gate_decision.json", source_row_or_key="aggregate.small_instance_enrichment_vs_random", verified=True),
        ledger_row("E039", research_question="Is the selected bbox mix size-selective?", dataset="DeepPCB", protocol="frozen mechanism audit", strategy="unchanged frozen selection", metric="small-share uplift", value=f"{phase_a['small_share_uplift']:.12f}", uncertainty=f"group bootstrap 95% CI [{phase_a['small_share_ci95_low']:.6f},{phase_a['small_share_ci95_high']:.6f}]", gate="CI low>1 plus six additional checks", result="PARTIAL_ONLY", evidence_status="exploratory_only", inferential_unit="six groups", allowed_claim="Small-share increased under the frozen selected set.", prohibited_claim="General small-defect utility was confirmed.", source_file="runs/deeppcb_small_defect_mechanism_audit/phase_a_decision.json", source_row_or_key="small_share_uplift; small_share_ci95_*", verified=True),
        ledger_row("E040", research_question="Is the small signal group-independent?", dataset="DeepPCB", protocol="frozen mechanism audit", strategy="group dominance audit", metric="max positive small-excess group share", value=f"{phase_a['dominance']['max_group_positive_small_excess_share']:.12f}", uncertainty="descriptive decomposition across six groups", gate="<0.40", result="FAIL", evidence_status="exploratory_only", inferential_unit="six groups", allowed_claim="One group explained 45.79% of positive excess, preventing A1 support.", prohibited_claim="The mechanism is group-independent.", source_file="runs/deeppcb_small_defect_mechanism_audit/phase_a_decision.json", source_row_or_key="dominance.max_group_positive_small_excess_share; checks.single_group_*", verified=True),
        ledger_row("E041", research_question="Which bbox size range drove the secondary signal?", dataset="DeepPCB", protocol="frozen mechanism audit", strategy="predefined size bins", metric="577-1024 px² bbox enrichment", value=f"{data['main_enrichment']:.12f}", uncertainty=f"selected={int(data['main_bin']['query_boxes'])}; expected_random={data['main_bin']['expected_random_boxes']:.6f}", gate="descriptive post-hoc bin", result="MAIN_ENRICHED_RANGE", evidence_status="exploratory_only", inferential_unit="pooled eligible six groups", allowed_claim="The main enriched small-size range was 577-1024 px².", prohibited_claim="All tiny defects were enriched.", source_file="runs/deeppcb_small_defect_mechanism_audit/size_bin_enrichment.csv", source_row_or_key="size_bin=577-1024; sum query_boxes / sum expected_random_boxes", verified=True),
        ledger_row("E042", research_question="Did the complete mechanism gate pass?", dataset="DeepPCB", protocol="frozen mechanism audit", strategy="spatial/confound/dominance checks", metric="Phase A decision", value=phase_a["phase_a_decision"], uncertainty="all seven frozen checks", gate="all checks required for A1", result="A2", evidence_status="confirmatory_negative", inferential_unit="one read-only audit", allowed_claim="Mechanism evidence remained ambiguous.", prohibited_claim="Spatial mechanism was confirmed.", source_file="runs/deeppcb_small_defect_mechanism_audit/phase_a_decision.json", source_row_or_key="phase_a_decision; checks", verified=True),
        ledger_row("E043", research_question="Did SSIM produce an adequate development candidate?", dataset="DeepPCB", protocol="group92000 development-only benchmark", strategy="M2 SSIM", metric="small recall; FP/image; brightness retention", value=f"{float(m2['small_recall']):.6f}; {float(m2['false_positives_per_image']):.6f}; {float(m2['brightness_recall_retention']):.6f}", uncertainty="development-only point estimate on 111 trainval images", gate="small recall>=0.50; FP/image<=1; brightness retention>=0.80", result="FAIL", evidence_status="confirmatory_negative", inferential_unit="group92000 trainval 111", allowed_claim="M2 met the FP budget but failed recall and brightness robustness.", prohibited_claim="SSIM methods are generally invalid.", source_file="runs/deeppcb_reference_residual_development/method_comparison.csv", source_row_or_key="method=M2_SSIM", verified=True),
        ledger_row("E044", research_question="Was any frozen candidate authorized for external confirmation?", dataset="DeepPCB", protocol="group92000 development-only benchmark", strategy="M1/M2/M3", metric="candidate decision", value=data["candidate"]["result"], uncertainty="exact frozen development decision", gate="all adequacy checks", result="NO_CANDIDATE", evidence_status="confirmatory_negative", inferential_unit="one development group", allowed_claim="No tested candidate was authorized for external confirmation.", prohibited_claim="External reference-residual utility was validated.", source_file="runs/deeppcb_reference_residual_development/frozen_candidate.json", source_row_or_key="result", verified=True),
        ledger_row("E045", research_question="Did the workflow authorize detector expansion?", dataset="DeepPCB", protocol="parent+mechanism+development chain", strategy="authorization workflow", metric="detector training/inference count", value="0/0", uncertainty="actual observed actions", gate="candidate required before detector screen", result="STOP", evidence_status="process_supported", inferential_unit="one prospective external branch", allowed_claim="The branch stopped before detector training and inference.", prohibited_claim="A counterfactual detector failure or improvement is known.", source_file="runs/deeppcb_reference_residual_development/frozen_candidate.json", source_row_or_key="result=NO_CANDIDATE; paired with parent/Phase A protected-state flags", verified=True),
        ledger_row("E046", research_question="Was the official/final test consumed?", dataset="DeepPCB", protocol="protected split lock", strategy="official-test lock", metric="official/final actual use", value="0/0", uncertainty="actual observed use", gate="must remain zero", result="PASS", evidence_status="process_supported", inferential_unit="one protected split", allowed_claim="Official and final test actual use remained zero.", prohibited_claim="Counterfactual final-test performance or avoided-use count is known.", source_file="runs/deeppcb_reference_residual_gate/prospective_main_20260718/gate_decision.json", source_row_or_key="official_test_used=false; final_test_used=false; split_lock", verified=True),
    ]
    existing = {row["evidence_id"] for row in rows}
    rows.extend(item for item in additions if item["evidence_id"] not in existing)
    write_csv(OUT / "research_evidence_ledger.csv", rows, fields)


def build_claim_boundary() -> None:
    text = """# Thesis claim boundary v3

Date: 2026-07-18  
Evidence Freeze v3 decision: **A. THESIS_CORE_STRENGTHENED_BY_PROSPECTIVE_STOP**

## Supported

- 산업 결함 AL 후보 신호의 `signal-discovery-safety-reproducibility-learning-operational` 단계는 서로 대체할 수 없다.
- GC10, MPDD, VisA의 fixed-pool discovery gain은 composition safety 또는 downstream utility와 분리됐다.
- DeepPCB에서 외부 데이터 protocol, score, query fraction, selection hash와 primary gate가 GT audit 전에 고정됐다.
- DeepPCB primary total-bbox gate가 실패한 뒤 detector screen이 중단됐다.
- Positive secondary small-box signal은 parent PASS로 승격되지 않고 mechanism audit과 development-only benchmark로 분리됐다.
- DeepPCB detector training/inference actual count는 0이고 official/final test actual use는 0이다.
- 이 사례는 workflow의 prospective authorization conformance와 claim-boundary enforcement를 강화한다.

## Exploratory only

- DeepPCB bbox area 577-1024 px² 구간과 frozen residual selection의 관계.
- DeepPCB small-share uplift 1.1828, group-bootstrap 95% CI [1.0814,1.3711].
- Parent secondary small enrichment 1.307591.
- V2.3 FN 1.379693, rare-FN 1.785910, confidence rare-FN 2.683022.

## Rejected or closed

- DeepPCB total bbox-richness utility: 1.107693 < frozen 1.20.
- General small-defect annotation triage: Phase A A2_MECHANISM_AMBIGUOUS.
- Independent reference-residual candidate: Phase B NO_CANDIDATE.
- DeepPCB detector improvement: not authorized and not tested.
- Universal VLM/DINO/detector-uncertainty selector superiority.

## Not identifiable

- Workflow의 prospective predictive accuracy, early-stop recall, false-advance rate와 good-selector sensitivity.
- STOP branch의 counterfactual detector outcome.
- Independent production generalization.
- Actual annotation-time or monetary-cost reduction.

## DeepPCB 금지 표현

- DeepPCB를 selector 성공 사례로 표현하지 않는다.
- 1.307591을 confirmatory PASS로 표현하지 않는다.
- bbox area 577-1024 px² 결과를 tiny defect 전반의 효과로 표현하지 않는다.
- M1/M2/M3 결과를 literature methods의 일반적 무효성으로 표현하지 않는다.
- DeepPCB STOP 사례를 predictive screening accuracy로 표현하지 않는다.
- 문서화되지 않은 prevented model-run 수를 추가하지 않는다.
- DeepPCB official test, detector 학습 또는 score/threshold rescue를 재개하지 않는다.

## 논문 본체 방화벽

1. 논문 본체의 성립 조건은 future positive selector가 아니다.
2. DeepPCB 결과가 A1 또는 candidate PASS였어도 external utility는 별도 독립 확인이 필요했다.
3. 현재 A2와 NO_CANDIDATE는 optional extension을 닫지만 retrospective core evidence를 소거하지 않는다.
4. 이번 사례가 강화하는 것은 success prediction이 아니라 `freeze -> audit -> authorization STOP -> locked test preservation`의 실제 준수다.
5. 교수님이 positive algorithm을 필수로 요구하면 해당 연구는 별도의 supervised/reference-inspection 주제와 split으로 분리한다.
"""
    write_text(DOCS / "thesis_claim_boundary_v3_20260718.md", text)


def build_closure_doc(data: dict[str, Any]) -> None:
    text = f"""# DeepPCB branch closure

Date: 2026-07-18  
Status: **CLOSED - NO RESCUE AUTHORIZED**

## Frozen chain

| Stage | Frozen result | Decision |
|---|---|---|
| Prospective parent G1 | Spearman 0.412174; positive 6/6 | relation PASS |
| Prospective parent G2 | total bbox enrichment 1.107693 < 1.20 | **FAIL_STOP** |
| Secondary | small enrichment 1.307591 | exploratory only |
| Mechanism audit | small-share 1.1828; dominant group 45.79%; main range 577-1024 px² | **A2_MECHANISM_AMBIGUOUS** |
| Development benchmark | M2 small recall 0.179; FP/image 0.973; brightness retention 0.20 | **NO_CANDIDATE** |
| External confirmation | candidate absent | **NOT AUTHORIZED** |
| Detector screen | candidate absent | **NOT AUTHORIZED** |
| Protected tests | detector 0; official/final actual use 0 | locked |

## Closure interpretation

이 branch는 방법론 성공 사례가 아니다. G1의 양의 relation과 small secondary signal은 실제였지만 primary effect size, group-independence, development robustness가 동시에 충족되지 않았다. 따라서 branch를 threshold/score rescue 없이 닫는다.

논문에서의 역할은 **Prospective External Authorization STOP Case**다. 외부 데이터에서 사전 freeze, selection hash, primary FAIL 수용, secondary 분리 감사, development NO_CANDIDATE, detector/official-test 미소비가 한 경로로 기록됐다.

## Allowed next action

- Evidence Freeze v3, 논문, figure와 지도교수 보고자료에 통합.
- 기존 run과 hash를 read-only archive로 유지.

## Prohibited next action

- threshold 32, min component area 9, 0.5/0.5 score 또는 query fraction rescue.
- eligible six groups, group92000, official test를 이용한 추가 candidate search.
- detector 학습/inference 또는 official-test annotation 접근.
- `1.307591`, `1.1828` 또는 spatial ratio를 selector 성공으로 승격.

## Source locators

- Parent decision: `runs/deeppcb_reference_residual_gate/prospective_main_20260718/gate_decision.json`
- Selection hash: `runs/deeppcb_reference_residual_gate/prospective_main_20260718/frozen_selection_sha256.txt`
- Phase A: `runs/deeppcb_small_defect_mechanism_audit/phase_a_decision.json`
- Size bins: `runs/deeppcb_small_defect_mechanism_audit/size_bin_enrichment.csv`
- Phase B: `runs/deeppcb_reference_residual_development/method_comparison.csv`
- Candidate decision: `runs/deeppcb_reference_residual_development/frozen_candidate.json`
"""
    write_text(DOCS / "deeppcb_branch_closure_20260718.md", text)


def build_evolution_v3() -> None:
    old = (DOCS / "research_evolution_and_evidence_freeze_v2_20260718.md").read_text(encoding="utf-8")
    prefix = old.split("## 9. Evidence Freeze v2", 1)[0].rstrip()
    prefix = prefix.replace("# Research evolution and Evidence Freeze v2", "# Research evolution and Evidence Freeze v3")
    prefix = prefix.replace("Final synthesis decision: **A. THESIS_REFRAME_STRONGLY_SUPPORTED**", "Final synthesis decision: **A. THESIS_CORE_STRENGTHENED_BY_PROSPECTIVE_STOP**")
    prefix = prefix.replace("This rating applies only to the evidence-based thesis reframe. It does not revive the rejected selector-superiority hypothesis.", "This rating reflects added prospective authorization evidence. It does not revive the rejected selector-superiority hypothesis or establish predictive screening accuracy.")
    addition = """

## 9. Prospective external authorization case: DeepPCB

DeepPCB는 defect-free template과 tested image가 정렬된 paired-reference 조건이므로, global DINO/VLM보다 local residual이 결함 annotation triage에 가까울 수 있다는 새로운 외부 조건이었다. Score, threshold 32, component minimum area 9, query fraction 20%, eligible six groups와 reserved group92000을 실행 전에 고정하고 selection CSV를 SHA-256으로 봉인한 뒤 GT를 결합했다(E035).

G1에서는 score와 bbox instance count의 mean group Spearman이 0.412174였고 6/6 group에서 양의 방향이었다(E036). 그러나 primary G2 total bbox enrichment는 1.107693으로 frozen 1.20에 못 미쳐 `FAIL_STOP`이었다(E037). Small bbox enrichment 1.307591은 secondary exploratory signal로만 분리했다(E038).

Read-only mechanism audit에서는 small-share uplift 1.1828, 95% CI [1.0814,1.3711], size-selectivity >1 in 5/6, spatial ratio >1 in 6/6이 관측됐다. 그러나 positive small-box excess의 45.79%가 한 group에서 발생하여 frozen <40% rule을 실패했고 Phase A는 `A2_MECHANISM_AMBIGUOUS`였다(E039-E042). Size-bin audit에서 <=256 px² box는 없었고 257-576 px²는 selected 0이었으며, 주요 농축 구간은 577-1024 px²였다. 따라서 tiny defect 전반의 효과로 표현하지 않는다.

Reserved development group92000의 M1 AbsDiff, M2 SSIM, M3 fused residual 비교에서도 candidate가 없었다. M2는 FP/image 0.973에서 small recall 0.179, brightness retention 0.20이었고 M1/M3는 <=1 FP/image에서 nonzero operating point를 유지하지 못했다(E043-E044). External confirmation과 detector screen은 미승인됐고 detector training/inference와 official/final test actual use는 모두 0이었다(E045-E046).

이 사례는 method success가 아니라 **research stopping success**다. Retrospective core가 실제 외부 branch에서 어떻게 적용되는지 보여주되, 한 STOP 사례로 predictive accuracy를 계산할 수 없다는 경계는 유지한다.

## 10. Evidence Freeze v3

### 최종 판정

**A. THESIS_CORE_STRENGTHENED_BY_PROSPECTIVE_STOP**

강화된 것은 selector 성능이 아니라 다음 세 가지다.

1. 외부 데이터에서 protocol과 selection을 결과 전에 고정한 prospective process evidence.
2. Primary FAIL 뒤 positive secondary signal을 parent success로 승격하지 않은 claim-boundary enforcement.
3. Mechanism ambiguity와 development NO_CANDIDATE 뒤 detector/official-test 소비를 중단한 authorization conformance.

### 유지 주장

- Fixed benchmark discovery, composition safety, acquisition reproducibility와 learner utility는 분리된다.
- Selection PASS는 bounded measurement authorization이지 success prediction이 아니다.
- DeepPCB primary FAIL은 detector screen을 중단했고 official/final test actual use는 0이었다.
- 명시적으로 문서화된 기존 minimum 45 planned detector runs stopped 수치는 그대로 유지하며 DeepPCB에 임의의 prevented-run 수를 더하지 않는다.

### 폐기 또는 종료 주장

- DeepPCB total bbox-richness utility, general small-defect triage, independent residual candidate와 detector improvement.
- Universal selector superiority, predictive screening accuracy, independent production generalization과 annotation-cost reduction.

### Protected status

- DeepPCB exact branch closed: **True**.
- Score/threshold rescue authorized: **False**.
- External confirmation authorized: **False**.
- Detector screen authorized: **False**.
- New training/inference in Evidence Freeze v3: **False**.
- Official/final test actual use: **0/0**.
"""
    write_text(DOCS / "research_evolution_and_evidence_freeze_v3_20260718.md", prefix + addition)


def build_advisor_brief() -> None:
    text = """# 지도교수 결정 보고: DeepPCB closure와 Evidence Freeze v3

## 요청드리는 결정

> Positive selector 성능이 아니라, 후보 신호가 discovery·safety·learning utility로 번역되는지 단계적으로 검증하고 불충분한 branch의 학습·official test·주장을 중단하는 empirical workflow를 학위논문의 중심 기여로 인정할 수 있는지 결정 부탁드립니다.

## 이번에 새로 추가된 근거

기존 논문의 약점은 generic workflow가 모든 주요 실험 종료 후 정식화됐다는 점이었습니다. DeepPCB에서는 외부 paired-reference 데이터를 대상으로 protocol, score, threshold, query fraction, eligible groups와 selection hash를 먼저 고정했습니다. G1 relation은 양성이었지만 primary G2 total bbox enrichment가 1.107693으로 frozen 1.20을 넘지 못해 `FAIL_STOP`을 수용했습니다.

이후 small enrichment 1.307591을 성공으로 바꾸지 않고 read-only mechanism audit으로 분리했습니다. Small-share uplift는 1.1828이었지만 positive excess의 45.79%가 한 group에 집중되어 `A2_MECHANISM_AMBIGUOUS`였습니다. Reserved development group에서 AbsDiff, SSIM, fused residual을 비교했지만 `NO_CANDIDATE`였습니다. 이에 external confirmation과 detector screen을 승인하지 않았고 detector training/inference와 official/final test use는 0으로 유지했습니다.

## 왜 thesis core가 강화되는가

1. **Prospective process evidence**: 외부 branch에서 결과 전 protocol·selection freeze가 실제 적용됐습니다.
2. **Claim-boundary enforcement**: 양의 secondary signal을 parent PASS나 tiny-defect generalization으로 승격하지 않았습니다.
3. **Authorization evidence**: mechanism과 development gate 실패 후 고비용 detector 단계가 실제로 중단됐습니다.
4. **Reproducibility**: selection hash, source locator, frozen decision, code와 integrity tests가 남아 있습니다.

이는 predictive screening accuracy를 증명하지 않습니다. 한 STOP case만으로 early-stop recall이나 good-method false-stop sensitivity를 계산할 수 없습니다. 강화된 기여는 success prediction이 아니라 prospective conformance와 cost containment입니다.

## 최종 연구 판정

**A. THESIS_CORE_STRENGTHENED_BY_PROSPECTIVE_STOP**

이 판정은 positive 성능을 의미하지 않습니다. 외부 데이터에서도 불충분한 positive signal을 detector-success 주장으로 확대하지 않는 workflow가 실제로 작동했다는 뜻입니다.

## 교수님께 보고할 1분 설명

“교수님, 기존 결과를 사후적으로만 정리했다는 약점을 보완하기 위해 DeepPCB 외부 paired-reference branch를 score와 gate를 먼저 고정해 실행했습니다. Score와 bbox 수의 관계는 양성이었지만 primary enrichment가 1.108로 기준 1.20을 넘지 못해 학습을 중단했습니다. Small-box 1.308배라는 secondary signal도 성공으로 바꾸지 않고 기전 감사를 했는데 한 group 의존성이 커 ambiguous였고, 별도 development group의 세 residual 방법도 외부 확인 후보를 만들지 못했습니다. 그래서 detector와 official test를 전혀 쓰지 않고 branch를 닫았습니다. 성능 좋은 selector를 만든 결과는 아니지만, 양의 일부 수치를 성공으로 과장하지 않고 외부 branch의 다음 단계 권한을 통제한 최초의 prospective 사례가 생겼습니다. 이 validity-gated empirical workflow와 failure-condition evidence를 학위논문의 중심 기여로 인정할 수 있는지 결정 부탁드립니다.”

## 승인 후 경로

- 승인 시: 새 selector 실험 없이 Evidence Freeze v3를 기준으로 논문 작성과 발표자료 정리에 들어갑니다.
- Positive algorithm이 필수라면: 현재 AL 논문에 사후 score search를 추가하지 않고, supervised reference-inspection/backbone 연구를 별도 주제·split·protocol로 분리합니다.
- DeepPCB branch는 어느 경우에도 재개하지 않습니다.
"""
    write_text(DOCS / "advisor_decision_brief_deeppcb_closure_20260718.md", text)


def build_mini_paper_v2() -> None:
    path = DOCS / "mini_paper_validity_gated_industrial_al_20260718.md"
    paper = path.read_text(encoding="utf-8")
    paper = paper.replace("# 산업 결함 능동학습 후보 신호의 단계적 타당성 평가: 발견-안전성-학습 효용 간극 분석", "# 산업 결함 능동학습 후보 신호의 단계적 타당성 평가: 발견-안전성-학습 효용 간극과 Prospective STOP Case")
    paper = paper.replace("![전체 연구 아키텍처](figures/full_page_validity_gated_architecture.png)", "![전체 연구 아키텍처 v2](figures/full_page_validity_gated_architecture_v2.png)")
    marker = "Temporal audit의 predictive confusion matrix는 generic policy의 사전동결 부재와 STOP branch의 counterfactual 부재 때문에 NOT IDENTIFIABLE이었다. 반면 명시적으로 계획된 detector model run 최소 45개를 실행하지 않았고 locked final test의 actual use는 0회였다."
    replacement = marker + " DeepPCB 외부 prospective branch에서는 G1 relation이 양성이었지만 primary total-bbox enrichment 1.107693이 frozen 1.20에 미달해 FAIL_STOP이 됐다. Secondary small signal은 mechanism audit A2와 development NO_CANDIDATE로 종료됐으며 detector training/inference와 official/final test actual use는 모두 0이었다."
    paper = paper.replace(marker, replacement)
    dataset_row = "| VisA | anomaly discovery, category collapse audit | 200 paired leave-20 perturbations | 미수행(safety rejected) | category를 independent capture session으로 해석 |"
    paper = paper.replace(dataset_row, dataset_row + "\n| DeepPCB | prospective aligned-reference authorization case | six eligible groups; group92000 development-only | 미수행(primary FAIL, NO_CANDIDATE) | secondary small signal을 selector success로 해석 |")
    deep_section = """

## 6.4 Prospective External Authorization Case: DeepPCB

DeepPCB는 aligned defect-free template과 tested image가 존재해 global representation보다 local residual이 task structure에 가까울 가능성이 있는 외부 조건이었다. Threshold 32, minimum component area 9, residual-area/component rank 0.5/0.5, query fraction 20%, eligible six groups와 reserved group92000을 사전에 고정하고 selection CSV를 SHA-256으로 봉인한 뒤 GT를 결합했다(E035).

G1 score-instance relation은 mean Spearman 0.412174였고 6/6 group에서 양의 방향으로 PASS했다(E036). 그러나 G2 primary total bbox enrichment는 1.107693, 95% CI [1.048980,1.171691]로 frozen 1.20을 넘지 못해 `FAIL_STOP`이었다(E037). Small enrichment 1.307591은 exploratory secondary로만 유지했다(E038).

Read-only audit에서 small-share uplift는 1.1828, 95% CI [1.0814,1.3711]였고 spatial ratio는 6/6 group에서 1보다 컸다. 그러나 한 group이 positive small-box excess의 45.79%를 설명해 <40% rule을 실패했고 Phase A는 `A2_MECHANISM_AMBIGUOUS`였다(E039-E042). Main enriched range는 bbox area 577-1024 px²였으며 <=256 px² box는 존재하지 않고 257-576 px² selected box는 0이었다. 따라서 tiny defect 전반의 효과로 해석하지 않는다.

Reserved development group92000의 M1 AbsDiff, M2 SSIM, M3 fused residual 비교 결과는 `NO_CANDIDATE`였다. M2는 FP/image 0.973에서 small recall 0.179, brightness retention 0.20으로 frozen adequacy를 통과하지 못했고 M1/M3도 FP budget 안의 nonzero point를 유지하지 못했다(E043-E044). External confirmation과 detector screen은 승인되지 않았으며 detector training/inference와 official/final test actual use는 0이었다(E045-E046).

![DeepPCB prospective STOP case](figures/deeppcb_prospective_stop_case.png)

**그림 4. DeepPCB prospective STOP case.** 외부 protocol freeze부터 parent primary FAIL, exploratory signal의 분리 감사, development NO_CANDIDATE와 detector/official-test STOP까지의 authorization chain을 보인다. 이는 selector 성공 사례나 predictive accuracy가 아니라 process conformance 사례다.
"""
    paper = paper.replace("\n<!-- PAGEBREAK -->\n\n# 7. Temporal Audit", deep_section + "\n\n<!-- PAGEBREAK -->\n\n# 7. Temporal Audit")
    paper = paper.replace("Evidence Freeze v2의 판정은 **A. THESIS_REFRAME_STRONGLY_SUPPORTED**", "Evidence Freeze v3의 판정은 **A. THESIS_CORE_STRENGTHENED_BY_PROSPECTIVE_STOP**")
    paper = paper.replace("- E033-E034 및 `docs/framework_cost_avoidance_summary_20260718.csv`: documented model-run lower bound와 final actual use.", "- E033-E034 및 `docs/framework_cost_avoidance_summary_20260718.csv`: documented model-run lower bound와 final actual use.\n- E035-E046: `runs/evidence_freeze_v3_20260718/research_evidence_ledger.csv`의 DeepPCB prospective freeze, primary FAIL, secondary audit, NO_CANDIDATE와 STOP rows.")
    reference_marker = "[5] Caron, M. et al. “Emerging Properties in Self-Supervised Vision Transformers.” ICCV, 2021."
    paper = paper.replace(reference_marker, reference_marker + "\n\n[6] Tang, S. et al. “Online PCB Defect Detector on a New PCB Defect Dataset.” arXiv:1902.06197, 2019.\n\n[7] Saiyod, S. et al. “Defect-Intent Ambiguity Addressing for Training-Free Deterministic PCB Defect Localization via Template Selection and Dissimilarity Mapping.” Sensors 26(5), 1541, 2026.\n\n[8] Zeng, Z. et al. “Reference-Based Defect Detection Network.” IEEE Transactions on Image Processing 30, 6637-6647, 2021.")
    defense_section = """

## 8.5 학위논문 방어를 위한 수용 조건과 보완책

이 논문의 생존 가능성은 positive selector 성능이 아니라 연구 질문의 적합성에 달려 있다. 학과와 지도교수가 **새 방법의 성능 우위**만을 필수 기여로 요구한다면 현재 패키지는 그 요건을 충족하지 못한다. 반대로 산업 AI 연구에서 후보 신호의 번역 실패, 단계별 authorization, 비용 억제와 claim-boundary enforcement를 방법론적·실증적 기여로 인정한다면 논문 중심부는 완결 가능하다.

방어 시에는 다음 네 계약을 고정한다. 첫째, DeepPCB는 selector 성공이 아니라 prospective process-conformance 사례다. 둘째, predictive screening accuracy는 counterfactual 부재로 `NOT IDENTIFIABLE`이다. 셋째, 방지된 실행 수는 문서로 확인되는 최소 45개만 보고하고 DeepPCB에 임의 수를 더하지 않는다. 넷째, future positive method는 본 논문의 성립 조건이 아니라 별도 후속 연구다. 이 계약을 Evidence Freeze v3 ledger, source registry, 자동 무결성 검사와 동일한 문구로 유지한다.

따라서 추가 보완의 우선순위는 새 실험이 아니라 (i) 심사 기준과 기여 형태의 지도교수 승인, (ii) claim-evidence crosswalk 검수, (iii) 재현 가능한 source locator와 hash 보존, (iv) limitations와 반증 가능성의 명시다. 이 네 항목이 승인되면 성능 결과가 추가로 나오지 않아도 원점 회귀하지 않는다. 승인되지 않으면 연구를 더 돌리기 전에 논문 유형 또는 주제 변경을 결정해야 한다.
"""
    paper = paper.replace("\n# 9. 결론", defense_section + "\n\n# 9. 결론")
    write_text(DOCS / "mini_paper_validity_gated_industrial_al_v2_20260718.md", paper)


def build_defense_readiness() -> None:
    text = f"""# Thesis defense readiness and non-regression contract

Date: 2026-07-18
Evidence decision: **A. THESIS_CORE_STRENGTHENED_BY_PROSPECTIVE_STOP**

## 1. 무엇이 추가로 강해졌는가

DeepPCB는 positive method 성능을 추가하지 않았다. 대신 외부 데이터에서 protocol·score·gate·selection hash를 먼저 고정하고, G1의 양의 relation 이후에도 G2 primary FAIL을 수용했으며, secondary signal을 mechanism/development audit으로 격리한 뒤 detector와 official test를 사용하지 않은 end-to-end authorization trace를 추가했다. 따라서 retrospective workflow라는 기존 약점은 일부 보완됐지만 predictive policy의 정확도는 여전히 식별되지 않는다.

## 2. 심사 전 수용 조건

| 질문 | 현재 답 | 교수 승인 필요성 |
|---|---|---|
| 학위 기여가 반드시 새 selector의 성능 우위여야 하는가? | 현재는 충족하지 못함 | **필수** |
| validity-gated empirical evaluation과 failure-condition map을 중심 기여로 인정할 수 있는가? | 원자료와 prospective STOP 사례가 존재 | **필수** |
| predictive accuracy를 주장하지 않는 범위를 수용하는가? | NOT IDENTIFIABLE로 고정 | **필수** |
| DeepPCB 추가 실험 없이 branch closure를 수용하는가? | exact branch closed | **필수** |

## 3. Non-regression contract

- `1.307591`을 confirmatory PASS로 승격하지 않는다.
- 577-1024 px² 관계를 tiny defect 전반으로 확대하지 않는다.
- DeepPCB detector improvement를 주장하지 않는다.
- literature method의 일반 무효를 주장하지 않는다.
- documented prevented model runs lower bound 45에 DeepPCB 수를 임의 추가하지 않는다.
- DeepPCB score/threshold rescue, detector training/inference, official/final test 접근을 금지한다.

## 4. 논문 완성도를 높이는 허용 작업

새 실험 없이 허용되는 작업은 evidence locator 검증, 표·그림과 본문 수치의 일치 검사, claim-evidence crosswalk, 지도교수용 구두 설명과 한계 문구 정제, DOCX/PDF 렌더링 QA다. 새로운 성능 탐색은 본 freeze의 보완이 아니라 별도 연구 질문이므로 분리해야 한다.

## 5. 최종 현실적 판정

현재 패키지는 “positive selector 논문”으로는 부족하지만 “산업 결함 AL 후보 신호가 discovery·safety·learning utility로 번역되는지를 단계적으로 제한하고, 불충분한 positive signal의 확장을 실제로 중단한 실증 workflow”로는 방어 가능한 토대가 있다. 원점 회귀를 막는 핵심은 추가 성능이 아니라 지도교수와 이 기여 형태를 먼저 합의하는 것이다.
"""
    write_text(DOCS / "thesis_defense_readiness_v3_20260718.md", text)


def build_stop_figure() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(19.2, 5.4))
    ax.set_xlim(0, 19.2)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    colors = ["#376F95", "#376F95", "#376F95", "#3F7D57", "#B84A4A", "#B77B24", "#B77B24", "#B84A4A", "#17324D"]
    titles = ["External\ndataset", "Protocol\nfreeze", "Selection\nhash", "G1 PASS", "G2 FAIL", "Exploratory\nsmall signal", "Mechanism\naudit A2", "Development\nNO CANDIDATE", "STOP"]
    details = ["DeepPCB", "score · gate\nquery 20%", "SHA-256\nfrozen", "ρ=0.412\n6/6 positive", "1.108 < 1.20", "small 1.308\nexploratory", "group 45.79%\n577-1024 px²", "M2 recall .179\nbrightness .20", "detector 0\nofficial/final 0"]
    xs = [0.30, 2.40, 4.50, 6.60, 8.70, 10.80, 12.90, 15.00, 17.10]
    for i, x in enumerate(xs):
        face = "#F7EDED" if i in (4, 7) else ("#F8F1E5" if i in (5, 6) else "#EAF1F7")
        rect = plt.Rectangle((x, 1.25), 1.75, 2.65, facecolor=face, edgecolor=colors[i], linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.875, 3.48, titles[i], ha="center", va="center", fontsize=11.2, weight="bold", color=colors[i])
        ax.text(x + 0.875, 2.18, details[i], ha="center", va="center", fontsize=9.0, color="#17324D", linespacing=1.25)
        if i < len(xs) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.08, 2.58), xytext=(x + 1.83, 2.58), arrowprops=dict(arrowstyle="-|>", color="#6B7280", lw=1.5))
    ax.text(0.35, 4.87, "DeepPCB Prospective External Authorization Case", fontsize=20, color="#17324D", weight="bold")
    ax.text(0.35, 4.48, "Positive secondary signal was audited, not promoted. The exact branch closes without detector or official-test consumption.", fontsize=10.5, color="#6B7280")
    ax.text(9.6, 0.50, "METHOD SUCCESS: NO    |    RESEARCH STOPPING SUCCESS: YES    |    PREDICTIVE ACCURACY: NOT IDENTIFIABLE", ha="center", fontsize=11.5, color="#17324D", weight="bold")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "deeppcb_prospective_stop_case.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def svg_text(x: float, y: float, lines: list[str], size: int, color: str, weight: int = 400, anchor: str = "start", line_height: int | None = None) -> str:
    line_height = line_height or int(size * 1.28)
    tspans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else line_height
        tspans.append(f'<tspan x="{x}" dy="{dy}">{html.escape(line)}</tspan>')
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">' + "".join(tspans) + "</text>"


def build_architecture_svg_html() -> None:
    navy, blue, green, red, amber, gray = "#17324D", "#376F95", "#3F7D57", "#B84A4A", "#B77B24", "#667085"
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#667085"/></marker><filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="4" stdDeviation="5" flood-opacity="0.12"/></filter></defs>',
        '<rect width="1920" height="1080" fill="#F7F9FC"/>',
        '<rect x="0" y="0" width="1920" height="116" fill="#17324D"/>',
        svg_text(58, 55, ["Validity-Gated Industrial Defect AL — Evidence Freeze v3"], 34, "#FFFFFF", 700),
        svg_text(58, 91, ["Prospective authorization evidence + retrospective translation-failure map"], 18, "#C9D6E2", 400),
        svg_text(1862, 60, ["2026-07-18"], 17, "#C9D6E2", 400, "end"),
    ]
    # Top thesis evolution cards.
    cards = [
        (58, 145, 330, 142, "01  INITIAL HYPOTHESIS", ["VLM consistency", "DINO diversity", "detector uncertainty", "→ selector superiority"], red, "#FCEEEE"),
        (420, 145, 420, 142, "02  RETROSPECTIVE CORE", ["Discovery ≠ safety", "coverage ≠ learning utility", "seed stability ≠ acquisition generalization"], blue, "#EAF1F7"),
        (872, 145, 470, 142, "03  DEEPPCB PROSPECTIVE CASE", ["External paired-reference data", "protocol + score + selection hash frozen", "primary FAIL accepted"], amber, "#FBF2E5"),
        (1374, 145, 488, 142, "04  V3 DECISION", ["THESIS CORE STRENGTHENED", "by prospective STOP evidence", "not by positive method performance"], green, "#E9F4ED"),
    ]
    for x, y, w, h, title, lines, edge, face in cards:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{face}" stroke="{edge}" stroke-width="2" filter="url(#shadow)"/>')
        parts.append(svg_text(x + 22, y + 32, [title], 16, edge, 700))
        parts.append(svg_text(x + 22, y + 65, lines, 17, navy, 500, line_height=24))
    for x1, x2 in ((388, 420), (840, 872), (1342, 1374)):
        parts.append(f'<line x1="{x1}" y1="216" x2="{x2-8}" y2="216" stroke="{gray}" stroke-width="2" marker-end="url(#arrow)"/>')

    # Six-stage gate band.
    parts.append(svg_text(58, 333, ["VALIDITY-GATED AUTHORIZATION PIPELINE"], 20, blue, 700))
    gate_x = [58, 350, 642, 934, 1226, 1518]
    gate_data = [
        ("G1", "Signal validity", ["grounding", "relation/confound"]),
        ("G2", "Target discovery", ["yield", "enrichment@budget"]),
        ("G3", "Composition safety", ["class/source", "rare safety"]),
        ("G4", "Acquisition repro.", ["new selections", "fixed ≠ new"]),
        ("G5", "Learning utility", ["mAP · recall", "rare AP"]),
        ("G6", "Operational validity", ["top-budget", "human/deploy"]),
    ]
    for i, (x, (gid, title, lines)) in enumerate(zip(gate_x, gate_data)):
        edge = blue if i < 4 else (green if i == 4 else amber)
        face = "#EDF4F8" if i < 4 else ("#ECF6EF" if i == 4 else "#FBF3E8")
        parts.append(f'<rect x="{x}" y="352" width="250" height="145" rx="15" fill="{face}" stroke="{edge}" stroke-width="2"/>')
        parts.append(svg_text(x + 18, 382, [gid], 17, edge, 700))
        parts.append(svg_text(x + 18, 414, [title], 18, navy, 700))
        parts.append(svg_text(x + 18, 448, lines, 15, gray, 400, line_height=21))
        if i < 5:
            parts.append(f'<line x1="{x+253}" y1="425" x2="{gate_x[i+1]-8}" y2="425" stroke="{gray}" stroke-width="2" marker-end="url(#arrow)"/>')
    parts.append(f'<rect x="58" y="515" width="1810" height="56" rx="14" fill="#E9F4ED" stroke="{green}" stroke-width="2"/>')
    parts.append(svg_text(963, 550, ["PASS = one bounded measurement authorization, not a success prediction    |    FAIL / NA = STOP + claim contraction"], 20, navy, 700, "middle"))

    # DeepPCB chain.
    parts.append(svg_text(58, 620, ["PROSPECTIVE EXTERNAL AUTHORIZATION CASE — DEEPPCB"], 20, amber, 700))
    chain = [
        (58, "External + freeze", ["paired reference", "selection SHA-256"], blue, "#EAF1F7"),
        (302, "G1 PASS", ["ρ = 0.412", "6/6 positive"], green, "#E9F4ED"),
        (546, "G2 FAIL_STOP", ["1.108 < 1.20", "primary gate"], red, "#FCEEEE"),
        (790, "Secondary", ["small 1.308", "exploratory"], amber, "#FBF2E5"),
        (1034, "Phase A — A2", ["group 45.79%", "577-1024 px²"], amber, "#FBF2E5"),
        (1278, "Phase B", ["NO CANDIDATE", "M2 recall .179"], red, "#FCEEEE"),
        (1522, "STOP", ["detector 0", "official/final 0"], navy, "#EEF1F4"),
    ]
    for i, (x, title, lines, edge, face) in enumerate(chain):
        parts.append(f'<rect x="{x}" y="646" width="210" height="128" rx="14" fill="{face}" stroke="{edge}" stroke-width="2"/>')
        parts.append(svg_text(x + 105, 682, [title], 17, edge, 700, "middle"))
        parts.append(svg_text(x + 105, 718, lines, 16, navy, 500, "middle", line_height=22))
        if i < len(chain) - 1:
            parts.append(f'<line x1="{x+212}" y1="710" x2="{chain[i+1][0]-8}" y2="710" stroke="{gray}" stroke-width="2" marker-end="url(#arrow)"/>')

    # Bottom claim boundaries.
    panels = [
        (58, "SUPPORTED", ["Prospective protocol + selection freeze", "Primary FAIL stopped detector expansion", "Secondary signal audited separately", "Official/final actual use = 0"], green, "#E9F4ED"),
        (662, "EXPLORATORY", ["Small-share uplift 1.1828", "Residual ↔ bbox area 577-1024 px²", "Spatial checks positive but A2", "No external utility claim"], amber, "#FBF2E5"),
        (1266, "REJECTED / CLOSED", ["Total bbox-richness utility", "General small-defect triage", "Independent residual candidate", "DeepPCB detector improvement"], red, "#FCEEEE"),
    ]
    for x, title, lines, edge, face in panels:
        parts.append(f'<rect x="{x}" y="820" width="570" height="188" rx="18" fill="{face}" stroke="{edge}" stroke-width="2" filter="url(#shadow)"/>')
        parts.append(svg_text(x + 24, 855, [title], 19, edge, 700))
        parts.append(svg_text(x + 24, 892, ["• " + line for line in lines], 16, navy, 500, line_height=25))
    parts.append(svg_text(960, 1052, ["METHOD SUCCESS: NO   |   PROSPECTIVE AUTHORIZATION CONFORMANCE: YES   |   PREDICTIVE ACCURACY: NOT IDENTIFIABLE"], 18, navy, 700, "middle"))
    parts.append("</svg>")
    svg = "\n".join(parts)
    svg_path = FIGURES / "full_page_validity_gated_architecture_v2.svg"
    write_text(svg_path, svg)
    html_text = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Validity-Gated Architecture v2</title>
<style>html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#F7F9FC}}body{{display:flex;align-items:center;justify-content:center}}svg{{width:100vw;height:100vh;display:block;font-family:'Malgun Gothic','Noto Sans KR',Arial,sans-serif}}</style></head>
<body>{svg}</body></html>"""
    write_text(FIGURES / "full_page_validity_gated_architecture_v2.html", html_text)


def build_source_registry(data: dict[str, Any]) -> None:
    sources = [
        (ORIGINAL / "gate_decision.json", "DeepPCB parent gate"),
        (ORIGINAL / "frozen_selected_images.csv", "frozen selection"),
        (ORIGINAL / "frozen_selection_sha256.txt", "selection hash"),
        (PHASE_A / "phase_a_decision.json", "mechanism decision"),
        (PHASE_A / "size_bin_enrichment.csv", "size-specific evidence"),
        (PHASE_B / "method_comparison.csv", "development benchmark"),
        (PHASE_B / "frozen_candidate.json", "candidate decision"),
        (ROOT / "runs" / "evidence_freeze_v2_20260718" / "research_evidence_ledger.csv", "v2 ledger"),
    ]
    rows = []
    for index, (path, role) in enumerate(sources, start=1):
        rows.append({"source_id": f"V3S{index:03d}", "path": path.relative_to(ROOT).as_posix(), "role": role, "exists": path.exists(), "sha256": sha256(path) if path.exists() else "", "source_locator_required": True})
    write_csv(OUT / "source_registry.csv", rows, ["source_id", "path", "role", "exists", "sha256", "source_locator_required"])


def write_config() -> None:
    config = {
        "freeze": "Evidence Freeze v3",
        "decision": DECISION,
        "decision_scope": "thesis core strengthened by one prospective STOP authorization case; not method performance or predictive accuracy",
        "deeppcb_branch_closed": True,
        "score_threshold_rescue_authorized": False,
        "external_confirmation_authorized": False,
        "detector_screen_authorized": False,
        "training_performed": False,
        "detector_inference_performed": False,
        "vlm_calls_performed": False,
        "embedding_extraction_performed": False,
        "official_test_used": False,
        "final_test_used": False,
        "documented_prevented_detector_runs_lower_bound": 45,
        "deeppcb_prevented_run_count_added": 0,
        "predictive_policy_claim": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "freeze_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    data = verify_frozen_inputs()
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "decision": DECISION, "checks": data["checks"], "training": False, "inference": False, "official_test": False, "final_test": False}, ensure_ascii=False, indent=2))
        return
    build_hypothesis_matrix()
    build_mechanism_matrix()
    build_ledger(data)
    build_claim_boundary()
    build_closure_doc(data)
    build_evolution_v3()
    build_advisor_brief()
    build_mini_paper_v2()
    build_defense_readiness()
    build_stop_figure()
    build_architecture_svg_html()
    build_source_registry(data)
    write_config()
    print(json.dumps({"status": "DONE", "decision": DECISION, "ledger": str(OUT / "research_evidence_ledger.csv"), "html": str(FIGURES / "full_page_validity_gated_architecture_v2.html"), "training": False, "inference": False, "official_test": False, "final_test": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

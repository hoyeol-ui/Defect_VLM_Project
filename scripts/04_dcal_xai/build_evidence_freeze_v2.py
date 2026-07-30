from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
OUT = ROOT / "runs" / "evidence_freeze_v2_20260718"
DATE = "2026-07-18"

STATUS = {
    "rejected",
    "partially_supported",
    "descriptive_positive",
    "confirmatory_negative",
    "exploratory_only",
    "not_tested",
    "not_identifiable",
    "out_of_scope",
    "implementation_asset_only",
}

INITIAL_PDFS = [
    Path(r"C:\Users\user\Downloads\Active Learning의발전과 VLM 결합 연구 동향_이호열.pdf"),
    Path(r"C:\Users\user\Downloads\발표_대본__Active_Learning의_발전과_VLM_결합_연구_동향.pdf"),
    Path(r"C:\Users\user\Downloads\474aae46-65df-4446-bb82-de5f46bfe19b__학위논문_연구계획서_(전체_문서).pdf"),
    Path(r"C:\Users\user\Downloads\ae434082-0e63-4b21-aba7-0e623431e317_VLM_설명_일관성Groundedness산업_결함_객체탐지_Active_Learning_문헌지도.pdf"),
    Path(r"C:\Users\user\Downloads\4d8e6f7e-b94d-41b7-a14d-5f8ecfda569f_VLM_Self-Consistency_기반_설명_가능한_능동_학습_제조_현장_수용성_향상을_위한_타당성_연구.pdf"),
]


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def ensure_no_locked_path(path: str | Path) -> None:
    lowered = str(path).replace("\\", "/").lower()
    forbidden = ("final_test", "final-test", "locked_test", "locked-test")
    if any(token in lowered for token in forbidden):
        raise RuntimeError(f"Protected path rejected: {path}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def row(**kwargs: Any) -> dict[str, Any]:
    return kwargs


HYPOTHESES = [
    row(hypothesis_id="H01", research_phase="initial_plan", original_hypothesis="VLM 설명 Self-Consistency는 epistemic uncertainty의 직접 근사치이다.", original_claim_strength="theoretical proof", original_source="초기 발표 p.11; 연구계획서 p.3-4", dataset="NEU-DET; GC10-DET", method="multi-prompt semantic consistency", primary_metric="consistency-error/groundedness correlation", frozen_gate="legacy Spearman>=0.20 and AUC>=0.60", observed_result="Spearman=-0.181199, CI [-0.389654, 0.036410]; severe-failure AUC=0.373433; gate 0/6", evidence_status="rejected", failure_or_support_mechanism="consistent hallucination and semantic consistency-grounding decoupling", revised_research_question="언어적 일관성이 어떤 조건에서도 acquisition signal로 번역되지 않는가?", allowed_claim="본 설정에서 consistency는 epistemic uncertainty의 유효한 직접 근사가 아니었다.", prohibited_claim="VLM epistemic uncertainty를 증명했다.", manuscript_section="4.2 VLM consistency validity", source_file="runs/vlm_consistency_groundedness_validity/legacy_pilot_audit_20260715_main/legacy_validity_metrics.csv"),
    row(hypothesis_id="H02", research_phase="initial_plan", original_hypothesis="낮은 consistency는 detector 오류와 양의 상관을 가진다.", original_claim_strength="H1 confirmatory", original_source="초기 발표 p.11; 타당성 연구 p.7", dataset="GC10-DET; NEU-DET", method="legacy consistency audit", primary_metric="Spearman and severe-failure AUC", frozen_gate="positive correlation in every dataset", observed_result="GC10 Spearman=-0.311545; NEU=-0.084942; expected direction not observed", evidence_status="confirmatory_negative", failure_or_support_mechanism="response agreement did not track visual correctness", revised_research_question="consistency와 grounding을 먼저 분리 검증해야 하는가?", allowed_claim="예상한 오류 상관 방향은 확인되지 않았다.", prohibited_claim="낮은 consistency가 높은 오류를 예측한다.", manuscript_section="4.2 VLM consistency validity", source_file="runs/vlm_consistency_groundedness_validity/legacy_pilot_audit_20260715_main/legacy_validity_metrics.csv"),
    row(hypothesis_id="H03", research_phase="initial_plan", original_hypothesis="Consistency 기반 selection은 Random/Entropy보다 detector learning curve가 우수하다.", original_claim_strength="H2 superiority", original_source="초기 발표 p.11; 연구계획서 p.6", dataset="NEU-DET", method="VLM consistency selector family", primary_metric="mAP50-95 vs label budget", frozen_gate="stable superiority across acquisition seeds", observed_result="VLM validity prerequisite failed; later GT-free selector branches did not stably beat Random", evidence_status="rejected", failure_or_support_mechanism="signal validity failed before learning utility", revised_research_question="selection 전에 signal validity를 gate할 수 있는가?", allowed_claim="VLM consistency selector branch was stopped at validity stage.", prohibited_claim="Consistency selection is superior to Random.", manuscript_section="4.4 branch decision", source_file="docs/branch_decision_table_20260718.csv"),
    row(hypothesis_id="H04", research_phase="vlm_grounding", original_hypothesis="Detection-oriented prompt와 groundedness를 결합하면 VLM selection utility가 복구된다.", original_claim_strength="method repair", original_source="VLM 문헌지도 p.1, p.7", dataset="GC10-DET", method="structured full-image prompts", primary_metric="schema, bbox coverage, IoU", frozen_gate="schema/informative>=0.90; bbox coverage>=0.80", observed_result="schema completeness=0.01; bbox coverage=0.30; median IoU=0; gate FAIL", evidence_status="rejected", failure_or_support_mechanism="structured compliance and visual grounding collapse", revised_research_question="oracle localization에서도 grounding이 복구되는가?", allowed_claim="Prompt 구조화만으로 grounded acquisition signal이 복구되지 않았다.", prohibited_claim="Groundedness 결합이 selection utility를 향상했다.", manuscript_section="4.3 groundedness audit", source_file="runs/vlm_consistency_groundedness_validity/structured_prompt_pilot20_gc10_20260715/groundedness_audit/structured_groundedness_metrics.csv"),
    row(hypothesis_id="H05", research_phase="vlm_grounding", original_hypothesis="Oracle crop이 위치 문맥을 제공하면 positive evidence와 bbox grounding이 복구된다.", original_claim_strength="strong diagnostic", original_source="oracle crop frozen protocol", dataset="GC10-DET", method="20 positive oracle crops", primary_metric="presence, evidence, median IoU", frozen_gate="presence>=0.50; evidence>=0.70", observed_result="parse=1.0, informative=1.0, presence=0, evidence=0, median IoU=0; FAIL", evidence_status="confirmatory_negative", failure_or_support_mechanism="all-negative presence collapse persisted under oracle localization", revised_research_question="collapse가 특정 모델에 한정되는가?", allowed_claim="더 쉬운 oracle crop 조건에서도 grounded defect evidence가 실패했다.", prohibited_claim="Full-image 난이도만이 VLM 실패 원인이다.", manuscript_section="4.3 oracle crop diagnostic", source_file="runs/vlm_consistency_groundedness_validity/oracle_crop_diagnostic20_gc10_20260715/oracle_crop_audit/oracle_crop_diagnostic_metrics.csv"),
    row(hypothesis_id="H06", research_phase="vlm_grounding", original_hypothesis="다른 소형 Qwen VLM은 compliance-grounding collapse를 복구한다.", original_claim_strength="cross-model confirmation", original_source="paired model comparison protocol", dataset="GC10-DET", method="3 models x 40 paired views", primary_metric="balanced accuracy and median bbox IoU", frozen_gate="all compliance, sensitivity, specificity and grounding checks", observed_result="0/3 pass; best balanced accuracy=0.70 with median IoU=0", evidence_status="confirmatory_negative", failure_or_support_mechanism="model-specific but persistent collapse modes", revised_research_question="VLM은 acquisition scorer가 아니라 evidence-constrained interface로만 제한해야 하는가?", allowed_claim="검토한 세 소형 모델에서 deployable grounded signal을 얻지 못했다.", prohibited_claim="모델 교체로 VLM signal validity가 해결됐다.", manuscript_section="4.4 paired model comparison", source_file="runs/vlm_consistency_groundedness_validity/paired_model_comparison_gc10_20260715/comparison/paired_model_gate_comparison.csv"),
    row(hypothesis_id="H07", research_phase="gt_free_selection", original_hypothesis="Global DINO diversity/coverage는 중복을 줄이고 범용 detector utility를 높인다.", original_claim_strength="cross-dataset utility", original_source="초기 AL diversity 근거; DINO branch protocols", dataset="NEU-DET; GC10-DET; MPDD; VisA", method="frozen global DINO", primary_metric="discovery, composition, downstream mAP", frozen_gate="dataset-specific selection and translation gates", observed_result="Discovery gains existed, but VisA category collapse, MPDD source confound, GC10 rare-AP loss and NEU non-generalization occurred", evidence_status="rejected", failure_or_support_mechanism="global representation follows dominant product/source/texture variation rather than local defect utility", revised_research_question="Discovery, composition safety and learner utility를 분리해야 하는가?", allowed_claim="Global DINO는 일부 fixed benchmark discovery endpoint를 개선했지만 범용 utility는 보장하지 않았다.", prohibited_claim="DINO diversity generally improves detector performance.", manuscript_section="5-6 selection and translation", source_file="docs/stratified_condition_robustness_decision_20260718.md"),
    row(hypothesis_id="H08", research_phase="cold_start", original_hypothesis="NEU seed45의 고정 Visual20 집합은 training seed 변화에도 Random20보다 우수하다.", original_claim_strength="fixed-set diagnostic", original_source="V8 seed45 post-hoc clue", dataset="NEU-DET", method="one acquisition set x 5 training seeds", primary_metric="mAP50-95 difference", frozen_gate="gain>=0.01; >=4/5 wins; safety", observed_result="+0.016236, 5/5 wins, descriptive CI about [0.006738, 0.027864]; fixed-set gate PASS", evidence_status="descriptive_positive", failure_or_support_mechanism="selected-set effect was not explained solely by training noise", revised_research_question="새 acquisition set에서도 반복되는가?", allowed_claim="그 한 고정 선택 집합은 다섯 training seeds에서 안정적이었다.", prohibited_claim="Cold-start DINO selector generally outperforms Random.", manuscript_section="6.3 seed45 fixed-set diagnostic", source_file="runs/seed45_fixed_set_stability/seed45_fixed_set_stability_main/screen_gate.csv"),
    row(hypothesis_id="H09", research_phase="cold_start_confirmation", original_hypothesis="Seed45 효과는 새로운 acquisition seeds에 일반화된다.", original_claim_strength="preregistered confirmation", original_source="V8 cold-start confirmation protocol", dataset="NEU-DET", method="10 new acquisition seeds", primary_metric="paired mAP50-95 difference", frozen_gate="gain>=0.01; >=7/10; p<=0.05; recall safety", observed_result="+0.007019; CI [-0.005211, 0.019678]; p=0.322266; 7/10; FAIL", evidence_status="confirmatory_negative", failure_or_support_mechanism="acquisition-set non-generalization", revised_research_question="training-seed stability와 acquisition robustness를 구분해야 하는가?", allowed_claim="Fixed-set stability did not establish acquisition-set generalization.", prohibited_claim="Seed45 result was independently confirmed.", manuscript_section="6.4 acquisition confirmation", source_file="runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main/confirmatory_gate.csv"),
    row(hypothesis_id="H10", research_phase="random_audit", original_hypothesis="Random은 class/instance/bbox coverage가 약한 baseline이다.", original_claim_strength="baseline assumption", original_source="early selector framing", dataset="NEU-DET", method="V10c24 Random post-hoc audit", primary_metric="class coverage, instance diversity, bbox richness", frozen_gate="audit only", observed_result="Random already supplied broad class coverage, instance diversity and bbox richness in the balanced large pool", evidence_status="rejected", failure_or_support_mechanism="balanced pool and non-tiny budget make Random robust", revised_research_question="Random이 강한 pool 조건은 무엇인가?", allowed_claim="Balanced large-pool NEU에서 Random은 강한 baseline이었다.", prohibited_claim="Random is a weak baseline.", manuscript_section="6.2 Random baseline strength", source_file="runs/random_baseline_audit_v10c24/random_baseline_audit_20260713_225216/random_baseline_audit_summary.md"),
    row(hypothesis_id="H11", research_phase="cross_dataset_selection", original_hypothesis="GC10 Frozen DINO가 rare/taxonomy discovery를 안전하게 개선한다.", original_claim_strength="selection-only confirmatory", original_source="GC10 frozen protocol", dataset="GC10-DET", method="initial20/query20; 200 paired seeds", primary_metric="rare images/classes and combined classes", frozen_gate="11 checks; rare gain>=0.75; class gain>=0.25", observed_result="rare images +2.720, CI [2.455, 2.980]; combined classes +0.825; selection gate PASS", evidence_status="partially_supported", failure_or_support_mechanism="fixed-pool rare discovery plus filename-sequence proxy diversification", revised_research_question="selection gain이 rare detector AP로 번역되는가?", allowed_claim="동일 fixed benchmark pool의 200 paired seeds에서 rare discovery가 증가했다.", prohibited_claim="GC10 detector performance improved generally.", manuscript_section="5.2 GC10 discovery", source_file="runs/stratified_condition_robustness_20260718/stratified_effect_summary.csv"),
    row(hypothesis_id="H12", research_phase="cross_dataset_selection", original_hypothesis="MPDD FrozenCategoryBalancedDINO가 category safety를 유지하며 anomaly discovery를 높인다.", original_claim_strength="selection-only", original_source="MPDD frozen protocol", dataset="MPDD", method="initial20/query20; 200 paired seeds", primary_metric="anomaly yield and category coverage", frozen_gate="category delta>=-0.25", observed_result="anomaly +6.245, CI [5.885, 6.605]; category -0.325; overall gate FAIL", evidence_status="descriptive_positive", failure_or_support_mechanism="source-confounded capture-day diversification; 67.3% source-composition contribution", revised_research_question="Discovery gain이 source/category composition에 의존하는가?", allowed_claim="같은 candidate pool에서 anomaly yield는 증가했지만 safety gate는 실패했다.", prohibited_claim="MPDD downstream anomaly utility or production generalization improved.", manuscript_section="5.3 MPDD discovery and confound", source_file="runs/stratified_condition_robustness_20260718/stratified_effect_summary.csv"),
    row(hypothesis_id="H13", research_phase="cross_dataset_selection", original_hypothesis="VisA Frozen DINO가 12-category safety를 유지하며 sparse anomaly discovery를 높인다.", original_claim_strength="selection-only", original_source="VisA frozen protocol", dataset="VisA", method="initial20/query20; 200 paired seeds", primary_metric="anomaly yield and category coverage", frozen_gate="category coverage not worse", observed_result="anomaly +14.480, CI [14.195, 14.770]; category -4.110; HHI +0.286025; all 200 seeds gain+safety loss", evidence_status="descriptive_positive", failure_or_support_mechanism="category collapse; 72.4% of gain associated with dominant category contribution", revised_research_question="Discovery와 composition safety를 별도 gate로 두어야 하는가?", allowed_claim="Anomaly discovery rose while category coverage collapsed.", prohibited_claim="VisA selector was successful or safe.", manuscript_section="5.4 VisA category collapse", source_file="runs/stratified_condition_robustness_20260718/stratified_effect_summary.csv"),
    row(hypothesis_id="H14", research_phase="selection_learning_translation", original_hypothesis="GC10 q20 discovery gain은 overall 및 rare detector utility로 번역된다.", original_claim_strength="frozen downstream confirmation", original_source="GC10 development detector confirmation", dataset="GC10-DET", method="5 acquisitions x 3 training seeds", primary_metric="mAP50-95 and rare macro AP", frozen_gate="mAP gain>=0.010 and rare gain>=0.020 plus safety", observed_result="overall mAP +0.017378; rare macro AP -0.019877; gate FAIL", evidence_status="partially_supported", failure_or_support_mechanism="overall mAP-rare safety trade-off", revised_research_question="평균 utility와 rare-class safety를 분리해야 하는가?", allowed_claim="Overall development mAP rose in this protocol, but rare utility worsened and the gate failed.", prohibited_claim="GC10 DINO detector translation succeeded.", manuscript_section="6.5 first detector translation", source_file="runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_development_confirmation_gate.csv"),
    row(hypothesis_id="H15", research_phase="budget_reset", original_hypothesis="K40 budget140은 Random140보다 taxonomy omission risk를 낮춘다.", original_claim_strength="selection holdout confirmation", original_source="V2 budget protocol", dataset="GC10-DET", method="200 design + 200 holdout seeds", primary_metric="all-class/all-rare/min-two coverage", frozen_gate="all-class>=0.90; all-rare>=0.90; min-two>=0.75; safety", observed_result="holdout all-class 0.955 vs 0.940; rare images +1.820; selection gate PASS", evidence_status="partially_supported", failure_or_support_mechanism="cluster coverage reduced zero-coverage risk", revised_research_question="taxonomy coverage가 detector learning utility로 번역되는가?", allowed_claim="K40 reduced fixed-benchmark taxonomy omission risk at budget140.", prohibited_claim="K40 improves detector utility.", manuscript_section="5.5 K40 coverage", source_file="runs/dcal_xai/v2_budget_extended/decision.json"),
    row(hypothesis_id="H16", research_phase="selection_learning_translation", original_hypothesis="K40 coverage gain은 YOLOv8n overall/rare detector utility로 번역된다.", original_claim_strength="confirmatory downstream screen", original_source="V2.2 protocol", dataset="GC10-DET", method="5 acquisitions x 3 training seeds x 2 policies", primary_metric="mAP50-95, recall, rare macro AP", frozen_gate="five downstream checks", observed_result="mAP -0.001678, rare AP -0.018290, recall -0.021871; FAIL", evidence_status="confirmatory_negative", failure_or_support_mechanism="coverage-utility gap", revised_research_question="Selection-stage coverage는 learner utility의 필요조건도 충분조건도 아닌가?", allowed_claim="Coverage PASS did not translate to detector utility in the frozen screen.", prohibited_claim="K40 downstream noninferiority or superiority.", manuscript_section="6.6 K40 translation", source_file="runs/dcal_xai/v2_backbone_main/v8n_screen_decision.json"),
    row(hypothesis_id="H17", research_phase="detector_coupled", original_hypothesis="Horizontal-flip disagreement는 epistemic detector uncertainty이다.", original_claim_strength="selection signal", original_source="DCAL-XAI R1", dataset="GC10-DET", method="DetectorDifficultyDiversity", primary_metric="coverage with instance noninferiority", frozen_gate="all seven R1 checks", observed_result="query instances -2.2; 94/100 selections had one-view detection dominance; FAIL", evidence_status="rejected", failure_or_support_mechanism="augmentation geometry confound", revised_research_question="Detector-native signals must be validated against actual error before use.", allowed_claim="Flip disagreement was dominated by geometry/view sensitivity in this setup.", prohibited_claim="Flip disagreement estimates epistemic uncertainty.", manuscript_section="7.2 flip disagreement", source_file="runs/dcal_xai/gc10_r1/selection_summary.md"),
    row(hypothesis_id="H18", research_phase="detector_signal_validity", original_hypothesis="Frozen detector ensemble uncertainty enriches top20% total FP+FN enough for AL.", original_claim_strength="preregistered primary", original_source="V2.3 protocol", dataset="GC10-DET", method="15 existing Random140 checkpoints", primary_metric="top20% total-error enrichment", frozen_gate="mean>=1.50; CI low>1; AUROC>=0.65; rho>=0.20; stability>=0.50", observed_result="1.115901, CI [1.034640, 1.221093]; AUROC=0.766432; six of seven checks but primary effect-size FAIL", evidence_status="confirmatory_negative", failure_or_support_mechanism="ranking signal without sufficient operational top-budget enrichment", revised_research_question="Ranking quality와 actionable review enrichment를 분리해야 하는가?", allowed_claim="The signal ranked error above chance but failed the frozen operational enrichment threshold.", prohibited_claim="Detector uncertainty AL gate passed.", manuscript_section="7.3 V2.3 operational gate", source_file="runs/dcal_xai/v2_detector_signal_validity/decision.json"),
    row(hypothesis_id="H19", research_phase="detector_signal_validity", original_hypothesis="Confidence/no-detection may enrich FN and rare-FN for recall-repair triage.", original_claim_strength="post-hoc hypothesis generation", original_source="V2.3 secondary endpoints", dataset="GC10-DET", method="same sealed development predictions", primary_metric="FN/rare-FN enrichment", frozen_gate="none; not primary", observed_result="combined FN=1.379693; rare-FN=1.785910; confidence rare-FN=2.683022; rare support 18 images/22 boxes", evidence_status="exploratory_only", failure_or_support_mechanism="possible recall-oriented ranking; small rare support and reused development split", revised_research_question="Untouched native-bbox pool에서 recall triage를 독립 검증할 수 있는가?", allowed_claim="FN endpoints motivate a future protocol only.", prohibited_claim="Recall-repair triage is validated.", manuscript_section="7.4 exploratory FN signal", source_file="docs/fn_failure_prediction_extension_feasibility_20260718.md"),
    row(hypothesis_id="H20", research_phase="condition_map", original_hypothesis="Target sparsity/prevalence가 selector discovery gain을 조절한다.", original_claim_strength="cross-dataset condition law", original_source="three-dataset condition-map hypothesis", dataset="GC10-DET; MPDD; VisA", method="200-seed stratified audit", primary_metric="prevalence-stratum effect", frozen_gate="target range>=20; relative range>=0.10; >=20 independent pools/tertile", observed_result="No dataset passed; pool Jaccard GC10=0.978446, MPDD=0.962827, VisA=0.995386", evidence_status="not_identifiable", failure_or_support_mechanism="leave-20 perturbations of one fixed pool; prevalence nearly fixed", revised_research_question="Independent pool realizations are required before condition-effect estimation.", allowed_claim="Same-pool selector effects are identifiable; prevalence moderation is not.", prohibited_claim="Sparse targets cause larger gains.", manuscript_section="5.6 identifiability audit", source_file="docs/stratified_audit_design_feasibility_20260718.md"),
    row(hypothesis_id="H21", research_phase="metadata_feasibility", original_hypothesis="GC10/MPDD/VisA metadata can create independent production pool realizations.", original_claim_strength="prospective feasibility", original_source="metadata feasibility protocol", dataset="GC10-DET; MPDD; VisA", method="manifest+EXIF audit", primary_metric="documented independent groups", frozen_gate=">=20 documented target-blind groups N>=40 and mixed-target support", observed_result="0/3 datasets pass; overall CURRENT_DATASETS_INDEPENDENT_POOL_NO_GO", evidence_status="not_identifiable", failure_or_support_mechanism="metadata proxy-production lot validity gap", revised_research_question="Real lot/time/source metadata must be prospectively collected.", allowed_claim="Current benchmarks cannot support independent-production generalization.", prohibited_claim="EXIF day, filename group, or category is a production lot.", manuscript_section="5.7 metadata feasibility", source_file="docs/metadata_feasibility_decision_20260718.md"),
    row(hypothesis_id="H22", research_phase="metadata_feasibility", original_hypothesis="Existing proxies can still diagnose composition sensitivity descriptively.", original_claim_strength="post-hoc mechanism audit", original_source="metadata audit protocol", dataset="GC10-DET; MPDD; VisA", method="frozen-selection concentration audit", primary_metric="group coverage/entropy/HHI deltas", frozen_gate="descriptive only", observed_result="GC10 filename groups +1.465; MPDD DINO capture days +1.130; VisA categories -4.110 and HHI +0.286025", evidence_status="descriptive_positive", failure_or_support_mechanism="proxy diversification differs from category collapse", revised_research_question="Composition safety should be reported beside discovery.", allowed_claim="The frozen selections altered proxy/session/category composition in dataset-specific ways.", prohibited_claim="These proxies prove production generalization.", manuscript_section="5.7 metadata mechanism", source_file="runs/metadata_feasibility_audit_20260718/metadata_feasibility_main/selection_session_concentration_summary.csv"),
    row(hypothesis_id="H23", research_phase="initial_plan", original_hypothesis="VLM-AL improves mAP by 2-3 percentage points over HDAL-LB.", original_claim_strength="quantitative superiority target", original_source="연구계획서 p.3, p.6, p.9", dataset="NEU-DET", method="planned VLM-AL vs HDAL-LB", primary_metric="mAP", frozen_gate="+0.02 to +0.03", observed_result="No confirmatory HDAL-LB superiority test meeting this target; tested selector families did not stably beat Random", evidence_status="rejected", failure_or_support_mechanism="signal and acquisition generalization failed before target claim", revised_research_question="Method superiority가 아니라 translation validity를 평가한다.", allowed_claim="The original +2-3% superiority target was not achieved.", prohibited_claim="+2-3% mAP improvement.", manuscript_section="1.3 scope transition", source_file=str(INITIAL_PDFS[2])),
    row(hypothesis_id="H24", research_phase="initial_plan", original_hypothesis="제안 시스템은 label cost를 50-80% 절감한다.", original_claim_strength="operational/ROI claim", original_source="초기 발표 p.2; 연구계획서 p.9; 타당성 연구 p.15", dataset="planned field setting", method="planned cost analysis", primary_metric="measured annotation cost at matched performance", frozen_gate="50-80% reduction", observed_result="No field annotation-time/cost study and no matched-performance cost curve", evidence_status="not_tested", failure_or_support_mechanism="operational outcome not measured", revised_research_question="Gate가 불필요한 model runs를 얼마나 방지하는가?", allowed_claim="At least 45 planned detector model runs and all final-test consumption were avoided; monetary savings were not measured.", prohibited_claim="50-80% label cost reduction or ROI.", manuscript_section="8.5 cost prevention", source_file="docs/three_dataset_condition_map_decision_20260718.md"),
    row(hypothesis_id="H25", research_phase="initial_plan", original_hypothesis="VLM language alignment improves domain adaptation by 3-5% mAP.", original_claim_strength="quantitative domain adaptation", original_source="연구계획서 p.3-6", dataset="NEU-DET to KAMP/MVTec 3D-AD", method="planned language alignment", primary_metric="target-domain mAP", frozen_gate="+0.03 to +0.05", observed_result="Not executed under the frozen evidence program", evidence_status="not_tested", failure_or_support_mechanism="scope was narrowed before implementation", revised_research_question="Domain shift requires a separate supervised study.", allowed_claim="Domain adaptation remains outside the tested thesis evidence.", prohibited_claim="Language alignment improved domain adaptation.", manuscript_section="9 future work", source_file=str(INITIAL_PDFS[2])),
    row(hypothesis_id="H26", research_phase="initial_plan", original_hypothesis="A local lightweight VLM runs at 0.2 s/image on one RTX 3090.", original_claim_strength="deployment target", original_source="연구계획서 p.3-4", dataset="deployment benchmark", method="planned QLoRA/vLLM stack", primary_metric="latency and memory", frozen_gate="0.2 s/image", observed_result="No frozen latency/memory benchmark matching the planned system", evidence_status="not_tested", failure_or_support_mechanism="implementation target was not the final research question", revised_research_question="Deployment benchmarking belongs to future engineering work.", allowed_claim="Local execution was explored in components only; the target was not validated.", prohibited_claim="0.2 s/image local deployment achieved.", manuscript_section="9 future work", source_file=str(INITIAL_PDFS[2])),
    row(hypothesis_id="H27", research_phase="initial_plan", original_hypothesis="Grounded explanations improve inspector trust and field acceptance.", original_claim_strength="human/field outcome", original_source="타당성 연구 p.14-15, p.21", dataset="field inspectors", method="planned human study", primary_metric="trust/reinspection/acceptance", frozen_gate="trust 27% to 60%; reinspection 73% to 40%", observed_result="No inspector study or field pilot was conducted", evidence_status="not_tested", failure_or_support_mechanism="human outcome not measured; VLM grounding validity also failed", revised_research_question="Explanations must be evidence-constrained before any human study.", allowed_claim="Explainability/trust remains an untested future-work objective.", prohibited_claim="Inspector trust or acceptance improved.", manuscript_section="9 future work", source_file=str(INITIAL_PDFS[4])),
]


HYPOTHESIS_FIELDS = [
    "hypothesis_id", "research_phase", "original_hypothesis", "original_claim_strength",
    "original_source", "dataset", "method", "primary_metric", "frozen_gate",
    "observed_result", "evidence_status", "failure_or_support_mechanism",
    "revised_research_question", "allowed_claim", "prohibited_claim",
    "manuscript_section", "source_file",
]


ASSETS = [
    row(asset_id="A01", asset_type="initial_presentation", file_or_directory=str(INITIAL_PDFS[0]), original_purpose="AL/VLM research trend and proposed H1/H2", current_classification="literature_background_asset", current_value="historical motivation and exact initial claims", reusable_for="introduction and research evolution", limitations="expected effects are not results", thesis_role="background + hypothesis provenance", keep_or_exclude="keep_reframed", provenance_verified=True),
    row(asset_id="A02", asset_type="presentation_script", file_or_directory=str(INITIAL_PDFS[1]), original_purpose="oral explanation of initial proposal", current_classification="literature_background_asset", current_value="documents original rationale and intended practical value", reusable_for="advisor briefing history", limitations="not empirical evidence", thesis_role="appendix provenance", keep_or_exclude="keep_appendix", provenance_verified=True),
    row(asset_id="A03", asset_type="research_plan", file_or_directory=str(INITIAL_PDFS[2]), original_purpose="full thesis plan and quantitative targets", current_classification="rejected_method_evidence", current_value="primary source for discarded claims and scope change", reusable_for="research evolution narrative", limitations="contains untested expected metrics and field claims", thesis_role="hypothesis provenance, not current methods", keep_or_exclude="keep_reframed", provenance_verified=True),
    row(asset_id="A04", asset_type="literature_map", file_or_directory=str(INITIAL_PDFS[3]), original_purpose="consistency/grounding/industrial AL literature synthesis", current_classification="literature_background_asset", current_value="correctly anticipated consistency!=truth, cross-model risk and evaluation pitfalls", reusable_for="related work and failure interpretation", limitations="citation tokens in PDF must be replaced with final bibliography", thesis_role="related work scaffold", keep_or_exclude="keep_revise_citations", provenance_verified=True),
    row(asset_id="A05", asset_type="feasibility_plan", file_or_directory=str(INITIAL_PDFS[4]), original_purpose="Self-Consistency feasibility and field acceptance plan", current_classification="future_work_asset", current_value="shows shift from SOTA claim toward signal validity", reusable_for="limitations and future human study", limitations="trust/cost numbers were targets, not measured outcomes", thesis_role="historical transition + future work", keep_or_exclude="keep_reframed", provenance_verified=True),
    row(asset_id="A06", asset_type="VLM_prompts_responses", file_or_directory="runs/vlm_consistency_groundedness_validity", original_purpose="generate and score structured explanations", current_classification="rejected_method_evidence", current_value="raw evidence of schema/presence/grounding collapse", reusable_for="VLM validity chapter and qualitative examples", limitations="small pilots; Qwen-family scope", thesis_role="chapter 4 empirical core", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A07", asset_type="consistency_scores", file_or_directory="runs/vlm_consistency_groundedness_validity/legacy_pilot_audit_20260715_main", original_purpose="validate linguistic uncertainty", current_classification="rejected_method_evidence", current_value="six frozen checks failed with traceable statistics", reusable_for="H1 falsification", limitations="legacy pilot composition", thesis_role="chapter 4 negative result", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A08", asset_type="groundedness_audits", file_or_directory="runs/vlm_consistency_groundedness_validity/oracle_crop_diagnostic20_gc10_20260715", original_purpose="test whether localization context restores validity", current_classification="failure_mechanism_evidence", current_value="strong oracle-condition falsification", reusable_for="grounding collapse mechanism", limitations="20 crops", thesis_role="chapter 4 mechanism", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A09", asset_type="DINO_embeddings_clusters", file_or_directory="runs/gc10_taxonomy_selection_audit; runs/gc10_discovery_representation_audit", original_purpose="global visual diversity and label-aware retrieval", current_classification="implementation_asset", current_value="reusable frozen representation cache plus documented failure boundary", reusable_for="reproduction and future local-feature comparison", limitations="global CLS representation; not local defect feature", thesis_role="methods appendix + mechanism evidence", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A10", asset_type="selection_manifests", file_or_directory="runs/gc10_taxonomy_selection_audit; runs/mpdd_selection_only_audit; runs/visa_selection_only_audit", original_purpose="freeze selected image IDs before GT audit", current_classification="reproducibility_asset", current_value="exact query membership for 200 paired seeds", reusable_for="selection audit reproduction", limitations="leave-20 perturbations of fixed pools", thesis_role="chapter 5 data provenance", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A11", asset_type="random_baseline_records", file_or_directory="runs/random_baseline_audit_v10c24", original_purpose="audit baseline class/instance/bbox properties", current_classification="evaluation_methodology_asset", current_value="establishes Random as a strong baseline in balanced large pools", reusable_for="baseline design and discussion", limitations="NEU protocol-specific", thesis_role="chapter 6 baseline audit", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A12", asset_type="paired_200_seed_records", file_or_directory="runs/stratified_condition_robustness_20260718", original_purpose="estimate fixed-pool selector variation", current_classification="partial_positive_evidence", current_value="paired discovery effects and safety diagnostics", reusable_for="chapter 5 effect estimates", limitations="not independent production pools", thesis_role="chapter 5 primary evidence", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A13", asset_type="YOLO_checkpoints", file_or_directory="runs/dcal_xai/v2_backbone_main; runs/seed45_fixed_set_stability; runs/active_learning_v8_cold_start_confirmation", original_purpose="downstream utility and stability evaluation", current_classification="reproducibility_asset", current_value="reusable frozen learners for audit", reusable_for="reproduction only under new authorization", limitations="development split reused; checkpoint sets are protocol-specific", thesis_role="chapter 6 provenance", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A14", asset_type="sealed_predictions", file_or_directory="runs/dcal_xai/v2_detector_signal_validity/predictions", original_purpose="GT-blind prediction freeze before error join", current_classification="reproducibility_asset", current_value="15-checkpoint sealed prediction set", reusable_for="V2.3 audit reproduction", limitations="same 232-image development split", thesis_role="chapter 7 provenance", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A15", asset_type="final_test_lock", file_or_directory="docs/branch_decision_table_20260718.csv; run configs and logs", original_purpose="protect locked evaluation", current_classification="evaluation_methodology_asset", current_value="documents zero final-test consumption across branches", reusable_for="research integrity statement", limitations="must remain protected", thesis_role="chapter 3 and limitations", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A16", asset_type="protocol_documents", file_or_directory="docs/*protocol*.md; scripts/04_dcal_xai/*protocol*.md", original_purpose="freeze endpoint/gate before runs", current_classification="evaluation_methodology_asset", current_value="traceable staged validation design", reusable_for="framework chapter", limitations="branch-specific gates are not one universal threshold", thesis_role="chapter 3 core", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A17", asset_type="gate_code", file_or_directory="scripts/03_analysis; scripts/04_dcal_xai", original_purpose="automated branch decisions", current_classification="implementation_asset", current_value="prevents silent post-hoc PASS promotion", reusable_for="reproduction and supplemental code", limitations="research prototype", thesis_role="methods supplement", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A18", asset_type="unit_tests", file_or_directory="scripts/04_dcal_xai/test_*.py; tests", original_purpose="validate audit invariants", current_classification="reproducibility_asset", current_value="machine-checkable claim/gate integrity", reusable_for="artifact evaluation", limitations="not a substitute for external replication", thesis_role="appendix", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A19", asset_type="metadata_audit", file_or_directory="runs/metadata_feasibility_audit_20260718; docs/metadata_feasibility_decision_20260718.md", original_purpose="find independent production pool groupings", current_classification="evaluation_methodology_asset", current_value="prevents proxy-to-production-lot overclaim", reusable_for="identifiability and future data collection design", limitations="no dataset passes independent-pool gate", thesis_role="chapter 5 identifiability", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A20", asset_type="condition_figures", file_or_directory="docs/figures", original_purpose="summarize discovery/safety/translation", current_classification="implementation_asset", current_value="publication-ready evidence maps", reusable_for="thesis figures", limitations="must retain scope labels", thesis_role="chapters 1, 5, 8", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A21", asset_type="branch_decision_tables", file_or_directory="docs/branch_decision_table_20260718.csv; docs/condition_map_master_20260718.csv", original_purpose="integrate branch outcomes", current_classification="evaluation_methodology_asset", current_value="single traceable branch registry", reusable_for="evidence synthesis", limitations="summary rows must point to raw sources", thesis_role="chapter 3 and appendix", keep_or_exclude="keep", provenance_verified=True),
    row(asset_id="A22", asset_type="stopped_selector_variants", file_or_directory="V10d/PDF V10c/D2R variants", original_purpose="search for stronger selectors", current_classification="exclude_from_main_thesis", current_value="documents stopping discipline", reusable_for="brief appendix chronology only", limitations="would look like post-hoc method search if foregrounded", thesis_role="appendix branch log", keep_or_exclude="exclude_main_keep_registry", provenance_verified=True),
    row(asset_id="A23", asset_type="FN_extension_plan", file_or_directory="docs/fn_failure_prediction_extension_feasibility_20260718.md", original_purpose="evaluate recall-repair follow-up", current_classification="future_work_asset", current_value="frozen restart conditions without implementation", reusable_for="future work", limitations="post-hoc and no untouched validation", thesis_role="chapter 9", keep_or_exclude="keep_future_only", provenance_verified=True),
    row(asset_id="A24", asset_type="domain_adaptation_plan", file_or_directory=str(INITIAL_PDFS[2]), original_purpose="language-aligned cross-domain adaptation", current_classification="exclude_from_main_thesis", current_value="separate future supervised study", reusable_for="future proposal", limitations="not tested", thesis_role="future work only", keep_or_exclude="exclude_main", provenance_verified=True),
    row(asset_id="A25", asset_type="human_trust_claims", file_or_directory=str(INITIAL_PDFS[4]), original_purpose="inspector acceptance and reinspection reduction", current_classification="exclude_from_main_thesis", current_value="motivation only after citation verification", reusable_for="future human-subject protocol", limitations="no human study; numerical targets unverified", thesis_role="do not present as result", keep_or_exclude="exclude_results", provenance_verified=True),
]

ASSET_FIELDS = ["asset_id", "asset_type", "file_or_directory", "original_purpose", "current_classification", "current_value", "reusable_for", "limitations", "thesis_role", "keep_or_exclude", "provenance_verified"]


MECHANISMS = [
    row(dataset="GC10-DET", strategy_or_signal="FrozenDINOVisualDiversity q20", signal_validity="selection discovery valid within paired fixed pool", discovery_gain="rare images +2.720; CI [2.455,2.980]", composition_change="combined classes +0.825; filename groups +1.465", diversification_or_concentration="diversification", category_safety="mostly nonnegative (172/200 Q1)", source_or_session_safety="filename proxy only; no production claim", acquisition_reproducibility="200 paired leave-20 perturbations", downstream_overall_utility="q20 mAP +0.017378", downstream_rare_utility="rare macro AP -0.019877", operational_utility="translation gate FAIL", dominant_mechanism="coverage_utility_gap", final_decision="DISCOVERY_APPROVE / DETECTOR_REJECT", evidence_level="paired_existing_record + confirmatory downstream fail", source_file="runs/stratified_condition_robustness_20260718/stratified_effect_summary.csv; runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_development_confirmation_gate.csv"),
    row(dataset="MPDD", strategy_or_signal="FrozenCategoryBalancedDINO", signal_validity="target discovery descriptive positive", discovery_gain="anomaly images +6.245; CI [5.885,6.605]", composition_change="category coverage -0.325; capture days +1.130", diversification_or_concentration="source_confounded_diversification", category_safety="frozen threshold failed", source_or_session_safety="official-test origin explains 67.3% of gain", acquisition_reproducibility="200 paired leave-20 perturbations", downstream_overall_utility="not tested; task mismatch", downstream_rare_utility="not applicable", operational_utility="selection safety FAIL", dominant_mechanism="source_confounded_diversification", final_decision="DISCOVERY_DESCRIBE / DOWNSTREAM_HOLD", evidence_level="paired_existing_record descriptive", source_file="runs/stratified_condition_robustness_20260718/stratified_effect_summary.csv; docs/metadata_feasibility_decision_20260718.md"),
    row(dataset="VisA", strategy_or_signal="FrozenDINOVisualDiversity", signal_validity="target discovery strong within fixed pool", discovery_gain="anomaly images +14.480; CI [14.195,14.770]", composition_change="categories -4.110; entropy -0.929016; HHI +0.286025", diversification_or_concentration="category_collapse", category_safety="failed in 200/200 paired seeds", source_or_session_safety="source not separable from category", acquisition_reproducibility="200 paired leave-20 perturbations", downstream_overall_utility="not tested; task mismatch", downstream_rare_utility="not applicable", operational_utility="SAFETY_REJECT", dominant_mechanism="category_collapse", final_decision="SAFETY_REJECT", evidence_level="paired_existing_record descriptive", source_file="runs/stratified_condition_robustness_20260718/stratified_effect_summary.csv"),
    row(dataset="GC10-DET", strategy_or_signal="First DINO detector translation q20", signal_validity="selection gate passed", discovery_gain="rare +2.720", composition_change="taxonomy +0.825", diversification_or_concentration="diversification", category_safety="selection positive", source_or_session_safety="proxy only", acquisition_reproducibility="5 acquisition units", downstream_overall_utility="mAP +0.017378", downstream_rare_utility="rare macro -0.019877", operational_utility="frozen gate FAIL", dominant_mechanism="overall_map_rare_safety_tradeoff", final_decision="DETECTOR_REJECT", evidence_level="confirmatory_fail", source_file="runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_development_confirmation_gate.csv"),
    row(dataset="GC10-DET", strategy_or_signal="DINOClusterCoverageK40/140", signal_validity="holdout coverage valid", discovery_gain="rare images +1.820", composition_change="all-class 0.955 vs 0.940", diversification_or_concentration="diversification", category_safety="selection PASS", source_or_session_safety="production-group field is proxy", acquisition_reproducibility="200 design + 200 holdout seeds", downstream_overall_utility="mAP -0.001678", downstream_rare_utility="rare -0.018290; recall -0.021871", operational_utility="downstream FAIL", dominant_mechanism="coverage_utility_gap", final_decision="CLOSE", evidence_level="confirmatory selection pass + downstream fail", source_file="runs/dcal_xai/v2_budget_extended/decision.json; runs/dcal_xai/v2_backbone_main/v8n_screen_decision.json"),
    row(dataset="NEU-DET", strategy_or_signal="Seed45 Visual20 fixed set", signal_validity="fixed selected-set effect", discovery_gain="not primary", composition_change="one fixed query set", diversification_or_concentration="not generalized", category_safety="4/6 class AP nonnegative", source_or_session_safety="not assessed", acquisition_reproducibility="one acquisition set only", downstream_overall_utility="mAP +0.016236; 5/5 training seeds", downstream_rare_utility="not separate endpoint", operational_utility="diagnostic only", dominant_mechanism="fixed_set_training_stability", final_decision="DIAGNOSTIC_ONLY", evidence_level="descriptive_positive", source_file="runs/seed45_fixed_set_stability/seed45_fixed_set_stability_main/screen_gate.csv"),
    row(dataset="NEU-DET", strategy_or_signal="Independent Visual acquisition confirmation", signal_validity="preregistered acquisition test", discovery_gain="not primary", composition_change="10 new selected sets", diversification_or_concentration="variable", category_safety="per-class confirmation not sufficient", source_or_session_safety="not assessed", acquisition_reproducibility="10 acquisition seeds", downstream_overall_utility="mAP +0.007019; CI crosses 0; p=0.322266", downstream_rare_utility="not confirmed", operational_utility="gate FAIL", dominant_mechanism="acquisition_non_generalization", final_decision="CLOSE", evidence_level="confirmatory_negative", source_file="runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main/confirmatory_gate.csv"),
    row(dataset="GC10-DET", strategy_or_signal="V2.3 ensemble detector uncertainty", signal_validity="above-chance error ranking", discovery_gain="not a discovery selector", composition_change="top20% review set", diversification_or_concentration="not evaluated", category_safety="rare support small", source_or_session_safety="not evaluated", acquisition_reproducibility="5 acquisition units x 3 training seeds", downstream_overall_utility="total-error enrichment 1.115901 < 1.50", downstream_rare_utility="FN 1.379693; rare-FN 1.785910 exploratory", operational_utility="primary gate FAIL", dominant_mechanism="ranking_without_operational_enrichment", final_decision="CLOSE_AS_AL", evidence_level="confirmatory fail + exploratory secondary", source_file="runs/dcal_xai/v2_detector_signal_validity/decision.json"),
    row(dataset="GC10-DET; NEU-DET", strategy_or_signal="VLM explanation consistency", signal_validity="invalid under frozen audits", discovery_gain="not authorized", composition_change="not applicable", diversification_or_concentration="grounding_collapse", category_safety="not reached", source_or_session_safety="not reached", acquisition_reproducibility="not reached", downstream_overall_utility="not authorized", downstream_rare_utility="not authorized", operational_utility="0/6 legacy gate checks", dominant_mechanism="grounding_collapse", final_decision="CLOSE", evidence_level="confirmatory_negative", source_file="runs/vlm_consistency_groundedness_validity/legacy_pilot_audit_20260715_main/legacy_validity_gate.csv"),
    row(dataset="GC10-DET", strategy_or_signal="Horizontal-flip disagreement", signal_validity="confounded", discovery_gain="rare +2.0 descriptive", composition_change="query instances -2.2", diversification_or_concentration="geometry_confound", category_safety="instance noninferiority failed", source_or_session_safety="not assessed", acquisition_reproducibility="5 acquisition seeds", downstream_overall_utility="not authorized", downstream_rare_utility="not authorized", operational_utility="selection gate FAIL", dominant_mechanism="geometry_confound", final_decision="CLOSE", evidence_level="diagnostic", source_file="runs/dcal_xai/gc10_r1/selection_summary.md"),
]

MECHANISM_FIELDS = ["dataset", "strategy_or_signal", "signal_validity", "discovery_gain", "composition_change", "diversification_or_concentration", "category_safety", "source_or_session_safety", "acquisition_reproducibility", "downstream_overall_utility", "downstream_rare_utility", "operational_utility", "dominant_mechanism", "final_decision", "evidence_level", "source_file"]


def evidence(eid: str, question: str, dataset: str, protocol: str, strategy: str,
             metric: str, value: Any, uncertainty: str, gate: str, result: str,
             status: str, unit: str, allowed: str, prohibited: str, source: str,
             locator: str) -> dict[str, Any]:
    return row(evidence_id=eid, research_question=question, dataset=dataset,
               protocol=protocol, strategy=strategy, metric=metric, value=value,
               uncertainty=uncertainty, gate=gate, result=result,
               evidence_status=status, inferential_unit=unit,
               allowed_claim=allowed, prohibited_claim=prohibited,
               source_file=source, source_row_or_key=locator, verified=True)


LEDGER = [
    evidence("E001", "Does VLM consistency track groundedness?", "GC10-DET; NEU-DET", "legacy validity audit", "semantic consistency", "Spearman consistency vs groundedness", -0.1811991161, "bootstrap 95% CI [-0.389654, 0.036410]", "rho>=0.20 and CI low>0", "FAIL", "confirmatory_negative", "image (n=95)", "Expected positive relation was not observed.", "Consistency is epistemic uncertainty.", "runs/vlm_consistency_groundedness_validity/legacy_pilot_audit_20260715_main/legacy_validity_metrics.csv", "scope=overall; metric=spearman_consistency_vs_groundedness"),
    evidence("E002", "Does inconsistency detect severe failure?", "GC10-DET; NEU-DET", "legacy validity audit", "1-consistency", "AUC severe failure", 0.3734329797, "bootstrap 95% CI [0.246360, 0.508573]", "AUC>=0.60 and CI low>0.50", "FAIL", "confirmatory_negative", "image (n=95)", "The frozen severe-failure endpoint failed.", "Inconsistency detects failure.", "runs/vlm_consistency_groundedness_validity/legacy_pilot_audit_20260715_main/legacy_validity_metrics.csv", "metric=inconsistency_auc_severe_failure"),
    evidence("E003", "Can structured prompts produce grounded evidence?", "GC10-DET", "structured pilot20", "structured VLM", "schema completeness", 0.01, "n=20 images; 100 responses", ">=0.90", "FAIL", "confirmatory_negative", "image", "Structured output validity collapsed.", "Structured prompting restores validity.", "runs/vlm_consistency_groundedness_validity/structured_prompt_pilot20_gc10_20260715/groundedness_audit/structured_groundedness_metrics.csv", "metric=mean_schema_complete_response_rate"),
    evidence("E004", "Does oracle cropping restore grounded defect evidence?", "GC10-DET", "oracle crop20", "Qwen2-VL", "positive presence/evidence/median IoU", "0 / 0 / 0", "20 positive crops", "presence>=0.50; evidence>=0.70", "FAIL", "confirmatory_negative", "crop image", "Failure persisted in an easier oracle condition.", "Full-image context was the sole failure cause.", "runs/vlm_consistency_groundedness_validity/oracle_crop_diagnostic20_gc10_20260715/oracle_crop_audit/oracle_crop_diagnostic_metrics.csv", "positive_presence_rate; defect_evidence_rate; median_bbox_iou"),
    evidence("E005", "Does another small Qwen model restore compliance and grounding?", "GC10-DET", "3-model paired comparison", "Qwen2/Qwen3/Qwen2.5", "models passing frozen gate", 0, "3 models x 40 paired views", "all paired checks", "FAIL", "confirmatory_negative", "paired view within model", "Different collapse modes were observed; none passed.", "A model swap solved VLM grounding.", "runs/vlm_consistency_groundedness_validity/paired_model_comparison_gc10_20260715/comparison/paired_model_gate_comparison.csv", "gate_pass=False for 3/3"),
    evidence("E006", "Is Random weak in balanced large pools?", "NEU-DET", "V10c24 Random audit", "GTFreeRandom", "class/instance/bbox selection properties", "not weak", "post-hoc audited selected sets", "audit", "SUPPORTED_BASELINE", "partially_supported", "selected set", "Random provided broad coverage and richness in this protocol.", "Random is universally optimal.", "runs/random_baseline_audit_v10c24/random_baseline_audit_20260713_225216/random_baseline_audit_summary.md", "summary conclusion"),
    evidence("E007", "Does V10c24 scale at round2 budget120?", "NEU-DET", "seed51 round2", "V10c24", "mAP50-95 difference vs Random", -0.0048630617, "single acquisition realization", "six frozen scale checks", "FAIL", "confirmatory_negative", "acquisition realization", "The scale extension failed.", "The selector scales to budget120.", "runs/active_learning_v10c24_round2_scale_smoke/v10c24_round2_scale_smoke_20260713_225326/recovered_v10c24_round2_scale_smoke_summary.md", "round2 mAP50-95 difference"),
    evidence("E008", "Is the seed45 fixed selected set stable to training seed?", "NEU-DET", "1 set x 5 train seeds", "Visual20", "mAP50-95 difference", 0.0162356649, "5/5 wins; descriptive CI approx [0.006738,0.027864]", "fixed-set gate", "PASS_DIAGNOSTIC", "descriptive_positive", "training seed conditional on one set", "One fixed set had a stable learner effect.", "Selector generalization was established.", "runs/seed45_fixed_set_stability/seed45_fixed_set_stability_main/screen_gate.csv", "treatment=Visual20"),
    evidence("E009", "Does seed45 effect generalize to new acquisition sets?", "NEU-DET", "10 acquisition seeds", "Visual diversity", "mAP50-95 difference", 0.0070187, "CI [-0.005211,0.019678]; p=0.322266; 7/10 wins", "gain>=0.01; p<=0.05; recall safety", "FAIL", "confirmatory_negative", "acquisition seed", "Acquisition-set generalization failed.", "Cold-start DINO superiority was confirmed.", "runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main/confirmatory_gate.csv", "primary paired mAP row"),
    evidence("E010", "Does GC10 DINO improve rare discovery?", "GC10-DET", "initial20/query20; 200 paired seeds", "FrozenDINOVisualDiversity", "rare images gain vs Random", 2.72, "paired seed CI [2.455,2.980]; positive=0.88", "selection-only 11 checks", "PASS_SELECTION", "partially_supported", "acquisition seed/leave-20 pool", "Rare discovery increased in the fixed benchmark.", "Detector utility or production generalization improved.", "runs/stratified_condition_robustness_20260718/stratified_effect_summary.csv", "dataset=GC10-DET; stratum=all"),
    evidence("E011", "Does GC10 discovery preserve taxonomy composition?", "GC10-DET", "initial20/query20; 200 paired seeds", "FrozenDINOVisualDiversity", "combined unique classes delta", 0.825, "paired records", "selection safety", "PASS_SELECTION", "partially_supported", "acquisition seed/leave-20 pool", "Taxonomy coverage increased in this selection endpoint.", "Coverage equals learning utility.", "runs/stratified_condition_robustness_20260718/stratified_effect_summary.csv", "coverage_delta"),
    evidence("E012", "Does MPDD DINO improve anomaly discovery?", "MPDD", "initial20/query20; 200 paired seeds", "FrozenCategoryBalancedDINO", "anomaly image gain", 6.245, "paired seed CI [5.885,6.605]; positive=0.985", "category delta>=-0.25", "FAIL_SAFETY", "descriptive_positive", "acquisition seed/leave-20 pool", "Anomaly discovery increased while safety failed.", "MPDD selector passed or improved downstream utility.", "runs/stratified_condition_robustness_20260718/stratified_effect_summary.csv", "dataset=MPDD; primary selector"),
    evidence("E013", "What composition change accompanies MPDD discovery?", "MPDD", "same frozen selections", "FrozenCategoryBalancedDINO", "category coverage delta", -0.325, "200 paired seeds", "category safety", "FAIL", "confirmatory_negative", "acquisition seed", "Category safety threshold failed.", "Discovery was composition-neutral.", "runs/stratified_condition_robustness_20260718/stratified_effect_summary.csv", "coverage_delta"),
    evidence("E014", "How much MPDD gain aligns with official-test source?", "MPDD", "source-origin audit", "FrozenCategoryBalancedDINO", "source-composition contribution", "67.3%", "descriptive decomposition", "not a causal gate", "DESCRIPTIVE", "descriptive_positive", "fixed dataset decomposition", "Source composition is a major confound.", "Source causes the full anomaly gain.", "docs/stratified_condition_robustness_decision_20260718.md", "MPDD dataset results"),
    evidence("E015", "Does VisA DINO improve anomaly discovery?", "VisA", "initial20/query20; 200 paired seeds", "FrozenDINOVisualDiversity", "anomaly image gain", 14.48, "paired seed CI [14.195,14.770]; positive=1.0", "category safety", "FAIL_SAFETY", "descriptive_positive", "acquisition seed/leave-20 pool", "Discovery increased in all paired seeds.", "The selector was safe or useful for learning.", "runs/stratified_condition_robustness_20260718/stratified_effect_summary.csv", "dataset=VisA; stratum=all"),
    evidence("E016", "What safety cost accompanies VisA discovery?", "VisA", "same frozen selections", "FrozenDINOVisualDiversity", "object-category coverage delta", -4.11, "200/200 gain-positive+safety-loss", "category coverage not worse", "FAIL", "confirmatory_negative", "acquisition seed", "Category collapse accompanied the gain.", "Anomaly discovery implies category safety.", "runs/stratified_condition_robustness_20260718/stratified_effect_summary.csv", "coverage_delta"),
    evidence("E017", "Does GC10 q20 selection translate to detector utility?", "GC10-DET", "5 acquisitions x 3 train seeds", "Frozen DINO vs Random", "overall mAP50-95 delta", 0.0173784069, "5 acquisition units; development", "mAP and rare safety jointly", "FAIL", "partially_supported", "acquisition seed", "Overall mAP rose in this protocol only.", "The downstream gate passed.", "runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_acquisition_seed_summary.csv", "overall paired difference"),
    evidence("E018", "Does GC10 q20 protect rare utility?", "GC10-DET", "same detector confirmation", "Frozen DINO vs Random", "rare macro AP50-95 delta", -0.0198767455, "classes 8/9/10; development", "rare gain>=0.020", "FAIL", "confirmatory_negative", "acquisition seed", "Rare-class utility declined.", "Overall mAP gain implies rare safety.", "runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_class_group_macro_summary.csv", "rare macro paired difference"),
    evidence("E019", "Does K40 reduce taxonomy omission?", "GC10-DET", "budget140 holdout; 200 seeds", "DINOClusterCoverageK40", "all-class rate", 0.955, "Random=0.940", "holdout coverage gate", "PASS_SELECTION", "partially_supported", "acquisition seed/leave-140 pool", "K40 reduced zero-coverage risk.", "K40 improves detector learning.", "runs/dcal_xai/v2_budget_extended/budget_summary.csv", "split=holdout; strategy=DINOClusterCoverageK40"),
    evidence("E020", "Does K40 coverage translate to mAP?", "GC10-DET", "5 acquisitions x 3 train seeds", "K40/140 vs Random140", "mAP50-95 delta", -0.0016777446, "bootstrap CI [-0.017424,0.018332]", "downstream screen", "FAIL", "confirmatory_negative", "acquisition seed", "Coverage did not translate to overall utility.", "K40 is noninferior or superior.", "runs/dcal_xai/v2_backbone_main/v8n_screen_contrasts.csv", "metric=map5095"),
    evidence("E021", "Does K40 protect rare AP?", "GC10-DET", "same screen", "K40/140 vs Random140", "rare macro AP delta", -0.0182895351, "classes 8/9/10", "rare noninferiority", "FAIL", "confirmatory_negative", "acquisition seed", "Rare AP declined.", "Coverage ensures rare-class utility.", "runs/dcal_xai/v2_backbone_main/v8n_screen_decision.json", "contrasts.rare_macro_ap5095_difference"),
    evidence("E022", "Does K40 protect recall?", "GC10-DET", "same screen", "K40/140 vs Random140", "recall delta", -0.0218708249, "5 acquisition units", "recall noninferiority", "FAIL", "confirmatory_negative", "acquisition seed", "Recall declined.", "Coverage ensures recall safety.", "runs/dcal_xai/v2_backbone_main/v8n_screen_contrasts.csv", "metric=recall"),
    evidence("E023", "Can label-aware global retrieval recover class 8?", "GC10-DET", "1,000 frozen replay states", "nearest target exemplar", "class8 top1 hit", 0.1538461538, "351 class8 states; 130 seeds", "top1>=0.50 plus six checks", "FAIL", "confirmatory_negative", "frozen replay state", "Global representation branch was not recovered.", "Label-aware retrieval solved local defect representation.", "runs/gc10_discovery_representation_audit/gc10_d2r_200seed_20260715/label_aware_retrieval_audit/label_aware_class8_summary.csv", "method=nearest_target_exemplar"),
    evidence("E024", "Is flip disagreement a safe detector difficulty signal?", "GC10-DET", "DCAL-XAI R1; 5 seeds", "DetectorDifficultyDiversity", "query instance delta", -2.2, "bootstrap CI [-5.2,0.8]", "instance noninferiority", "FAIL", "confirmatory_negative", "acquisition seed", "Geometry-dominated selection failed a safety gate.", "Flip disagreement is epistemic uncertainty.", "runs/dcal_xai/gc10_r1/selection_comparisons.csv", "metric=query_instances"),
    evidence("E025", "Does V2.3 provide actionable total-error enrichment?", "GC10-DET", "15 frozen checkpoints; 232 development images", "ensemble combined uncertainty", "top20% total-error enrichment", 1.1159008048, "bootstrap CI [1.034640,1.221093]", ">=1.50", "FAIL", "confirmatory_negative", "acquisition realization (n=5)", "The ranking was above chance but operational effect size was insufficient.", "V2.3 AL signal passed.", "runs/dcal_xai/v2_detector_signal_validity/signal_validity_summary.csv", "signal=ensemble_combined_uncertainty"),
    evidence("E026", "Does V2.3 rank majority error?", "GC10-DET", "same V2.3 audit", "ensemble combined uncertainty", "majority-error AUROC", 0.7664320783, "5/5 acquisition seeds >0.5", ">=0.65", "PASS_COMPONENT", "partially_supported", "acquisition realization", "Above-chance ranking does not equal top-budget utility.", "AUROC alone validates AL utility.", "runs/dcal_xai/v2_detector_signal_validity/signal_validity_summary.csv", "signal=ensemble_combined_uncertainty; auroc_mean"),
    evidence("E027", "Does V2.3 enrich false negatives?", "GC10-DET", "same reused development audit", "ensemble combined uncertainty", "FN enrichment", 1.3796926705, "secondary endpoint", "not preregistered primary", "HYPOTHESIS_ONLY", "exploratory_only", "acquisition realization", "Motivates a future untouched-pool protocol only.", "Recall-repair triage is confirmed.", "runs/dcal_xai/v2_detector_signal_validity/signal_validity_summary.csv", "fn_enrichment_mean"),
    evidence("E028", "Does V2.3 enrich rare false negatives?", "GC10-DET", "same reused development audit", "ensemble combined uncertainty", "rare-FN enrichment", 1.7859103354, "18 rare images / 22 boxes", "not preregistered primary", "HYPOTHESIS_ONLY", "exploratory_only", "acquisition realization", "Rare-FN signal is small-sample hypothesis generation.", "Rare-defect safety improved.", "runs/dcal_xai/v2_detector_signal_validity/signal_validity_summary.csv", "rare_fn_enrichment_mean"),
    evidence("E029", "Are 200 seeds independent production pools?", "GC10-DET", "pool design audit", "leave-20 pool perturbation", "mean pool Jaccard", 0.978446, "200 highly overlapping pools", "independent pool gate", "NO", "not_identifiable", "fixed benchmark pool", "Paired same-pool effects are estimable.", "Seeds are independent production pools.", "docs/stratified_audit_design_feasibility_20260718.md", "GC10 design row"),
    evidence("E030", "Are 200 seeds independent production pools?", "MPDD", "pool design audit", "leave-20 pool perturbation", "mean pool Jaccard", 0.962827, "200 highly overlapping pools", "independent pool gate", "NO", "not_identifiable", "fixed benchmark pool", "Paired same-pool effects are estimable.", "Seeds are independent production pools.", "docs/stratified_audit_design_feasibility_20260718.md", "MPDD design row"),
    evidence("E031", "Are 200 seeds independent production pools?", "VisA", "pool design audit", "leave-20 pool perturbation", "mean pool Jaccard", 0.995386, "200 highly overlapping pools", "independent pool gate", "NO", "not_identifiable", "fixed benchmark pool", "Paired same-pool effects are estimable.", "Seeds are independent production pools.", "docs/stratified_audit_design_feasibility_20260718.md", "VisA design row"),
    evidence("E032", "Can benchmark metadata define independent pools?", "GC10-DET; MPDD; VisA", "metadata feasibility audit", "EXIF/filename/category groups", "datasets passing independent-pool gate", 0, "0/3 datasets", ">=20 documented target-blind mixed groups", "NO_GO", "not_identifiable", "dataset", "Current benchmarks cannot test production generalization.", "Metadata proxies are production lots.", "docs/metadata_feasibility_decision_20260718.md", "Dataset decisions table"),
    evidence("E033", "How much planned compute was explicitly avoided?", "all branches", "branch registry", "validity gates", "prevented detector model runs", 45, "minimum documented: D2R 15 + YOLOv8s 30", "documented only", "SUPPORTED", "partially_supported", "planned model run", "At least 45 detector model runs were not launched.", "GPU-hours or money saved.", "docs/three_dataset_condition_map_decision_20260718.md", "Validity-gated workflow value"),
    evidence("E034", "Was locked final test consumed?", "all branches", "branch registry", "final-test protection", "final-test evaluations", 0, "all documented branches", "must equal 0", "PASS", "partially_supported", "branch", "The locked final test remained unused.", "Final-test performance or generalization.", "docs/branch_decision_table_20260718.csv", "final_test_consumed=False for all rows"),
]

LEDGER_FIELDS = ["evidence_id", "research_question", "dataset", "protocol", "strategy", "metric", "value", "uncertainty", "gate", "result", "evidence_status", "inferential_unit", "allowed_claim", "prohibited_claim", "source_file", "source_row_or_key", "verified"]


KOREAN_TITLES = [
    (1, "산업 결함 능동학습 후보 신호의 단계적 타당성 평가: 발견-안전성-학습 효용 간극 분석", "초기 문제와 현재 증거를 가장 정확히 묶는다.", "새 selector가 없어 방법론 기여가 약하다는 질문을 받을 수 있다."),
    (2, "산업 결함 검출을 위한 Validity-Gated Active Learning 평가 체계", "평가 workflow를 전면에 둔다.", "실증된 failure mechanism을 부제나 초록에서 구체화해야 한다."),
    (3, "산업 결함 Active Learning에서 후보 신호의 발견 효용과 검출 학습 효용의 분리", "selection-learning gap이 선명하다.", "VLM과 metadata 감사 범위가 제목에서 덜 드러난다."),
    (4, "산업 결함 데이터의 Annotation-Efficient Learning을 위한 Acquisition Signal Audit", "annotation triage와 AL을 함께 포괄한다.", "국문 제목에 영문 용어가 많다."),
    (5, "산업 결함 능동학습의 실패 조건과 중단 규칙에 관한 실증 연구", "negative/conditional contribution이 솔직하다.", "긍정적 discovery 결과가 제목에서 약해 보일 수 있다."),
]

ENGLISH_TITLES = [
    (1, "Validity-Gated Evaluation of Acquisition Signals for Industrial Defect Active Learning: Discovery, Safety, and Learning-Utility Gaps", "Covers the full empirical contribution without a superiority claim.", "Long title; may be shortened after advisor review."),
    (2, "A Validity-Gated Audit of Active Learning Signals for Industrial Defect Detection", "Compact and methodologically clear.", "The discovery-safety distinction must be explicit in the abstract."),
    (3, "When Selection Gains Do Not Translate: An Empirical Study of Industrial Defect Active Learning", "Memorable and centered on the strongest translation result.", "Could sound exclusively negative unless balanced in the subtitle."),
    (4, "Separating Discovery, Composition Safety, and Detector Utility in Industrial Active Learning", "Directly names the three stages.", "The VLM validity branch is implicit rather than explicit."),
    (5, "Failure-Condition Mapping and Stopping Rules for Annotation-Efficient Industrial Defect Learning", "Highlights operational value and stopping discipline.", "Active Learning should be prominent in keywords and abstract."),
]


def make_evolution_doc() -> str:
    return f"""# Research evolution and Evidence Freeze v2

Date: {DATE}  
Final synthesis decision: **A. THESIS_REFRAME_STRONGLY_SUPPORTED**  
This rating applies only to the evidence-based thesis reframe. It does not revive the rejected selector-superiority hypothesis.

## 1. 최초 연구 질문

초기 연구는 생성형 VLM의 반복 설명이 흔들릴수록 모델 지식이 부족하다는 직관에서 출발했다. 초기 발표의 H1은 explanation consistency와 실제 분류/위치 오류의 상관, H2는 동일 label budget에서 Random·Entropy보다 빠른 mAP 향상이었다. 전체 연구계획서는 이를 (i) linguistic epistemic uncertainty의 형식화, (ii) HDAL-LB 대비 +2~3% mAP, (iii) 언어 정렬 기반 domain adaptation +3~5% mAP, (iv) 0.2 s/image 로컬 배포, (v) 50~80% label-cost 절감과 현장 신뢰 향상으로 확장했다.

당시 문헌적 출발점은 반복 생성의 semantic disagreement, AL uncertainty/diversity, VLM prompting, visual grounding이었다. 그러나 문헌지도는 동시에 `consistency != truth`, selector(VLM)-learner(YOLO) cross-model risk, seed/retraining 교락, strong Random 가능성을 명시했다. 이후 실험은 바로 이 경고를 단계별로 검증한 과정이 되었다.

## 2. 최초 가설의 단계적 검증

| 단계 | 질문과 protocol | 핵심 결과 | frozen gate | 결정과 다음 단계 |
|---|---|---|---|---|
| VLM consistency | 95-image legacy consistency-groundedness audit | Spearman -0.181199; severe-failure AUC 0.373433 | 0/6 | **FAIL**. H1을 기각하고 grounding을 직접 감사했다. |
| Groundedness | GC10 structured prompt pilot20 | schema 0.01; bbox coverage 0.30; median IoU 0 | FAIL | Prompt 재작성만으로 부족하여 oracle crop diagnostic으로 이동했다. |
| Oracle crop | positive oracle crops 20 | presence 0; evidence 0; median IoU 0 | FAIL | Full-image 난이도 설명을 기각했다. |
| Cross-model paired probe | 3 Qwen-family models x 40 paired views | 0/3 pass; best balanced accuracy 0.70 with IoU 0 | FAIL | VLM acquisition-score branch를 종료했다. |
| DINO diversity | NEU/GC10/MPDD/VisA frozen global representation | fixed-pool discovery gain은 존재했으나 category/source/local-defect 문제가 반복 | dataset-specific | 범용 utility 주장 대신 discovery와 composition을 분리했다. |
| Random baseline audit | NEU V10c24 selected-set properties | class coverage, instance diversity, bbox richness가 이미 강함 | audit | Random을 weak baseline으로 취급하지 않도록 고정했다. |
| Acquisition confirmation | seed45 fixed set 후 10 new acquisition seeds | fixed set +0.016236; new sets +0.007019, CI crosses 0, p=0.322266 | FAIL | training-seed stability != acquisition generalization을 확인했다. |
| Coverage-utility translation | GC10 q20 and K40/140 detector screens | q20 mAP +0.017378 but rare -0.019877; K40 mAP -0.001678, rare -0.018290, recall -0.021871 | FAIL | coverage와 learner utility를 별도 gate로 고정했다. |
| Detector uncertainty validity | 15 existing checkpoints, sealed predictions, 232 development images | total-error enrichment 1.115901 <1.50; AUROC 0.766432 | FAIL | AL signal은 종료; FN endpoints는 exploratory only로 남겼다. |
| Cross-dataset discovery/safety | 200 paired seeds per dataset | GC10 +2.720; MPDD +6.245; VisA +14.480, but dataset-specific safety losses | mixed | same-pool discovery effect만 유지했다. |
| Stratified identifiability | reconstructed candidate pools | mean Jaccard 0.978446/0.962827/0.995386 | prevalence gate 0/3 | sparsity law를 `not identifiable`로 고정했다. |
| Metadata feasibility | acquisition manifests + EXIF + frozen selections | documented independent pool groups 0/3 | NO-GO | production generalization 주장을 차단했다. |

## 3. 명확히 기각된 중심 주장

- 설명 consistency는 epistemic uncertainty의 직접 근사치다.
- VLM consistency selector는 Random보다 안정적으로 우수하다.
- Global DINO coverage는 detector learning utility를 보장한다.
- Horizontal-flip disagreement는 epistemic detector uncertainty다.
- 범용 detector uncertainty는 top-budget total-error utility를 충분히 농축한다.
- 현재 benchmark에서 target sparsity가 selector gain을 조절한다는 법칙을 식별할 수 있다.
- 현재 metadata로 independent-production generalization을 검증할 수 있다.
- HDAL-LB 대비 +2~3% mAP, label cost 50~80% 절감, domain adaptation +3~5%, inspector trust 향상을 달성했다.

## 4. 실제로 확인된 positive evidence와 동반 위험

| Evidence | Positive effect | 동시에 확인된 경계 |
|---|---:|---|
| GC10 q20 discovery | rare images +2.720; CI [2.455,2.980] | q20 detector rare macro AP -0.019877 |
| MPDD q20 discovery | anomaly images +6.245; CI [5.885,6.605] | category -0.325; source-composition contribution 67.3% |
| VisA q20 discovery | anomaly images +14.480; CI [14.195,14.770] | categories -4.110; HHI +0.286025; 200/200 safety loss |
| GC10 first detector translation | overall mAP +0.017378 | rare macro AP -0.019877; frozen gate FAIL |
| Seed45 fixed set | mAP +0.016236; 5/5 training seeds | independent acquisition confirmation FAIL |
| V2.3 ranking | majority-error AUROC 0.766432 | total-error enrichment 1.115901 <1.50 |
| V2.3 secondary FN | FN 1.379693; rare-FN 1.785910 | post-hoc, 18 rare images/22 boxes, no untouched validation |

## 5. 반복 확인된 failure mechanisms

1. **Consistent hallucination / grounding collapse**: 문법적으로 유효한 응답과 시각적으로 grounded한 응답은 달랐다. Oracle crop과 model swap에서도 복구되지 않았다.
2. **Global-representation/local-defect mismatch**: DINO는 제품·배경·세션·texture 차이를 포착했지만 작은 결함 semantics와 detector utility를 일관되게 포착하지 못했다.
3. **Category/source/session concentration**: VisA category collapse와 MPDD source-origin contribution이 discovery gain의 안전성을 제한했다.
4. **Acquisition-set non-generalization**: 한 고정 집합의 training-seed 안정성은 새 선택 집합에서 재현되지 않았다.
5. **Coverage-utility gap**: taxonomy/rare coverage 증가는 rare AP, recall, mAP를 자동 보장하지 않았다.
6. **Overall mAP-rare safety trade-off**: 평균 성능의 양의 차이가 희소 class 안전성과 공존하지 않았다.
7. **Ranking-operational enrichment gap**: AUROC와 correlation이 높아도 작은 review budget의 enrichment는 운영 threshold에 못 미칠 수 있다.
8. **Metadata-independent-pool validity gap**: metadata 존재 자체가 independent production pool을 의미하지 않는다.

## 6. 살아남은 중심 연구 질문

> 산업 결함 데이터에서 Active Learning 후보 신호의 **signal validity, target discovery, composition safety, acquisition-set 재현성, downstream detector utility와 operational enrichment는 어떻게 분리되며**, 고비용 학습과 locked evaluation 전에 이 번역 실패를 어떻게 판정할 수 있는가?

## 7. 현재 중심 기여

**Primary contribution**  
산업 결함 AL 후보 신호를 `Signal -> Discovery -> Composition Safety -> Acquisition Reproducibility -> Learning Utility -> Operational Validity`로 분해하고, 다음 단계의 고비용 실행을 frozen gate로만 허용하는 **validity-gated empirical evaluation framework**를 제시·적용했다.

**Secondary contributions**

1. 세 fixed benchmark에서 same-pool discovery gain과 category/source safety가 분리됨을 paired records로 정량화했다.
2. GC10의 두 detector translation protocol과 NEU acquisition confirmation을 통해 selection coverage/training stability가 learner utility/acquisition generalization과 다름을 보였다.
3. VLM grounding, flip disagreement, detector uncertainty, metadata를 각각 신호·운영·식별 가능성 단계에서 감사하여 구체적 stopping condition을 제시했다.

## 8. 연구 범위와 한계

- GC10/MPDD/VisA 결과는 fixed benchmark의 same-pool paired effects다.
- Bootstrap CI는 acquisition-seed perturbation variation이며 production-population CI가 아니다.
- 독립 production lot/time/source pool이 없다.
- Locked final test는 한 번도 사용하지 않았다. 따라서 final-test 성능을 보고하지 않는다.
- FN/rare-FN은 exploratory이고 untouched validation이 없어 실행을 중단했다.
- GC10은 multi-class object detection, MPDD/VisA는 anomaly localization/discovery이므로 raw yield나 downstream metric을 합산·평균하지 않는다.
- 동일 workflow를 적용했지만 branch별 gate와 target 정의는 task-specific이다.

## 9. Evidence Freeze v2

### 유지 주장

- Fixed benchmark same-pool에서 target discovery gain은 존재한다.
- Discovery와 composition safety는 분리될 수 있다.
- GC10에서 selection coverage는 rare AP/recall/mAP로 자동 번역되지 않았다.
- Training-seed stability는 acquisition-set generalization이 아니다.
- Validity gate는 최소 45개의 명시된 detector run과 final-test 소비를 방지했다.

### 폐기 주장

- 범용 selector superiority, VLM epistemic proof, sparsity law, production generalization, 성능 향상 보장, 비용·신뢰·도메인 적응 개선.

### Exploratory only

- FN enrichment 1.379693, rare-FN 1.785910, confidence rare-FN 2.683022.

### Protected status

- Final test used: **False**.
- New training/inference/embedding/selector/FN screen executed in this synthesis: **False**.

### Stopped branches

- VLM consistency/grounding, small-model search, V10c/V10d scaling, D2R representation repair, flip disagreement, K40 detector expansion, V2.3 general-purpose AL, FN implementation.

### Reusable assets

- Paired selection records, manifests, DINO caches, detector checkpoints, sealed predictions, gate code, protocol documents, metadata provenance, unit tests and evidence ledger.
"""


def make_claim_boundary_doc() -> str:
    return """# Thesis claim boundary

## 반드시 주장해야 하는 것

- 산업 결함 AL 후보 신호의 `signal-discovery-safety-reproducibility-learning-operational` 단계는 서로 대체할 수 없다.
- Fixed benchmark same-pool paired records에서 GC10, MPDD, VisA의 target discovery gain이 확인됐다.
- Discovery gain은 VisA category collapse, MPDD source confound, GC10 rare-utility loss와 공존했다.
- GC10 K40 selection coverage PASS는 downstream mAP/rare AP/recall PASS로 번역되지 않았다.
- NEU 한 고정 선택 집합의 training-seed 안정성은 새로운 acquisition-set confirmation을 통과하지 못했다.
- 모든 branch에서 final test를 사용하지 않았고, frozen gate가 최소 45개 후속 detector model run을 차단했다.

## 조건부로 주장할 수 있는 것

- GC10 rare discovery +2.720, MPDD anomaly discovery +6.245, VisA anomaly discovery +14.480은 **동일 fixed acquisition pool의 200 paired leave-20 perturbations** 범위에서만 말한다.
- GC10 first translation의 overall mAP +0.017378은 해당 development protocol의 부분 효과이며 rare AP -0.019877과 항상 함께 보고한다.
- MPDD capture-day 및 GC10 filename-sequence 결과는 mechanism sensitivity proxy다. Production lot/generalization이 아니다.
- V2.3 AUROC 0.766432는 above-chance ranking evidence이며 operational AL validity가 아니다.

## Exploratory로만 남겨야 하는 것

- Combined FN enrichment 1.379693.
- Combined rare-FN enrichment 1.785910.
- Confidence-only rare-FN enrichment 2.683022.
- Local feature misalignment, recall-repair triage, human-facing explanation interface.

## 절대로 주장하면 안 되는 것

- 새 selector가 Random보다 일반적으로 우수하다.
- Target sparsity가 gain을 조절한다는 법칙을 확인했다.
- 200 seeds는 200 independent production pools다.
- MPDD EXIF day는 production lot이다.
- GC10 filename group은 official production group이다.
- VisA category는 capture session이다.
- VLM consistency가 epistemic uncertainty임을 증명했다.
- Label cost 50-80% 절감, detector performance improvement guarantee, inspector trust 향상, domain adaptation 향상, 0.2 s/image deployment를 달성했다.

## 용어 사용 규칙

| 용어 | 허용 사용 | 금지 사용 |
|---|---|---|
| validity | 사전 정의된 endpoint/gate 충족 여부 | 일반적 진실성의 동의어 |
| robustness | 명시된 training/acquisition seed 범위 | production robustness |
| generalization | 새 acquisition sets 등 실제 독립 축을 명시 | fixed-set 반복을 일반화로 표현 |
| independent pool | source-documented, target-blind production unit만 | leave-k perturbation, category, EXIF proxy |
| uncertainty | 계산된 signal 이름 또는 후보 신호 | epistemic uncertainty로 단정 |
| diversity | 사용한 metric(pairwise similarity, coverage, entropy)을 명시 | 시각적으로 다양하다는 모호한 표현 |
| discovery | fixed query budget의 target image count | detector utility와 동일시 |
| safety | category/source/rare/recall gate의 구체적 endpoint | 모든 위험을 포괄하는 표현 |
| utility | detector mAP/rare AP/recall 또는 review enrichment를 명시 | coverage/discovery를 utility로 승격 |
| efficiency | 실제 model-run prevention처럼 측정된 값 | 시간·비용 미측정 상태의 경제성 주장 |
| label cost reduction | matched-performance annotation cost를 측정한 경우만 | 50-80% 기대값을 결과로 사용 |
| explainability | 생성 설명의 존재 또는 interface 기능 | grounding/인간 신뢰가 검증됐다는 표현 |
"""


def make_outline_doc() -> str:
    title_lines = ["# Reframed thesis outline", "", "## Recommended titles", "", "### Korean"]
    for rank, title, strength, risk in KOREAN_TITLES:
        title_lines += [f"{rank}. **{title}**", f"   - 장점: {strength}", f"   - 위험: {risk}"]
    title_lines += ["", "### English"]
    for rank, title, strength, risk in ENGLISH_TITLES:
        title_lines += [f"{rank}. **{title}**", f"   - Strength: {strength}", f"   - Risk: {risk}"]

    chapters = [
        ("제1장 서론", "산업 결함 annotation 비용; acquisition signal 검증 문제; 강한 Random; 연구 질문과 기여", "Table 1 initial-to-revised hypothesis summary", "Fig. 1 research hypothesis evolution", "초기 계획과 최종 질문의 차이", "비용 50-80%, 신뢰 향상 수치", "왜 새 알고리즘이 아니라 평가 연구인가?"),
        ("제2장 관련 연구", "AL uncertainty/diversity; object detection AL; VLM semantic uncertainty; groundedness/hallucination; industrial anomaly/defect; evaluation pitfalls", "Table 2 literature-to-risk map", "없음 또는 taxonomy diagram", "문헌이 예고한 consistency!=truth, cross-model risk, Random strength", "검증되지 않은 현장 통계", "Negative result의 novelty는 무엇인가?"),
        ("제3장 Validity-Gated Evaluation Framework", "Signal, Selection, Composition Safety, Acquisition Generalization, Learning Utility, Operational Validity; gates and stopping", "Table 3 gate definitions and authorization", "Fig. 2 evidence pyramid", "다음 단계는 이전 gate PASS로만 허용", "하나의 universal threshold 주장", "Branch-specific gate가 사후적이지 않은가?"),
        ("제4장 VLM 신호 타당성 감사", "legacy consistency; structured grounding; oracle crop; paired model comparison", "Table 4 VLM validity results", "response-collapse examples; claim boundary inset", "0/6 legacy and 0/3 model pass; grounding collapse", "VLM acquisition superiority", "모델 크기를 키우면 달라지지 않는가?"),
        ("제5장 Selection Discovery와 Composition Safety", "GC10, MPDD, VisA paired effects; DINO mechanism; metadata audit", "Table 5 dataset-specific discovery/safety; Table 6 metadata feasibility", "Fig. 3 discovery-composition-utility matrix", "Same-pool discovery exists, safety is dataset-specific", "Cross-dataset raw average; sparsity law", "200 seeds가 독립 pool인가?"),
        ("제6장 Selection-Learning Translation", "Random audit; q20 translation; K40; seed45; independent confirmation", "Table 7 translation contrasts; Table 8 Random properties", "selection-learning translation diagram", "coverage/training stability do not guarantee utility/generalization", "positive mAP without rare loss; training seeds as independent acquisitions", "왜 overall mAP +0.017을 성공이라 하지 않는가?"),
        ("제7장 Detector-Native Signal과 Operational Gate", "flip disagreement; V2.3 sealed predictions; ranking vs enrichment; FN exploratory", "Table 9 signal validity and operational thresholds", "AUROC vs enrichment schematic", "above-chance ranking can fail operational threshold", "FN triage confirmation", "AUROC 0.766이면 충분하지 않은가?"),
        ("제8장 종합 논의", "discovery!=safety; safety!=utility; stability!=generalization; metadata!=production; cost prevention/final protection", "Table 10 mechanism matrix; Table 11 claim boundary", "Fig. 4 claim boundary map", "Repeated translation failures form the empirical contribution", "production generalization or universal laws", "Checklist 이상의 학술적 기여는 무엇인가?"),
        ("제9장 결론 및 향후 연구", "primary/secondary contributions; limits; independent production pools; FN restart conditions; separate supervised detector branch", "Table 12 future authorization criteria", "none", "현재 evidence freeze와 future protocol의 분리", "추가 selector 결과 예측", "추가 실험 없이 논문을 써도 되는가?"),
    ]
    lines = title_lines + ["", "## Chapter plan", ""]
    for name, content, tables, figures, core, excluded, question in chapters:
        lines += [f"### {name}", "", f"- 내용: {content}", f"- 사용할 표: {tables}", f"- 사용할 그림: {figures}", f"- 사용할 실험/핵심 주장: {core}", f"- 제외할 내용: {excluded}", f"- 예상 심사 질문: {question}", ""]
    return "\n".join(lines)


def make_advisor_brief() -> str:
    return """# 지도교수 결정 보고: Evidence Freeze v2

## 연구 방향 변경 요약

최초 연구는 VLM Self-Consistency를 epistemic uncertainty로 간주하고, 이를 DINO diversity·detector uncertainty와 결합한 selector가 Random보다 detector mAP를 안정적으로 높인다는 가설에서 출발했습니다. 초기 계획에는 +2~3% mAP, 50~80% label-cost 절감, domain adaptation, local deployment, 검사원 신뢰 향상까지 포함됐습니다.

현재 원자료는 이 중심 가설을 지지하지 않습니다. VLM consistency-groundedness gate, oracle crop, 세 소형 VLM 비교가 모두 실패했고, NEU의 seed45 positive set은 새 acquisition seeds에서 재현되지 않았습니다. GC10에서는 selection coverage가 좋아져도 rare AP·recall·mAP로 번역되지 않았습니다. 따라서 논문의 중심을 새 selector의 성능 우위가 아니라 **산업 결함 AL 후보 신호의 단계적 타당성과 번역 실패 조건을 검증하는 empirical evaluation**으로 변경하고자 합니다.

## 무엇이 실패했는가

`낮은 consistency = 높은 epistemic uncertainty`, `VLM/DINO/uncertainty selector > Random`, `global coverage = detector utility`, `flip disagreement = epistemic uncertainty`, `above-chance error ranking = operational AL utility`, `200 seeds = independent production generalization`이라는 연결이 성립하지 않았습니다. 초기의 +2~3% mAP, label-cost 50~80%, domain adaptation, inspector trust 목표도 달성 또는 검증되지 않았습니다.

## 무엇이 연구 자산으로 남았는가

- **Positive evidence**: GC10 rare discovery +2.720, MPDD anomaly +6.245, VisA anomaly +14.480; seed45 fixed-set mAP +0.016236; V2.3 AUROC 0.766432.
- **Failure mechanisms**: VLM grounding collapse, global/local representation mismatch, category/source concentration, acquisition-set non-generalization, coverage-utility gap, ranking-operational enrichment gap, metadata-independent-pool gap.
- **Workflow**: Signal -> Discovery -> Safety -> Acquisition Reproducibility -> Learning Utility -> Operational Validity의 단계별 frozen gate.
- **Cost prevention**: 명시적으로 계획된 detector model run 최소 45개(D2R 15 + YOLOv8s 30)를 시작하지 않았고 final test는 0회 사용했습니다. GPU-hours나 금액은 측정 자료가 없어 주장하지 않습니다.
- **Reusable assets**: 200-seed selection records, manifests, DINO cache, YOLO checkpoints, 15개 sealed Random140 predictions, gate code, metadata provenance, unit tests.

## 새 중심 연구 질문

> 산업 결함 데이터에서 Active Learning 후보 신호의 signal validity, target discovery, composition safety, acquisition-set 재현성, downstream detector utility와 operational enrichment는 어떻게 분리되며, 고비용 학습과 locked evaluation 전에 이 번역 실패를 어떻게 판정할 수 있는가?

## 새 중심 기여

1. 산업 결함 AL 후보 신호를 여섯 단계로 분해하고 다음 단계 실행을 frozen gate로 제한하는 validity-gated evaluation framework.
2. GC10·MPDD·VisA에서 fixed-pool discovery gain과 composition safety가 분리됨을 보인 paired empirical evidence.
3. GC10/NEU에서 selection coverage와 training-seed stability가 detector utility/acquisition generalization을 보장하지 않음을 보인 translation evidence.

## 현재 추가 실험이 불필요한 이유

추가 selector search는 기존 failure mechanism을 해결하는 독립 자원 없이 development reuse와 사후 method search를 늘립니다. 현재 benchmark metadata는 independent production pools를 만들지 못하며, FN branch도 untouched non-final bbox validation이 없습니다. 따라서 지금은 결과를 더 만드는 것보다 claim boundary와 evidence provenance를 동결하고 본문을 작성하는 편이 연구 윤리에 맞습니다. Locked final은 계속 보호합니다.

## 교수님께 결정받아야 할 항목

- **A. 권고**: Validity-gated empirical evaluation을 학위 중심 기여로 인정하고 논문 작성을 시작한다.
- **B. 조건부**: 제한적 positive algorithmic contribution을 추가로 요구한다면, 새 independent production pool/untouched validation 확보 전까지 구현은 보류한다.
- **C. 분리**: supervised detector/backbone/small-defect capacity 연구가 필요하면 본 AL evidence와 별도 연구 질문·split·protocol로 분리한다.

## 교수님께 구두로 말씀드릴 1분 설명

“교수님, 처음에는 VLM 설명 일관성과 DINO·detector uncertainty를 결합한 selector가 Random보다 안정적으로 성능을 높일 것으로 예상했습니다. 하지만 consistency와 grounding은 oracle crop과 세 모델에서도 유효성을 통과하지 못했고, NEU의 positive seed도 새 acquisition set에서는 재현되지 않았습니다. 반면 GC10, MPDD, VisA에서 target discovery 자체는 실제로 증가했지만, category/source safety 또는 rare-class detector utility와 분리됐습니다. 특히 GC10에서는 coverage가 좋아져도 mAP, rare AP, recall로 번역되지 않는 결과가 두 protocol에서 확인됐습니다. 그래서 selector 성공을 주장하는 대신, 신호·발견·안전성·재현성·학습 효용·운영 효용을 단계적으로 검증하고 실패 시 학습과 final test를 중단하는 validity-gated 평가를 논문의 중심으로 바꾸고 싶습니다. 현재 결과만으로 본문 작성은 가능하다고 판단하지만, 이 empirical evaluation을 학위의 중심 기여로 인정할지, 아니면 별도의 supervised detector 연구를 분리해 요구하실지 결정 부탁드립니다.”
"""


def build_source_registry() -> list[dict[str, Any]]:
    source_paths: set[str] = set()
    metric_registry = ROOT / "runs" / "integrated_condition_map_20260718" / "source_metric_registry.csv"
    if metric_registry.exists():
        with metric_registry.open("r", encoding="utf-8-sig", newline="") as handle:
            for item in csv.DictReader(handle):
                for part in item["source_file"].split(";"):
                    if part.strip():
                        source_paths.add(part.strip())
    for collection in (HYPOTHESES, ASSETS, MECHANISMS, LEDGER):
        for item in collection:
            key = "source_file" if "source_file" in item else "file_or_directory"
            for part in str(item.get(key, "")).split(";"):
                if part.strip():
                    source_paths.add(part.strip())
    source_paths.update(str(p) for p in INITIAL_PDFS)

    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(sorted(source_paths), start=1):
        ensure_no_locked_path(raw)
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / raw.replace("/", "\\")
        exists = path.exists()
        kind = "external_pdf" if path.suffix.lower() == ".pdf" else ("directory" if path.is_dir() else "file")
        digest = sha256(path) if exists and path.is_file() and path.stat().st_size < 100_000_000 else ""
        rows.append(row(source_id=f"S{index:03d}", path=rel(path), source_kind=kind,
                        exists=exists, sha256=digest, provenance_status="verified" if exists else "referenced_missing_or_compound",
                        note="Compound semicolon entries are split; directories are existence-checked only."))
    return rows


def missing_evidence_rows() -> list[dict[str, Any]]:
    return [
        row(item="independent production pools", status="missing_resource", reason="0/3 datasets pass the documented metadata gate", impact="production generalization and prevalence moderation are not identifiable", required_for_restart="source-documented target-blind lot/time/source pools"),
        row(item="untouched non-final GC10 bbox validation", status="missing_resource", reason="development already reused; final locked", impact="FN/rare-FN branch remains exploratory", required_for_restart="new untouched native-bbox pool"),
        row(item="field annotation cost/time", status="not_measured", reason="no matched-performance annotation study", impact="50-80% cost reduction cannot be claimed", required_for_restart="prospective annotation-time/cost protocol"),
        row(item="inspector trust/acceptance", status="not_measured", reason="no human-subject or field pilot", impact="explainability and trust improvement cannot be claimed", required_for_restart="ethics-approved human study after grounding validity"),
        row(item="domain adaptation benefit", status="not_tested", reason="language alignment experiment not executed", impact="+3-5% transfer claim excluded", required_for_restart="separate supervised domain-shift study"),
        row(item="local deployment latency", status="not_tested", reason="no frozen 0.2 s/image benchmark", impact="deployment target excluded", required_for_restart="hardware-specific latency/memory benchmark"),
        row(item="GPU-hours and monetary savings", status="missing_original_artifact", reason="runtime registry is incomplete across branches", impact="only prevented model-run count is reported", required_for_restart="prospective resource accounting"),
        row(item="MPDD/VisA downstream detector utility", status="task_mismatch", reason="pixel-mask anomaly localization/discovery tasks", impact="not zero-filled or pooled with GC10", required_for_restart="task-appropriate downstream model and independent protocol"),
    ]


def configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = [
        Path(r"C:\Windows\Fonts\malgun.ttf"),
        Path(r"C:\Windows\Fonts\malgunsl.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            name = font_manager.FontProperties(fname=str(candidate)).get_name()
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    return plt


def save_figures() -> None:
    plt = configure_matplotlib()
    from matplotlib.patches import FancyBboxPatch, Polygon

    FIGURES.mkdir(parents=True, exist_ok=True)
    navy, blue, green, amber, red, gray = "#16324F", "#2F6B9A", "#3F7D57", "#C58A2B", "#B84A4A", "#6B7280"

    # 1. Research hypothesis evolution
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.2); ax.axis("off")
    ax.text(0.4, 6.72, "연구 가설의 전이: selector 우위에서 번역 타당성 감사로", fontsize=19, weight="bold", color=navy)
    stages = [
        (0.5, "초기 가설", "VLM consistency\n→ epistemic uncertainty\n→ Random보다 우수", red, "H1/H2 기각\nSpearman -0.181\nAUC 0.373"),
        (4.8, "중간 확장", "Groundedness + DINO diversity\n+ detector uncertainty", amber, "부분 신호와 붕괴\nOracle presence 0\nK40 utility FAIL"),
        (9.1, "최종 연구 질문", "Signal → Discovery → Safety\n→ Reproducibility → Utility\n→ Operational validity", green, "Validity-gated audit\n최소 45 runs 방지\nFinal test 0회"),
    ]
    for x, title, body, color, result in stages:
        ax.add_patch(FancyBboxPatch((x, 3.15), 3.5, 2.55, boxstyle="round,pad=0.18", facecolor="#F7F9FB", edgecolor=color, linewidth=2.2))
        ax.text(x+0.2, 5.28, title, fontsize=15, weight="bold", color=color)
        ax.text(x+0.2, 4.62, body, fontsize=12.2, va="top", color=navy, linespacing=1.45)
        ax.add_patch(FancyBboxPatch((x+0.15, 1.05), 3.2, 1.45, boxstyle="round,pad=0.15", facecolor=color, edgecolor="none", alpha=0.12))
        ax.text(x+0.3, 2.18, result, fontsize=11.5, va="top", color=navy, linespacing=1.35)
    for start in (4.1, 8.4):
        ax.annotate("", xy=(start+0.52, 4.4), xytext=(start-0.05, 4.4), arrowprops=dict(arrowstyle="->", lw=2.2, color=gray))
    ax.text(0.5, 0.35, "핵심: 초기 답은 기각되었지만, 무엇이 다음 단계로 번역되지 않는지 판정하는 연구 질문은 살아남았다.", fontsize=11.8, color=gray)
    fig.tight_layout(); fig.savefig(FIGURES / "research_hypothesis_evolution.png", bbox_inches="tight"); plt.close(fig)

    # 2. Discovery-composition-utility matrix
    fig, ax = plt.subplots(figsize=(14, 7.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.4); ax.axis("off")
    ax.text(0.4, 6.92, "Discovery–Composition Safety–Learning Utility Matrix", fontsize=19, weight="bold", color=navy)
    headers = ["Target discovery", "Composition / proxy safety", "Downstream learning utility"]
    for j, header in enumerate(headers):
        ax.text(4.1+j*3.15, 6.28, header, ha="center", fontsize=12.5, weight="bold", color=navy)
    rows = [
        ("GC10-DET", [("+2.720 rare images", green), ("+0.825 classes\n+1.465 filename groups", green), ("q20 rare AP -0.019877\nK40 mAP -0.001678", red)]),
        ("MPDD", [("+6.245 anomalies", green), ("category -0.325\ncapture-day +1.130\nsource confound 67.3%", amber), ("Not tested\n(task mismatch)", gray)]),
        ("VisA", [("+14.480 anomalies", green), ("categories -4.110\nHHI +0.286025", red), ("Not tested\n(safety rejected)", gray)]),
    ]
    for i, (name, cells) in enumerate(rows):
        y = 4.85 - i*1.65
        ax.text(0.55, y+0.55, name, fontsize=15, weight="bold", color=navy, va="center")
        for j, (label, color) in enumerate(cells):
            x = 2.58 + j*3.15
            ax.add_patch(FancyBboxPatch((x, y), 2.95, 1.18, boxstyle="round,pad=0.12", facecolor=color, alpha=0.14, edgecolor=color, linewidth=1.5))
            ax.text(x+1.475, y+0.59, label, ha="center", va="center", fontsize=11.4, color=navy, linespacing=1.25)
    ax.text(0.55, 0.15, "Green = endpoint-positive within frozen scope · Amber = confounded/trade-off · Red = safety/translation failure · Gray = not tested", fontsize=10.8, color=gray)
    fig.tight_layout(); fig.savefig(FIGURES / "discovery_composition_utility_matrix.png", bbox_inches="tight"); plt.close(fig)

    # 3. Evidence pyramid
    fig, ax = plt.subplots(figsize=(12, 8.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(0.4, 8.52, "Evidence pyramid: 확인된 층과 남은 공백", fontsize=19, weight="bold", color=navy)
    levels = [
        (0.9, 10.2, "Raw data · code · checkpoints", "확보", green),
        (1.55, 8.9, "Signal diagnostics", "확보: VLM / DINO / detector", green),
        (2.2, 7.6, "Selection evidence", "확보: 200 paired seeds", green),
        (2.85, 6.3, "Acquisition confirmation", "부분: NEU 10 seeds, FAIL", amber),
        (3.5, 5.0, "Downstream utility", "GC10 두 protocol, mixed/FAIL", amber),
        (4.15, 3.7, "Operational / production generalization", "미확보: independent pools 없음", red),
    ]
    base_y = 0.75
    for idx, (left, width, label, state, color) in enumerate(levels):
        y0 = base_y + idx*1.15
        x0, x1 = left, left+width
        top_inset = 0.45
        poly = Polygon([(x0,y0),(x1,y0),(x1-top_inset,y0+0.92),(x0+top_inset,y0+0.92)], closed=True, facecolor=color, edgecolor="white", alpha=0.18)
        ax.add_patch(poly)
        ax.text(6.0, y0+0.58, label, ha="center", va="center", fontsize=12.6, weight="bold", color=navy)
        ax.text(6.0, y0+0.24, state, ha="center", va="center", fontsize=10.5, color=gray)
    ax.text(0.65, 0.23, "논문은 현재 도달한 층까지만 주장한다. 상위 operational/production 층은 future work다.", fontsize=11.2, color=gray)
    fig.tight_layout(); fig.savefig(FIGURES / "evidence_pyramid.png", bbox_inches="tight"); plt.close(fig)

    # 4. Claim boundary map
    fig, ax = plt.subplots(figsize=(15, 8.4))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8.4); ax.axis("off")
    ax.text(0.4, 7.95, "Claim boundary map", fontsize=19, weight="bold", color=navy)
    columns = [
        ("SUPPORTED", green, ["단계별 utility 분리", "GC10 coverage–utility gap", "Final test 0회", "최소 45 runs 방지"]),
        ("CONDITIONAL", blue, ["GC10 +2.720", "MPDD +6.245", "VisA +14.480", "fixed-pool paired scope"]),
        ("EXPLORATORY", amber, ["FN 1.379693", "rare-FN 1.785910", "local feature branch", "human interface"]),
        ("REJECTED", red, ["VLM epistemic proxy", "selector superiority", "global DINO utility", "flip epistemic signal"]),
        ("NOT IDENTIFIABLE", gray, ["sparsity law", "production generalization", "population CI", "metadata lot claim"]),
    ]
    for j, (title, color, items) in enumerate(columns):
        x = 0.35 + j*2.92
        ax.add_patch(FancyBboxPatch((x, 1.05), 2.68, 5.95, boxstyle="round,pad=0.14", facecolor=color, alpha=0.10, edgecolor=color, linewidth=1.5))
        ax.text(x+1.34, 6.55, title, ha="center", fontsize=12.5, weight="bold", color=navy)
        for i, item in enumerate(items):
            ax.text(x+0.22, 5.75-i*1.08, "• " + item, fontsize=10.8, color=navy, va="top", wrap=True)
    ax.text(0.4, 0.38, "Supported ≠ universal. Conditional claims must retain dataset, protocol, budget, and inferential unit.", fontsize=11.2, color=gray)
    fig.tight_layout(); fig.savefig(FIGURES / "claim_boundary_map.png", bbox_inches="tight"); plt.close(fig)


def validate_internal() -> None:
    if any(h["evidence_status"] not in STATUS for h in HYPOTHESES):
        raise RuntimeError("Invalid hypothesis evidence_status")
    if any(e["evidence_status"] not in STATUS for e in LEDGER):
        raise RuntimeError("Invalid ledger evidence_status")
    ids = [e["evidence_id"] for e in LEDGER]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate evidence IDs")
    if any(not e["source_file"] or not e["source_row_or_key"] for e in LEDGER):
        raise RuntimeError("Every ledger item must have a source locator")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Evidence Freeze v2 without training or inference.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_internal()
    for path in INITIAL_PDFS:
        if not path.exists():
            raise FileNotFoundError(path)

    config = {
        "date": DATE,
        "decision": "A_THESIS_REFRAME_STRONGLY_SUPPORTED",
        "decision_scope": "thesis reframe only; original selector superiority rejected",
        "training_performed": False,
        "inference_performed": False,
        "vlm_calls_performed": False,
        "embedding_extraction_performed": False,
        "selector_implementation_performed": False,
        "fn_screen_performed": False,
        "final_test_used": False,
        "evidence_freeze": True,
    }
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_OK", "hypotheses": len(HYPOTHESES), "assets": len(ASSETS), "mechanisms": len(MECHANISMS), "ledger": len(LEDGER), **config}, ensure_ascii=False, indent=2))
        return

    DOCS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    write_csv(DOCS / "hypothesis_transition_matrix_20260718.csv", HYPOTHESES, HYPOTHESIS_FIELDS)
    write_csv(DOCS / "research_asset_reclassification_20260718.csv", ASSETS, ASSET_FIELDS)
    write_csv(DOCS / "acquisition_mechanism_matrix_20260718.csv", MECHANISMS, MECHANISM_FIELDS)
    write_csv(OUT / "research_evidence_ledger.csv", LEDGER, LEDGER_FIELDS)
    sources = build_source_registry()
    write_csv(OUT / "source_registry.csv", sources, ["source_id", "path", "source_kind", "exists", "sha256", "provenance_status", "note"])
    write_csv(OUT / "missing_evidence.csv", missing_evidence_rows(), ["item", "status", "reason", "impact", "required_for_restart"])
    write_text(DOCS / "research_evolution_and_evidence_freeze_v2_20260718.md", make_evolution_doc())
    write_text(DOCS / "thesis_claim_boundary_20260718.md", make_claim_boundary_doc())
    write_text(DOCS / "reframed_thesis_outline_20260718.md", make_outline_doc())
    write_text(DOCS / "advisor_decision_brief_evidence_freeze_v2_20260718.md", make_advisor_brief())
    save_figures()
    (OUT / "freeze_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    commands = [
        r".\.venv\Scripts\python.exe scripts\04_dcal_xai\build_evidence_freeze_v2.py --dry-run",
        r".\.venv\Scripts\python.exe scripts\04_dcal_xai\build_evidence_freeze_v2.py",
        r".\.venv\Scripts\python.exe scripts\04_dcal_xai\test_evidence_freeze_v2.py",
    ]
    write_text(OUT / "executed_commands.txt", "\n".join(commands))
    expected = [
        DOCS / "hypothesis_transition_matrix_20260718.csv",
        DOCS / "research_asset_reclassification_20260718.csv",
        DOCS / "research_evolution_and_evidence_freeze_v2_20260718.md",
        DOCS / "acquisition_mechanism_matrix_20260718.csv",
        DOCS / "thesis_claim_boundary_20260718.md",
        DOCS / "reframed_thesis_outline_20260718.md",
        DOCS / "advisor_decision_brief_evidence_freeze_v2_20260718.md",
        FIGURES / "research_hypothesis_evolution.png",
        FIGURES / "discovery_composition_utility_matrix.png",
        FIGURES / "evidence_pyramid.png",
        FIGURES / "claim_boundary_map.png",
        OUT / "research_evidence_ledger.csv",
        OUT / "source_registry.csv",
        OUT / "missing_evidence.csv",
        OUT / "freeze_config.json",
        OUT / "executed_commands.txt",
        OUT / "generated_file_manifest.txt",
        OUT / "test_results.txt",
        ROOT / "scripts" / "04_dcal_xai" / "build_evidence_freeze_v2.py",
        ROOT / "scripts" / "04_dcal_xai" / "test_evidence_freeze_v2.py",
    ]
    write_text(OUT / "generated_file_manifest.txt", "\n".join(rel(path) for path in expected))
    print(json.dumps({"status": "DONE", "decision": config["decision"], "hypotheses": len(HYPOTHESES), "assets": len(ASSETS), "mechanisms": len(MECHANISMS), "ledger": len(LEDGER), "sources": len(sources), "final_test_used": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

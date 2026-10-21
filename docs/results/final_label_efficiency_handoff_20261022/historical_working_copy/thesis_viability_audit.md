# Thesis Viability Audit

## Executive verdict

**종합 판정: B — 조건부로 석사학위논문 성립 가능.**

현재 repository는 새 selector의 우월성을 입증하는 패키지는 아니다. 그러나 평가한 산업 결함 Active Learning branch에서 selection-stage discovery, selected-set composition, acquisition/fixed-set reproducibility, downstream learner utility, rare-class safety가 서로 다른 판정을 만들 수 있음을 추적한 **bounded multi-branch empirical audit**로는 학위논문이 성립한다.

이 판정에는 조건이 붙는다.

- 공통 benchmark, 보편 법칙, Gate의 예측 정확도, 실제 비용 절감, 독립 production pool 일반화를 주장하지 않는다.
- `docs/translation_atlas/`는 유용하지만 **post-hoc integration**임을 본문 전면에 표시한다.
- 서로 다른 데이터셋·task·metric의 효과를 수치적으로 pooling하지 않는다.
- KSDD2의 `TRANSLATION_NOT_OBSERVED_STOP`을 “효과 없음”으로 번역하지 않고, **통계적으로 양수이나 사전 최소효과 미확립**으로 쓴다.
- DeepPCB와 MPDD/VisA처럼 learner outcome이 없는 STOP branch는 downstream 실패 사례가 아니라 **outcome unavailable / boundary case**로 둔다.

이 조건을 지키지 않고 “검증된 framework”, “산업 benchmark”, “좋은 selector를 사전에 판별”, “비용 절감”을 전면에 내세우면 판정은 **C 이하**로 내려간다.

## 감사 범위와 판정 기준

본 감사는 repository 안에 실제 존재하는 다음 계층을 대조했다.

- 통합 레지스트리: `docs/translation_atlas/translation_branch_inventory.csv`
- 통합 통계: `docs/translation_atlas/translation_atlas_statistics.json`
- claim audit: `docs/claim_evidence_gap_matrix.csv`
- 원자료/결정 파일: 각 `runs/**` CSV/JSON과 `docs/results/**`
- 제한 및 설계 감사: `docs/framework_holdout_confusion_matrix_20260718.csv`, `docs/metadata_feasibility_decision_20260718.md`, `docs/stratified_audit_design_feasibility_20260718.md`
- 시간적·연구사 감사: `docs/final_repository_research_audit_20260730.md`

“강하게 주장 가능”은 해당 데이터셋·selector·budget·split·learner 범위 안에서 직접 결과와 source locator가 있고, 반대 endpoint도 함께 보고할 수 있다는 뜻이다. 외부 산업 일반화나 인과성을 뜻하지 않는다.

## 1. 현재 자료만으로 강하게 주장 가능한 명제 3개

### 1.1 GC10 q20에서는 rare-image discovery와 rare-class learner safety가 같은 방향으로 움직이지 않았다

Frozen DINO visual diversity는 Random보다 20-image query에서 rare-class image를 평균 `+2.720`장 더 선택했고 95% CI는 `[+2.455, +2.980]`이었다. Downstream YOLOv8n의 전체 mAP50-95 delta도 `+0.017378`, acquisition-seed bootstrap CI `[+0.000580, +0.036332]`로 양수였다. 그러나 rare-class macro AP delta는 `-0.019877`이었고 frozen rare-safety Gate는 실패했다.

이 자료로 강하게 말할 수 있는 것은 “GC10의 이 frozen branch에서 discovery와 overall learner metric의 개선이 rare-class safety를 보장하지 않았다”이다. DINO가 일반적으로 위험하다거나 rare-class 성능을 항상 악화시킨다고 말할 수는 없다.

근거:

- `runs/gc10_taxonomy_selection_audit/gc10_random_vs_dino_200seed_20260715/gc10_metric_summary.csv`
- `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_acquisition_seed_summary.csv`
- `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_class_group_macro_summary.csv`
- `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_development_confirmation_gate.csv`

### 1.2 MPDD와 VisA에서는 discovery gain과 selected-set composition safety가 동시에 성립하지 않았다

MPDD에서 FrozenCategoryBalancedDINO는 query anomaly image를 Random보다 평균 `+6.245`장 더 선택했고 CI는 `[+5.885, +6.605]`이었다. 동시에 unique product-category coverage delta는 `-0.325`였고 frozen noninferiority threshold `-0.25`를 통과하지 못했다. VisA에서는 anomaly image delta가 `+14.480`, CI `[+14.195, +14.770]`이었지만 unique object-category coverage delta는 `-4.110`으로 악화됐다.

따라서 이 두 fixed-pool selection audit에서는 target discovery positivity와 composition safety가 분리된 endpoint라는 주장이 강하다. 다만 learner를 실행하지 않았으므로 이를 “learning failure”라고 부르면 안 된다.

근거:

- `runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_metric_summary.csv`
- `runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_selection_only_gate.csv`
- `runs/visa_selection_only_audit/visa_random_vs_visual_200seed_20260715/visa_metric_summary.csv`
- `runs/visa_selection_only_audit/visa_random_vs_visual_200seed_20260715/visa_selection_only_gate.csv`

### 1.3 KSDD2에서는 강한 selection discovery와 양의 learner gain이 관찰됐지만 사전 최소효과는 확립되지 않았다

Prospective KSDD2 branch에서 frozen PatchCore score는 100-image query당 positive defect image를 Random보다 평균 `+88.8`장 더 선택했고 CI는 `[+86.9, +90.6]`이었다. Compact U-Net 60-run package의 macro Dice delta는 `+0.037565`, CI `[+0.019581, +0.057029]`, positive acquisition-seed fraction은 `0.8`이었다. Pixel AP delta `+0.049824`, PatchCore normal FPR `0.160129`, FPR delta `+0.010573`도 기록돼 있다. 그러나 사전 primary minimum인 mean Dice delta `≥ +0.05`는 실패했다.

방어 가능한 문장은 “learner effect는 통계적으로 양의 방향이었지만 frozen practical minimum을 확립하지 못했다”이다. “translation이 없었다”, “PatchCore가 무효다”, “PatchCore가 우월하다”는 세 문장 모두 현재 결과보다 강하다.

근거:

- `scripts/06_ksdd2_prospective/config.yaml`
- `scripts/07_ksdd2_g5_bounded_learner/config.yaml`
- `docs/results/ksdd2_patchcore_prospective_20260726/g2_g4/per_seed_metrics.csv`
- `docs/results/ksdd2_g5_single_unet_20260727/g5_authorization_decision.json`
- `docs/results/ksdd2_g5_single_unet_20260727/paired_seed_metrics.csv`

## 2. 제한적으로만 주장 가능한 명제 3개

### 2.1 평가가 완료된 translation branch에서 selection positivity는 full downstream Gate PASS를 보장하지 않았다

GC10 q20, GC10 K40/140, KSDD2의 completed translation branch는 모두 full frozen Gate PASS가 아니었다(`0/3`). 이 관찰은 현재 branch 집합의 기술적 결론으로는 유효하다. 그러나 n=3이고 task·metric·threshold·retrospective/prospective status가 다르므로 실패 확률, meta-effect, 산업 일반 법칙으로 해석할 수 없다.

허용 문장: “평가한 completed branches에서는 positive selection endpoint가 full downstream authorization을 보장하지 않았다.”

금지 문장: “positive selection은 일반적으로 learning으로 번역되지 않는다.”

근거: `docs/translation_atlas/translation_branch_inventory.csv`, `docs/translation_atlas/translation_atlas_statistics.json`.

### 2.2 한 fixed selected set의 training-seed 안정성은 acquisition reproducibility와 같지 않았다

NEU seed45 fixed-set diagnostic은 Visual20의 mAP50-95 delta `+0.016236`과 `5/5` training-seed wins를 기록했다. 그러나 10개의 새 acquisition seed confirmation에서는 mAP delta가 `+0.007019`, CI `[-0.005211, +0.019678]`, exact sign-flip `p=0.322266`이었고 frozen minimum과 유의성 check를 통과하지 못했다.

이는 재현성 대상을 분리해야 한다는 좋은 사례지만, favorable fixed set이 결과 관찰 후 선택된 diagnostic이고 새 acquisition에서도 별도 selection-primary endpoint가 없었다. 따라서 한 데이터셋의 methodological warning으로만 사용해야 한다.

근거:

- `runs/seed45_fixed_set_stability/seed45_fixed_set_stability_main/screen_gate.csv`
- `runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main/confirmatory_metric_summary.csv`
- `runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main/confirmatory_gate.csv`

### 2.3 단계 분리형 translation taxonomy는 기존 결과를 정직하게 감사하는 데 유용하다

Composition failure, learning-direction failure, rare-safety conflict, positive-but-below-minimum이라는 네 fate는 repository의 실제 결과를 손실 적게 정리한다. 그러나 taxonomy와 공통 six-stage framing은 여러 branch 결과를 본 뒤 만들어진 post-hoc synthesis다. 따라서 검증된 예측 framework나 새로운 보편 이론이 아니라 **분석·보고 체계**로만 주장해야 한다.

근거: `docs/translation_atlas/four_fates_taxonomy.csv`, `docs/translation_atlas/translation_atlas_analysis.md`, `docs/final_repository_research_audit_20260730.md`.

## 3. 삭제하거나 축소해야 하는 주장 3개

### 3.1 “Authorization Workflow가 좋은 selector를 높은 정확도로 예측한다” — 삭제

`docs/framework_holdout_confusion_matrix_20260718.csv`의 predicted-decision과 true/false stop·advance 항목은 `NOT_IDENTIFIABLE` 또는 `NA`다. STOP branch의 downstream counterfactual을 관찰하지 않았고 generic policy도 전체 branch 전에 공통으로 동결되지 않았다. 남길 수 있는 것은 “일부 prospective branch에서 정해진 STOP 규칙을 준수했다”는 process-conformance 주장뿐이다.

### 3.2 “실제 GPU-hour, annotation time 또는 금액을 절감했다” — 삭제 또는 계획 containment로 축소

Repository에는 prevented model runs `45`라는 계획상 집계가 있지만 실제 hardware-normalized GPU-hour, annotation active time, energy, 금액 원장은 없다. KSDD2의 defect-pixel/component burden proxy도 annotation time이 아니다.

허용 문장: “사전에 정의한 규칙에 따라 계획된 후속 run을 실행하지 않았다.”

금지 문장: “GPU 비용과 annotation 비용을 절감했다.”

근거: `docs/branch_decision_table_20260718.csv`, `runs/evidence_freeze_v3_20260718/research_evidence_ledger.csv`, `docs/results/ksdd2_g5_single_unet_20260727/posthoc/audit_summary.json`.

### 3.3 “현재 패키지는 benchmark이거나 특정 selector/원인의 보편성을 입증한다” — 삭제

데이터셋마다 task, selector, budget, selection endpoint, learner, threshold가 다르고 독립 production pool과 공통 evaluation API가 없다. 따라서 benchmark, universal selector superiority, 최초성, global-DINO/learner mismatch의 인과적 확정은 방어할 수 없다. Representation mismatch는 가능한 설명으로만 남기고, threshold는 “산업적으로 최적”이 아니라 “outcome 전에 고정한 decision threshold”로 불러야 한다.

근거: `docs/claim_evidence_gap_matrix.csv`의 C015, C018, C019, C020과 `docs/metadata_feasibility_decision_20260718.md`.

## 4. 이 논문이 왜 필요한가를 정당화하는 가장 설득력 있는 논리 2개

### 4.1 Selection metric은 downstream utility의 단일 대리변수가 아니다

현재 repository에는 같은 “selection positivity”가 composition failure(MPDD/VisA), learner direction failure(GC10 K40), rare-class safety conflict(GC10 q20), positive-but-practically-insufficient learner gain(KSDD2)으로 갈라지는 직접 사례가 있다. 이 결과는 새 selector를 제안하지 않아도 가치가 있다. 실제 AL 연구에서 discovery, set composition, learner average performance, subgroup safety, practical minimum을 한 성공 지표로 축약하면 서로 다른 실패를 놓칠 수 있음을 수치로 보여 주기 때문이다.

### 4.2 실패·부분 성공·관찰 불가능을 분리하는 evidence discipline이 필요하다

DeepPCB는 selection Gate 미달 뒤 learner를 실행하지 않았고, KSDD2는 learner delta가 양수였지만 minimum effect 미달로 STOP했다. 두 branch를 모두 단순 “실패”로 쓰면 정보가 사라진다. 반대로 STOP을 예측 정확도나 비용 절감으로 부풀리면 식별 불가능한 주장이 된다. 본 논문의 실질적 필요성은 **어디까지 관찰됐고 어디서 주장을 멈춰야 하는지**를 endpoint·시간적 status·NA와 함께 보여 주는 데 있다.

## 5. 심사에서 공격받을 가장 큰 약점 5개

| 순위 | 공격 지점 | 심사위원이 물을 질문 | 냉정한 판정 |
|---:|---|---|---|
| 1 | Post-hoc 통합과 이질적 branch | “결과를 본 뒤 실패 사례를 한 taxonomy로 묶은 것 아닌가?” | 맞다. Atlas는 post-hoc이며 공통 사전등록 연구가 아니다. |
| 2 | 독립 production pool 부재 | “200 seeds가 정말 200개의 독립 산업 조건인가?” | 아니다. 같은 fixed pool의 leave-20 perturbation이며 mean Jaccard가 GC10 `0.978446`, MPDD `0.962827`, VisA `0.995386`이다. |
| 3 | 불완전한 downstream outcome matrix | “STOP branch가 실제로는 downstream에서 성공했을 가능성은?” | 배제할 수 없다. MPDD, VisA, DeepPCB는 learner outcome이 NA이고 Gate precision/recall도 식별 불가능하다. |
| 4 | Practical threshold의 외부 타당성 | “DeepPCB 1.20, KSDD2 Dice +0.05가 왜 산업적으로 의미 있는가?” | outcome 전 동결됐다는 근거는 있지만 stakeholder cost/quality나 외부 calibration 근거는 약하다. |
| 5 | 연구사 적응성·제한된 learner/selector 범위 | “여러 번의 방향 전환과 개발 split 재사용이 결론 선택에 영향을 준 것 아닌가?” | 위험이 있다. 많은 branch가 retrospective이고 NEU favorable fixed set은 post-observation diagnostic이다. KSDD2만 상대적으로 강한 prospective anchor다. |

## 6. 약점을 방어하기 위해 실제 자료로 제시할 수 있는 근거

| 약점 | 제시할 파일/표/그래프 | 제시할 수치·사실 | 방어 가능한 선 |
|---|---|---|---|
| Post-hoc·이질성 | `docs/translation_atlas/translation_branch_inventory.csv`; `docs/translation_atlas/translation_atlas_statistics.json`; `docs/translation_atlas/translation_atlas_main_figure.png` | `analysis_status=POSTHOC_INTEGRATION_NOT_GATE_CHANGING`; cross-dataset numeric pooling `false`; branch별 native metric 표시 | Atlas는 meta-analysis가 아니라 auditable case matrix라고 명시한다. |
| 독립 pool 부재 | `docs/stratified_audit_design_feasibility_20260718.md`; `docs/metadata_feasibility_decision_20260718.md` | GC10/MPDD/VisA mean pool Jaccard `0.978446/0.962827/0.995386`; no dataset passes independent-pool Gate | “fixed-pool acquisition perturbation reproducibility”로 이름을 제한한다. |
| downstream NA·counterfactual 부재 | `docs/framework_holdout_confusion_matrix_20260718.csv`; `docs/translation_atlas/four_fates_taxonomy.csv` | predicted decision과 true/false stop이 NA; MPDD/VisA/DeepPCB learner `NOT_AVAILABLE` | NA를 실패로 채우지 않고 별도 boundary row로 공개한다. |
| threshold calibration | `runs/deeppcb_reference_residual_gate/prospective_main_20260718/gate_decision.json`; `docs/results/ksdd2_g5_single_unet_20260727/g5_authorization_decision.json`; 두 protocol/config | DeepPCB `1.107693 < 1.20`; KSDD2 `+0.037565 < +0.05`; KSDD2 CI lower `+0.019581` | Threshold는 precommitted decision rule이며 optimal/경제적 threshold가 아니라고 쓴다. |
| 연구사 적응성 | `docs/final_repository_research_audit_20260730.md`; `docs/research_evolution_and_evidence_freeze_v3_20260718.md`; `docs/claim_evidence_gap_matrix.csv` | branch별 retrospective/prospective status, 지원 수준, overclaim risk가 기록됨 | retrospective evidence, post-hoc diagnostic, prospective screen을 표에서 분리한다. |
| rare-class 안전성 공격 | GC10 acquisition/class summary와 Gate CSV | overall mAP `+0.017378`, rare macro AP `-0.019877`, recall `+0.035381`; rare check만 실패 | 평균 성능과 subgroup safety가 충돌한 한 bounded 사례로 제시한다. |
| “KSDD2도 사실 성공 아닌가?” | `g5_authorization_decision.json`, `paired_seed_metrics.csv` | Dice CI는 양수지만 mean `+0.037565`, threshold `+0.05`; 60/60 runs, rescue/test access false | statistical positivity와 practical sufficiency를 분리한다. |
| “KSDD2도 사실 실패 아닌가?” | 같은 파일 및 `RESULT_ANALYSIS_KO.md` | CI `[+0.019581,+0.057029]`, pixel AP `+0.049824`, FPR checks PASS | effect absence가 아니라 positive-but-below-minimum이라고 답한다. |
| 재현성 개념 혼동 | NEU fixed-set screen과 10-acquisition confirmation | fixed set `+0.016236`, 5/5; acquisition `+0.007019`, CI crosses zero, `p=0.322266` | training-seed 안정성과 acquisition reproducibility를 별도 열로 둔다. |
| final-test leakage 의심 | `runs/evidence_freeze_v3_20260718/research_evidence_ledger.csv`; KSDD2 freeze/decision files | `final_test_used=False`, `official_test_accessed=false`, rescue unauthorized | 날짜·package 범위를 명시해 locked-test 보존 사실만 주장한다. |

핵심 방어 전략은 약점을 부정하는 것이 아니다. 각 약점을 표의 열로 노출해 “분석자가 뒤늦게 감춘 조건”이 아니라 “논문이 명시적으로 평가한 경계”로 바꾸는 것이다.

## 7. 현재 상태의 성립 가능성 평가

### 최종 등급: B

#### B를 주는 이유

1. 단일 anecdote가 아니라 GC10, MPDD, VisA, NEU, DeepPCB, KSDD2에 걸쳐 서로 다른 stage relation이 실제 파일과 수치로 남아 있다.
2. GC10 q20과 KSDD2에는 selection과 downstream learner를 연결한 직접 record가 있고, KSDD2는 60-run prospective screen-stage package다.
3. negative, NA, partial positive를 구분한 claim boundary와 source locator가 비교적 강하다.
4. 석사논문으로서 방법론적 질문은 명확하다. “어떤 selector가 최고인가?”가 아니라 “selection signal의 positivity가 이후 어떤 endpoint와 일치·충돌하는가?”이다.

#### A가 아닌 이유

1. 공통 protocol로 설계된 독립 multi-dataset confirmatory study가 아니다.
2. 독립 production pool/site/lot 검증이 없다.
3. completed translation branch는 3개뿐이며 STOP branch counterfactual이 없다.
4. practical threshold가 외부 요구로 calibration되지 않았다.
5. Atlas와 taxonomy가 post-hoc이고 causal mechanism test가 없다.

#### C로 내려가는 조건

- 제목·초록에서 benchmark, framework validation, cost savings, selector superiority를 주장할 때
- MPDD/VisA/DeepPCB의 NA learner outcome을 downstream failure로 셀 때
- KSDD2를 단순 success 또는 no-effect로 이분화할 때
- 200 fixed-pool perturbations를 독립 산업 pool replication으로 부를 때
- 서로 다른 native effect를 평균내어 보편 translation rate를 만들 때

#### 등급을 A에 가깝게 올리려면

처음부터 공통 protocol로 동결한 independent pool/site 연구, 모든 PASS/FAIL branch의 별도 validation-track downstream truth, 외부 요구에 근거한 practical threshold, 공통 baseline/metric 또는 사전 정의된 relation test가 필요하다. 이는 현재 논문의 최소 보완 범위를 넘어 후속 연구에 가깝다.

## 8. 추가 실험 없이 방어 가능한 버전과 최소 추가 검증 버전

### 8.1 추가 실험 없이 방어 가능한 버전

제목 수준:

> 산업 결함 Active Learning 후보 신호의 selection-stage 결과와 bounded downstream learning utility 간 관계에 대한 multi-branch empirical audit

핵심 연구 질문:

> 평가한 branch에서 selection discovery, composition, fixed/acquisition reproducibility, downstream learner outcome, rare-class safety는 어떤 관계와 불일치를 보였는가?

필수 구성:

1. `dataset × signal × stage`를 공통 행 단위로 사용한다.
2. 각 endpoint에 `FROZEN_PRIMARY`, `POSTHOC_DESCRIPTIVE`, `NOT_AVAILABLE` 상태를 표시한다.
3. MPDD/VisA는 composition STOP, DeepPCB는 prospective boundary, NEU는 reproducibility warning, GC10/KSDD2는 completed translation case로 역할을 분리한다.
4. 통합 그림은 `docs/translation_atlas/translation_atlas_main_figure.png`를 출발점으로 쓰되 post-hoc·native metric·no pooling footnote를 유지한다.
5. 새 learner/selector run 없이 source table 재검산, 문서 시간축 동기화, claim wording audit를 완료한다.

이 버전에서 가능한 결론:

> 평가한 산업 결함 AL branch에서 selection positivity는 하나의 downstream fate로 수렴하지 않았다. Discovery, composition, acquisition reproducibility, learner direction, rare-class safety, practical sufficiency는 분리해 보고해야 한다.

이 버전에서는 cost, independent-pool generalization, Gate predictive accuracy를 결과로 쓰지 않는다.

### 8.2 최소 추가 검증이 필요한 강화 버전

실제 operational/cost relevance를 논문 결과에 넣고 싶다면 repository가 제안한 최소안은 **기존 KSDD2 acquisition manifest를 이용한 blinded annotation-time pilot**이다.

- 기존 acquisition seed 2개를 사전 무작위 선택
- PatchCore와 Random 각 40 images/seed
- annotator 2명 권장
- 총 `2 seeds × 2 strategies × 40 images × 2 annotators = 320 image-tasks`
- primary endpoint: image당 active annotation seconds
- quality control: 기존 GT와의 Dice/IoU
- 새 GPU learner training: 0
- official/final test 접근: 금지

근거 설계: `docs/minimum_additional_experiments.md`, `protocols/gc10_cost_translation_v1/`의 logging/quality protocol 자산.

이 pilot이 완료돼도 허용되는 주장은 “기존 KSDD2 pool의 제한된 blinded pilot에서 equal image budget이 equal annotation time을 의미했는지”까지다. 전체 산업 비용이나 PatchCore의 cost-normalized 우월성은 여전히 주장할 수 없다.

독립 production generalization이나 Gate predictive validity를 주장하려면 이보다 훨씬 큰 새 연구가 필요하다. 최소한 fresh independent pool에서 selector와 Random, 모든 Gate outcome을 별도 validation track으로 관찰해야 한다. 현재 repository의 `3 pools × 2 arms × 2 budgets × 8 acquisition seeds × 2 training seeds = 192 runs` 예시는 후속 연구 규모이며, 현 thesis의 필수 실험으로 간주해서는 안 된다.

## 최종 심사 대응 문장

> 본 논문은 특정 selector의 우월성을 주장하지 않는다. 평가한 산업 결함 AL branch에서 selection-stage positivity가 composition failure, acquisition reproducibility failure, learner-direction failure, rare-class safety conflict, 또는 통계적으로 양수이나 사전 최소효과에 미달하는 결과로 이어질 수 있음을 repository-traceable evidence로 구분한다. 이 결과는 benchmark나 보편 법칙이 아니라, selection signal과 downstream utility를 동일시하지 말아야 한다는 bounded empirical evidence다.


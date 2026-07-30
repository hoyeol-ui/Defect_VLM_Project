# 미니 논문 핵심 표

이 파일은 동결된 결과를 논문 표로 재사용하기 위한 단일 출처다. 숫자는 `research_evidence_ledger.csv`와 각 원본 summary/CSV에서만 가져왔다.

## 표 1. Hypothesis transition

| Original hypothesis | Evidence | Result | Revised interpretation | Thesis role |
|---|---|---|---|---|
| VLM explanation consistency는 grounded defect uncertainty다 | consistency–groundedness Spearman -0.181199; severe-failure AUC 0.373433; oracle presence/evidence/IoU 0; paired model 0/3 PASS | Rejected | 사용한 VLM·prompt·데이터 조건에서 semantic consistency와 visual grounding이 분리됨 | Signal-validity failure mechanism |
| Frozen DINO diversity는 범용 annotation utility proxy다 | GC10 rare +2.720; MPDD anomaly +6.245; VisA anomaly +14.480, 그러나 구성 변화 상이 | Rejected as universal guarantee | DINO는 dominant visual variance를 추적하지만 그 분산이 local defect utility라는 보장은 없음 | Discovery–composition mechanism comparison |
| Discovery/coverage 개선은 detector utility로 번역된다 | GC10 overall +0.017378와 rare AP -0.019877; K40 coverage .955 vs .940이나 mAP -0.001678 | Rejected | discovery, composition safety, learning utility를 별도 endpoint로 검증해야 함 | Selection–learning translation gap |
| seed45 fixed-set gain은 selector 일반화다 | fixed +0.016236, 5/5; 새 acquisition +0.007019, CI [-0.005211, 0.019678], p=.322266 | Not confirmed | training-seed stability와 acquisition-set generalization은 다른 재현성 축 | Reproducibility separation |
| detector error AUROC가 높으면 작은 annotation budget에서 유용하다 | AUROC 0.766432; total-error enrichment 1.115901 < 1.50; FN 1.379693 | Primary gate FAIL; FN exploratory | ranking association과 top-budget operational enrichment를 분리해야 함 | Operational-validity boundary |
| 단계적 gate는 미래 branch 실패를 예측한다 | generic precommitted holdout 0/6; predictive confusion matrix 불가 | Not identifiable | predictive policy가 아니라 retrospective authorization and cost-containment workflow | Thesis reframe and claim control |

## 표 2. Discovery–Composition–Utility Matrix

| Case | Signal | Discovery result | Composition change | Acquisition reproducibility | Downstream utility | Final decision |
|---|---|---|---|---|---|---|
| GC10 | Frozen DINO visual diversity | rare images +2.720; classes +0.825 | proxy groups +1.465; relative safety better than MPDD/VisA | fixed-pool paired seeds only | q20 overall +0.017378; rare AP -0.019877 | Partial discovery; rare-safety FAIL |
| MPDD | Hierarchical/global DINO | anomalies +6.245 | category -0.325; capture-day +1.130; source contribution 67.3% | 200 highly overlapping leave-20 perturbations | Not tested because safety FAIL | Source-confounded discovery only |
| VisA | Frozen DINO visual diversity | anomalies +14.480 | categories -4.110; entropy -0.929016; HHI +0.286025 | discovery gain and safety loss in 200/200 fixed-pool seeds | Not authorized | Category collapse; STOP |
| Seed45 | Visual diversity fixed set | fixed-set mAP50-95 +0.016236 | one fixed selected set; composition audit does not establish mechanism | 5/5 learner seeds, but new acquisition +0.007019; CI crosses 0 | independent acquisition gate FAIL | Training stability, not acquisition generalization |
| K40 | ClusterK40/140 coverage | all-class coverage .955 vs Random .940 | coverage positive, rare composition not sufficient | 200 selection holdout seeds; one bounded learner family | mAP -0.001678; rare AP -0.018290; recall -0.021871 | Selection–learning translation FAIL |
| V2.3 | Detector ensemble uncertainty | majority-error AUROC 0.766432 | ranking association, not a composition guarantee | reused development audit; no external FN pool | total-error enrichment 1.115901 < 1.50; FN 1.379693 exploratory | Primary operational gate FAIL |

## 표 3. Gate and authorization rule

| Stage | Primary question | PASS meaning | FAIL meaning | Authorized next step | Prohibited claim |
|---|---|---|---|---|---|
| G1 Signal validity | 신호가 주장한 현상을 실제로 측정하는가? | 최소한 주장한 semantic/physical endpoint와 frozen relation 확인 | 신호의 의미가 붕괴하거나 confound됨 | G2–G3 selection-only audit | 신호를 epistemic uncertainty 또는 utility로 단정 |
| G2 Target discovery | 같은 budget에서 원하는 표본을 더 찾는가? | target yield/enrichment가 frozen 기준 충족 | Random 대비 discovery 근거 부족 | G3와 결합 판정 | detector 성능 향상 주장 |
| G3 Composition safety | discovery가 class/category/source/rare 구성을 훼손하지 않는가? | discovery와 safety를 동시에 충족 | concentration 또는 rare-safety 손실 | G4 acquisition confirmation; 또는 discovery-only claim | safety 실패를 숨긴 범용 성공 주장 |
| G4 Acquisition reproducibility | 새 선택에서도 효과가 유지되는가? | 새 acquisition seed 또는 독립 pool unit에서 유지 | fixed-set learner stability에만 국한 | 정확히 1회 `BOUNDED_SCREEN` | training stability를 selector 일반화로 표현 |
| G5 Learning utility | learner가 실제로 좋아지는가? | overall·recall·rare endpoint의 frozen gate 충족 | selection positive가 learning으로 번역되지 않음 | G6 operational audit | backbone/seed 확장, locked final 사용 |
| G6 Operational validity | 작은 review budget에서 실제 가치가 있는가? | enrichment와 human/deployment endpoint 충족 | ranking score가 운영 효용으로 번역되지 않음 | locked final 1회 사용 | production effectiveness 주장 |

> **핵심 규칙:** Selection PASS는 성공 예측이 아니라 learning utility를 측정할 수 있는 1회 bounded detector screen 권한이다.

## 표 4. Cost containment

| Branch | Planned next experiment | Prevented model runs | Evidence source | Conservative treatment |
|---|---|---:|---|---|
| GC10 D2R representation | 5 acquisition units × 3 detector training seeds | 15 | `docs/gc10_d2r_representation_branch_closure_20260715.md`; E033 | 계획 문서에 명시된 detector runs만 포함 |
| GC10 K40 downstream | YOLOv8s expansion: 5 acquisition units × 3 train seeds × 2 compared policies | 30 | `docs/framework_cost_avoidance_summary_20260718.csv`; E033 | YOLOv8n bounded screen 이후 실행하지 않은 expansion만 포함 |
| 합계 | 서로 겹치지 않는 두 계획 | **최소 45** | Evidence Freeze v2 ledger | GPU-hours·전력·금액으로 환산하지 않음 |
| Locked final | 모든 gate 이후 1회 사용 가능 | actual use 0 | E034 | counterfactual avoided uses는 `NA` |

## 표 5. 논문 claim boundary

| Supported | Conditional | Exploratory | Rejected | Not identifiable |
|---|---|---|---|---|
| discovery–safety separation | fixed-benchmark acquisition robustness | FN enrichment 1.379693 | VLM consistency as direct epistemic uncertainty | prospective predictive accuracy |
| selection–learning separation | dataset-specific discovery benefit | rare-FN 1.785910 | universal selector superiority | independent production generalization |
| acquisition stability–generalization separation | MPDD source-confounded diversification | local feature misalignment future hypothesis | sparsity law | annotation cost reduction |
| branch-specific gate compliance | GC10 rare discovery |  | global DINO utility guarantee | inspector trust improvement |
| documented lower bound 45 |  |  | flip disagreement as epistemic uncertainty | good-selector sensitivity |
| final-test actual use 0 |  |  |  | stopped-branch counterfactual outcomes |

## 원본 locator

- `runs/evidence_freeze_v2_20260718/research_evidence_ledger.csv`
- `docs/hypothesis_transition_matrix_20260718.csv`
- `docs/acquisition_mechanism_matrix_20260718.csv`
- `docs/framework_temporal_validation_decision_20260718.md`
- `docs/framework_cost_avoidance_summary_20260718.csv`
- `docs/framework_holdout_confusion_matrix_20260718.csv`


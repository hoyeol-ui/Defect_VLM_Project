# DeepPCB branch closure

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

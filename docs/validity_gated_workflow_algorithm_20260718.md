# Algorithm 1. Validity-Gated Authorization for Industrial Active Learning Research

## 목적

후보 신호의 부분적 성공을 곧바로 detector learning utility로 해석하지 않고, 각 번역 단계의 타당성을 확인한 뒤 다음 고비용 단계의 **수행 권한**만 제한적으로 부여한다. 이 알고리즘은 downstream 성공을 예측하는 정책이 아니다.

## 입력과 출력

**입력**

- candidate signal `s`
- unlabeled pool `U`
- labeled set `L`
- acquisition budget `B`
- frozen endpoints and gates `G`
- learner `M`
- locked final test `T_final`
- 비교 기준(최소한 Random), acquisition seed 정책, evidence provenance

**출력**

- `STOP`: 다음 고비용 단계 수행 금지
- `BOUNDED_SCREEN`: 정확히 한 번의 제한된 detector screen 권한
- `ADVANCE`: 다음 gate 또는 조건부 locked-final 단계로 진행
- `CLAIM_SCOPE`: `DISCOVERY_ONLY`, `CONDITIONAL`, `EXPLORATORY`, `NOT_IDENTIFIABLE` 등 허용 주장 범위

## 절차

1. **Protocol freeze**: 가설, 데이터 역할, endpoint, threshold, acquisition seed, learner seed, stopping rule, final-test lock, provenance 규칙을 결과 열람 전에 고정한다.
2. **G1—Signal validity**: VLM consistency, DINO distance, detector uncertainty와 같은 신호가 주장한 의미를 측정하는지 독립 endpoint로 감사한다. 예: grounding, 실제 error ranking, transform/source confound.
3. G1이 `FAIL` 또는 `NOT_IDENTIFIABLE`이면 branch를 중단하고 신호가 측정한다고 주장한 의미를 축소한다. selector 확장, detector 학습, final 사용을 허용하지 않는다.
4. **G2—Target discovery**: 동일 budget에서 Random 대비 rare/anomaly/target yield를 계산하고 uncertainty를 함께 기록한다.
5. **G3—Composition safety**: G2와 동시에 class/category/source/session concentration, entropy, HHI, bbox·instance richness, rare safety를 평가한다.
6. G2는 통과하지만 G3가 실패하면 `DISCOVERY_ONLY`로 분류한다. 발견량 증가는 보고할 수 있으나 학습 효용 또는 범용 selector 우월성은 주장하지 않는다.
7. **G4—Acquisition reproducibility**: 고정 선택 집합의 learner-seed 반복과 새 acquisition seed 또는 독립 pool realization을 구분한다. 전자는 training stability, 후자는 acquisition generalization을 평가한다.
8. G2–G4 중 필수 gate가 실패하거나 식별 불가능하면 `STOP`한다. 동일 fixed pool에서 seed만 바꾼 결과를 독립 production generalization으로 해석하지 않는다.
9. G2–G4를 통과하면 `BOUNDED_SCREEN`을 부여한다. 이는 **성공 예측이 아니라 learning utility를 측정할 수 있는 정확히 한 번의 detector screen 권한**이다.
10. **G5—Learning utility**: 사전 고정 learner와 development validation에서 mAP50-95, recall, rare AP 또는 task-specific AU(L)C를 Random과 비교한다. overall improvement와 rare/safety degradation을 동시에 판정한다.
11. G5가 실패하면 추가 backbone, 추가 seed, 후속 selector 조합, locked final 사용을 중단한다. 이 downstream FAIL은 selection gate의 `false advance`가 아니라 bounded measurement의 허용된 결과다.
12. **G6—Operational validity**: top-budget enrichment, 실제 annotation/deployment 가치, human trust 또는 생산 조건의 relevance를 독립 endpoint로 확인한다.
13. G5와 G6를 모두 통과하고 사전 등록 조건이 충족된 경우에만 locked final을 1회 사용한다. 그렇지 않으면 final은 봉인하고 결과를 단계별 claim boundary와 함께 보고한다.

## 판정 의사코드

```text
freeze(protocol, endpoints, thresholds, costs, provenance, final_lock)

g1 = audit_signal_validity(S)
record(g1)
if g1 != PASS:
    return STOP

g2 = audit_target_discovery(S, B, budget)
g3 = audit_composition_safety(S, B, budget)
record(g2, g3)
if g2 == PASS and g3 != PASS:
    return DISCOVERY_ONLY
if g2 != PASS or g3 != PASS:
    return STOP

g4 = audit_acquisition_reproducibility(S, acquisition_units)
record(g4)
if g4 != PASS:
    return STOP

authorization = BOUNDED_SCREEN  # success prediction이 아님
g5 = run_exactly_one_frozen_learner_screen(S, B)
record(g5, consumed_cost, avoided_cost)
if g5 != PASS:
    keep_final_locked()
    return STOP

g6 = audit_operational_validity(S)
record(g6)
if g6 != PASS:
    keep_final_locked()
    return STOP

authorize_locked_final_once()
return FINAL_AUTHORIZATION
```

## 불변 규칙

- `Discovery gain ≠ composition safety ≠ learning utility`를 항상 분리한다.
- `Selection PASS ≠ detector success`; PASS는 측정 권한이다.
- 고정 선택 집합의 learner-seed 안정성을 acquisition 일반화로 바꾸어 말하지 않는다.
- `NOT_IDENTIFIABLE`은 0이나 실패율로 대체하지 않는다.
- early-stop recall, false-advance rate, correct-stop precision은 현재 시간적 감사에서 `NA`다.
- 문서화된 비용 억제는 D2R 15회와 K40 YOLOv8s 30회, 합계 **최소 45 planned detector model runs**다. 이를 45회를 초과한 추정치로 확대하지 않는다.
- locked final의 실제 사용은 0회다. counterfactual final-test avoidance는 식별 불가능하므로 `NA`다.
- GT/XML은 selection 완료 후 post-hoc mechanism audit에만 사용하며 selector 입력으로 역류시키지 않는다.

## 현재 프로젝트에서의 해석

현재 자료는 이 workflow의 미래 실패 예측 성능을 입증하지 않는다. 15개 branch 중 결과 전 명시적 protocol artifact가 확인된 것은 11개이고, 나머지 4개는 script-only chronology다. 또한 사전 고정된 generic gate policy를 검증할 holdout branch가 0/6이므로 confusion matrix와 predictive operating characteristics는 식별 불가능하다. 따라서 정당한 정체성은 **retrospective validity-gated empirical evaluation and cost-containment workflow**다.

## 증거 추적

- Temporal 판정: `docs/framework_temporal_validation_decision_20260718.md`
- Branch chronology: `docs/framework_branch_timeline_20260718.csv`
- Holdout identifiability: `docs/framework_holdout_confusion_matrix_20260718.csv`
- Cost lower bound: `docs/framework_cost_avoidance_summary_20260718.csv`
- Claim boundary: `docs/thesis_claim_boundary_20260718.md`
- Frozen evidence ledger: `runs/evidence_freeze_v2_20260718/research_evidence_ledger.csv`
- Flowchart: `docs/figures/validity_gated_algorithm_flowchart.png`

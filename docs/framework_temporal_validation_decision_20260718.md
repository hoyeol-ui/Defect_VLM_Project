# Framework temporal validation decision

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

- Development chronology (9): V10c24 scale extension, NEU seed45 fixed-set stability, NEU independent acquisition confirmation, VisA global DINO selection, MPDD global DINO selection, GC10 global DINO selection, GC10 first detector translation, VLM consistency validity, VLM oracle crop groundedness
- Temporal holdout (6): Paired VLM model comparison, GC10 D2R label-aware retrieval, DCAL-XAI flip disagreement, K40 budget140 selection holdout, K40 YOLOv8n downstream, V2.3 detector uncertainty

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

# 지도교수 결정용 미니 논문 패키지

## 한 문장 결론

초기 ‘범용 GT-free selector’ 가설은 지지되지 않았지만, 세 산업 결함 데이터셋과 detector audit에서 반복된 **discovery–composition–learning translation gap**을 단계별로 검증하고 최소 45 planned detector runs의 확장을 중단한 **retrospective validity-gated empirical evaluation and cost-containment workflow**로 학위논문을 재구성할 수 있습니다.

# 기존 연구가 어디서 출발했는가

초기 연구는 VLM explanation consistency, DINO visual diversity, detector uncertainty를 조합하면 Random보다 detector learning에 유용한 이미지를 선택할 수 있다고 가정했습니다. 목표는 높은 selection quality, detector 성능 향상, annotation 비용 감소를 하나의 연결된 기여로 보이는 것이었습니다.

# 무엇이 기각됐는가

- VLM consistency를 direct epistemic uncertainty로 해석하는 가설: Spearman -0.181199, severe-failure AUC 0.373433, oracle presence/evidence/IoU 0, paired model 0/3 PASS.
- global DINO diversity를 범용 learning-utility proxy로 해석하는 가설: discovery는 증가했으나 MPDD source confound와 VisA category collapse가 발생했습니다.
- selection-stage coverage/discovery가 detector utility를 보장한다는 가설: GC10 q20과 K40에서 overall·rare·recall endpoint가 분리되었습니다.
- training-seed stability를 acquisition generalization으로 해석하는 가설: Seed45 fixed set은 +0.016236 및 5/5 승리였으나 새 acquisition은 +0.007019, CI [-0.005211, 0.019678], p=.322266이었습니다.

# 무엇이 실제 결과로 남았는가

| 사례 | Positive | 동반된 한계 또는 harm | 판정 |
|---|---:|---|---|
| GC10 | rare discovery +2.720; q20 overall mAP +0.017378 | rare AP -0.019877 | 평균 이득과 rare safety 충돌 |
| MPDD | anomaly +6.245 | category -0.325; source contribution 67.3% | source-confounded discovery |
| VisA | anomaly +14.480 | category -4.110; HHI +0.286025 | category collapse |
| K40 | coverage .955 vs .940 | mAP -0.001678; rare -0.018290; recall -0.021871 | selection–learning FAIL |
| V2.3 | error AUROC .766432 | total-error enrichment 1.115901 < 1.50 | ranking–operation FAIL |

# 왜 하나의 논문으로 묶이는가

각 branch는 별개의 selector 경진대회가 아니라 하나의 질문을 다른 단계에서 검증합니다.

> 후보 acquisition signal의 discovery 또는 selection positive만으로 composition safety와 실제 detector learning utility를 주장할 수 있는가?

현재 답은 ‘아니다’입니다. Signal validity, Target discovery, Composition safety, Acquisition reproducibility, Learning utility, Operational validity를 분리해야 합니다. GC10·MPDD·VisA의 차이는 약점이 아니라 서로 다른 failure mechanism을 드러내는 비교 자산입니다.

# 새 중심 기여

Primary contribution은 새 score가 아니라 다음 고비용 검증의 수행 권한을 통제하는 방법론입니다.

1. 후보 신호를 여섯 validity endpoint로 분리합니다.
2. discovery gain과 category/source/rare safety를 동시에 기록합니다.
3. fixed-set learner stability와 새 acquisition 재현성을 분리합니다.
4. Selection PASS를 성공 선언이 아니라 **정확히 한 번의 bounded downstream screen 권한**으로 정의합니다.
5. FAIL/NOT IDENTIFIABLE 뒤에는 backbone·seed 확장과 locked final 사용을 금지합니다.

# Predictive framework라고 주장하지 않는 이유

15개 branch 중 outcome 이전 explicit protocol artifact가 확인된 것은 11개이며 4개는 script-only chronology입니다. 후기 6개 중 generic policy가 precommitted된 holdout branch는 0개입니다. STOP branch의 downstream counterfactual도 없습니다. 따라서 predictive confusion matrix, early-stop recall, false-advance rate, correct-stop precision은 모두 `NOT IDENTIFIABLE/NA`입니다.

그러므로 temporal 판정은 `C. RETROSPECTIVE_AUDIT_ONLY`입니다. 이는 Evidence Freeze v2의 `A. THESIS_REFRAME_STRONGLY_SUPPORTED`와 모순되지 않습니다. 전자는 predictive claim 금지, 후자는 제한된 학위논문 재구성의 증거 충분성을 뜻합니다.

# 비용 억제와 final-test 보호

- D2R detector confirmation 15 planned runs를 중단했습니다.
- K40 YOLOv8s expansion 30 planned runs를 중단했습니다.
- 겹치지 않는 문서화된 lower bound는 **최소 45 planned detector model runs**입니다.
- GPU-hours, 전력, 금액, 성공/실패 counterfactual은 추정하지 않습니다.
- locked final test의 **actual use는 0회**입니다.
- 몇 회의 final 사용을 막았는지는 counterfactual이므로 `NA`입니다.

이 효과는 predictor의 정확도가 아니라 branch-level authorization, resource containment, claim-boundary enforcement의 운영적 효과입니다.

# 추가 실험 없이 작성 단계로 이동하려는 이유

현재 논문 질문은 positive selector 우월성이 아니라 동결된 후보 신호의 translation gap과 authorization 기록입니다. 핵심 수치, 실패 mechanism, source locator, temporal 한계, cost lower bound가 이미 고정되었습니다. 같은 fixed benchmarks에서 score를 더 조합하면 사후 method search가 될 위험이 큽니다.

추가 confirmatory experiment가 정당화되는 경우는 target-blind production unit이 문서화된 외부 독립 pool과 사전 등록 generic policy가 확보되는 때입니다. 그 전에는 논문 본체를 작성하고, FN 1.379693과 local feature hypothesis는 exploratory future work로 격리하는 것이 타당합니다.

# 교수님께 요청드리는 결정

다음 중 하나를 선택해 주시기를 요청드립니다.

## A. Validity-gated empirical evaluation을 석사논문 중심 기여로 인정 — 권고

현재 evidence freeze와 claim boundary를 유지한 채 논문 작성에 들어갑니다. 성능 우월성 논문이 아니라 산업 AL 후보 신호의 단계적 타당성, translation gap, cost containment 연구로 심사 범위를 고정합니다.

## B. 제한적인 추가 positive method result를 요구 — 조건부

논문 본체는 A로 고정하되, 독립 pool이 확보될 때 사전 등록된 단 하나의 confirmatory method만 추가합니다. 현재 데이터에서 post-hoc 조합을 계속 찾지는 않습니다.

## C. 별도 supervised detector 연구로 전환

Active Learning 연구와 분리하여 detector/backbone capacity 또는 supervised defect detection을 새 주제로 시작합니다. 이 경우 현재 결과는 타당성 감사와 부록/선행 연구로 보존하되, 새 주제의 데이터·시간·학위 일정 비용을 별도로 승인받아야 합니다.

# 1분 구두 보고 대본

> 교수님, 처음에는 VLM consistency와 DINO diversity, detector uncertainty를 결합하면 Random보다 좋은 GT-free selector를 만들 수 있다고 생각했습니다. 하지만 여러 데이터셋에서 공통적으로 확인된 것은 discovery가 늘어도 category나 rare safety가 무너질 수 있고, selection positive가 detector utility로 자동 연결되지 않는다는 점이었습니다. 예를 들어 GC10은 rare discovery가 2.720장 늘고 overall mAP도 0.017378 올랐지만 rare AP는 0.019877 떨어졌습니다. VisA는 anomaly를 14.480장 더 찾았지만 category coverage가 4.110 줄었습니다. 그래서 성능이 오를 때까지 selector를 더 찾기보다, 신호 타당성부터 discovery, safety, acquisition 재현성, learning utility, operational validity까지 단계별로 검증하는 workflow로 연구 질문을 바꿨습니다. 다만 temporal audit상 미래 실패를 예측하는 policy라고 말할 수는 없습니다. 확인된 가치는 branch별 주장 통제, 최소 45 planned detector runs의 중단, final test 실제 사용 0회입니다. 이 범위를 지키는 validity-gated empirical evaluation을 학위논문 중심 기여로 인정할지, 제한된 positive result를 더 요구할지, 아니면 supervised detector 주제로 분리할지 결정 부탁드립니다.


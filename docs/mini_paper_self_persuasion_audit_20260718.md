# 자기설득 방지 감사: 왜 이것이 하나의 연구인가

이 문서는 연구자가 먼저 반대편 심사자의 입장에서 논문을 읽도록 만든다. 각 문항의 ‘방어 가능한 답변’만 논문과 발표에 사용하고, ‘피해야 할 과장’은 금지한다.

## 1. 이 연구가 단순 실패 목록이 아닌 이유는 무엇인가?

- **냉정한 답변:** 범용 GT-free selector가 Random을 안정적으로 이긴다는 초기 중심 가설은 지지되지 않았다.
- **방어 가능한 답변:** 각 실패를 같은 잣대로 나열한 것이 아니라 Candidate Signal → Discovery → Safety → Reproducibility → Learning → Operational Validity의 번역 단계로 재분류했다. 서로 다른 실험이 동일한 translation failure를 반복 검출했고, 이를 evidence locator와 stopping rule로 연결했다.
- **피해야 할 과장된 답변:** 실패가 많기 때문에 그 자체로 강한 논문이다.

## 2. 여러 데이터셋과 branch를 하나로 묶는 공통 연구 질문은 무엇인가?

- **냉정한 답변:** GC10은 object detection, MPDD와 VisA는 anomaly 중심이므로 동일 benchmark 점수로 합칠 수 없다.
- **방어 가능한 답변:** 공통 질문은 ‘후보 신호의 discovery 또는 selection positive가 composition safety와 detector utility로 번역되는가’다. GC10은 rare taxonomy와 downstream, MPDD는 source confound, VisA는 category collapse를 각각 식별한다.
- **피해야 할 과장된 답변:** 세 데이터셋이 같은 산업 분포를 대표하므로 결과를 평균할 수 있다.

## 3. 새 알고리즘이 없는데도 방법론적 기여라고 부를 수 있는 근거는 무엇인가?

- **냉정한 답변:** 새 acquisition score나 optimization rule은 제안하지 않았다.
- **방어 가능한 답변:** 기여는 모델 알고리즘이 아니라 실험 authorization 알고리즘이다. signal validity, discovery, safety, acquisition reproducibility, learning utility, operational validity를 분리하고, PASS가 성공 선언이 아니라 다음 bounded measurement 권한이 되도록 정의했다.
- **피해야 할 과장된 답변:** gate 자체가 detector 성능을 높이는 새 AL 알고리즘이다.

## 4. validity gate가 사후 체크리스트와 다른 점은 무엇인가?

- **냉정한 답변:** 완전한 prospective generic policy는 아니며, 15개 branch 중 4개는 script-only chronology다.
- **방어 가능한 답변:** 체크리스트는 결과를 설명하는 데 그칠 수 있지만 이 workflow는 branch별 frozen endpoint와 STOP/ADVANCE를 실행 권한에 연결했다. D2R 15회와 K40 YOLOv8s 30회가 실제로 다음 단계로 진행되지 않았고 locked final도 열지 않았다. 동시에 temporal 한계를 공개해 predictive policy 주장을 포기했다.
- **피해야 할 과장된 답변:** 모든 gate가 첫 실험 이전부터 prospectively 사전 등록되었다.

## 5. predictive validation이 실패했는데도 thesis reframe이 유지되는 이유는 무엇인가?

- **냉정한 답변:** generic precommitted holdout은 0/6이므로 early-stop recall과 false-advance rate를 계산할 수 없다.
- **방어 가능한 답변:** Evidence Freeze v2의 `A. THESIS_REFRAME_STRONGLY_SUPPORTED`는 학위논문 재구성의 증거 충분성을 뜻하고, temporal audit의 `C. RETROSPECTIVE_AUDIT_ONLY`는 predictive claim 금지를 뜻한다. 서로 다른 주장 수준의 판정이므로 모순이 아니다.
- **피해야 할 과장된 답변:** prediction은 검증되지 않았지만 실제로는 잘 맞았을 것이다.

## 6. 45개 model run 방지는 과학적 기여인가, 운영적 기여인가?

- **냉정한 답변:** 45회 중단 자체는 detector 성능 향상도, predictive accuracy도 아니다.
- **방어 가능한 답변:** 주로 운영적 기여이며, 과학적 기여를 보조한다. 명시된 next experiment를 증거 부족 후 중단함으로써 research resource와 claim expansion을 통제했다. 실제 GPU-hour·금액은 재지 않았으므로 ‘최소 45 planned model runs stopped’로만 보고한다.
- **피해야 할 과장된 답변:** 45개의 실패 모델을 정확히 예측하여 막았고 그만큼 비용을 절약했다.

## 7. 가장 약한 주장과 가장 강한 주장은 각각 무엇인가?

- **냉정한 답변:** 가장 약한 것은 gate가 미래 성공·실패를 예측하거나 production에서 일반화한다는 주장이다. 현재 식별 불가능하다.
- **방어 가능한 답변:** 가장 강한 것은 discovery–safety, selection–learning, training-stability–acquisition-generalization이 분리된다는 관찰이다. 이는 GC10, MPDD, VisA, Seed45, K40의 원자료로 추적 가능하다.
- **피해야 할 과장된 답변:** 가장 강한 주장은 제안 selector가 Random보다 우수하다는 것이다.

## 8. “Random보다 이긴 방법이 없는데 왜 논문인가?”라고 물으면 어떻게 답할 것인가?

- **냉정한 답변:** 우월한 production selector를 제안한 논문은 아니다.
- **방어 가능한 답변:** 산업 AL에서 Random이 강한 조건과 proxy 최적화가 category·rare safety를 해치는 조건, 그리고 selection positive가 learner utility로 번역되지 않는 조건을 정량화했다. 이는 새 method를 비싼 학습 전에 걸러 내는 연구방법론적 결과다.
- **피해야 할 과장된 답변:** Random보다 이기지 않아도 우리 방법이 실용적으로 더 좋다.

## 9. “gate를 결과를 보고 만든 것 아닌가?”라고 물으면 어떻게 답할 것인가?

- **냉정한 답변:** 전체 여섯 단계 generic policy는 후기 branch 이전에 완전 동결되지 않았다.
- **방어 가능한 답변:** 그래서 predictive screening으로 주장하지 않는다. 다만 11/15 branch에는 outcome 이전 explicit protocol artifact가 있고, 각 branch의 endpoint·stop decision과 실제 중단 기록은 retrospective process evidence로 남는다.
- **피해야 할 과장된 답변:** 파일 timestamp만으로 독립 preregistration이 입증된다.

## 10. “이건 future method를 위한 실험 노트 아닌가?”라고 물으면 어떻게 답할 것인가?

- **냉정한 답변:** future method 가설과 미검증 endpoint가 포함되어 있다.
- **방어 가능한 답변:** 논문 본체는 future method가 아니라 이미 닫힌 evidence ledger를 분석한다. VLM grounding collapse, dataset별 discovery-composition mechanism, selection-learning translation, temporal identifiability와 cost containment가 완결된 분석 단위다. FN과 local feature는 명시적으로 exploratory/future work에 격리한다.
- **피해야 할 과장된 답변:** 이 workflow가 곧바로 다음 positive method의 성공을 보장한다.

## 11. 현재 논문이 무너지려면 어떤 핵심 증거가 틀려야 하는가?

- **냉정한 답변:** 원본 CSV locator가 잘못 연결되거나, discovery와 harm 수치가 같은 비교 단위가 아니거나, 45회 planned run 문서가 사후 생성되었다면 핵심 신뢰가 약해진다.
- **방어 가능한 답변:** 그래서 frozen ledger의 E001–E034, paired-unit 정의, temporal artifact chronology, cost source를 source registry로 추적한다. 반대로 새 selector가 미래에 성공해도 ‘translation은 단계별로 검증해야 한다’는 현재 결론 자체는 무너지지 않는다.
- **피해야 할 과장된 답변:** 어떤 후속 결과가 나와도 이 논문은 반증될 수 없다.

## 12. 추가 실험 없이 논문 작성을 시작해도 되는 이유는 무엇인가?

- **냉정한 답변:** independent production generalization과 positive selector는 여전히 없다.
- **방어 가능한 답변:** 현재 논문의 질문은 positive selector 우월성이 아니라 동결된 candidate signals의 translation gap과 authorization 기록이다. 필요한 결과와 한계가 이미 식별되었고, 추가 post-hoc search는 claim을 강화하기보다 연구 질문을 다시 흔들 가능성이 크다. 외부 독립 pool이 생기면 별도 prospective confirmation으로 다룬다.
- **피해야 할 과장된 답변:** 더 할 실험이 전혀 없고 현재 결과가 production 적용에 충분하다.

## 최종 자기검증 문장

> 이 연구는 후보 신호가 성공할 것이라고 예측하지 않는다. 후보 신호가 실제로 무엇을 개선하고 무엇을 훼손했는지, 다음 고비용 검증을 수행할 근거가 있는지, 어떤 claim을 금지해야 하는지를 단계별로 기록한다.


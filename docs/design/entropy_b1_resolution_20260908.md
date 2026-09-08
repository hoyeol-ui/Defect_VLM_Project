# Entropy policy learner A 사전 고정

## 결론

B1은 RESOLVED다. 기존 Random/DINO의 한 acquisition trajectory를 A/B learner가 공유하는 계산 구조와 정합적이며 구현을 막는 문제가 없다. 사용자가 2026-09-08 이번 요청에서 모든 dataset × acquisition index에 같은 규칙을 지정했다. 이 결정은 Entropy 신규 학습·AP 결과를 보기 전이다. 이전 R0 inference smoke의 점수 건전성 통계는 이미 관측했으므로 “모든 종류의 결과를 보기 전”이라는 표현은 사용하지 않는다.

## 고정 규칙

- Policy learner는 항상 기존 seed manifest의 slot A다. 유리한 checkpoint/seed를 보고 A/B를 교체하지 않는다.
- R0 A의 historical validation-best best.pt로 R1 query를 선택한다.
- 선택된 같은 cumulative manifest를 A/B가 공유한다. 각 learner는 해당 acquisition에 고정된 seed와 동일 pretrained yolov8n.pt에서 cold-start한다.
- Rr A의 validation-selected best.pt만 다음 Rr+1 query를 만든다. B checkpoint, B validation 성능, A/B 평균 성능은 acquisition 입력이 아니다.
- R0~R5의 labeled sets는 누적되지만 학습 가중치는 round 간 이어받지 않는다. best.pt는 다음 query 추론에만 사용한다.
- 2 datasets × 10 acquisition indices = 20 trajectories; 각 trajectory 5 rounds × 2 learners = 10 신규 cells, 총 200 cells다. R0의 40 checkpoint는 신규 학습 cell이 아닌 read-only reference다.

## 코드와 통계 구조 검증

기존 `confirmatory_v1/confirmatory_protocol_v1.yaml`의 `seed_freeze`는 acquisition별 A/B learner seed를 고정한다. `random.trajectory_reused_by_both_training_seeds`, `frozen_dino.training_seed_input: forbidden`, `primary_endpoint.learner_repeat_reduction: arithmetic_mean_of_two_training_seed_AULCs`, `primary_unit: acquisition_seed`가 기존 구조다. `materialize_all_trajectories.py`는 A/B를 구분하지 않는 cumulative manifest를 만들고, `build_full_registry.py`는 그 manifest에 A/B train cells를 연결한다.

새 규칙은 manifest 공유, seed pairing, labeled budget, cold start, validation/test 경계를 보존한다. 기존의 “selector는 learner seed와 무관”이라는 방법별 특성은 detector-output uncertainty에 그대로 적용할 수 없으므로, Entropy 전용 protocol에서 A를 정책 모델로 명시한다. 기존 Random/DINO protocol을 소급 수정하지 않는다.

향후 비교 단위는 데이터셋별 acquisition index 10개다. 각 learner에서 기존과 같은 6점 AULC를 산출하고 A/B를 산술 평균한 후 동일 acquisition index의 Random/DINO와 paired contrast를 계산하는 구조를 보존할 수 있다. AULC는 선형이므로 6개 budget의 A/B 평균 후 적분과 동치다. 이번 작업에서는 실제 AULC/AP를 계산하지 않는다.

Entropy의 A는 이전 round 선택에도 관여한다. 따라서 A/B는 조건부로 교환가능한 독립 평가 반복이 아니며 “순수한 learner seed 변동성만 분리했다”고 주장하지 않는다. 목표량은 **사전 지정 A-policy가 만든 경로에서 A/B 두 learner의 평균 성능**이다. B를 독립 policy trajectory로 늘리거나 A/B를 n=20 독립 표본으로 세지 않는다. 필요 시 나중에 A/B 별 결과를 역할을 구분해 설명할 수 있지만, 이를 보고 policy를 바꾸지 않는다.

20개 trajectory는 실행 dependency가 서로 독립이라는 의미다. 같은 데이터와 test split을 공유하므로 이를 데이터셋을 합친 n=20 iid 표본이라는 주장으로 바꾸지 않는다. 신규 통계 검정의 contrast/multiplicity 세부 계획은 공식 성능 평가 전에 별도 고정해야 한다.

## 기록용 정의문

“모델 예측에 의존하는 Entropy 방식에서는 각 acquisition index의 learner A를 표본 선택 정책 모델로 사전 지정하였다. 각 round에서 A의 validation-selected best.pt가 선택한 이미지 집합을 두 learner seed가 공통으로 사용하며, 두 learner는 동일 pretrained checkpoint에서 각각 cold-start 학습하였다. learner B는 다음 round의 표본 선택에 관여하지 않았다.”

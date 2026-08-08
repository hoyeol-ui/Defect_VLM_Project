# FN / missed-defect extension feasibility decision

## 최종 판정

**DESIGN_PROTOCOL_ONLY**

현재 결과는 새 feature-misalignment 구현, selection screen 또는 학습을 정당화하지 않는다. FN 신호는 별도 research question을 작성할 최소한의 hypothesis-generation 근거이지만, 독립 검증 자원과 구현 hook이 없다. 따라서 프로토콜 요건만 문서화하고 실행은 중단한다. 이 판정은 V2.3 FAIL을 사후 PASS로 바꾸지 않는다.

## 1. FN enrichment 1.379693의 의미

V2.3 combined signal의 FN enrichment는 **1.379693**로 무작위보다 높다. 동일 primary signal이 majority-error AUROC **0.766432**, Spearman **0.293777**, rank stability **0.652379**를 보였으므로 완전한 noise로 보기는 어렵다. 그러나 preregistered primary endpoint는 total FP+FN enrichment였고 **1.115901 < 1.50**로 FAIL했다. FN은 secondary endpoint이므로 독립 가설의 “동기”까지만 제공한다.

## 2. rare-FN 수치와 표본 한계

- combined rare-FN enrichment: **1.785910**
- confidence-only rare-FN enrichment: **2.683022**
- rare GT support: **18 images / 22 boxes**

이 표본에서는 클래스·이미지 몇 개의 순위 변화가 enrichment를 크게 바꾼다. confidence-only 2.683022는 사후 secondary signal이며 independent realization별 rare capture의 안정성·CI가 primary gate로 사전 고정되지 않았다. 안전성 또는 rare-defect recall 개선으로 주장할 수 없다.

## 3. untouched validation resource

현재 확인된 GC10 protocol은 acquisition, 반복 사용된 232-image development, locked final로 구성된다. V2.3은 development를 이미 사용했다. **현재 연구 주장에 쓸 수 있는 untouched non-final bbox validation resource는 확인되지 않았다.** Locked final은 계속 접근 금지다.

## 4. GC10 FN-box endpoint 정의 가능성

기술적으로는 class-aware IoU 0.50 matching과 confidence 0.25를 이미 사용하므로 `FN-box capture@20% review budget`을 정의할 수 있다. 다만 한 GT box가 여러 checkpoint에서 반복되는 구조이므로 이미지/checkpoint 반복을 독립 표본으로 취급하면 안 된다. acquisition realization 또는 독립 pool split이 inferential unit이어야 한다.

## 5. MPDD/VisA 외부 일반화

MPDD/VisA는 pixel-mask anomaly localization/discovery task다. GC10 detector FN-box와 같은 endpoint가 아니므로 기존 결과로 직접 외부 일반화를 주장할 수 없다. 별도의 anomaly score/mask miss definition 없이 YOLO FN-box를 강제하면 task mismatch가 크다.

## 6. 기존 checkpoint event-count audit

가능하다. 15개 Random140 checkpoint, sealed predictions, per-image FP/FN/rare-FN audit가 이미 존재하므로 **추가 학습·추론 없이** event count와 acquisition-realization별 capture 분포를 재집계할 수 있다. 그러나 이는 동일 development 결과의 descriptive feasibility audit일 뿐 독립 confirmation이 아니다. 현재 결과에는 5 acquisition units밖에 없다.

## 7. local feature hook 존재 여부

저장소 검색 결과, 기존 DINO cache 코드는 `last_hidden_state[:, 0]`의 global CLS embedding을 저장한다. detector 중간 feature를 위한 `register_forward_hook`/동등 hook과 DINO patch-token cache는 현재 `scripts/04_dcal_xai`에 존재하지 않는다. 따라서 local misalignment는 단순 분석이 아니라 새 구현 branch다.

## 8. global DINO 실패 원인 해결 여부

local alignment는 small/local defect semantics를 겨냥한다는 점에서 기존 실패 가설과 방향은 맞지만, **실제로 해결한다는 증거는 없다**. D2R label-aware global retrieval도 class8 best top1 0.153846으로 복구에 실패했다. 새 local method는 이름 변경이 아니라 별도 signal-validity 가설과 독립 split을 요구한다.

## 9. 계산비용

기존 event-count 재집계는 CSV 연산만 필요하다. 새 local detector feature를 동일 232장에 계산하면 최소 15 checkpoints × 232 images = **3,480 detector-image forward passes**와 DINO patch feature 232 image passes가 필요하다. selection-only라면 학습은 없지만, feature storage·spatial alignment 구현/검증 비용이 추가된다. 원본에 측정 GPU-hours가 없어 시간은 추정하지 않는다.

## 10. 학위 일정 적합성

event-count/프로토콜 설계는 감당 가능하지만, 현재 untouched resource가 없는 상태에서 local hook 구현까지 들어가면 결과가 다시 development overfitting이 된다. 학위 본문에는 exploratory FN signal과 실행 조건을 future work로 고정하는 편이 안전하다.

## 실행 재개를 위한 필요조건

1. locked final과 분리된 untouched native-bbox pool 확보.
2. primary endpoint와 inferential unit 사전 고정.
3. Random review, confidence deficit, no-detection, frozen V2.3 combined baseline을 동일 budget으로 비교.
4. rare-FN은 primary gate에서 제외.
5. 위 자원이 없으면 구현·feature extraction·학습을 시작하지 않음.

`ONE_GC10_SELECTION_ONLY_SCREEN_JUSTIFIED`가 아니므로 `v3_local_feature_misalignment_protocol.md`와 구현 코드는 생성하지 않았다.

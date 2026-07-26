# KSDD2 G5 냉정·객관적 post-hoc audit

상태: **POSTHOC_NOT_GATE_CHANGING**  
Frozen decision: **TRANSLATION_NOT_OBSERVED_STOP**

## 1. 결론

강한 discovery 이득이 이 동결 learner의 frozen minimum learning endpoint로
충분히 번역되지 않았다. 이는 PatchCore나 모든 selector가 무의미하다는
증명이 아니라, 이 조건에서 자동 전이 가정이 지지되지 않았다는 뜻이다.

## 2. Frozen primary 결과

- PatchCore macro Dice: **0.6084**
- Random macro Dice: **0.5709**
- Paired Dice delta: **+0.0376**
- 95% CI: **[+0.0196, +0.0570]**
- Positive acquisition-seed fraction: **80.0%**
- Pixel AP delta: **+0.0498**
- PatchCore normal FPR: **0.1601**
- Normal FPR delta: **+0.0106**

## 3. 가장 불편하지만 중요한 사실

### 동일 image budget은 동일 annotation cost가 아니다

- Query positive-image delta: 평균 **+88.8장**
- Query defect-pixel burden ratio: 평균 **15.68×**
- Query component burden ratio: 평균 **11.87×**

PatchCore arm은 Random보다 훨씬 많은 positive mask와 결함 pixel을 제공받는다.
이는 AL discovery의 의도된 효과지만, mask 작성 시간이 더 길 수 있으므로
`같은 100장 = 같은 annotation 비용`이라고 해석하면 안 된다. 학습 이득이
있더라도 그것은 동일 **image-count budget**의 이득이지 동일 노동시간의
비용 효율성 증거가 아니다.

### 열 acquisition seed는 독립 생산환경 열 개가 아니다

- PatchCore query pairwise Jaccard mean: **0.813**
- PatchCore query union: **119장**
- 열 seed 모두에서 선택: **51장**

동일 pool에서 candidate set이 크게 겹치므로 10개 seed는 독립 site나 lot
일반화가 아니다. 현재 CI는 acquisition perturbation에 대한 안정성이지
production generalization의 CI가 아니다.

### Development split은 final evidence가 아니다

Development 466장은 G1에서 이미 사용됐고 G5에서도 평가됐다. 학습이나
checkpoint 선택에는 사용하지 않았지만 완전히 새로운 final test는 아니다.
따라서 어떤 양의 결과도 `screen-stage evidence`까지만 허용한다.

### FPR은 부차적인 장식이 아니다

정상 FPR 기준은 Dice 이득을 무효화하기 위한 임의 페널티가 아니다. 산업
검사에서 정상 샘플의 반복 오탐은 review workload를 직접 증가시킨다.
이번 평균 FPR Gate는 통과했지만 seed별 이질성은 post-hoc 한계로 남는다.

## 4. 추가 진단의 위치

- Training-seed pair 결과는 robustness 설명용이며 primary n을 30으로
  늘리지 않는다.
- 결함 크기 tertile 분석은 사후 분석이며 Gate 판정을 변경하지 않는다.
- Discovery delta와 Dice delta의 Spearman은 n=10의 기술 통계일 뿐
  인과관계나 일반화를 주장하지 않는다.
- Threshold sweep, epoch rescue, backbone 교체는 현재 판정을 구조하기
  위한 근거로 사용할 수 없다.

## 5. 결과별 허용되는 다음 행동

- `LEARNING_TRANSLATION_OBSERVED`: 별도 protocol과 지도교수 승인 후에만
  locked official-test confirmation 검토
- `LEARNING_GAIN_WITH_SAFETY_FAIL_STOP`: 확장 중단, sensitivity–review
  burden trade-off 사례로 보고
- `TRANSLATION_NOT_OBSERVED_STOP`: rescue 금지, subthreshold positive
  learning signal과 selection–learning gap 사례로 보고
- `NOT_IDENTIFIABLE_STOP`: infrastructure 원인을 outcome과 분리해 보고

## 6. 금지되는 주장

- PatchCore의 보편적 우위
- learning effect가 전혀 없었다는 단정
- 동일 annotation cost에서의 효율 향상
- 독립 production-pool 일반화
- 실제 annotation 시간 또는 금액 절감
- Official-test 성능
- Workflow의 predictive accuracy

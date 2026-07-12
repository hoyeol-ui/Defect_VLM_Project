# Final Detector-aware Pivot Protocol

작성일: 2026-07-12  
문서 상태: V10b seed42 development gate 이후 업데이트  
Final test 상태: locked / unused  
현재 다음 단계: seed43-46 frozen V10b multiseed one-cycle validation

## 1. 현재 결론 요약

V8 NEU-only 및 V9/V10 계열 실험을 거치며 연구 방향은 다음처럼 정리되었다.

> VLM 설명 일관성(consistency) 또는 image-level visual diversity만으로는 object detection active learning에서 Random baseline을 안정적으로 넘기 어렵다. 현재 detector가 실제로 불확실해하는 샘플을 먼저 반영하고, 그 안에서 DINO visual diversity와 class/instance balance를 결합해야 detector utility를 더 잘 포착할 수 있다.

V10b는 이 pivot의 최신 candidate method이다. Seed42 development gate에서 V10b는 Random보다 aggregate mAP, precision, recall이 모두 높았고, V9b 대비 NEU 6개 클래스 모두에서 AP50-95가 향상되었다. 하지만 seed42는 방법 개발에 사용된 tuning/development seed이므로, 아직 일반화 성능이나 final evidence로 주장하지 않는다.

## 2. 연구 흐름의 변천

### 2.1 중단 또는 보조화된 방향

- VLM explanation consistency-only acquisition
- Consistency를 main acquisition score로 미는 방향
- OWL-ViT pseudo groundedness / pseudo-instance를 main score로 사용하는 방향
- 결과를 보며 seed, weight, epoch을 반복적으로 바꾸는 비고정 탐색

### 2.2 유지되는 요소

- DINOv2 visual diversity는 standalone winning strategy가 아니라 auxiliary component로 유지한다.
- Random은 강한 baseline으로 계속 유지한다.
- Final test는 method/protocol lock 이전에는 사용하지 않는다.
- GT-free acquisition과 post-hoc XML analysis를 명확히 분리한다.

### 2.3 최신 pivot

최신 가설은 다음과 같다.

> 산업 결함 객체 탐지 active learning에서는 “어려운 샘플”보다 “학습 가능한 instance-rich 샘플”이 중요하지만, instance-rich에 과도하게 치우치면 recall과 detector utility가 떨어질 수 있다. 따라서 detector uncertainty와 DINO visual diversity를 강화하고 pseudo-instance count 비중을 낮춘 V10b가 더 안정적인 후보 전략이다.

## 3. 현재 고정 프로토콜

### Dataset split

- Dataset: NEU-DET 전체 1,800장
- Acquisition pool: 900장, class당 150장
- Development evaluation: 300장, class당 50장
- Final test: 300장, class당 50장, locked / unused
- Unused reserve: 300장
- Pool/dev/final overlap: 0

### Budget and training

- Initial labeled set: 60장
- Query size: 30장
- One-cycle labeled budget: 90장
- Detector: YOLOv8n
- Epochs: 100
- Image size: 640
- Batch: 8
- Acquisition seed 42: development/tuning seed
- Training seed 1042 for seed42
- Independent validation seeds: 43, 44, 45, 46
- Training seed rule: `training_seed = 1000 + acquisition_seed`

## 4. Acquisition score

V10b acquisition score는 다음 형태로 고정한다.

```text
S(x) =
  w_u * U(x)
+ w_d * D(x)
+ w_b * B(x)
+ w_i * I(x)
```

각 component의 의미:

- `U(x)`: detector uncertainty
- `D(x)`: DINO visual distance / diversity
- `B(x)`: predicted class deficit
- `I(x)`: pseudo instance count score

각 component는 iterative selection 과정에서 정규화된다. 특히 pseudo-instance score는 상한에서 쉽게 포화될 수 있으므로, 단순히 bbox가 많은 샘플을 더 고르는 것이 detector utility를 보장하지 않는다.

## 5. V9b에서 V10b로의 수정

| Component | V9b | V10b | 변경 방향 |
|---|---:|---:|---|
| Detector uncertainty | 0.20 | 0.25 | 증가 |
| DINO visual distance | 0.25 | 0.35 | 증가 |
| Predicted class deficit | 0.15 | 0.15 | 유지 |
| Pseudo instance count | 0.40 | 0.25 | 감소 |

### 설계 의도

V9b는 실제 XML 기준으로 bbox instance와 entropy가 Random보다 높았지만, mAP와 recall은 Random보다 낮았다. 이는 “많은 객체 수”가 곧 “좋은 detector 학습 샘플”을 의미하지 않음을 보여준다.

V10b는 다음 방향으로 수정되었다.

- 과도한 instance-rich 편향 완화
- initial set과 중복되지 않는 visual coverage 강화
- detector decision boundary가 불확실한 이미지 확보
- precision-recall balance 회복

이 가정들은 현재 supported interpretation이며 causal proof는 아니다.

## 6. Seed42 development gate 결과

### Aggregate metrics

| Strategy | Budget | mAP50 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Round0 | 60 | 0.581204 | 0.306094 | 0.566488 | 0.599504 |
| Random | 90 | 0.641054 | 0.329700 | 0.620629 | 0.610587 |
| V9b | 90 | 0.623766 | 0.317276 | 0.661013 | 0.579154 |
| V10b | 90 | 0.647079 | 0.340866 | 0.641340 | 0.621586 |

### V10b minus Random

- mAP50: +0.006025
- mAP50-95: +0.011166
- Precision: +0.020711
- Recall: +0.010999

### V10b minus V9b

- mAP50: +0.023313
- mAP50-95: +0.023590
- Precision: -0.019673
- Recall: +0.042432

### Per-class AP50-95

V10b minus Random:

| Class | Δ AP50-95 |
|---|---:|
| crazing | +0.0244 |
| inclusion | -0.0596 |
| patches | +0.0247 |
| pitted_surface | +0.0496 |
| rolled-in_scale | -0.0138 |
| scratches | +0.0417 |

V10b minus V9b:

| Class | Δ AP50-95 |
|---|---:|
| crazing | +0.0103 |
| inclusion | +0.0193 |
| patches | +0.0302 |
| pitted_surface | +0.0055 |
| rolled-in_scale | +0.0188 |
| scratches | +0.0574 |

## 7. Selection-only evidence

V10b는 V9b와 실제로 다른 query batch를 만들었다.

- V9b size: 30
- V10b size: 30
- Overlap: 18
- V9b only: 12
- V10b only: 12
- Jaccard: 0.428571

Selection geometry:

| Strategy | Pairwise cosine similarity | Distance to initial |
|---|---:|---:|
| V9b | 0.500067 | 0.229034 |
| V10b | 0.444252 | 0.280558 |
| Random | 0.556216 | 0.162501 |

해석:

- V10b는 V9b보다 batch redundancy가 낮다.
- V10b는 initial set에서 더 멀리 떨어진 샘플을 고른다.
- V10b는 V9b보다 pseudo box 과다 선택을 줄였다.
- 이 결과는 “다른 selector로 동작한다”는 selection-only gate evidence이지, detector 성능 향상의 causal proof는 아니다.

## 8. Annotation efficiency proxy

| Strategy | Total bbox | Additional bbox | mAP50-95 gain | Gain / additional bbox |
|---|---:|---:|---:|---:|
| Random | 191 | 52 | 0.023606 | 0.000454 |
| V9b | 224 | 85 | 0.011182 | 0.000132 |
| V10b | 206 | 67 | 0.034772 | 0.000519 |

주의:

- 이 값은 실제 annotation time이 아니다.
- bbox당 gain은 보조 proxy로만 사용한다.
- 같은 이미지 안의 bbox 수가 실제 작업 비용에 선형으로 반영된다는 보장이 없다.

## 9. Evidence level

### Directly verified facts

- V10b seed42 mAP50-95 = 0.340866
- Random seed42 mAP50-95 = 0.329700
- V10b는 V9b보다 NEU 6개 클래스 모두에서 AP50-95가 높다.
- validation-only recovery가 aggregate 기록과 일치한다.
- Final test는 사용되지 않았다.

### Supported interpretations

- V10b가 V9b의 instance-rich 편향을 완화했다.
- Visual redundancy가 감소했다.
- Recall이 V9b 대비 회복되었다.
- V10b는 candidate method로 고정하고 독립 seed 검증을 수행할 가치가 있다.

### Unverified hypotheses

- Uncertainty 증가가 recall 향상의 직접 원인이다.
- DINO diversity 증가가 성능 향상의 직접 원인이다.
- Class entropy 증가가 성능 향상의 직접 원인이다.
- V10b가 모든 seed에서 Random보다 우수하다.
- V10b가 final test에서도 Random보다 우수하다.

## 10. 다음 실행 단계

현재는 다음 단계를 수행한다.

```text
for acquisition_seed in [43, 44, 45, 46]:
    training_seed = 1000 + acquisition_seed
    build independent split
    sample initial 60
    train shared Round0
    score pool with Round0 detector
    select Random query 30
    select frozen V10b query 30
    train Random budget 90
    train V10b budget 90
    recover per-class dev metrics
```

금지 사항:

- seed42 재사용 금지
- V10b weight 추가 수정 금지
- V9b 재학습 금지
- full curve 실행 금지
- final test 평가 금지
- seed43-46 결과를 본 뒤 weight 수정 금지

## 11. Gate 판정

| 항목 | 결과 |
|---|---|
| V9b 대비 성능 향상 | 통과 |
| Random 대비 mAP50-95 우위 | seed42에서 통과 |
| Random 대비 mAP50 우위 | seed42에서 통과 |
| Precision/recall 동시 개선 | seed42에서 통과 |
| Annotation efficiency proxy | seed42에서 통과 |
| Per-class recovery | 완료 |
| Final test lock | 유지 |
| 독립 multiseed 일반화 | 진행 전 / pending |

최종 문장:

> V10b는 V9b의 instance-rich 편향을 완화하고 detector uncertainty와 visual diversity를 강화함으로써 seed42 development one-cycle에서 Random보다 높은 mAP, precision, recall을 달성했다. 그러나 seed42는 방법 개발에 사용된 development seed이므로, 현재 결론은 candidate method의 development gate 통과이며 일반화 검증은 seed43-46 frozen V10b validation에서 확인해야 한다.

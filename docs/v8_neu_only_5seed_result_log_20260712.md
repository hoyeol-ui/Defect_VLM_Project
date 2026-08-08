# V8 NEU-only 5-seed 결과 로그

작성일: 2026-07-12  
실험 폴더: `runs/active_learning_ablation_v8_neu_only/v8_neu_only_20260712_105644`  
요약 파일: `runs/active_learning_ablation_v8_neu_only/v8_neu_only_20260712_105644/v8_neu_only_summary.md`  
Final test 사용 여부: 사용 안 함

## 1. 실험 목적

V7 mixed setting에서는 acquisition pool과 evaluation split의 분포가 맞지 않았고, 특히 GC10 pool이 `crease` 중심으로 극단적으로 편향되어 있었다. 이 때문에 DINO visual diversity가 Random 대비 안정적으로 우세한지 판단하기 어려웠다.

V8 NEU-only 실험은 이 문제를 줄이기 위해 다음처럼 프로토콜을 단순화했다.

- Acquisition pool: NEU-DET only, 50 images
- Development evaluation: NEU-DET only, 177 images
- Final test: locked / unused
- Acquisition seeds: 42, 43, 44, 45, 46
- Label budget: 15 → 20 → 25 → 30 → 35
- Detector: YOLOv8n
- Epochs: 100
- Strategies:
  - `GTFreeRandom`
  - `GTFreeDatasetBalancedConsistency`
  - `GTFreeDatasetBalancedVisualDiversity`

핵심 질문은 다음과 같다.

> GC10 skew와 mixed-protocol mismatch를 제거했을 때 DINO visual diversity가 Random보다 반복적으로 좋은가?

## 2. 실행 상태

- Successful/shared rows: 75
- Failed rows: 0
- Pool filter: `NEU-DET`
- Development eval filter: `NEU-DET`
- Final test: unused
- Canonical sampling: enabled

## 3. 주요 평균 결과

| Strategy | Final mAP@50 | Final mAP@50-95 | AULC mAP@50 | AULC mAP@50-95 |
|---|---:|---:|---:|---:|
| Random | **0.4755** | **0.2244** | 0.4164 | **0.1905** |
| DINO Visual | 0.4686 | 0.2117 | **0.4211** | 0.1867 |
| Dataset-balanced Consistency | 0.4561 | 0.2128 | 0.4012 | 0.1811 |

## 4. 핵심 판정

5 seeds 평균에서는 Random이 최종 성능 기준 1위였다.

- Final mAP@50: Random 0.4755 > Visual 0.4686
- Final mAP@50-95: Random 0.2244 > Visual 0.2117
- AULC mAP@50-95: Random 0.1905 > Visual 0.1867

하지만 DINO Visual은 AULC mAP@50에서만 Random보다 높았다.

- AULC mAP@50: Visual 0.4211 > Random 0.4164
- Visual vs Random AULC mAP@50: 5승 0패
- 개선 폭: +0.0047, 약 +1.13%

따라서 결론은 다음처럼 정리하는 것이 가장 안전하다.

> NEU-only로 프로토콜을 정리하자 전체 성능은 크게 상승했지만, DINO visual diversity를 Random보다 효과적인 독립 acquisition strategy라고 주장할 근거는 확보되지 않았다. DINO Visual은 consistency-only보다 나았고 mAP@50 AULC에서 작은 개선을 보였지만, 최종 mAP와 localization-sensitive mAP@50-95 지표에서는 Random을 안정적으로 넘지 못했다.

## 5. Paired comparison 요약

Visual vs Random:

| Metric | Mean diff | Wins | Losses | 해석 |
|---|---:|---:|---:|---|
| AULC mAP@50 | +0.0047 | 5 | 0 | Visual 우세 |
| AULC mAP@50-95 | -0.0037 | 1 | 4 | Random 우세 |
| Final mAP@50 | -0.0068 | 2 | 3 | Random 근소 우세 |
| Final mAP@50-95 | -0.0126 | 2 | 3 | Random 우세 |

Visual vs Dataset-balanced Consistency:

| Metric | Mean diff | Wins | Losses | 해석 |
|---|---:|---:|---:|---|
| AULC mAP@50 | +0.0199 | 5 | 0 | Visual 명확 우세 |
| AULC mAP@50-95 | +0.0056 | 5 | 0 | Visual 우세 |
| Final mAP@50 | +0.0126 | 3 | 2 | Visual 근소 우세 |
| Final mAP@50-95 | -0.0011 | 2 | 3 | 거의 동률 |

## 6. 라운드별 평균 learning curve

mAP@50:

| Budget | Random | Visual | Consistency |
|---:|---:|---:|---:|
| 15 | 0.3140 | 0.3140 | 0.3140 |
| 20 | 0.3759 | 0.3909 | 0.3882 |
| 25 | 0.4294 | **0.4364** | 0.3844 |
| 30 | 0.4655 | **0.4657** | 0.4472 |
| 35 | **0.4755** | 0.4686 | 0.4561 |

mAP@50-95:

| Budget | Random | Visual | Consistency |
|---:|---:|---:|---:|
| 15 | 0.1407 | 0.1407 | 0.1407 |
| 20 | 0.1677 | **0.1710** | 0.1706 |
| 25 | 0.1920 | **0.1925** | 0.1669 |
| 30 | **0.2196** | 0.2072 | 0.2101 |
| 35 | **0.2244** | 0.2117 | 0.2128 |

해석:

- Visual은 20~30장 구간의 mAP@50 효율이 좋았다.
- 하지만 mAP@50-95에서는 30~35장 구간에서 Random이 더 강했다.
- Random은 최종 localization-quality 지표에서 여전히 강한 baseline이다.

## 7. Instance/class composition 분석

최종 35장 기준 평균:

| Strategy | Total bbox instances | bbox/image | Multi-object ratio | Class entropy |
|---|---:|---:|---:|---:|
| Random | 80.8 | 2.31 | 0.834 | **2.529** |
| DINO Visual | **82.8** | **2.37** | 0.829 | 2.502 |
| Consistency | 77.8 | 2.22 | **0.840** | 2.547 |

DINO Visual은 최종 선택 세트에서 bbox instance 수와 bbox/image가 Random보다 약간 높았다. 그러나 class entropy는 Random이 조금 더 높았다. 즉 Visual은 더 많은 instance를 확보했지만, Random보다 class coverage가 항상 더 균형적이지는 않았다.

최종 35장 전체 seeds 합산 class instance:

| Class | Random | Visual | Consistency |
|---|---:|---:|---:|
| crazing | 63 | 63 | 53 |
| inclusion | 90 | **102** | 60 |
| patches | **74** | 68 | 66 |
| pitted_surface | 41 | **60** | 53 |
| rolled-in_scale | **73** | 34 | 72 |
| scratches | 63 | **87** | 85 |

중요한 관찰:

- Visual은 `inclusion`, `pitted_surface`, `scratches`를 많이 확보했다.
- Random은 `rolled-in_scale`을 훨씬 많이 확보했다.
- Visual의 mAP@50-95 약세는 특정 클래스, 특히 `rolled-in_scale` 부족과 관련될 가능성이 있다.

## 8. V7 mixed 대비 V8 NEU-only 변화

V8 NEU-only는 V7 mixed보다 모든 전략에서 성능이 크게 상승했다.

| Strategy | Δ Final mAP@50 | Δ Final mAP@50-95 | Δ AULC mAP@50 | Δ AULC mAP@50-95 |
|---|---:|---:|---:|---:|
| Random | +0.1132 | +0.0578 | +0.1163 | +0.0546 |
| DINO Visual | +0.1093 | +0.0528 | +0.1289 | +0.0591 |
| Consistency | +0.1052 | +0.0645 | +0.1230 | +0.0591 |

이 결과는 다음 판단을 뒷받침한다.

> V7 mixed setting의 낮은 성능과 불안정성은 방법론 자체의 실패만이 아니라, GC10 skew와 evaluation distribution mismatch의 영향을 크게 받았다.

## 9. 연구적 결론

이번 결과는 “DINO Visual이 Random을 이겼다”는 결과가 아니다. 또한 “DINO visual diversity가 독립적인 active learning strategy로 유효하다”는 주장도 현재 결과만으로는 과장이다.

가장 정확한 결론은 다음이다.

> Consistency-only는 실패했다. DINO Visual은 consistency보다 나았지만, Random 대비 이점은 mAP@50 AULC의 약 1.13% 개선에 한정되었고, localization-sensitive 지표에서는 열세였다. 따라서 현재 형태의 DINO visual diversity는 독립적인 acquisition strategy로서의 유효성을 입증하지 못했다.

논문에서 방어 가능한 표현:

- Consistency-only acquisition is not sufficient for detector utility.
- DINO visual diversity is better treated as an auxiliary component than as a standalone acquisition strategy.
- Random remains a strong baseline, especially for localization-sensitive metrics (`mAP@50-95`).
- Dataset/protocol alignment is critical; mixed skewed pools can obscure acquisition effects.
- A final method should pivot toward detector-aware acquisition rather than relying on VLM explanation consistency alone.

## 10. 다음 단계 제안

사용자가 “실험을 그만하고 문서화”하기로 했으므로, 당장 새 장시간 실험은 시작하지 않는다.

문서화 우선순위:

1. V7 mixed-skewed 결과를 limitation으로 정리
2. V8 NEU-only 결과를 main development result로 정리
3. Random이 강한 이유를 instance/class composition 관점에서 분석
4. DINO Visual은 standalone method가 아니라 consistency보다 나은 auxiliary diversity component로 격하
5. 마지막 피벗 후보로 YOLO localization/instance uncertainty + DINO diversity + class/instance balance를 사전 기준과 함께 설계

최종 test는 아직 사용하지 않았으므로, method lock 전까지 보류한다.

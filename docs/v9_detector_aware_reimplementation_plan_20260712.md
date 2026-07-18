# V9 Detector-aware Reimplementation Plan

작성일: 2026-07-12  
상태: 설계 + seed42 diagnostic probe 완료  
Final test 상태: locked / unused

## 1. 현재 판단

V8 NEU-only 결과는 `DINO Visual`이 `Consistency`보다 낫다는 점은 보였지만, `Random` 대비 독립적인 우월성은 입증하지 못했다. 따라서 다음 구현은 설명 일관성이나 image-level diversity를 더 미는 방향이 아니라, detector가 실제로 어려워하는 샘플과 class/instance balance를 acquisition에 직접 반영하는 방향이어야 한다.

다만 이번 seed42 probe는 중요한 경고도 보여줬다.

> Detector uncertainty를 단순히 low confidence 기준으로만 쓰면 Random보다 더 나쁜 batch를 고를 수 있다.

즉 V9의 핵심은 “uncertainty 추가”가 아니라, uncertainty, pseudo-instance richness, predicted-class balance, DINO diversity를 어떤 순서와 제약으로 결합할지다.

## 2. 완료한 작은 seed42 diagnostic

실행 스크립트:

```powershell
.\.python311\python.exe .\scripts\02_active_learning\probe_v9_detector_aware_selection.py
```

출력 폴더:

```text
runs/v9_detector_aware_selection_probe/v9_detector_probe_20260712_154007
```

이 probe는 재학습 없이 기존 V8 NEU-only seed42 round0 checkpoint를 사용했다.

사용한 checkpoint:

```text
runs/active_learning_ablation_v8_neu_only/v8_neu_only_20260712_105644/yolo_train_runs/seed42___SHARED_ROUND0___R0_trainseed1042/weights/best.pt
```

평가하지 않은 것:

- final test
- 새 YOLO training
- 새 acquisition full curve

## 3. Probe 결과 요약

### Round1 batch 5장 post-hoc XML 통계

| Strategy | Images | BBox instances | Mean bbox/image | Actual classes | Class entropy | Large bbox ratio |
|---|---:|---:|---:|---:|---:|---:|
| GTFreeRandom | 5 | 13 | 2.6 | 5 | 2.2955 | 0.6923 |
| GTFreeDatasetBalancedConsistency | 5 | 12 | 2.4 | 3 | 1.1887 | 0.3333 |
| GTFreeDatasetBalancedVisualDiversity | 5 | 11 | 2.2 | 2 | 0.6840 | 0.4545 |
| DetectorUncertainty | 5 | 9 | 1.8 | 2 | 0.7642 | 1.0000 |
| DetectorUncertaintyDINO | 5 | 8 | 1.6 | 3 | 1.5000 | 1.0000 |
| DetectorUncertaintyDINOBalanced | 5 | 8 | 1.6 | 3 | 1.4056 | 1.0000 |

### Cumulative 20장 post-hoc XML 통계

| Strategy | Images | BBox instances | Mean bbox/image | Actual classes | Class entropy | Large bbox ratio |
|---|---:|---:|---:|---:|---:|---:|
| GTFreeRandom | 20 | 47 | 2.35 | 6 | 2.3565 | 0.5957 |
| GTFreeDatasetBalancedConsistency | 20 | 46 | 2.30 | 6 | 2.4071 | 0.5000 |
| GTFreeDatasetBalancedVisualDiversity | 20 | 45 | 2.25 | 6 | 2.4087 | 0.5333 |
| DetectorUncertainty | 20 | 43 | 2.15 | 5 | 2.1642 | 0.6512 |
| DetectorUncertaintyDINO | 20 | 42 | 2.10 | 6 | 2.3630 | 0.6429 |
| DetectorUncertaintyDINOBalanced | 20 | 42 | 2.10 | 6 | 2.3120 | 0.6429 |

## 4. 해석

이번 probe만 보면 naive detector-aware 전략은 아직 성공 후보가 아니다.

관찰:

1. `DetectorUncertainty`는 low confidence 샘플을 잘 찾았지만, 실제 XML 기준 class diversity와 instance 수가 Random보다 낮았다.
2. `DetectorUncertaintyDINO`와 `DetectorUncertaintyDINOBalanced`는 class 수를 일부 회복했지만, bbox instance richness는 Random보다 낮았다.
3. detector가 낮은 confidence를 보이는 샘플이 반드시 “라벨링하면 detector 성능을 가장 많이 올리는 샘플”은 아니었다.
4. Random이 강했던 이유는 여전히 class spread와 instance richness를 우연히 잘 확보했기 때문으로 보인다.

따라서 V9는 다음 원칙으로 다시 설계해야 한다.

> uncertainty는 후보 신호 중 하나일 뿐이며, pseudo-instance richness와 predicted-class/batch balance를 더 강하게 넣어야 한다.

## 5. 코드 재구현 방향

기존 `run_al_yolo_ablation_v7_full_curve.py`를 계속 키우지 말고, 다음처럼 분리한다.

```text
scripts/02_active_learning/
  canonical_sampling_v7.py                  # 유지
  detector_prediction_cache_v9.py           # 신규: YOLO prediction cache
  acquisition_strategies_v9.py              # 신규: GT-free acquisition functions
  run_al_yolo_ablation_v9_detector_aware.py # 신규: V9 runner
  launch_v9_neu_only_seed42_smoke.py        # 신규: 최소 seed42 test launcher
```

### 공통 원칙

- NEU-only protocol 유지
- canonical sample identity 유지
- development_eval_v7 only
- final_test_v7 사용 금지
- DINO embedding cache 재사용
- detector prediction은 round별 CSV cache로 저장
- acquisition 함수는 XML/class_hint/actual bbox 통계를 절대 읽지 않음
- XML 통계는 post-hoc diagnosis에서만 사용

## 6. V9 acquisition 설계 수정안

Naive top uncertainty gate는 피한다. 대신 다음 feature를 계산한다.

### GT-free detector features

```text
detector_uncertainty = 1 - max_conf
detector_pseudo_instance_score = log1p(num_pred_boxes)
predicted_class = top confidence predicted class
no_box_flag = whether no detection exists at low threshold
```

### Visual feature

```text
dino_distance = min cosine distance to current labeled set and selected batch
```

### Balance feature

```text
predicted_class_deficit =
    target count per class
    - current labeled actual class count
    - current selected predicted class count
```

여기서 labeled set은 이미 annotation이 끝난 상태이므로 실제 class distribution을 사용할 수 있다. unlabeled 후보에는 detector predicted class만 사용한다.

### 추천 selection logic

기존 probe의 실패를 반영해 다음 제약을 넣는다.

1. no-box sample은 batch당 최대 1장으로 제한
2. predicted class당 batch 내 최대 2장 제한
3. detector_pseudo_instance_score가 너무 낮은 샘플만 연속 선택하지 않음
4. uncertainty 단독 top-k가 아니라, instance/balance/dino와 함께 greedy score로 선택

초기 고정 score:

```text
S(x) =
0.25 * normalized(detector_uncertainty)
+ 0.30 * normalized(detector_pseudo_instance_score)
+ 0.25 * normalized(predicted_class_deficit)
+ 0.20 * normalized(dino_distance)
```

중요: 이 weight는 성능을 보고 튜닝하지 않는다. 바꾸려면 실행 전에 문서에 고정한다.

## 7. 권장 테스트 순서

### Stage 0: selection-only sanity across 5 seeds

YOLO 재학습 없이 기존 V8 round0 checkpoints를 사용한다.

목표:

- V9 candidate가 Random보다 post-hoc XML instance/class 통계에서 심하게 나쁘지 않은지 확인
- 특히 total bbox instances, class entropy, predicted class coverage를 확인

통과 조건:

- 5 seeds 평균에서 Random 대비 bbox instances가 크게 낮지 않음
- class entropy가 Random과 동급
- 특정 class에 batch가 붕괴하지 않음

이 조건을 못 넘으면 YOLO training으로 가지 않는다.

### Stage 1: seed42 round1 only training

V9가 고른 round1 batch 5장을 initial 15장에 추가해 budget 20 모델만 학습한다.

비교:

- 기존 V8 seed42 Random round1 결과
- 기존 V8 seed42 DINO Visual round1 결과
- 신규 V9 seed42 round1 결과

장점:

- 새 YOLO training 1회만 필요
- Random 대비 가능성이 있는지 빠르게 확인 가능

### Stage 2: seed42 full curve

Stage 1이 희망적일 때만 15→20→25→30→35 seed42 full curve를 돌린다.

### Stage 3: 5-seed full curve

seed42 full curve에서 Random 대비 mAP@50-95 또는 AULC mAP@50-95 개선이 보일 때만 실행한다.

## 8. 현재 결론

내 판단은 다음이다.

1. 이 연구 방향 자체는 아직 버릴 필요 없다.
2. 하지만 `explanation consistency`는 메인 acquisition signal에서 내려야 한다.
3. `DINO visual diversity`도 standalone이 아니라 redundancy control 역할로 제한해야 한다.
4. detector-aware pivot은 타당하지만, naive low-confidence uncertainty로는 부족하다.
5. V9는 `pseudo-instance richness + predicted-class balance + DINO diversity + detector uncertainty` 순서로 재설계해야 한다.

가장 중요한 문장:

> V9의 목표는 “VLM signal로 Random을 이기는 것”이 아니라, Random이 우연히 확보하던 instance richness와 class spread를 GT-free detector prediction으로 의도적으로 재현하면서, DINO diversity로 중복을 줄이는 것이다.

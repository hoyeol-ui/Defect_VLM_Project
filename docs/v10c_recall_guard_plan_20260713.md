# V10c recall-guard plan — 2026-07-13

오늘의 목표는 V10b의 “거의 Random 동률” 상태를 깨기 위한 새 후보를 빠르게 검증하는 것이다.

## 출발점

V10b는 seed42에서는 Random보다 좋았지만, seed43~46 독립 one-cycle에서는 평균 mAP50-95 차이가 `+0.000949`에 불과했다.

```text
Random mean mAP50-95: 0.299207
V10b mean mAP50-95:   0.300157
paired mean diff:     +0.000949
precision mean diff:  +0.023781
recall mean diff:     -0.026170
```

따라서 문제는 단순히 “더 어려운 샘플”을 고르는 것이 아니라, V10b가 precision 쪽으로 치우치면서 recall을 잃는 구조를 고치는 것이다.

## V10c 아이디어

V10c는 query 30장을 두 부분으로 나눈다.

```text
core 24장:
  - V10b 계열 instance-rich + DINO diversity + predicted-class balance
  - no-box 제외
  - pseudo box 2개 이상

recall guard 6장:
  - low pseudo-box / no-box / low-confidence 후보를 제한적으로 허용
  - DINO distance와 uncertainty를 같이 본다
  - no-box는 최대 2장으로 제한
```

즉 V10c는 V9처럼 hard/noisy sample로 회귀하지 않고, V10b의 강한 core는 유지하면서 recall 회복용 탐색 슬롯만 따로 둔다.

## 기본 가중치

Core selector:

```text
detector_uncertainty:     0.30
dino_visual_distance:    0.35
predicted_class_deficit: 0.20
pseudo_instance_count:   0.15
```

Recall guard:

```text
detector_uncertainty:     0.25
dino_visual_distance:    0.30
predicted_class_deficit: 0.20
low_coverage:            0.25
```

## 왜 seed47~50인가?

seed43~46 결과를 이미 보고 V10c를 설계했기 때문에, 같은 seed에서 좋은 결과가 나와도 “진짜 독립 검증”이라고 말하기 어렵다.

그래서 기본 runner는 seed47~50을 사용한다.

```text
AL_ACQUISITION_SEEDS = 47,48,49,50
```

만약 seed43~46에서 돌리면 development/tuning evidence로만 해석한다.

## 실행 스크립트

```text
scripts/02_active_learning/run_v10c_recall_guard_onecycle.py
```

Final test는 계속 locked이며, 이 runner는 final test를 평가하지 않는다.

## 성공 기준

V10c가 오늘 보고자료에 “반전”으로 들어가려면 최소한 다음 조건을 만족해야 한다.

1. seed47~50에서 Random 대비 paired mean mAP50-95가 명확히 양수.
2. recall mean diff가 V10b처럼 크게 음수가 아니거나, mAP gain이 recall loss를 상쇄.
3. wins/losses가 우연처럼 보이지 않을 것.
4. per-class 손실이 특정 클래스에 반복적으로 몰리지 않을 것.
5. final test를 사용하지 않았음을 명시할 것.

## 실패해도 얻는 것

V10c가 Random을 확실히 이기지 못해도, 다음 중 하나는 확인할 수 있다.

- recall guard가 실제로 recall penalty를 줄이는가?
- no-box/low-coverage 샘플이 도움이 되는가, 노이즈인가?
- query 30장 안에서 core/guard 비율이 너무 공격적인가?
- V10 계열의 한계가 selection score가 아니라 data budget/query size에 있는가?


# Frozen V8 Visual Cold-Start Confirmation Protocol

작성일: 2026-07-14  
상태: 실행 전 사전 고정  
Final test: locked / unused

## 1. 냉정한 출발점

seed45 fixed-set retraining에서 Visual20은 Random20 대비 mAP50-95 `+0.016236`, 5/5 training-seed 승리를 보였다. 그러나 이 분석은 전체 V8 결과에서 좋은 seed45를 사후 선택한 조건부 안정성 감사다. n=5 paired exact sign-flip 양측 p-value는 `0.0625`이며, 기존 acquisition seeds 42-46의 Visual-Random round1 차이는 평균 `+0.003253`, 3승 2패, exact p=`0.625`다.

따라서 현재 evidence는 다음으로 제한한다.

> seed45에서 선택된 Visual batch의 효용은 training-seed 반복에서도 유지됐지만, Visual acquisition rule의 acquisition-seed 일반화는 입증되지 않았다.

## 2. 고정 가설

NEU-only pool의 cold-start budget 15→20에서 frozen `GTFreeDatasetBalancedVisualDiversity`가 `GTFreeRandom`보다 development mAP50-95와 recall을 반복적으로 개선하는가?

## 3. 고정 protocol

- Acquisition pool: 기존 V8 NEU-only 50장
- Development evaluation: 기존 NEU-only 177장
- Final test: 사용 금지
- Acquisition seeds: 52-61
- Initial labeled size: 15
- Query size: 5
- 평가 budget: 20만
- Strategies:
  - `GTFreeRandom`
  - `GTFreeDatasetBalancedVisualDiversity`
- Detector: `yolov8n.pt`
- Epochs/patience: 100/100
- Image size/batch: 640/8
- Training seed: `1000 + acquisition_seed`
- DINO cache와 selection implementation: V8에서 변경 금지

같은 50장 pool을 재표집하므로 이 실험은 dataset generalization이 아니라 acquisition-seed robustness만 검증한다.

## 4. Primary analysis

각 acquisition seed에서 budget20의 paired difference를 계산한다.

```text
delta_s = Visual_mAP50-95(s) - Random_mAP50-95(s)
```

보고 항목:

- paired mean/median difference
- wins/losses/ties
- exact paired sign-flip p-value
- bootstrap CI는 descriptive 보조값으로만 사용

## 5. 사전 고정 success gate

아래 조건을 모두 만족해야 cold-start Visual을 독립 확인 후보로 유지한다.

1. mean mAP50-95 difference ≥ `+0.010`
2. wins ≥ `7/10`
3. exact paired sign-flip two-sided p ≤ `0.05`
4. mean recall difference ≥ `0`
5. 어떤 단일 seed를 제거해도 leave-one-out mean mAP50-95 difference > `0`

판정:

- 5/5: confirmatory extension 가치 있음
- 3-4/5: 제한적 신호; selector claim 금지, workflow/triage 사례로만 유지
- 0-2/5: frozen Visual 종료

## 6. 금지 사항

- seed52-61 결과를 본 뒤 selector, DINO distance, query size 수정 금지
- YOLO 모델 동시 변경 금지
- round2 이상 확장 금지
- final test 접근 금지
- seed45 결과와 새 seed 결과를 합쳐 confirmatory p-value를 만드는 행위 금지

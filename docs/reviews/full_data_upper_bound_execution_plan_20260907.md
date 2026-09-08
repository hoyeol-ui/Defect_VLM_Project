# 100% Full-data Reference 실행 계획 — 2026-09-07

## 오늘의 결정

**추천안은 Plan A, 총 4 runs로 고정한다.** 현재 목적은 새로운 주효과를 검정하는 것이 아니라 교수님의 “능동학습 없이 100%를 쓰면 얼마인가?”라는 질문에 동일 조건의 기술적 reference로 답하는 것이다. 오늘은 계획만 확정했으며 학습·평가를 실행하지 않았다.

Plan B는 현재 learner-seed 분포 전체와 대응해야 한다고 지도교수가 명시적으로 요구할 때만 선택한다. Plan A 결과를 본 뒤 유리하거나 불리하다는 이유로 Plan B로 전환하지 않는다.

## 목적

Train pool의 100% 라벨 이미지를 사용한 YOLOv8n의 절대 성능을 측정하고, 현재 R5 약 35% 결과가 100% reference와 얼마나 차이 나는지 제시한다. 이는 AL arm 간 우열 검정이나 실제 annotation 비용 절감의 증거가 아니다.

## 데이터와 실행 단위

| Dataset | 100% train 이미지 | Validation | 잠긴 test | 현재 R5 |
|---|---:|---:|---:|---:|
| GC10-DET | 1,605 | 229 | 458 | 561 (34.953%) |
| NEU-DET | 1,259 | 180 | 359 | 441 (35.028%) |

- 100% run에는 acquisition과 acquisition seed가 없다.
- 각 run은 해당 데이터셋 train pool 전체로 독립 cold-start한다.
- validation으로 `best.pt`를 선택하고, 모든 training integrity가 PASS한 뒤 기존과 같은 분리 절차로 잠긴 test를 한 번 평가한다.
- 기존 440-run manifest, result, checkpoint, figure는 수정하거나 재사용해 덮어쓰지 않는다. 별도의 `full_data_reference` 실행 루트와 manifest를 만든다.

## 반드시 동일하게 유지할 조건

### 데이터 경계

- 현재와 같은 split 및 canonical sample ID를 사용한다.
- Train/validation/test 수는 각각 GC10-DET `1605/229/458`, NEU-DET `1259/180/359`로 고정한다.
- Validation은 checkpoint 선택, 잠긴 test는 최종 평가에만 사용한다.
- 잠긴 test의 이미지·라벨·metric은 training, early stopping 조정, seed 선택, hyperparameter 선택에 사용하지 않는다.

### 모델과 학습

| 항목 | 고정값 |
|---|---|
| Learner | YOLOv8n |
| Ultralytics | 8.4.90 |
| 초기 checkpoint SHA-256 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| Lifecycle | run마다 같은 checkpoint에서 독립 cold-start; warm-start/resume 금지 |
| Epochs / patience | 최대 100 / 100 |
| Input / batch / workers | 640 / 8 / 4 |
| Device / cache / AMP | GPU 0 / false / true |
| Optimizer | SGD |
| lr0 / lrf / momentum / weight decay | 0.01 / 0.01 / 0.937 / 0.0005 |
| Scheduler | `cos_lr=false`, warmup 3 epochs |
| Mosaic | 1.0, 마지막 10 epochs에서 종료 |
| 기타 augmentation | hsv 0.015/0.7/0.4, translate 0.1, scale 0.5, fliplr 0.5; 나머지는 current protocol 유지 |
| Checkpoint 선택 | development validation의 Ultralytics fitness 기준 `best.pt` |
| 최종 평가 | conf 0.001, IoU 0.7, max_det 300, plots/json off |

440-run 설정의 정본은 `confirmatory_v1_1/confirmatory_protocol_v1_1.yaml`이며 SHA-256은 `6836f452d22c3de26dfa2680c45d0b5cb8f925b1a9274789c4ef81a1770f0b10`이다. Base data/split 규약은 `confirmatory_v1/confirmatory_protocol_v1.yaml`이며 SHA-256은 `fa6ad82f5f6b282b1fde5529af41cb54eb5471d3f2950f00d32f491b8ed0c663`이다. Full-data manifest는 v1.1 정본의 학습값과 v1 정본의 data boundary를 읽어 별도로 봉인하되 두 파일을 수정하지 않는다.

## Learner seed 정책

### Plan A — 최소 reference, 4 runs

결과를 보지 않고 기존 sealed seed manifest의 acquisition index 0에 이미 고정된 A/B learner seeds를 사용한다.

| Dataset | Seed A | Seed B |
|---|---:|---:|
| GC10-DET | 1356694077 | 850898698 |
| NEU-DET | 838661549 | 2073899285 |

계산은 `2 datasets × 2 learner seeds = 4 runs`이다.

**장점:** 가장 빠르게 동일 조건의 100% 절대 기준을 제공한다. 기존 R5의 A/B seed와 직접 대응되는 사전 고정 seed라 outcome screening이 없다.

**한계:** 데이터셋별 n=2이므로 learner 변동성의 전체 분포나 정밀한 CI를 주장할 수 없다. 기술적 reference와 seed별 절대값만 보고해야 한다.

**가능한 주장:** “이 두 사전 고정 seed와 동일 lifecycle에서 100% train의 성능은 X였고, R5 약 35%와의 절대 gap은 Y였다.”

### Plan B — 현재 learner 분포 대응, 40 runs

`confirmatory_v1/data_freeze/seed_manifest.csv`의 데이터셋별 20개 `training_seed_a/b`를 모두 사용한다. Manifest SHA-256은 `31372e88b78478fc3400ed363030681e48c0468b8bc599af7d86fbc33defd93d`이다.

계산은 `2 datasets × 20 learner seeds = 40 runs`이다. Full-data에는 acquisition seed가 없으므로 같은 full train을 acquisition index별로 다시 정의하지 않고, 기존 20개 learner seed 값만 한 번씩 사용한다.

**장점:** 현재 AL 결과의 20 learner-seed 분포와 직접 대응하며 평균, 분산, paired seed gap을 보고할 수 있다.

**한계:** Plan A의 10배 학습량이다. 교수 질문에 필요한 단일 기술적 기준보다 강한 설계이며 9월 일정에서 본문 수정 시간을 줄일 수 있다.

**가능한 주장:** “현재 사전 고정 learner seed 분포에서 100% reference의 평균·변동성과 R5의 paired gap은 X였다.” 단, annotation 비용 절감으로 번역하지 않는다.

## 정확한 GPU 작업량

| 항목 | Plan A | Plan B |
|---|---:|---:|
| YOLO training runs | 4 | 40 |
| 최대 epoch 합 | 400 | 4,000 |
| GC10 train image-epochs | 321,000 | 3,210,000 |
| NEU train image-epochs | 251,800 | 2,518,000 |
| 전체 train image-epochs | **572,800** | **5,728,000** |
| 잠긴 test image-evaluations | 1,634 | 16,340 |

`image-epochs`는 `train 이미지 수 × 최대 100 epochs × seed 수`이며 validation 계산은 별도 숫자로 더하지 않았다.

원 실행 환경의 R5 완료 기록에서 중앙 training wall time은 GC10-DET 561장 약 13.88분, NEU-DET 441장 약 10.63분이었다. 이미지 수에 선형 비례한다고 단순 추정하면 full-data 한 run은 각각 약 39.7분과 30.3분이고, 순차 단일 GPU 기준 Plan A 약 2.3 GPU-hours, Plan B 약 23.3 GPU-hours이다. 이는 data loading·validation·test·full-data scaling 차이를 제외한 계획용 추정치이므로 첫 run 완료 후 실제 시간을 기록하되 run 수나 protocol은 바꾸지 않는다.

## 결과표와 해석 규칙

각 dataset·seed에 대해 mAP50-95, mAP50, precision, recall을 저장하고 다음을 보고한다.

1. 100% reference의 seed별 절대값과 단순 평균
2. 같은 learner seed에서 R5 Random, R5 DINOv2 표현 기반 선택, 100% reference의 절대값
3. `full-data - R5` 절대 gap과 `R5 / full-data` 기술 비율
4. R5가 우연히 100%보다 높은 경우도 그대로 보고하며 upper bound라는 보장 표현을 피하고 `100% reference`라고 쓴다.
5. nAULC는 R0~R5 곡선 지표이므로 100% 단일점에 계산하거나 직접 같은 지표처럼 비교하지 않는다.
6. 이미지 수 비율을 annotation 시간·금액 절감률로 바꾸지 않는다.

## 실행 전 완료 조건과 STOP gate

1. 지도교수가 Plan A 4 runs 범위를 확인한다.
2. 별도 manifest에 data hashes, checkpoint hash, 네 seed, 모든 학습·평가 인자를 기록한다.
3. Dry-run은 manifest/hash/경로/CLI 파싱까지만 하고 GPU 학습은 승인 뒤 시작한다.
4. 기존 잠긴 test 절차를 수정해야만 실행할 수 있거나, split/checkpoint/config가 current protocol과 달라지면 **STOP-6**로 중단한다.
5. 모든 4 training runs와 integrity audit가 완료되기 전에는 잠긴 test 평가를 열지 않는다.
6. 한 run의 결과가 마음에 들지 않아 seed, epoch, augmentation, checkpoint selection을 바꾸지 않는다.

## 최종 추천

**Plan A, 4 runs를 현재 논문의 100% reference로 실행한다.** 목적은 빠르고 비교 가능한 절대 기준 확보이며, Plan B 수준의 분포 주장은 하지 않는다. 실행 시작은 지도교수 범위 확인과 별도 manifest 봉인 이후로 제한한다.

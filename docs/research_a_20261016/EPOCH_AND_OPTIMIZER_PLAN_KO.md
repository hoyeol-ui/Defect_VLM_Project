# 후속 실험 설계: SGD 100 → 200 epoch, AdamW는 별도

상태: **설계만 완료 / 새 학습 미승인·미실행 / 새 실행기 미구현**. 작성일 2026-10-16 KST.

## 목적과 대상

- GC10-DET: YOLO26n. 기존 100 epoch sealed dev mAP50-95 27.6485%, mAP50 54.4875%, best epoch 86.
- NEU-DET: YOLO11n. 기존 100 epoch sealed dev mAP50-95 36.2192%, mAP50 68.4967%, best epoch 99.
- 기존 DINO_F518 R5, acquisition seed 2026091901, learner seed 2026091900, 동일 train/dev membership을 고정한다. GC10 train/dev 561/343장, NEU 440/270장.
- 주 지표는 전체 sealed dev mAP50-95, 보조 지표는 mAP50·모든 클래스 AP50-95·시간·메모리·AMP/update 상태다.
- 100 epoch 결과는 완료된 동일 환경 run을 재사용한다. 이번 설계 때문에 재학습하지 않는다. K8, final test, 새로운 표본 선택은 제외한다.

## 먼저 해결할 문제: epochs 하나만 바꿔도 실제 스케줄은 여러 개가 바뀐다

현재 설치·봉인된 Ultralytics 8.4.90 코드에서:

1. Linear LR 배율: `max(1 - x / self.epochs, 0) * (1 - lrf) + lrf`.
2. Mosaic 종료: 0-based `epoch == self.epochs - close_mosaic`. 기존 100/10은 표시 epoch 91부터 off, native 200/10은 191부터 off다.
3. YOLO26 E2ELoss의 one-to-many 배율: `max(1 - updates / (hyp.epochs - 1), 0) * (0.8 - 0.1) + 0.1`. 총 epoch 수가 첫 100 epoch의 loss 가중치까지 바꾼다.
4. 기존 `patience=100`은 100 epoch에서는 조기 종료하지 않았지만 200 epoch에서는 조기 종료할 수 있다.

따라서 native `epochs=200`만 적용한 결과를 '첫 100 epoch와 완전히 같은 학습을 100 epoch 추가한 효과'라고 부르면 안 된다. 이는 전체 학습 길이에 맞춘 스케줄 확장 실험이다.

## 주 설계 E1: 기존 첫 100 epoch를 보존하는 SGD 추가 학습 길이 비교

사용자의 'epoch 효과만'을 엄밀히 해석한 **제안**이다. 승인된 기존 실행기를 그대로 돌리는 200 epoch 명령은 아직 없으며, 이 제안의 구현·검토·실행 승인이 필요하다.

| 항목 | 기존 100 epoch 대조 | 제안 200 epoch |
|---|---|---|
| 시작 | 공식 pretrained cold start, 완료됨 | 같은 pretrained에서 cold start |
| 종료 예산 | 100 | 200 |
| Optimizer | SGD | SGD |
| Batch / imgsz / AMP | 8 / 640 / true | 동일 |
| lr0 / lrf / momentum / weight_decay | .01 / .01 / .937 / .0005 | 동일 |
| nbs / warmup | 64 / 3 epoch | 동일 |
| LR 계산 기준 길이 | 100 | **100으로 고정**, 이후 하한 .0001 유지 |
| Mosaic off 시작 | 표시 epoch 91 | **동일하게 91**, 200까지 off 유지 |
| YOLO26 loss decay 기준 길이 | 100 | **100으로 고정**, 이후 o2m=.1/o2o=.9 유지 |
| Early stopping | patience=100, 실제 미발동 | **patience=0 제안**, 정확히 200 완료 |
| Epoch dev | rect=True, batch16, AMP | 동일 |
| 최종 sealed dev | rect=False, batch8, quantize32 | 동일 |

LR의 x와 scheduler.step 호출 순서는 기존 구현 그대로 보존한다. epoch 100에서 기록된 LR이 하한보다 약간 높을 수 있으므로 100번째 값 자체를 임의로 바꾸지 않는다. 뒤의 scheduler step부터 원래 100 기준 함수의 하한으로 고정한다. 공유 `hyp.epochs`를 100으로 덮어쓰면 전체 학습 길이까지 변할 수 있으므로 loss 기준 길이는 별도 필드로 구현해야 한다.

`patience=0`은 optimizer 변경이 아닌 완료 예산 보장 조치다. 원래 100 epoch에서는 발동하지 않은 조기 종료만 제거한다. 이 차이와 horizon 고정 필드를 새 resolved 기록에 명시한다. loss 계산·추가 증강·LR 크기·SGD momentum·weight decay를 탐색하지 않는다.

### 기존 100 epoch 결과 재사용을 위한 일치 검사

- 새 실행기의 첫 100 epoch LR/accumulation/mosaic 상태와 YOLO26 loss 계수가 기존 규칙과 일치해야 한다.
- epoch 100에서 별도 진단 checkpoint를 저장한다. optimizer·scaler·RNG·loader generator·EMA/model tensor와 기존 last checkpoint의 비교 가능한 수치 상태를 비교한다. epoch 예산, 파일 경로, 시간 등 의도적인 메타데이터 차이 때문에 `.pt` 전체 파일 SHA 동일성을 요구하지 않는다.
- epoch 1~100 dev metric/loss 및 tensor의 정확 일치를 우선 기대한다. 불일치를 사후 허용오차로 덮지 않는다. 차이가 나면 epoch 101 전에 멈추고 원인을 검토한다. 기존 대조군을 자동 재학습하지 않는다.
- 구현 시 기존 checkpoint에 실제로 저장된 상태 범위부터 확인한다. criterion 등 저장되지 않은 상태는 기존 수식·로그에서 비교하고, 관측하지 못한 상태까지 동일하다고 주장하지 않는다.
- 기존 100 epoch best.pt는 원본 그대로 사용한다. 새 200 epoch는 전체 1~200에서 epoch dev mAP50-95 최대, exact tie 최신 epoch를 고른다.

### 해석 경계

이 비교는 **기존 스케줄 종료 이후 낮은 LR로 100 epoch를 더 학습하는 효과**다. '200 epoch에 최적화된 native recipe'의 성능을 뜻하지 않는다. 200 epoch는 best 후보를 더 많이 검사하므로 checkpoint 선택 기회 증가도 효과에 포함된다. 최종 dev 개선이 보장되지 않으며 best가 100 이전에 남을 수도 있다. epoch 100/200의 last checkpoint 추가 dev 평가는 이번 주 분석에 포함하지 않는다.

## 실행 순서와 승인 후 구현 범위

1. 원본 sealed 연구 A 실행기·결과는 그대로 두고 별도 E1 작업 폴더/실행기를 준비한다.
2. `training_epochs`, `schedule_horizon_epochs`, `mosaic_close_epoch`, `loss_decay_horizon`을 분리한다. 기존 run.py/worker.py의 100 epoch 제한과 검증도 새 실행기에서만 확장한다.
3. model별 epoch 경계 1/3/90/91/99/100/101/199/200의 schedule trace를 검증하고, fixed-budget stop과 파일 충돌 방지·AMP·체크포인트·dev 규칙을 확인한다. 현재 작업에서는 이 구현·기술 시험도 실행하지 않았다.
4. 승인 후 GC10 × YOLO26n 200 epoch 1회, NEU × YOLO11n 200 epoch 1회를 GPU 하나에서 순차 실행한다. 기존 checkpoint resume 금지.
5. 환경·입력·코드·pretrained SHA, 출력 충돌을 확인하고 실행한다. 기존 native AMP scale adjustment는 기록하되 finite loss/state 실패·OOM·무결성 실패·설정 drift는 즉시 중단한다. batch/LR 자동 변경·재시도 금지.
6. 기존 100 epoch와 sealed dev 전체/클래스별 Δ, best epoch, 시간, VRAM, optimizer updates/AMP skips, SHA를 비교한다. 낮은 점수도 보존한다.

## 예상 시간: 실측 기반 추정, 새 학습은 아직 0회

| 대상 | 완료된 100 epoch 총 시간 | 200 epoch 중심 추정 | 계획 범위 |
|---|---:|---:|---:|
| GC10 × YOLO26n | 33.14분 | 64.29분 | 55~75분 |
| NEU × YOLO11n | 18.78분 | 36.09분 | 30~45분 |
| 추가 2회 합계 | 대조군 재학습 없음 | **100.38분** | **85~120분** |

중심 추정은 `실측 100 epoch 총 시간 + 100 × mosaic 종료 후 epoch 시간 중앙값`이다. 중앙값은 GC10 18.690초, NEU 10.385초다. 단순 2배 추정은 각각 66.28/37.56분이다. 위 시간은 데이터 준비·학습·dev·백업을 포함한 GPU 실행 예측이며 새 실행기 구현·검토 시간은 제외한다. 학습 실패 시 소요된 시간도 보존한다. 근거 수치는 TIME_ESTIMATES.json에 있다.

## E2: AdamW 비교는 별도, 100 epoch 고정

- E1과 독립된 승인 단계로 두고 자동 시작하지 않는다. E1 성능을 보고 AdamW의 epoch를 100/200 중 유리하게 바꾸지 않는다.
- 두 데이터셋의 선택 모델은 그대로, **100 epoch SGD 기존 결과 vs 100 epoch AdamW**를 비교한다. AdamW도 pretrained cold start이며 E1 checkpoint를 이어 쓰지 않는다.
- optimizer 단독 비교의 초기 제안은 lr0=.01, lrf=.01, weight_decay=.0005와 기존 100 epoch schedule/augmentation을 유지하는 것이다. `optimizer=auto` 금지. 이는 AdamW 최적 설정이라는 뜻이 아니다.
- 설치 코드에서 momentum=.937은 AdamW의 beta1=.937로, beta2=.999로 매핑된다. SGD의 momentum warmup과 AdamW beta 동작은 동일하지 않으며 decay도 SGD L2와 AdamW decoupled 방식의 본질적 차이다. 이들을 기록해야 한다.
- AdamW용 lr0=.001 등을 따로 쓰려면 optimizer+LR이 동시에 달라지므로 'optimizer 단독 효과'라고 부르지 않는다. 별도의 LR 대조 설계로 분리하고 실행 전에 고정한다.
- 현재 ModernTrainer의 SGD 전용 guard를 조용히 해제하지 않는다. 승인 후 별도 AdamW 실행기·설정·검증을 준비한다.
- AdamW 2회는 SGD 100 epoch 합계 51.92분을 참고한 **잠정 45~70분**만 제시한다. optimizer별 실측이 없어 불확실하며 지금은 실행하지 않는다.

## 보존과 제한

모델과 조건을 같은 dev를 보고 선택하는 후속 탐색이다. 단일 acquisition/learner seed만으로 일반적 우월성·유의성을 주장하지 않는다. 이미 사용한 test에서 출발한 연구라는 한계도 유지한다. 새로운 final test 평가와 K8 실험은 이 계획에 없다. 기존 논문/9월 결과/연구 A 8개 실행/연구 B 봉인은 수정하지 않는다.

## 코드 근거

- Ultralytics 8.4.90 `engine/trainer.py` `_setup_scheduler`: LR horizon.
- 봉인된 `strict_loop.py`: mosaic 종료, scheduler.step, criterion.update, early-stop 호출.
- Ultralytics `utils/loss.py` `E2ELoss.decay`: 총 epochs 의존 loss 가중치.
- Ultralytics `utils/torch_utils.py` `EarlyStopping`: patience=0이면 infinity.
- Ultralytics `engine/trainer.py` `build_optimizer`: AdamW betas와 parameter group 처리.

이 문서는 코드의 실제 동작을 근거로 작성한 제안이며, 기존 100 epoch 실행 조건을 사후 변경한 기록이 아니다.

## 공통 자료 점검 연결 — 2026-10-16 추가 검토

[클래스별 데이터 지원량 독립 검토](../../research_tools/ff_coverage_preflight/CLASS_SUPPORT_REVIEW_KO.md)를 연구 A·B 공통 사후 점검 D0(P1)로 연결한다. 기존 train/R5 이미지·box/AP 표는 재사용하고, 원본이 있을 때만 다중객체·bbox geometry의 누락 부분을 CPU로 보완한다. 낮은 AP를 표본 부족이나 학습 부족으로 단정하지 않는다. 이 연결은 위 모델·epoch·optimizer·스케줄 계획이나 완료 결과를 변경하거나 추가 학습을 승인하지 않는다.

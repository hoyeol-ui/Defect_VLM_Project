# 연구 A: 8개 YOLO 본 학습 최종 비교

작성 시각: 2026-10-16T01:11:14.059642+09:00

**학습 8/8 성공, 새 실행 7회 및 보존된 기존 GC10 × YOLO26n 1회.** 원본 로그, best/last 및 백업 가중치를 보존했습니다. GPU 학습은 순차 실행했고 학습 재시도·resume·batch/LR/epoch 임의 변경은 없었습니다. K8 학습, 추가 recipe 실험, test 평가도 하지 않았습니다.

최종 집계 프로그램에는 오류가 있었습니다. 모든 학습과 cell 검증이 끝난 뒤 CSV를 작성하면서 기존 GC10 × YOLO26n에 없는 launcher_wall_seconds를 참조해 KeyError가 발생했습니다. queue와 CAMPAIGN_FAILURE.json, 부분 CSV는 그대로 보존했습니다. 이 보고서는 별도 final_review 디렉터리에 기존 결과를 읽어서 작성했으며 누락된 외부 launcher 시간은 null로 두었습니다. 모든 cell에 존재하는 worker total_seconds를 공통 시간 지표로 사용합니다. 실패 기록의 마지막 run_id는 queue 문맥이며 해당 NEU YOLOv8n 학습 실패를 의미하지 않습니다.

## 최종 sealed dev 성능

모든 점수는 %이며 Δ는 같은 데이터셋·동일 환경의 신규 YOLOv8n 대비 %p입니다.

| Dataset | Model | mAP50-95 | mAP50 | Δ mAP50-95 | Δ mAP50 | Best epoch | 총 시간(분) |
|---|---|---:|---:|---:|---:|---:|---:|
| GC10-DET | yolov8n | 26.4381 | 56.3168 | +0.0000 | +0.0000 | 50 | 21.66 |
| GC10-DET | yolo11n | 26.0425 | 53.0232 | -0.3956 | -3.2936 | 90 | 23.98 |
| GC10-DET | yolo12n | 25.4050 | 52.8718 | -1.0331 | -3.4450 | 86 | 30.27 |
| GC10-DET | yolo26n | 27.6485 | 54.4875 | +1.2104 | -1.8293 | 86 | 33.14 |
| NEU-DET | yolov8n | 34.5247 | 66.7292 | +0.0000 | +0.0000 | 99 | 15.03 |
| NEU-DET | yolo11n | 36.2192 | 68.4967 | +1.6946 | +1.7675 | 99 | 18.78 |
| NEU-DET | yolo12n | 34.1993 | 66.4291 | -0.3254 | -0.3001 | 87 | 23.74 |
| NEU-DET | yolo26n | 34.8907 | 63.2978 | +0.3660 | -3.4315 | 92 | 26.86 |

총 시간은 worker 준비·학습·epoch별 dev·최종 dev·입력 확인·백업 포함입니다. 순수 학습만의 시간이 아닙니다. 이번 7개 순차 실행 구간은 2026-10-15 22:27:37부터 2026-10-16 01:08:27까지 약 2시간 40분 50초입니다.

## GC10-DET: 전체 클래스 AP50-95 (%)

| Class | YOLOv8n | YOLO11n | YOLO12n | YOLO26n |
|---|---:|---:|---:|---:|
| punching_hole | 50.4442 | 54.1253 | 53.2322 | 53.4631 |
| welding_line | 42.3524 | 41.1756 | 34.0958 | 48.1319 |
| crescent_gap | 51.4061 | 60.3631 | 58.1916 | 56.7298 |
| water_spot | 35.0944 | 35.1538 | 34.7614 | 34.9878 |
| oil_spot | 15.5549 | 13.9785 | 13.2351 | 14.9894 |
| silk_spot | 13.5046 | 13.3962 | 9.8354 | 8.9395 |
| inclusion | 5.9892 | 3.0092 | 4.0332 | 3.9152 |
| rolled_pit | 11.3226 | 10.1779 | 4.9652 | 20.8154 |
| crease | 14.3580 | 5.2350 | 15.9335 | 8.9724 |
| waist_folding | 24.3547 | 23.8107 | 25.7662 | 25.5406 |

## NEU-DET: 전체 클래스 AP50-95 (%)

| Class | YOLOv8n | YOLO11n | YOLO12n | YOLO26n |
|---|---:|---:|---:|---:|
| crazing | 11.0301 | 13.5937 | 9.6933 | 6.4165 |
| inclusion | 36.3854 | 38.7582 | 38.8748 | 38.0941 |
| patches | 49.0754 | 53.6112 | 53.3035 | 53.6632 |
| pitted_surface | 48.3902 | 51.8670 | 46.1149 | 46.3073 |
| rolled-in_scale | 15.6346 | 15.9664 | 16.8120 | 16.9118 |
| scratches | 46.6323 | 43.5190 | 40.3970 | 47.9509 |

## GPU 메모리와 AMP

| Dataset | Model | 학습 최대 allocated GiB | 학습 최대 reserved GiB | Optimizer updates | AMP overflow skips |
|---|---|---:|---:|---:|---:|
| GC10-DET | yolov8n | 0.959 | 1.072 | 917 | 9 |
| GC10-DET | yolo11n | 1.107 | 1.244 | 917 | 9 |
| GC10-DET | yolo12n | 1.568 | 1.768 | 916 | 10 |
| GC10-DET | yolo26n | 1.225 | 1.330 | 918 | 8 |
| NEU-DET | yolov8n | 0.946 | 1.252 | 708 | 9 |
| NEU-DET | yolo11n | 1.094 | 1.514 | 707 | 10 |
| NEU-DET | yolo12n | 1.554 | 2.131 | 707 | 10 |
| NEU-DET | yolo26n | 1.211 | 1.592 | 710 | 7 |

GPU 메모리는 PyTorch 실행기가 전체 학습에 걸쳐 기록한 최대치입니다. 장치 전체 메모리 사용량은 아닙니다. 외부 반복 GPU 모니터링은 실행하지 않았습니다. 모든 AMP overflow skip은 첫 epoch에만 발생했습니다. 모든 epoch에 optimizer update가 있었고 loss와 학습 state는 유한값이며 NaN recovery는 0회입니다. 일반적인 AMP 스케일 조정과 치명적인 nonfinite loss/state를 구분했습니다.

## 9월 결과와 이번 동일 환경 대조군

| Dataset | 9월 YOLOv8n mAP50-95 | 이번 YOLOv8n mAP50-95 | 차이(%p) |
|---|---:|---:|---:|
| GC10-DET | 24.0281% | 26.4381% | +2.4100 |
| NEU-DET | 32.0508% | 34.5247% | +2.4738 |

9월은 RTX A4000 / PyTorch 2.8.0+cu128, 이번은 RTX 3070 / PyTorch 2.12.1+cu126입니다. 이번 8개 실행은 Python 3.11.9, Ultralytics 8.4.90, torchvision 0.27.1+cu126, CUDA 12.6, driver 560.94까지 동일한 환경임을 확인했습니다. 과거 점수는 참조값이며 신규 동일 환경 대조군을 대체하지 않습니다. 환경에 따른 차이와 단일 실행 변동을 분리 추정하지 않았습니다.

## 후속 학습 조건 개선 우선 후보

**GC10-DET는 YOLO26n, NEU-DET는 YOLO11n을 우선 후보로 권합니다.** 사전에 정한 전체 dev mAP50-95를 기준으로 각 데이터셋에서 가장 높습니다. 하나의 모델이 두 데이터셋에서 모두 우세하다고 결론내리지 않습니다.

- GC10 YOLO26n: 신규 YOLOv8n 대비 mAP50-95 +1.2104%p. 그러나 mAP50은 -1.8293%p이고 학습 총 시간은 21.66분에서 33.14분으로 증가합니다. 10개 중 5개 클래스 상승/5개 하락입니다. rolled_pit +9.4928%p, welding_line +5.7795%p인 반면 silk_spot -4.5651%p, inclusion -2.0740%p, crease -5.3856%p로 취약 클래스 개선은 일관되지 않습니다.
- NEU YOLO11n: mAP50-95 +1.6946%p, mAP50 +1.7675%p. 6개 중 5개 클래스 상승하고 crazing은 +2.5636%p입니다. scratches는 -3.1133%p입니다. 시간은 15.03분에서 18.78분으로 증가합니다.
- YOLO12n: 두 데이터셋 모두 전체 mAP50-95가 신규 v8보다 낮고 시간·메모리 비용도 증가했습니다. 현재 고정 recipe에서 후속 조건 개선의 우선순위는 낮습니다.
- YOLO11n은 GC10에서 -0.3956%p, YOLO26n은 NEU에서 +0.3660%p입니다. 특히 NEU YOLO26n은 mAP50 -3.4315%p, crazing -4.6136%p라 전체 지표를 함께 보아야 합니다.

단일 acquisition seed·learner seed의 dev 관측값입니다. 통계적 유의성이나 재현 가능한 우월성을 입증하지 않습니다. 사전학습 이력·native loss·head·후처리 차이도 동반되어 순수 architecture 효과로 해석할 수 없습니다. Best epoch가 늦다는 사실만으로 epoch 연장의 이득을 단정하지 않으며, 추가 epoch/optimizer 실험은 별도 승인 후 분리 설계해야 합니다. 이미 사용한 test에 기반한 후속 연구라는 한계도 유지합니다.

## 프로토콜과 무결성

동일 DINO_F518 R5, acquisition seed 2026091901, learner seed 2026091900, 공식 pretrained cold start, 100 epoch·SGD·batch 8·640·AMP를 사용했습니다. Epoch dev는 실제 rect=True·batch 16·AMP, 최종 dev는 rect=False·batch 8·quantize=32입니다. epoch dev mAP50-95 최대값을 선택하고 정확한 동률은 최신 epoch입니다. 최종 sealed dev로 best epoch를 다시 선정하지 않았습니다.

- 새 7개 cell의 PASS ledger, 기존 1개 result 봉인, 8개 실행 인덱스의 총 13154개 파일 항목 SHA256/크기를 다시 확인했습니다.
- 100 epoch, 입력 ID와 선택 manifest, 실제 rect/batch, resolved 설정, finite loss/state, AMP/update 집계, best/last 및 backup SHA, 모든 클래스 AP 평균이 통과했습니다.
- 원본 연구 결과, 기존 GC10 YOLO26n 실행, K8 봉인과 이전 manifest·best.pt는 보존했습니다.
- COMPARISON.json의 PASS는 학습 결과 집계 상태이며 queue 프로세스 전체 성공을 뜻하지 않습니다. 전체 queue는 마지막 보고서 생성 오류로 종료됐습니다. 이 보고서는 그 오류를 별도로 공개하고 기존 결과만 검증했습니다.

## 가중치 SHA256

| Run | best.pt | last.pt |
|---|---|---|
| GC10_a2026091901_yolov8n_e100 | `813a4392b916290921745f1ce3b1da4fef7d5f8e8ecc502d278b231356d55dd7` | `f0de97fe8a2fe3f30f1bd5cc7d51c81f3184eb3dfac2e5325b1f5ca43e49ec39` |
| GC10_a2026091901_yolo11n_e100 | `f3ba3f208855ed7151108e5c4b1838aa1f00332fbac978331412884e552312e2` | `98db544d7b82e28a38e7ed6eb2f63445cb125ce4234c311f3bcdf0b5280a347a` |
| GC10_a2026091901_yolo12n_e100 | `383df13dab79bb16bb4e1de96c8e86c929cf9ae076e6d2c9c708f61047a8c72a` | `9e81cbe08b36a62902fac5e69e73a9ad5d58f6368111217f77cd5a308aaf8ccc` |
| GC10_a2026091901_yolo26n_e100 | `4aab15e5ecea24c65bd735a0e1e902ce0bc0259c1aa1afd6454e72c0d5620223` | `6c37f60d7a88da9d733694c438fa4fd9431200e80148721eda047dee14ef7bbb` |
| NEU_a2026091901_yolov8n_e100 | `f6ec661581334ea810dc23f0bda901835ac1c203f2581edc1fc3b1773f75735b` | `1f1a82b175ff111fbd307a3d56d3811ec2843bd753ea2f644af3a46a5bb5a207` |
| NEU_a2026091901_yolo11n_e100 | `61bf207240e68fbfaf22f463bc448005672a65b6485779a4f9a0a7d11122fc35` | `c35012f99995d0708833de19324aa8529562ff0285378d2d5ae3fc4faafd6ad3` |
| NEU_a2026091901_yolo12n_e100 | `f1507db94ff0135d6cde7d1d06e86a0fa470d8008e25caf5a7baaccb493c4ac6` | `4a972f0965373ca7f0c4c268eae4cf79111862d71eb3168a0713307fa46f5ab2` |
| NEU_a2026091901_yolo26n_e100 | `4d38f679b73118c79494a2a72868d02fd6c5851470d528bf0cb3f6bc39bac74d` | `cd0ded64b72707b0ebfa4fe9d251eb55de5eff247d29414a6d11cf1bdafe854b` |

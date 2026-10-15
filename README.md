# 연구 A: 최신 YOLO 비교 결과와 후속 실험 설계

이 브랜치는 완료된 연구 A 결과와 다음 실험의 **설계만** 정리합니다. 신규 200 epoch/AdamW 학습은 시작하지 않았습니다. 연구 B K8와 final test 평가는 제외합니다.

## 먼저 읽을 자료

- [8개 실행 전체 결과·클래스 AP·SHA](docs/research_a_20261016/results/FULL_COMPARISON_REPORT.md)
- [SGD 100↔200 epoch 및 별도 AdamW 계획](docs/research_a_20261016/EPOCH_AND_OPTIMIZER_PLAN_KO.md)
- [Jupyter/Colab 분석용 노트북](docs/research_a_20261016/experiment_design.ipynb)
- [NotebookLM 인계 문서](docs/research_a_20261016/NOTEBOOKLM_HANDOFF.md)
- [계획 매트릭스](docs/research_a_20261016/PLANNED_EPOCH_MATRIX.csv) · [모델 CSV](docs/research_a_20261016/results/MODEL_COMPARISON.csv) · [클래스 AP CSV](docs/research_a_20261016/results/CLASS_AP_COMPARISON.csv)

## 완료 결과: 신규 동일 환경 비교

RTX 3070 8GB, Ultralytics 8.4.90, PyTorch 2.12.1+cu126. 기존 DINO_F518 R5, acquisition seed 2026091901, learner seed 2026091900. 각 공식 pretrained에서 100 epoch·SGD·batch8·640·AMP cold start. 아래 Δ는 이번 동일 환경 YOLOv8n 대비입니다.

| Dataset | Model | dev mAP50-95 (%) | dev mAP50 (%) | Δ mAP50-95 (%p) | Best epoch | 총 시간(분) |
|---|---|---:|---:|---:|---:|---:|
| GC10-DET | yolov8n | 26.4381 | 56.3168 | +0.0000 | 50 | 21.66 |
| GC10-DET | yolo11n | 26.0425 | 53.0232 | -0.3956 | 90 | 23.98 |
| GC10-DET | yolo12n | 25.4050 | 52.8718 | -1.0331 | 86 | 30.27 |
| GC10-DET | yolo26n | 27.6485 | 54.4875 | +1.2104 | 86 | 33.14 |
| NEU-DET | yolov8n | 34.5247 | 66.7292 | +0.0000 | 99 | 15.03 |
| NEU-DET | yolo11n | 36.2192 | 68.4967 | +1.6946 | 99 | 18.78 |
| NEU-DET | yolo12n | 34.1993 | 66.4291 | -0.3254 | 87 | 23.74 |
| NEU-DET | yolo26n | 34.8907 | 63.2978 | +0.3660 | 92 | 26.86 |

**우선 후보: GC10은 YOLO26n, NEU는 YOLO11n.** GC10 YOLO26n은 전체 mAP50-95가 높지만 mAP50 및 silk_spot/inclusion/crease는 신규 v8보다 낮습니다. NEU YOLO11n은 6개 중 5개 클래스가 상승하고 scratches는 하락했습니다. 단일 seed의 관측값이며 통계적 유의성이나 순수 architecture 효과를 주장하지 않습니다.

## 프로토콜 정정과 결과 상태

Epoch별 dev는 **실제 rect=True·batch16·AMP**, 최종 sealed dev는 **rect=False·batch8·quantize32**입니다. 학습 args의 rect=False를 실제 epoch dev 로더 값으로 해석하면 안 됩니다. 9월 trainer와 현재 Ultralytics Python source digest가 일치하고, 이번 런타임 로더 기록에서도 확인했습니다. 이전 계획서의 epoch dev rect=False 설명은 문서 오류였습니다.

8개 학습은 모두 성공했습니다. 마지막 CSV 집계에서 기존 실행의 launcher_wall_seconds 누락으로 오류가 났지만 학습/평가/체크포인트는 성공했고, 실패 로그를 보존한 채 기존 결과만 읽어 final_review 보고서를 작성했습니다. 실패를 숨기거나 학습을 재시도하지 않았습니다. 13,154개 run 파일 항목의 SHA/크기 검증이 통과했습니다.

9월 YOLOv8n의 GC10 24.0281%, NEU 32.0508%는 A4000/PyTorch2.8/CUDA12.8 환경의 과거 참조값입니다. 이번 동일 환경 대조군 26.4381%/34.5247%와 구분합니다.

## 다음 단계: 설계만, 학습 미실행

- 기존 SGD 100 epoch 결과를 재사용하고, 두 모델에 각각 200 epoch cold start 1회를 계획합니다.
- 단순 epochs=200은 LR·mosaic·YOLO26 loss schedule까지 바꿉니다. 주 제안은 첫 100 epoch 스케줄을 유지하고 이후 100 epoch를 추가해 학습 길이 효과를 분리하는 방식입니다. 새 실행기와 prefix 일치 검증이 필요하며 아직 구현·실행하지 않았습니다.
- 예상 GPU 실행시간: GC10 약 64분(55~75분), NEU 약 36분(30~45분), 합계 약 100분(85~120분).
- AdamW는 별도 100 epoch 비교 단계로 분리하며 자동 시작하지 않습니다.

## 파일과 원본 보존

이 브랜치에는 결과 표·설계·설정 제안·가중치 SHA만 포함합니다. 이미지·라벨·.pt 가중치·가상환경·인증정보는 포함하지 않습니다. 기존 연구 원본, 봉인된 실행 코드, 선택 manifest와 결과는 로컬에서 그대로 보존했습니다. [출처와 SHA](docs/research_a_20261016/PROVENANCE.json)를 참고하세요.

기존 저장소 README는 [보존본](docs/research_a_20261016/archive/README_before_research_A.md)에 그대로 남겼습니다. 이 브랜치는 2026-10 후속 연구의 진입점이며 기존 main이나 논문 결론을 변경하지 않습니다.

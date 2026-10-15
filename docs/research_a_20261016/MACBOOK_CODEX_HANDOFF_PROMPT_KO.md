# 맥북 Codex 인계 프롬프트 — 연구 A 후속 실험 설계

아래 구분선 이후의 내용을 맥북 Codex의 새 대화에 그대로 붙여넣으세요. 여기서 '노트북'은 **맥북 기기**를 뜻합니다. Jupyter나 NotebookLM 작업을 요청하는 것이 아닙니다.

---

지금부터 맥북 Codex에서 기존 산업 결함 탐지 연구의 후속 **실험 설계만** 이어서 진행해 주세요. 이 대화의 이전 이력을 안다고 가정하지 말고 아래 GitHub 브랜치의 자료를 먼저 읽어 주세요.

## 1. 저장소와 작업 범위

- 저장소: https://github.com/hoyeol-ui/Defect_VLM_Project
- 작업 브랜치: `research/modern-yolo-results-epoch-plan-20261016`
- 브랜치 URL: https://github.com/hoyeol-ui/Defect_VLM_Project/tree/research/modern-yolo-results-epoch-plan-20261016
- 결과와 최초 후속 계획이 올라간 기준 커밋: `3a8e7d736b9f3a5d4b691e7c1558beb4503f64d2`. 이후 이 인계 문서가 추가됐으므로 **브랜치 최신 HEAD를 사용**하고 이 커밋으로 reset하지 마세요.
- 현재 체크아웃과 미커밋 변경을 먼저 확인하세요. 다른 작업이 있으면 덮어쓰거나 stash/reset하지 말고 별도 안전한 체크아웃으로 읽으세요. 저장소가 없으면 위 브랜치를 별도 폴더에 clone해도 됩니다.
- `main`, 기존 논문 결과, 완료된 실험, 가중치와 봉인된 설정은 수정하지 마세요. 이 브랜치의 설계 문서만 후속 검토 대상으로 삼으세요.
- 이번 요청은 자료 읽기·설계 검토·새 검토 문서 작성까지입니다. **학습·평가·GPU 기술 시험·학습 queue 시작은 승인하지 않습니다.**

## 2. 먼저 읽을 파일과 순서

경로는 모두 저장소 루트 기준입니다.

1. `README.md` — 연구 A의 현재 요약과 결과.
2. `docs/research_a_20261016/results/FULL_COMPARISON_REPORT.md` — 8개 실행, 전체 클래스 AP, 메모리·AMP, 무결성 및 한계.
3. `docs/research_a_20261016/results/MODEL_COMPARISON.csv`
4. `docs/research_a_20261016/results/CLASS_AP_COMPARISON.csv`
5. `docs/research_a_20261016/results/HISTORICAL_VS_NEW_CONTROL.csv` — 9월 결과와 신규 동일 환경 대조군 구분.
6. `docs/research_a_20261016/EPOCH_AND_OPTIMIZER_PLAN_KO.md` — 아직 승인·구현되지 않은 후속 설계 제안.
7. `docs/research_a_20261016/PLANNED_EPOCH_MATRIX.csv`
8. `docs/research_a_20261016/configs/common_args_100epoch.json` — 실제 완료된 학습의 공통 인자.
9. `docs/research_a_20261016/configs/epoch_extension_proposal.json`
10. `docs/research_a_20261016/configs/adamw_separate_proposal.json`
11. `docs/research_a_20261016/TIME_ESTIMATES.json`
12. `docs/research_a_20261016/PROVENANCE.json`, `results/FINAL_VERIFICATION.json`, `VALIDATION.md` — 출처와 검증 범위.

`experiment_design.ipynb`와 `NOTEBOOKLM_HANDOFF.md`는 '노트북'을 잘못 이해해 만들어진 보조자료입니다. 이번 맥북 인계의 필수 자료가 아니며, 새 노트북 제작이나 NotebookLM 작업으로 전환하지 마세요. `archive/README_before_research_A.md`는 과거 저장소 설명의 보존본이며 현재 연구 A의 우선 계획이 아닙니다.

## 3. 연구의 현재 상태

- 연구 A: 기존 R5 선택 집합을 고정하고 탐지 모델·학습 조건을 개선하는 후속 연구.
- 연구 B: DINOv2 표본 선택 개선. K=8 군집 균형 CPU 진단은 완료·봉인됐고 YOLO 학습은 미뤄둔 상태. 이번 작업에서 다루지 않습니다.
- 연구 A의 2개 데이터셋 × 4개 모델 **총 8개 100 epoch 학습이 완료**됐습니다. 완료 실행을 다시 돌리지 마세요.
- 첫 GC10 × YOLO26n 1회 후 나머지 7개를 GPU 하나에서 순차 실행했습니다. 각 학습과 체크포인트 검증은 모두 성공했습니다.
- 마지막 비교 CSV 작성 중 기존 첫 실행에 없는 `launcher_wall_seconds`를 읽어 집계 오류가 발생했습니다. 학습 오류가 아니며, 오류 로그·부분 CSV를 보존하고 기존 결과만 읽어 `final_review` 보고서를 작성했습니다. 이를 숨기거나 학습 실패/재시도로 잘못 해석하지 마세요.
- 200 epoch 및 AdamW 학습은 **아직 0회**이며 해당 실행기도 미구현입니다.

## 4. 완료된 동일 환경 결과

모든 값은 **최종 sealed dev** 점수입니다. Δ는 같은 데이터셋의 **신규 동일 환경 YOLOv8n** 대비 mAP50-95 차이(%p)입니다.

| 데이터셋 | 모델 | mAP50-95 (%) | mAP50 (%) | Δ (%p) | Best epoch | 총 시간(분) |
|---|---|---:|---:|---:|---:|---:|
| GC10-DET | YOLOv8n | 26.4381 | 56.3168 | 0 | 50 | 21.66 |
| GC10-DET | YOLO11n | 26.0425 | 53.0232 | −0.3956 | 90 | 23.98 |
| GC10-DET | YOLO12n | 25.4050 | 52.8718 | −1.0331 | 86 | 30.27 |
| GC10-DET | YOLO26n | 27.6485 | 54.4875 | +1.2104 | 86 | 33.14 |
| NEU-DET | YOLOv8n | 34.5247 | 66.7292 | 0 | 99 | 15.03 |
| NEU-DET | YOLO11n | 36.2192 | 68.4967 | +1.6946 | 99 | 18.78 |
| NEU-DET | YOLO12n | 34.1993 | 66.4291 | −0.3254 | 87 | 23.74 |
| NEU-DET | YOLO26n | 34.8907 | 63.2978 | +0.3660 | 92 | 26.86 |

사용자가 선택한 후속 모델은 **GC10-DET → YOLO26n, NEU-DET → YOLO11n**입니다.

- GC10의 YOLO26n은 전체 mAP50-95가 가장 높지만, mAP50과 silk_spot/inclusion/crease는 신규 YOLOv8n보다 낮습니다.
- NEU의 YOLO11n은 6개 중 5개 클래스가 개선됐고 scratches는 하락했습니다.
- 9월 YOLOv8n mAP50-95는 GC10 24.0281%, NEU 32.0508%입니다. RTX A4000/PyTorch 2.8.0/CUDA 12.8의 과거 참조값이므로 이번 RTX 3070 환경의 대조값과 혼동하지 마세요.

## 5. 유지할 데이터·학습·평가 프로토콜

- 기존 **DINO_F518 R5** 선택 집합. K8 선택 집합이 아닙니다.
- Acquisition seed `2026091901`, learner seed `2026091900`.
- GC10 train/dev `561/343`, NEU train/dev `440/270`장, final split 고정.
- Windows RTX 3070 8GB, Python 3.11.9, PyTorch 2.12.1+cu126, torchvision 0.27.1+cu126, Ultralytics 8.4.90, CUDA 12.6, driver 560.94.
- 공식 pretrained에서 cold start. 기존 checkpoint에서 resume하지 않습니다.
- 기존 SGD: batch 8, imgsz 640, AMP, lr0=.01, lrf=.01, momentum=.937, weight_decay=.0005, nbs=64, warmup 3 epoch, linear scheduler, close_mosaic=10.
- **Epoch별 dev: 실제 rect=True, batch16, FP32 EMA + AMP autocast.**
- **최종 sealed dev: rect=False, batch8, quantize=32.**
- imgsz640, conf=.001, IoU=.7, max_det300. Epoch dev mAP50-95 최대, 정확한 동률은 최신 epoch를 선택합니다. 최종 sealed dev로 best epoch를 다시 고르지 않습니다.
- 과거 계획서의 epoch dev `rect=False` 설명은 문서 오류였습니다. 실제 9월과 이번 실행의 로더 동작은 같습니다. 학습 args의 rect=False를 실제 dev 로더 값으로 오해하지 마세요.
- 동일 seed 한 번의 관측값입니다. 통계적 유의성·일반적 우월성·순수 architecture 인과 효과를 주장하지 마세요. Native loss/head/사전학습/후처리 차이도 있습니다.

## 6. 다음에 결정할 사항 — 기존 제안을 확정 사항으로 취급하지 말 것

사용자 요청은 먼저 **SGD를 유지한 100 대 200 epoch 비교로 epoch 효과를 확인**하고, **AdamW 비교를 별도로 분리**하는 것입니다. 모델 선택은 위 두 모델로 정해졌지만 아래 구현 세부안은 아직 제안 단계입니다.

### A. 'Epoch 효과'의 정확한 정의

현재 코드에서 단순히 `epochs=200`으로 바꾸면 LR 감소 길이, mosaic 종료 시점, YOLO26 E2ELoss의 one-to-many/one-to-one 배율 변화까지 달라집니다. 따라서 아래 두 질문을 구분하고 어떤 것을 주 실험으로 삼을지 판단할 자료를 제시하세요.

1. **기존 경로 보존 후 추가 학습:** LR horizon=100, mosaic off=표시 epoch91, YOLO26 loss decay horizon=100을 유지하고 이후 낮은 LR·mosaic off·종단 loss 배율로 100 epoch를 추가합니다. 현재 계획서의 주 제안입니다. 이는 '기존 스케줄 종료 후 추가 학습'이지 native 200 epoch 최적 recipe가 아닙니다.
2. **Native 200 epoch 스케줄 확장:** 총 길이에 맞춰 LR·mosaic·loss 스케줄도 늘립니다. 실제 epoch 인자 변화 효과를 보지만, 첫 100 epoch까지 같은 경로라고 주장할 수 없습니다. 현재 주 제안과 다른 실험으로 명시해야 합니다.

둘을 자동으로 모두 실행하거나 결과를 본 뒤 유리한 정의로 바꾸지 마세요.

### B. 정확히 200 epoch 완료와 기존 100 epoch 결과 재사용

- 기존 `patience=100`은 200 epoch에서 조기 종료를 유발할 수 있습니다. `patience=0`은 현재 제안이며, 완료 예산 보장 조치로 명시해야 합니다.
- 기존 100 epoch 대조군은 재사용하는 방향입니다. 같은 pretrained에서 cold start하는 200 epoch 실행의 첫 100 epoch 일치 검사가 얼마나 가능한지 설계하세요.
- 전체 `.pt` SHA는 경로/시간/총 epoch 메타데이터 때문에 달라질 수 있습니다. 비교 가능한 EMA/model tensor, optimizer/scaler/RNG/loader 상태와 LR·mosaic·loss 배율을 구분해 검사하는 기준이 필요합니다.
- 불일치하면 epoch101 전에 중단·보고하는 안을 검토하세요. 체크포인트에 없는 상태를 관측했다고 가정하지 말고, 재현 실패를 임의 허용오차·자동 재학습으로 덮지 마세요.
- Best-of-200은 best-of-100보다 dev를 보고 선택할 기회가 많습니다. 추가 학습 효과와 checkpoint 선택 기회 증가를 해석에 포함하세요.

### C. AdamW 비교 분리

- E1과 별도 승인 단계이며 **100 epoch 고정 SGD 대 AdamW**가 현재 제안입니다. E1 결과를 보고 AdamW epoch를 임의로 바꾸지 마세요.
- 현재 문서의 AdamW lr0=.01은 동일 수치 LR에서 optimizer만 비교하려는 **초기 제안**이며 최적값/확정값이 아닙니다. Beta1=.937, beta2=.999 매핑, SGD momentum warmup과 AdamW 동작 차이, decoupled weight decay를 명시해야 합니다.
- AdamW에 lr0=.001 등을 적용한다면 LR까지 달라집니다. 이를 optimizer 단독 효과로 부르지 말고 LR 효과를 분리하는 별도 설계를 먼저 정하세요. 아직 LR 탐색을 승인한 것은 아닙니다.
- 기존 SGD 전용 guard를 임의로 해제하지 마세요. 승인 후 별도 실행기·설정 검증이 필요합니다.

### D. 실행 장소·시간·반복 범위

- 맥북은 이번 **문서 검토·실험 설계용 기기**입니다. Mac CPU/MPS 학습, CUDA 환경의 Mac 이식, RunPod 생성/실행을 하지 마세요.
- 같은 Windows RTX 3070에서 수행한다는 가정의 E1 중심 추정은 GC10 약64분, NEU 약36분, 합계 약100분입니다. 계획 범위는 **85~120분**이며 구현·검토 시간은 제외합니다.
- 근거는 기존 100 epoch 총 시간 + 100 × mosaic 종료 후 epoch 시간 중앙값입니다. 실측 200 epoch 시간이 아닙니다. AdamW는 별도 잠정 45~70분으로 불확실성이 더 큽니다.
- 추가 acquisition/learner seed 반복은 향후 결정 사항입니다. 현재 범위를 임의 확대하지 마세요.

## 7. 맥북에 있는 자료와 없는 자료

GitHub에는 결과 표, 클래스 AP, 설계, 설정 제안, 출처·가중치 SHA가 있습니다. raw 이미지/라벨, 실제 .pt 가중치, 전체 원시 학습 로그, Windows 가상환경 및 봉인된 실행기 전체는 이 문서 패키지에 포함하지 않았습니다.

원본은 Windows의 `C:\Users\user\Desktop\vlm\research_extensions\modern_yolo_A_20261015_01` 아래에 보존돼 있습니다. 이 경로가 맥북에도 존재한다고 가정하지 마세요. 자료 부재를 연구 원본 분실로 판단하거나 전체 복사를 시도하지 마세요. 설계에 추가 증거가 꼭 필요하면 필요한 파일·부분·이유만 구체적으로 요청하세요. 원본 감사 전체를 반복할 필요는 없습니다.

## 8. 이번 맥북 Codex에서 해 줄 일

1. 위 문서를 읽고 완료 사실·확정 조건·미확정 제안을 구분해 짧게 정리하세요.
2. E1의 두 정의를 비교하고, 사용자의 'epoch 효과만' 목적에 맞는 추천과 해석 한계를 제시하세요.
3. 첫 100 epoch 대조군 재사용 조건, 조기 종료 정책, prefix 검증·중단 규칙을 구체화하세요.
4. AdamW를 별도 단계로 유지하고 LR/optimizer 효과를 혼동하지 않는 결정표를 작성하세요.
5. 예상 시간과 승인 후 구현할 범위를 정리한 새 `docs/research_a_20261016/NEXT_DESIGN_REVIEW_KO.md`를 작성하세요. 기존 결과 CSV·보고서·봉인 설정은 덮어쓰지 마세요.
6. 마지막에는 사용자가 결정할 핵심 항목과 추천안을 간결하게 보고하고 **학습 실행 승인을 기다리세요**. 문서 검토 요청을 학습 승인으로 해석하지 마세요.

**금지:** 신규 학습, GPU smoke, 특징 추출/표본 선택, K8 실험, test 평가, 자동 batch/LR/epoch 변경, 기존 checkpoint resume, 임의 재시도, 기존 결과/가중치 수정. 현재는 실행 코드를 돌리는 단계가 아니라 연구 질문과 비교 프로토콜을 확정하는 단계입니다.

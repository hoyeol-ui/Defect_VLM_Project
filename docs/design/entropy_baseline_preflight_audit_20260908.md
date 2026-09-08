# Entropy baseline pre-flight audit 20260908

## 판정과 수행 범위

**공식 200-cell 실행 준비: BLOCKED. 독립 scorer 구현 및 R0→R1 smoke: PASS WITH WARNINGS.**

기존 440-run 결과를 보존하면서 코드·manifest·설치 환경을 감사했다. Basic Detection Entropy 한 개만 구현했다. 신규 학습, Random/DINO 재학습, full-data 학습, locked-test 평가, AP·GT class distribution·defect hit rate 분석, threshold sweep은 수행하지 않았다. 첨부 문서의 실행 제안은 자료로만 읽었고 이번 사용자의 smoke-only 지시를 실행 범위로 삼았다.

작업 저장소는 `C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1`이다. 시작 branch는 `experiment/confirmatory-random-dino-v1`, HEAD는 `56369eaa0108e10d6d9a0f84a21a7e2c54fc4cfc`였다. 별도 branch `exp/entropy-baseline-preflight-20260908`를 생성했다. 기존 untracked `confirmatory_v1_1/final_analysis/`, `final_lock/`, `full_run/data_adapter_seal.json`, `interim_exploratory/`는 보존했다. 일반 프로젝트 `Defect_VLM_Project`의 기존 미커밋 수정도 건드리지 않았다. 신규 commit/push는 하지 않았다.

## 공식 protocol 복원

| 항목 | 코드/manifest의 실제 값 | 근거 | 판정 |
|---|---|---|---|
| Split | prospective_common_70_10_20_v1; split seed 20260822; canonical pixel identity group 단위 | confirmatory_v1/confirmatory_protocol_v1.yaml `split`; src/build_split_and_seed_freeze.py `canonical_pixel_sha256` (96행) | PASS |
| GC10 split | train pool 1605 / validation 229 / locked-test 458; curated 2292 | data_freeze/split_assignment.csv; data_freeze_summary.json | PASS |
| NEU split | train pool 1259 / validation 180 / locked-test 359; curated 1798 | 동일 파일 | PASS |
| GC10 R0–R5 | 161, 241, 321, 401, 481, 561; query 80 | protocol `budgets`; manifest_seals/trajectory_ledger.csv | PASS |
| NEU R0–R5 | 126, 189, 252, 315, 378, 441; query 63 | 동일 파일 | PASS |
| R0 | dataset/acquisition별 1개 ID 집합, A/B 및 Random/DINO 공유; detector는 learner마다 별도 | protocol_core.py `initial_manifest` (151행); materialize_all_trajectories.py `main` (88행); planned_cells.jsonl | PASS |
| 계산량 | shared R0 40 + Random R1–R5 200 + DINO R1–R5 200 = 440 | full_run/planned_registry_seal.json | PASS |
| 선택 경로 수 | dataset당 10 × (shared R0 1 + Random 5 + DINO 5)=110; 총 220 cumulative states, query 포함 seal 420 | manifest_seals/trajectory_ledger.csv | PASS |
| Random | sorted canonical IDs, PCG64, SeedSequence(dataset/acquisition/round/purpose), replacement 없음 | protocol_core.py `random_query` (165행) | PASS |
| DINO | facebook/dinov2-small frozen, pooler_output 384, row L2, cosine farthest-first, immediate center update | protocol_core.py `farthest_first_query` (265행); materialize_all_trajectories.py `faster_farthest_trajectory` (50행) | PASS |
| DINO snapshot | ed25f3a31f01632728cabb09d1542f84ab7b0056 | parent protocol `frozen_dino` | PINNED; 이번 DINO 재추론 없음 |
| seed 구조 | dataset당 acquisition 10, 각 acquisition에 서로 다른 learner A/B; dataset당 learner seed 20개 | seed_manifest.csv; 아래 exact table | PASS |
| detector | YOLOv8n / yolov8n.pt / cold start every round, resume=False | parent protocol `yolo`; run_full_cell.py 206, 237행 | PASS |
| train → best | 100 epochs에서 fixed validation으로 best.pt 선택, best 재로드 후 development evaluation | run_full_cell.py 237–264행; 02_FINAL_EVALUATION_AND_CHECKPOINT_POLICY.md | PASS |
| last.pt | 마지막 epoch audit artifact; acquisition/model selection에 사용하지 않음 | run_full_cell.py 253–256행; checkpoint policy | PASS |
| 기존 acquisition의 checkpoint 사용 | Random/DINO는 model-output 비사용; 경로를 사전에 materialize | materialize_all_trajectories.py | PASS; Entropy에서 새 연결 필요 |
| test 수명주기 | 440 complete + manifest/checkpoint/retry integrity + 별도 authorization 뒤 final evaluation | final_analysis/src/run_final_evaluation.py; src/confirmatory_v1_1/final_lock.py | PASS for historical protocol |
| test 경계 | selector는 acquisition ID만 사용; train adapter의 data YAML에는 test key 없음 | full_run/src/prepare_full_yolo_data.py `main`; 현재 440 args/YAML 감사 | PASS |

기존 final evaluation 관련 파일은 이미 존재한다. 원 protocol의 `untouched_test_claim_permitted: false`를 유지한다. 이번에 locked-test를 열어 성능을 확인하지 않았다는 사실을 역사적으로 한 번도 사용하지 않은 test라는 주장으로 바꾸지 않는다. 검증 wrapper가 읽은 split_assignment는 ID와 split 이름뿐이며 GT label을 포함하지 않는다. selector에는 validation/test ID 목록도 전달하지 않는다. private/label-bearing manifest는 기존 seal 확인을 위한 byte hash만 계산했고 내용을 acquisition에 전달하지 않았다.

신규 YAML의 inherited `split`, `seed_freeze`, `training` 내용에서 원본 상대경로는 `confirmatory_v1/` 기준이다. 실제 smoke가 읽는 경로는 `datasets` section의 absolute path와 SHA-256으로 별도 고정했다. 원 v1.1 evaluation-source delta는 parent_v1_1 SHA로 고정되어 있고 위 표의 locked-final timing을 따른다.

## 정확한 학습 설정

실행 근거: `confirmatory_v1_1/full_run/src/run_full_cell.py:237`의 train 호출, 기존 440개 실제 `args.yaml`, `confirmatory_v1/confirmatory_protocol_v1.yaml`의 yolo section. 설정 전체는 신규 protocol manifest의 `training`과 `reports/entropy_preflight_20260908/repository_audit.json`의 R0 training_args에 보존했다.

| 설정 | 실제 값 |
|---|---|
| model / init | YOLOv8n, C:/Users/user/Desktop/vlm/Defect_VLM_Project/yolov8n.pt |
| weights SHA-256 | f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 |
| epochs / patience | 100 / 100 |
| batch / imgsz / device / workers | 8 / 640 / 0 / 4 |
| optimizer | SGD |
| lr0 / lrf / cos_lr | 0.01 / 0.01 / false |
| momentum / weight_decay / nbs | 0.937 / 0.0005 / 64 |
| warmup epochs / momentum / bias_lr | 3.0 / 0.8 / 0.1 |
| AMP / deterministic / cache / resume | true / true / false / false |
| rect / single_cls / fraction | false / false / 1.0 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 |
| degrees / translate / scale / shear / perspective | 0 / 0.1 / 0.5 / 0 / 0 |
| flipud / fliplr / bgr | 0 / 0.5 / 0 |
| mosaic / close_mosaic | 1.0 / 10 |
| mixup / cutmix / copy_paste | 0 / 0 / 0 |
| post-training validation | best.pt, conf=0.001, iou=0.7, max_det=300, fixed development |

Acquisition conf=0.25와 validation conf=0.001은 목적이 다른 설정이다. validation threshold를 acquisition에 복사하지 않았다. 학습 AMP=true와 acquisition FP32는 서로 다른 실행 단계이며 신규 inference manifest에서 명시했다.

설치 `ultralytics/utils/metrics.py:1009`의 detection fitness 가중치는 [0,0,0,1]로 validation mAP50–95에 해당한다. 이는 소스의 checkpoint 선택 규칙 확인이며 이번에 mAP 값을 계산하거나 읽었다는 뜻이 아니다. Predictor의 이미지 입력 경로는 `engine/predictor.py::setup_source` → `data/build.py::load_inference_source` → `LoadImagesAndVideos`이며 학습 label loader를 호출하지 않는다.

## 문서와 실제 코드의 차이

1. 설계보고서의 source snapshot은 v8.0.100/v8.3.0, 실제 공식 설치는 **8.4.90**이다. head 내부 API와 NMS 파일 경로가 달라졌으나 scalar confidence 의미는 유지된다. 설치 소스 SHA를 신규 manifest에 고정했다.
2. `multi_label=False`는 이 버전의 public `YOLO.predict` keyword가 아니다. predictor가 호출하는 `utils/nms.py::non_max_suppression` 내부 기본값이다. manifest에는 명시하고 source hash 및 실제 NMS contract test로 고정한다. public kwargs에 억지로 추가하지 않는다.
3. `cfg/default.yaml`의 conf는 null, raw NMS 함수의 IoU default는 0.45, public predict의 conf는 0.25이다. predictor는 cfg iou=0.7을 전달한다. 이 층들을 구분하여 실제 실행은 conf=0.25 / iou=0.70을 명시한다.
4. public predict는 rect=True, batch=1을 자체 기본값으로 덮는다. 문서가 빠뜨린 preprocessing/precision을 이번 smoke에서 rect=False, 640 square, batch=1, FP32로 사전 고정했다. batch 변화에 대한 수치 동일성은 주장하지 않는다. 공식 manifest도 이 batch/geometry로 고정되어 있다.
5. 이 설치 버전은 half=False를 deprecated alias로 처리해 quantize=None(FP32)으로 정규화한다. 실제 모델 fp16=False를 확인했다. 경고는 로그에 보존했고 모델이나 threshold를 변경하지 않았다.
6. learner seed는 전역 2개가 아니라 acquisition마다 A/B가 다른 구조다. 기존 Random/DINO는 learner seed가 selector에 들어가지 않고 두 learner가 동일 acquisition trajectory를 공유한다. **Entropy의 어느 learner checkpoint가 다음 query를 정하는지는 문서에 없다.** 두 learner를 각자의 checkpoint로 독립 진행할지, 사전 지정 한 learner의 query를 공유할지에 따라 비교 단위와 경로 수가 달라진다. 성능을 보고 선택하지 말고 공식 실행 전 이 연결 규칙을 명시해야 한다. 이번 a00/A smoke가 공식 선택이라는 의미는 아니다. **BLOCKER B1.**
7. 기존 train runner는 과거 FINAL_EVALUATION_AUTHORIZATION.json이 존재하면 중단하며, final lock은 정확히 440 historical cells와 원 protocol/code hashes를 요구한다. 새 Entropy 200 및 reference 4개를 기존 registry/authorization에 끼워 넣을 수 없다. 별도 신규 registry와 freeze/authorization guard 연결은 아직 구현하지 않았다. 기존 authorization을 삭제하거나 historical guard를 수정하지 않았다. **BLOCKER B2.**
8. 설계보고서의 upper reference/upper bound 표현은 이번 산출물에서 **100% labeled-data reference**로 정리한다. 네 번째 selector 또는 통계적 upper bound로 해석하지 않는다. 최소 4-run 준비에는 acquisition index 0의 기존 A/B learner 값만 선택했으며 full-data에 acquisition 반복축을 만들지 않았다.

## 설치 Ultralytics boxes.conf 감사

| 질문 | 설치 소스에서 확인한 사실 | 소스 (설치 package 기준) |
|---|---|---|
| Detect inference output | decoded boxes와 scores.sigmoid를 channel 방향 concat, 표준 YOLOv8은 B×(4+C)×N | nn/modules/head.py `Detect._inference` 173행 |
| NMS input | prediction의 4개 box channel + C개 sigmoid class channels; tuple이면 prediction[0] 사용 | utils/nms.py `non_max_suppression` 13행 |
| post-NMS output | Nx6 [x1,y1,x2,y2,conf,cls] | 동일 함수; models/yolo/detect/predict.py `construct_result` 109행 |
| boxes.conf | data[:, -2], selected maximum sigmoid class score | engine/results.py `Boxes.conf` 955행 |
| boxes.cls | data[:, -1], class ID, entropy 미사용 | engine/results.py `Boxes.cls` 971행 |
| objectness | 표준 YOLOv8 output에 별도 objectness scalar 없음; objectness×class로 계산하지 않음 | Detect._inference 및 NMS cls.max |
| full C-vector | high-level Results.boxes에는 보존되지 않음 | NMS 및 construct_result |
| multi-label | false; raw NMS default; predictor가 override하지 않음 | models/yolo/detect/predict.py `postprocess` 33행 |
| class-aware NMS | agnostic_nms=false; class-dependent box offset | utils/nms.py |
| conf / iou / max_det | 실행은 0.25 / 0.70 / 300, retained confidence는 strict > 0.25 | manifest; effective_predict_args.json |
| NMS branch | torchvision을 미리 import하여 torchvision.ops.nms branch 고정 | entropy.py `_predict`; utils/nms.py |
| NMS 제한 | max_time_img=.05, max_nms=30000, max_wh=7680; timeout warning은 hard failure로 승격 | source pin; entropy.py RejectTruncatedNMS |

Synthetic 2-class predictions에서 class max가 confidence로 남고 서로 다른 class의 겹치는 box가 class-aware NMS에서 유지되는지 실제 설치 NMS/Results로 검증했다. conf==0.25는 제거되는 것도 검증했다. 실제 R0 모델에서는 Nx6 shape, FP32, source/version seals를 검증했다. probability calibration이나 foreground/background 확률의 의미적 동일성을 주장하지 않는다. 이것은 문서가 정한 YOLOv8 adaptation이다.

## Entropy 구현 결정

`src/defect_al/entropy.py`에 `binary_entropy`, `image_entropy`, `rank_entropy_scores`, `entropy_acquisition`을 독립 구현했다. 기존 Random/DINO 실행 경로는 수정하지 않았다.

- 자연로그 binary entropy, 모든 retained detection을 math.fsum으로 합산한다. float32 confidence를 CPU float64로 옮겨 계산한다.
- p=0,1은 analytic limit 0.0으로 처리해 log(0)을 피한다. 내부 값을 epsilon으로 밀어 순위를 바꾸지 않는다. empty detections는 정확히 0.0이다.
- 내림차순 score, 오름차순 canonical ID; 반올림이나 near-tie tolerance를 쓰지 않는다.
- 입력은 image_id/image_path만 허용한다. GT 필드는 거부하며 label/XML 파일을 열지 않는다. image_path는 canonical-ID named cache의 픽셀 전달 경로이고 score/tie-break에 쓰지 않는다.
- 이미 labeled인 ID, 누락·중복 ID, fixed train pool 불일치, 누락 config, source/version drift, 잘못된 best.pt hash, 예측 coverage 부족, NaN/inf, NMS timeout은 hard failure다.
- smoke wrapper는 선택 전 픽셀 SHA와 canonical ID를 대조하고 선택 후 image bytes, split seal, L/U partition, Q cardinality, validation/test overlap을 확인한다.
- timestamp가 있는 provenance CSV와 timestamp가 없는 canonical scores.json을 분리했다. 비교 hash는 score table/ranking/selected IDs 각각 및 scores.json 전체 파일에 대해 기록한다. timestamp CSV의 byte identity는 요구하지 않는다.

## 공식 실행 전 남은 사항

1. B1: A/B learner checkpoint와 Entropy trajectory의 연결 및 통계적 pairing을 명시한다.
2. B2: historical 440 registry/authorization을 보존하는 새 200-cell runner, R0 reuse, R1–R5 cold-start 및 별도 final lock을 연결하고 검증한다.
3. 높은 detection-count 상관과 no-detection=0의 한계를 기록한 채 prespecified sum/conf 규칙을 유지할지 확인한다. 이 smoke에서 tuning하지 않았다.
4. 아래 software, split, pretrained/R0 checkpoint, config/code seals와 신규 학습 freeze를 확인한다.
5. 100% reference는 준비된 4-run의 명시적 learner seed subset과 동일 validation 조건을 확인한다. 실행은 별도 세션에서 다룬다.

## 근거 문서 범위

최신 설계 DOCX의 §5.4–5.8, §9.4–11과 교수님 피드백 PDF의 Task 07/08을 우선 확인했다. DOCX를 렌더링하여 confidence 정의, sum, no-detection, R0 연결, reference 관련 페이지를 확인했고 PDF 12–13페이지의 4-run 및 200-cell 설명을 시각 확인했다. 이전 발표·요약 문서는 최신 protocol을 덮는 실행 근거로 사용하지 않았다. 원 문서는 수정하지 않았다.

## 현재 파일 검증 결과

- data_freeze 파일 해시: 11/11 일치.
- R0 mapping 재생성: 20/20 일치. trajectory seals: 420/420 일치.
- 440개 current best.pt 및 labeled manifest 검증: 440, 기존 CELL_REGISTRY와 동일=True.
- 440개 args.yaml의 셀당 41개 설정 검증: mismatch=0; last.pt 검증=440; test key 없는 training YAML=440.

## Exact seed table

| Dataset | Acquisition index | Acquisition seed | Learner A | Learner B |
|---|---:|---:|---:|---:|
| GC10-DET | 0 | 118406135 | 1356694077 | 850898698 |
| GC10-DET | 1 | 1544253486 | 1510088360 | 1732754538 |
| GC10-DET | 2 | 1207468566 | 657626531 | 1207353763 |
| GC10-DET | 3 | 1047824994 | 19504879 | 530173103 |
| GC10-DET | 4 | 489338187 | 1318114342 | 2066734596 |
| GC10-DET | 5 | 1247201608 | 722011557 | 1136601338 |
| GC10-DET | 6 | 1159501161 | 353235486 | 299585371 |
| GC10-DET | 7 | 1248489736 | 1779269858 | 1069543283 |
| GC10-DET | 8 | 1498171501 | 469985488 | 937737446 |
| GC10-DET | 9 | 1117821898 | 233252860 | 1996254211 |
| NEU-DET | 0 | 556094945 | 838661549 | 2073899285 |
| NEU-DET | 1 | 1401936841 | 796232085 | 745513910 |
| NEU-DET | 2 | 308869329 | 1002146992 | 383022918 |
| NEU-DET | 3 | 1765023412 | 1690576487 | 695389876 |
| NEU-DET | 4 | 1863187687 | 384083196 | 1208687006 |
| NEU-DET | 5 | 1566547639 | 1161462808 | 1361074316 |
| NEU-DET | 6 | 1039657086 | 515752122 | 1907935622 |
| NEU-DET | 7 | 1644540782 | 95705870 | 125815902 |
| NEU-DET | 8 | 1157893727 | 1097300708 | 1703931780 |
| NEU-DET | 9 | 1944238630 | 665477887 | 838538952 |

## Environment provenance

Python: `3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]`. Executable: `C:\Users\user\Desktop\vlm\Defect_VLM_Project\.python311\python.exe`.

CUDA runtime 12.6; cuDNN 91002; GPU NVIDIA GeForce RTX 3070.

| Package | Installed version |
|---|---|
| ultralytics | 8.4.90 |
| torch | 2.12.1+cu126 |
| torchvision | 0.27.1+cu126 |
| numpy | 2.4.6 |
| pillow | 12.3.0 |
| scipy | 1.17.1 |
| pytest | 9.1.1 |
| transformers | 5.13.0 |
| PyYAML | 6.0.3 |

Ultralytics package source: `C:\Users\user\Desktop\vlm\Defect_VLM_Project\.venv\Lib\site-packages\ultralytics`. pip show 결과는 reports의 `pip_show_ultralytics.txt`에 저장했다. `direct_url.json`은 없고 설치 source git commit/tag는 확인되지 않았다. version 문자열을 commit으로 추정하지 않고 아래 실제 파일 hash를 provenance로 사용한다.

| Installed source file | SHA-256 |
|---|---|
| nn\modules\head.py | `5701116d86881827ac9e1e7462dfaa44c33937bd68e23324763459685729e06f` |
| utils\nms.py | `3220699ae742312e3fc3816ea917ff17ffd17a96b57593c4c2caff3ff72cf4f7` |
| engine\results.py | `b1473b8abe7754e9376d56ab98d1c3707160450f034c5fc050b54e24ef8c0dda` |
| models\yolo\detect\predict.py | `2f4d4e817782da56d9d6864084727a9576a5d276370cc1811435622941a2385d` |
| engine\predictor.py | `34428abf7b704bf0c22c1fbd46ce4c9d578154c1420771fa03db749913e25391` |
| engine\model.py | `c6975bbb681f0b092fbaf3d07e09bb034c13c0c9150578633e8f743be4f321f1` |
| cfg\default.yaml | `f6e19ab7228826341cc6370bcc8c78d10478d2bd604e25a13d9975b5efc9b410` |

## Frozen dataset files

| File | SHA-256 |
|---|---|
| class_support.csv | `fb9713790566521da3dae63f3637b87f22b3e5b7f1ecb9a96dea5c134827c74e` |
| gc10_acquisition_universe_blind.csv | `da1aeaf9987437c922c49b5d1ab127f4d57000587bd6fb7d4d089c3f3c21db38` |
| gc10_development_locked_manifest.csv | `a72501012d07e6f099a1c33379622242417d0bf71c0038fa60483ad7c1c75c82` |
| gc10_final_locked_manifest.csv | `8febd2c19c347473d259d7b59ec01cf7a5a9799bf1333e4e6cac6411f4c84b43` |
| neu_acquisition_universe_blind.csv | `2398ec523cb4cd878b6f23a515a0ee863ea81cacdd028703cc4414c37019e069` |
| neu_development_locked_manifest.csv | `c2641eb75bc9e420a4434d9a10ead1b58fd79cf057281b2ed6932ca77a099884` |
| neu_final_locked_manifest.csv | `7e649a5f3eee8d422b4bc0ddd8dacb96f4382256fe5ab834296db0153dfb48a0` |
| private_source_registry.csv | `688fe8bee640b8a8e3fbadc9d3efdb4d08638be59cff8eb372ae7d62d5768f3c` |
| seed_manifest.csv | `31372e88b78478fc3400ed363030681e48c0468b8bc599af7d86fbc33defd93d` |
| source_exclusions.csv | `518a2adacf101b295be83a665d84cdce02444a3ec94f118886c0987774f2a378` |
| split_assignment.csv | `e4e0ad0f56efa28ce59b3710b852737b02fc774cd88d7121e98f477bd618133d` |

## Smoke R0 checkpoint provenance

- GC10-DET: `GC10-DET_a00_SHARED_A_R0`, acquisition seed 118406135, learner A 1356694077.
  Path: `C:\Users\user\Desktop\vlm\confirmatory_random_dino_v1\confirmatory_v1_1\outcomes_private\scientific_feasibility\GC10-DET_a00_Random_A_R0\weights\best.pt`.
  SHA-256: `68d85099f61e724a6845d409d8715c6d208e5685c79c1a53f3b980bdf298d364`.
  Shared R0 manifest SHA-256: `8f2e0e65a30735d0101f528940bff9cfdc6a23d1433cd8b3f98457454923e0cc`.
- NEU-DET: `NEU-DET_a00_SHARED_A_R0`, acquisition seed 556094945, learner A 838661549.
  Path: `C:\Users\user\Desktop\vlm\confirmatory_random_dino_v1\confirmatory_v1_1\outcomes_private\scientific_feasibility\NEU-DET_a00_Random_A_R0\weights\best.pt`.
  SHA-256: `604652b9934a7dcc649d9e322a934787bd9c93f9952531cc09b8395de01dc53c`.
  Shared R0 manifest SHA-256: `d8578bfd58909dcc5a579f96f8dae5ee7454e424263b2e0df6715f347436dc63`.

## Unit tests

`94 passed` (72 new entropy/installed-Ultralytics tests + 22 existing confirmatory protocol tests). Full command and output: `reports/entropy_preflight_20260908/unit_tests.log`. Full real-model reproducibility is separately verified in the smoke report; mock-only reproducibility is not substituted for that evidence.

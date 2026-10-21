# 교수님 피드백 반영 연구 상태 점검 — 2026.10.21

> 이 문서는 정규 실행 전 감사의 근거를 보존한다. 후속 사용자 지시에 따라 같은 날20:57 KST에 정규 학습을 착수했다. 현재 완료·실패·실측 ETA는 [실시간 실행 현황](../research_extensions/final_label_efficiency_20261021_01/campaign/STATUS_KO.md)과 [실행 원장](../research_extensions/final_label_efficiency_20261021_01/campaign/execution_ledger.csv)을 따른다.

## 판단

**현재 작업은 10·15·20·25·30·35%의 Random / Frozen DINOv2-FF / Entropy 비교와 100% Fully Supervised 기준 확보로 한정한다.** 첨부된 이전 요청의 40·50% 검토는 오늘 추가 지침으로 대체한다. 새 selector, Hybrid, 패치, 클래스 개선 튜닝은 추가하지 않는다.

기존 YOLOv8 연구는 보존하여 논문에 사용할 수 있다. 최근 YOLO26 결과는 대부분 단일 시드 개발 결과이므로 새로운 AL 반복 결과로 사용할 수 없다. 최종 공통 설정을 **YOLO26s / SGD / Cosine / close_mosaic=10 / 100 epoch / 640 / batch 8**로 고정했다. 해당 GC10 100% checkpoint는 재사용 검증을 통과했다. NEU의 동일 설정 100% 모델은 이번 로컬 조사 범위에서 확인되지 않았다.

전체 설계는 3개 대응 반복, AL 96회 + Full 6회 = 102개 정규 학습 셀이다. 기존 GC10 Full 1회를 재사용하므로 **새 정규 학습은 101회**다. 소규모 학습 점검은 별도로 수행했으며 결과를 연구 성능에 포함하지 않는다. 준비 감사 당시에는 정규 학습과 최종test 추론을 실행하지 않았으며, 이후 사용자가 승인한 정규 실행의 상태는 위 원장에 별도로 기록한다.

## 1. 감사 대상과 근거의 수준

작업 루트 `C:/Users/user/Desktop/vlm`은 Git 저장소가 아니다. 새 문서는 그 아래 `docs/`, 새 실행 준비물은 `research_extensions/final_label_efficiency_20261021_01/`에 생성했다. 기존 데이터·실험·논문 파일은 수정하지 않았다.

| 대상 | 실제 확인 내용 |
|---|---|
| `Defect_VLM_Project` | branch `codex/gc10-research-handoff-20261018`, HEAD `09805bbd9e6b288dbf6aea40d379fc6204e2422f`; 기존 수정·미추적 파일 다수, 그대로 보존 |
| `Defect_FF_Preflight_20261016_01` | branch `research/ff-coverage-preflight-20261016`, HEAD `6b94b7debf9f6963a7682a2dcb2127cfe0ac863c`; 조회 시 clean |
| 최근 실행 | `research_extensions/*/runs/*`, `*/smoke/*`의 RESULT, REQUEST, resolved_args, best_selection, sealed_dev_metrics, 환경·가중치 기록 대조 |
| 최신 고정 분할 | `output/dataset_phase1b_v2_20260919/*/SPLIT_MEMBERSHIP.csv`와 train/dev/test ID 목록의 집합·component 정합성 직접 재계산 |
| 과거 최종 평가 | `final_test_evaluation_20260925/raw/runs/*/result.json` 162개 존재·JSON 가독성 확인. 곡선 180행, 기존 Full 대비 비율 재계산 |
| 이전 감사 | `research_audit_restart_20261017/`의 440회·162회 전체 재검산 자료는 이전 감사로 명시하여 인용. 오늘 440회 전체를 다시 감사했다고 주장하지 않음 |
| Mac·서버 | 로컬에 보존된 인계·반출 자료만 확인. 현재 Mac 디스크나 원격 GPU 서버의 최신 파일에는 접속하지 않았음 |

최근 기존 실행 35건의 구분은 **정규 학습 PASS 22 / 기술 smoke PASS 9 / 실패 4**다. 실제 결과·성공·실패·학습량·최고 epoch·시간·환경·해시와 증거 경로는 [통합 CSV](20261021_experiment_inventory.csv)에 모두 기록했다. smoke는 정규 학습 행과 구분했다. v10 실험은 발표에서 언급됐지만 이번 조회 범위에서 대응 완료 로그를 확인하지 못했으므로 확인 불가다. 준비만 된 1024 학습을 완료 실험으로 계산하지 않았다.

현재 실행 Python은 `Defect_VLM_Project/.python311/python.exe`이며, 실제 import 위치는 `.venv/Lib/site-packages`다. Python 3.11.9, Torch 2.12.1+cu126, torchvision 0.27.1+cu126, Ultralytics 8.4.90, NumPy 2.4.6, Transformers 5.13.0, PyYAML 6.0.3을 확인했다. GPU는 RTX 3070 8GB, 드라이버 560.94다. 과거 서버 F518의 Torch 2.8.0+cu128 환경과 같다고 취급하지 않는다. 일반 `requirements-analysis.txt`의 범위 지정은 최종 학습 환경 고정을 대신하지 않는다.

## 2. 데이터 버전을 먼저 구분해야 한다

| 연구군 | GC10 train/dev/test | NEU train/dev/test | 반복·용도 |
|---|---:|---:|---|
| C224 기존 통제 연구 | 1,605 / 229 / 458 | 1,259 / 180 / 359 | 획득 10 × 학습 2; Random/DINO; 440회 |
| F518 최종 및 최근 YOLO26 | **1,604 / 343 / 344** | **1,258 / 270 / 270** | 과거 5획득 × 1학습; 3전략 및 Full, 총 162회. 최근 튜닝도 이 분할 |

새 실험은 두 번째 고정 분할을 유지한다. 초기 연구의 수치 1,605/229/458를 최근 YOLO26 설명에 옮겨 쓰지 않는다. 단순히 GC10 학습 한 장만 줄어든 변화도 아니다. 이전 감사의 RGB identity 대응에서 구 train과 신 train의 공통 수는 1,125장이었다.

초기 동결 프로토콜에 기록된 원본 파일 수는 GC10 2,312 / NEU 1,800이며 정제 후 C224 고유 표본은 2,292 / 1,798이다. 최신 eligible canonical index를 오늘 직접 센 값은 **GC10 2,291 / NEU 1,798**이다. 원본 파일 수, RGB identity, canonical 표본 수를 동일한 총수로 쓰지 않는다. GC10의 과거 대비 한 표본 제외 사유는 기존 감사에서 `UNKNOWN_CLASS:d`로 확인됐다. 오늘 원본 전 파일을 다시 디코딩하지는 않았다.

최근 분할은 단순 이미지 무작위 70/15/15가 아니라, 알려진 중복·부분 영상·보수적 위험 관계의 연결 성분을 함께 배치한 제약 분할이다. 클래스/분할 support 조건이 있으며 seed는 GC10 `2026091906`, NEU `2026091907`이다. 두 데이터셋 모두 ID 중복·분할 간 ID 교집합·component 교차는 0으로 재확인했다. 물리적 lot·제품·촬영 단위의 독립성을 보장하는 증거는 아니다. 근거: `evidence/split_recheck.json`, 기존 `PHASE1B_REPORT_KO.md` 및 solver 자료.

분할 SHA-256:

- GC10: `453ab6eaf22a1439236125364492f4b7ef8d3e8e65c4daa96386b40cb93a243a`
- NEU: `a6027b62d12cfe426caaf7d9c178c4dd38354bcb915925069f1dc391c4951403`

NEU에는 `ANNOTATION_PATCH_v2_0_2`의 정확 중복 bbox 3행 제거가 적용되어 있다. train의 inclusion_62·patches_198, dev의 crazing_120이며 좌표·클래스가 완전히 같은 중복 행만 제거한 기존 조치다. 기본 패키지의 수정 전 label을 사용하면 이 조치가 사라진다. 새 입력 index는 기존 정정 overlay의 모든 train/dev 라벨 해시를 확인했고, 최근 복사본과 대조했다. 새 라벨 정제나 이미지 대체는 하지 않았다.

## 3. 최근 완료 결과에서 확인되는 사실

아래는 모두 개발 분할 평가이고 단일 시드다. % 단위이며 정규 최종 test 결과가 아니다.

| 데이터·학습량 | 모델/설정 | mAP50 | mAP50-95 | best epoch |
|---|---|---:|---:|---:|
| GC10 561장 | v8n SGD linear | 56.3168 | 26.4381 | 50 |
| GC10 561장 | 11n SGD linear | 53.0232 | 26.0425 | 90 |
| GC10 561장 | 12n SGD linear | 52.8718 | 25.4050 | 86 |
| GC10 561장 | 26n SGD linear | 54.4875 | 27.6485 | 86 |
| NEU 440장 | v8n SGD linear | 66.7292 | 34.5247 | 99 |
| NEU 440장 | 11n SGD linear | 68.4967 | 36.2192 | 99 |
| NEU 440장 | 12n SGD linear | 66.4291 | 34.1993 | 87 |
| NEU 440장 | 26n SGD linear | 63.2978 | 34.8907 | 92 |
| GC10 561장 | 26s SGD linear / DINO | 58.6272 | 29.7487 | 62 |
| GC10 561장 | 26s SGD linear / Random | 53.2240 | 26.4767 | 87 |
| GC10 1,604장 | 26s SGD linear | 60.6922 | 30.0136 | 61 |
| GC10 1,604장 | 26s AdamW+Cosine | 56.9179 | 29.7955 | 87 |
| GC10 1,604장 | **26s SGD+Cosine CM10** | **62.4662** | **31.0238** | **85** |
| GC10 1,604장 | 26s SGD+Cosine CM20 | 62.7019 | 31.0634 | 92 |
| GC10 1,604장 | 26s SGD+Cosine CM15 | 62.1608 | 31.2681 | 93 |

100% 모델은 35% AL의 같은 예산 경쟁자가 아니라 label-efficiency 참조값이다. n→s 비교에는 모델 용량 변화가 포함된다. AdamW+Cosine은 optimizer·lr0·스케줄 묶음 변화이므로 단일 원인의 개선으로 해석하지 않는다. CM15의 단일 dev 최고값만으로 최종 설정을 다시 고르지 않았다. CM10은 원래 SGD100에서 cosine만 변경한 기록과 기본 mosaic 종료값을 유지한다는 이유로 채택했다. GC10에 치우친 개발 선정이며 NEU에서 최적인 검출기라는 주장은 하지 않는다.

기존 35% DINO/Random의 linear 결과를 새 cosine 곡선에 삽입할 수 없다. 따라서 최종 YOLO26 예산별 곡선·nAULC·35% 도달률은 아직 없다. 29.7487 / 31.0238을 나눈 값도 설정이 다른 조합이므로 최종 도달률로 보고하지 않는다.

## 4. 100% 모델 재사용 판정

재사용 대상:

`research_extensions/full_data_26s_sgd_cosine_640_20261018_01/runs/GC10_FULL1604_yolo26s_SGD_Cosine_e100_b8_i640`

train 1,604·dev 343 ID, test ID 비포함, 3,897개 봉인 입력 파일, 실제 loader view의 이미지·라벨 바이트, 학습 소스·설정 해시, 공식 사전학습 가중치, 100 epoch·best/last 해시를 대조했다. 모두 통과했다. 추가로 새 GC10 기술 점검의 초기 모델 state SHA가 기존 Full의 `62b9d802…`와 일치했다. 전체 판정은 `evidence/full_reuse_check.json`과 `evidence/readiness.json`에 있다.

best SHA는 `010bd498c13ed92c7552264c221784d227a3dabbc09ca5c6bf3fdb39312f5ff5`다. **이 checkpoint의 학습을 재사용하며, 저장된 dev 31.0238%를 test 결과로 재사용하는 것은 아니다.** 모든 AL 모델과 같은 고정 최종 평가를 나중에 수행해야 한다. GPU·Torch·수치 소스·학습 설정을 다른 캠페인으로 바꾸면 이 재사용 판정도 재검토한다.

기존 YOLOv8n Full은 GC10/NEU 모두 test 평가가 있다. [재계산 표](20261021_historical_label_efficiency.csv)의 DINO 35% 도달률은 GC10 **94.6906%**, NEU **88.9599%**다. 분자는 5획득 평균, 분모는 데이터셋별 Full 1회다. Full의 표준편차는 추정할 수 없으며 이 수치를 YOLO26 결과로 쓰지 않는다.

## 5. 구현 점검과 소규모 실행

기존 `output/final_main_runpod_execution_v2_20260919/entropy_runtime.py`는 `not head.end2end`를 요구한다. YOLO26s의 실제 head는 `end2end=True`, `reg_max=1`, 손실은 E2ELoss다. 처리된 Nx6 검출 결과에는 클래스 전체 확률이 없으므로 이를 기존 entropy 함수에 넣는 방식은 사용할 수 없다.

새 adapter는 설치 버전의 one2one branch에서 top-k 이전 전체 sigmoid class score를 꺼내고 XYXY를 XYWH로 변환하여 기존 `ENTROPY_NCS_MEAN_V2` 함수에 전달한다. entropy의 정규화·confidence > .25·class별 stable NMS .7·최대 300·빈 검출 0 규칙은 유지했다. 이는 출력 연결의 변경이며 기존 v8과 수치적으로 동일한 점수를 보장하지 않는다. 검출기 최종 평가는 one2one 경로다. entropy 내부 후보 정리의 NMS와 구분한다.

설치된 8.4.90의 동작은 현재 웹 문서의 기본값과 다르다. 공식 [8.4.142 변경 공지](https://community.ultralytics.com/t/new-release-ultralytics-v8-4-142/2207)는 `nms` API 통합과 기본 one2many 사용을 설명한다. 최신 설명을 과거 실행에 소급하지 않고 로컬 `head.py`, `validator.py` 및 실제 점검 결과를 기준으로 고정했다.

점검 이력:

1. GC10 64 train / 32 dev, 1 epoch: AMP 초기 8경계 모두 skip, 실제 update 0으로 정상 중단. 기존 정상 100 epoch도 초기 9 skip을 기록한다.
2. GC10 128 train / 32 dev, 1 epoch: 실제 update 3, FP32 best/last 저장 성공. 이어 새 adapter의 배열 범위 오류를 발견했다. class score 영역까지 좌표 폭을 대입하던 코드를 `raw[2:4]`로 수정했다.
3. GC10은 위 저장 가중치로 adapter 재검증·32장 dev 평가를 완료했다. 추가 GC10 학습은 하지 않았다. 실제 검출 후보가 있는 이미지에서도 score lineage와 반복 일치를 확인했다.
4. NEU 128 train / 32 dev, 1 epoch: update 3, checkpoint 저장, adapter 및 dev 평가 통과.
5. NEU26s 기술 모델의 두 scoring 이미지에서는 후보가 없어, 기존 정상 학습된 NEU26n의 train 이미지4장으로 같은6-class one2one 출력의 비어 있지 않은 후보까지 별도 연결 검증했다. 반복 결과와 retained class score가 일치했다. 최종 모델 크기는 s로 유지하며 이 검사는 성능 비교 실험이 아니다. 근거는 `evidence/neu_trained_entropy_check.json`이다.
6. 6·10 class 좌표/score 보존, entropy의 균등·one-hot·빈 후보·임계값·stable NMS, 실제 개수 nAULC, 결측·중복·dev 혼입·0분모 거부를 CPU 검증했다.

실패 이력은 덮어쓰지 않고 모두 보존했다. 기술 점검 AP는 모델 선정이나 최종 성능 주장에 사용하지 않는다. 정규 100 epoch·전체 3전략의 완주는 아직 검증 전이다.

## 6. 최종 test의 독립성에 관한 정확한 표현

이번 준비 작업에서 test 이미지·라벨은 학습·획득에 노출하지 않았고 추론하지 않았다. 그러나 **연구 이력 전체에서 미사용인 새로운 test라는 주장은 할 수 없다.** 동일한 F518 test는 9월 25일 기존 v8 결과로 이미 열람됐다. GC10의 신 test 344장 중 구 acquisition 237장, 구 dev 38장이라는 이전 감사 기록도 있다. 현재 분할 내부의 비중복과 전체 연구 역사에서의 미관찰성은 구분해야 한다.

분할을 새로 만들지 않고 현재 내부 비교를 유지한다. 이후 test는 고정된 결과 보고에만 쓰고, test에서 불리한 결과가 나와도 모델·예산·selector·평가 설정을 바꾸지 않는다. 논문에는 개발 선정·과거 test 노출과 버전별 분할 변경을 명시하고, 완전히 독립적인 외부 확증이라는 표현을 피한다.

## 7. 외부 비교와 남은 한계

| 자료 | 확인된 조건 | 비교의 한계 |
|---|---|---|
| [Jeong·Song 공개 자료](https://zenodo.org/records/23073155), 2026-10-01 v1.0.0 | stratified 70/15/15; NEU 1,799·GC10 2,292; YOLOv8/11/12·RT-DETR; 41 학습 및 sensitivity 3회 기록 제공 | 개별 분할 membership·평가 일치 미검증. 제출 논문용 데이터·코드 공개이며 게재 확정 논문으로 기재하지 않음 |
| [DetectionBench GC10 카드](https://huggingface.co/datasets/dronefreak/GC10-DET) | 공개 train 1,840, valid 230, test 230; 80/10/10 | 우리 1,604/343/344 및 annotation 동결과 다름 |

오늘은 원문 공개 페이지까지 확인했다. 외부 ZIP 전체 로그를 재현하거나 외부 mAP를 동일 조건 검증했다고 주장하지 않는다. GPU가 더 빠르다고 반드시 mAP가 높아지는 것은 아니다. 학습 수치·환경 차이와 처리 속도 차이를 구분한다.

## 네 질문에 대한 답

1. **논문에 쓸 수 있는 결과:** C224 440회, F518 162회의 기존 고정 프로토콜 결과와 그 통계; 최근 YOLO26의 dev 튜닝·클래스 오류 자료는 개발 근거·보조 분석으로 구분해서 사용한다. F518의 DINO−Random test nAULC 차이는 GC10 +1.4716%p, NEU +0.2952%p이며 기존 Holm 보정에서 유의한 우위는 확인되지 않았다.
2. **반드시 해결할 문제:** 분할 버전 혼동, NEU 라벨 overlay 누락, YOLO26 Entropy 출력 호환, head·환경·초기화 고정, 반복 단위와 test 노출의 정확한 서술. 준비 단계의 구현 문제는 이번 점검에서 수정·검증했고 정규 캠페인 완주와 최종 평가는 남아 있다.
3. **방어 가능한 설계:** 같은 로컬 캠페인에서 3개의 대응 획득+학습 seed 조합, 6개 예산, 3전략, 각 학습 seed에 대응하는 Full. n=3에서 유의성 확보를 약속하지 않고 효과량·SD·seed별 차이를 보고한다.
4. **작업 순서:** 10/21~22 프로토콜·소스 고정 → 공유 10% 및 NEU Full 우선 → 나머지 예산과 모든 대응 반복 → 설정 변경 없이 최종 평가 → 결과·그림 채움. 실험 동안 기존 논문의 서론·관련 연구·방법·한계와 학회 초안을 병행 작성한다.

실행 순서·시간·통계 정의는 [최종 프로토콜](20261021_final_experiment_protocol.md), 제출 일정은 [논문 작업 계획](20261021_thesis_paper_plan.md)을 따른다.

# 최종 Label Efficiency 실험 프로토콜 — 2026.10.21

## 1. 고정 범위

연구 질문은 동일한 검출기에서 Frozen DINOv2 다양성 선택이 Random·Entropy 대비 같은 라벨 예산의 검출 성능에 어떤 차이를 만드는지, 35% 라벨로 100% 전체 학습 기준의 어느 정도 성능을 확보하는지다. 어느 전략이 높게 나오더라도 프로토콜을 유지한다.

| 항목 | 고정 내용 |
|---|---|
| 데이터 | 기존 F518 고정 분할. GC10 1,604/343/344, NEU 1,258/270/270 |
| AL 예산 | **10·15·20·25·30·35%**, 6지점. 40·50% 추가 없음 |
| 실제 라벨 이미지 수 | GC10 `[160,241,321,401,481,561]`; NEU `[126,189,252,315,377,440]` |
| Full | GC10 train 1,604 / NEU train 1,258. dev/test는 제외 |
| 비교 | Random, Frozen DINOv2 Farthest-first, 기존 정규화 class-score Entropy |
| 검출기 | **YOLO26s**, 공식 pretrained `yolo26s.pt` |
| pretrained SHA | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |
| 학습 | 매 예산 cold-start; 100 epoch; imgsz640; batch8; nbs64; SGD lr0=.01, lrf=.01, momentum=.937, weight_decay=.0005 |
| 스케줄·증강 | cosine, warmup3, close_mosaic10, mosaic1, mixup0, cutmix0, 기존 공통 설정의 나머지 증강·손실 계수 유지 |
| 수치 처리 | AMP=True, 기존 strict_loop+guarded_step, 실제 batch 고정, OOM시 batch 변경·NaN 복구·예산 간 이어 학습 없음 |
| best 선택 | 학습 중 dev mAP50-95 최대, 동률 latest epoch; 실제 dev batch16, rect=True |
| 보고 평가 | FP32, 640, batch8, rect=False, conf=.001, iou=.7, max_det300. P/R는 설치 validator가 smoothed mean F1 최대점에서 반환하는 요약값이며 고정conf=.25 운영점과 구분 |
| YOLO26 head | 설치 버전 8.4.90의 one2one / end2end=True. 최신 라이브러리 기본값으로 대체하지 않음 |
| 주 지표 | mAP50-95의 예산별 곡선과 10~35% nAULC |
| 보조 | mAP50, Precision, Recall, 클래스별 AP50/AP50-95, 35% 도달률 및 절대 차이 |

전체 kwargs는 `research_extensions/final_label_efficiency_20261021_01/config/common_args.json`에 저장했다. CM15가 dev 최고였다는 이유로 사후 최종 선택을 바꾸지 않고, SGD에서 스케줄만 바꾼 CM10을 공통 설정으로 고정한다. 이 선택에 dev 탐색이 사용됐다는 사실을 논문에 밝힌다. NEU 최적화·추가 모델 선정 실험은 하지 않는다.

예산은 train pool의 이미지 비율이다. 실제 35% 값은 GC10 561/1604=34.9751%, NEU 440/1258=34.9762%다. 학습용 이미지 라벨의 약 65%를 덜 사용한다는 설명은 가능하지만, bbox annotation 시간이나 총 연구 라벨 비용을 65% 줄였다고 환산하지 않는다. dev 라벨·사전학습 비용도 별도다.

## 2. 반복 단위와 학습 횟수

| 대응 반복 | Acquisition seed | Learner seed | Full learner seed |
|---|---:|---:|---:|
| 1 | 2026091901 | 2026091900 | 2026091900 |
| 2 | 2026091902 | 2026091908 | 2026091908 |
| 3 | 2026091903 | 2026091909 | 2026091909 |

각 행 안에서 3전략은 초기 10% 이미지·learner seed·초기화 조건을 공유한다. Full은 같은 행의 learner seed로 전체 train을 학습한다. 반복은 **획득 seed와 학습 seed를 함께 바꾼 3개 대응 설정**이다. 두 변동 원인을 분리 추정하는 crossed design이나 3×3 실험이 아니다. 과거 5 acquisition × 고정 learner 또는 10×2 결과와 같은 반복 구조라고 쓰지 않는다.

Random과 DINO는 현재 train pool·seed와 일치하는 기존 R0~R5 선택 manifest를 검증하여 계승한다. 72개 복사본에는 각 원본 경로와 SHA를 남겼다. DINO 특징을 다시 추출하거나 FF를 새로 튜닝하지 않는다. F518은 facebook/dinov2-small, 고정 revision, 518 letterbox·마지막 CLS·L2 특징과 기존 거리/동률 규칙이다.

Entropy는 **각 반복·각 직전 예산의 새 YOLO26 best**로 남은 train pool 이미지만 scoring한다. YOLOv8 Entropy 선택 목록을 재사용하지 않는다. 후보 GT·dev AP·test 값은 acquisition에 사용하지 않는다. one2one의 top-k 이전 raw class sigmoid를 원래 NCS-mean entropy에 연결하며 .25 confidence·stable class NMS .7·300개 상한·빈 후보 0 규칙을 유지한다. 획득 점수 동률은 score 후 ID 오름차순이다.

학습 수 계산:

- AL: `2 datasets × 3 repeats × (shared10% 1 + 3methods × later5) = 96`
- Full: `2 datasets × 3 learner seeds = 6`
- 합계 **102개 정규 셀**, 재사용 GC10 Full seed1900 **1개**, **새 정규 학습 101회**
- 개념적 AL 결과는 108행이다. 공유 10% 결과를 3전략 곡선에 붙이며 이를 3회의 독립 학습으로 세지 않는다. Full 6행을 더한 집계 입력은 114행이다.
- 기술 smoke·재평가·Entropy 추론은 위 학습 수와 별도다.

기존 GC10 35% linear SGD 결과는 새 cosine 조건과 다르므로 재사용 0개다. 같은 정규 설정의 NEU Full은 미확인이다. 5개 대응 반복을 처음부터 고정하면 AL160+Full10=170, 같은 기준 1개 재사용 시 새169회다. 이번 일정에는 **3개를 고정**하며 결과를 본 뒤 유리할 때만 반복을 추가하지 않는다.

물리 셀 목록은 `execution_matrix.csv`, 결정 가능한 설정과 seed는 `config/protocol.json`이 기준이다.

## 3. 결과 계산 규칙

1. 각 예산의 mAP는 3개 대응 반복의 **평균 ± 표본 SD(ddof=1)**로 보고한다. mAP 원자료 0~1, 본문 % 단위, 방법 간 차이는 %p다.
2. nAULC는 각 반복에서 6개 실제 누적 이미지 수를 x축으로 사다리꼴 적분하고 `(N35−N10)`으로 나눈다. 100% 지점은 적분에 넣지 않는다. 반복별 nAULC를 만든 뒤 평균·SD를 계산한다.
3. 주 도달률은 `100 × mean(mAP35) / mean(mAPFull)`이다. 별도로 같은 learner seed에 대응하는 `100 × mAP35_i / mAPFull_i` 3개의 평균±SD도 보고하며 두 통계량을 같은 값으로 부르지 않는다. Full SD와 35% SD, 대응 절대 gap(%p)도 함께 제시한다. 분모가 0이면 비율을 계산하지 않는다.
4. Full은 실측 참조이며 절대 상한이 아니다. 100%를 넘는 도달률도 그대로 기록하고 분산·epoch 정책을 함께 해석한다. 성능 동등성이나 우월성의 증명이 아니다.
5. DINO−Random과 DINO−Entropy의 nAULC 차이를 대응 반복별로 산출한다. 두 데이터셋×두 비교의 4개 primary family에 기존과 같은 Holm 정의를 적용한다. n=3의 양측 exact sign-flip 최소 p는 .25여서 p<.05를 확보할 수 없다. 효과 방향·크기·SD·원자료를 우선 보고하고 유의한 우위라고 쓰지 않는다. 6예산을 독립 n=18로 늘려 세지 않는다.
6. 클래스별 AP는 보조 분석이다. 기존 취약 클래스와 전체 클래스를 모두 제시하고 결과를 보고 클래스·IoU·confidence를 새 primary endpoint로 고르지 않는다.

입력 형식·결측/중복/partition 검사는 `summarize.py`에 구현했다. test 과학 평가만 받으며, 미완료 곡선·공유 R0 불일치·0분모를 거부한다. 오늘 생성한 성능 없는 표 틀에는 0이나 예상 AP를 넣지 않는다.

## 4. 시간·자원

실측 GC10 561장 26s SGD100은 약 35.0분, Full1604 SGD+Cosine100은 약 92.6분이다. 두 점을 이용한 단순 선형 근사와 기존 26n의 NEU/GC10 이미지당 시간 비율로 추산한 **새101회 학습 중앙 추정은 약44.3 GPU시간**, 계획 범위는 **33~66시간**이다. NEU26s 정규100 epoch 시간은 아직 실측하지 않았고 작은 기술 smoke로 대신 추정하지 않았다. Entropy 획득 추론·최종 평가·파일 전송·다른 작업의 GPU 경합·장애 대응은 이 범위에 별도로 더한다. 전체 일정에는 대략 **40~80시간의 연속 GPU 가용 시간**을 확보한다.

`evidence/runtime_estimate.json`에 근거와 추정식을 저장했다. 이 수치는 완료 보장이나 5090 성능 예측이 아니다. 최초 공유10% 및 NEU Full의 정상 완료 시간으로 남은 시간을 다시 산출하되 과학 설정은 변경하지 않는다.

RTX 3070 로컬 실행을 기본으로 하므로 GPU 임대료는 0이며 전기료는 계산하지 않았다. 과거 대화의 $/h를 현재 임대료로 재인용하지 않았다. 원격 요금·가용성은 아직 미확정이다. **원격 GPU로 전환하면 새 환경을 동결하고 해당 환경에서 AL·Full을 함께 실행하는 것이 원칙이며, 이번 1회 재사용을 제외한 102회 전체 예산으로 다시 산정한다.** 빠른 GPU가 정확도를 자동 향상시킨다고 가정하지 않는다.

26s의 best/last 전체상태 checkpoint는 각각 약80.3MB다. 신규101회 두 가중치는 약16.2GB이며 로그·temporary·평가 결과·여유 공간을 포함해 최소25GB를 확보한다. 준비 시 C: 여유는 약57GB였다. 새 학습은 고유 디렉터리와 GPU lock을 사용한다. 이미지/라벨은 검증된 원본의 고유 loader view에 연결하고 cache는 새 출력 안에 쓴다.

## 5. 실행 순서와 준비 상태

1. `audit.py` / `prepare.py`가 수행한 입력·분할·재사용 판정을 확인한다. 실행 소스·전체 kwargs·입력 index·선택 manifest·설치된 Ultralytics 수치 소스를 봉인한다.
2. 두 데이터셋의 128 train/32 dev 1 epoch 기술 점검과 CPU 검증 결과를 확인한다. 이 점검에서 사용한 checkpoint로 정규 학습을 시작하지 않는다.
3. 각 반복의 공유10%를 한 번 학습한다. 우선 NEU Full seed1900을 확보하고, 나머지 Full 및 모든 AL 곡선을 완성한다. 첫 반복 결과가 유리해도 반복 수나 방법을 줄이지 않는다.
4. 각 예산에서 Random/DINO는 계승 manifest로 학습하고, Entropy는 직전 checkpoint→획득→학습 순서로 수행한다. 모든 셀은 원래 pretrained에서 cold-start한다.
5. 전체 모델·설정·best 선택을 마친 뒤 고정된 test 평가 manifest를 별도로 봉인한다. 기존 Full을 포함해 동일한 head·FP32·batch·rect·IoU 조건으로 평가한다. 준비 단계에서는 test 추론을 실행하지 않는다.
6. `summarize.py`로 곡선·nAULC·도달률·대응차이를 집계하고 원시 행과 대조한다. test 결과를 이용해 설정을 다시 고르지 않는다.

단일 셀 명령 예시(PowerShell, workspace 루트):

```powershell
& 'Defect_VLM_Project/.python311/python.exe' -I -B 'research_extensions/final_label_efficiency_20261021_01/cell.py' --dataset GC10-DET --method Random --pair 1 --round 0
& 'Defect_VLM_Project/.python311/python.exe' -I -B 'research_extensions/final_label_efficiency_20261021_01/acquire.py' --dataset GC10-DET --pair 1 --round 1
& 'Defect_VLM_Project/.python311/python.exe' -I -B 'research_extensions/final_label_efficiency_20261021_01/cell.py' --dataset GC10-DET --method Entropy --pair 1 --round 1
```

round0은 어느 AL 방법을 지정해도 공유 Random 셀 이름으로 통일하여 중복 학습을 막는다. 정규 셀 덮어쓰기와 기존 GC10 Full seed1900의 재학습은 거부한다. 준비 보고 이후 사용자 지시에 따라 같은 날20:57 KST에 `campaign_controller.py`로 순차101회 실행을 착수했다. NEU Full과 여섯 공유10% 모델을 먼저 수행하며 공유 모델별 Entropy 검증을 완료한 뒤 후속 예산을 진행한다. 과학적 설정은 이 문서대로 유지한다.

**준비 판정:** 로컬 train/dev 실행용 셀 러너·입력·Entropy adapter·집계 처리를 준비하고 기술 점검을 통과했다. 정규100 epoch 전체 행렬 완주, 원격 환경 검증, 최종test 평가 실행은 미완료다. 최종test는 과거 연구에서 열람한 고정holdout이며 완전히 새롭게 확보한 미관찰test라고 서술하지 않는다.

# Distinct Active Learning Baseline 사전 설계 — 2026-09-07

## 오늘의 결정

Random과 DINOv2 표현 기반 farthest-first 외의 단일 비교 방법으로 **image-level Minmax least-confidence**를 추천안으로 고정한다. 이는 Roy et al.의 Minmax를 YOLO의 post-NMS confidence에 적용한 사전 고정 adaptation이며 논문의 전체 방법을 그대로 재현한다는 뜻은 아니다. 현재 방법과 기하적으로 중복되는 Core-set/k-center는 제외한다. 오늘은 문헌·점수식·연산량만 확정했으며 코드 구현, detector inference, 학습, 잠긴 test 평가는 수행하지 않았다.

## 후보 비교

| 후보 | Image-level score 정의 | No-detection 이미지 처리 | 기존 코드 재사용 | 현재 방법과 차별성 | 구현 위험 | 추천 |
|---|---|---|---|---|---|---|
| **Minmax least-confidence** | `U(x)=1-max_{d∈D(x)} q_d` | `max ∅=0`, 따라서 `U=1` | 기존 학습/eval runner와 YOLO predict는 재사용 가능; current selector 구현은 없음 | 표현 공간의 다양성이 아니라 현재 detector의 불확실성 | 낮음~중간. 다중 객체에서 한 고신뢰 box가 다른 불확실성을 가릴 수 있음 | **채택** |
| Basic detection entropy | `H(x)=Σ_d[-q_d log q_d-(1-q_d)log(1-q_d)]` | 빈 합 `0` | 같은 confidence 추론 재사용 | detector confidence 기반 | 중간. 검출 수가 많은 이미지에 점수가 커지는 혼입과 빈 이미지 처리 문제가 있음 | 예비안만 유지 |
| 분류·위치 확률 불확실성 | probabilistic head의 class·box aleatoric/epistemic uncertainty를 객체·이미지 단위로 집계 | 별도 규칙 필요 | 현재 learner를 그대로 재사용하기 어려움 | localization까지 직접 반영 | 높음. head/loss 및 학습 정의가 바뀌어 별도 방법론 연구가 됨 | 이번 범위 제외 |

근거 문헌은 object detection의 Minmax를 정의한 [Roy et al., BMVC 2018](https://bmva-archive.org.uk/bmvc/2018/contents/papers/0287.pdf), 단일단계 detector에서 `1-confidence`를 사용한 [Yamani et al., WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Yamani_Active_Learning_for_Single-Stage_Object_Detection_in_UAV_Images_WACV_2024_paper.pdf), 기본 detection entropy를 정의한 [Wu et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Entropy-Based_Active_Learning_for_Object_Detection_With_Progressive_Diversity_Constraint_CVPR_2022_paper.pdf), probabilistic localization/classification uncertainty를 다룬 [Choi et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Choi_Active_Learning_for_Deep_Object_Detection_via_Probabilistic_Modeling_ICCV_2021_paper.pdf)이다. 전체 논문 방법을 가져왔다고 주장하지 않고, 재현 가능한 단순 기준선의 정의 근거로만 사용한다.

과거 repository의 복합 detector-uncertainty 실험은 current protocol의 기준선 구현이 아니며 primary gate 실패 기록이 있다. 그 결과를 되살리거나 성공 근거로 재사용하지 않고, 위 단순 규칙을 별도 arm으로 새로 사전 고정한다.

## 추천 baseline의 정확한 규칙

### 점수

남은 train pool의 이미지 `x`에 대해 직전 round의 policy detector가 post-NMS로 출력한 최종 box confidence 집합을 `D(x)`라 한다.

`U(x) = 1 - max_{d∈D(x)} q_d`

- `q_d`는 Ultralytics가 최종 출력한 `boxes.conf`이다. 내부 objectness와 class probability를 새로 분해하거나 재조합하지 않는다.
- 검출이 없으면 `max ∅=0`으로 정의해 `U(x)=1`로 둔다. 이는 Roy et al.에 직접 명시된 규칙이 아니라 본 연구가 결과 전에 고정하는 확장 규칙이다.
- `U`가 큰 이미지부터 query count만큼 선택한다.
- 점수가 정확히 같으면 canonical sample ID 오름차순으로 결정한다.
- Class quota, confidence calibration, entropy, diversity 재정렬, threshold tuning은 추가하지 않는다.

### Policy detector와 round 순서

1. 각 dataset·acquisition seed의 기존 shared R0 learner slot A `best.pt`로 남은 pool을 추론해 R1을 선택한다.
2. 선택된 R1 집합으로 learner A/B를 각각 current protocol과 동일하게 cold-start 학습한다.
3. R1 learner A `best.pt`로 R2를 선택하고 같은 방식으로 R5까지 진행한다.
4. 선택 집합은 learner A/B에 공용이며, slot A만 다음 acquisition의 policy detector로 사용한다.
5. 이 설계는 단일 policy seed에 조건부인 detector-driven trajectory라는 한계가 있다. 이를 숨기지 않으며 결과 후 A/B ensemble이나 다른 seed로 바꾸지 않는다.

### Detector inference 고정값

- Ultralytics 8.4.90, `imgsz=640`, `conf=0.001`, `iou=0.7`, `max_det=300`
- Deterministic inference, 같은 이미지 전처리와 canonical sample ID 순서
- 직전 round의 validation-selected `best.pt`만 사용
- Acquisition scorer는 남은 pool의 GT class/bbox와 잠긴 test 데이터·metric을 보지 않는다. Development label/metric은 current lifecycle의 사전 고정 규칙에 따라 직전 round `best.pt`를 고르는 용도로 쓰이므로 acquisition에 간접 영향을 준다. 다만 baseline score·threshold·집계 규칙의 결과 기반 tuning에는 사용하지 않는다.
- Query count: GC10-DET round당 80장, NEU-DET round당 63장

`conf=0.001`은 current detector 평가 설정에서 가져와 사전 고정하는 최소 출력 floor이며 결과를 보고 조정하지 않는다.

## 실제 training cell 계산

### 공유 가능한 R0

기존 R0는 strategy 간 selected set과 config가 동일하다.

`2 datasets × 10 acquisition seeds × 2 learner seeds × 1 round = 40 conceptual cells`

이 40개 checkpoint는 이미 존재하므로 **새 training cells는 0**이다. 특히 R1 acquisition에는 각 trajectory의 기존 R0 slot A checkpoint를 사용한다.

### 새 acquisition arm의 R1~R5

`2 datasets × 10 acquisition seeds × 2 learner seeds × 5 rounds = 200 new training cells`

따라서 새 arm 전체는 R0를 포함하면 240 conceptual cells이지만, 실제 추가 YOLO 학습은 **200 cells**가 맞다. 기존 Response Matrix의 계산에는 correction이 필요 없다.

## 계산량 분리

### Acquisition inference

| Dataset | Round별 남은 pool 이미지 | Acquisition seeds | 총 image inferences |
|---|---|---:|---:|
| GC10-DET | 1,444 + 1,364 + 1,284 + 1,204 + 1,124 = 6,420 | 10 | 64,200 |
| NEU-DET | 1,133 + 1,070 + 1,007 + 944 + 881 = 5,035 | 10 | 50,350 |
| **합계** | 100 checkpoint-pool passes |  | **114,550** |

Slot A checkpoint 한 개만 각 round에 사용한 계산이다. A/B ensemble을 쓰면 inference가 두 배가 되므로 결과 후 변경하지 않는다.

### YOLO training

- 새 cells: 200
- 최대 epoch 합: `200 × 100 = 20,000 cell-epochs`
- GC10-DET: `(241+321+401+481+561) × 10 × 2 × 100 = 4,010,000 image-epochs`
- NEU-DET: `(189+252+315+378+441) × 10 × 2 × 100 = 3,150,000 image-epochs`
- 합계: **7,160,000 maximum image-epochs**

기존 동일 round의 완료 시간 중앙값을 단순 합산하면 순차 단일 GPU에서 약 32 GPU-hours이며, 여기에 114,550장의 acquisition inference, validation, artifact audit 시간이 추가된다. 실제 시간은 첫 cell로 갱신하되 cells, seeds, rounds는 바꾸지 않는다.

### 잠긴 test 평가

- GC10-DET: `10 acquisition × 2 learner × 5 rounds × 458 = 45,800 image-evaluations`
- NEU-DET: `10 acquisition × 2 learner × 5 rounds × 359 = 35,900 image-evaluations`
- 합계: **81,700 locked-test image-evaluations**이며, 200개 새 모델을 각각 한 번 평가한다.

잠긴 test는 200/200 training과 integrity audit가 완료되고 별도 authorization을 통과한 뒤에만 평가한다.

## 비교와 주장 범위

- 같은 R0, 같은 query count, 같은 data split, 같은 YOLO lifecycle, 같은 learner seeds를 유지한다.
- Primary 비교는 Random, DINOv2 표현 기반 farthest-first, Minmax least-confidence의 R0~R5 nAULC로 한다.
- Least-confidence가 더 좋거나 나쁠 것이라는 방향 가설로 hyperparameter를 조정하지 않는다.
- 이 기준선은 localization uncertainty를 직접 측정하지 않고, 한 고신뢰 box가 다중 객체 이미지의 불확실성을 가릴 수 있다.
- 단순성은 현재 방법에 detector-dependent uncertainty라는 다른 축을 추가하는 통제 기준선의 장점이지, 새로운 방법론 기여가 아니다.

## 실행 진입 조건

1. 지도교수가 이 한 가지 기준선과 200-cell 범위를 확인한다.
2. 위 점수, empty rule, policy checkpoint, inference 인자, tie-break, seeds, query count를 별도 protocol/manifest에 봉인한다.
3. 기존 production code를 직접 수정하지 않고 별도 arm runner를 만들고 selection-only dry-run으로 manifest와 GT 비개입을 감사한다.
4. 기존 잠긴 test protocol 수정이 필요하거나 GT/사람 정보가 acquisition에 들어간 흔적이 발견되면 STOP한다.
5. 승인과 dry-run PASS 전에는 학습을 시작하지 않는다.

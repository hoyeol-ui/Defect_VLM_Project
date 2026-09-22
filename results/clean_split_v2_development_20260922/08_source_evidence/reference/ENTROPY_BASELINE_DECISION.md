# Entropy baseline 재검토와 최종 설계 정의

**결정: OPTION B를 채택한다.** 최종 방법명은 **Entropy — mean normalized class-score entropy**이며 수식 ID는 `ENTROPY_NCS_MEAN_V2`다. 기존 binary confidence SUM은 과거 근거로만 남기고 새 scientific arm으로 실행하지 않는다. 새 baseline의 학습 성능은 아직 확인하지 않았고, 여러 수식을 실행해 좋은 것을 고른 결정이 아니다.

## 1. 기존 정의의 한계

기존 코드는 post-NMS `boxes.conf` 하나에 `h(p)=−p ln p−(1−p)ln(1−p)`를 적용해 이미지 내 SUM을 사용했다. 기존 100개 query 기록의 score–detection-count Spearman 평균은 GC10 .918631, NEU .943865였다. 이 수치만으로 SUM이 틀렸다고 단정하지 않지만, 이미지 단위 비교에서 uncertainty와 예측 개수 효과를 분리하기 어렵다. max confidence 하나로는 같은 최고 점수에서 클래스 간 혼동 정도도 구분할 수 없다.

수치 근거는 기존 `Defect_VLM_Project_development_release/results/development_20260913/tables/entropy_query_diagnostics.csv`의 dataset별 50행이다. 이번 검증기는 이 저장된 query별 상관의 평균을 재집계했다. 원 이미지에 새 scoring을 수행하거나 원 score–count 상관 자체를 다시 추정한 것은 아니다. 원 CSV의 SHA는 DESIGN_MANIFEST_v2.json에 기록한다.

[Learning Loss, CVPR 2019 §4.2](https://dgyoo.github.io/papers/cvpr19.pdf)는 detection box의 softmax entropy를 평균하는 비교군을 사용했다. [Brust et al., Active Learning for Deep Object Detection §3.1](https://arxiv.org/html/1809.09875)는 SUM/average/max와 빈 검출을 비교하며 average의 개수 배수 효과 완화를 설명한다. 후자는 margin 기반 점수이므로 그 논문이 우리 entropy 수식을 사용했다고 말하지 않는다.

## 2. 후보 비교

| 후보 | YOLOv8에서 가능한가 | 장점 | 한계·결정 |
|---|---|---|---|
| A. binary confidence SUM | 기존 코드로 가능 | 단순하며 과거와 같은 정의 | 최고 confidence만 사용하고 검출 개수에 직접 비례. 최종 arm에서는 제외 |
| binary confidence MEAN | 최종 boxes.conf만으로 가능 | SUM의 직접 개수 배수 제거 | class-vector 혼동 정보는 여전히 없음. 제외 |
| 모든 sigmoid의 Bernoulli entropy 평균 | raw C차원 score로 가능 | native sigmoid/BCE 형태 유지 | 다수의 거의 0인 클래스와 background 성격에 좌우되며 categorical entropy와 다름. 제외 |
| native box softmax entropy MEAN | SSD처럼 softmax head인 경우 직접 가능 | 문헌의 명확한 비교군 | YOLOv8 출력은 native softmax가 아니므로 그대로라고 부를 수 없음 |
| **정규화 class-score entropy MEAN** | **raw C차원 sigmoid score와 NMS anchor index 보존 필요** | 클래스 간 상대적 혼동을 사용하고 직접적인 개수 배수 효과를 제거 | **최종 채택. YOLO에 맞춘 명시적 변환이며 calibrated posterior가 아님** |
| entropy MAX | 동일 score에서 가능 | 한 불확실 box를 강조 | outlier 한 개에 의존하고 다른 box 정보 손실. 제외 |
| pre-NMS 모든 anchor entropy | 가능 | 필터에서 버린 예측도 포함 | background·중복 anchor가 압도. 제외 |
| localization variance·MC dropout·ensemble | 추가 모델/반복 inference/학습 변경 필요 | 더 넓은 uncertainty 표현 | 별도 연구와 비용 증가. 이번 한 arm에는 포함하지 않음 |

## 3. frozen 수식

각 이미지에서 최종 NMS에 남은 box 집합을 B라 한다. 해당 anchor의 **원래 C개 sigmoid class score**를 s_bc로 보존한다. GC10 C=10, NEU C=6이다.

`q_bc = s_bc / sum(k=1..C, s_bk)`

`h_b = −sum(c=1..C, q_bc * ln(q_bc)) / ln(C)`

`U(image) = mean(b in B, h_b)`; B가 비면 **0**.

s는 sigmoid 이후이며 softmax를 sigmoid 값에 다시 적용하지 않는다. pre-sigmoid logits에 softmax를 적용한 별도 수식도 사용하지 않는다. q는 foreground class score의 정규화이며 모델이 학습한 native categorical posterior가 아니다. background class를 임의로 추가하지 않는다. q=0 항은 0·ln0=0으로 처리한다. 유한하고 0≤s≤1인지 검사한다. 통과 box는 max(s)>.25이므로 분모는 0이 될 수 없다. 분모 오류를 임의 epsilon이나 uniform vector로 덮지 않고 실패로 처리한다.

집계는 FP64 자연로그와 안정적인 합산으로 수행한다. 0≤h,U≤1을 검사한다. 동일 box 집합을 반복 복제해도 MEAN은 같지만, 서로 다른 box를 추가하면 평균이 달라질 수 있다. 따라서 detection-count 상관이 반드시 0이 된다고 주장하지 않는다.

이 변환은 단일 클래스 우세와 클래스 혼동을 비교할 수 있는, 계산 가능한 predictive classification uncertainty **대리값**이다. 객체 존재 확률의 calibration, 놓친 객체, localization uncertainty, epistemic uncertainty를 측정한다고 주장하지 않는다. 성능이 낮아도 A나 MAX로 교체하지 않는다.

## 4. 실제 YOLOv8 출력과 구현 가능성

로컬 Ultralytics **8.4.90** 소스를 읽었다. `nn/modules/head.py:173–184`의 Detect._inference는 decoded box와 `scores.sigmoid()`를 결합한다. 별도 objectness 채널을 곱하는 옛 YOLO 공식을 적용하지 않는다. `utils/nms.py`는 원 anchor index를 `return_idxs=True`로 추적할 수 있지만 기본 출력은 class vector를 버리고 box/conf/class만 남긴다. 따라서 `Results.boxes.conf`만 저장한 기존 파일로 새 entropy를 복원할 수 없다. [공식 해당 버전 소스](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/nn/modules/head.py).

향후 adapter는 같은 한 번의 forward에서 raw score tensor를 clone해 보존하고 NMS가 반환한 원 index로 C차원 score를 모은다. IoU나 좌표 근접도로 사후 매칭하지 않는다. source shape는 표준 640 YOLOv8n에서 B×(4+C)×8400을 예상하되 실제 gate에서 확인한다. end-to-end/다른 head/추가 output이면 조용히 대체하지 않고 실패시킨다.

## 5. scoring과 선택 명세

- checkpoint: 동일 dataset·acquisition seed의 이전 **Entropy 경로 dev-best**. R1만 새 shared R0 best를 사용한다.
- imgsz640, rect=False인 square letterbox, FP32 forward, batch1, AMP/TF32/augmentation off, 전체 미선택 pool 처리.
- confidence **max class score >.25**, class-aware NMS, IoU .70, max_det300, multi_label=False, classes=None, GT autolabel 없음.
- NMS는 목표 환경에서 CPU FP32의 사전 고정 backend와 순서로 수행한다. 동일 confidence는 원 anchor index 오름차순, 동일 class argmax는 작은 class index, IoU>.70이면 suppress하는 tie 규칙을 adapter 명세에 넣는다. raw index 계보와 비동점 사례의 pinned native NMS 일치를 검증한다.
- max_nms=30000, NMS timeout이나 scoring 누락은 실패. 300개 cap에 도달한 이미지는 로그에 기록하며 threshold/cap을 사후 변경하지 않는다.
- U 내림차순, 이미지 점수가 정확히 같으면 opaque ID 오름차순. 분모·range·finite 오류는 후보를 빼고 계속하지 않고 해당 scoring job을 실패 처리한다.
- no detection=0. 그런 이미지를 자동으로 가장 불확실하게 올리지 않는다. 그 비율이 높으면 한계로 보고하고 confidence를 결과에 맞춰 낮추지 않는다.

## 6. 기술 gate와 보고

**아직 scoring/selection/inference를 실행하지 않았다.** 본학습 전 adapter smoke에는 새 v2 기술 smoke에서 만든 dataset별 1 epoch checkpoint를 사용한다. 이 checkpoint는 scientific R0나 실제 acquisition parent로 사용하지 않는다. 따라서 본학습 전 검증이 미래 scientific R0의 존재를 요구하지 않는다. 실제 main에서는 매 round의 정식 parent로 아래 계약을 다시 확인한다.

1. raw vector ↔ NMS index ↔ 최종 conf/class/box가 일대일이며 retained conf=max(s), class=argmax(s).
2. 재현성과 tie, threshold 경계, no detection, 한 box, 여러 box, 모든 클래스, duplicate box, range·finite·NMS timeout 처리를 synthetic fixture로 확인.
3. 고정 train 이미지의 forward 한 번으로 계산했고 전체 pool coverage와 SHA가 완전한지 확인. GT에 접근하지 않음.
4. 각 round의 score–검출 수 Spearman, score quantile, zero detection 비율, max_det 도달 비율, class score 합 분포, 선택 구성은 선택 봉인 뒤 기록.
5. 이전 binary SUM과 수치가 다르다는 것은 예상된 새 정의의 결과다. 과거 Entropy 결과와 같은 arm으로 합치지 않음.

이 baseline은 문헌의 box class entropy 평균이라는 구조를 참고해 YOLOv8 sigmoid 출력에 맞게 정의한 것이다. 문헌과 완전히 같은 구현, 보편적인 최선의 uncertainty, 최적의 threshold라고 표현하지 않는다. 이 한계는 baseline을 모호하게 두는 이유가 아니라 명확한 비교 범위를 정하는 이유다.

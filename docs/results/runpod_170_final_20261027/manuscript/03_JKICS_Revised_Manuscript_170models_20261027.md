# 금속 표면 결함 객체탐지에서 DINOv2 기반 표본 선택의 라벨 효율과 클래스별 성능 분석

Label Efficiency and Class Specific Performance of DINOv2 Based Sample Selection for Metallic Surface Defect Detection

요    약

본 논문은 금속 표면 결함 객체탐지에서 고정 DINOv2 특징 기반 표본 선택의 라벨 효율과 클래스별 성능을 분석한다. GC10-DET와 NEU-DET에서 Random, DINOv2 farthest-first, Entropy를 동일한 YOLO26s의 200epoch 학습과 약 10~35% 이미지 예산으로 비교하였다. 획득·학습 시드 5쌍을 대응시켜 170개 모델을 학습하고 고정 Test로 평가하였다. DINOv2와 Random의 평균 Test nAULC 차이는 각각 +1.27%p와 −0.17%p였으며, Holm 보정 p값은 0.50과 1.00이었다. GC10에서는 35% 예산의 평균 성능이 높았으나 inclusion 등 일부 클래스는 낮아졌다. NEU에서는 정밀도 상승과 재현율 하락이 함께 나타났고, 최저 클래스 nAULC도 0.94%p 낮았다. 결과는 특징 기반 선택의 효과가 데이터셋·예산·클래스에 따라 달라짐을 보여주며, 지정한 검정에서 통계적 우위나 방법 간 동등성을 확립하지 않는다.

ABSTRACT

This study evaluates the label efficiency and class specific effects of sample selection using frozen DINOv2 features for metallic surface defect detection. Random, DINOv2 farthest-first, and entropy sampling are compared on GC10-DET and NEU-DET using YOLO26s trained for 200 epochs at six image budgets from approximately 10% to 35%. Five acquisition and learner seed pairs yield 170 trained models evaluated on fixed test splits. Mean differences in normalized area under the test learning curve (nAULC) between DINOv2 and random sampling are +1.27 and -0.17 percentage points, with Holm-adjusted p-values of 0.50 and 1.00. At the 35% budget, DINOv2 yields higher mean overall performance on GC10 while reducing AP for some classes. On NEU, higher precision accompanies lower recall, and the minimum-class normalized area is lower by 0.94 percentage points. The findings show dataset, budget, and class dependence without establishing statistical superiority or equivalence. Their scope is limited by fixed splits, the combined seed design, and documented historical data exposure.

키워드: 능동학습, 결함 객체탐지, DINOv2, 표본 선택, 라벨 효율

Key Words: Active learning, Defect detection, DINOv2, Sample selection, Label efficiency

## Ⅰ. 서론

금속 표면의 결함 검사는 제품의 상태를 확인하는 작업이다. 객체탐지 모델은 영상에서 결함의 종류와 위치를 함께 예측하지만, 학습 이미지마다 클래스와 경계 상자 주석이 필요하다. 공개 결함 자료인 GC10-DET[1]와 NEU 계열 자료[2]를 활용하면 제한된 학습 이미지에서 표본 선택의 영향을 비교할 수 있다. 이때 중요한 문제는 높은 탐지 성능을 얻는 것과 함께, 같은 수의 이미지를 사용할 때 어떤 표본을 먼저 선택할 것인지 정하는 데 있다.

풀 기반 능동학습(active learning)은 아직 선택하지 않은 이미지 중 일부를 추가로 주석하고 모델을 다시 학습하는 절차다. 이미지의 대표성이나 예측 불확실성을 선택 기준으로 사용할 수 있으나, 선택 점수가 높다는 사실만으로 학습 후 탐지 성능의 향상이 보장되지는 않는다. 학습 이미지 수와 성능의 관계를 분석한 기존 연구[3]처럼, 선택 전략도 여러 라벨 예산에서 비교할 필요가 있다. 또한 전체 평균이 높아져도 특정 결함의 성능은 낮아질 수 있으므로 평균과 클래스별 결과를 함께 살펴보아야 한다.

본 연구는 고정된 DINOv2 특징을 이용한 farthest-first 선택(DINOv2-FF)을 무작위 선택(Random) 및 클래스 점수의 엔트로피를 이용한 선택(Entropy)과 비교한다. 두 자료에서 YOLO26s[9]와 학습 조건을 고정하고, 획득 시드와 학습 시드를 묶은 다섯 대응 반복을 구성한다. 동일한 초기 라벨 집합과 학습 시드를 방법 간에 공유하여 예산별 효과와 반복 변동을 함께 분석한다.

연구 질문은 두 가지다. 첫째, 같은 검출기와 라벨 예산에서 DINOv2-FF의 학습곡선과 반복별 효과가 Random 및 Entropy와 어떻게 다른가. 둘째, 전체 평균의 차이가 정밀도·재현율과 개별 결함 클래스에서도 같은 방향으로 나타나는가. 본 연구는 예산별 탐지 성능, 대응 nAULC 차이, 클래스별 AP와 선택 이미지 구성을 연결해 이 질문에 답한다. 새로운 검출기나 선택 알고리즘을 제안하는 대신, 두 결함 자료에서 관찰한 선택 효과와 그 적용 범위를 구체적으로 제시한다.

## Ⅱ. 관련 연구

### 2.1 산업 영상과 학습 데이터의 구성

공개 결함 자료를 이용한 실험에서는 검출기뿐 아니라 학습에 포함할 이미지의 구성이 결과에 영향을 준다. Lee와 Kang[3]은 YOLO의 학습 이미지 수에 따른 탐지 성능을 분석하였다. 본 연구도 한 예산의 점수만으로 라벨 효율을 판단하지 않고, 같은 누적 이미지 수에서 표본 선택 전략의 학습곡선을 비교한다.

Lv 등[4]은 표면 결함 탐지에서 불확실성 기반 후보 선택과 클래스별 표본 규모 조절을 결합한 능동학습을 제시하였다. 본 연구는 새로운 결함 탐지기나 클래스별 할당 알고리즘을 제안하지 않는다. 고정된 범용 특징의 기하학적 선택과 두 기준선을 동일 조건에서 평가하고, 평균 성능과 결함별 결과의 관계를 살펴본다.

### 2.2 특징 기반 선택과 능동학습 평가

Sener와 Savarese[5]는 능동학습을 특징 공간의 core-set 선택 문제로 다루었다. 본 연구의 farthest-first는 현재 라벨 집합과 거리가 먼 표본을 순차적으로 추가하는 기하학적 선택이다. 원 연구의 특징 공간과 이론적 가정이 본 실험과 다르므로 그 보장을 DINOv2 특징과 객체탐지 AP에 그대로 적용하지 않는다. DINOv2[6]의 범용 특징은 선택 표현으로 사용하며 검출기의 백본을 대체하지 않는다.

Munjal 등[7]은 학습 무작위성과 실험 조건에 따라 능동학습과 Random의 상대 결과가 달라질 수 있음을 보였다. Probst 등[8]은 객체탐지에서 박스 점수를 이미지 점수로 집계하는 방식의 중요성을 보고하였다. 이에 본 연구는 반복 단위, 초기화, Entropy의 원래 클래스 점수와 집계 규칙을 명시하고, 전체 지표와 클래스별 지표를 구분한다.

## Ⅲ. 연구 방법

### 3.1 데이터 분할과 라벨 예산

분석에는 GC10-DET의 10개 클래스와 NEU-DET의 6개 클래스를 사용하였다. 원 아카이브에서 이미지·주석 대응과 RGB 내용, 중복 주석을 감사하여 정제한 후, 중복 위험 관계의 연결성분이 분할을 넘지 않도록 약 70:15:15의 Train·Dev·Test를 고정하였다. 연결성분은 실제 생산 lot을 뜻하지 않는다. 정제 후 이미지는 각각 2,291장과 1,798장이며 실제 입력 수는 표 1과 같다. 분할 시드는 GC10 2026102306, NEU 2026102307이다.

표 1. 고정 데이터 분할

Table 1. Fixed dataset splits

| Dataset | Classes | Train | Dev | Test |
| --- | --- | --- | --- | --- |
| GC10-DET | 10 | 1,604 | 344 | 343 |
| NEU-DET | 6 | 1,258 | 270 | 270 |

학습 풀은 Train으로 한정하였다. 이미지 한 장을 선택하면 그 이미지에 포함된 모든 결함 주석을 학습에 사용한다. 누적 라벨 예산은 학습 풀의 약 10%, 15%, 20%, 25%, 30%, 35%로 두었다. 실제 누적 수는 GC10에서 160, 241, 321, 401, 481, 561장, NEU에서 126, 189, 252, 315, 377, 440장이다. 초기 10%를 R0, 이후 다섯 단계를 R1~R5로 구분하며 이미 선택한 이미지는 계속 유지한다.

획득 시드 2026102401~2026102405와 학습 시드 2026102501~2026102505를 일대일로 묶은 다섯 반복을 사용하였다. 같은 쌍에서는 세 방법이 R0와 학습 시드를 공유하며, 예산이 바뀌어도 같은 공개 가중치에서 새로 학습한다. 반복은 획득과 학습 무작위성의 결합 단위다. 두 요인의 분산을 따로 추정하는 교차 설계가 아니며, 결과에 따라 시드나 초기 집합을 다시 고르지 않았다.

### 3.2 표본 선택 방법

Random은 정렬된 미선택 이미지 목록에서 필요한 수를 동일 확률로 비복원 추출한다. 데이터셋·획득 시드·라운드·목적을 구분한 PCG64 규칙을 고정하였다. DINOv2-FF는 facebook/dinov2-small의 고정 revision(ed25f3a31f01632728cabb09d1542f84ab7b0056)에서 마지막 CLS 토큰을 추출한 384차원 단위 벡터를 사용한다. 입력의 긴 변을 518로 맞추고 비율을 유지한 채 RGB(124,116,104)로 중앙 패딩하며 자르지 않는다. 특징 추출기는 갱신하지 않는다.

학습 풀을 U, 현재 라벨 집합을 L, 정규화 특징을 z(x)라 하자. 후보와 L의 거리는 δ(x,L)=min{1−z(x)ᵀz(l): l∈L}로 계산한다. 식 (1)로 하나의 후보를 추가한 뒤 남은 후보의 최근접 거리를 갱신한다. 단위 벡터에서 이 거리는 L2 거리와 같은 순서를 주지만, 실제 비교값과 동점 규칙은 1−내적에 맞춰 고정하였다. float32 저장 특징을 float64로 비교하고 거리 차이가 10⁻¹² 이내이면 이미지 식별자 순서를 사용한다.

$$
x^*=\arg\max_{x\in U\setminus L}\delta(x,L)
$$

Entropy는 직전 라운드의 해당 반복 모델로 미선택 이미지를 추론한다. YOLO26s의 one-to-one 헤드에서 최종 top-k 이전의 8,400개 후보와 C차원 sigmoid 클래스 점수 벡터를 보존한다. 최대 클래스 점수가 0.25를 넘는 후보에 CPU 클래스별 NMS(IoU 0.70, 최대 300개)를 적용하고, 남은 원래 점수 벡터를 성분의 합으로 나누어 q를 구한다. 식 (2)의 정규화 엔트로피를 박스별로 계산한 뒤 이미지 안에서 산술평균한다. 검출이 없으면 0, 정확한 동점이면 이미지 식별자 순서를 사용한다.

$$
h(b)=-\frac{\sum_{c=1}^C q_{bc}\log q_{bc}}{\log C}
$$

이 점수는 클래스 점수의 상대적인 분산을 나타내는 대리값이며 보정된 확률이나 위치 불확실성 자체가 아니다. 획득용 NMS는 이 점수의 정의에 포함되지만, 최종 성능은 one-to-one 헤드의 NMS 없는 end-to-end 경로로 평가한다. 획득에 직전 모델을 사용하는 것과 다음 예산 모델의 초기화는 별개이며, 다음 모델도 공개 가중치에서 새로 학습한다.

### 3.3 검출기 학습과 최종평가

주실험은 Ultralytics 8.4.90의 YOLO26s[9]를 사용하였다. 모든 방법과 예산은 동일한 yolo26s.pt에서 시작하며 이전 라운드 가중치를 이어받지 않는다. 학습은 200epoch로 고정하고 AMP와 TF32를 끈 FP32, SGD와 cosine schedule을 사용하였다. Dev mAP50-95가 최대인 best checkpoint를 선택하며 동률이면 마지막 해당 epoch를 사용한다. 최종 Test 점수로 모델이나 설정을 선택하지 않았다. 주요 조건은 표 2와 같다.

표 2. 주실험의 학습 및 평가 조건

Table 2. Training and evaluation settings of the main experiment

| Item | Setting |
| --- | --- |
| Detector / initialization | YOLO26s / same pretrained weights; cold start |
| Epochs / patience | 200 / 200 |
| Image / train batch | 640 × 640 / 32 |
| Optimizer | SGD; lr0=0.01; lrf=0.01 |
| Momentum / decay | 0.937 / 0.0005 |
| Schedule / warmup | Cosine / 3 epochs |
| Nominal batch / mosaic | 64 / close in last 10 epochs |
| Precision / TF32 | FP32; AMP off / TF32 off |
| Training device | RTX PRO 4500 Blackwell (32 GB) |
| Training software | Python 3.11.9; Torch 2.12.1+cu129 |
| Validation batch | 64 during training; 8 for final Test |
| Evaluation device | RTX3070; Torch 2.12.1+cu126 |
| Test settings | 640; FP32; rect=False; conf=0.001; max_det=300 |
| Detection head | one-to-one; end2end=True; NMS-free Test |
| Acquisition NMS | conf>0.25; class-wise IoU 0.70; max300 |
| Ultralytics | 8.4.90 in both stages |

공통 R0는 반복마다 한 번만 학습하였다. 데이터셋당 R0 5개, 각 방법의 R1~R5 모델 25개씩과 Full 5개로 85개, 두 자료를 합쳐 170개 모델이다. Full은 같은 학습 시드의 전체 Train 참조다. 200epoch와 best/last 파일 해시를 확인한 후 고정 Test로 각 best 모델을 한 번씩 평가하였다. 공통 R0를 세 방법에 연결하면 논리 결과는 190행이다. 서버 학습과 로컬 평가의 GPU·CUDA 환경은 달라 비트 단위 동일성을 주장하지 않는다.

### 3.4 지표와 통계 분석

주 검출 지표는 IoU 0.50~0.95를 0.05 간격으로 바꾸어 계산한 클래스별 AP의 평균인 mAP50-95다. mAP50, 정밀도(precision, P)와 재현율(recall, R)을 보조 지표로 제시한다. 식 (3)에서 TP, FP와 FN은 각각 정탐, 오탐과 미검출을 뜻한다.

$$
P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN}
$$

저장된 P/R는 IoU 0.50에서 평활한 클래스 평균 F1이 최대인 confidence의 클래스 평균값이다. 평가 입력 conf=0.001은 후보 하한이며 P/R를 보고한 공통 운영 임계값이 아니다. 모델별 요약 평가점이 다를 수 있으므로 이 값만으로 같은 운영 조건의 오탐·누락 위험을 비교하지 않는다. 평균과 표본 SD(ddof=1)는 다섯 획득·학습 시드 쌍으로 계산한다.

각 반복의 누적 이미지 수를 n₀,…,n₅, 해당 mAP50-95를 A₀,…,A₅라 하면 정규화 학습곡선 아래 면적(nAULC)을 식 (4)로 계산한다. 실제 이미지 수 간격을 사용해 사다리꼴 적분하고 관측 예산 구간으로 나눈다. Full 참조는 적분에 포함하지 않는다.

$$
\mathrm{nAULC}=\frac{\sum_{r=0}^4(n_{r+1}-n_r)(A_{r+1}+A_r)}{2(n_5-n_0)}
$$

반복별 nAULC를 먼저 계산한 뒤 같은 시드 쌍의 대응 차이를 구하였다. 두 데이터셋의 DINOv2-FF−Random과 DINOv2-FF−Entropy, 총 네 사전 지정 비교에 대응 차이의 부호 교환 가능성을 가정한 평균 차이 통계량의 양측 exact sign-flip과 Holm 보정[10]을 적용하였다. 5개 차이의 32개 부호 조합을 모두 열거하므로 최소 양측 p값은 0.0625이며 보정 전에도 0.05 미만에 도달할 수 없다. 효과량·표본 SD·개별 대응 값을 함께 보고하고 여러 예산점을 독립 반복으로 세지 않는다. 클래스별·구성별 분석은 탐색적으로 해석한다.

## Ⅳ. 실험 결과 및 분석

### 4.1 라벨 예산에 따른 성능 변화

표 3과 그림 1은 동일한 YOLO26s의 Test mAP50-95 학습곡선이다. 같은 반복의 R0는 세 방법이 공유한다. GC10에서 DINOv2-FF의 평균은 15~35%의 모든 예산에서 Random보다 높았다. 30%에서는 26.05% 대 24.48%, 35%에서는 26.97% 대 24.74%였다. Entropy와 비교하면 20%에서는 DINOv2-FF가 낮고 이후 예산에서는 높아, 우위의 크기와 방향이 비교 방법과 예산에 따라 달랐다.

표 3. 예산별 Test mAP50-95

Table 3. Test mAP50-95 (%) across label budgets

| Data /<br>budget | N | Random | DINOv2-FF | Entropy |
| --- | --- | --- | --- | --- |
| GC10 10% | 160 | 19.83±0.60 | 19.83±0.60 | 19.83±0.60 |
| GC10 15% | 241 | 22.29±0.91 | 23.79±1.14 | 21.95±1.95 |
| GC10 20% | 321 | 23.97±1.49 | 24.38±1.93 | 24.78±1.11 |
| GC10 25% | 401 | 23.75±1.83 | 25.53±1.76 | 25.21±1.21 |
| GC10 30% | 481 | 24.48±1.39 | 26.05±1.87 | 25.94±1.07 |
| GC10 35% | 561 | 24.74±0.37 | 26.97±1.22 | 26.21±0.92 |
| NEU 10% | 126 | 26.98±2.22 | 26.98±2.22 | 26.98±2.22 |
| NEU 15% | 189 | 28.66±0.94 | 28.55±1.56 | 28.01±1.73 |
| NEU 20% | 252 | 29.76±1.23 | 29.52±1.15 | 29.23±1.57 |
| NEU 25% | 315 | 30.13±1.09 | 30.31±2.20 | 30.14±1.80 |
| NEU 30% | 377 | 31.47±0.87 | 30.86±0.62 | 30.87±0.59 |
| NEU 35% | 440 | 31.18±1.72 | 31.11±0.97 | 30.62±1.10 |

Mean±sample SD over five paired acquisition and learner seeds. R0 is shared.

![Figure1](../figures/Fig1.png)

그림 1. 라벨 이미지 수에 따른 Test 성능곡선

Fig. 1. Test learning curves with sample SD over five paired repetitions

NEU에서는 DINOv2-FF가 15%, 20%, 30%, 35%에서 Random보다 낮았고 25%에서만 소폭 높았다. 30%는 30.86% 대 31.47%, 35%는 31.11% 대 31.18%였다. 최종 예산의 작은 차이나 특정 구간만으로 전 구간의 선택 효과를 설명하지 않으며, 두 데이터셋의 점수를 직접 합산하지 않는다.

### 4.2 학습곡선의 요약과 반복별 효과

표 4는 실제 약 10~35% 이미지 수 구간의 평균 nAULC와 35% 성능이다. GC10에서 DINOv2-FF는 24.62%로 Random의 23.35%보다 1.27%p 높았으나, NEU에서는 29.65%로 Random의 29.82%보다 0.17%p 낮았다. 이는 같은 특징 선택법의 관측 효과가 데이터셋에 따라 달라졌음을 뜻한다.

표 4. nAULC와 최종 예산 성능

Table 4. nAULC and detection performance at the 35% budget

| Data | Method | nAULC (%) | AP35 (%) |
| --- | --- | --- | --- |
| GC10 | Random | 23.35±0.58 | 24.74±0.37 |
| GC10 | DINOv2-FF | 24.62±0.78 | 26.97±1.22 |
| GC10 | Entropy | 24.17±0.89 | 26.21±0.92 |
| NEU | Random | 29.82±0.82 | 31.18±1.72 |
| NEU | DINOv2-FF | 29.65±1.19 | 31.11±0.97 |
| NEU | Entropy | 29.41±1.18 | 30.62±1.10 |

Mean±sample SD, n=5 paired repetitions. AP35 denotes mAP50-95 at R5.

표 5와 그림 2에서 DINOv2-FF−Random의 반복별 nAULC 차이는 GC10에서 4개가 양수, 1개가 음수였고 NEU에서는 2개가 양수, 3개가 음수였다. Entropy 대비 차이도 두 자료 모두 3개만 양수였다. 따라서 평균의 크기와 개별 반복의 방향을 함께 살펴보아야 한다.

표 5. 대응 nAULC 차이와 정확 검정

Table 5. Paired nAULC differences and exact tests

| Data | Pair | Delta±SD<br>(pp) | Exact p | Holm p |
| --- | --- | --- | --- | --- |
| GC10 | D−R | +1.27±1.09 | 0.1250 | 0.5000 |
| GC10 | D−E | +0.45±0.99 | 0.4375 | 1.0000 |
| NEU | D−R | -0.17±0.74 | 0.6875 | 1.0000 |
| NEU | D−E | +0.25±0.74 | 0.5625 | 1.0000 |

Delta: mean±sample SD (pp), n=5. D: DINOv2-FF; R: Random; E: Entropy. Holm family: four comparisons.

![Figure2](../figures/Fig2.png)

그림 2. 반복별 대응 nAULC 차이

Fig. 2. Five paired nAULC differences for each comparison; black marks show means

GC10의 DINOv2-FF−Random은 양측 p=0.125, Holm p=0.50이었고, 나머지 세 비교의 Holm p값은 모두 1.00이었다. 지정한 검정으로 통계적 우위를 확립하지 못했다. 이 결과는 효과가 없거나 방법들이 동등하다는 증명이 아니며, 다른 검정을 사후 선택하여 유의성을 주장하지 않는다.

Full의 Test mAP50-95는 GC10 28.56±0.80%, NEU 34.72±1.16%였다. 데이터셋당 같은 학습 시드의 다섯 모델을 사용하였다. 예산별 성능과 대응 Full 비율은 별개 지표이며, Full을 학습곡선의 적분 구간에 포함하지 않는다.

### 4.3 정밀도와 재현율의 차이

표 6과 그림 3은 같은 R5 best checkpoint의 전체 지표다. GC10에서 DINOv2-FF는 Random보다 mAP50-95가 2.23%p, 정밀도가 3.87%p, 재현율이 3.16%p 높았다. 이 자료에서는 평균 AP와 P/R가 함께 높아졌지만 이 양상이 다른 데이터셋이나 모든 클래스에서도 유지되는지는 별도로 확인해야 한다.

표 6. 35% 예산의 Test 전체 지표

Table 6. Overall test metrics (%) at the 35% budget

| Metric | Random | DINOv2-FF | Entropy |
| --- | --- | --- | --- |
| GC10-DET (n=5) | GC10-DET (n=5) | GC10-DET (n=5) | GC10-DET (n=5) |
| mAP50 | 51.34±1.96 | 56.17±2.02 | 54.98±1.53 |
| mAP50-95 | 24.74±0.37 | 26.97±1.22 | 26.21±0.92 |
| Precision | 58.75±4.08 | 62.61±4.13 | 64.28±2.39 |
| Recall | 52.60±3.51 | 55.76±2.25 | 54.09±2.78 |
| NEU-DET (n=5) | NEU-DET (n=5) | NEU-DET (n=5) | NEU-DET (n=5) |
| mAP50 | 60.64±3.06 | 59.78±1.40 | 58.93±1.79 |
| mAP50-95 | 31.18±1.72 | 31.11±0.97 | 30.62±1.10 |
| Precision | 63.56±6.43 | 66.65±3.53 | 62.90±2.63 |
| Recall | 59.78±2.45 | 56.34±2.00 | 56.96±0.87 |

Mean±sample SD, n=5 paired repetitions. P/R use each model's smoothed mean-F1 operating point at IoU 0.50.

![Figure3](../figures/Fig3.png)

그림 3. 35% 예산의 Test 정밀도와 재현율

Fig. 3. Precision and recall at R5 with sample SD; DINO denotes DINOv2-FF

NEU에서는 DINOv2-FF의 정밀도가 Random보다 3.09%p 높았지만 재현율은 3.44%p 낮았다. mAP50-95는 0.07%p, mAP50는 0.86%p 낮았다. 따라서 정밀도의 증가만으로 평균 탐지 성능이나 누락 위험의 개선을 주장할 수 없다. 또한 모델별 F1 요약점이 달라 공통 confidence에서의 인과적 절충으로 해석하지 않는다.

### 4.4 클래스별 성능과 최저 클래스

그림 4에서 GC10의 DINOv2-FF−Random AP50-95 차이는 crease +9.37%p, waist folding +6.01%p인 반면 inclusion −1.30%p, silk spot −1.07%p였다. NEU에서도 scratches +4.33%p와 rolled-in scale −4.07%p, crazing −2.42%p가 함께 나타났다. 클래스 AP를 같은 가중치로 평균하는 mAP에서는 서로 다른 클래스의 증감이 상쇄될 수 있으므로, 평균 차이를 모든 결함의 향상으로 표현하지 않는다.

![Figure4](../figures/Fig4.png)

그림 4. 35% 예산에서 클래스별 Test AP50-95 차이

Fig. 4. Class-wise DINOv2-FF minus Random AP50-95 differences at R5 with paired sample SD

표 7. 예시 클래스의 Test AP

Table 7. Test AP for illustrative classes at R5

| Data / class | Rnd<br>AP50 | DINO<br>AP50 | Rnd<br>AP50-95 | DINO<br>AP50-95 |
| --- | --- | --- | --- | --- |
| GC10 / inclusion | 13.91 | 11.45 | 4.77 | 3.47 |
| GC10 / rolled pit | 18.12 | 27.46 | 6.34 | 8.11 |
| NEU / crazing | 29.13 | 23.55 | 8.93 | 6.51 |
| NEU / inclusion | 66.64 | 66.70 | 30.80 | 31.03 |
| NEU / rolled-in scale | 48.24 | 42.02 | 19.62 | 15.55 |
| NEU / scratches | 74.14 | 77.48 | 37.37 | 41.70 |

Means over five paired repetitions. Rnd: Random; DINO: DINOv2-FF. Descriptive examples.

표 7은 일부 클래스에서 AP50와 AP50-95가 어떻게 달라졌는지 보여준다. 클래스별 AP와 P/R 원배열도 보존하였다. 표의 사례는 전체 클래스 그림을 보완하는 탐색적 설명이며, 클래스별 우위를 검정하거나 원인을 분리한 결과가 아니다.

그림 5는 반복별·예산별 최저 클래스 AP50-95를 연결한 탐색적 요약이다. 실제 이미지 수로 적분한 최저 클래스 nAULC는 GC10에서 Random 1.97%, DINOv2-FF 2.69%였고, NEU에서는 7.26%와 6.32%로 0.94%p 낮았다. 각 점의 최저 클래스는 바뀔 수 있어 하나의 고정 클래스를 추적한 곡선과 다르며, 이 사후 요약에 별도의 유의성 주장을 부여하지 않는다.

![Figure5](../figures/Fig5.png)

그림 5. 예산별 최저 클래스 Test AP50-95

Fig. 5. Minimum-class AP50-95 per budget and repetition; bands show sample SD

### 4.5 선택 이미지 구성과 성능의 관계

표 8은 선택된 R5 이미지 중 해당 클래스를 포함한 평균 개수와 AP 차이를 비교한다. GC10 inclusion은 48.2장에서 27.0장으로 줄고 AP도 1.30%p 낮아졌다. 그러나 waist folding은 35.2장에서 20.6장으로 줄었는데 AP는 6.01%p 높았다. NEU에서는 crazing과 rolled-in scale의 선택 수와 AP가 함께 낮아졌고 scratches는 함께 높아졌다. 개수만으로 모든 클래스의 성능 변화를 설명하지는 못한다.

표 8. 선택 이미지 구성과 클래스 AP 변화

Table 8. Selected-image composition and class-wise AP differences at R5

| Data / class | Random<br>images | DINO<br>images | Delta AP<br>(pp) |
| --- | --- | --- | --- |
| GC10 / inclusion | 48.2 | 27.0 | -1.30 |
| GC10 / waist folding | 35.2 | 20.6 | +6.01 |
| GC10 / crease | 14.4 | 27.4 | +9.37 |
| NEU / crazing | 75.6 | 34.2 | -2.42 |
| NEU / rolled-in scale | 74.2 | 29.4 | -4.07 |
| NEU / scratches | 71.0 | 121.0 | +4.33 |

Image counts: means over five repetitions; classes may coexist in one image. Delta: DINOv2-FF minus Random AP50-95 (pp).

이 비교는 선택 완료 후 정답 주석을 이용한 기술적 분석이다. 이미지의 상자 수, 결함 크기, 배경과 공존 클래스도 함께 달라질 수 있다. 전역 DINOv2 특징이 배경이나 결함 구성을 반영했을 가능성은 있으나, 이를 AP 변화의 원인으로 확인한 대조 실험이나 새 예측 오류 감사는 수행하지 않았다.

### 4.6 대응 Full 참조와 라벨 예산

표 9는 같은 반복의 35% 점수를 해당 Full 점수로 나눈 비율과 AP 차이다. 비율을 반복별로 계산한 뒤 다섯 값의 평균과 표본 SD를 구하였다. 분모가 0이면 비율은 정의하지 않는 규칙을 사용하며, 이번 Full 점수는 모두 0보다 컸다.

표 9. 35% 예산의 대응 Full 도달률

Table 9. Paired Full attainment at the 35% budget

| Data | Method | Attainment<br>(%) | Gap<br>(pp) |
| --- | --- | --- | --- |
| GC10 | Random | 86.66±2.71 | -3.83±0.84 |
| GC10 | DINOv2-FF | 94.58±7.02 | -1.59±2.00 |
| GC10 | Entropy | 91.78±2.75 | -2.35±0.81 |
| NEU | Random | 89.88±5.38 | -3.54±1.90 |
| NEU | DINOv2-FF | 89.65±3.26 | -3.61±1.18 |
| NEU | Entropy | 88.30±4.96 | -4.10±1.80 |

Attainment: 100×AP35/paired Full. Gap: AP35 minus paired Full (pp). Mean±sample SD, n=5.

DINOv2-FF의 평균 Full 도달률은 GC10 94.58±7.02%, NEU 89.65±3.26%였고, 평균 AP 차이는 각각 −1.59%p와 −3.61%p였다. GC10에서 35% 이미지로 Full의 약 95% 수준에 접근했다는 기술적 해석은 가능하지만, 모든 반복이나 두 데이터셋에서 95%를 달성했다는 결론은 아니다.

Full도 200epoch를 학습하므로 이미지가 많은 만큼 더 많은 업데이트를 포함한다. 이는 같은 학습 시드의 경험적 참조이며 절대 상한이 아니다. 이미지 예산은 박스 수, 주석 난도, DINOv2 특징 추출 비용과 실제 작업 시간을 포함하지 않아 이 비율을 주석 비용 절감률로 바꾸지 않는다.

## Ⅴ. 논의

첫 번째 질문에 대해 DINOv2-FF의 평균 효과는 데이터셋에 따라 달랐다. GC10의 Random 대비 nAULC 차이는 +1.27%p, NEU는 −0.17%p였으며 개별 반복의 부호도 완전히 일치하지 않았다. 전체 곡선과 대응 효과를 함께 보고해야 하며, 작은 표본에서 유의성을 확립하지 못한 결과를 우위나 동등성으로 바꾸어 해석하지 않는다.

두 번째 질문에서는 GC10의 평균 AP와 P/R가 함께 높아졌어도 일부 클래스 AP는 낮아졌다. NEU에서는 정밀도 상승과 재현율 하락, 클래스별 상반된 변화와 낮은 최저 클래스 요약이 관찰되었다. 선택 이미지 구성은 이를 설명하는 단서지만, 이미지 수·결함 크기·배경 등의 영향을 분리하지 못해 선택 특징의 인과 효과로 단정할 수 없다.

다섯 반복은 획득·학습 시드를 함께 바꾼 결합 단위여서 두 무작위성의 기여를 분리하지 못한다. 두 공개 자료의 한 고정 분할과 YOLO26s 한 검출기로 얻은 결과이므로 다른 분할·검출기·실제 생산 환경으로의 일반화에는 검증이 더 필요하다. 이번 학습과 선택은 Test를 사용하지 않았으나 원천 자료에는 과거 평가·이전 분할 사용 이력이 있어 연구 전 과정에서 미관찰인 외부 검증으로 부르지 않는다.

서버 Blackwell/CUDA 12.9 학습과 로컬 RTX3070/CUDA 12.6 평가의 차이를 기록했으며 비트 단위 재현을 주장하지 않는다. P/R는 모델별 F1 요약점이므로 공통 운영 confidence나 클래스별 오탐·미검출 비용을 반영한 산업 위험 지표가 아니다. 클래스별 원지표를 보존했지만 고정 운영점의 오류 수와 새로운 예측 기반 기전 분석은 본 결과에 포함하지 않았다.

## Ⅵ. 결론

본 연구는 금속 표면 결함 탐지에서 고정 DINOv2 기반 표본 선택을 동일 YOLO26s·200epoch·다섯 획득/학습 시드 쌍으로 비교하고 170개 모델의 고정 Test를 완료하였다. DINOv2-FF와 Random의 평균 nAULC 차이는 GC10 +1.27%p, NEU −0.17%p였고 지정한 네 비교에서 Holm 보정 후 통계적 우위는 확립되지 않았다.

GC10의 35% 평균 성능과 대응 Full 비율이 높아진 결과를 모든 데이터셋·클래스의 개선으로 일반화할 수 없었다. NEU에서는 정밀도와 재현율의 방향이 갈렸고 최저 클래스 nAULC도 낮아졌다. 따라서 특징 기반 선택은 평균 학습곡선, 반복별 효과와 클래스 구성을 함께 평가해야 한다. 향후에는 독립 분할, 두 무작위성의 교차 설계와 공통 운영점의 오류 분석을 통해 관측 효과의 범위를 확인할 필요가 있다.

## References

[1] X. Lv, F. Duan, J.-J. Jiang, X. Fu, and L. Gan, “Deep metallic surface defect detection: The new benchmark and detection network,” Sensors, vol. 20, no. 6, article 1562, Mar. 2020.
https://doi.org/10.3390/s20061562

[2] K. Song and Y. Yan, “A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects,” Applied Surface Science, vol. 285, pp. 858–864, Nov. 2013.
https://doi.org/10.1016/j.apsusc.2013.09.002

[3] Y. Lee and J. Kang, “Performance analysis by the number of learning images on anti-drone object detection system with YOLO,” J. Korean Inst. Commun. Inf. Sci., vol. 49, no. 3, pp. 356–360, Mar. 2024.
https://doi.org/10.7840/kics.2024.49.3.356

[4] X. Lv, F. Duan, J.-J. Jiang, X. Fu, and L. Gan, “Deep active learning for surface defect detection,” Sensors, vol. 20, no. 6, article 1650, Mar. 2020.
https://doi.org/10.3390/s20061650

[5] O. Sener and S. Savarese, “Active learning for convolutional neural networks: A core-set approach,” in Proc. ICLR, Vancouver, Canada, 2018.
https://arxiv.org/abs/1708.00489

[6] M. Oquab et al., “DINOv2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.
https://arxiv.org/abs/2304.07193

[7] P. Munjal, N. Hayat, M. Hayat, J. Sourati, and S. Khan, “Towards robust and reproducible active learning using neural networks,” in Proc. IEEE/CVF CVPR, pp. 223–232, Jun. 2022.
https://doi.org/10.1109/CVPR52688.2022.00032

[8] D. Probst, H. Raza, and E. Rodner, “Evaluating zero-cost active learning for object detection,” arXiv:2212.04211, 2022.
https://doi.org/10.48550/arXiv.2212.04211

[9] Ultralytics, “Ultralytics YOLO26,” accessed Oct. 27, 2026.
https://docs.ultralytics.com/models/yolo26/

[10] S. Holm, “A simple sequentially rejective multiple test procedure,” Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65–70, 1979.
https://www.jstor.org/stable/4615733

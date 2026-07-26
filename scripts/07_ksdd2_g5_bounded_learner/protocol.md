# KSDD2 G5 Single-Learner Bounded Translation Protocol

Prepared: 2026-07-26  
Protocol ID: `KSDD2_G5_SINGLE_UNET_V1`  
Status before execution: `READY_TO_FREEZE`

## 1. 연구 역할

G1–G4의 prospective 결과는 PatchCore가 KSDD2에서 selection-only gate를
통과했음을 보여주었지만, 선택된 이미지가 실제 segmentation 학습 효용으로
번역되는지는 보여주지 않았다.

G5의 질문은 하나뿐이다.

> 동일한 200-image label budget에서, frozen PatchCore query로 구성한
> 학습 집합이 frozen Random query 학습 집합보다 하나의 고정된 U-Net에서
> 더 높은 segmentation utility를 만드는가?

G5는 더 좋은 learner를 찾는 benchmark가 아니다. 설정 하나가 실패하면
다른 backbone, loss, epoch 또는 threshold로 구조하지 않고
`TRANSLATION_NOT_OBSERVED_STOP`으로 종료한다.

## 2. Prospective 범위와 한계

이 protocol은 G5 learner outcome을 보기 전에 동결된다. 하지만 G1–G4 결과와
development label 통계가 이미 알려진 뒤 작성되었으므로, 전체 연구 시작
시점부터의 preregistration은 아니다. 따라서 결과는 **screen-stage
prospective evidence**로만 해석한다.

Development 466장은 G1에서 signal validity 평가에 이미 사용됐다. G5에서는
학습·early stopping·checkpoint 선택에 사용하지 않고 최종 epoch가 끝난 뒤
한 번만 평가한다. 이 재사용 때문에 G5 결과는 final generalization evidence가
아니다.

Official test 1,004장의 image/mask content는 계속 봉인한다.

## 3. Upstream authorization

G5는 다음 upstream artifact가 모두 유효할 때만 시작한다.

- G1–G4 protocol: `KSDD2_PATCHCORE_G1_G4_V1`
- Frozen commit:
  `f10fd92985d63408212edd24e80d741de30ba5f7`
- G1/G2/G3/G4: 모두 `PASS`
- Authorization: `BOUNDED_SCREEN`
- Official-test authorization: `false`
- Freeze manifest, authorization decision, selection CSV의 SHA-256 검증

Upstream hash와 하나라도 다르면 learner 학습을 시작하지 않는다.

## 4. Paired training sets

Acquisition seed 0–9 각각에 대해 두 학습 집합을 만든다.

### PatchCore arm

`initial_seedXX` 100장 + `selected_patchcore_seedXX` 100장 = 200장

### Random arm

동일한 `initial_seedXX` 100장 + `selected_random_seedXX` 100장 = 200장

Initial과 query는 seed 내에서 중복되지 않아야 한다. 각 queried image는
mask annotation이 완료됐다고 가정하며 정상 이미지는 empty mask를 가진다.

두 arm은 image count가 같지만 실제 mask annotation 시간은 측정하지 않았다.
따라서 동일 annotation cost 또는 비용 절감 주장은 금지한다.

## 5. 단 하나의 learner 설정

Learner는 repository-native PyTorch compact U-Net 하나로 고정한다.
U-Net 계열은 encoder context와 decoder localization을 skip connection으로
결합하는 단일 binary segmentation learner로 사용한다.

- Encoder channels: `16, 32, 64, 128, 256`
- Normalization: GroupNorm, 8 groups
- Activation: ReLU
- Downsampling: 2×2 max pooling
- Upsampling: bilinear + 3×3 convolution
- Skip connection: concatenation
- Pretrained encoder: 없음
- Dropout: 없음
- Input: RGB `704×256` (`H×W`; native portrait orientation preserved)
- Augmentation: horizontal flip `p=0.5`만 허용

Architecture, parameter initialization, preprocessing, optimizer와 training
seed는 모든 arm/seed에서 동일하다.

## 6. 고정 학습 설정

- Training seeds: `260727`, `260728`, `260729`
- Epoch: `80`
- Batch size: `4`
- Optimizer: AdamW
- Learning rate: `3e-4`
- Weight decay: `1e-4`
- Scheduler: 없음
- Gradient clipping: `1.0`
- Precision: float32
- AMP: 사용하지 않음
- Loss: `0.5 × BCEWithLogits + 0.5 × soft Dice`
- BCE `pos_weight`: 사용하지 않음
- Early stopping: 없음
- Validation during training: 없음
- 평가 checkpoint: epoch 80 final state 하나

각 acquisition seed와 strategy는 세 training seed로 반복한다. 같은
training seed의 PatchCore/Random pair는 동일한 initialization으로 시작한다.
총 run 수는 `2 strategies × 10 acquisition seeds × 3 training seeds = 60`이다.
Infrastructure 오류는 동일 설정으로 한 번만 재시도할 수 있다. 재시도에서도
실패하면 결과를 대체하거나 설정을 낮추지 않고
`NOT_IDENTIFIABLE_STOP`으로 종료한다.

## 7. 평가

Development 466장을 epoch 80 이후 한 번만 평가한다.

### Primary outcome

53개 positive development image 각각의 hard Dice를 구한 뒤 macro 평균한다.

- Prediction: `sigmoid(logit) >= 0.5`
- Positive target에서 empty prediction의 Dice: `0`
- 비교 단위:
  `PatchCore macro Dice − Random macro Dice`, acquisition seed별 paired delta

정상 이미지의 empty-mask Dice는 primary에 넣지 않는다. 정상 이미지가
대부분인 데이터에서 trivial empty prediction이 Dice를 부풀리는 것을
방지하기 위해서다.

### Secondary outcomes

- 전체 pixel average precision의 고정 4,096-bin histogram 근사
- Max-pixel probability 기반 image AUROC
- 정상 이미지 false-positive rate
- Positive image에서 16-pixel 이상 component를 하나라도 찾는 recall

Normal false positive는 threshold 0.5에서 면적 16 pixel 이상의 predicted
component가 하나라도 있을 때로 정의한다.

## 8. 통계

Inferential unit은 열 개 acquisition seed다. 세 training seed를 독립
표본처럼 세지 않는다. 먼저 각 acquisition-seed/strategy 안에서 세 결과를
평균한 뒤, 열 개 paired acquisition delta에 대해서만 추론한다.

- Paired bootstrap 5,000회
- 95% confidence interval
- Primary comparison 하나만 confirmatory로 사용
- Seed 삭제와 missing-run imputation 금지
- Leave-one-seed-out mean delta를 stability check로 사용

## 9. Frozen G5 gate

### Run conformance

- 60/60 run 완료
- NaN/non-finite metric 0개

### Primary learning translation

모두 만족해야 한다.

- Mean macro-Dice delta ≥ `+0.05`
- Paired 95% CI lower bound > `0`
- Positive delta seed fraction ≥ `0.80`
- Leave-one-seed-out minimum mean delta > `0`

### Corroboration

- Mean all-pixel AP delta > `0`

### Safety

- PatchCore arm normal-image FPR mean ≤ `0.20`
- PatchCore−Random normal-image FPR mean delta ≤ `+0.05`

## 10. 결정 규칙

| 조건 | 결정 |
|---|---|
| 모든 translation·corroboration·safety check PASS | `LEARNING_TRANSLATION_OBSERVED` |
| Primary/corroboration FAIL | `TRANSLATION_NOT_OBSERVED_STOP` |
| Learning PASS, safety FAIL | `LEARNING_GAIN_WITH_SAFETY_FAIL_STOP` |
| 동일 재시도 후 infrastructure failure | `NOT_IDENTIFIABLE_STOP` |

어떤 결과도 automatic official-test 사용, hyperparameter rescue 또는 두 번째
learner를 허용하지 않는다.

## 11. 허용되는 주장

PASS 시:

> 하나의 사전 동결된 compact U-Net screen에서 PatchCore selection의
> discovery advantage가 KSDD2 development segmentation utility로
> 번역됐다.

FAIL 시:

> 강한 selection-only discovery가 동일한 label-image budget의 frozen
> segmentation learner utility로 번역되지 않아, selection–learning
> translation gap이 prospective하게 관찰됐다.

## 12. 금지되는 주장

- U-Net 또는 PatchCore의 보편적 우위
- SOTA segmentation 성능
- Independent production-pool generalization
- Official-test 성능
- 동일 mask-annotation 시간 또는 비용
- 최적 learner configuration
- Workflow predictive accuracy
- Category/source safety

## 13. 참고 근거

- Ronneberger, Fischer, Brox, *U-Net: Convolutional Networks for Biomedical
  Image Segmentation*, 2015:
  https://arxiv.org/abs/1505.04597
- PyTorch `BCEWithLogitsLoss`: sigmoid와 BCE를 결합한 numerically stable
  binary loss 구현:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

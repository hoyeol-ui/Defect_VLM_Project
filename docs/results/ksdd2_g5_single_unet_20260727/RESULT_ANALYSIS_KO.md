# KSDD2 G5 결과 해석

Protocol ID: `KSDD2_G5_SINGLE_UNET_V1`  
판정: **`TRANSLATION_NOT_OBSERVED_STOP`**  
해석 범위: KSDD2의 재사용 development split과 사전 동결된 compact U-Net
한 설정에서 수행한 bounded screen

## 1. 한 문장 결론

PatchCore selection은 Random보다 segmentation 성능을 **작지만 일관되게**
높였으나, 사전에 정한 최소 학습 이득 `ΔDice ≥ +0.05`를 충족하지 못했다.
따라서 “효과가 전혀 없다”가 아니라 **최소한 의미 있다고 정의한 크기의
learning translation을 입증하지 못했으므로 STOP**이 정확한 해석이다.

## 2. Frozen Gate 판정

| 판정 항목 | 관찰값 | 기준 | 결과 |
|---|---:|---:|---|
| Mean paired macro Dice delta | `+0.0376` | `≥ +0.05` | **FAIL** |
| Bootstrap CI lower bound | `+0.0196` | `> 0` | PASS |
| Positive acquisition seeds | `8/10` | `≥ 8/10` | PASS |
| Leave-one-seed-out minimum | `+0.0312` | `> 0` | PASS |
| Pixel AP delta | `+0.0498` | `> 0` | PASS |
| PatchCore normal FPR | `0.1601` | `≤ 0.20` | PASS |
| FPR delta vs Random | `+0.0106` | `≤ +0.05` | PASS |

Gate를 실패시킨 것은 정상 오탐이나 방향 불안정이 아니라 **오직 frozen
minimum effect size 미달**이다. PatchCore Dice `0.6084`와 Random Dice
`0.5709`의 차이는 절대 `+0.0376`, Random 대비 약 `+6.6%`의 상대 증가다.
하지만 protocol은 상대 증가가 아니라 절대 Dice `+0.05`를 요구했다.

95% CI가 0을 배제하므로 평균적인 이득 방향에 대한 증거는 있다. 그러나
CI `[+0.0196, +0.0570]`는 작은 이득부터 frozen threshold를 조금 넘는
이득까지 포함한다. 따라서 현재 표본은 “양의 효과”와는 양립하지만,
“최소 `+0.05`의 유용한 효과가 확립됐다”는 주장에는 충분하지 않다.

## 3. 60-run이 실제로 말해 주는 것

60개 job은 `2 strategies × 10 acquisition seeds × 3 training seeds`다.
세 training seed를 먼저 평균한 뒤 acquisition seed 10개를 primary 통계
단위로 사용했으므로 `n=60`으로 표본을 부풀리지 않았다.

training seed별 평균 Dice delta는 `+0.0353`에서 `+0.0399`로 모두
비슷하다. 따라서 전체 평균 이득이 특정 training seed 하나 때문에 생긴
것으로 보이지는 않는다. 반면 개별 strategy pair의 minimum delta는
`-0.0949`에서 `-0.0537`이고 positive pair fraction은 `0.6–0.9`다.
즉 평균 이득은 존재하지만, 특정 acquisition/training 조합에서는 Random이
더 좋았으며 learner-level 안정성이 완전하지 않다.

## 4. Discovery가 강했는데 Dice 이득이 제한된 이유

PatchCore는 query 100장에서 Random보다 평균 `+88.8`개의 positive image를
더 선택했다. 그런데 Dice 이득은 `+0.0376`이었다. 이 불균형은 다음을
시사한다.

1. 결함 이미지를 더 많이 확보하는 것과 새로운 이미지의 mask를 더 정확히
   예측하는 것은 같은 목표가 아니다.
2. 유사한 positive sample이 반복되면 추가 mask의 marginal information이
   빠르게 감소할 수 있다.
3. learner capacity, defect morphology, background texture 및 selected-set
   diversity가 discovery advantage의 번역을 제한할 수 있다.

사후 Spearman은 `ρ=0.394`, `p=0.260`, `n=10`이다. 방향은 약한 양의
연관성과 양립하지만 표본이 작고 유의하지 않다. 따라서 “더 많이 발견한
seed일수록 Dice가 증가한다”거나 인과관계를 주장할 수 없다.

## 5. Annotation burden과 공정성

두 arm은 동일한 **100-image query budget**이지만 동일 annotation cost가
아니다.

- Positive-image delta: `+88.8장`
- Defect-pixel burden ratio: `15.68×`
- Component burden ratio: `11.87×`

PatchCore arm은 훨씬 많은 결함 mask를 제공받았다. 이는 selector의
discovery 성과인 동시에 사람이 그 mask를 작성해야 한다면 추가 노동이다.
그러므로 이번 결과는 동일 image-count budget 비교이지 동일 시간·금액
budget에서의 효율성 검증이 아니다.

## 6. Query overlap이 의미하는 한계

PatchCore query의 pairwise Jaccard 평균은 `0.813`이고, 열 seed의 query
union은 119장뿐이며 51장은 모든 seed에서 반복 선택됐다. Random의 Jaccard
평균은 `0.026`, union은 799장이다.

따라서 PatchCore의 열 acquisition seed는 사실상 비슷한 핵심 positive
집합을 반복 측정한다. 현재 CI는 동일 pool 내부 acquisition perturbation에
대한 안정성을 보여 주지만, 독립 생산라인·lot·site 일반화를 뜻하지 않는다.

## 7. Safety 해석

우려했던 safety failure는 발생하지 않았다.

- PatchCore normal FPR `0.1601 ≤ 0.20`
- PatchCore−Random FPR `+0.0106 ≤ +0.05`

즉 이 결과는 `LEARNING_GAIN_WITH_SAFETY_FAIL_STOP`이 아니다. Pixel AP도
양수이므로 “Dice만 좋아지고 pixel ranking은 악화됐다”는 패턴도 아니다.
현재 STOP은 안전성 문제가 아니라 사전 최소 효과크기 미달 때문이다.

단, acquisition seed 0과 2에서는 FPR delta가 각각 약 `+0.147`,
`+0.282`로 컸다. 평균 Gate는 통과했지만 seed별 운영 부담은 이질적이다.
이는 post-hoc 진단이며 frozen 판정을 변경하지 않는다.

## 8. 연구적으로 갖는 의미

이 결과는 기존 발표의 핵심 명제인 `Selection ≠ Learning`을 구체화한다.

- G1–G4: PatchCore는 강한 signal/discovery advantage를 보였다.
- G5: 그 이득은 learner에 일부 번역됐지만, frozen minimum effect에는
  도달하지 못했다.
- Safety: 평균 정상 오탐 기준은 통과했다.
- Decision: Official test와 추가 rescue를 허가할 만큼 충분하지 않았다.

따라서 가장 정확한 논문 문장은 다음과 같다.

> KSDD2의 사전 동결 compact U-Net screen에서 PatchCore selection은
> Random 대비 양의 segmentation 이득을 보였으나, 사전 정의한 최소
> translation effect를 충족하지 못해 확장 권한이 부여되지 않았다.

이 결과는 PatchCore가 무의미하다는 증거도, translation gap이 언제나
발생한다는 증명도 아니다. 강한 discovery advantage가 downstream utility로
자동 또는 비례 번역된다고 가정해서는 안 된다는 prospective 사례다.

## 9. 허용되는 주장과 금지되는 주장

허용:

- 동일 image-count budget에서 PatchCore가 Random보다 훨씬 많은 결함
  이미지를 선택했다.
- 동결 learner에서 평균 Dice와 Pixel AP는 양의 방향이었다.
- frozen minimum Dice effect가 충족되지 않아 protocol에 따라 STOP했다.
- Official test를 열지 않은 것은 workflow conformance의 성공이다.

금지:

- PatchCore의 보편적 우위
- learning effect가 없었다는 단정
- 동일 annotation cost 또는 실제 시간 절감
- 독립 production pool 일반화
- official-test 성능
- threshold, epoch, loss 또는 backbone 변경으로 현재 판정을 rescue

## 10. 다음 행동

현재 G5는 종료한다. 60개 checkpoint, frozen manifest, paired metrics 및
판정 JSON을 봉인하고 official test는 계속 잠근다.

후속 연구를 한다면 현재 결과의 rescue가 아니라 별도 질문과 protocol이어야
한다. 우선순위는 다음과 같다.

1. 실제 mask annotation time을 측정하는 cost-normalized acquisition
2. 동일 pool seed 반복이 아닌 새로운 lot/site에서의 external replication
3. positive yield와 morphology diversity를 분리하는 selection diagnosis

이 세 실험은 새 protocol로 사전 동결해야 하며 현재 frozen decision을
변경하는 용도로 사용하면 안 된다.

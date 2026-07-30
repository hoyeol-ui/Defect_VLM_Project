# V7 DINO full learning-curve 결과 해석

작성일: 2026-07-12  
원본 실행 폴더: `runs/active_learning_ablation_v7_full_curve/v7_full_curve_20260711_211848`  
문서화 폴더: `docs/results/v7_full_curve_20260711_211848`  
Final test 사용 여부: **False**

## 1. 실험 목적

이 실험은 최종 35장 한 번의 성능이 아니라, 라벨 예산이 15 → 20 → 25 → 30 → 35장으로 증가할 때 acquisition 전략별 학습 곡선과 AULC를 비교하기 위한 본격 개발 세트 실험이다.

비교 전략은 다음 세 가지로 고정했다.

| 표기 | 전략 |
|---|---|
| Random | `GTFreeRandom` |
| DBC | `GTFreeDatasetBalancedConsistency` |
| Visual | `GTFreeDatasetBalancedVisualDiversity` |

이전 gate 실험에서는 DINO Visual-only가 최종 35장 기준으로 Random과 DBC보다 좋았지만, 이번 full learning-curve에서는 acquisition round 전체의 안정성과 효율을 다시 검증했다.

## 2. 핵심 결과 요약

| 전략 | final mAP@50 | final mAP@50-95 | normalized AULC mAP@50 | normalized AULC mAP@50-95 |
|---|---:|---:|---:|---:|
| DBC | 0.3341 ± 0.0325 | 0.1473 ± 0.0198 | 0.2718 | 0.1175 |
| Visual | 0.3698 ± 0.0180 | 0.1617 ± 0.0091 | 0.2837 | 0.1230 |
| Random | **0.3779 ± 0.0227** | **0.1690 ± 0.0159** | **0.2853** | **0.1257** |

결론적으로 Visual은 DBC보다 일관되게 낫지만, Random을 넘는 전략으로 확정하기에는 부족하다.

## 3. 쌍대 비교 해석

### Visual vs DBC

Visual은 DBC 대비 명확한 개선을 보였다.

| metric | 평균 차이 | seed 승/패 |
|---|---:|---:|
| final mAP@50 | +0.0357 | 4승 1패 |
| final mAP@50-95 | +0.0143 | 5승 0패 |
| normalized AULC mAP@50 | +0.0119 | 3승 2패 |
| normalized AULC mAP@50-95 | +0.0055 | 4승 1패 |

즉, DINO visual diversity는 기존 consistency-balanced 계열보다 detector 학습에 더 유효한 샘플을 선택하는 경향이 있다.

### Visual vs Random

Visual은 Random 대비 AULC에서는 일부 seed에서 이겼지만, 전체 평균과 최종 성능에서는 Random을 넘지 못했다.

| metric | 평균 차이 | seed 승/패 |
|---|---:|---:|
| final mAP@50 | -0.0080 | 2승 3패 |
| final mAP@50-95 | -0.0073 | 1승 4패 |
| normalized AULC mAP@50 | -0.0017 | 3승 2패 |
| normalized AULC mAP@50-95 | -0.0027 | 3승 2패 |

따라서 현재 단계에서 “Visual-only가 Random보다 우수하다”고 주장하면 방어가 어렵다. 더 안전한 표현은 “Visual diversity는 consistency-balanced baseline을 개선했지만, 강한 Random baseline을 안정적으로 초과하지는 못했다”이다.

### DBC vs Random

DBC는 Random 대비 최종 성능에서 뚜렷하게 낮았다.

| metric | 평균 차이 | seed 승/패 |
|---|---:|---:|
| final mAP@50 | -0.0437 | 0승 5패 |
| final mAP@50-95 | -0.0216 | 0승 5패 |

기존 consistency/dataset-balanced acquisition만으로는 이 데이터 구성에서 Random 대비 우위를 만들기 어렵다는 신호다.

## 4. 왜 Visual이 Random을 못 넘었나?

선택 샘플 분석에서는 Visual 전략이 의도한 동작을 실제로 수행한 것이 확인된다.

| 전략 | 선택 batch 평균 cosine similarity | 기존 labeled set과 평균 거리 |
|---|---:|---:|
| DBC | 0.4379 | 0.2792 |
| Random | 0.4466 | 0.2766 |
| Visual | **0.3985** | **0.3833** |

cosine similarity가 낮을수록 선택 batch 내부 중복이 낮고, labeled set과의 거리가 클수록 기존 labeled set에 없는 visual novelty를 더 많이 뽑았다는 뜻이다. 따라서 Visual 전략은 구현상 실패한 것이 아니라, DINO 공간에서 더 다양한 샘플을 실제로 선택했다.

다만 최종 35장 기준 실제 bbox/클래스 통계는 Random과 큰 차이가 나지 않았다.

| 전략 | 평균 bbox 수 | 평균 bbox/image | multi-object 비율 | 실제 class 수 | class entropy |
|---|---:|---:|---:|---:|---:|
| DBC | 66.6 | 1.9029 | 0.5600 | 8.0 | 2.5772 |
| Visual | 66.4 | 1.8971 | 0.5257 | 8.6 | **2.6367** |
| Random | **66.8** | **1.9086** | 0.5486 | **9.2** | 2.5995 |

Visual은 visual novelty를 확보했지만, detector 성능에 직접적으로 유리한 instance richness, class coverage, hard-but-learnable sample 구성에서는 Random과 비슷하거나 일부 지표에서 밀렸다. 이것이 gate 실험의 낙관적 결과가 full-curve에서 약해진 주된 이유로 보인다.

## 5. 연구적 결론

이번 결과는 실패라기보다 좋은 분리 결과다.

1. **학습 부족 문제는 이미 상당히 해소됐다.**  
   100 epoch, patience 100 조건에서 mAP가 이전 20 epoch 실험보다 크게 올라갔다.

2. **Consistency-only 또는 DBC-only는 현재 데이터 구성에서 Random을 넘기 어렵다.**  
   DBC는 Visual보다도 낮고 Random 대비 final mAP에서 0승 5패였다.

3. **DINO visual diversity는 acquisition 품질을 일부 개선한다.**  
   DBC 대비 final mAP@50-95에서 5승 0패이고, 선택 batch 중복도도 가장 낮다.

4. **하지만 Visual-only는 Random 대비 아직 충분하지 않다.**  
   Random은 우연히 bbox 수, 실제 class coverage, instance-richness를 잘 가져가는 강한 baseline이다.

5. **Final test를 아직 쓰지 않은 것은 매우 중요하다.**  
   이 결과는 개발 세트에서 방법론을 정리하기 위한 evidence이며, 최종 테스트는 방법을 잠근 뒤 한 번만 사용해야 한다.

## 6. 다음 실험 제안

바로 final test로 가지 말고, 다음 후보를 개발 세트에서 제한적으로 한 번 더 검증하는 것이 안전하다.

### 후보 A: Visual + GT-free pseudo-instance proxy

Visual diversity가 중복을 줄이는 데는 성공했으므로, 부족한 축은 detector utility에 더 가까운 instance richness다. 실제 XML bbox를 acquisition에 쓰지 않고, GT-free pseudo box/objectness proxy를 추가하는 방향이 타당하다.

예시:

```text
score(x) = 0.5 * visual_diversity(x) + 0.5 * pseudo_instance_richness(x)
```

단, pseudo-instance 가중치를 넓게 탐색하지 말고 1~2개 후보만 고정해야 한다.

### 후보 B: Budget 45 또는 55까지 확장

35장은 너무 작은 예산이라 seed 변동과 Random 우연성이 크다. 논문 주장 자체가 low-budget active learning이면 35장을 유지할 수 있지만, 방법론 검증용 개발 실험에서는 45 또는 55까지 확장해 learning curve의 안정성을 확인할 가치가 있다.

### 후보 C: Visual 후보군 + class/dataset quota 유지

Visual-only가 novelty를 잘 확보하는 대신 class coverage에서 Random 대비 부족한 seed가 있었다. 따라서 visual diversity를 쓰되, dataset/class-hint quota 또는 cumulative deficit constraint를 완전히 버리지 않는 조합을 제한적으로 검증할 수 있다.

## 7. 현재 주장 가능 문장

현재 결과만으로 안전하게 쓸 수 있는 문장은 다음과 같다.

> In the development split, DINOv2-based visual diversity consistently improved over the dataset-balanced consistency baseline, but did not consistently outperform the strong random baseline across the full active-learning curve. This suggests that visual diversity reduces selection redundancy, yet detector utility also depends on instance richness and actual defect coverage, which random sampling can capture surprisingly well under the current small-budget setting.

한국어로는 다음처럼 정리할 수 있다.

> 개발 세트 기준으로 DINOv2 visual diversity는 consistency-balanced baseline보다 안정적으로 개선되었지만, full active-learning curve 전체에서 강한 Random baseline을 일관되게 넘지는 못했다. 이는 visual diversity가 선택 중복을 줄이는 데는 효과적이나, detector 성능에는 bbox instance richness와 실제 결함 class coverage가 함께 중요하며, 작은 라벨 예산에서는 Random이 이 요소들을 우연히 잘 확보할 수 있음을 시사한다.


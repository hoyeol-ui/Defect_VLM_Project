# 논문 핵심 정의 정본 — 2026-09-07

## 현재 연구의 환경

제한된 라벨 이미지 조건의 산업 결함 데이터이다. 데이터셋은 GC10-DET와 NEU-DET이며, 고정된 train pool 안에서 학습에 사용하는 이미지 수를 약 10%에서 약 35%까지 늘린다. validation은 각 학습 단계의 `best.pt` 선택에만, 잠긴 test는 최종 평가에만 사용한다.

## Task

산업 결함 객체 탐지이다.

## 연구 목적

동일한 수의 라벨 이미지를 사용할 때 능동학습 표본 선택 방법이 detector 학습 성능에 제공하는 효용과 한계를 확인하는 것이다.

## 현재 방법

- **Representation:** DINOv2-small ViT-S/14의 384차원 이미지 표현. 모델 가중치는 업데이트하지 않는다.
- **Acquisition:** L2 정규화 표현의 cosine distance를 이용한 greedy farthest-first. 현재 라벨 집합에서 가장 멀리 있는 이미지를 한 장씩 자동 선택하고 즉시 기준 집합에 추가한다.
- **Learner:** 매 데이터셋·표본 수 단계에서 동일한 봉인 checkpoint로 독립 cold-start하는 YOLOv8n.
- **Comparator:** 남은 train pool에서 균등 비복원 추출하는 Random.

따라서 DINOv2는 이미지를 표현하는 구성요소이고, 실제 선택 규칙은 cosine greedy farthest-first이다. 현재 효과는 두 구성요소를 결합한 표본 선택 방법의 효과로 해석한다.

## 현재 주요 결과

Primary endpoint는 R0~R5, 즉 train pool의 약 10~35% 구간에서 잠긴 test의 mAP50-95 곡선을 요약한 nAULC이다.

| Dataset | Random nAULC | DINOv2 표현 기반 선택 nAULC | Paired 차이 | 95% CI | Acquisition 방향 |
|---|---:|---:|---:|---:|---:|
| GC10-DET | 0.244643 | 0.250454 | +0.005812 (+0.581%p) | [0.003970, 0.007576] | 10/10 개선 |
| NEU-DET | 0.351950 | 0.357731 | +0.005782 (+0.578%p) | [0.003100, 0.008119] | 9/10 개선 |

두 데이터셋의 Holm 보정 exact sign-flip p-value는 각각 0.003906이다. 그러나 평균 이득이 모든 클래스의 이득을 뜻하지는 않는다. NEU-DET의 worst-class nAULC 차이는 -0.014879이고, R5 worst-class AP50-95 차이는 -0.031182로 악화됐다. 사후 표현 감사에서도 GC10-DET 8번 클래스 `rolled_pit`은 32장 중 최근접 이웃이 같은 클래스를 보존한 비율이 0.25에 그쳤다.

## 현재 주장 가능한 것

1. 봉인된 현재 protocol의 약 10~35% 이미지 구간에서, DINOv2 표현과 farthest-first를 결합한 선택은 Random보다 두 데이터셋의 평균 nAULC를 각각 약 0.58%p 높였다.
2. 이 평균 차이는 acquisition seed를 나눈 반복 실험에서 GC10-DET 10/10, NEU-DET 9/10의 방향 일관성을 보였다.
3. acquisition seed와 learner seed를 분리하고, validation checkpoint 선택과 잠긴 test 평가를 분리한 통제 절차로 downstream 학습 효용을 확인했다.
4. 평균적인 이득과 취약 클래스의 악화가 함께 나타날 수 있으므로, 표본 선택의 전체 효용과 클래스별 안전성은 별도로 평가해야 한다.
5. 사후 정량 감사에서 두 데이터셋의 DINOv2 표현에 라벨 관련 이웃 구조가 확인됐지만, GC10-DET의 클래스별 보존 정도는 불균일했다.

## 주장하면 안 되는 것

- 모든 클래스의 성능이 향상됐다는 주장
- 새로운 DINO 알고리즘 또는 새로운 보편적 능동학습 알고리즘을 개발했다는 주장
- 실제 annotation 시간이나 금전 비용을 절감했다는 주장
- SOTA를 달성했다는 주장
- 산업 결함 전체 또는 다른 산업 domain으로 일반화된다는 주장

## 현재 최종 제목 후보

### 1순위

**제한된 라벨 환경의 산업 결함 객체 탐지를 위한 DINOv2 표현 기반 능동학습 표본 선택의 학습 효용 검증**

**Validating the Learning Utility of DINOv2 Representation-Based Active Learning Sample Selection for Industrial Defect Object Detection under Limited Label Budgets**

### 2순위

**제한된 라벨 환경의 산업 결함 객체 탐지를 위한 능동학습 표본 선택: DINOv2 표현 기반 선택의 통제 검증**

**Active Learning Sample Selection for Industrial Defect Object Detection under Limited Label Budgets: A Controlled Validation of DINOv2 Representation-Based Selection**

# KSDD2 PatchCore Prospective G1–G4 결과 분석

실행일: 2026-07-26  
Protocol ID: `KSDD2_PATCHCORE_G1_G4_V1`  
최종 판정: **BOUNDED_SCREEN**

## 1. 실행 무결성

- Clean branch: `codex/ksdd2-prospective-run`
- Frozen Git commit: `f10fd92985d63408212edd24e80d741de30ba5f7`
- Freeze 시 worktree: clean
- Freeze 파일: 26개
- KSDD2 ZIP SHA-256:
  `edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8`
- PatchCore backbone:
  `timm/wide_resnet50_2.racm_in1k`
- Backbone state SHA-256:
  `67569d0ff3ed2e26eb2add3e28437ebfc218713bd1052e0b91418207883d677f`
- GPU: NVIDIA GeForce RTX 3070
- 모든 score·selection·decision CSV/JSON: SHA-256 seal 생성
- Development score 확인 전 freeze: 준수
- Query mask 확인 전 selection seal: 준수
- Official test image/mask 접근: 0
- Detector 학습: 0

## 2. 데이터 구성

| 구분 | 이미지 수 | 결함 이미지 수 | 역할 |
|---|---:|---:|---|
| Development | 466 | 53 | G1 signal validity |
| Acquisition | 1,865 | 193 | G2–G4 selection-only |
| Official test | 1,004 | 열지 않음 | final locked |

각 acquisition seed는 initial labeled 100장과 query budget 100장을 사용했다.
Initial set에서 확인된 정상 이미지만 PatchCore memory bank에 들어갔으며
seed별 정상 reference 수는 83–90장이었다.

## 3. G1 — Signal Validity

| 지표 | 사전 기준 | 관측값 | 95% CI | 판정 |
|---|---:|---:|---:|---|
| Image AUROC | ≥ 0.65; CI 하한 ≥ 0.55 | **0.942** | 0.905–0.971 | PASS |
| BBox-count Spearman | ≥ 0.10; CI 하한 ≥ 0 | **0.487** | 0.420–0.546 | PASS |
| Top-100 enrichment | ≥ 1.50×; CI 하한 ≥ 1.20× | **3.957×** | 3.429–4.333× | PASS |

해석: PatchCore image score는 frozen development split에서 결함 존재와
연결성분 수를 유의미하게 rank했다. 이는 selection signal의 타당성이지
detector/segmentation 학습 효용의 증거는 아니다.

## 4. G2 — Target Discovery

| 지표 | 사전 기준 | 결과 | 판정 |
|---|---:|---:|---|
| PatchCore−Random 결함 이미지 차이 | 평균 > 0 | **+88.8장/100장** | PASS |
| Paired bootstrap 95% CI 하한 | > 0 | **86.9–90.6장** | PASS |
| 평균 enrichment | ≥ 1.25× | **10.546×** | PASS |
| 양의 효과 seed 비율 | ≥ 80% | **100% (10/10)** | PASS |

PatchCore top-100에는 seed별 97–100장의 결함 이미지가 포함됐다
(평균 98.5장). Random-100에는 seed별 6–15장, 평균 9.7장이 포함됐다.
Acquisition prevalence 10.35%와 Random 결과가 서로 일치하므로 Random
baseline은 예상 범위에서 동작했다.

## 5. G3 — Task-specific Composition Safety

| 지표 | 사전 기준 | 결과 | 판정 |
|---|---:|---:|---|
| 작은 결함 이미지 차이 | 평균 ≥ 0 | **+5.7장** | PASS |
| Size-bin coverage 차이 | 평균 ≥ −0.25 | **+0.2 bin** | PASS |
| Pool 대비 JSD의 Random 대비 차이 | 평균 ≤ 0.05 | **−0.0146** | PASS |
| Positive excess 최대 size-bin 비중 | ≤ 0.60 | **0.494** | PASS |

해석: PatchCore의 강한 discovery가 frozen small/medium/large 진단에서
Random보다 나쁜 구성 붕괴를 만들지는 않았다. 다만 KSDD2는 단일 의미
결함 클래스이고 생산 lot 정보가 없으므로, 이 결과를 category safety나
source safety로 확장하면 안 된다.

## 6. G4 — Acquisition Reproducibility

| 지표 | 사전 기준 | 결과 | 판정 |
|---|---:|---:|---|
| Seed 수 | 10 | **10** | PASS |
| Positive delta seed 비율 | ≥ 80% | **100%** | PASS |
| Leave-one-seed-out 최소 평균 delta | > 0 | **+88.22장** | PASS |

열 개 candidate set의 pairwise Jaccard는 평균 0.813, 최소 0.739였다.
Candidate union 119장 중 115장(96.6%)이 결함이었고 51장은 모든 seed의
top-100에 반복해서 포함됐다. 이는 initial-reference perturbation에 대한
ranking 안정성을 보여준다.

단, 열 개 seed는 하나의 KSDD2 train pool을 공유한다. 따라서 G4는
독립 production-pool generalization이 아니다.

## 7. 사후 shortcut 무결성 점검

이 절은 **POSTHOC_NOT_PREREGISTERED**이며 G1–G4 판정을 변경하지 않는다.

- Train exact duplicate hash group: 0
- Development/acquisition cross-role exact duplicate group: 0
- Cross-role 63-bit pHash 최소 거리: 8
- pHash 거리 ≤ 4인 cross-role pair: 0
- Numeric ID directional AUROC: 0.503
- 이미지 크기 directional AUROC: 약 0.50
- Mean intensity directional AUROC: 0.540
- Pixel standard deviation directional AUROC: 0.743
- 여섯 개 단순 global feature logistic 5-fold CV AUROC: 0.723
- PatchCore acquisition-pool AUROC: 평균 0.962, 최소 0.960

Exact duplicate, 파일 번호, 해상도 또는 매우 단순한 global statistic만으로
PatchCore의 약 0.96 AUROC를 설명하는 명백한 shortcut은 발견되지 않았다.
그러나 pHash와 global feature 검사는 acquisition-session, semantic texture,
고차원 near-duplicate shortcut을 완전히 배제하지 못한다.

## 8. 최종 결론과 주장 범위

사전 동결된 PatchCore 후보는 KSDD2에서 G1 signal validity, G2 discovery,
G3 task-specific composition safety, G4 acquisition perturbation
reproducibility를 모두 통과했다. 따라서 규약에 따라 **딱 하나의 별도 동결
learner configuration에 대한 G5 bounded screen**만 허용된다.

이 결과가 허용하는 주장은 다음과 같다.

> 사전 동결된 단일 local patch-memory 후보가 KSDD2의 selection-only
> G1–G4를 통과했으며, validity-gated workflow가 처음으로 다음 단계에
> 대한 제한적 실험 권한을 부여했다.

허용되지 않는 주장은 다음과 같다.

- PatchCore의 보편적 selector 우위
- 독립 production pool 일반화
- Detector/segmentation 성능 향상
- Annotation time, GPU hour 또는 비용 절감
- Workflow의 predictive accuracy나 false-stop rate
- Official test 성능

## 9. 발표용 한 장 구성

제목:

> External prospective case — frozen gate가 처음으로 BOUNDED_SCREEN을 허용

왼쪽에는 사전 동결 조건을 배치한다.

- KSDD2 official train only
- Initial 100 / Query 100 / 10 seeds
- Random vs unmodified PatchCore
- Score·selection manifest SHA-256 seal
- Official test locked

가운데에는 세 개의 핵심 결과만 배치한다.

- G1 AUROC **0.942**
- PatchCore−Random **+88.8 defect images / 100**
- Positive effect **10/10 seeds**

오른쪽에는 authorization을 배치한다.

- G1 PASS
- G2 PASS
- G3 PASS
- G4 PASS
- **Next: one separately frozen G5 learner screen**

하단 문장:

> 중요한 결과는 “PatchCore가 이겼다”가 아니라, 사전 규약이 결과에 따라
> downstream 실험 권한을 실제로 부여했다는 점이다.

## 10. 다음 단계

Official test는 계속 열지 않는다. 다음 작업은 G5를 실행하는 것이 아니라
먼저 별도 G5 protocol을 작성하고 freeze하는 것이다. 하나의 segmentation
learner 설정, 동일한 initial/query budget, PatchCore 대 Random, 고정된
training seed와 evaluation split, 사전 정의된 learning-utility threshold를
명시한 뒤에만 bounded learner screen을 실행한다.

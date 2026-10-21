# 제4장 실험 결과 Writing Packets v3

> 교차 장 일관성 감사의 수정 권고를 반영한 버전이다. 수치와 claim 범위는 변경하지 않았으며, 제4장의 관찰 역할을 강화했다.

## 사용 원칙

이 문서는 제4장 4.1–4.5를 실제로 집필하기 직전의 문단 설계와 증거 배치를 고정한다. 제4장에서는 관찰된 수치, 비교 방향, 신뢰구간, 분석의 시간적 지위, 사전 기준 충족 여부만 쓴다. 원인 설명, 일반화, 방법론적 시사점, 산업적 의미는 제5장으로 넘긴다.

- 4.1에는 최종 claim을 직접 배치하지 않는다.
- 4.2–4.5는 각각 EP02, EP03, EP04, EP05의 결과 절로 고정한다.
- EP01의 통합 문장은 제4장 본문이나 소절 제목으로 사용하지 않는다.
- 데이터셋·과업·지표가 다른 결과를 합산하거나 공통 효과크기로 환산하지 않는다.
- `translation branch`, `Gate PASS`, `positive endpoint` 같은 내부 용어는 각각 `평가 분기`, `사전에 정한 평가 기준 충족`, `양의 방향으로 나타난 선택 단계 결과`로 쓴다.

## 4.1 후보 선택 신호별 선택 단계 관찰

### 절의 목적

후속 절의 결과를 읽기 위해 후보 선택 신호가 각 평가 분기에서 어떤 선택 단계 native endpoint로 측정되었고 Random 대비 어떤 차이가 기록되었는지를 제시한다. 이 절은 선택 단계 측정 결과만 제시하며, 다운스트림 예측력 또는 selector 우월성은 판단하지 않는다.

### 다룰 claim_id

- 직접 배치 claim 없음
- EP02–EP05의 결과를 읽기 위한 평가 전제만 제공
- EP01은 언급하지 않음

### 첫 문단에서 바로 제시할 핵심 관찰 문장

1. 각 평가 분기의 후보 선택 신호는 사전에 정한 native endpoint에서 Random 대비 차이로 기록되었으며, 서로 다른 단위의 효과는 합산하지 않았다.
2. GC10 q20에서는 희소 클래스 이미지 발견량이 Random보다 평균 `+2.720`장 많았고 95% 신뢰구간은 `[+2.455,+2.980]`이었으며, KSDD2의 100-image query에서는 positive defect image가 평균 `+88.8`장 많았고 신뢰구간은 `[+86.9,+90.6]`이었다.
3. MPDD와 VisA의 선택 단계 평가에서는 anomaly image 차이가 각각 `+6.245`와 `+14.480`으로 관찰되었으며, 구성 기준 결과는 4.2에서 별도로 제시한다.

### 반드시 포함할 핵심 수치

- GC10 q20 rare-class image discovery: `+2.720`, 95% CI `[+2.455,+2.980]`
- KSDD2 100-image query positive defect image: `+88.8`, CI `[+86.9,+90.6]`
- MPDD anomaly image: `+6.245`, CI `[+5.885,+6.605]`
- VisA anomaly image: `+14.480`, CI `[+14.195,+14.770]`
- MPDD·VisA 평가 단위: 각 `200` paired fixed-pool perturbations

이 수치들은 분기별 선택 단계 측정값으로만 병렬 기재하고 평균·순위·통합 효과를 계산하지 않는다.

### 연결할 figure/table

- **표 4-1(신규 본문 표):** `데이터셋 × 후보 선택 신호 × query budget × 선택 단계 native endpoint × Random 대비 차이 × 신뢰구간 × 평가 단위`
- 별도 통합 그림은 배치하지 않는다. `docs/translation_atlas/translation_atlas_main_figure.png`는 post-hoc 통합 자료이므로 제4장 4.1의 주 그림으로 사용하지 않는다.

### 문단 전개 순서

1. **문단 1 — 판독 규칙:** 후보 신호의 선택 단계 측정 결과를 각 분기의 native endpoint와 Random 대비 차이로 제시한다고 밝힌다.
2. **문단 2 — 분기별 관찰:** GC10 q20, KSDD2, MPDD, VisA의 선택 단계 차이와 신뢰구간을 데이터셋별로 나열한다.
3. **문단 3 — 표 제시:** 표 4-1로 데이터셋, budget, endpoint, 반복 단위를 한 화면에 정리하되 수치적 pooling을 하지 않는다.
4. **문단 4 — 다음 절 연결:** 선택 단계에서 함께 측정된 발견량과 범주 구성 결과를 4.2에서 제시한다고만 안내한다.

### 허용되는 표현

- “Random 대비 선택 단계 지표가 양의 방향을 보였다.”
- “각 결과는 해당 분기의 native endpoint와 평가 단위로 제시하였다.”
- “서로 다른 데이터셋의 효과크기는 합산하지 않았다.”
- “구성 기준과 다운스트림 지표는 후속 절에서 별도로 제시한다.”

### 금지되는 표현

- “후보 선택 신호의 타당성이 검증되었다.”
- “선택 단계 성능이 다운스트림 효용을 예측했다.”
- “해당 selector가 Random보다 우월하다.”
- “선택 단계 gain은 학습으로 이어지지 않는다.”
- “여러 산업 데이터셋에서 일반적으로 재현되었다.”
- “평가 framework가 검증되었다.”

### 제5장으로 넘길 해석

- 선택 단계에서 양의 방향으로 나타난 결과와 다운스트림 기준 충족의 관계
- 분기별 endpoint가 서로 다른 이유와 이를 함께 해석하는 범위
- post-hoc 통합, 이질적 분기 및 독립 production pool 부재의 의미

### 한 문장 요약

GC10 q20, KSDD2, MPDD 및 VisA에서는 각 분기의 선택 단계 native endpoint에 Random 대비 양의 차이가 기록되었고, 수치는 분기별 단위로 제시되었다.

## 4.2 결함 발견량과 선택 집합 구성

### 절의 목적

MPDD와 VisA의 동일한 선택 단계 평가에서 결함 이미지 발견량과 선택 집합의 범주 구성 결과를 나란히 보고한다. Learner outcome이나 두 지표의 방법론적 의미는 해석하지 않는다.

### 다룰 claim_id

- EP02

### 첫 문단에서 바로 제시할 핵심 관찰 문장

1. MPDD에서 anomaly image 차이는 `+6.245`였고 신뢰구간은 `[+5.885,+6.605]`였으나, product-category coverage 차이는 `-0.325`로 사전 noninferiority 기준 `-0.25`를 충족하지 못했다.
2. VisA에서 anomaly image 차이는 `+14.480`이었고 신뢰구간은 `[+14.195,+14.770]`이었으나, object-category coverage 차이는 `-4.110`이었고 coverage check는 충족되지 않았다.
3. 두 결과는 각각 `200`회의 paired fixed-pool perturbation에서 산출되었다.

### 반드시 포함할 핵심 수치

- MPDD anomaly images: `+6.245`, CI `[+5.885,+6.605]`
- MPDD product-category coverage: `-0.325`; 사전 noninferiority 기준 `-0.25`; 기준 미충족
- VisA anomaly images: `+14.480`, CI `[+14.195,+14.770]`
- VisA object-category coverage: `-4.110`; coverage check 미충족
- 반복 단위: 데이터셋별 `200` paired fixed-pool perturbations

### 연결할 figure/table

- **그림:** `docs/notion_professor_update_20260803/assets/figure06_discovery_composition_contrast.png`
- **표 4-2(신규 본문 표):** `MPDD·VisA의 anomaly discovery와 category coverage 대비표`
- 그림 source 설명: `docs/notion_professor_update_20260803/figure_manifest.csv`

### 문단 전개 순서

1. **문단 1 — 비교 설계:** 두 데이터셋에서 candidate와 Random을 같은 fixed pool의 paired perturbation으로 비교했다는 평가 단위를 제시한다.
2. **문단 2 — MPDD:** anomaly image 차이와 CI를 먼저 제시한 뒤 product-category coverage 차이, 기준값 및 미충족 여부를 쓴다.
3. **문단 3 — VisA:** anomaly image 차이와 CI를 먼저 제시한 뒤 object-category coverage 차이와 check 미충족 여부를 쓴다.
4. **문단 4 — 병렬 표시:** 그림과 표에서 discovery와 coverage를 같은 행에 배치하고, learner outcome은 이 평가 범위에 포함되지 않았다고 명시한다.
5. **문단 5 — 관찰 요약:** 두 데이터셋에서 발견량 증가와 구성 기준 미충족이 함께 기록되었다는 범위 한정 문장으로 끝낸다.

### 허용되는 표현

- “결함 이미지 발견량은 증가했으나 범주 구성 기준은 충족하지 못했다.”
- “선택 단계의 두 측정 결과가 같은 방향의 판정을 보이지 않았다.”
- “각 `200`회 반복은 같은 fixed pool에서의 paired perturbation이었다.”
- “Learner outcome은 이 평가에서 관찰하지 않았다.”

### 금지되는 표현

- “MPDD와 VisA에서 학습이 실패했다.”
- “DINO가 category collapse를 유발했다.”
- “200개의 독립 산업 pool에서 재현되었다.”
- “구성 기준 미충족 때문에 downstream 성능이 저하되었다.”
- “발견량과 구성 안전성은 반드시 별도로 평가해야 한다.”

### 제5장으로 넘길 해석

- Discovery와 composition을 동일 성공 지표로 취급할 수 없는 이유
- Category coverage가 실제 learner utility와 어떤 관계를 갖는지에 대한 미확인 범위
- Fixed-pool perturbation과 독립 production pool 재현성의 차이

### 한 문장 요약

MPDD와 VisA에서는 anomaly image 차이가 각각 `+6.245`와 `+14.480`이었지만, 두 데이터셋 모두 선택 집합의 범주 구성 기준을 충족하지 못했다.

## 4.3 고정 선택 집합 재학습과 acquisition 반복 결과

### 절의 목적

NEU-DET에서 결과 관찰 후 선택된 favorable fixed set의 재학습 결과를 `POSTHOC_DESCRIPTIVE`로 제시하고, selector를 여러 acquisition seed에서 다시 실행한 반복 결과를 별도 관찰로 제시한다. 두 재현성 대상을 구분하는 해석은 제5장으로 넘긴다.

### 다룰 claim_id

- EP03

### 첫 문단에서 바로 제시할 핵심 관찰 문장

1. 결과 관찰 후 선택된 favorable fixed set의 `POSTHOC_DESCRIPTIVE` 재학습에서는 mAP50-95 차이가 `+0.016236`이었고 `5/5` 학습 seed에서 양의 차이가 기록되었다.
2. 선택 과정을 `10`개 acquisition seed에서 반복한 평가에서는 평균 mAP50-95 차이가 `+0.007019`, 신뢰구간이 `[-0.005211,+0.019678]`, 양의 차이가 나타난 반복이 `7/10`이었다.
3. Acquisition 반복의 exact sign-flip 검정은 `p=0.322266`이었고, 평균 차이는 사전 기준 `+0.010`을 충족하지 못했다.

### 반드시 포함할 핵심 수치

- 고정 선택 집합 mAP50-95 delta: `+0.016236`
- 고정 선택 집합 training-seed wins: `5/5`
- Acquisition 반복 수: `10`
- Acquisition 반복 mAP50-95 delta: `+0.007019`
- Descriptive CI: `[-0.005211,+0.019678]`
- Acquisition-seed wins: `7/10`
- Exact sign-flip: `p=0.322266`
- 결과 확인 전에 고정한 최소 기준: `+0.010`; 미충족
- 시간적 지위: 결과 관찰 후 선택된 favorable fixed set이며, fixed-set arm은 `POSTHOC_DESCRIPTIVE`

### 연결할 figure/table

- **그림:** `docs/notion_professor_update_20260803/assets/figure05_neu_two_reproducibilities.png`
- **표 4-3(신규 본문 표):** `결과 관찰 후 선택된 favorable fixed set의 사후 재학습과 acquisition 반복 비교`
- 그림 source 설명: `docs/notion_professor_update_20260803/figure_manifest.csv`

### 문단 전개 순서

1. **문단 1 — 시간적 지위와 두 평가 단위:** Fixed-set arm이 결과 관찰 후 선택된 favorable 사례의 `POSTHOC_DESCRIPTIVE` 분석임을 먼저 밝히고, training-seed 반복과 selector 재실행을 포함한 acquisition-seed 반복을 구분해 제시한다.
2. **문단 2 — 고정 집합 관찰:** Favorable fixed set의 사후 선택과 `POSTHOC_DESCRIPTIVE` 지위를 다시 표시한 뒤 mAP 차이와 `5/5`를 보고한다.
3. **문단 3 — Acquisition 반복 결과:** 평균 차이, CI, `7/10`, `p`값을 순서대로 보고한다.
4. **문단 4 — 사전 기준 판정:** 평균 `+0.007019`와 사전 `+0.010` 기준을 직접 대조해 미충족 여부를 쓴다.
5. **문단 5 — 병렬 요약:** 그림과 표를 참조하며 두 평가 결과를 같은 용어로 합치지 않고 관찰값만 요약한다.

### 허용되는 표현

- “결과 관찰 후 선택된 favorable fixed set의 사후 기술 분석에서 5개 학습 seed 모두 양의 차이가 기록되었다.”
- “Acquisition 반복의 신뢰구간은 0을 포함했다.”
- “Acquisition 반복은 사전 `+0.010` 기준을 충족하지 못했다.”
- “고정 집합 결과는 사후 기술 분석이다.”

### 금지되는 표현

- “NEU에서 visual diversity가 재현에 실패했다.”
- “Training-seed 안정성은 무의미하다.”
- “고정 집합 결과가 confirmatory PASS 또는 재현 성공이었다.”
- “10 acquisition seeds가 independent pool 일반화를 입증했다.”
- “두 재현성 개념은 반드시 분리해야 한다.”

### 제5장으로 넘길 해석

- Training-seed 안정성과 acquisition reproducibility가 서로 다른 평가 대상인 이유
- Favorable fixed set의 사후 선택이 해석 범위에 미치는 영향
- 같은 물리적 pool 안의 acquisition 반복과 독립 pool 일반화의 차이

### 한 문장 요약

NEU-DET에서 결과 관찰 후 선택된 favorable fixed set의 `POSTHOC_DESCRIPTIVE` 재학습은 mAP 차이 `+0.016236`과 `5/5` 양의 결과를 보였고, acquisition 반복의 평균 차이 `+0.007019`는 신뢰구간이 0을 포함했으며 사전 `+0.010` 기준을 충족하지 못했다.

## 4.4 전체 성능과 희소 클래스 성능

### 절의 목적

GC10 q20의 동일한 다운스트림 평가에서 전체 detector 지표와 frequent·rare class-group 지표를 함께 보고한다. 지표가 상반된 방향을 보인 원인이나 산업 안전상의 의미는 해석하지 않는다.

### 다룰 claim_id

- EP04

### 첫 문단에서 바로 제시할 핵심 관찰 문장

1. GC10 q20에서 전체 mAP50-95 차이는 `+0.017378`이었고 acquisition-seed 신뢰구간은 `[+0.000580,+0.036332]`였으며, `5`개 acquisition seed 중 `4`개에서 양의 차이가 기록되었다.
2. 전체 recall 차이는 `+0.035381`, frequent-class macro AP 차이는 `+0.033345`였다.
3. 같은 평가의 rare-class macro AP 차이는 `-0.019877`이었고 사전 rare-class 기준은 충족되지 않았다.

### 반드시 포함할 핵심 수치

- Overall mAP50-95 delta: `+0.017378`
- Acquisition-seed CI: `[+0.000580,+0.036332]`
- Positive acquisition seeds: `4/5`
- Recall delta: `+0.035381`
- Frequent-class macro AP delta: `+0.033345`
- Rare-class macro AP delta: `-0.019877`
- Rare macro AP check: 사전 기준 미충족
- 평가 구조: `5` acquisition seeds × 각 `3` training seeds

### 연결할 figure/table

- **그림:** `docs/notion_professor_update_20260803/assets/figure04_gc10_endpoint_conflict.png`
- **표 4-4(신규 본문 표):** `GC10 q20 overall·frequent·rare endpoint 비교표`
- 그림 source 설명: `docs/notion_professor_update_20260803/figure_manifest.csv`

### 문단 전개 순서

1. **문단 1 — 평가 구조:** `5` acquisition seeds와 각 `3` training seeds에서 candidate와 Random을 비교했음을 제시한다.
2. **문단 2 — 전체 성능:** mAP50-95 차이, CI, `4/5` 결과와 recall 차이를 보고한다.
3. **문단 3 — 클래스 그룹:** Frequent macro AP와 rare macro AP의 point estimate를 같은 문단과 표에서 제시한다.
4. **문단 4 — 기준 판정:** Rare macro AP 기준 미충족을 명시하고 별도 CI가 제시되지 않았음을 함께 표시한다.
5. **문단 5 — 그림 요약:** 전체·frequent·rare 지표의 방향을 그림에서 확인하도록 안내하고 관찰 문장으로 끝낸다.

### 허용되는 표현

- “전체 mAP50-95와 recall은 양의 차이를 보였다.”
- “Frequent macro AP는 양수였고 rare macro AP는 음수였다.”
- “희소 클래스 기준은 충족하지 못했다.”
- “Rare macro AP는 별도 CI가 없는 point estimate다.”

### 금지되는 표현

- “Overall mAP는 rare class를 항상 희생한다.”
- “DINO가 희소 클래스를 악화시켰다.”
- “Rare-class harm가 통계적으로 확정되었다.”
- “산업 안전 위험이 입증되었다.”
- “평균 지표와 subgroup 지표는 반드시 함께 보고해야 한다.”

### 제5장으로 넘길 해석

- 평균 detector 지표와 희소 클래스 지표를 함께 해석해야 하는 이유
- Rare-class point estimate의 불확실성과 별도 CI 부재
- 한 GC10 설정의 상반된 지표 방향을 다른 데이터셋으로 일반화할 수 없는 이유
- 지표 충돌이 산업 안전 또는 인과적 harm를 의미하는지에 대한 미확인 범위

### 한 문장 요약

GC10 q20에서는 전체 mAP50-95가 `+0.017378` 증가한 반면 rare-class macro AP는 `-0.019877` 감소했고, 희소 클래스 기준은 충족되지 않았다.

## 4.5 KSDD2 Dice 차이와 사전 최소효과 기준 판정

### 절의 목적

KSDD2의 제한적 다운스트림 평가에서 Dice 차이와 신뢰구간, 결과 확인 전에 고정한 최소효과 기준, 보조 지표·정상 이미지 FPR 점검을 함께 보고한다. 기준의 산업적 적절성이나 실용적 의미는 판단하지 않는다.

### 다룰 claim_id

- EP05

### 첫 문단에서 바로 제시할 핵심 관찰 문장

1. KSDD2 제한적 다운스트림 평가는 요구된 `60/60` runs를 완료했다.
2. Macro Dice 차이는 `+0.037565`였고 신뢰구간은 `[+0.019581,+0.057029]`로 하한이 0을 초과했다.
3. 평균 Dice 차이는 결과 확인 전에 고정한 최소 기준 `+0.05`를 충족하지 못했다.

### 반드시 포함할 핵심 수치

- 완료 runs: `60/60`
- Macro Dice delta: `+0.037565`
- CI: `[+0.019581,+0.057029]`
- 결과 확인 전에 고정한 mean Dice 기준: `+0.05`; 미충족
- Leave-one-seed-out minimum: `+0.031234`
- Positive acquisition-seed fraction: `0.8`
- Pixel AP delta: `+0.049824`
- PatchCore normal FPR: `0.160129`
- FPR delta: `+0.010573`

### 연결할 figure/table

- **그림:** `docs/notion_professor_update_20260803/assets/figure08_ksdd2_positive_below_threshold.png`
- **표 4-5(신규 본문 표):** `KSDD2 primary Dice·보조 지표·정상 이미지 FPR 점검 결과표`
- 그림 source 설명: `docs/notion_professor_update_20260803/figure_manifest.csv`

### 문단 전개 순서

1. **문단 1 — 평가 범위와 완료 상태:** 제한적 다운스트림 평가의 범위와 `60/60` 완료를 제시한다.
2. **문단 2 — Primary estimate:** Macro Dice 차이와 CI를 제시하고 CI 하한이 0을 초과했음을 쓴다.
3. **문단 3 — 사전 기준 판정:** 평균 `+0.037565`와 사전 `+0.05` 기준을 직접 대조해 미충족을 명시한다.
4. **문단 4 — Supporting checks:** Leave-one-seed-out minimum, positive seed fraction, pixel AP 차이를 순서대로 보고한다.
5. **문단 5 — 정상 이미지 FPR 점검:** PatchCore normal FPR과 FPR delta를 기록하고 표 4-5를 참조한다.
6. **문단 6 — 관찰 요약:** 양의 추정 방향과 사전 평균 기준 미충족을 동시에 보존한 문장으로 끝낸다.

### 허용되는 표현

- “Dice 차이는 양의 방향이었고 CI 하한은 0을 초과했다.”
- “평균 Dice 차이는 사전 `+0.05` 기준을 충족하지 못했다.”
- “제한적 다운스트림 평가에서 `60/60` runs가 완료되었다.”
- “보조 지표와 정상 이미지 FPR 관찰값은 각각 다음과 같았다.”

### 금지되는 표현

- “Translation effect가 없었다.”
- “KSDD2 평가가 성공했다” 또는 “KSDD2 평가가 실패했다”라는 무조건적 이분법
- “PatchCore 우월성이 입증되었다.”
- “실용적 개선이 입증되었다.”
- “`+0.05`가 산업적으로 최적인 threshold다.”
- “Official test에서 개선되었다.”

### 제5장으로 넘길 해석

- 통계적으로 양의 방향인 추정치와 사전 최소효과 기준 충족의 차이
- `+0.05` 기준의 외부 stakeholder·비용 calibration 부재
- 한 acquisition pool과 한 learner에 한정된 결과의 일반화 범위
- Decision label을 success 또는 no-effect로 단순화하면 안 되는 이유

### 한 문장 요약

KSDD2의 macro Dice 차이는 `+0.037565`이고 신뢰구간 하한은 0을 초과했지만, 평균 차이는 사전에 고정한 `+0.05` 기준을 충족하지 못했다.

## 제4장 각 절의 첫 문장 후보

- **4.1:** 각 평가 분기의 후보 선택 신호는 사전에 정한 native endpoint에서 Random 대비 차이로 기록되었으며, 서로 다른 단위의 효과는 합산하지 않았다.
- **4.2:** MPDD에서 anomaly image 차이는 `+6.245`였고 신뢰구간은 `[+5.885,+6.605]`였으나, product-category coverage 차이는 `-0.325`로 사전 noninferiority 기준 `-0.25`를 충족하지 못했다.
- **4.3:** 결과 관찰 후 선택된 favorable fixed set의 `POSTHOC_DESCRIPTIVE` 재학습에서는 mAP50-95 차이가 `+0.016236`이었고 `5/5` 학습 seed에서 양의 차이가 기록되었다.
- **4.4:** GC10 q20에서 전체 mAP50-95 차이는 `+0.017378`이었고 acquisition-seed 신뢰구간은 `[+0.000580,+0.036332]`였으며, `5`개 acquisition seed 중 `4`개에서 양의 차이가 기록되었다.
- **4.5:** KSDD2 제한적 다운스트림 평가는 요구된 `60/60` runs를 완료했다.

## 제4장에서 쓰면 안 되는 해석 문장 목록

- “Selection gain은 learning으로 이어지지 않는다.”
- “선택 단계 지표는 다운스트림 효용의 충분조건이 아니다.”
- “후보 선택 신호 또는 특정 selector의 우월성이 검증되었다.”
- “평가 framework 또는 Gate의 예측 정확도가 검증되었다.”
- “Discovery와 composition은 반드시 별도로 평가해야 한다.”
- “Training-seed 안정성과 acquisition reproducibility는 대체 불가능한 개념이다.”
- “평균 성능과 희소 클래스 지표는 반드시 함께 보고해야 한다.”
- “Overall mAP 개선은 희소 클래스를 희생한다.”
- “Rare-class harm 또는 산업 안전 위험이 입증되었다.”
- “KSDD2에서 실용적인 개선이 입증되었다” 또는 “효과가 없었다.”
- “사전 `+0.05` 기준은 산업적으로 최적이다.”
- “같은 fixed pool의 반복이 independent production pool 일반화를 입증했다.”
- “관찰 결과의 원인은 representation, clustering 또는 selector 설계다.”
- “현재 결과는 비용 절감을 입증한다.”

## 제5장으로 넘겨야 할 논의 포인트 목록

- **EP01 / 5.1·5.6:** 선택 단계와 다운스트림 결과가 모두 관찰된 세 평가 분기의 제한적 통합과 일반화 경계
- **EP02 / 5.2:** Discovery와 selected-set composition을 동일한 성공 지표로 취급할 수 없는 이유
- **EP03 / 5.3:** 고정 집합의 training-seed 안정성과 selector를 다시 실행한 acquisition reproducibility를 구분해야 하는 이유
- **EP04 / 5.4:** 전체 detector 지표와 희소 클래스 지표의 상반된 방향을 함께 해석하는 방법과 rare AP 불확실성
- **EP05 / 5.6:** 통계적 양의 방향, 사전 최소효과 기준 충족, 외부 실용성을 구분하는 이유
- **공통 / 5.7:** Post-hoc 통합, 이질적 평가 분기, independent production pool 부재, learner outcome이 없는 분기, threshold calibration 부재


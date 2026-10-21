# 제5장 논의 Writing Packets v3

## 사용 원칙

이 문서는 제5장 5.1–5.7을 실제로 집필하기 직전의 해석 범위와 문장 경계를 고정한다. 제4장의 수치와 기준 판정은 필요한 만큼만 교차참조하며, 새 계산·새 효과크기·새 claim을 추가하지 않는다.

> **v3 봉합 원칙:** 5.1은 세 분기의 이질성과 미충족 지점 차이만 해석하고, 비용·annotation-time·예산 한계는 5.5에 집중하며, 5.6은 KSDD2 핵심 Dice 판정과 표 4-5 교차참조만 유지한다.

- 5.1은 EP01을 **결과가 모두 관찰된 세 평가 분기**에 한정해 통합한다.
- 5.2–5.4는 각각 EP02–EP04의 의미를 해석한다.
- 5.5는 이미지 예산 기반 평가의 해석 한계이며 독립적인 연구 기여로 제시하지 않는다.
- 5.6은 EP05의 판정 구조와 EP01의 일반화 제한을 연결한다.
- 5.7은 EP01–EP05 전체에 적용되는 방화벽을 명시한다.
- EP01의 최종 통합 문장 자체는 제6장 6.1로 넘기고, 제5장에서는 그 문장이 성립하는 범위와 제한 조건을 설명한다.
- Selector 우월성, 검증된 framework, Gate 예측 정확도, 비용 절감 입증, 산업 benchmark, 독립 production pool 일반화는 주장하지 않는다.

## 5.1 평가 분기별 선택 단계 결과와 다운스트림 기준 충족의 관계

### 절의 목적

선택 단계와 다운스트림 결과가 모두 관찰된 세 평가 분기의 미충족 지점이 어떻게 달랐는지와 평가 조건의 이질성을 해석한다. 제6장 6.1의 통합 결론과 동등한 관계 문장은 두지 않으며, 세 분기의 결과를 합산하거나 보편적인 선택–학습 법칙으로 확대하지 않는다.

### 연결되는 claim_id

- EP01: 통합 해석
- EP06: 부록 A의 보조 평가 분기이며 본 절 말미에서 1문장 교차참조만 허용

### 제4장에서 가져올 직접 결과 요약

- **4.1:** GC10 q20과 KSDD2의 선택 단계 native endpoint는 각각 Random 대비 양의 방향을 보였다. 분기별 지표는 서로 다른 단위로 유지되었고 합산되지 않았다.
- **4.4 / GC10 q20:** 전체 mAP50-95 차이는 `+0.017378`이었으나 rare-class macro AP 차이는 `-0.019877`이었고 희소 클래스 기준은 충족되지 않았다.
- **4.5 / KSDD2:** Macro Dice 차이는 `+0.037565`, CI `[+0.019581,+0.057029]`였으나 결과 확인 전에 고정한 평균 `+0.05` 기준은 충족되지 않았다.
- **부록 A / GC10 K40/140:** Coverage 차이는 `+0.015`였으나 mAP50-95 차이는 `-0.001678`, CI `[-0.017424,+0.018332]`였고 rare AP와 recall 기준도 충족되지 않았다. 이는 EP06의 보조 결과이며 제4장의 핵심 소절로 승격하지 않는다.
- 통합 레지스트리에서 결과가 모두 관찰된 평가 분기는 `3`개이고 사전에 정한 전체 다운스트림 기준 충족은 `0/3`으로 기록되었다. 이 수치는 실패 확률이 아니라 현재 세 분기의 기술적 집계다.

### 이 절에서 말할 수 있는 해석 문장

1. 세 평가 분기의 다운스트림 미충족 지점은 희소 클래스 지표, 학습 방향, 결과 확인 전에 고정한 평균 최소효과 기준으로 서로 달랐다.
2. 데이터셋, task, selector, budget, learner, threshold 및 시간적 지위가 이질적이므로, 분기별 native metric과 판정 근거를 유지한 채 해석해야 한다.
3. `0/3`은 이질적인 세 평가 분기를 요약한 기술적 결과이며, 새로운 데이터에서의 실패 확률이나 selector의 일반적 성능을 뜻하지 않는다.
4. GC10 K40/140은 EP01의 범위를 보조하지만 EP04와 논리적 기능이 겹치고 mAP 신뢰구간이 0을 포함하므로 부록 근거 이상으로 확대하지 않는다.

### 이 절에서 말하면 안 되는 과장/일반화 문장

- “Selection gain은 learning으로 이어지지 않는다.”
- “선택 단계 신호는 다운스트림 utility의 충분조건이 아니다”를 범위 제한 없이 보편 명제로 제시한다.
- “Full translation 실패율은 100%다.”
- “모든 selector가 다운스트림에서 실패했다.”
- “Gate가 좋은 selector를 정확히 예측하거나 걸러냈다.”
- “세 분기의 효과를 종합하면 공통 음의 효과가 있다.”

### 필요한 figure/table 또는 교차참조

- `docs/translation_atlas/translation_atlas_main_figure.png`
- `docs/translation_atlas/four_fates_taxonomy.csv`
- **표 5-1(신규 본문 표):** `평가 분기 × 선택 단계 관찰 × 다운스트림 기준 판정 × 시간적 지위`
- 제4장 4.1, 4.4, 4.5 교차참조
- 부록 A 교차참조 문장: “추가 GC10 K40/140 보조 평가는 부록 A에 제시하였다.”
- 통합 그림과 표에는 `post-hoc integration`, `native metric 유지`, `cross-dataset numeric pooling 없음`을 각주로 명시한다.

### 남는 한계 또는 예상 반론

- 세 분기는 데이터셋, task, selector, budget, learner, threshold 및 retrospective/prospective 지위가 다르다.
- 결과가 모두 관찰된 분기가 `3`개뿐이므로 `0/3`을 비율 추정이나 일반 법칙으로 해석할 수 없다.
- GC10 K40/140의 mAP 신뢰구간은 0을 포함하며 noninferiority는 통과했다.
- MPDD·VisA·DeepPCB처럼 learner outcome이 없는 분기는 이 통합의 실패 사례로 포함할 수 없다.

### 한 문장 요약

5.1은 세 평가 분기의 서로 다른 미충족 지점과 이질적 평가 조건을 정리하고, native metric 비합산과 `0/3`의 비율 해석 금지를 명시하는 데 한정한다.

## 5.2 결함 발견량과 구성 기준의 비일치

### 절의 목적

MPDD와 VisA의 선택 단계에서 결함 이미지 발견량과 선택 집합의 범주 구성 기준이 서로 다른 판정을 만든 의미를 해석한다. 후속 learner가 실행되지 않았다는 경계를 유지한다.

### 연결되는 claim_id

- EP02

### 제4장에서 가져올 직접 결과 요약

- **4.2 / MPDD:** Anomaly image 차이는 `+6.245`, CI `[+5.885,+6.605]`였으나 product-category coverage 차이는 `-0.325`로 사전 noninferiority 기준 `-0.25`를 충족하지 못했다.
- **4.2 / VisA:** Anomaly image 차이는 `+14.480`, CI `[+14.195,+14.770]`였으나 object-category coverage 차이는 `-4.110`이었고 coverage check는 충족되지 않았다.
- 두 평가는 각각 같은 fixed pool 안의 `200`회 paired perturbation으로 수행되었으며 learner outcome은 관찰하지 않았다.

### 이 절에서 말할 수 있는 해석 문장

1. MPDD와 VisA의 관찰은 결함 이미지 발견량과 선택 집합의 범주 구성이 선택 단계 안에서도 서로 다른 평가 대상임을 보여준다.
2. 동일한 query set에서 target image yield가 증가하더라도 범주 coverage 기준이 자동으로 충족되는 것은 아니었다.
3. 따라서 이 두 데이터셋의 선택 단계 성능은 discovery와 composition을 각각 보고해야 손실 없이 기술할 수 있다.
4. 이 해석은 선택 집합 구성에 한정되며, 후속 learner 성능이나 composition 기준의 인과적 유용성을 확정하지 않는다.

### 이 절에서 말하면 안 되는 과장/일반화 문장

- “MPDD와 VisA에서 학습이 실패했다.”
- “DINO가 category collapse를 유발했다.”
- “Category coverage 미충족이 downstream 저하의 원인이다.”
- “200개의 독립 산업 pool에서 재현되었다.”
- “모든 산업 결함 Active Learning에서 discovery와 composition은 충돌한다.”
- “Global representation 때문에 범주 구성이 악화되었다.”

### 필요한 figure/table 또는 교차참조

- 제4장 4.2 및 표 4-2 교차참조
- `docs/notion_professor_update_20260803/assets/figure06_discovery_composition_contrast.png`
- 5.2에서는 새 결과 그림을 만들지 않고 4.2 그림·표를 재참조한다.

### 남는 한계 또는 예상 반론

- MPDD와 VisA에서는 learner outcome이 존재하지 않으므로 composition 차이의 downstream 효용을 확인할 수 없다.
- `200`회 반복은 독립 production pool이 아니라 동일 fixed pool에서 initial set을 달리한 paired perturbation이다.
- Category coverage가 실제 작업 가치나 오류 비용과 어떻게 연결되는지는 측정하지 않았다.

### 한 문장 요약

MPDD와 VisA의 결과는 결함 이미지 발견량과 선택 집합의 범주 구성을 별도 평가 대상으로 해석할 근거를 제공하지만, 후속 학습 효용까지 확장하지는 않는다.

## 5.3 고정 선택 집합의 training-seed 안정성과 acquisition reproducibility의 구분

### 절의 목적

NEU-DET의 고정 선택 집합의 training-seed 안정성(fixed-set training stability)과 동일 fixed pool에서의 acquisition reproducibility가 각각 어떤 변동성을 다루는지 해석한다. 고정 집합의 양의 결과를 selector의 confirmatory 우위로 읽지 않는다.

### 연결되는 claim_id

- EP03

### 제4장에서 가져올 직접 결과 요약

- **4.3 / 고정 집합:** `POSTHOC_DESCRIPTIVE` 지위의 mAP50-95 차이는 `+0.016236`이었고 `5/5` training seeds에서 양의 차이가 나타났다.
- **4.3 / Acquisition 반복:** `10`개 acquisition seeds에서 평균 mAP50-95 차이는 `+0.007019`, CI `[-0.005211,+0.019678]`, 양의 반복은 `7/10`, exact sign-flip은 `p=0.322266`이었다.
- Acquisition 반복의 평균 차이는 결과 확인 전에 고정한 `+0.010` 기준을 충족하지 못했다.

### 이 절에서 말할 수 있는 해석 문장

1. 고정 선택 집합의 training-seed 반복은 선택 집합을 조건으로 한 학습 초기화·최적화 변동성을 다룬다.
2. Acquisition-seed 반복은 selector를 다시 실행해 선택 집합이 달라지는 과정까지 포함하므로 고정 집합 재학습과 다른 재현성 대상을 측정한다.
3. 한 고정 집합에서 `5/5` 양의 차이가 관찰되었다는 사실만으로 선택 과정을 다시 수행했을 때 유사한 학습 효과가 재현된다고 해석할 수 없었다.
4. 이 구분은 training-seed 안정성을 무의미하게 만드는 것이 아니라 각 결과가 답하는 질문을 제한한다.

### 이 절에서 말하면 안 되는 과장/일반화 문장

- “NEU에서 visual diversity가 재현에 실패했다.”
- “Training-seed 안정성은 무의미하다.”
- “고정 집합 결과는 confirmatory PASS였다.”
- “10 acquisition seeds가 독립 production pool 일반화를 입증했다.”
- “Acquisition 변동성이 성능 저하의 원인이다.”
- “모든 Active Learning 연구에서 fixed-set 재학습은 쓸모가 없다.”

### 필요한 figure/table 또는 교차참조

- 제4장 4.3 및 표 4-3 교차참조
- `docs/notion_professor_update_20260803/assets/figure05_neu_two_reproducibilities.png`
- 그림 캡션 또는 본문에 fixed-set arm의 `POSTHOC_DESCRIPTIVE` 지위를 유지한다.

### 남는 한계 또는 예상 반론

- 고정 집합은 결과를 확인한 뒤 선택된 favorable 사례이므로 공정한 confirmatory arm이 아니다.
- Acquisition 반복도 동일한 물리적 pool 안에서 수행되어 외부 site·lot·production pool 재현성을 검증하지 않는다.
- Acquisition 반복에 별도의 selection-primary endpoint가 없어 변동의 원인을 단계별로 분해할 수 없다.

### 한 문장 요약

NEU-DET 사례에서 고정 선택 집합의 training-seed 안정성(fixed-set training stability)과 동일 fixed pool에서 selector 재실행을 포함한 acquisition reproducibility는 서로 다른 조건부 질문에 답하는 결과로 해석해야 한다.

## 5.4 전체 성능과 희소 클래스 성능의 충돌 가능성

### 절의 목적

GC10 q20에서 전체 detector 지표와 희소 클래스 지표가 상반된 방향을 보인 의미를 해석한다. 이를 일반적인 희소 클래스 harm나 산업 안전 위험의 인과적 증거로 확대하지 않는다.

### 연결되는 claim_id

- EP04

### 제4장에서 가져올 직접 결과 요약

- **4.4 / 전체 지표:** mAP50-95 차이는 `+0.017378`, CI `[+0.000580,+0.036332]`, 양의 acquisition seeds는 `4/5`였고 recall 차이는 `+0.035381`이었다.
- **4.4 / Class-group 지표:** Frequent-class macro AP 차이는 `+0.033345`였으나 rare-class macro AP 차이는 `-0.019877`이었고 희소 클래스 기준은 충족되지 않았다.
- Rare-class macro AP는 별도 CI가 없는 한 GC10 q20 설정의 point estimate다.

### 이 절에서 말할 수 있는 해석 문장

1. GC10 q20의 전체 mAP 개선만으로 모든 클래스 그룹의 개선을 기술할 수는 없었다.
2. 같은 다운스트림 평가에서 전체·frequent 지표와 rare 지표가 상반된 방향을 보였으므로 평균 지표와 subgroup 지표를 함께 보고할 필요가 있다.
3. 본 사례에서 희소 클래스 기준은 평균 성능 지표가 포착하지 않은 별도 판정 정보를 제공했다.
4. 다만 rare AP의 별도 신뢰구간이 없으므로 해석은 point-estimate conflict와 사전 기준 미충족에 한정해야 한다.

### 이 절에서 말하면 안 되는 과장/일반화 문장

- “Overall mAP는 rare class를 항상 희생한다.”
- “DINO가 희소 클래스 성능을 악화시켰다.”
- “Rare-class harm가 통계적으로 확정되었다.”
- “산업 안전 위험이 입증되었다.”
- “모든 데이터셋에서 평균 성능과 희소 클래스 안전성이 충돌한다.”
- “희소 클래스 기준이 실제 사고 위험을 예측한다.”

### 필요한 figure/table 또는 교차참조

- 제4장 4.4 및 표 4-4 교차참조
- `docs/notion_professor_update_20260803/assets/figure04_gc10_endpoint_conflict.png`
- 5.4에서는 전체·frequent·rare 지표의 방향을 재계산하지 않고 4.4의 그림과 표를 인용한다.

### 남는 한계 또는 예상 반론

- Rare-class macro AP에는 별도 CI가 없어 통계적으로 확정된 저하라고 부를 수 없다.
- 한 데이터셋, 한 budget, 한 learner 설정의 결과이므로 일반적인 rare-class harm를 추정하지 못한다.
- Rare-class 지표와 실제 산업 안전·품질 비용의 연결은 측정하지 않았다.

### 한 문장 요약

GC10 q20의 상반된 지표 방향은 전체 성능과 희소 클래스 성능을 함께 해석해야 할 사례를 제공하지만, 희소 클래스 harm의 일반적·인과적 증거는 아니다.

## 5.5 이미지 예산 기반 평가의 한계

### 절의 목적

본 연구가 query image 수를 공통적인 예산 표현으로 사용했을 때 생기는 해석 한계를 정리한다. 비용·annotation-time·계산 자원·금액 원장 부재에 관한 상세 한계는 이 절에 집중하고 5.7에서는 본 절을 교차참조하며, 이를 비용 절감이나 새로운 cost-aware 평가 기여로 제시하지 않는다.

### 연결되는 claim_id

- 직접 연결되는 최종 claim 없음
- EP01–EP05의 산업적·비용적 해석 범위를 제한하는 보조 절

### 제4장에서 가져올 직접 결과 요약

- **4.1:** 평가 분기별 query budget과 선택 단계 native endpoint를 표 4-1에 구분해 제시했다.
- **4.4:** GC10 결과는 `q20` 이미지 예산에서 관찰되었다.
- **4.5:** KSDD2 선택 단계는 `100-image query`였고 제한적 learner package는 `60/60` runs를 완료했다.
- 현재 자료에는 실제 active annotation time, hardware-normalized GPU-hour, energy 또는 금액 원장이 없다. KSDD2의 defect-pixel/component burden proxy도 annotation time 측정값이 아니다.

### 이 절에서 말할 수 있는 해석 문장

1. 동일한 이미지 수 예산은 선택 집합의 크기를 통제하지만 이미지별 결함 수, 주석 복잡도 및 실제 작업 시간을 동일하게 만들지는 않는다.
2. 따라서 본 연구의 budget 비교는 image-count 기준의 평가로 한정되며 annotation effort나 총비용 비교로 바꿀 수 없다.
3. 결함 발견량과 범주 구성의 차이가 실제 주석 부담에 영향을 줄 가능성은 있으나 현재 자료에서는 그 크기와 방향을 측정하지 않았다.
4. 이미지 예산의 한계는 EP01–EP05의 해석 범위를 제한하는 조건이며 독립적으로 검증된 cost-aware contribution이 아니다.

### 이 절에서 말하면 안 되는 과장/일반화 문장

- “동일 이미지 예산은 동일한 주석 비용을 의미한다.”
- “본 연구가 annotation cost를 절감했다.”
- “Prevented model runs가 실제 GPU-hour 또는 금액 절감이다.”
- “KSDD2 burden proxy는 annotation time이다.”
- “후보 selector가 cost-normalized 효율에서 우월하다.”
- “이미지 예산 기반 평가의 새로운 framework를 검증했다.”

### 필요한 figure/table 또는 교차참조

- 제4장 4.1의 표 4-1 교차참조
- **표 5-2(선택적 보조 표):** `평가 분기 × 관찰된 예산 단위 × 관찰되지 않은 자원 변수`
- 새 cost figure는 만들지 않는다.
- 표의 미관찰 항목은 `annotation time`, `GPU-hour`, `energy`, `currency cost`로 명시하되 수치를 추정하지 않는다.

### 남는 한계 또는 예상 반론

- 이미지별 주석 난이도와 구성요소 수가 달라 equal image budget의 운영상 등가성을 확인할 수 없다.
- 실제 시간·비용 원장이 없어 비용 절감 또는 비용효율 주장은 식별 불가능하다.
- 데이터셋 간 query budget과 task가 달라 image count 자체도 공통 산업 기준이 아니다.

### 한 문장 요약

본 연구의 예산 비교는 선택 이미지 수를 통제한 결과로만 해석해야 하며, 실제 주석 시간·계산 비용·금액의 등가성이나 절감을 의미하지 않는다.

## 5.6 선택–학습 연결 해석의 범위와 제한 조건

### 절의 목적

EP05의 Dice 차이와 사전 최소효과 기준 미충족을 EP01의 제한 조건과 연결해, 선택–학습 연결을 해석할 때 구분해야 할 판정 수준을 정리한다. 제6장의 최종 통합 문장을 반복하지 않고 해석 조건만 고정한다.

### 연결되는 claim_id

- EP05: 중심 사례
- EP01: 범위와 제한 조건

### 제4장에서 가져올 직접 결과 요약

- **4.5 / KSDD2 핵심 판정:** Macro Dice 차이는 `+0.037565`, CI `[+0.019581,+0.057029]`였고, 결과 확인 전에 고정한 평균 `+0.05` 기준은 충족하지 못했다.
- **보조 결과 처리:** Leave-one-seed-out, positive seed fraction, pixel AP, normal FPR 및 run completeness는 본 절에서 수치를 반복하지 않고 **표 4-5**를 교차참조한다.
- **5.1 연결:** 다른 completed 분기들도 각자의 downstream 기준 미충족 지점이 달랐으며 native metric을 합산하지 않았다.

### 이 절에서 말할 수 있는 해석 문장

1. KSDD2에서는 통계적으로 양의 방향인 Dice 추정치와 결과 확인 전에 고정한 평균 `+0.05` 기준의 충족이 서로 다른 판정을 만들었다.
2. 결과 확인 전에 기준을 고정했다는 사실은 판정 절차의 시간적 투명성을 제공하지만, 그 기준이 산업적으로 최적이거나 경제적으로 보정되었음을 의미하지 않는다.
3. EP01의 세 분기는 task·metric·threshold가 다르므로 선택–학습 연결은 분기별 사전 기준에 대한 판정으로만 비교해야 하며 공통 효과크기로 합산할 수 없다.
4. 통계적 양의 방향, 사전 기준 충족, 외부 실용성은 서로 다른 해석 수준이며 현재 자료는 마지막 수준을 직접 검증하지 않았다.

### 이 절에서 말하면 안 되는 과장/일반화 문장

- “KSDD2에서 translation effect가 없었다.”
- “PatchCore 우월성이 입증되었다.”
- “KSDD2에서 실용적 개선이 입증되었다.”
- “`+0.05`는 산업적으로 최적인 threshold다.”
- “Official test 성능이 개선되었다.”
- “선택 단계 gain은 일반적으로 학습으로 이어지지 않는다.”
- “현재 기준은 좋은 selector를 사전에 정확히 판별한다.”

### 필요한 figure/table 또는 교차참조

- 제4장 4.5 및 **표 4-5를 주 교차참조**한다. 보조 지표와 정상 이미지 FPR 수치는 본 절에서 다시 열거하지 않는다.
- `docs/notion_professor_update_20260803/assets/figure08_ksdd2_positive_below_threshold.png`
- 5.1의 표 5-1 교차참조
- **표 5-3(선택적 해석 표):** `관찰된 효과 방향 × 결과 확인 전 기준 판정 × 외부 calibration 상태`
- 표 5-3은 새 수치를 만들지 않고 기존 판정 수준을 병렬 정리한다.

### 남는 한계 또는 예상 반론

- `+0.05` 기준은 결과 확인 전에 고정되었지만 stakeholder cost·quality 요구에 의해 외부 calibration되지 않았다.
- KSDD2 결과는 한 acquisition pool과 한 learner에 한정된다.
- 신뢰구간 상한이 `+0.05`를 넘더라도 primary rule은 mean estimate `≥ +0.05`였으므로 기준 충족으로 바꿀 수 없다.
- 반대로 decision label이 기준 미충족이라고 해서 effect absence로 해석할 수도 없다.

### 한 문장 요약

5.6은 KSDD2의 Dice 방향과 사전 기준 판정을 구분하고, 이를 세 평가 분기에 연결할 때 필요한 native metric·threshold·외부 calibration의 제한을 명시하는 데 한정한다.

## 5.7 연구의 한계

### 절의 목적

EP01–EP05의 주장을 성립시키는 범위와 현재 자료가 답하지 못하는 질문을 명시한다. 한계를 사후 방어 문구로 축소하지 않고 각 claim의 해석 경계로 제시한다.

### 연결되는 claim_id

- EP01–EP05 공통 한계
- 새로운 claim 없음

### 제4장에서 가져올 직접 결과 요약

- 제4장의 결과는 데이터셋·budget·selector·learner·metric이 다른 평가 분기를 각자의 native metric으로 제시했으며 cross-dataset numeric pooling을 수행하지 않았다.
- MPDD와 VisA의 `200`회 반복은 동일 fixed pool의 paired perturbation이었고, NEU의 acquisition 반복도 같은 물리적 pool 안에서 수행되었다.
- NEU 고정 집합 결과는 `POSTHOC_DESCRIPTIVE`였고, 통합 Atlas와 taxonomy도 `post-hoc integration`이다.
- KSDD2의 `+0.05` 기준은 결과 확인 전에 고정되었으나 외부 stakeholder·비용 근거로 calibration되지 않았다.
- MPDD·VisA·DeepPCB의 learner outcome은 `NOT_AVAILABLE`이므로 사전 기준 미충족으로 후속 평가가 중단된 분기의 counterfactual downstream outcome은 관찰되지 않았다.

### 이 절에서 말할 수 있는 해석 문장

1. 본 연구의 통합은 여러 결과를 본 뒤 구성된 post-hoc 분석·보고 체계이며, 공통 protocol로 사전등록된 framework 검증이 아니다.
2. Fixed-pool perturbation은 같은 pool 안에서 initial set 또는 acquisition seed 변화에 대한 민감도를 보여주지만 independent production pool 재현성을 제공하지 않는다.
3. 결과 확인 전에 고정한 threshold는 절차적 판정 기준으로 사용할 수 있으나 외부 stakeholder 요구나 비용에 근거한 최적 threshold로 해석할 수 없다.
4. Learner outcome이 없는 STOP 분기는 downstream 실패도 성공도 확인되지 않은 경계 사례이며, Gate 예측 정확도의 근거로 사용할 수 없다.

### 이 절에서 말하면 안 되는 과장/일반화 문장

- “본 연구는 검증된 선택–학습 translation framework를 제안한다.”
- “결과가 independent production pool에서 일반화되었다.”
- “200 seeds는 200개 독립 산업 조건의 반복이다.”
- “STOP branch는 downstream failure를 정확히 예측했다.”
- “Gate precision과 recall이 입증되었다.”
- “사전 threshold는 산업적으로 최적이다.”
- 비용·annotation-time·계산 자원·금액 관련 금지선은 5.5를 교차참조하며 본 절에서 반복하지 않는다.
- “본 연구는 산업 Active Learning benchmark다.”

### 필요한 figure/table 또는 교차참조

- **표 5-4(신규 한계 매트릭스):** `한계 × 영향받는 claim × 관찰된 범위 × 미관찰 범위 × 허용되는 해석`
- `docs/translation_atlas/translation_atlas_main_figure.png`에는 post-hoc 및 no-pooling 각주를 유지한다.
- 4.2의 fixed-pool 반복, 4.3의 post-hoc fixed-set 및 acquisition 반복, 4.5의 threshold 판정을 교차참조한다.
- 이미지 예산과 실제 비용·annotation-time의 관계는 5.5를 교차참조한다.
- STOP branch의 learner outcome은 `NOT_AVAILABLE`로 표시하고 빈칸이나 실패로 대체하지 않는다.

### 남는 한계 또는 예상 반론

1. **Post-hoc integration 한계:** Atlas와 taxonomy는 여러 분기 결과를 본 뒤 구성되었으며 공통 사전등록 통합이 아니다.
2. **Independent production pool 부재:** Fresh site·lot·production pool에서의 selector와 Random 비교가 없다.
3. **Fixed-pool perturbation의 해석 한계:** GC10·MPDD·VisA mean pool Jaccard는 각각 `0.978446`, `0.962827`, `0.995386`으로 높으며, 반복은 외부 환경 변화가 아니라 같은 pool 안의 perturbation이다.
4. **Threshold calibration 외부 근거 부재:** KSDD2 `+0.05` 등은 결과 확인 전 고정 기준이지만 stakeholder cost·quality 또는 외부 효용에 근거한 최적값이 아니다.
5. **STOP branch의 counterfactual outcome 부재:** MPDD·VisA·DeepPCB는 learner outcome이 없으므로 중단하지 않았을 때의 downstream 성능과 Gate predictive validity를 식별할 수 없다.
6. **이질적 분기:** Dataset, task, budget, selector, learner, metric, threshold와 시간적 지위가 달라 공통 effect 또는 실패율을 추정할 수 없다.
7. **희소 클래스 불확실성:** GC10 rare macro AP에는 별도 CI가 없다.
8. **이미지 예산의 비용 해석 제한:** 미관찰 자원 변수와 비용 관련 해석 경계는 5.5를 교차참조하며 본 절에서 상세 항목을 반복하지 않는다.

### 한 문장 요약

따라서 본 연구는 평가한 분기 안에서 선택 단계와 후속 평가의 일치·불일치를 감사한 제한적 실증 연구로 해석해야 하며, 검증된 framework·산업 benchmark·독립 생산 환경 일반화로 확대할 수 없다.

## 제5장에서만 말할 수 있는 학술적 기여 문장 후보

- 본 연구는 평가한 산업 결함 Active Learning 분기에서 결함 발견량, 선택 집합 구성, acquisition reproducibility, 다운스트림 평균 성능, 희소 클래스 지표 및 사전 최소효과가 동일한 판정으로 환원되지 않는 사례를 단계별로 구분해 제시하였다.
- MPDD와 VisA의 관찰은 selection-stage discovery와 selected-set composition을 별도 평가 대상으로 기술할 필요성을 보여준다.
- NEU-DET 사례는 고정 선택 집합의 training-seed 안정성과 selector 재실행을 포함한 acquisition reproducibility가 서로 다른 조건부 질문임을 보여준다.
- GC10 q20 사례는 aggregate metric과 rare-class subgroup metric의 병렬 보고가 단일 평균 지표가 놓치는 판정 정보를 보존할 수 있음을 보여준다.
- KSDD2 사례는 통계적 양의 방향과 결과 확인 전에 고정한 최소효과 기준의 충족을 구분해 보고하는 사례를 제공한다.

## 제5장에서만 말할 수 있는 산업적 시사점 문장 후보

- 평가한 분기에서는 선택 집합의 결함 이미지 수뿐 아니라 범주 구성과 희소 클래스 지표를 함께 확인해야 선택 결과의 범위를 더 손실 없이 기술할 수 있었다.
- 한 고정 선택 집합의 반복 학습이 안정적이더라도 실제 획득 과정을 다시 수행한 결과는 별도로 확인할 필요가 있다.
- 결과 확인 전에 기준을 고정하는 절차는 판정의 투명성을 높일 수 있지만, 실제 운영 기준으로 사용하려면 stakeholder cost·quality 요구에 대한 외부 calibration이 추가로 필요하다.
- 이미지 수 예산은 주석 대상을 통제하는 단위로는 사용할 수 있으나 실제 주석 시간이나 총비용의 대리변수로 간주해서는 안 된다.
- 이 시사점들은 현재 데이터셋·budget·pool·learner 범위에 한정되며 비용 절감이나 생산 환경 일반화를 입증하지 않는다.

## 제6장으로 넘겨야 할 통합 문장

다음 EP01 문장은 제5장 본문에서 결론처럼 반복하지 않고 제6장 6.1의 마지막 통합 문장으로 넘긴다.

> **선택 단계와 다운스트림 결과가 모두 관찰된 세 평가 분기에서, 선택 단계에서 양의 방향으로 나타난 결과만으로는 사전에 정한 다운스트림 평가 기준의 충족을 확인할 수 없었다.**

## 과장 위험이 큰 금지 문장 목록

- “Selection gain은 learning으로 이어지지 않는다.”
- “선택 단계 신호는 다운스트림 utility의 충분조건이 아니다”를 범위 제한 없이 보편 명제로 제시한다.
- “특정 selector가 Random보다 일반적으로 우월하다.”
- “검증된 framework가 좋은 selector를 사전에 판별한다.”
- “Gate의 예측 정확도·precision·recall이 입증되었다.”
- “Full translation 실패율은 100%다.”
- “MPDD·VisA·DeepPCB의 STOP은 downstream failure였다.”
- “200 fixed-pool perturbations는 200개의 독립 생산 조건이다.”
- “10 acquisition seeds가 independent production pool 일반화를 입증했다.”
- “Overall mAP 개선은 rare class를 항상 희생한다.”
- “Rare-class harm 또는 산업 안전 위험이 통계적으로 입증되었다.”
- “KSDD2에서 effect가 없었다” 또는 “실용적 개선이 입증되었다.”
- “`+0.05` threshold는 산업적으로 최적이다.”
- “이미지 예산은 실제 annotation time 또는 비용과 동일하다.”
- “실제 GPU-hour, annotation cost 또는 금액 절감을 입증했다.”
- “본 연구는 산업 결함 Active Learning benchmark다.”
- “관찰된 endpoint conflict의 원인이 DINO, representation 또는 clustering임이 확정되었다.”



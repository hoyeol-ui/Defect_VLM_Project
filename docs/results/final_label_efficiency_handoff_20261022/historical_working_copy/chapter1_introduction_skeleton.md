# 제1장 서론 문단 골격

## 사용 원칙

이 문서는 완성 문장이 아니라 문단별 기능과 메시지의 순서를 고정하는 writing skeleton이다. 선행연구 anchor는 기존 positioning memo의 항목만 사용하며, 산업 결함 축의 `IDD-01`–`IDD-04`는 입력 CSV의 `IND-01`–`IND-04`에 대응한다.

- 서술 흐름은 **상황 → 평가 문제 → 연구 공백 → bounded empirical audit의 목적 → 범위**로 고정한다.
- 서론에서 EP01–EP05의 결과를 미리 선언하거나 수치를 제시하지 않는다.
- “필요하다”는 표현은 평가 단위의 분리 필요성에만 사용하고, 새 framework나 보편 규범의 정립을 뜻하게 쓰지 않는다.
- 자기 규정 3문장은 아래 지정 위치에서 원문 그대로 사용한다.

## 1.1 연구 배경

| 문단 | 문단 기능 | 핵심 메시지 1문장 | 선행연구 anchor id | 서술 범위 제한 | 다음 문단으로의 연결 |
|---|---|---|---|---|---|
| 1 | 산업 맥락과 주석 병목 제시 | 산업 결함 탐지에서는 희소·미세 결함과 전문가 판독 요구 때문에 labeled data 확보가 연구 및 적용의 주요 제약으로 다뤄져 왔다. | `IDD-01`, `IDD-02` | 실제 비용 규모나 본 연구의 비용 절감 효과는 쓰지 않음 | 제한된 label에서 어떤 표본을 선택할지로 전환 |
| 2 | 산업 결함 AL의 기존 목표 정리 | 산업 결함 AL 연구는 uncertainty·diversity·hybrid sampling과 human feedback을 활용해 제한된 label에서 학습 자료를 구성하는 방향으로 전개되어 왔다. | `IDD-03`, `IDD-04` | 기존 방법 순위나 본 연구의 우월성을 비교하지 않음 | label efficiency 중심 흐름의 평가 단위를 질문 |
| 3 | pretrained representation 시대의 확장 제시 | VFM/VLM과 pretrained vision encoder의 활용은 representativeness, diversity, calibration 및 class balance를 결합한 새로운 selection 신호를 가능하게 했다. | `VFM-01`, `VFM-02` | Foundation representation의 일반적 우수성 또는 산업 결함 전이 성능을 주장하지 않음 | 강한 신호와 후속 학습 결과를 구분할 필요로 연결 |
| 4 | 배경에서 문제 설정으로 전환 | 선택 단계 신호가 정교해졌다는 사실과 selected-set composition, acquisition 반복, downstream 성능 및 희소 클래스 지표가 같은 판정을 보인다는 사실은 서로 다른 평가 문제다. | `VFM-03`, `VFM-04`, `ODA-04` | “항상 불일치한다” 또는 인과적 representation mismatch를 쓰지 않음 | 1.2에서 평가·재현성 문제를 구체화 |

### 1.1 집필 잠금

- 첫 문단은 산업 현장 일반론보다 expert labeling 병목으로 시작한다.
- pretrained representation은 연구 환경 변화로만 제시하며, 결과의 원인으로 소급하지 않는다.
- image count는 예산 표현일 뿐 annotation time·비용의 대리변수로 단정하지 않는다.

## 1.2 연구 문제의 설정

| 문단 | 문단 기능 | 핵심 메시지 1문장 | 선행연구 anchor id | 서술 범위 제한 | 다음 문단으로의 연결 |
|---|---|---|---|---|---|
| 1 | AL 성능 비교의 조건 민감성 제시 | Deep AL의 Random 대비 결과는 stochasticity, seed, protocol 및 domain에 민감할 수 있어 단일 점 추정치나 한 번의 실행만으로 일반적 우열을 확정하기 어렵다. | `ALR-01`, `ALR-04` | 본 연구가 이 문제를 해결했다고 쓰지 않음 | 어떤 randomness를 반복했는지의 구분으로 이동 |
| 2 | 재현성 단위의 분리 | 초기 집합·학습 seed·query batch·acquisition seed는 서로 다른 변동 원천이므로 고정 집합 재학습과 선택 재실행 결과가 답하는 질문을 구분해야 한다. | `ALR-02`, `ALR-03` | 통합 재현성 표준이나 benchmark를 제안하지 않음 | Object detection의 복수 endpoint로 확장 |
| 3 | Detection과 rare-class 평가 문제 | Object detection AL에서는 classification·localization 신호, batch composition, aggregate 성능 및 rare-slice 성능이 단일 success label로 환원되지 않을 수 있다. | `ODA-01`, `ODA-02`, `ODA-03` | rare-class harm·safety hazard나 일반적 충돌을 선취하지 않음 | 선택 단계와 후속 평가의 연결 문제로 수렴 |
| 4 | 핵심 연구 공백 명시 | 기존 연구가 선택 신호와 label efficiency를 발전시켜 왔더라도, selection-stage discovery를 composition, acquisition reproducibility, downstream utility, rare-class 지표 및 사전 최소효과 판정과 구분해 연결 상태를 기록하는 질문은 별도로 남는다. | `VFM-02`, `VFM-03`, `IDD-02`, `IDD-04` | “기존 연구가 전혀 다루지 않았다”는 novelty 표현을 쓰지 않음 | bounded audit의 필요성으로 연결 |
| 5 | 문제를 bounded audit로 한정 | 필요한 것은 selector의 단일 우승자를 정하는 것이 아니라, 현재 평가한 분기에서 각 평가 단위가 어디까지 관찰되고 어떤 지점에서 해석이 제한되는지를 명시하는 제한적 실증 감사다. | `ALR-01`–`ALR-04`; `ODA-02`–`ODA-04` | 세 분기의 결과를 모집단 실패율이나 보편 법칙으로 확대하지 않음 | 1.3의 목적과 연구 질문 제시 |

### 1.2 서술 범위 제한

- 연구 결과를 미리 요약하는 EP01 문장은 사용하지 않는다.
- “Selection gain은 learning으로 이어지지 않는다”와 같은 범위 없는 일반화를 사용하지 않는다.
- fixed-set stability와 acquisition reproducibility를 동의어로 쓰지 않는다.
- rare-class 지표의 별도 확인을 실제 산업 안전성의 입증으로 바꾸지 않는다.
- protocol 민감성을 본 연구 framework의 필요성이나 검증 성공으로 연결하지 않는다.

## 1.3 연구 목적 및 연구 질문

### 문단 골격

| 문단 | 문단 기능 | 핵심 메시지 1문장 | 배치할 자기 규정 | 서술 범위 제한 |
|---|---|---|---|---|
| 1 | 연구 목적 선언 | 본 연구의 목적은 평가한 산업 결함 AL 분기에서 선택 단계, 선택 집합, 반복 단위, 후속 학습 및 희소 클래스 지표를 분리해 그 연결의 관찰 범위와 제한을 정리하는 것이다. | 아래 ‘무엇인가’ 문장을 문단 마지막에 원문 그대로 배치 | 방법 개발 또는 selector ranking을 목적에 포함하지 않음 |
| 2 | 연구 질문 제시 | 연구 질문은 선택 단계 관찰, 두 재현성 단위, 후속 utility 판정, 증거 상태와 해석 범위의 네 묶음으로 구성한다. | 없음 | 결과 방향을 질문 안에 선취하지 않음 |
| 3 | 평가 원칙 고정 | 각 분기는 native metric과 기존 판정 기준을 유지하며, 이질적인 효과를 합산하지 않고 `POSTHOC_DESCRIPTIVE`와 `NOT_AVAILABLE`을 별도 상태로 기록한다. | 없음 | 공통 효과크기·실패율·Gate 정확도를 제안하지 않음 |

### 자기 규정 — 이 논문이 무엇인가

> 본 논문은 평가한 산업 결함 Active Learning 분기에서 selection-stage discovery, selected-set composition, fixed-set training stability, acquisition reproducibility, downstream learner utility, rare-class safety를 분리하고, 이들 사이의 연결이 어디까지 관찰되며 어디서 제한되는지를 정리한 bounded empirical audit이다.

### 최종 연구 질문

1. **RQ1 — 선택 단계:** 평가한 산업 결함 AL 분기에서 후보 선택 신호의 Random 대비 결과는 해당 지표가 관찰된 분기의 discovery 및 selected-set composition 지표에서 어떻게 나타나는가?
2. **RQ2 — 반복 단위:** 고정 선택 집합의 training seed 반복 결과와 selector를 다시 실행하는 acquisition seed 반복 결과는 각각 어떤 변동성을 보여 주는가?
3. **RQ3 — 후속 평가:** 선택 단계 결과와 downstream learner utility, aggregate 성능, 희소 클래스 지표 및 사전 최소효과 기준의 판정은 해당 결과가 관찰된 평가 분기에서 어떻게 함께 또는 다르게 나타나는가?
4. **RQ4 — 증거 경계:** Post-hoc 결과, fixed-pool 반복, learner outcome의 `NOT_AVAILABLE`, 결과 확인 전에 고정한 기준을 구분할 때 현재 자료가 허용하는 선택–학습 연결 해석의 범위는 어디까지인가?

### 연구 질문별 연결 위치

| 연구 질문 | 방법 장 | 결과 장 | 논의 장 | claim 경계 |
|---|---|---|---|---|
| RQ1 | 3.2, 3.4 | 4.1, 4.2 | 5.2 | EP02 범위; learner 결과가 없는 분기는 downstream 실패로 세지 않음 |
| RQ2 | 3.5 | 4.3 | 5.3 | EP03 범위; favorable fixed set은 `POSTHOC_DESCRIPTIVE` |
| RQ3 | 3.6, 3.7 | 4.4, 4.5 | 5.4, 5.6 | EP04·EP05 범위; 인과·success/no-effect 이분법 금지 |
| RQ4 | 3.1, 3.7 | 상태 표기만 수행 | 5.1, 5.5–5.7 | EP01의 해석 범위; 최종 통합 문장은 6.1 마지막 1회 |

## 1.4 연구 범위와 논문 구성

### 문단 골격

| 문단 | 문단 기능 | 핵심 메시지 1문장 | 반드시 포함할 내용 | 서술 범위 제한 |
|---|---|---|---|---|
| 1 | 기여 층위와 비범위 확정 | 본 연구는 방법 우승이나 예측 framework가 아니라 평가한 분기에서 평가 단위와 증거 상태를 분리하는 연구다. | 아래 ‘무엇이 아닌가’와 ‘기여 층위’ 문장을 원문 그대로 배치 | identity 문구를 약화·강화해 변형하지 않음 |
| 2 | 실증 범위 명시 | 분석 범위는 현재 자료의 데이터셋·budget·pool·selector·learner·metric·threshold 및 시간적 지위에 한정된다. | native metric 유지, cross-dataset no-pooling, post-hoc integration | 새로운 공통 효과나 대표성 주장 금지 |
| 3 | 미관찰 범위 명시 | Learner outcome이 없는 분기, 외부 보정되지 않은 threshold, 동일 fixed pool 반복 및 비용 원장 부재는 결론의 범위를 제한한다. | `NOT_AVAILABLE`, independent production pool 부재, 비용 측정 부재 | 미관찰 결과를 실패·성공으로 대체하지 않음 |
| 4 | 논문 구성 안내 | 제2장은 네 선행연구 축과 연구 위치, 제3장은 분기별 평가 설계, 제4장은 관찰, 제5장은 해석과 한계, 제6장은 제한된 통합 요약을 담당한다. | 제4장=관찰, 제5장=해석, 제6장=통합 문장 | 장별 역할 혼합 금지 |

### 자기 규정 — 이 논문이 무엇이 아닌가

> 본 논문은 특정 selector의 일반적 우월성, 선택–학습 연결을 예측하는 검증된 framework, 산업 benchmark, 실제 비용 절감 또는 독립 production pool 일반화를 입증하는 연구가 아니다.

### 자기 규정 — 기여의 층위

> 본 논문의 기여는 새로운 선택 방법의 제안이 아니라, 평가한 분기 안에서 선택 단계와 후속 학습의 평가 단위 및 증거 상태를 분리하고 그 연결의 관찰 범위와 제한을 실증적으로 정리하는 데 있다.

### 이 논문이 다루지 않는 것

- 새로운 selector 또는 VFM/VLM sampling method의 제안
- 특정 selector의 Random 대비 일반적 우월성
- 선택–학습 translation framework 또는 Gate 예측 정확도의 검증
- 산업 결함 AL benchmark 또는 공통 method ranking
- 실제 annotation time, GPU-hour, energy, 금액 비용의 절감 입증
- 독립 site·lot·production pool에 대한 외부 일반화
- fixed-pool perturbation을 독립 생산 조건으로 해석하는 주장
- rare-class harm, safety hazard 또는 causal representation mismatch의 입증
- STOP 분기의 counterfactual downstream outcome 추정
- 이질적인 평가 분기의 수치 합산, 공통 효과크기 또는 일반 실패율 추정

## 제1장에서 반드시 등장해야 하는 문장 5개

1. 산업 결함 Active Learning은 전문가 주석이 제한된 조건에서 학습 자료를 선별하려는 문제를 다루지만, 선택 이미지 수의 감소만으로 실제 주석 시간이나 비용 절감을 확인할 수는 없다.
2. Deep Active Learning의 Random 대비 결과는 seed, 초기 집합, 학습 protocol 및 domain에 따라 달라질 수 있으므로, 선택 방법의 단일 점 추정치만으로 일반적 우월성을 판단하기 어렵다.
3. Object detection 및 VFM/VLM 기반 연구가 선택 신호를 정교화해 왔더라도, selection-stage discovery와 selected-set composition, acquisition reproducibility, downstream learner utility 및 희소 클래스 지표 사이의 관계는 구분해 확인할 필요가 있다.
4. 본 논문은 평가한 산업 결함 Active Learning 분기에서 selection-stage discovery, selected-set composition, fixed-set training stability, acquisition reproducibility, downstream learner utility, rare-class safety를 분리하고, 이들 사이의 연결이 어디까지 관찰되며 어디서 제한되는지를 정리한 bounded empirical audit이다.
5. 본 논문의 기여는 새로운 선택 방법의 제안이 아니라, 평가한 분기 안에서 선택 단계와 후속 학습의 평가 단위 및 증거 상태를 분리하고 그 연결의 관찰 범위와 제한을 실증적으로 정리하는 데 있다.

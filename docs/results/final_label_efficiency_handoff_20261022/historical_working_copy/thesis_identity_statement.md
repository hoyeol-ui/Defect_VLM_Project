# Thesis Identity Statement

## 0. 문서 지위와 적용 원칙

이 문서는 논문 전체에서 사용할 자기 규정과 주장 경계를 고정한다. `related_work_positioning_memo.md`에서 채택된 포지셔닝을 우선하며, EP01–EP05의 최종 claim과 제4·5장 v2 writing packet의 역할 분리를 변경하지 않는다.

- `bounded`는 현재 자료에서 실제로 평가된 데이터셋·budget·pool·selector·learner·metric·threshold·시간적 지위의 범위를 뜻한다.
- `empirical`은 repository에 존재하는 관찰과 판정만 사용한다는 뜻이며, 미관찰 결과나 인과 기제를 보충하지 않는다.
- `audit`은 평가 단위와 증거 상태를 분리해 대조한다는 뜻이며, 검증된 방법론·framework·benchmark를 뜻하지 않는다.
- 이 문서에서 `rare-class safety`는 희소 클래스 지표를 전체 성능과 별도로 확인하는 평가 항목을 뜻한다. 실제 산업 안전 위험이나 rare-class harm의 인과적 입증을 뜻하지 않는다.

## 1. 확정 자기 규정 3문장

### 1.1 이 논문이 무엇인가

> 본 논문은 평가한 산업 결함 Active Learning 분기에서 selection-stage discovery, selected-set composition, fixed-set training stability, acquisition reproducibility, downstream learner utility, rare-class safety를 분리하고, 이들 사이의 연결이 어디까지 관찰되며 어디서 제한되는지를 정리한 bounded empirical audit이다.

### 1.2 이 논문이 무엇이 아닌가

> 본 논문은 특정 selector의 일반적 우월성, 선택–학습 연결을 예측하는 검증된 framework, 산업 benchmark, 실제 비용 절감 또는 독립 production pool 일반화를 입증하는 연구가 아니다.

### 1.3 기여의 층위

> 본 논문의 기여는 새로운 선택 방법의 제안이 아니라, 평가한 분기 안에서 선택 단계와 후속 학습의 평가 단위 및 증거 상태를 분리하고 그 연결의 관찰 범위와 제한을 실증적으로 정리하는 데 있다.

세 문장은 제1장과 제2장 2.5에서 동일한 강도와 문구로 사용한다. 제4장에서는 자기 규정 문구를 결과 claim처럼 반복하지 않고, native metric 유지·비합산·증거 상태 표기·사전 기준 판정으로 구현한다. 제5장에서는 분기별 해석 범위와 한계를 설명하고, EP01의 최종 통합 문장은 제6장 6.1 마지막에 한 번만 둔다.

## 2. 고정 용어와 사용 경계

| 내부 또는 혼동 가능 표현 | 논문에서 사용할 표현 | 고정 의미 | 허용하지 않는 확장 |
|---|---|---|---|
| bounded empirical audit | 제한된 범위의 실증적 감사 또는 원어 병기 | 평가한 분기에서 평가 단위·증거 상태·연결 제한을 대조 | 검증된 audit framework, benchmark |
| translation branch | 평가 분기 / 선택–학습 연결 평가 분기 | 선택 단계와 후속 평가가 연결된 개별 실험 단위 | 공통 모집단에서 추출한 동질적 반복 |
| full downstream Gate PASS | 사전에 정한 다운스트림 평가 기준 충족 | 분기별 native metric과 threshold에 따른 판정 | selector 품질의 보편적 인증, 예측 정확도 |
| positive endpoint | 양의 방향의 선택 단계 결과 | 해당 분기의 선택 단계 지표가 기준선보다 양의 방향 | downstream utility의 충분조건 |
| STOP branch | 사전 기준 미충족으로 후속 학습이 중단된 분기 | learner outcome이 `NOT_AVAILABLE`인 분기 | downstream 실패 또는 성공 |
| fixed-set stability | 고정 선택 집합의 training-seed 안정성 | 선택 집합을 고정한 조건에서의 학습 변동 | acquisition 재현성 또는 외부 일반화 |
| acquisition reproducibility | acquisition seed별 선택 재실행 결과 | 동일 물리적 pool 안에서 선택 집합 변화까지 포함한 반복 | 독립 production pool 재현성 |
| rare-class safety | 희소 클래스 지표의 별도 판정 | aggregate 지표와 병렬 보고하는 subgroup 평가 | rare-class harm, safety hazard, 산업 안전성 입증 |
| frozen threshold | 결과 확인 전에 고정한 판정 기준 | 시간적 투명성을 갖는 분기별 기준 | 산업적으로 최적이거나 외부 보정된 기준 |
| image budget | 선택 이미지 수 예산 | query image 수로 표현한 평가 범위 | annotation time·GPU-hour·금액과의 등가성 |

## 3. 장별 자기 규정 구현

| 위치 | 자기 규정의 역할 | 허용되는 구현 | 금지되는 구현 |
|---|---|---|---|
| 1.1 | 산업 결함 주석 병목과 선택 신호 확장의 배경 제시 | 왜 단계별 확인이 필요한지로 연결 | 비용 절감이 이미 입증된 문제처럼 서술 |
| 1.2 | 단일 selection 지표와 후속 평가의 구분 필요성 제기 | seed·protocol·domain 민감성, multi-endpoint 문제 | 본 연구 결과를 보편 법칙으로 선취 |
| 1.3 | 목적과 연구 질문을 bounded audit로 고정 | 평가 단위와 증거 상태를 묻는 RQ | 새 selector 개발·성능 우승을 목적화 |
| 1.4 | 범위·비범위와 장별 역할 명시 | fixed-pool, post-hoc, no-pooling 경계 | 독립 production pool·비용·benchmark 주장 |
| 2.1–2.4 | 네 선행연구 축의 문제와 본 연구의 접점 제시 | 대표 anchor를 평가 단위 관점에서 배열 | 기존 연구를 본 논문이 해결했다고 서술 |
| 2.5 | 자기 규정 3문장을 동일 문구로 확정 | 다른 질문과 기여 층위를 명시 | identity의 강도를 높이거나 낮춘 변형 |
| 3.1 | 실제 실험 설계로 조작화 | 분기, native metric, 증거 상태, 판정 시점 | 통합 framework 검증 절차로 명명 |
| 4장 | 관찰 | 수치·방향·CI·증거 상태·사전 기준 판정 | audit의 학술적 의미나 일반화 주장 |
| 5장 | 해석 | 분기별 연결 제한, 반론, 미관찰 범위 | EP01 결론문 반복, 인과 기제 확정 |
| 6.1 | 통합 요약 | EP01 최종 문장을 마지막에 한 번 사용 | 새 claim 또는 새로운 범위의 일반화 |

## 4. 선행연구 anchor 표기 규칙

제1·2장에서는 positioning memo와 `contribution_positioning_table.csv`에 이미 있는 anchor만 사용한다.

- AL 평가·재현성: `ALR-01`–`ALR-04`
- Object detection AL·rare class: `ODA-01`–`ODA-04`
- VFM/VLM 시대 AL: `VFM-01`–`VFM-04`
- 산업 결함 AL: `IDD-01`–`IDD-04`

산업 결함 축은 사용자가 지정한 `IDD-` 표기로 통일한다. 이는 입력 CSV의 `IND-01`–`IND-04`와 각각 일대일 대응하며 새로운 문헌이나 ID를 추가한 것이 아니다.

## 5. v2 패킷과의 정합성 점검

| 점검 위치 | 판정 | 충돌 또는 위험 | 고정 조치 |
|---|---|---|---|
| 제4장 전체 | 정합 | v2가 관찰·native metric·비합산 원칙을 유지함 | `bounded empirical audit`를 제4장의 결과 claim으로 승격하지 않음 |
| 4.1 제목·목적 | 정합 | 제목은 이미 관찰형으로 수정됨 | 현 제목 `후보 선택 신호별 선택 단계 관찰` 유지 |
| 4.1 문단 전개 1 | 경미한 잔여 충돌 | 본문 지침에 남은 “후보 신호의 타당성”은 검증·정당화로 읽힐 수 있음 | 초안 전 “후보 신호의 선택 단계 측정 결과를 각 분기의 native endpoint로 제시한다”로 축소 |
| 4.3 | 정합 | favorable fixed set을 confirmatory하게 읽을 위험이 상존 | `POSTHOC_DESCRIPTIVE`를 최초 언급·표·그림에 유지 |
| 4.5 | 정합 | success/no-effect 이분법 위험이 상존 | Dice 차이와 사전 최소효과 기준 판정을 병렬 유지 |
| 5.1 | 정합 | identity 설명이 6.1 EP01 결론문과 중복될 수 있음 | 세 분기의 이질성과 미충족 지점 차이만 설명하고 EP01 원문은 쓰지 않음 |
| 5.3 | 정합 | acquisition 반복을 외부 재현성으로 확대할 위험 | 동일 fixed pool의 조건부 반복임을 유지 |
| 5.5·5.7 | 정합 | 비용·예산 한계 중복 가능성 | 상세 한계는 5.5, 5.7은 5.5 교차참조만 사용 |
| 5.6 | 정합 | KSDD2를 성공 또는 무효과로 축약할 위험 | 표 4-5를 참조하고 양의 Dice 차이와 최소효과 기준 미충족을 동시에 유지 |
| 5.7 | 정합 | “감사”가 formal framework validation으로 오독될 수 있음 | 제한된 실증 연구라는 의미와 비검증 경계를 함께 명시 |

실질적인 claim 충돌은 없다. 초안 전에 필요한 직접 수정은 4.1 문단 전개 지침에 남은 `타당성` 표현 한 곳이며, 나머지는 현재 v2의 경계 문구를 유지하는 문제다.

## 논문 전체에서 반복적으로 사용할 자기 규정 3문장

1. 본 논문은 평가한 산업 결함 Active Learning 분기에서 selection-stage discovery, selected-set composition, fixed-set training stability, acquisition reproducibility, downstream learner utility, rare-class safety를 분리하고, 이들 사이의 연결이 어디까지 관찰되며 어디서 제한되는지를 정리한 bounded empirical audit이다.
2. 본 논문은 특정 selector의 일반적 우월성, 선택–학습 연결을 예측하는 검증된 framework, 산업 benchmark, 실제 비용 절감 또는 독립 production pool 일반화를 입증하는 연구가 아니다.
3. 본 논문의 기여는 새로운 선택 방법의 제안이 아니라, 평가한 분기 안에서 선택 단계와 후속 학습의 평가 단위 및 증거 상태를 분리하고 그 연결의 관찰 범위와 제한을 실증적으로 정리하는 데 있다.

## 제1장 서론에서 반드시 등장해야 하는 문장 5개

1. 산업 결함 Active Learning은 전문가 주석이 제한된 조건에서 학습 자료를 선별하려는 문제를 다루지만, 선택 이미지 수의 감소만으로 실제 주석 시간이나 비용 절감을 확인할 수는 없다.
2. Deep Active Learning의 Random 대비 결과는 seed, 초기 집합, 학습 protocol 및 domain에 따라 달라질 수 있으므로, 선택 방법의 단일 점 추정치만으로 일반적 우월성을 판단하기 어렵다.
3. Object detection 및 VFM/VLM 기반 연구가 선택 신호를 정교화해 왔더라도, selection-stage discovery와 selected-set composition, acquisition reproducibility, downstream learner utility 및 희소 클래스 지표 사이의 관계는 구분해 확인할 필요가 있다.
4. 본 논문은 평가한 산업 결함 Active Learning 분기에서 selection-stage discovery, selected-set composition, fixed-set training stability, acquisition reproducibility, downstream learner utility, rare-class safety를 분리하고, 이들 사이의 연결이 어디까지 관찰되며 어디서 제한되는지를 정리한 bounded empirical audit이다.
5. 본 논문의 기여는 새로운 선택 방법의 제안이 아니라, 평가한 분기 안에서 선택 단계와 후속 학습의 평가 단위 및 증거 상태를 분리하고 그 연결의 관찰 범위와 제한을 실증적으로 정리하는 데 있다.

## 제2장 관련연구에서 각 절 마지막에 붙일 “접점·경계” 문장 5개

1. **2.1:** 산업 결함 AL이 전문가 주석 부족과 label efficiency를 다뤄 왔다는 점에서 본 논문과 맞닿지만, 본 논문은 실제 annotation time·비용 또는 생산라인 효율을 검증하지 않고 평가한 분기의 단계별 연결만 다룬다.
2. **2.2:** Detection-specific uncertainty, balance 및 rare-slice 평가는 본 논문의 평가 단위 분리와 맞닿지만, 본 논문은 새로운 detection selector나 일반적인 rare-class harm·safety hazard를 주장하지 않는다.
3. **2.3:** VFM/VLM 연구가 pretrained representation 기반 selection 가능성을 확장했다는 점에서 본 논문과 맞닿지만, 본 논문은 VFM/VLM selector의 우월성이나 representation mismatch의 인과성을 검증하지 않는다.
4. **2.4:** AL 평가 연구의 seed·protocol·domain 민감성은 본 논문의 반복 단위 구분과 no-pooling 원칙을 뒷받침하지만, 본 논문은 표준 benchmark나 재현성 framework를 제안·검증하지 않는다.
5. **2.5:** 따라서 본 논문은 네 연구 흐름을 새로운 방법으로 통합하는 것이 아니라, 평가한 산업 결함 AL 분기에서 선택 단계와 후속 평가의 단위를 분리하고 연결의 관찰 범위를 제한적으로 정리한다.

## 본문 어디에서도 사용 금지 문장 목록

- “특정 selector가 Random보다 일반적으로 우월하다.”
- “본 연구는 선택–학습 translation framework를 검증했다.”
- “Gate가 우수한 selector를 정확하게 예측하거나 선별한다.”
- “본 연구는 산업 결함 Active Learning benchmark를 제공한다.”
- “이미지 예산 감소가 실제 annotation time 또는 비용 절감을 입증한다.”
- “Fixed-pool 반복 결과가 독립 production pool에 일반화된다.”
- “Selection-stage gain은 downstream learning utility로 이어지지 않는다.”
- “선택 단계 신호는 downstream utility의 충분조건이 아니다.”라는 문장을 관찰 분기 범위 없이 보편 명제로 쓰는 표현
- “Overall 성능 향상이 rare-class harm 또는 산업 안전 위험을 초래했다.”
- “Representation mismatch가 관찰된 불일치의 원인이다.”
- “STOP 분기는 downstream 실패를 예측했다.”
- “KSDD2는 성공했다” 또는 “KSDD2에는 효과가 없었다.”
- “사전 threshold는 산업적으로 최적이다.”
- “세 평가 분기의 `0/3`은 일반적 실패율이다.”

## 자기 규정과 v2 패킷 간 충돌 지점(있다면)

- **직접 수정 필요:** 4.1 패킷의 문단 전개 지침에 남은 “후보 신호의 타당성”은 자기 규정의 관찰형 역할보다 강하다. 실제 초안에서는 “후보 신호의 선택 단계 측정 결과”로 바꾼다.
- **표현 관리 필요:** 5.7의 “감사”는 `bounded empirical audit`의 평가 단위 대조라는 의미로만 사용하고 formal framework의 검증으로 설명하지 않는다.
- **중복 방지 필요:** 5.1에서는 자기 규정의 범위를 설명할 수 있으나 EP01의 최종 통합 문장을 반복하지 않는다. 해당 문장은 제6장 6.1 마지막에 한 번만 둔다.
- **그 외 실질 충돌 없음:** 4.3의 `POSTHOC_DESCRIPTIVE`, 4.5의 Dice 차이/기준 판정 병렬화, 5.5의 비용 경계, 5.6의 success/no-effect 금지는 bounded empirical audit 규정과 일치한다.

# 제2장 관련연구 문단 골격

## 사용 원칙과 anchor 대응

이 문서는 관련연구의 완성 초안이 아니라 절별 문단 순서·핵심 메시지·비교 경계를 고정한다. 대표 연구는 `related_work_positioning_memo.md`와 `contribution_positioning_table.csv`에 이미 채택된 anchor만 사용한다.

- `ALR-01`–`ALR-04`: AL 평가·재현성
- `ODA-01`–`ODA-04`: Object detection AL·rare class
- `VFM-01`–`VFM-04`: VFM/VLM 시대 AL
- `IDD-01`–`IDD-04`: 산업 결함 AL. 입력 CSV의 `IND-01`–`IND-04`와 일대일 대응하는 표기 정규화다.
- 각 절은 **연구 흐름 → 대표 접근 → 평가 관점의 공백 → 본 논문과의 접점·경계** 순서로 끝낸다.
- “기존 연구와 달리”를 반복하기보다, 기존 연구가 답한 질문과 본 논문이 답하려는 제한된 질문을 구분한다.

## 2.1 산업 결함 탐지와 데이터 주석화

| 문단 | 문단 기능 | 핵심 문장 1개 | 선행연구 anchor id | 경계 |
|---|---|---|---|---|
| 1 | 적용 맥락 | 산업 결함 탐지는 희소·미세 결함과 전문가 판독 요구 때문에 labeled data 확보가 병목인 영역으로 다뤄져 왔다. | `IDD-01`, `IDD-02` | 실제 비용 규모나 본 연구의 절감 효과를 쓰지 않음 |
| 2 | 산업 결함 AL의 주요 접근 | 산업 결함 AL 연구는 uncertainty, category-aware sampling 및 hybrid sampling을 이용해 제한된 label에서 학습 성능을 확보하는 문제에 초점을 두었다. | `IDD-02`, `IDD-04` | 특정 접근의 재검증이나 우열을 주장하지 않음 |
| 3 | inspectability와 human feedback | 최근 연구는 explanation과 operator feedback을 결합해 선택 근거의 inspectability를 산업 적용 논의에 포함한다. | `IDD-03` | 본 연구가 human factors, explanation quality 또는 operator trust를 평가했다고 쓰지 않음 |
| 4 | 본 논문으로의 연결 | 이 흐름은 label efficiency뿐 아니라 선택 집합의 내용과 후속 평가 상태를 구분해 기록할 배경을 제공한다. | `IDD-01`–`IDD-04` | evidence packet을 explainable AL 방법으로 부르지 않음 |

**본 논문과의 접점 및 경계:** 산업 결함 AL이 전문가 주석 부족과 label efficiency를 다뤄 왔다는 점에서 본 논문과 맞닿지만, 본 논문은 실제 annotation time·비용 또는 생산라인 효율을 검증하지 않고 평가한 분기의 단계별 연결만 다룬다.

## 2.2 객체 탐지 Active Learning

| 문단 | 문단 기능 | 핵심 문장 1개 | 선행연구 anchor id | 경계 |
|---|---|---|---|---|
| 1 | Detection-specific selection | 객체 탐지 AL은 classification과 localization을 함께 다루므로 이미지 분류의 단일 uncertainty를 그대로 적용하기 어려운 문제를 갖는다. | `ODA-01` | 본 논문이 uncertainty의 최적 정의를 제안했다고 쓰지 않음 |
| 2 | Consistency와 batch composition | Consistency와 batch balance를 함께 다루는 연구는 informativeness와 selected-set composition이 구분되는 평가 대상임을 보여준다. | `ODA-02` | 균형 sampling의 일반적 우월성이나 composition 원인론을 주장하지 않음 |
| 3 | Rare slice의 별도 평가 | Rare·critical slice 연구는 aggregate 성능과 slice-level 성능을 병렬로 확인하는 평가 관점을 제시했다. | `ODA-03` | GC10 결과를 일반 rare-class harm 또는 산업 안전 위험으로 확대하지 않음 |
| 4 | Class imbalance와 예산 단위 | Class-aware detection AL 연구는 class imbalance를 acquisition에 반영하는 동시에 선택 이미지 수가 실제 labeling cost와 같지 않을 수 있음을 지적한다. | `ODA-04` | 본 연구를 cost-aware AL 또는 비용 절감 증거로 부르지 않음 |
| 5 | 본 논문으로의 연결 | 본 논문은 detection selector를 새로 제안하지 않고 discovery, composition, aggregate 성능 및 rare-class 지표가 동일한 판정으로 환원되는지를 평가한 분기 안에서 구분한다. | `ODA-01`–`ODA-04` | 새 method, 일반 법칙, causal harm 주장 금지 |

**본 논문과의 접점 및 경계:** Detection-specific uncertainty, balance 및 rare-slice 평가는 본 논문의 평가 단위 분리와 맞닿지만, 본 논문은 새로운 detection selector나 일반적인 rare-class harm·safety hazard를 주장하지 않는다.

## 2.3 VLM 및 시각 표현 기반 선택 신호

| 문단 | 문단 기능 | 핵심 문장 1개 | 선행연구 anchor id | 경계 |
|---|---|---|---|---|
| 1 | Pretraining–finetuning 전환 | Pretrained vision representation의 활용은 제한된 annotation budget에서 representativeness와 diversity를 이용한 subset selection을 별도 문제로 부각했다. | `VFM-01` | 본 연구가 ActiveFT류 distribution matching을 검증했다고 쓰지 않음 |
| 2 | VFM 시대의 AL 재검토 | VFM 환경에서는 initial pool, diversity 및 representativeness–uncertainty 관계가 기존 AL 조건과 다르게 관찰될 수 있어 실제 평가 설정의 확인이 중요하다. | `VFM-02` | DINO 또는 foundation representation을 불일치의 원인으로 확정하지 않음 |
| 3 | VLM candidate composition | VLM 기반 AL 연구는 강한 pretrained knowledge가 있더라도 candidate class composition과 adaptation setting이 결과에 영향을 줄 수 있음을 보고했다. | `VFM-03` | VLM class imbalance 해결이나 prompt adaptation을 본 논문의 성과로 귀속하지 않음 |
| 4 | Calibration과 neighborhood | Calibrated uncertainty와 neighborhood information을 결합하는 접근은 selection score의 의미가 downstream task와 평가 설정에 의존함을 보여준다. | `VFM-04` | Classification 결과를 산업 결함 detection·segmentation에 직접 일반화하지 않음 |
| 5 | 본 논문으로의 연결 | 본 논문은 pretrained representation 기반 선택 신호의 양의 방향을 곧바로 downstream utility의 대리변수로 두지 않고 관찰 가능한 평가 단위를 분리한다. | `VFM-01`–`VFM-04` | VFM/VLM selector 우월성이나 causal representation mismatch 주장 금지 |

**본 논문과의 접점 및 경계:** VFM/VLM 연구가 pretrained representation 기반 selection 가능성을 확장했다는 점에서 본 논문과 맞닿지만, 본 논문은 VFM/VLM selector의 우월성이나 representation mismatch의 인과성을 검증하지 않는다.

## 2.4 Active Learning의 평가와 재현성

| 문단 | 문단 기능 | 핵심 문장 1개 | 선행연구 anchor id | 경계 |
|---|---|---|---|---|
| 1 | 반복 평가의 필요성 | Deep AL의 Random 대비 gain은 stochasticity와 regularization 조건에 따라 달라질 수 있어 반복과 통제된 비교가 요구된다. | `ALR-01` | 본 연구가 AL 전반의 재현성 문제를 해결했다고 쓰지 않음 |
| 2 | Randomness source 구분 | 초기 집합, 모델 학습, query batch 및 acquisition 실행은 서로 다른 randomness source이므로 각 반복이 답하는 질문을 분리해야 한다. | `ALR-02` | fixed-set stability와 acquisition reproducibility를 합치지 않음 |
| 3 | Protocol 표준화 연구 | Benchmark 연구는 동일한 training·testing protocol이 method 비교의 전제가 됨을 보여준다. | `ALR-03` | 이질적인 본 연구 분기를 benchmark나 공정한 method ranking으로 부르지 않음 |
| 4 | Domain과 반복에 따른 순위 변화 | Cross-domain 평가에서는 method 순위와 Random 대비 판정이 domain 및 반복 seed에 따라 달라질 수 있다. | `ALR-04` | 세 completed branch를 일반 실패율이나 pooled cross-domain effect로 해석하지 않음 |
| 5 | 본 논문으로의 연결 | 본 논문은 고정 집합의 learner randomness와 선택 재실행의 acquisition randomness를 분리하고 이질적인 분기의 native metric을 합산하지 않는다. | `ALR-01`–`ALR-04` | 새로운 재현성 표준 또는 framework 검증 주장 금지 |

**본 논문과의 접점 및 경계:** AL 평가 연구의 seed·protocol·domain 민감성은 본 논문의 반복 단위 구분과 no-pooling 원칙을 뒷받침하지만, 본 논문은 표준 benchmark나 재현성 framework를 제안·검증하지 않는다.

## 2.5 본 연구의 위치

| 문단 | 문단 기능 | 핵심 문장 1개 | 선행연구 anchor id | 경계 |
|---|---|---|---|---|
| 1 | 네 축의 교차점 | AL 평가·재현성, detection의 composition·rare slice, VFM/VLM 기반 신호, 산업 결함의 label-efficiency 흐름은 서로 다른 평가 단위를 제공하지만 이들을 하나의 success label로 합칠 근거를 자동으로 제공하지는 않는다. | `ALR-01`, `ALR-02`, `ALR-04`; `ODA-02`, `ODA-03`, `ODA-04`; `VFM-02`, `VFM-03`; `IDD-02`, `IDD-04` | “기존 연구가 전혀 다루지 않았다”는 novelty 주장 금지 |
| 2 | 무엇인가 | 본 논문의 대상과 범위를 첫 번째 자기 규정 문장으로 확정한다. | 위 대표 anchor | 아래 문장을 원문 그대로 사용 |
| 3 | 무엇이 아닌가 | 본 논문이 주장하지 않는 method·framework·benchmark·cost·generalization 범위를 두 번째 자기 규정 문장으로 확정한다. | `ALR-03`, `ALR-04`, `ODA-04`, `IDD-04` | 아래 문장을 원문 그대로 사용 |
| 4 | 기여 층위 | novelty를 새 방법이 아니라 평가 단위 및 증거 상태의 분리와 연결 제한의 실증 정리로 고정한다. | 네 축 대표 anchor | 아래 문장을 원문 그대로 사용 |
| 5 | 후속 장 연결 | 제3장은 분기별 평가 설계, 제4장은 관찰, 제5장은 제한된 해석, 제6장은 EP01 통합 문장 1회라는 역할 분리를 예고한다. | 해당 없음 | 결과 방향·수치·EP01 원문을 여기서 선취하지 않음 |

### 자기 규정 1 — 이 논문이 무엇인가

> 본 논문은 평가한 산업 결함 Active Learning 분기에서 selection-stage discovery, selected-set composition, fixed-set training stability, acquisition reproducibility, downstream learner utility, rare-class safety를 분리하고, 이들 사이의 연결이 어디까지 관찰되며 어디서 제한되는지를 정리한 bounded empirical audit이다.

### 자기 규정 2 — 이 논문이 무엇이 아닌가

> 본 논문은 특정 selector의 일반적 우월성, 선택–학습 연결을 예측하는 검증된 framework, 산업 benchmark, 실제 비용 절감 또는 독립 production pool 일반화를 입증하는 연구가 아니다.

### 자기 규정 3 — 기여의 층위

> 본 논문의 기여는 새로운 선택 방법의 제안이 아니라, 평가한 분기 안에서 선택 단계와 후속 학습의 평가 단위 및 증거 상태를 분리하고 그 연결의 관찰 범위와 제한을 실증적으로 정리하는 데 있다.

**본 논문과의 접점 및 경계:** 따라서 본 논문은 네 연구 흐름을 새로운 방법으로 통합하는 것이 아니라, 평가한 산업 결함 AL 분기에서 선택 단계와 후속 평가의 단위를 분리하고 연결의 관찰 범위를 제한적으로 정리한다.

## 제2장 관련연구에서 각 절 마지막에 붙일 “접점·경계” 문장 5개

1. **2.1:** 산업 결함 AL이 전문가 주석 부족과 label efficiency를 다뤄 왔다는 점에서 본 논문과 맞닿지만, 본 논문은 실제 annotation time·비용 또는 생산라인 효율을 검증하지 않고 평가한 분기의 단계별 연결만 다룬다.
2. **2.2:** Detection-specific uncertainty, balance 및 rare-slice 평가는 본 논문의 평가 단위 분리와 맞닿지만, 본 논문은 새로운 detection selector나 일반적인 rare-class harm·safety hazard를 주장하지 않는다.
3. **2.3:** VFM/VLM 연구가 pretrained representation 기반 selection 가능성을 확장했다는 점에서 본 논문과 맞닿지만, 본 논문은 VFM/VLM selector의 우월성이나 representation mismatch의 인과성을 검증하지 않는다.
4. **2.4:** AL 평가 연구의 seed·protocol·domain 민감성은 본 논문의 반복 단위 구분과 no-pooling 원칙을 뒷받침하지만, 본 논문은 표준 benchmark나 재현성 framework를 제안·검증하지 않는다.
5. **2.5:** 따라서 본 논문은 네 연구 흐름을 새로운 방법으로 통합하는 것이 아니라, 평가한 산업 결함 AL 분기에서 선택 단계와 후속 평가의 단위를 분리하고 연결의 관찰 범위를 제한적으로 정리한다.

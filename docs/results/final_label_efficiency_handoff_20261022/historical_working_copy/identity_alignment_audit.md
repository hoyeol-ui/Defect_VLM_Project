# Thesis Identity Alignment Audit

## 0. 감사 목적과 판정

이 문서는 본문 초안 작성 직전의 최종 정합성 감사다. 지정된 15개 입력만 사용했으며 저장소 재스캔, 새 실험, 새 수치 계산, 새 claim 및 새 선행연구 추가를 수행하지 않았다.

**종합 판정: 조건부 통과.** EP01–EP05의 최종 주장과 자기 규정 사이에는 논리적 충돌이 없고, 연구 질문 네 개도 다섯 claim을 모두 포괄한다. 다만 실제 초안 작성 전에 아래의 필수 봉합 표현을 적용해야 한다.

1. 4.1 v2의 실제 문단 지침에서 `타당성`을 제거한다.
2. KSDD2의 `safety check`를 `정상 이미지 FPR 점검`으로 바꿔 `rare-class safety`와 분리한다.
3. 5.3의 `선택 재현성`을 `acquisition reproducibility`로 통일한다.
4. 구버전 outline map의 4.1·4.3·4.5·5.3·5.6 제목과 final adoption 문서의 과거 배치를 최신 v2 기준으로 대체한다.
5. 1.3에 EP01이 RQ3·RQ4의 통합 답변임을 예고하는 비결과형 문장 한 개를 둔다.

이 감사의 `수정 후 표현`은 기존 문서의 같은 항목보다 우선하는 초안 작성 규칙이다. 사용자가 산출물을 세 파일로 제한했으므로 기존 v2·map·adoption 파일 자체는 수정하지 않는다.

## 1. 수정 불가 자기 규정 3문장

1. 본 논문은 평가한 산업 결함 Active Learning 분기에서 selection-stage discovery, selected-set composition, fixed-set training stability, acquisition reproducibility, downstream learner utility, rare-class safety를 분리하고, 이들 사이의 연결이 어디까지 관찰되며 어디서 제한되는지를 정리한 bounded empirical audit이다.
2. 본 논문은 특정 selector의 일반적 우월성, 선택–학습 연결을 예측하는 검증된 framework, 산업 benchmark, 실제 비용 절감 또는 독립 production pool 일반화를 입증하는 연구가 아니다.
3. 본 논문의 기여는 새로운 선택 방법의 제안이 아니라, 평가한 분기 안에서 선택 단계와 후속 학습의 평가 단위 및 증거 상태를 분리하고 그 연결의 관찰 범위와 제한을 실증적으로 정리하는 데 있다.

## 2. 초안 작성 시 문서 우선순위

| 우선순위 | 문서 | 사용할 내용 | 사용하지 않을 내용 |
|---|---|---|---|
| 1 | `thesis_identity_statement.md` | 자기 규정 3문장, 용어 정의, 주장 경계 | 없음 |
| 2 | 본 감사 문서와 CSV | 수정 후 표현, 용어 규칙, RQ·anchor 매핑 | 수정 전 표현 |
| 3 | `chapter1_introduction_skeleton.md`, `chapter2_related_work_skeleton.md` | 제1·2장 문단 순서와 anchor | 본 감사에서 교정한 용어의 이전 표현 |
| 4 | `chapter4_writing_packets_v2.*`, `chapter5_writing_packets_v2.*` | 수치·그림·표·문단 순서·장별 역할 | 본 감사에서 교정한 4.1·safety·5.3 표현 |
| 5 | `final_claim_adoption_*` | `논문용 최종 주장 문장` 열의 EP01–EP05 | `현재 주장 문장` 열과 구버전 장절 배치 |
| 6 | `claim_to_outline_*` | claim의 결과/논의/결론 역할 원칙 | 최신 v2와 다른 구버전 제목 |
| 7 | `cross_chapter_consistency_audit.*` | 수정 이력과 위험 근거 | 과거 문제 표현을 현재 제목·문장으로 재사용 |

## 3. 문서별 자기 규정 정합성

| 문서 | 무엇인가 문장과의 정합성 | 무엇이 아닌가 문장과의 정합성 | 기여 층위와의 정합성 | 판정 | 필요한 조치 |
|---|---|---|---|---|---|
| `thesis_identity_statement.md` | 기준 문서 | 기준 문서 | 기준 문서 | 통과 | 수정 금지 |
| `chapter1_introduction_skeleton.md` | 동일 자기 규정을 사용함 | 비범위가 명시됨 | 새 방법이 아닌 평가 단위 분리로 고정됨 | 조건부 통과 | EP01의 RQ3·RQ4 통합 역할을 알리는 예고문 1개 추가 |
| `chapter2_related_work_skeleton.md` | 2.5에서 원문 그대로 사용함 | 경계 문장을 절별로 유지함 | 네 축을 방법 통합이 아닌 평가 질문으로 연결함 | 통과 | 현재 문구 유지 |
| `chapter4_writing_packets_v2.md` | 관찰 역할과 대체로 정합 | 우월성·framework·일반화 금지가 명시됨 | 새 방법 주장을 하지 않음 | 수정 필요 | 4.1 `타당성` 제거; 4.5 KSDD2 `safety check` 구체화 |
| `chapter4_writing_packets_v2.csv` | 관찰 역할과 정합 | 금지 표현 열은 경계로 기능함 | 새 기여를 주장하지 않음 | 수정 필요 | 4.1 purpose의 `타당성 검증` 삭제; 4.5 safety 용어 구체화 |
| `chapter5_writing_packets_v2.md` | 분기별 제한 해석과 정합 | framework·benchmark·cost·production 일반화가 금지됨 | 해석·한계 역할을 유지함 | 수정 필요 | 5.3 용어 통일; 5.1 최소효과 기준의 시간적 지위 명시; 5.6 safety 용어 구체화 |
| `chapter5_writing_packets_v2.csv` | 분기별 제한 해석과 정합 | 금지 표현 열은 경계로 기능함 | 독립 기여 확대를 금지함 | 수정 필요 | 5.3·5.6 표현을 MD와 함께 통일 |
| `claim_to_outline_map.md` | 역할 분리 원칙은 정합 | 과장 금지는 정합 | 구버전 제목이 v2보다 강함 | 수정 필요 | 4.1·4.3·4.5·5.3·5.6 표기를 최신 v2와 본 감사 기준으로 대체 |
| `claim_to_outline_table.csv` | claim 역할은 정합 | 과장 금지는 정합 | 4.3·4.5 제목과 safety 명칭이 구버전임 | 수정 필요 | 최신 v2 제목 및 구체 지표명 사용 |
| `final_claim_adoption_memo.md` | 최종 claim 열은 정합 | 금지 표현과 취약점이 명시됨 | 최종 문장은 안전함 | 조건부 통과 | 과거 `현재 주장 문장`과 구버전 장절 위치를 초안에 사용하지 않음 |
| `final_claim_adoption_table.csv` | 최종 claim 열은 정합 | 금지 표현 열은 정합 | `현재 주장 문장`은 최종 문장보다 강함 | 조건부 통과 | `논문용 최종 주장 문장` 열만 claim 원문으로 사용 |
| `related_work_positioning_memo.md` | bounded empirical audit를 명시함 | 방법·비용·일반화 경계가 명시됨 | 평가 단위 분리로 기여를 제한함 | 경미한 수정 필요 | `안전한 composition`, `별도 안전성 판정`을 구체 지표 표현으로 축소 |
| `contribution_positioning_table.csv` | 기여·경계가 정합 | framework·benchmark·cost 경계가 명시됨 | safe contribution 수준을 유지함 | 표기 수정 필요 | 산업 결함 anchor의 `IND-`를 제1·2장 표기인 `IDD-`로 정규화 |
| `cross_chapter_consistency_audit.md` | 현재 v2를 만든 진단 근거임 | 과장 위험을 명시함 | 과거 문제를 기록함 | 역사 자료로 통과 | 옛 제목과 문장을 초안 source로 사용하지 않음 |
| `cross_chapter_consistency_audit.csv` | 현재 v2를 만든 진단 근거임 | 과장 위험을 명시함 | 과거 문제를 기록함 | 역사 자료로 통과 | 수정 전 표현을 현재 지침으로 해석하지 않음 |

위험 키워드가 선행연구의 실제 논문 제목, 금지 표현 목록, 부재·미검증 설명에 등장하는 것은 identity 위반이 아니다. 위반 판정은 해당 표현이 본 연구의 제목·목적·허용 문장·요약 문장으로 사용될 때만 적용했다.

## 4. 필수 표현 봉합표

| 대상 문서·위치 | 위험 유형 | 수정 전 표현 | 수정 후 표현 | 조치 수준 | 이유 |
|---|---|---|---|---|---|
| Ch4 v2 MD, 4.1 목적 | 타당성·검증 | “이 절은 타당성 검증, 다운스트림 예측력 또는 selector 우월성을 판단하는 절이 아니다.” | “이 절은 선택 단계 측정 결과만 제시하며, 다운스트림 예측력 또는 selector 우월성은 판단하지 않는다.” | 필수 | 부정문이더라도 4.1의 역할을 `타당성` 축으로 정의하지 않기 위함 |
| Ch4 v2 MD, 4.1 문단 1 | claim 강도 상승 | “후보 신호의 타당성을 각 분기의 선택 단계 native endpoint와 Random 대비 차이로만 제시한다.” | “후보 신호의 선택 단계 측정 결과를 각 분기의 native endpoint와 Random 대비 차이로 제시한다.” | 필수 | 측정 결과를 construct validity 검증처럼 읽는 위험 제거 |
| Ch4 v2 CSV, 4.1 purpose | 타당성·검증 | “타당성 검증, 다운스트림 예측력 또는 selector 우월성을 판단하지 않는다.” | “선택 단계 측정 결과만 제시하며 다운스트림 예측력 또는 selector 우월성은 판단하지 않는다.” | 필수 | MD와 CSV의 역할 문장 통일 |
| Ch4 v2 MD·CSV, 4.5 목적·표·문단·허용 표현 | safety 의미 충돌 | `supporting·safety check`, `Safety checks`, `보조·안전성 관찰값` | `보조 지표·정상 이미지 FPR 점검`, `정상 이미지 FPR 점검`, `보조 지표와 정상 이미지 FPR 관찰값` | 필수 | `rare-class safety`는 희소 클래스 평가 항목이고 KSDD2 FPR은 다른 점검이므로 분리 |
| Ch5 v2 MD·CSV, 5.3 제목 | 재현성 단위 모호성 | “학습 안정성과 선택 재현성의 구분” | “고정 선택 집합의 training-seed 안정성과 acquisition reproducibility의 구분” | 필수 | 선택 재현성을 독립 pool 재현성으로 오독하는 위험 제거 |
| Ch5 v2 MD·CSV, 5.3 본문·요약 | 용어 변형 | `training-seed 안정성`, `selector 재실행을 포함한 acquisition reproducibility`를 혼용 | 첫 언급은 `고정 선택 집합의 training-seed 안정성(fixed-set training stability)`과 `동일 fixed pool에서의 acquisition reproducibility`로 정의하고 이후 축약 | 필수 | 자기 규정의 두 반복 단위를 정확히 고정 |
| Ch5 v2 MD·CSV, 5.1 | 최소효과 기준의 시간적 지위 누락 | “평균 최소효과 기준” | “결과 확인 전에 고정한 평균 최소효과 기준” | 필수 | 사후 threshold처럼 읽히는 위험 제거 |
| Ch5 v2 MD·CSV, 5.6 교차참조 | safety 의미 충돌 | “보조·안전성 수치” | “보조 지표와 정상 이미지 FPR 수치” | 필수 | rare-class 평가와 KSDD2 FPR 점검을 분리 |
| Ch5 v2 MD, 5.7 요약 | audit 의미 확대 가능성 | “선택 단계와 후속 평가의 일치·불일치를 감사한 제한적 실증 연구” | “평가한 분기에서 선택 단계와 후속 평가의 단위 및 증거 상태를 분리하고 연결의 관찰 범위와 제한을 정리한 제한적 실증 연구” | 권고 | `audit`를 formal framework validation처럼 읽는 위험 축소 |
| Outline map MD, 4.1 | 구버전 제목 | “후보 선택 신호의 타당성” | “후보 선택 신호별 선택 단계 관찰” | 필수 | 최신 v2와 identity의 관찰 역할 반영 |
| Outline map MD·CSV, 4.3 | 구버전 결과 제목 | “고정 집합 학습 안정성과 획득 재현성” | “고정 선택 집합 재학습과 acquisition 반복 결과” | 필수 | 결과 장에서 해석을 선취하지 않음 |
| Outline map MD·CSV, 4.5 | 구버전 결과 제목 | “통계적 양의 효과와 사전 최소효과 기준” | “KSDD2 Dice 차이와 사전 최소효과 기준 판정” | 필수 | 인과적 effect·success 인상을 제거 |
| Outline map MD, 5.3 | 용어 모호성 | “학습 안정성과 선택 재현성의 구분” | “고정 선택 집합의 training-seed 안정성과 acquisition reproducibility의 구분” | 필수 | 자기 규정 용어와 일치 |
| Outline map MD, 5.6 | 범위·역할 정렬 | “선택 단계와 다운스트림 성능 간 연결의 제한 조건” | “선택–학습 연결 해석의 범위와 제한 조건” | 권고 | 최신 v2 제목과 동일하게 유지 |
| Outline map/table, EP05 표명 | safety 의미 충돌 | `primary·supporting·safety check 표` | `primary Dice·보조 지표·정상 이미지 FPR 점검 표` | 필수 | 두 safety 개념 분리 |
| Final adoption memo/table, `현재 주장 문장` | 채택 전 과장 표현 | `Gate PASS를 보장하지 않았다`, `composition safety를 보장하지 않았다`, `rare-class safety의 악화`, `양의 Dice effect` 등 | 각 행의 `논문용 최종 주장 문장`을 그대로 사용 | 필수 | 채택 전 문장은 provenance이며 본문용 문장이 아님 |
| Final adoption memo/table, 본문 위치 | 구버전 배치 | EP02 5.2, EP03 5.3, EP04 6.1, EP05 6.3 중심 표기 | EP02 4.2/5.2/6.1; EP03 4.3/5.3/6.1; EP04 4.4/5.4/6.1; EP05 4.5/5.6/6.1 | 필수 | 최신 목차·v2 역할 분리 반영; 6.3은 현 목차에 없음 |
| Positioning memo, VFM-03 접점 | safety 뉘앙스 | “안전한 selected-set composition” | “범주 구성 기준을 충족하는 selected-set composition” | 권고 | 산업 안전 개념의 유입 방지 |
| Positioning memo, 희소 클래스 비교·답변 | safety 뉘앙스 | “별도 안전성 판정”, “aggregate improvement를 rare-class safety로 번역하지 않는” | “희소 클래스 지표의 별도 판정”, “aggregate improvement를 희소 클래스 지표 개선으로 간주하지 않는” | 권고 | `rare-class safety`를 구체 평가 항목으로 조작화 |
| Contribution positioning CSV, 산업 anchor | anchor 표기 불일치 | `IND-01`–`IND-04` | 제1·2장 표시에서는 `IDD-01`–`IDD-04` | 필수 | skeleton의 요청 표기와 일치; 문헌 자체는 동일 |
| Cross-chapter audit MD·CSV | 역사적 표현 | 4.1 `타당성`, 4.3·4.5 구버전 제목 | 변경 이력으로만 보존하고 초안에는 최신 v2와 본 감사의 표현 사용 | 사용 규칙 | 역사적 진단 자료를 현행 집필 지침으로 오독하지 않기 위함 |

## 5. 위험 키워드 판정 규칙

| 위험 표현 | 허용되는 경우 | 본 연구를 서술할 때 사용할 표현 | 금지되는 사용 |
|---|---|---|---|
| 타당성·유효성 | 인용한 선행연구의 개념 또는 `유효하지 않다`가 아닌 자료 상태 설명 | 선택 단계 측정 결과, 관찰된 차이 | 후보 신호의 타당성·유효성 입증 |
| 검증 | 선행연구 원문 설명, 미검증·추가 검증 필요라는 경계 | 평가했다, 관찰했다, 확인했다, 판정했다 | framework·selector·Gate를 검증했다 |
| framework | 인용 논문의 정식 명칭 또는 금지 경계 | 분석·보고 체계, 평가 단위 구분 | 본 연구의 검증된 framework |
| benchmark | 인용 논문의 정식 명칭 또는 본 연구가 아님을 명시 | 평가한 분기, 제한적 실증 감사 | 산업 benchmark 또는 method ranking |
| 예측 정확도 | 식별하지 못했다는 한계 | 분기별 기준 판정 | Gate precision·recall·예측 정확도 입증 |
| 비용 절감 | 측정하지 않았다는 한계 | image-count budget, 비용 원장 부재 | annotation·GPU·금액 절감 입증 |
| 일반화 | 일반화할 수 없다는 경계 또는 향후 검증 필요 | 평가한 분기·동일 fixed pool 범위 | 독립 production pool 또는 산업 전반 일반화 |

## 6. 핵심 용어 통일 규칙

| 용어 | 통일된 논문 표현 | 결과 장 사용 | 논의·결론 사용 | 금지되는 뉘앙스 |
|---|---|---|---|---|
| `rare-class safety` | 자기 규정에서 `rare-class safety(희소 클래스 지표의 별도 판정)`로 1회 정의 | `희소 클래스 macro AP`, `희소 클래스 기준`, `희소 클래스 성능 지표`처럼 관찰값으로 씀 | aggregate 지표와 별도 판정했다는 범위에서만 사용 | rare-class harm, safety hazard, 산업 안전성 입증; KSDD2 FPR check와 혼용 |
| `acquisition reproducibility` | `동일 fixed pool에서 acquisition seed별로 선택을 재실행한 결과` | 4.3에서는 관찰형 `acquisition 반복 결과` 사용 가능 | 5.3 첫 언급에서 formal term을 쓰고 동일 fixed pool 범위를 병기 | independent production pool 재현성, selector의 일반적 재현 성공·실패 |
| `fixed-set training stability` | `고정 선택 집합의 training-seed 안정성` | 4.3에서는 `고정 선택 집합 재학습 결과`로 관찰 | 5.3에서 선택 집합을 조건으로 한 학습 변동성으로 해석 | acquisition reproducibility와 동의어, confirmatory selector 우위 |
| `sufficient condition` | 본문 claim으로 사용하지 않고 EP01의 확정 문장으로 대체 | 사용 금지 | 범위 없는 보편 명제는 사용 금지 | “선택 신호는 downstream utility의 충분조건이 아니다”라는 일반 법칙 |
| `minimum-effect criterion` | KSDD2 첫 언급은 `결과 확인 전에 고정한 평균 Dice 최소효과 기준`; 이후 `사전 최소효과 기준` | Dice 차이·CI와 기준 충족 여부를 병렬 보고 | 통계적 방향, 사전 기준, 외부 실용성을 구분 | 산업 최적 threshold, 비용 보정 기준, 기준 미충족=무효과 |

`safety`라는 단어는 서로 다른 두 평가를 묶지 않는다. EP04 관련 희소 클래스 평가는 `희소 클래스 지표의 별도 판정`으로, KSDD2의 normal FPR은 `정상 이미지 FPR 점검`으로 쓴다.

## 7. 연구 질문–claim 대응

### 7.1 연구 질문별 직접 답변

| 연구 질문 | 직접 답하는 claim | 보조·범위 claim | 결과 위치 | 해석 위치 | 미연결 여부 |
|---|---|---|---|---|---|
| RQ1 선택 단계 | EP02 | EP01의 통합 입력으로만 사용 | 4.1 전제, 4.2 | 5.2 | 없음 |
| RQ2 반복 단위 | EP03 | EP03의 post-hoc·fixed-pool 경계는 RQ4에도 연결 | 4.3 | 5.3 | 없음 |
| RQ3 후속 평가 | EP04, EP05, EP01의 통합 답변 | EP03의 acquisition learner-effect 반복 | 4.3–4.5 | 5.1, 5.3, 5.4, 5.6 | 없음 |
| RQ4 증거 경계 | EP01, EP03, EP05 | EP02의 `NOT_AVAILABLE`·fixed-pool, EP04의 point-estimate·단일 분기 경계 | 상태 표기 전역 | 5.1, 5.3, 5.5–5.7 | 없음 |

### 7.2 Claim별 회수 여부

| claim_id | 연결 연구 질문 | 관계 | 서론 예고 상태 | 추가 조치 |
|---|---|---|---|---|
| EP01 | RQ3, RQ4 | 두 질문의 통합 답변 | 연구 질문 개별 문구에는 포함되지만 통합 역할이 암묵적 | 1.3 연구 질문 목록 직후에 아래 예고문 1개 추가 |
| EP02 | RQ1; RQ4 보조 | 선택 단계의 직접 답변과 증거 경계 | 충분히 예고됨 | 추가 RQ 불필요 |
| EP03 | RQ2; RQ3 보조; RQ4 | 반복 단위의 직접 답변과 evidence-status 경계 | 충분히 예고됨 | 추가 RQ 불필요 |
| EP04 | RQ3; RQ4 보조 | aggregate·rare 지표의 직접 답변 | 충분히 예고됨 | 추가 RQ 불필요 |
| EP05 | RQ3, RQ4 | Dice 방향·사전 기준·시간적 지위의 직접 답변 | 충분히 예고됨 | 추가 RQ 불필요 |

어떤 연구 질문에도 걸리지 않는 claim은 없다. 새 연구 질문은 필요하지 않다.

### 7.3 1.3에 추가할 비결과형 예고문

> 이들 연구 질문은 개별 평가 단위의 관찰뿐 아니라, 선택 단계에서 양의 방향의 결과가 나타난 분기에서 사전에 정한 다운스트림 평가 기준이 어떤 판정을 내렸는지를 평가 분기별로 함께 확인하도록 구성하였다.

이 문장은 결과 방향을 선취하지 않으며 EP01을 RQ3·RQ4의 통합 답변으로 예고한다. EP01의 확정 결론문은 1장에서 사용하지 않는다.

## 8. 선행연구 anchor의 최초 등장과 재사용

| anchor_id | 제1장 최초 등장 | 제2장 주 재사용 | 2.5 재사용 | 정합성 판정 |
|---|---|---|---|---|
| ALR-01 | 1.2 문단 1 | 2.4 문단 1·5 | 2.5 문단 1 | 정합 |
| ALR-02 | 1.2 문단 2 | 2.4 문단 2·5 | 2.5 문단 1 | 정합 |
| ALR-03 | 1.2 문단 2 | 2.4 문단 3·5 | 2.5 문단 3의 benchmark 경계 | 정합 |
| ALR-04 | 1.2 문단 1 | 2.4 문단 4·5 | 2.5 문단 1·3 | 정합 |
| ODA-01 | 1.2 문단 3 | 2.2 문단 1·5 | 직접 재사용 없음 | 정합 |
| ODA-02 | 1.2 문단 3 | 2.2 문단 2·5 | 2.5 문단 1 | 정합 |
| ODA-03 | 1.2 문단 3 | 2.2 문단 3·5 | 2.5 문단 1 | 정합 |
| ODA-04 | 1.1 문단 4 | 2.2 문단 4·5 | 2.5 문단 1·3 | 정합 |
| VFM-01 | 1.1 문단 3 | 2.3 문단 1·5 | 직접 재사용 없음 | 정합 |
| VFM-02 | 1.1 문단 3 | 2.3 문단 2·5 | 2.5 문단 1 | 정합 |
| VFM-03 | 1.1 문단 4 | 2.3 문단 3·5 | 2.5 문단 1 | 정합 |
| VFM-04 | 1.1 문단 4 | 2.3 문단 4·5 | 직접 재사용 없음 | 정합 |
| IDD-01 | 1.1 문단 1 | 2.1 문단 1·4 | 직접 재사용 없음 | 정합 |
| IDD-02 | 1.1 문단 1 | 2.1 문단 1·2·4 | 2.5 문단 1 | 정합 |
| IDD-03 | 1.1 문단 2 | 2.1 문단 3·4 | 직접 재사용 없음 | 정합 |
| IDD-04 | 1.1 문단 2 | 2.1 문단 2·4 | 2.5 문단 1·3 | 정합 |

각 anchor는 제1장에서 문제를 예고하고 제2장의 해당 축에서 적어도 한 번 직접 회수된다. 2.5에서 모든 anchor를 다시 열거할 필요는 없으며 대표 anchor만 통합한다. 새 anchor는 필요하지 않다.

## 9. EP01과 결론 6.1의 논리 정합성

### 확정 문장

> 선택 단계와 다운스트림 결과가 모두 관찰된 세 평가 분기에서, 선택 단계에서 양의 방향으로 나타난 결과만으로는 사전에 정한 다운스트림 평가 기준의 충족을 확인할 수 없었다.

| 점검 항목 | 판정 | 이유 |
|---|---|---|
| 평가 범위가 bounded한가 | 통과 | `모두 관찰된 세 평가 분기`로 모집단을 명시함 |
| selector 우월성·열등성을 주장하는가 | 통과 | selector ranking이나 방법 우열을 말하지 않음 |
| framework·Gate 예측 정확도를 주장하는가 | 통과 | 사전 기준의 충족 여부만 기술함 |
| 충분조건의 보편 법칙으로 확대되는가 | 통과 | 세 분기의 관찰로 한정되고 `충분조건`이라는 보편 용어를 쓰지 않음 |
| 미관찰 STOP 분기를 실패로 포함하는가 | 통과 | 선택 단계와 downstream 결과가 모두 있는 분기만 포함함 |
| 비용·production pool 일반화를 주장하는가 | 통과 | 비용·외부 pool에 관한 문장이 없음 |
| 자기 규정의 기여 층위와 맞는가 | 통과 | 평가 단위 연결의 관찰 범위를 제한적으로 요약함 |

사용 조건은 세 가지다. EP02–EP05의 데이터셋별 요약 뒤 6.1 마지막에 한 번만 쓴다. `0/3`, 실패율, 모든 selector, 충분조건, 비용 및 일반화 문장을 덧붙이지 않는다. 5.1에서는 이 문장을 반복하지 않는다.

## 자기 규정을 위반하는 남은 표현 목록과 수정안

| 남은 표현 | 위치 | 수정안 | 초안 전 상태 |
|---|---|---|---|
| “후보 신호의 타당성” | Ch4 v2 4.1 문단 지침; 구버전 outline map 4.1 | “후보 신호의 선택 단계 측정 결과”; 제목은 “후보 선택 신호별 선택 단계 관찰” | 반드시 적용 |
| `supporting·safety check` | Ch4 v2 4.5, Ch5 v2 5.6, outline map/table | `보조 지표·정상 이미지 FPR 점검` | 반드시 적용 |
| “학습 안정성과 선택 재현성” | Ch5 v2 5.3, outline map | “고정 선택 집합의 training-seed 안정성과 acquisition reproducibility” | 반드시 적용 |
| “평균 최소효과 기준” | Ch5 v2 5.1 | “결과 확인 전에 고정한 평균 최소효과 기준” | 반드시 적용 |
| 4.3·4.5·5.6 구버전 제목 | outline map/table | 최신 v2 제목으로 대체 | 반드시 적용 |
| 채택 전 `Gate PASS`, `composition safety`, `rare-class safety 악화`, `양의 effect` 문장 | final adoption의 `현재 주장 문장` 열 | 같은 행의 `논문용 최종 주장 문장`만 사용 | 반드시 적용 |
| `안전한 selected-set composition`, `별도 안전성 판정` | positioning memo | `범주 구성 기준을 충족하는 selected-set composition`, `희소 클래스 지표의 별도 판정` | 권고 적용 |
| `IND-01`–`IND-04` | contribution positioning CSV | 제1·2장 표시에서는 `IDD-01`–`IDD-04` | 반드시 적용 |

금지 목록, 선행연구 정식 제목, 과거 수정 이력에 포함된 `검증`, `framework`, `benchmark`, `일반화`, `비용 절감`은 본 연구의 주장으로 사용된 것이 아니므로 별도 위반으로 세지 않는다.

## 서론 연구 질문 4개와 EP01–EP05 매핑 요약

- RQ1은 EP02가 직접 답한다.
- RQ2는 EP03이 직접 답한다.
- RQ3은 EP04·EP05가 직접 답하고 EP03이 보조하며, EP01이 분기 간 통합 답변을 제공한다.
- RQ4는 EP01·EP03·EP05가 직접 답하고 EP02·EP04가 각 claim의 증거 경계를 보조한다.
- 어떤 RQ에도 연결되지 않는 claim은 없다. 새 RQ는 추가하지 않는다.
- EP01의 통합 역할만 1.3의 비결과형 예고문으로 명시하고, EP01 결론문 자체는 6.1 마지막에만 둔다.

## 초안 작성 직전 최종 통과 여부 판정 (통과 / 조건부 통과 / 재봉합 필요)

**조건부 통과.** Claim·RQ·anchor 구조에는 누락이나 논리 충돌이 없다. 그러나 기존 v2와 구버전 map의 원문이 물리적으로 수정된 것은 아니므로, 본 감사의 필수 `수정 후 표현`을 초안 source에 적용하기 전에는 완전 통과로 판정하지 않는다. 특히 4.1의 `타당성`, KSDD2의 `safety check`, 5.3의 `선택 재현성` 세 항목은 초안에 그대로 유입되면 안 된다.

## 초안을 어떤 절부터 시작하는 것이 가장 안전한지 순서 권고

1. **4.2** — EP02의 선택 단계 관찰과 learner `NOT_AVAILABLE` 경계가 이미 명확하다.
2. **4.4** — EP04를 전체·희소 클래스 지표의 관찰 방향으로 제한하기 쉽다.
3. **4.5** — `safety check`를 정상 이미지 FPR 점검으로 바꾼 뒤 작성한다.
4. **4.3** — `POSTHOC_DESCRIPTIVE`와 동일 fixed pool 경계를 첫 문장부터 고정한 뒤 작성한다.
5. **4.1** — `타당성` 표현을 모두 제거한 뒤 평가 전제만 작성한다.
6. **5.2 → 5.4 → 5.5 → 5.3 → 5.6 → 5.1 → 5.7** — 직접 claim 해석부터 시작하고 통합·한계 절은 뒤에 쓴다.
7. **제1장 → 제2장** — 결과·논의의 실제 경계가 문장에 반영된 뒤 skeleton을 문단으로 확장한다.
8. **6.1** — EP02–EP05 요약 뒤 EP01 통합 문장을 마지막 한 번만 작성한다.

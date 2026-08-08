# Thesis defense readiness and non-regression contract

Date: 2026-07-18
Evidence decision: **A. THESIS_CORE_STRENGTHENED_BY_PROSPECTIVE_STOP**

## 1. 무엇이 추가로 강해졌는가

DeepPCB는 positive method 성능을 추가하지 않았다. 대신 외부 데이터에서 protocol·score·gate·selection hash를 먼저 고정하고, G1의 양의 relation 이후에도 G2 primary FAIL을 수용했으며, secondary signal을 mechanism/development audit으로 격리한 뒤 detector와 official test를 사용하지 않은 end-to-end authorization trace를 추가했다. 따라서 retrospective workflow라는 기존 약점은 일부 보완됐지만 predictive policy의 정확도는 여전히 식별되지 않는다.

## 2. 심사 전 수용 조건

| 질문 | 현재 답 | 교수 승인 필요성 |
|---|---|---|
| 학위 기여가 반드시 새 selector의 성능 우위여야 하는가? | 현재는 충족하지 못함 | **필수** |
| validity-gated empirical evaluation과 failure-condition map을 중심 기여로 인정할 수 있는가? | 원자료와 prospective STOP 사례가 존재 | **필수** |
| predictive accuracy를 주장하지 않는 범위를 수용하는가? | NOT IDENTIFIABLE로 고정 | **필수** |
| DeepPCB 추가 실험 없이 branch closure를 수용하는가? | exact branch closed | **필수** |

## 3. Non-regression contract

- `1.307591`을 confirmatory PASS로 승격하지 않는다.
- 577-1024 px² 관계를 tiny defect 전반으로 확대하지 않는다.
- DeepPCB detector improvement를 주장하지 않는다.
- literature method의 일반 무효를 주장하지 않는다.
- documented prevented model runs lower bound 45에 DeepPCB 수를 임의 추가하지 않는다.
- DeepPCB score/threshold rescue, detector training/inference, official/final test 접근을 금지한다.

## 4. 논문 완성도를 높이는 허용 작업

새 실험 없이 허용되는 작업은 evidence locator 검증, 표·그림과 본문 수치의 일치 검사, claim-evidence crosswalk, 지도교수용 구두 설명과 한계 문구 정제, DOCX/PDF 렌더링 QA다. 새로운 성능 탐색은 본 freeze의 보완이 아니라 별도 연구 질문이므로 분리해야 한다.

## 5. 최종 현실적 판정

현재 패키지는 “positive selector 논문”으로는 부족하지만 “산업 결함 AL 후보 신호가 discovery·safety·learning utility로 번역되는지를 단계적으로 제한하고, 불충분한 positive signal의 확장을 실제로 중단한 실증 workflow”로는 방어 가능한 토대가 있다. 원점 회귀를 막는 핵심은 추가 성능이 아니라 지도교수와 이 기여 형태를 먼저 합의하는 것이다.

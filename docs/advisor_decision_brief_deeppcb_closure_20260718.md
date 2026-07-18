# 지도교수 결정 보고: DeepPCB closure와 Evidence Freeze v3

## 요청드리는 결정

> Positive selector 성능이 아니라, 후보 신호가 discovery·safety·learning utility로 번역되는지 단계적으로 검증하고 불충분한 branch의 학습·official test·주장을 중단하는 empirical workflow를 학위논문의 중심 기여로 인정할 수 있는지 결정 부탁드립니다.

## 이번에 새로 추가된 근거

기존 논문의 약점은 generic workflow가 모든 주요 실험 종료 후 정식화됐다는 점이었습니다. DeepPCB에서는 외부 paired-reference 데이터를 대상으로 protocol, score, threshold, query fraction, eligible groups와 selection hash를 먼저 고정했습니다. G1 relation은 양성이었지만 primary G2 total bbox enrichment가 1.107693으로 frozen 1.20을 넘지 못해 `FAIL_STOP`을 수용했습니다.

이후 small enrichment 1.307591을 성공으로 바꾸지 않고 read-only mechanism audit으로 분리했습니다. Small-share uplift는 1.1828이었지만 positive excess의 45.79%가 한 group에 집중되어 `A2_MECHANISM_AMBIGUOUS`였습니다. Reserved development group에서 AbsDiff, SSIM, fused residual을 비교했지만 `NO_CANDIDATE`였습니다. 이에 external confirmation과 detector screen을 승인하지 않았고 detector training/inference와 official/final test use는 0으로 유지했습니다.

## 왜 thesis core가 강화되는가

1. **Prospective process evidence**: 외부 branch에서 결과 전 protocol·selection freeze가 실제 적용됐습니다.
2. **Claim-boundary enforcement**: 양의 secondary signal을 parent PASS나 tiny-defect generalization으로 승격하지 않았습니다.
3. **Authorization evidence**: mechanism과 development gate 실패 후 고비용 detector 단계가 실제로 중단됐습니다.
4. **Reproducibility**: selection hash, source locator, frozen decision, code와 integrity tests가 남아 있습니다.

이는 predictive screening accuracy를 증명하지 않습니다. 한 STOP case만으로 early-stop recall이나 good-method false-stop sensitivity를 계산할 수 없습니다. 강화된 기여는 success prediction이 아니라 prospective conformance와 cost containment입니다.

## 최종 연구 판정

**A. THESIS_CORE_STRENGTHENED_BY_PROSPECTIVE_STOP**

이 판정은 positive 성능을 의미하지 않습니다. 외부 데이터에서도 불충분한 positive signal을 detector-success 주장으로 확대하지 않는 workflow가 실제로 작동했다는 뜻입니다.

## 교수님께 보고할 1분 설명

“교수님, 기존 결과를 사후적으로만 정리했다는 약점을 보완하기 위해 DeepPCB 외부 paired-reference branch를 score와 gate를 먼저 고정해 실행했습니다. Score와 bbox 수의 관계는 양성이었지만 primary enrichment가 1.108로 기준 1.20을 넘지 못해 학습을 중단했습니다. Small-box 1.308배라는 secondary signal도 성공으로 바꾸지 않고 기전 감사를 했는데 한 group 의존성이 커 ambiguous였고, 별도 development group의 세 residual 방법도 외부 확인 후보를 만들지 못했습니다. 그래서 detector와 official test를 전혀 쓰지 않고 branch를 닫았습니다. 성능 좋은 selector를 만든 결과는 아니지만, 양의 일부 수치를 성공으로 과장하지 않고 외부 branch의 다음 단계 권한을 통제한 최초의 prospective 사례가 생겼습니다. 이 validity-gated empirical workflow와 failure-condition evidence를 학위논문의 중심 기여로 인정할 수 있는지 결정 부탁드립니다.”

## 승인 후 경로

- 승인 시: 새 selector 실험 없이 Evidence Freeze v3를 기준으로 논문 작성과 발표자료 정리에 들어갑니다.
- Positive algorithm이 필수라면: 현재 AL 논문에 사후 score search를 추가하지 않고, supervised reference-inspection/backbone 연구를 별도 주제·split·protocol로 분리합니다.
- DeepPCB branch는 어느 경우에도 재개하지 않습니다.

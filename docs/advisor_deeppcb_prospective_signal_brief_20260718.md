# 지도교수 보고용: DeepPCB Prospective Reference-Residual Signal 결과

## 한 문장 판정

새 외부 bbox 데이터에서 사전 고정한 reference-residual selection branch는 전체 annotation-richness gate에는 미달했지만, **local signal validity와 small-defect enrichment에서 일관된 양의 결과**를 보여 `reference-grounded small-defect annotation triage`라는 좁은 후속 가설을 정당화했다.

## 왜 이 실험을 추가했는가

기존 GC10 development는 반복 사용됐고 locked final은 접근할 수 없었다. MPDD와 VisA는 GC10 FN-box와 task가 달랐다. 따라서 기존 split을 다시 나누어 긍정 결과를 만드는 대신, 처음 사용한 외부 산업 bbox 데이터 DeepPCB에서 결과 확인 전 protocol과 gate를 고정했다.

- DeepPCB official trainval 1,000장만 사용
- 6개 source/design group 889장에서 selection-only audit
- `group92000` 111장은 future detector development용으로 예약
- official test 500장은 목록 hash만 봉인하고 image/annotation/evaluation 미사용
- detector training 0회
- detector inference 0회
- final test 사용 0회

## 사전 고정한 신호

DeepPCB는 defect image와 정렬된 clean template을 함께 제공한다. 두 이미지의 absolute residual에서 area와 connected-component count를 계산하고, group 내부 percentile rank 평균을 `reference_residual_richness`로 고정했다. GT annotation은 score와 top-20% selection을 저장·해시한 뒤에만 결합했다.

이 신호는 global DINO representation과 달리 제품 전체 형상보다 정렬된 local change를 직접 측정한다.

## 결과

| endpoint | 결과 | 95% CI | frozen 판정 |
|---|---:|---:|---|
| Signal–bbox count Spearman | **0.412174** | [0.290463, 0.540022] | G1 PASS; 6/6 group 양의 관계 |
| 전체 bbox instance enrichment@20% | **1.107693** | [1.048980, 1.171691] | 6/6 group > 1이지만 1.20 threshold 미달 |
| Small-box enrichment@20% | **1.307591** | [1.186109, 1.474079] | G3 safety endpoint PASS |
| Bbox-area enrichment | 1.073711 | [0.962446, 1.192316] | secondary, 불확정 |
| Six-class coverage delta | **0.000000** | [0, 0] | coverage 보존 |
| Max-class-share delta | **-0.002072** | [-0.020487, 0.017331] | concentration 악화 없음 |

## 냉정한 해석

전체 branch 판정은 `FAIL_STOP`이다. Primary G2는 전체 bbox instance enrichment 1.20을 요구했으나 1.107693이었다. 따라서 이 결과는 detector screen을 허가하지 않으며, Random보다 detector mAP가 높아진다고 말할 수 없다.

그러나 다음 결과는 결과 전 고정된 endpoint에서 관찰됐다.

1. local reference signal과 bbox richness의 양의 관계가 6/6 group에서 반복됐다.
2. 같은 image budget에서 전체 bbox는 Random보다 평균 10.8% 많았다.
3. 특히 32×32 이하 small box는 평균 30.8% 더 많이 포함됐고 CI lower bound도 1보다 컸다.
4. 여섯 클래스 coverage와 class concentration은 악화되지 않았다.

따라서 허용되는 주장은 다음이다.

> Aligned reference가 존재하는 산업 검사 조건에서는 global representation보다 local reference residual이 small-defect annotation triage의 유망한 후보 신호일 수 있다. 다만 현재 결과는 selection-only evidence이며, 독립 데이터와 사전 등록된 downstream 검증이 필요하다.

## 교수님께 말씀드릴 문장

> 교수님, 기존 데이터의 development split을 다시 사용하면 또 사후 분석이 되기 때문에, 처음 사용하는 외부 bbox 데이터인 DeepPCB에서 protocol과 threshold를 먼저 고정하고 selection-only 실험을 한 번 진행했습니다. 전체 gate는 통과하지 못했습니다. 전체 bbox 수는 Random보다 평균 10.8% 많았지만 사전에 정한 20% 기준에는 못 미쳤습니다. 다만 예상보다 명확했던 결과는 작은 결함 box가 평균 30.8% 더 많이 선택됐고, 95% 신뢰구간도 18.6%에서 47.4%로 전부 양수였다는 점입니다. 여섯 클래스 coverage와 특정 클래스 집중도도 악화되지 않았습니다. 아직 detector 성능 향상이라고 말할 수는 없지만, 기존 global DINO와 달리 정렬된 clean reference를 이용한 local residual 신호가 small-defect annotation triage에는 실제 가능성이 있다는 데이터는 얻었습니다. 이 결과를 논문의 성공 결과로 바로 승격하기보다, 현재 validity-gated 논문의 prospective 적용 사례이자 독립 데이터에서 검증할 좁은 후속 가설로 포함해도 될지 의견을 받고 싶습니다.

## 설득력이 있는 이유

- 기존 데이터의 반복 사용이 아니라 처음 사용한 외부 bbox 데이터다.
- protocol, signal, threshold, gate를 결과 전에 문서화했다.
- 6개 source/design group을 inferential unit으로 사용했다.
- 전체 gate 실패를 그대로 유지했다.
- primary 실패와 positive safety endpoint를 함께 제시한다.
- small-defect enrichment의 CI가 1보다 높다.
- official test와 reserved development group을 남겨 두었다.

## 피해야 할 표현

- “새 selector가 Random을 이겼다.”
- “detector 성능이 향상됐다.”
- “annotation cost가 30% 절감됐다.”
- “산업 데이터 전반에 일반화된다.”
- “전체 gate가 사실상 통과했다.”

## 교수님께 요청할 결정

1. 이 결과를 validity-gated workflow의 prospective external case study로 논문에 포함할지.
2. small-defect triage를 독립 데이터가 확보될 때의 후속 confirmatory hypothesis로 승인할지.
3. detector screen은 현재 gate에 따라 실행하지 않고, 별도 데이터와 사전 등록 protocol이 확보될 때만 허용할지.

## Evidence locator

- Protocol: `docs/deeppcb_reference_residual_prospective_protocol_20260718.md`
- Summary: `runs/deeppcb_reference_residual_gate/prospective_main_20260718/deeppcb_reference_residual_gate_summary.md`
- Group metrics: `runs/deeppcb_reference_residual_gate/prospective_main_20260718/posthoc_group_metrics.csv`
- Frozen selection: `runs/deeppcb_reference_residual_gate/prospective_main_20260718/frozen_selected_images.csv`
- Gate decision: `runs/deeppcb_reference_residual_gate/prospective_main_20260718/gate_decision.json`
- Integrity tests: `runs/deeppcb_reference_residual_gate/prospective_main_20260718/test_results.txt`


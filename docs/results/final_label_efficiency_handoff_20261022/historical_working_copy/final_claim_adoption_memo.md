# Final Claim Adoption Memo

## 결정 목적

이 메모는 기존 evidence packet 6개를 논문 본문에 실제 채택할 최종 주장 세트로 축소·고정한다. 새 주장을 추가하지 않았으며, `evidence_packets.md`, `evidence_packets.csv`, `thesis_viability_audit.md`, `thesis_asset_inventory.md`, `docs/thesis_claim_boundary_v3_20260718.md`의 범위 안에서만 판정했다.

최종 결정은 다음과 같다.

- **A — 본문 핵심 주장:** EP01, EP02, EP03, EP04, EP05
- **B — 제한적 사용:** 없음
- **C — 부록/보조 논의:** EP06
- **D — 삭제 권고:** 없음

EP01은 논문의 통합 결론을 담당하지만 반드시 “다운스트림 결과가 존재하는 세 평가 분기”로 범위를 고정한다. EP02–EP05는 논문이 구분하려는 네 평가 대상인 선택 집합 구성, acquisition 재현성, 희소 클래스 안전성, 사전 최소효과를 각각 담당한다. EP06은 EP01의 추가 사례로는 유효하지만, GC10 사례가 EP04와 중복되고 mAP50-95 신뢰구간이 0을 포함하므로 핵심 주장 수를 늘릴 만큼 독립적인 기능이 없다.

## 논문용 용어 고정

| 내부 용어 | 논문용 표현 | 사용 원칙 |
|---|---|---|
| translation branch | 평가 분기 또는 선택–학습 연결 평가 분기 | 처음에는 데이터셋·예산·selector·후속 모델이 고정된 평가 단위로 정의한다. |
| completed translation branch | 선택 단계와 다운스트림 결과가 모두 관찰된 평가 분기 | “완료”가 성공을 뜻하지 않도록 관찰 범위를 함께 쓴다. |
| full downstream Gate PASS | 사전에 정한 다운스트림 평가 기준 충족 | 여러 기준의 동시 충족임을 방법 장에서 정의한다. |
| positive endpoint | 양의 방향으로 나타난 선택 단계 결과 | 통계적 유의성이나 전체 기준 통과와 동일시하지 않는다. |
| STOP branch | 사전 기준 미충족으로 후속 평가가 중단된 분기 | learner outcome이 없으면 “학습 실패”라고 쓰지 않는다. |
| frozen criterion / frozen threshold | 결과 확인 전에 고정한 평가 기준 | 산업적으로 최적화된 기준이라는 의미를 부여하지 않는다. |
| bounded learner screen | 정해진 데이터·예산·모델 범위의 제한적 다운스트림 평가 | 외부 일반화나 최종 test 평가와 구분한다. |
| learner utility | 다운스트림 학습 모델의 성능 변화 | 가능한 경우 mAP, Dice, recall 등 실제 측정 지표를 바로 쓴다. |
| composition safety | 선택 집합의 범주 구성 기준 | 실제 작업 안전 또는 산업 안전으로 확대하지 않는다. |
| acquisition reproducibility | 선택 과정을 다른 acquisition seed에서 반복했을 때의 결과 재현성 | training-seed 안정성 및 독립 production pool 재현성과 구분한다. |

## Packet별 최종 판정표

| claim_id | 현재 주장 문장 | 논문용 최종 주장 문장 | 판정 등급 | 판정 이유 | 직접 근거 파일 | 핵심 수치 | 본문 위치(장/절) | 연결할 figure/table | 허용 표현 | 금지 표현 | 남는 취약점 또는 예상 반론 |
|---|---|---|:---:|---|---|---|---|---|---|---|---|
| EP01 | 평가가 완료된 세 translation branch에서 selection-stage의 positive endpoint는 full downstream Gate PASS를 보장하지 않았다. | **선택 단계와 다운스트림 결과가 모두 관찰된 세 평가 분기에서, 선택 단계에서 양의 방향으로 나타난 결과만으로는 사전에 정한 다운스트림 평가 기준의 충족을 확인할 수 없었다.** | A | 논문의 통합 결론을 담당한다. 대상이 세 분기뿐이고 서로 이질적이므로 “보장하지 않았다”를 “충족을 확인할 수 없었다”로 낮추고 평가 범위를 문장 안에 고정했다. | `docs/translation_atlas/translation_branch_inventory.csv`<br>`docs/translation_atlas/translation_atlas_statistics.json`<br>`runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_development_confirmation_gate.csv`<br>`runs/dcal_xai/v2_backbone_main/v8n_screen_decision.json`<br>`docs/results/ksdd2_g5_single_unet_20260727/g5_authorization_decision.json` | 관찰 완료 분기 `3`; 전체 기준 충족 `0/3`; GC10 q20 rare AP `-0.019877`; K40 mAP `-0.001678`; KSDD2 Dice `+0.037565 < +0.05` | 7장 7.1 “평가 분기별 선택–학습 연결 결과” | `docs/translation_atlas/translation_atlas_main_figure.png`<br>`docs/translation_atlas/four_fates_taxonomy.csv`<br>본문 표 “데이터셋 × 후보 신호 × 평가 단계” | “본 연구에서 결과가 모두 관찰된 세 평가 분기에서는”; “다운스트림 기준 충족을 확인할 수 없었다”; “분기별 native metric을 유지했다” | “selection gain은 learning으로 이어지지 않는다”; “실패율 100%”; “모든 selector가 실패했다”; “Gate가 좋은 selector를 판별했다” | n=3이며 task·metric·threshold·시간적 지위가 다르다. `0/3`은 확률이나 일반 실패율이 아니다. |
| EP02 | MPDD와 VisA의 frozen selection-only audit에서 anomaly discovery improvement는 selected-set composition safety를 보장하지 않았다. | **MPDD와 VisA의 선택 단계 평가에서 결함 이미지 발견량의 증가는 선택 집합의 범주 구성 기준 충족과 일치하지 않았다.** | A | Discovery와 composition을 분리해야 한다는 직접 근거이며 두 데이터셋에서 동일한 방향의 endpoint conflict가 존재한다. Learner가 실행되지 않았으므로 문장을 선택 단계 안에 한정했다. | `runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_metric_summary.csv`<br>`runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_selection_only_gate.csv`<br>`runs/visa_selection_only_audit/visa_random_vs_visual_200seed_20260715/visa_metric_summary.csv`<br>`runs/visa_selection_only_audit/visa_random_vs_visual_200seed_20260715/visa_selection_only_gate.csv` | MPDD anomaly images `+6.245`, CI `[5.885,6.605]`, coverage `-0.325 < -0.25`; VisA anomaly images `+14.480`, CI `[14.195,14.770]`, coverage `-4.110`; 각 `200` paired fixed-pool perturbations | 5장 5.2 “결함 발견량과 선택 집합 구성의 분리” | `docs/notion_professor_update_20260803/assets/figure06_discovery_composition_contrast.png`<br>`docs/translation_atlas/translation_atlas_main_figure.png`의 MPDD·VisA 행 | “결함 발견량은 증가했으나 범주 구성 기준은 충족하지 못했다”; “선택 단계의 두 지표가 일치하지 않았다” | “MPDD와 VisA에서 학습이 실패했다”; “DINO가 category collapse를 유발했다”; “200개 산업 pool에서 재현됐다”; “원인이 global representation이다” | Learner utility는 관찰하지 않았다. 200회는 독립 production pool이 아니라 같은 fixed pool의 paired perturbation이다. |
| EP03 | NEU-DET에서 한 favorable fixed selected set의 training-seed 안정성은 새로운 acquisition seed에서 유사한 learner gain이 재현됨을 의미하지 않았다. | **NEU-DET 평가에서 한 고정 선택 집합의 학습 seed 간 안정성은 선택 과정을 다른 acquisition seed에서 반복했을 때의 학습 효과 재현성과 동일하게 해석할 수 없었다.** | A | 논문 목표에 포함된 acquisition reproducibility를 직접 다룬다. 다만 fixed set이 사후 선택된 favorable 사례이므로 selector 실패가 아니라 재현성 단위의 구분으로만 채택했다. | `runs/seed45_fixed_set_stability/seed45_fixed_set_stability_main/paired_differences_vs_random.csv`<br>`runs/seed45_fixed_set_stability/seed45_fixed_set_stability_main/screen_gate.csv`<br>`runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main/confirmatory_metric_summary.csv`<br>`runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main/confirmatory_gate.csv` | 고정 집합 mAP delta `+0.016236`, training-seed wins `5/5`; acquisition 반복 mAP delta `+0.007019`, CI `[-0.005211,0.019678]`, wins `7/10`, `p=0.322266`, 사전 기준 `+0.010` 미달 | 5장 5.3 “학습 seed 안정성과 acquisition 재현성의 구분” | `docs/notion_professor_update_20260803/assets/figure05_neu_two_reproducibilities.png`<br>본문 표 “고정 선택 집합 재학습과 acquisition 반복 비교” | “서로 다른 재현성 대상이다”; “고정 집합에서는 안정적이었으나 acquisition 반복 결과는 사전 기준을 충족하지 못했다”; “고정 집합 결과는 사후 기술 분석이다” | “NEU에서 visual diversity가 재현에 실패했다”; “training-seed 안정성은 무의미하다”; “10 acquisition seeds가 independent pool 일반화를 입증했다”; “고정 집합 결과가 confirmatory PASS였다” | 고정 집합은 결과 관찰 후 선택된 favorable 사례다. Acquisition 반복도 같은 물리적 pool을 사용하므로 외부 일반화가 아니다. |
| EP04 | GC10 q20 branch에서는 overall detector performance의 개선과 rare-class safety의 악화가 동시에 관찰됐다. | **GC10 q20 평가에서 전체 mAP50-95는 증가했으나 희소 클래스 macro AP는 감소하여, 전체 성능과 희소 클래스 성능 지표가 상반된 방향을 보였다.** | A | 같은 downstream 실험에서 overall endpoint와 rare-class endpoint가 반대 방향을 보인 직접 사례다. “안전성 악화”보다 관찰된 지표 방향을 그대로 명시해 인과·유의성 과장을 줄였다. | `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_acquisition_seed_differences.csv`<br>`runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_acquisition_seed_summary.csv`<br>`runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_class_group_macro_summary.csv`<br>`runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_development_confirmation_gate.csv` | mAP50-95 delta `+0.017378`, CI `[0.000580,0.036332]`, wins `4/5`; recall `+0.035381`; frequent AP `+0.033345`; rare AP `-0.019877` | 6장 6.1 “전체 성능과 희소 클래스 지표의 충돌” | `docs/notion_professor_update_20260803/assets/figure04_gc10_endpoint_conflict.png`<br>`docs/translation_atlas/translation_atlas_main_figure.png`의 GC10 q20 행 | “전체 mAP는 증가했으나 rare macro AP는 감소했다”; “전체 개선이 모든 클래스 개선을 의미하지 않았다”; “희소 클래스 기준은 충족하지 못했다” | “overall mAP는 rare class를 항상 희생한다”; “DINO가 rare class를 악화시켰다”; “rare-class harm가 통계적으로 확정됐다”; “산업 안전 위험이 입증됐다” | Rare macro AP에는 별도 CI가 없고 한 GC10 설정의 point-estimate 결과다. 일반적인 harm로 확대할 수 없다. |
| EP05 | KSDD2의 bounded learner screen에서는 통계적으로 양의 Dice effect가 관찰됐지만 사전에 고정한 mean Dice minimum +0.05는 확립되지 않았다. | **KSDD2의 제한적 다운스트림 평가에서 Dice 차이는 양의 방향을 보였고 신뢰구간 하한도 0을 초과했지만, 결과 확인 전에 고정한 평균 Dice 차이 `+0.05` 기준은 충족하지 못했다.** | A | 가장 강한 prospective anchor이며 statistical positivity와 사전 decision threshold 충족을 분리한다. Decision label을 success/no-effect로 이분화하지 않고 실제 수치와 기준을 동시에 보존한다. | `scripts/07_ksdd2_g5_bounded_learner/protocol.md`<br>`scripts/07_ksdd2_g5_bounded_learner/config.yaml`<br>`docs/results/ksdd2_g5_single_unet_20260727/g5_authorization_decision.json`<br>`docs/results/ksdd2_g5_single_unet_20260727/paired_seed_metrics.csv`<br>`docs/results/ksdd2_g5_single_unet_20260727/posthoc/audit_summary.json` | Dice delta `+0.037565`, CI `[0.019581,0.057029]`; LOO minimum `+0.031234`; positive seeds `0.8`; pixel AP `+0.049824`; FPR `0.160129`; FPR delta `+0.010573`; 사전 기준 `+0.05`; 완료 `60/60` runs | 6장 6.3 “통계적 양의 효과와 사전 최소효과 기준의 구분” | `docs/notion_professor_update_20260803/assets/figure08_ksdd2_positive_below_threshold.png`<br>`docs/translation_atlas/translation_atlas_main_figure.png`의 KSDD2 행 | “Dice 차이는 양수였으나 평균 +0.05 기준은 충족하지 못했다”; “통계적 양의 방향과 사전 기준 충족은 다른 판정이다” | “translation effect가 없었다”; “PatchCore 우월성이 입증됐다”; “실용적 개선이 입증됐다”; “+0.05가 산업적으로 최적이다”; “official test가 개선됐다” | `+0.05`는 결과 전에 고정됐지만 외부 stakeholder나 비용으로 calibration된 threshold가 아니다. 한 acquisition pool과 한 learner에 한정된다. |
| EP06 | GC10 K40/140 branch의 frozen coverage·composition PASS는 downstream detector의 positive learning direction을 보장하지 않았다. | **GC10 K40/140 보조 평가에서 선택 집합의 coverage 기준은 충족했지만, 다운스트림 mAP50-95의 추정 차이는 양의 방향이 아니었고 희소 클래스 macro AP와 recall 기준도 충족하지 못했다.** | C | EP01의 추가 사례로는 유효하지만 EP04와 동일한 GC10 downstream 축에 속한다. mAP CI가 0을 포함하고 noninferiority는 통과했으므로 핵심 주장으로 전면화하면 “성능 저하”로 오독될 위험이 크다. | `runs/dcal_xai/v2_budget_extended/budget_summary.csv`<br>`runs/dcal_xai/v2_backbone_main/v8n_screen_contrasts.csv`<br>`runs/dcal_xai/v2_backbone_main/v8n_screen_decision.json` | coverage delta `+0.015`(`0.955 vs 0.940`); mAP delta `-0.001678`, CI `[-0.017424,0.018332]`; rare AP `-0.018290`; recall `-0.021871`; models `30`; expansion `false` | 부록 A “추가 평가 분기”; 6장 6.2에서 1문장 교차참조만 허용 | `docs/translation_atlas/translation_atlas_main_figure.png`의 GC10 K40 행<br>`docs/translation_atlas/four_fates_taxonomy.csv` | “mAP의 양의 방향은 확인되지 않았다”; “rare AP와 recall 기준을 충족하지 못했다”; “EP01의 보조 사례다” | “K40이 detector 성능을 유의하게 악화시켰다”; “coverage selection은 무효다”; “DINO clustering은 learning과 음의 상관이다”; “false advance가 입증됐다” | mAP CI가 0을 포함하고 mAP noninferiority는 통과했다. EP04와 사례·논리 기능이 중복된다. |

## 채택 순서와 본문 사용 규칙

1. 5장에서는 EP02와 EP03을 사용해 선택 단계 안에서도 발견량·집합 구성·재현성이 서로 대체되지 않음을 제시한다.
2. 6장에서는 EP04와 EP05를 사용해 다운스트림 평균 성능·희소 클래스 지표·사전 최소효과가 서로 다른 판정을 만들었음을 제시한다.
3. 7장에서는 EP01을 앞선 네 핵심 사례의 제한적 합성으로만 제시한다. EP01을 독립적인 보편 명제로 먼저 제시하지 않는다.
4. EP06은 부록 표에 남기고, 본문에서는 EP01을 뒷받침하는 추가 평가 분기가 있다는 한 문장만 허용한다.
5. A 등급 문장은 제목·초록·결론에 그대로 사용할 수 있지만 데이터셋과 평가 범위를 삭제해서는 안 된다.

## 최종 본문 핵심 주장 4~5개

1. **선택 단계와 다운스트림 결과가 모두 관찰된 세 평가 분기에서, 선택 단계에서 양의 방향으로 나타난 결과만으로는 사전에 정한 다운스트림 평가 기준의 충족을 확인할 수 없었다.** (`EP01`)
2. **MPDD와 VisA의 선택 단계 평가에서 결함 이미지 발견량의 증가는 선택 집합의 범주 구성 기준 충족과 일치하지 않았다.** (`EP02`)
3. **NEU-DET 평가에서 한 고정 선택 집합의 학습 seed 간 안정성은 선택 과정을 다른 acquisition seed에서 반복했을 때의 학습 효과 재현성과 동일하게 해석할 수 없었다.** (`EP03`)
4. **GC10 q20 평가에서 전체 mAP50-95는 증가했으나 희소 클래스 macro AP는 감소하여, 전체 성능과 희소 클래스 성능 지표가 상반된 방향을 보였다.** (`EP04`)
5. **KSDD2의 제한적 다운스트림 평가에서 Dice 차이는 양의 방향을 보였고 신뢰구간 하한도 0을 초과했지만, 결과 확인 전에 고정한 평균 Dice 차이 `+0.05` 기준은 충족하지 못했다.** (`EP05`)

## 삭제 또는 부록 전환 주장

- **EP06 — 부록 전환(C):** GC10 K40/140 사례는 EP01의 추가 근거로 보존하되, mAP CI가 0을 포함하고 EP04와 논리 기능이 겹치므로 본문 핵심 주장에서는 제외한다.
- **삭제 권고(D):** 없음. 다만 EP01–EP06의 기존 금지 표현은 모두 삭제 대상으로 유지한다.
- **본문에서 자동 제외할 해석:** selector 우월성, 검증된 framework, Gate 예측 정확도, 실제 비용 절감, independent production pool 일반화, 인과 mechanism 확정.

## 지도교수 면담에서 먼저 말해야 할 주의 문장 3개

1. **통합 결론은 다운스트림 결과가 존재하는 세 평가 분기의 기술적 관찰이며, 서로 다른 데이터셋의 효과를 합산한 일반 실패율이 아닙니다.**
2. **MPDD·VisA의 후속 학습 결과는 관찰되지 않았고 NEU의 고정 집합 결과는 사후 기술 분석이므로, 이를 selector 또는 학습의 일반적 실패로 표현하지 않겠습니다.**
3. **KSDD2에서는 Dice 차이가 양수였지만 사전에 정한 `+0.05` 기준을 충족하지 못했으며, 이 기준을 산업적으로 최적화된 threshold나 실제 비용 절감 근거로 해석하지 않겠습니다.**


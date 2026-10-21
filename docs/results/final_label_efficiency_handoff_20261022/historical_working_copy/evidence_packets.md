# Thesis Evidence Packets

## 사용 원칙

이 문서는 논문 본문 문장과 repository evidence 사이의 연결을 고정하기 위한 claim packet이다. 각 packet의 첫 문장은 그대로 본문 후보로 사용할 수 있도록 범위를 제한했다. “직접 근거”는 가능한 한 통합 문서가 아니라 원 run CSV/JSON을 우선하며, 통합 Atlas는 연결 그림과 cross-branch index로만 사용한다.

공통 금지 원칙은 다음과 같다.

- evaluated branch의 결과를 모든 산업 결함 데이터셋·selector에 일반화하지 않는다.
- selection-only STOP의 `learner=NOT_AVAILABLE`을 downstream failure로 바꾸지 않는다.
- fixed-pool perturbation을 independent production pool replication으로 부르지 않는다.
- 서로 다른 task·metric·threshold의 effect size를 합산하지 않는다.
- frozen threshold를 산업적으로 최적화된 minimum으로 부르지 않는다.

## EP01 — Selection-stage gain은 full downstream utility의 충분조건이 아니었다

- **claim_id:** `EP01`
- **주장 문장 1개:** 평가가 완료된 세 translation branch에서 selection-stage의 positive endpoint는 full downstream Gate PASS를 보장하지 않았다.
- **이 주장을 왜 말할 수 있는가:** GC10 q20, GC10 K40/140, KSDD2는 selection-stage endpoint가 candidate 방향으로 양수였고 learner outcome도 존재하는 completed translation branch다. 그러나 GC10 q20은 rare-class safety, GC10 K40/140은 learner direction·rare AP·recall, KSDD2는 사전 최소 Dice 효과에서 각각 full Gate를 통과하지 못했다. 통합 레지스트리의 completed translation full-pass count는 `0/3`이다.
- **직접 근거 파일 목록:**
  - `docs/translation_atlas/translation_branch_inventory.csv`
  - `docs/translation_atlas/translation_atlas_statistics.json`
  - `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_development_confirmation_gate.csv`
  - `runs/dcal_xai/v2_backbone_main/v8n_screen_decision.json`
  - `docs/results/ksdd2_g5_single_unet_20260727/g5_authorization_decision.json`
- **사용 가능한 핵심 수치:** completed translation branch `3`; full frozen-Gate PASS `0/3`; GC10 q20 mAP delta `+0.017378`이지만 rare macro AP `-0.019877`; GC10 K40 mAP delta `-0.001678`; KSDD2 Dice delta `+0.037565 < +0.05`.
- **연결할 그림/표:** `docs/translation_atlas/translation_atlas_main_figure.png`; `docs/translation_atlas/four_fates_taxonomy.csv`; 본문 통합표 “dataset × signal × stage translation matrix”.
- **예상 반론:** n=3이고 데이터셋, selector, budget, learner, threshold가 달라 `0/3`은 일반적인 실패율을 뜻하지 않는다.
- **반론에 대한 방어 가능 범위:** 반론이 맞다. 이 claim은 확률 추정이나 meta-analysis가 아니라 **평가가 완료된 세 branch의 관찰 결과**로만 방어한다. Native metric을 유지하고 cross-dataset pooling을 하지 않는다.
- **과장하면 안 되는 표현:** “selection gain은 learning으로 이어지지 않는다”; “모든 selector가 실패한다”; “full translation 성공률은 0%다”; “stage Gate가 좋은 selector를 판별한다”.
- **본문 위치(장/절):** 7장 통합 분석, 7.1 “Completed translation branches의 downstream fate”.

## EP02 — Discovery improvement와 composition safety는 별개였다

- **claim_id:** `EP02`
- **주장 문장 1개:** MPDD와 VisA의 frozen selection-only audit에서 anomaly discovery improvement는 selected-set composition safety를 보장하지 않았다.
- **이 주장을 왜 말할 수 있는가:** MPDD와 VisA 모두 candidate selector가 Random보다 anomaly image를 더 많이 선택했지만, 사전에 정의된 category-coverage endpoint를 통과하지 못했다. Discovery와 composition이 동일한 query set에서 동시에 측정됐으므로 두 endpoint의 충돌을 직접 보일 수 있다.
- **직접 근거 파일 목록:**
  - `runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_metric_summary.csv`
  - `runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_selection_only_gate.csv`
  - `runs/visa_selection_only_audit/visa_random_vs_visual_200seed_20260715/visa_metric_summary.csv`
  - `runs/visa_selection_only_audit/visa_random_vs_visual_200seed_20260715/visa_selection_only_gate.csv`
- **사용 가능한 핵심 수치:** MPDD anomaly images `+6.245`, CI `[+5.885,+6.605]`, product-category coverage `-0.325 < -0.25`; VisA anomaly images `+14.480`, CI `[+14.195,+14.770]`, object-category coverage `-4.110`, coverage check FAIL; 각 audit `200` paired fixed-pool perturbations.
- **연결할 그림/표:** `docs/notion_professor_update_20260803/assets/figure06_discovery_composition_contrast.png`; source 설명은 `docs/notion_professor_update_20260803/figure_manifest.csv`; `docs/translation_atlas/translation_atlas_main_figure.png`의 MPDD·VisA rows.
- **예상 반론:** category coverage가 실제 learner performance에 중요한지 검증하지 않았고, 200 seeds도 독립 산업 pool이 아니다.
- **반론에 대한 방어 가능 범위:** learner utility가 아니라 **composition-safety endpoint의 실패**로 한정한다. 200 seeds는 같은 fixed pool에서 initial set을 달리한 paired perturbation이며 independent production replication이라고 하지 않는다.
- **과장하면 안 되는 표현:** “MPDD와 VisA에서 learning이 실패했다”; “DINO는 category collapse를 일으킨다”; “200개 산업 pool에서 재현됐다”; “composition failure의 원인은 global representation이다”.
- **본문 위치(장/절):** 5장 선택 단계 결과, 5.2 “Discovery–composition conflict”.

## EP03 — Fixed-set stability와 acquisition reproducibility는 구분되어야 했다

- **claim_id:** `EP03`
- **주장 문장 1개:** NEU-DET에서 한 favorable fixed selected set의 training-seed 안정성은 새로운 acquisition seed에서 유사한 learner gain이 재현됨을 의미하지 않았다.
- **이 주장을 왜 말할 수 있는가:** 동일 Visual20 set을 5개 training seed로 재학습한 diagnostic은 모두 Random보다 높은 mAP를 보였지만, selector를 10개 acquisition seed에서 다시 실행한 confirmation의 평균 gain은 frozen minimum에 못 미쳤고 CI가 0을 포함했으며 exact sign-flip test도 실패했다.
- **직접 근거 파일 목록:**
  - `runs/seed45_fixed_set_stability/seed45_fixed_set_stability_main/paired_differences_vs_random.csv`
  - `runs/seed45_fixed_set_stability/seed45_fixed_set_stability_main/screen_gate.csv`
  - `runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main/confirmatory_metric_summary.csv`
  - `runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main/confirmatory_gate.csv`
- **사용 가능한 핵심 수치:** fixed set mAP50-95 delta `+0.016236`, training-seed wins `5/5`; new acquisitions mAP delta `+0.007019`, descriptive CI `[-0.005211,+0.019678]`, wins `7/10`, exact sign-flip `p=0.322266`; frozen minimum `+0.010` FAIL.
- **연결할 그림/표:** `docs/notion_professor_update_20260803/assets/figure05_neu_two_reproducibilities.png`; source 설명은 `docs/notion_professor_update_20260803/figure_manifest.csv`; 본문 표 “training-seed stability vs acquisition-seed reproducibility”.
- **예상 반론:** fixed-set 결과는 favorable acquisition realization을 결과 관찰 후 선택한 post-hoc diagnostic이라 공정한 confirmatory 비교가 아니다.
- **반론에 대한 방어 가능 범위:** 그 점을 숨기지 않고 fixed-set 결과를 `POSTHOC_DESCRIPTIVE`로 표시한다. Claim은 selector 우위가 아니라 **재현성 대상이 다르다**는 methodological warning으로만 유지한다.
- **과장하면 안 되는 표현:** “NEU에서 visual diversity가 재현에 실패했다”; “training seed 안정성은 무의미하다”; “10 acquisition seeds가 독립 production generalization을 검증했다”; “고정 집합 결과가 confirmatory PASS였다”.
- **본문 위치(장/절):** 5장 선택 단계 결과, 5.3 “Training-set stability와 acquisition reproducibility의 구분”.

## EP04 — Overall performance와 rare-class safety는 충돌할 수 있었다

- **claim_id:** `EP04`
- **주장 문장 1개:** GC10 q20 branch에서는 overall detector performance의 개선과 rare-class safety의 악화가 동시에 관찰됐다.
- **이 주장을 왜 말할 수 있는가:** 동일한 5 acquisition seeds와 각 3 training seeds에서 DINO-selected set과 Random을 비교했을 때 overall mAP50-95와 recall delta는 양수였지만 rare-class macro AP delta는 음수였고 frozen rare-safety check가 실패했다.
- **직접 근거 파일 목록:**
  - `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_acquisition_seed_differences.csv`
  - `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_acquisition_seed_summary.csv`
  - `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_class_group_macro_summary.csv`
  - `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_development_confirmation_gate.csv`
- **사용 가능한 핵심 수치:** mAP50-95 delta `+0.017378`, acquisition-seed CI `[+0.000580,+0.036332]`, wins `4/5`; recall delta `+0.035381`; frequent macro AP `+0.033345`; rare macro AP `-0.019877`; rare macro AP Gate FAIL.
- **연결할 그림/표:** `docs/notion_professor_update_20260803/assets/figure04_gc10_endpoint_conflict.png`; source 설명은 `docs/notion_professor_update_20260803/figure_manifest.csv`; `docs/translation_atlas/translation_atlas_main_figure.png`의 GC10 q20 row.
- **예상 반론:** rare macro AP는 낮은 데이터 regime에서 분산이 크고 별도 CI가 없으며, 한 GC10 설정의 결과만으로 일반 안전성 문제를 말할 수 없다.
- **반론에 대한 방어 가능 범위:** rare AP의 불확실성을 인정하고 frozen point-estimate safety endpoint의 conflict로만 기술한다. “전체 개선이 모든 class 개선을 뜻하지 않았다”는 branch-specific 사실은 유지 가능하다.
- **과장하면 안 되는 표현:** “overall mAP는 rare class를 항상 희생한다”; “DINO가 rare class를 악화시켰다”; “rare-class harm가 통계적으로 확정됐다”; “산업 안전 위험이 입증됐다”.
- **본문 위치(장/절):** 6장 다운스트림 학습 효용, 6.1 “Overall utility와 rare-class safety conflict”.

## EP05 — Positive learner effect도 사전 최소효과에는 미달할 수 있었다

- **claim_id:** `EP05`
- **주장 문장 1개:** KSDD2의 bounded learner screen에서는 통계적으로 양의 Dice effect가 관찰됐지만 사전에 고정한 mean Dice minimum `+0.05`는 확립되지 않았다.
- **이 주장을 왜 말할 수 있는가:** G5 decision file은 60/60 required runs의 완료, 양의 mean delta와 CI lower bound, 0.8 positive acquisition-seed fraction, positive leave-one-seed-out minimum, pixel AP와 FPR safety PASS를 기록한다. 동시에 유일한 primary failure는 mean delta `≥ +0.05` check다.
- **직접 근거 파일 목록:**
  - `scripts/07_ksdd2_g5_bounded_learner/protocol.md`
  - `scripts/07_ksdd2_g5_bounded_learner/config.yaml`
  - `docs/results/ksdd2_g5_single_unet_20260727/g5_authorization_decision.json`
  - `docs/results/ksdd2_g5_single_unet_20260727/paired_seed_metrics.csv`
  - `docs/results/ksdd2_g5_single_unet_20260727/posthoc/audit_summary.json`
- **사용 가능한 핵심 수치:** macro Dice delta `+0.037565`, CI `[+0.019581,+0.057029]`, leave-one-seed-out minimum `+0.031234`, positive seed fraction `0.8`, pixel AP delta `+0.049824`, PatchCore normal FPR `0.160129`, FPR delta `+0.010573`, frozen minimum `+0.05`, completed runs `60/60`.
- **연결할 그림/표:** `docs/notion_professor_update_20260803/assets/figure08_ksdd2_positive_below_threshold.png`; source 설명은 `docs/notion_professor_update_20260803/figure_manifest.csv`; `docs/translation_atlas/translation_atlas_main_figure.png`의 KSDD2 row.
- **예상 반론:** CI upper bound가 `+0.05`를 넘고 effect가 양수이므로 사실상 성공으로 봐야 하거나, 반대로 decision label이 STOP이므로 효과가 없다고 봐야 한다.
- **반론에 대한 방어 가능 범위:** 두 이분법 모두 거부한다. 추정된 effect는 양수지만 primary decision rule은 **mean estimate** `≥ +0.05`였고 충족하지 못했다. Threshold가 최적이라는 주장 없이 statistical positivity와 frozen practical sufficiency를 분리한다.
- **과장하면 안 되는 표현:** “translation effect가 없었다”; “PatchCore가 Random보다 우월함이 입증됐다”; “실용적 개선이 입증됐다”; “+0.05는 산업적으로 최적인 threshold다”; “공식 test에서 개선됐다”.
- **본문 위치(장/절):** 6장 다운스트림 학습 효용, 6.3 “Statistical positivity와 practical minimum의 분리”.

## EP06 — Selection coverage PASS와 downstream learning direction은 별개였다

- **claim_id:** `EP06`
- **주장 문장 1개:** GC10 K40/140 branch의 frozen coverage·composition PASS는 downstream detector의 positive learning direction을 보장하지 않았다.
- **이 주장을 왜 말할 수 있는가:** K40 selected set은 holdout에서 all-class/rare/min-two 및 composition safety Gate를 통과했지만, 동일한 bounded YOLOv8n screen에서 mAP50-95 point estimate, rare macro AP, recall이 모두 Random 대비 음수였고 expansion이 승인되지 않았다.
- **직접 근거 파일 목록:**
  - `runs/dcal_xai/v2_budget_extended/budget_summary.csv`
  - `runs/dcal_xai/v2_backbone_main/v8n_screen_contrasts.csv`
  - `runs/dcal_xai/v2_backbone_main/v8n_screen_decision.json`
- **사용 가능한 핵심 수치:** all-class coverage rate delta `+0.015`(`0.955 vs 0.940`); mAP50-95 delta `-0.001678`, CI `[-0.017424,+0.018332]`; rare macro AP `-0.018290`; recall `-0.021871`; 30 models; `authorizes_v8s_expansion=false`.
- **연결할 그림/표:** `docs/translation_atlas/translation_atlas_main_figure.png`의 GC10 K40 row; `docs/translation_atlas/four_fates_taxonomy.csv`의 `B_LEARNING_DIRECTION_FAILURE`; 본문 대비표 “coverage endpoint vs learner endpoint”.
- **예상 반론:** mAP CI가 0을 포함하고 noninferiority check는 통과했으므로 명확한 성능 저하라고 단정할 수 없다.
- **반론에 대한 방어 가능 범위:** 맞다. “통계적으로 유의한 harm”가 아니라 **positive learning direction이 확인되지 않았고 rare AP·recall safety Gate가 실패했다**고 쓴다. 핵심은 selection PASS가 learner improvement를 보장하지 않았다는 endpoint contrast다.
- **과장하면 안 되는 표현:** “K40이 detector 성능을 유의하게 악화시켰다”; “coverage selection은 무효다”; “DINO clustering은 learning과 음의 상관이다”; “holdout PASS가 false advance임이 입증됐다”.
- **본문 위치(장/절):** 6장 다운스트림 학습 효용, 6.2 “Coverage PASS와 learner direction failure”.

## Packet 간 역할 구분

| Packet | 분석 수준 | 본문 기능 | 증거 강도 |
|---|---|---|---|
| EP01 | cross-branch synthesis | 핵심 결론의 범위를 고정 | 중간 — n=3, heterogeneous |
| EP02 | selection-only, two datasets | discovery와 composition 분리 | 강함 — learner claim은 NA |
| EP03 | one-dataset reproducibility contrast | 재현성 단위 구분 | 중간 — fixed-set arm은 post-hoc diagnostic |
| EP04 | completed downstream branch | overall–rare safety conflict | 강함 — rare AP uncertainty caveat 필요 |
| EP05 | prospective bounded learner branch | statistical–practical distinction | 강함 — threshold calibration caveat 필요 |
| EP06 | completed downstream branch | coverage–learner direction contrast | 강함 — harm significance는 주장 불가 |

논문 초록과 결론에서는 EP01을 단독 근거로 쓰지 말고 EP02–EP06의 구체적 사례를 먼저 제시한 뒤 제한적 합성으로 연결한다.


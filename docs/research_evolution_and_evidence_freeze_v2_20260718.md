# Research evolution and Evidence Freeze v2

Date: 2026-07-18  
Final synthesis decision: **A. THESIS_REFRAME_STRONGLY_SUPPORTED**  
This rating applies only to the evidence-based thesis reframe. It does not revive the rejected selector-superiority hypothesis.

## 1. 최초 연구 질문

초기 연구는 생성형 VLM의 반복 설명이 흔들릴수록 모델 지식이 부족하다는 직관에서 출발했다. 초기 발표의 H1은 explanation consistency와 실제 분류/위치 오류의 상관, H2는 동일 label budget에서 Random·Entropy보다 빠른 mAP 향상이었다. 전체 연구계획서는 이를 (i) linguistic epistemic uncertainty의 형식화, (ii) HDAL-LB 대비 +2~3% mAP, (iii) 언어 정렬 기반 domain adaptation +3~5% mAP, (iv) 0.2 s/image 로컬 배포, (v) 50~80% label-cost 절감과 현장 신뢰 향상으로 확장했다.

당시 문헌적 출발점은 반복 생성의 semantic disagreement, AL uncertainty/diversity, VLM prompting, visual grounding이었다. 그러나 문헌지도는 동시에 `consistency != truth`, selector(VLM)-learner(YOLO) cross-model risk, seed/retraining 교락, strong Random 가능성을 명시했다. 이후 실험은 바로 이 경고를 단계별로 검증한 과정이 되었다.

## 2. 최초 가설의 단계적 검증

| 단계 | 질문과 protocol | 핵심 결과 | frozen gate | 결정과 다음 단계 |
|---|---|---|---|---|
| VLM consistency | 95-image legacy consistency-groundedness audit | Spearman -0.181199; severe-failure AUC 0.373433 | 0/6 | **FAIL**. H1을 기각하고 grounding을 직접 감사했다. |
| Groundedness | GC10 structured prompt pilot20 | schema 0.01; bbox coverage 0.30; median IoU 0 | FAIL | Prompt 재작성만으로 부족하여 oracle crop diagnostic으로 이동했다. |
| Oracle crop | positive oracle crops 20 | presence 0; evidence 0; median IoU 0 | FAIL | Full-image 난이도 설명을 기각했다. |
| Cross-model paired probe | 3 Qwen-family models x 40 paired views | 0/3 pass; best balanced accuracy 0.70 with IoU 0 | FAIL | VLM acquisition-score branch를 종료했다. |
| DINO diversity | NEU/GC10/MPDD/VisA frozen global representation | fixed-pool discovery gain은 존재했으나 category/source/local-defect 문제가 반복 | dataset-specific | 범용 utility 주장 대신 discovery와 composition을 분리했다. |
| Random baseline audit | NEU V10c24 selected-set properties | class coverage, instance diversity, bbox richness가 이미 강함 | audit | Random을 weak baseline으로 취급하지 않도록 고정했다. |
| Acquisition confirmation | seed45 fixed set 후 10 new acquisition seeds | fixed set +0.016236; new sets +0.007019, CI crosses 0, p=0.322266 | FAIL | training-seed stability != acquisition generalization을 확인했다. |
| Coverage-utility translation | GC10 q20 and K40/140 detector screens | q20 mAP +0.017378 but rare -0.019877; K40 mAP -0.001678, rare -0.018290, recall -0.021871 | FAIL | coverage와 learner utility를 별도 gate로 고정했다. |
| Detector uncertainty validity | 15 existing checkpoints, sealed predictions, 232 development images | total-error enrichment 1.115901 <1.50; AUROC 0.766432 | FAIL | AL signal은 종료; FN endpoints는 exploratory only로 남겼다. |
| Cross-dataset discovery/safety | 200 paired seeds per dataset | GC10 +2.720; MPDD +6.245; VisA +14.480, but dataset-specific safety losses | mixed | same-pool discovery effect만 유지했다. |
| Stratified identifiability | reconstructed candidate pools | mean Jaccard 0.978446/0.962827/0.995386 | prevalence gate 0/3 | sparsity law를 `not identifiable`로 고정했다. |
| Metadata feasibility | acquisition manifests + EXIF + frozen selections | documented independent pool groups 0/3 | NO-GO | production generalization 주장을 차단했다. |

## 3. 명확히 기각된 중심 주장

- 설명 consistency는 epistemic uncertainty의 직접 근사치다.
- VLM consistency selector는 Random보다 안정적으로 우수하다.
- Global DINO coverage는 detector learning utility를 보장한다.
- Horizontal-flip disagreement는 epistemic detector uncertainty다.
- 범용 detector uncertainty는 top-budget total-error utility를 충분히 농축한다.
- 현재 benchmark에서 target sparsity가 selector gain을 조절한다는 법칙을 식별할 수 있다.
- 현재 metadata로 independent-production generalization을 검증할 수 있다.
- HDAL-LB 대비 +2~3% mAP, label cost 50~80% 절감, domain adaptation +3~5%, inspector trust 향상을 달성했다.

## 4. 실제로 확인된 positive evidence와 동반 위험

| Evidence | Positive effect | 동시에 확인된 경계 |
|---|---:|---|
| GC10 q20 discovery | rare images +2.720; CI [2.455,2.980] | q20 detector rare macro AP -0.019877 |
| MPDD q20 discovery | anomaly images +6.245; CI [5.885,6.605] | category -0.325; source-composition contribution 67.3% |
| VisA q20 discovery | anomaly images +14.480; CI [14.195,14.770] | categories -4.110; HHI +0.286025; 200/200 safety loss |
| GC10 first detector translation | overall mAP +0.017378 | rare macro AP -0.019877; frozen gate FAIL |
| Seed45 fixed set | mAP +0.016236; 5/5 training seeds | independent acquisition confirmation FAIL |
| V2.3 ranking | majority-error AUROC 0.766432 | total-error enrichment 1.115901 <1.50 |
| V2.3 secondary FN | FN 1.379693; rare-FN 1.785910 | post-hoc, 18 rare images/22 boxes, no untouched validation |

## 5. 반복 확인된 failure mechanisms

1. **Consistent hallucination / grounding collapse**: 문법적으로 유효한 응답과 시각적으로 grounded한 응답은 달랐다. Oracle crop과 model swap에서도 복구되지 않았다.
2. **Global-representation/local-defect mismatch**: DINO는 제품·배경·세션·texture 차이를 포착했지만 작은 결함 semantics와 detector utility를 일관되게 포착하지 못했다.
3. **Category/source/session concentration**: VisA category collapse와 MPDD source-origin contribution이 discovery gain의 안전성을 제한했다.
4. **Acquisition-set non-generalization**: 한 고정 집합의 training-seed 안정성은 새 선택 집합에서 재현되지 않았다.
5. **Coverage-utility gap**: taxonomy/rare coverage 증가는 rare AP, recall, mAP를 자동 보장하지 않았다.
6. **Overall mAP-rare safety trade-off**: 평균 성능의 양의 차이가 희소 class 안전성과 공존하지 않았다.
7. **Ranking-operational enrichment gap**: AUROC와 correlation이 높아도 작은 review budget의 enrichment는 운영 threshold에 못 미칠 수 있다.
8. **Metadata-independent-pool validity gap**: metadata 존재 자체가 independent production pool을 의미하지 않는다.

## 6. 살아남은 중심 연구 질문

> 산업 결함 데이터에서 Active Learning 후보 신호의 **signal validity, target discovery, composition safety, acquisition-set 재현성, downstream detector utility와 operational enrichment는 어떻게 분리되며**, 고비용 학습과 locked evaluation 전에 이 번역 실패를 어떻게 판정할 수 있는가?

## 7. 현재 중심 기여

**Primary contribution**  
산업 결함 AL 후보 신호를 `Signal -> Discovery -> Composition Safety -> Acquisition Reproducibility -> Learning Utility -> Operational Validity`로 분해하고, 다음 단계의 고비용 실행을 frozen gate로만 허용하는 **validity-gated empirical evaluation framework**를 제시·적용했다.

**Secondary contributions**

1. 세 fixed benchmark에서 same-pool discovery gain과 category/source safety가 분리됨을 paired records로 정량화했다.
2. GC10의 두 detector translation protocol과 NEU acquisition confirmation을 통해 selection coverage/training stability가 learner utility/acquisition generalization과 다름을 보였다.
3. VLM grounding, flip disagreement, detector uncertainty, metadata를 각각 신호·운영·식별 가능성 단계에서 감사하여 구체적 stopping condition을 제시했다.

## 8. 연구 범위와 한계

- GC10/MPDD/VisA 결과는 fixed benchmark의 same-pool paired effects다.
- Bootstrap CI는 acquisition-seed perturbation variation이며 production-population CI가 아니다.
- 독립 production lot/time/source pool이 없다.
- Locked final test는 한 번도 사용하지 않았다. 따라서 final-test 성능을 보고하지 않는다.
- FN/rare-FN은 exploratory이고 untouched validation이 없어 실행을 중단했다.
- GC10은 multi-class object detection, MPDD/VisA는 anomaly localization/discovery이므로 raw yield나 downstream metric을 합산·평균하지 않는다.
- 동일 workflow를 적용했지만 branch별 gate와 target 정의는 task-specific이다.

## 9. Evidence Freeze v2

### 유지 주장

- Fixed benchmark same-pool에서 target discovery gain은 존재한다.
- Discovery와 composition safety는 분리될 수 있다.
- GC10에서 selection coverage는 rare AP/recall/mAP로 자동 번역되지 않았다.
- Training-seed stability는 acquisition-set generalization이 아니다.
- Validity gate는 최소 45개의 명시된 detector run과 final-test 소비를 방지했다.

### 폐기 주장

- 범용 selector superiority, VLM epistemic proof, sparsity law, production generalization, 성능 향상 보장, 비용·신뢰·도메인 적응 개선.

### Exploratory only

- FN enrichment 1.379693, rare-FN 1.785910, confidence rare-FN 2.683022.

### Protected status

- Final test used: **False**.
- New training/inference/embedding/selector/FN screen executed in this synthesis: **False**.

### Stopped branches

- VLM consistency/grounding, small-model search, V10c/V10d scaling, D2R representation repair, flip disagreement, K40 detector expansion, V2.3 general-purpose AL, FN implementation.

### Reusable assets

- Paired selection records, manifests, DINO caches, detector checkpoints, sealed predictions, gate code, protocol documents, metadata provenance, unit tests and evidence ledger.

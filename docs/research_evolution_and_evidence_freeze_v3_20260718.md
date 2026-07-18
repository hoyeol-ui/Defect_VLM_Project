# Research evolution and Evidence Freeze v3

Date: 2026-07-18  
Final synthesis decision: **A. THESIS_CORE_STRENGTHENED_BY_PROSPECTIVE_STOP**  
This rating reflects added prospective authorization evidence. It does not revive the rejected selector-superiority hypothesis or establish predictive screening accuracy.

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

## 9. Prospective external authorization case: DeepPCB

DeepPCB는 defect-free template과 tested image가 정렬된 paired-reference 조건이므로, global DINO/VLM보다 local residual이 결함 annotation triage에 가까울 수 있다는 새로운 외부 조건이었다. Score, threshold 32, component minimum area 9, query fraction 20%, eligible six groups와 reserved group92000을 실행 전에 고정하고 selection CSV를 SHA-256으로 봉인한 뒤 GT를 결합했다(E035).

G1에서는 score와 bbox instance count의 mean group Spearman이 0.412174였고 6/6 group에서 양의 방향이었다(E036). 그러나 primary G2 total bbox enrichment는 1.107693으로 frozen 1.20에 못 미쳐 `FAIL_STOP`이었다(E037). Small bbox enrichment 1.307591은 secondary exploratory signal로만 분리했다(E038).

Read-only mechanism audit에서는 small-share uplift 1.1828, 95% CI [1.0814,1.3711], size-selectivity >1 in 5/6, spatial ratio >1 in 6/6이 관측됐다. 그러나 positive small-box excess의 45.79%가 한 group에서 발생하여 frozen <40% rule을 실패했고 Phase A는 `A2_MECHANISM_AMBIGUOUS`였다(E039-E042). Size-bin audit에서 <=256 px² box는 없었고 257-576 px²는 selected 0이었으며, 주요 농축 구간은 577-1024 px²였다. 따라서 tiny defect 전반의 효과로 표현하지 않는다.

Reserved development group92000의 M1 AbsDiff, M2 SSIM, M3 fused residual 비교에서도 candidate가 없었다. M2는 FP/image 0.973에서 small recall 0.179, brightness retention 0.20이었고 M1/M3는 <=1 FP/image에서 nonzero operating point를 유지하지 못했다(E043-E044). External confirmation과 detector screen은 미승인됐고 detector training/inference와 official/final test actual use는 모두 0이었다(E045-E046).

이 사례는 method success가 아니라 **research stopping success**다. Retrospective core가 실제 외부 branch에서 어떻게 적용되는지 보여주되, 한 STOP 사례로 predictive accuracy를 계산할 수 없다는 경계는 유지한다.

## 10. Evidence Freeze v3

### 최종 판정

**A. THESIS_CORE_STRENGTHENED_BY_PROSPECTIVE_STOP**

강화된 것은 selector 성능이 아니라 다음 세 가지다.

1. 외부 데이터에서 protocol과 selection을 결과 전에 고정한 prospective process evidence.
2. Primary FAIL 뒤 positive secondary signal을 parent success로 승격하지 않은 claim-boundary enforcement.
3. Mechanism ambiguity와 development NO_CANDIDATE 뒤 detector/official-test 소비를 중단한 authorization conformance.

### 유지 주장

- Fixed benchmark discovery, composition safety, acquisition reproducibility와 learner utility는 분리된다.
- Selection PASS는 bounded measurement authorization이지 success prediction이 아니다.
- DeepPCB primary FAIL은 detector screen을 중단했고 official/final test actual use는 0이었다.
- 명시적으로 문서화된 기존 minimum 45 planned detector runs stopped 수치는 그대로 유지하며 DeepPCB에 임의의 prevented-run 수를 더하지 않는다.

### 폐기 또는 종료 주장

- DeepPCB total bbox-richness utility, general small-defect triage, independent residual candidate와 detector improvement.
- Universal selector superiority, predictive screening accuracy, independent production generalization과 annotation-cost reduction.

### Protected status

- DeepPCB exact branch closed: **True**.
- Score/threshold rescue authorized: **False**.
- External confirmation authorized: **False**.
- Detector screen authorized: **False**.
- New training/inference in Evidence Freeze v3: **False**.
- Official/final test actual use: **0/0**.

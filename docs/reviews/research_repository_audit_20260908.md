# 검토자 의견서와 repository 결과의 교차 감사

감사 기준일: 2026-09-08 (Asia/Seoul)  
감사 범위: 검토자 검토 의견서, 추가 피드백, 현재 `al-selector` release와 Entropy a00 공식 tranche의 보존된 산출물  
실행 범위: 새 학습·새 selector·protocol 변경·잠긴 test 접근·논문/PPT 수정 0회

첨부된 학위논문 검토 의견서는 연구 방향과 보강 우선순위를 제시한 검토 자료다. 그것을 새 실행 authorization이나 protocol 변경 명령으로 해석하지 않았다. 수치의 판정에는 저장소의 config, source, sealed result table과 audit record를 우선 사용했다.

## 1. Executive conclusion

검토자가 제안한 최종 논리는 제한된 라벨 예산의 산업 결함 object detection에서 Frozen DINOv2 표현 기반 farthest-first 선택의 평균 학습 효용을 검증하되, 전체 mAP가 가릴 수 있는 취약 클래스 악화를 함께 드러내는 것이다. 이 방향은 현재 repository의 역사적 440-cell 결과와 맞는다. Frozen DINO는 GC10-DET와 NEU-DET에서 primary budget-curve AULC가 Random보다 각각 +0.581 pp와 +0.578 pp였지만, NEU의 worst-class AULC는 -1.488 pp, R5는 -3.118 pp였다. 검토자가 인용한 NEU R5 클래스별 선택 수와 AP 변화는 최종 보고서의 값과 일치하고, 여섯 개의 반올림된 점만으로 계산한 Spearman 0.942857과 Pearson 0.991712도 각각 0.943과 0.992를 재현한다. 다만 그 상관계수는 현재 최종 보고서에 별도의 sealed artifact로 저장되어 있지 않고, n=6인 사후 요약의 기술통계다. 기존 class-wise kNN/distance 분석은 이미 있으므로 DINO 표현에 라벨 관련 국소 구조가 있다는 근거는 충분하지만, class compactness나 farthest-first under-selection의 원인을 직접 검증한 분석은 아니다. Entropy는 제목의 중심을 바꾸는 새 연구가 아니라 detector uncertainty 계열의 보조 비교군이며, 현재 공식 결과는 acquisition index 10개 중 a00 하나뿐이므로 최종 비교나 일반화 주장을 할 단계가 아니다. 따라서 논문의 방향은 일관되지만, 상관·kNN·Entropy를 논문에 넣을 때 사후 분석, chance 기준, worst-class endpoint, 1/10 범위를 명시해야 한다.

## 2. research feedback vs repository truth

| 항목 | 현재 상태 | 근거 | 추가 작업 여부 | 논문 위치 |
|---|---|---|---|---|
| 제목 방향 | 대체로 정합. `성능을 높이기 위한`은 연구 목적 표현으로 쓰고 보편적 달성 주장으로 쓰지 않음 | 검토자 의견서, `final_analysis/CONFIRMATORY_FINAL_ANALYSIS_REPORT.md` | 제목 문구 결정만 필요 | 제목·초록·서론 목적 |
| NEU 선택 수 ↔ AP | 값과 방향 일치. 선택 수 CI 5/6, AP CI 4/6이 0을 배제 | `confirmatory_v1_1/final_analysis/CONFIRMATORY_FINAL_ANALYSIS_REPORT.md` | 상관표를 별도 post-hoc 표로 봉인하면 완료 | 3.10.4, 결과/논의 |
| Spearman .943 / Pearson .992 | 반올림된 6개 점으로 정확히 재현. 현재 보고서에는 correlation row/hash가 없음 | 위 보고서의 R5 class rows; 본 감사 산술 재확인 | 상관계수 산출 방식과 p-value를 명시할 문서화 필요 | 3.10.4 |
| “왜 덜 뽑혔나” | 기존 kNN은 있음. 그러나 within-class dispersion/density와 선택 trajectory의 직접 연결은 없음 | `docs/reviews/dinov2_representation_validity_audit_20260907.md`, `classwise_nearest_neighbor_agreement.csv` | 새 학습 불필요. 기존 값으로 가능한 주장만 사용 | DINO 진단/한계 |
| 81.25% / 99.21% | 1-NN 이웃이 query와 annotation class를 하나라도 공유하는 비율. chance-adjusted 값 아님 | `analyze_dinov2_representation_validity.py`, `dataset_metrics.csv` | prevalence 보정 기준을 표에 추가할지 결정 | 3.6.2 또는 사후 진단 |
| GC10 multi-label | label set bit mask의 교집합이 비어 있지 않으면 match. 전체 set equality가 아님 | `analyze_dinov2_representation_validity.py` | 정의를 본문에 한 문장으로 고정 | 방법/사후 분석 |
| 숫자→이미지 순서 | repository 수치와 검토자 요구가 일치 | classwise kNN CSV와 기존 figure package | 새 PPT는 이번 감사 범위 밖 | 발표자료 |
| DINO 설정 | 모델, revision, 전처리, 384-D pooler output, FP32, L2, deterministic, batch 32 모두 일치 | `confirmatory_v1/confirmatory_protocol_v1.yaml`, `confirmatory_v1/src/confirmatory_v1/frozen_dino_ff_v1.py` | 없음 | 3.6.2 |
| 선택 신호의 성격 | detector 예측을 선택 점수에 쓰지 않음. DINO는 feature extractor, farthest-first가 selector | protocol YAML, frozen DINO source | 용어만 일관되게 교정 | 3.6.2 |
| best.pt 의미 | fixed validation에서 선택된 checkpoint. budget/라벨 비율이 아님 | `confirmatory_v1_1/src/run_yolo_phase2b_cell.py`, Entropy training audit | 없음 | 학습 protocol |
| Entropy | 현재 freeze와 a00 공식 tranche는 존재. 20 cells/10 queries complete; 전체는 1/10 trajectory | `entropy_extension_v1/official_release_v1/*`, `reports/entropy_postrun_a00_20260908/POST_RUN_REPORT.md` | 남은 180 cells 후 class-wise validation 분석 | 보조 비교군/논의 |
| Lv et al. 선례 | Sensors 2020 논문은 NEU-DET active learning에서 uncertainty와 class-wise sampling을 다룸. 검토자가 제시한 최저 클래스 수치는 논문 표와 함께 인용할 수 있음 | [Lv et al. (2020)](https://doi.org/10.3390/s20061650) | 원문 표의 dataset subset과 metric 정의를 함께 적기 | 관련 연구/논의 |

## 3. Already completed

- Historical Random/Frozen DINO 440-cell registry, final lock, result tables, and hashes are preserved in the `al-selector` lineage. The audit did not modify them.
- Frozen DINO implementation is sealed: `facebook/dinov2-small`, ViT-S/14, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, shortest-edge 256 bicubic resize, 224 center crop, RGB conversion, ImageNet normalization, `outputs.pooler_output` 384-D, FP32, no autocast/TF32, deterministic inference, row-wise L2 normalization, batch 32.
- The existing representation audit covers 1-NN/5-NN overlap, class-wise preservation, pairwise distances, nominal NEU category diagnostics, cache identity, split boundaries, and label-source integrity. It reports GC10 1-NN annotation overlap 0.8125 and NEU 0.9921.
- Class-wise preservation is already available: GC10 `rolled_pit` 0.2500 and `crease` 0.3243 are low; NEU `crazing` and `rolled-in_scale` are 1.0000, while `scratches` is 0.9905. This is descriptive evidence, not a compactness proof.
- Historical final analysis already contains the NEU R5 selection-image changes and AP changes with intervals. Five of six selection-count intervals exclude zero; four of six AP intervals exclude zero. The distinction matters.
- Entropy B1/B2 is frozen with policy learner A, A/B shared manifests, A-best parent edges, cold-start sealed YOLOv8n, SUM binary entropy of post-NMS `boxes.conf`, no-detection score 0, `conf=0.25`, `iou=0.70`, `max_det=300`, canonical ID tie-break, and locked-test endpoint disabled.
- Entropy official a00 is audited as 20/20 training cells, 10/10 acquisition jobs, 10/10 round gates, historical protected hashes unchanged, policy B edges 0, duplicate/reselection/leakage 0, and integration artifacts not reused. It remains interim descriptive evidence only.

## 4. Still missing

1. The research correlation should be recorded as an explicitly post-hoc, six-class descriptive row with the exact input values, rounding rule, and correlation method. For six observations, the usual asymptotic Spearman p-value should not be treated as decisive; a permutation p-value would be the defensible inferential option if a formal test is later required. This is a reporting issue, not a reason to change the selector.
2. Existing class-wise kNN does not report within-class radius, average kNN distance by class, or the distance from each candidate to the current farthest-first reference set. It cannot by itself establish “dense class → fewer selections.” Use it to state label locality and the observed boundary, not as a causal explanation.
3. The 81.25% and 99.21% numbers need an explicit chance reference if shown as validity evidence. Uniform `1/10` and `1/6` are only toy baselines. A fair baseline preserves the observed image-level label-set prevalence (and multi-label co-occurrence), for example by permutation or an empirical independent-neighbor calculation. The repository currently labels these values as raw descriptive, chance-unadjusted.
4. Entropy a00 stores aggregate development mAP50-95 per checkpoint, not class-wise AP or worst-class AP. The official protocol is fixed, but the requested class-wise/worst-class Entropy comparison is not complete. It should be derived from retained validation checkpoints only after the official tranche is complete; it does not require changing training.
5. The official Entropy namespace has 180 training cells and 90 acquisition jobs outstanding. The 2-cell integration namespace is correctly excluded from official results.
6. A 100%-labeled reference is prepared as a separate plan but is not historical evidence yet. It should not be described as an observed upper bound until those reference runs exist.

## 5. DINO industrial-data validity

The repository supports a bounded claim: the frozen representation contains label-related local structure in these acquisition pools. It does not support a claim that DINOv2 is universally suitable for industrial defects or that it preserves every class equally.

The primary 1-NN annotation measure is `any shared annotation class`. For an image with label set `S_i` and its nearest neighbor `j`, it is one when `S_i ∩ S_j` is nonempty. GC10's multi-label images are therefore retained; they are not collapsed to one class. The class-wise table conditions on the query containing class `c` and asks whether the nearest neighbor also contains `c`. That is a different estimand from exact label-set equality.

The existing pairwise distance result points in the same direction: mean cosine distance for annotation-overlap pairs is 0.3943 vs 0.4542 for GC10 and 0.3111 vs 0.5129 for NEU. These are pairwise descriptive summaries, not independent observations for a p-value.

The class-wise result is especially useful for limiting the claim. GC10 has very low nearest-neighbor preservation for `rolled_pit` (0.25) and `crease` (0.3243), while NEU has high preservation for `crazing` and `rolled-in_scale` even though those classes are under-selected in the DINO R5 composition. Thus high kNN agreement is compatible with a compact class, but it is neither a direct density measure nor sufficient to explain farthest-first selection. The current evidence supports “representation geometry is class-dependent,” not “this geometry caused every class-wise AP change.”

For a chance comparison, use an empirical label-preserving permutation over the fixed acquisition pool, keeping each image's complete label set and its class co-occurrence structure. Report the observed 1-NN overlap against the permutation mean and interval. If only a simple reference is shown, label it explicitly as a toy uniform baseline and do not call 81.25% or 99.21% a chance-adjusted effect.

The DINO model family itself is correctly identified. The official DINOv2 implementation documents ViT-S/14 with LVD-142M pretraining and 384-dimensional embeddings; the repository pins a recovered revision and local artifacts. DINOv2's foundation-model paper describes features intended to transfer across image distributions and tasks, but that does not remove the need for this dataset-specific validity audit. See [DINOv2](https://arxiv.org/abs/2304.07193) and the [official model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md).

## 6. Entropy role

Entropy is a method-family control. The line that must remain explicit is: **탐지 모델의 예측을 선택 점수에 사용하지 않음** for Frozen DINO, whereas Entropy uses detector predictions from the frozen protocol checkpoint. DINOv2 is the feature extractor and greedy farthest-first is the selection rule; “DINO가 selector” is shorthand that should be avoided.

Entropy therefore does not require a title centered on uncertainty. The proposed title remains coherent if Entropy is described as a secondary detector-aware baseline. The official Entropy rule is already result-independent and must remain unchanged: binary entropy of post-NMS `boxes.conf`, SUM across boxes, no detection = 0, fixed inference thresholds, deterministic tie-break, fixed query sizes and seeds. The smoke correlation between entropy and detection count is a limitation to disclose, not a tuning signal.

The a00 interim curve is not a final method comparison. It is one of ten acquisition indices and uses development validation only. The retained aggregate values are useful for orchestration and descriptive monitoring, but they do not establish Entropy superiority, inferiority, a confidence interval, or a general budget transition.

The class-wise requirement is substantive. Lv et al. report a NEU-DET case where an uncertainty arm raises overall mAP while the weakest class declines; their paper also separates category-level sampling from the global score. This supports the research's request to inspect worst-class and class-wise AP, but it does not predict the outcome of the current protocol. The PMLR paper *Active Learning on a Budget* likewise motivates budget-dependent behavior: typical samples can be preferable at low budgets while unrepresentative/uncertain samples can become preferable at larger budgets. These are literature motivations, not evidence from the current a00.

## 7. Strongest thesis contribution

1. **A controlled utility result with a clearly bounded estimand.** Across two industrial defect datasets and six fixed label budgets, the study connects frozen representation-based acquisition to downstream YOLOv8n learning with separate acquisition and learner seeds, paired A/B learners, and an nAULC primary endpoint.
2. **A diagnostic finding about average performance and vulnerable classes.** The NEU result shows that an overall positive learning-curve effect can coexist with a reproducible worst-class decline and a class-selective change in the selected pool. That is more informative than a single mean-mAP claim.
3. **A reproducible boundary between selection signal, composition, and detector outcome.** Sealed manifests, validation-only checkpoint selection, historical integrity guards, post-hoc composition analysis, and the distinct Entropy baseline make it possible to state what the evidence does and does not identify. The kNN audit strengthens the representation discussion while also exposing its class-dependent limits.

## 8. Recommended next actions

| Priority | Action | GPU | Difficulty | Existing reuse | Thesis effect |
|---:|---|---|---|---|---|
| 1 | Write the NEU composition/AP table and the six-point correlation as a post-hoc diagnostic, including the selection-CI vs AP-CI distinction | No | Low | Direct reuse of final tables | Closes the “what happened” diagnosis |
| 2 | Present existing class-wise kNN/distance results with the exact multi-label definition and a clearly labeled prevalence-preserving chance reference | No for the writing; any new baseline is CPU analysis | Low–medium | Reuses sealed cache and existing script | Makes DINO validity defensible without overstating causality |
| 3 | Finish the already frozen Entropy official tranche, then extract class-wise AP and dynamic worst-class AP from validation checkpoints | Yes: remaining 180 cells | High | Reuses frozen runner and a00 orchestration | Tests whether the vulnerability pattern is DINO-specific or broader |
| 4 | Run the separate 100%-labeled reference plan if the thesis needs a full-data anchor | Yes: 4 planned runs minimum | Medium | Existing run plan and seeds | Adds an interpretable reference; do not call it an upper bound before completion |
| 5 | Align title, section 3.6.2/3.10.4, figures, and captions around “numbers first, examples second” | No | Low | Reuses existing figures and tables | Prevents the known reviewer objections |

## 9. Risks / claims to avoid

- Do not call `.943/.992` an independently sealed repository result until the six inputs and calculation method are recorded. With six classes, the asymptotic p-value is fragile; the effect should be described first as a strong descriptive monotone association.
- Do not infer a causal density explanation from kNN preservation. Compactness, farthest-first coverage, class prevalence, annotation co-occurrence, and image-level composition are distinct quantities.
- Do not describe 81.25% or 99.21% as chance-adjusted or as a universal DINO validity score. They are raw 1-NN shared-label rates.
- Do not conflate selection-count confidence intervals with AP confidence intervals: five selection classes exclude zero, while four NEU AP intervals exclude zero in the historical report.
- Do not pool Entropy a00 with the historical 10-index Random/DINO estimate, and do not reuse the 2-cell integration results as official results.
- Do not claim Entropy's current interim curve proves a budget-dependent winner, a final ranking, or a worst-class effect.
- Do not claim annotation-time savings, industrial deployment generalization, universal DINO superiority, or an observed 100%-labeled upper bound.
- Do not present representative images before the numerical class-wise table. Image panels should be captioned as post-hoc examples that were not used for selection.
- Do not describe a human-picked sample, DINO itself as the selector, or `best.pt` as a label percentage. The code selected samples, farthest-first implemented the rule, and validation selected the checkpoint.

## 판정

**THESIS DIRECTION COHERENT**

The research's proposed direction matches the strongest existing repository evidence: positive overall learning utility with a measurable vulnerable-class boundary. The direction needs careful post-hoc labeling and completion of the frozen Entropy tranche, but it does not require changing the title's central DINOv2 representation-based selection question.

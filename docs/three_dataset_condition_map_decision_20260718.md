# 냉정한 결론

**B. 일부는 성립하지만 한 가지 추가 분석이 필요하다.**

기존 결과만으로 세 데이터셋을 동일 알고리즘의 성능 검증으로 묶을 수는 없다. 그러나 동일한 20장 review budget에서 VisA anomaly +14.480, MPDD anomaly +6.245, GC10 rare image +2.720이라는 discovery gain과, 동시에 VisA category -4.110, MPDD category -0.325, GC10 downstream rare macro AP -0.019877이라는 안전·번역 실패가 확인된다. 따라서 “효용의 조건부 분리”를 보여주는 다중 case-study는 성립한다. 다만 dataset identity와 pool sparsity/source condition이 완전히 교락되어 있으므로, **기존 200-seed selection records만 이용한 prevalence/source/category-matched stratified reanalysis 한 번**이 필요하다. 새 selector나 학습은 필요하지 않다.

# 확인된 positive evidence

| Dataset / protocol | Random | Strategy | Effect | Evidence | 함께 발생한 위험 |
|---|---:|---:|---:|---|---|
| VisA anomaly query@20 | 2.155 | 16.635 | +14.480; enrichment 7.719x | frozen selection gate의 discovery 항목 통과, 전체 gate FAIL | object category -4.110 |
| MPDD anomaly query@20 | 4.060 | 10.305 | +6.245; enrichment 2.538x | frozen selection gate의 discovery 항목 통과, 전체 gate FAIL | product category -0.325; official-test origin +6.810/20 |
| GC10 rare-image query@20 | 2.170 | 4.890 | +2.720; enrichment 2.253x | confirmatory selection-only PASS | detector rare macro AP -0.019877 |
| GC10 K40 holdout coverage@140 | 0.940 | 0.955 | all-class rate +0.015 | independent holdout selection PASS | downstream mAP -0.001678; recall -0.021871 |
| NEU seed45 fixed set | 0.187550 | 0.203786 | mAP +0.016236; 5/5 | diagnostic fixed-set PASS | independent acquisition confirmation FAIL |

Positive는 모두 endpoint와 gate 범위 안에서만 해석했다. VisA/MPDD는 discovery effect가 크지만 전체 safety gate는 FAIL이므로 `confirmatory_pass`로 승격하지 않았다.

# 확인된 negative evidence

- NEU 독립 acquisition 10 seeds: mAP50-95 +0.007019, descriptive CI [-0.005211, 0.019678], p=0.322266; frozen gate FAIL.
- NEU V10c24 budget120: mAP50-95 -0.004863; scale gate FAIL.
- GC10 최초 DINO detector translation: mAP +0.017378이지만 rare macro AP -0.019877; 전체 gate FAIL.
- GC10 K40/140: selection holdout는 PASS였지만 YOLOv8n mAP -0.001678, rare AP -0.018290, recall -0.021871; downstream FAIL.
- VLM oracle crop: parse/informative 1.0에도 presence/evidence/median IoU가 모두 0; paired 3-model comparison 0/3 PASS.
- DCAL-XAI flip disagreement: query instances -2.2, 94/100 선택이 one-view detection에 지배되어 geometry confound로 종료.
- MPDD effect에는 official train/test-origin composition이 강하게 개입하고, VisA는 anomaly-rich category concentration을 동반한다.

# 3-dataset condition map

| Hypothesis | Supporting dataset | Contradicting dataset | Evidence strength | Remaining uncertainty |
|---|---|---|---|---|
| 1. Sparse-target pool에서 discovery gain이 커진다 | VisA +14.480; MPDD +6.245 | GC10 all-defect rare gain +2.720로 더 작음; NEU balanced pool에서 selector 우위 불안정 | medium | dataset와 prevalence가 교락; within-dataset prevalence strata 필요 |
| 2. Discovery gain은 category/source safety를 자동 보장하지 않는다 | VisA category -4.110; MPDD category -0.325/source-origin +6.810 | GC10 최초 selection은 combined class +0.825 | high | coverage 정의가 dataset별로 다름; 평균 금지 |
| 3. Representation coverage는 learning utility를 자동 보장하지 않는다 | GC10 K40 holdout PASS → mAP -0.001678, rare -0.018290 | GC10 q20 detector overall mAP +0.017378 | high for GC10 | learner/data budget 한 조건; 인과 메커니즘은 미확정 |
| 4. Learner alignment가 downstream translation을 결정한다 | GC10 q20 rare translation 실패, D2R class8 recovery 실패 | 직접 learner 교체의 positive evidence 없음 | low-medium | 현재 결과는 alignment 설명과 양립하지만 직접 조작 실험은 없음 |
| 5. Random 강도는 pool balance와 target prevalence에 의존한다 | NEU balanced large-pool Random 강세; VisA sparse target에서 큰 discovery gap | MPDD source confound와 GC10 rare definition이 단순 prevalence 해석을 방해 | medium | stratified matched analysis 필요 |

# Validity-gated workflow의 실제 가치

- 명시적으로 종료된 branch: **11개** (`CLOSE` 또는 `CLOSE_AS_AL`; diagnostic/translation-required 행 제외).
- 정확히 문서화된 방지 detector model runs: **최소 45개** = D2R 15 + K40 후속 YOLOv8s 30. 정의되지 않은 계획은 수치에 넣지 않았다.
- 보호된 locked final-test 평가: **모든 branch에서 소비 0회**.
- 구분된 failure mechanism: **6개** — VLM compliance/presence collapse, source/category concentration, global-representation/local-defect mismatch, acquisition-set non-generalization, flip geometry confound, coverage–learner utility gap.
- 재사용 자산: **15개 hash-verified Random140 checkpoints**, sealed V2.3 prediction/inference manifest 1세트, GC10/MPDD/VisA protocol·selection manifest 3계열, 각 branch gate/audit 코드와 CSV.
- GPU-hours는 원본 runtime artifact로 확인되지 않아 추정하지 않았다.

# 논문화 가능성 판정

| 수준 | 판정 | 이유 |
|---|---|---|
| 석사학위 연구로 방어 가능성 | **high** | 실패를 PASS로 바꾸지 않고, 3개 task 조건의 discovery/coverage boundary와 GC10 downstream translation을 traceable gate로 연결함 |
| 국내/국제 학술대회 논문화 가능성 | **medium** | 산업 AL의 negative/conditional evidence는 유효하나, condition effect의 within-dataset matched analysis와 명확한 scope 제한이 필요함 |
| 방법론 중심 저널 논문화 가능성 | **low** | 새 방법 성능, 외부 downstream replication, 독립 operational validation이 없고 dataset-condition 교락이 큼 |

# 반드시 줄여야 할 주장

- “새로운 AL selector가 Random을 능가한다.”
- “VLM이 detector utility를 예측한다.”
- “DINO가 rare defect selection에 일반적으로 유효하다.”
- “detector uncertainty gate가 성공했다.”
- “세 데이터셋에서 동일 알고리즘이 downstream 성능으로 검증됐다.”
- “pool sparsity가 discovery gain의 원인이다.” — 현재는 dataset과 교락된 연관이다.

# 방어 가능한 중심 주장

> 본 연구는 서로 다른 산업 결함 review pool에서 frozen visual 후보 신호의 target-discovery benefit이 category/source safety와 분리되고, GC10-DET에서는 selection coverage가 acquisition-set generalization·rare-class detector utility로 자동 번역되지 않음을 실증하며, 고비용 학습과 locked final 평가 전에 이 분리를 판정하는 validity-gated evaluation workflow를 제시한다.

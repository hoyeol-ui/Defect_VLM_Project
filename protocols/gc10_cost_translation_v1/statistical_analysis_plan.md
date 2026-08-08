# GC10 Cost-Translation Statistical Analysis Plan

Protocol version: `gc10-cost-translation-v1-candidate`

Status: candidate. Fields marked unresolved must be signed before Phase A.

## 1. Study position

This is a **prospective cost-endpoint validation of a retrospectively identified frozen candidate**. The DINO candidate and earlier learner outcome are known. Human-time, human-label quality, equal-time learner, and rare-safety endpoints can become prospective only if this plan is frozen before those outcomes are collected.

## 2. Fixed study sets

- Primary acquisition seeds: 5–14, selected by chronological deterministic rule
- Strategies: `GTFreeRandom`, `FrozenDINOVisualDiversity`
- Query rank per seed/strategy: 1–20
- Phase A unique union: 233 images
- Annotators: unresolved, minimum candidate 2 and preferred candidate 4
- Stage 0: 24 disjoint images; not an effect-analysis dataset
- Fixed pool: GC10 acquisition split only
- Official/final test: locked and excluded

The sample-ID-only overlap-minimized seed set is appendix sensitivity design only. It must not replace seeds 5–14.

## 3. Estimands and units

### Phase A primary estimand

Let \(M_{s,k,i}=1\) when image \(i\) is present in acquisition seed \(s\), strategy \(k\), query rank 1–20. Let \(t_{i,a}\) be the validated active time for image \(i\) and annotator \(a\).

\[
T_{s,k,a}=\sum_i M_{s,k,i}t_{i,a}
\]

\[
D_s=\frac{1}{A}\sum_a(T_{s,DINO,a}-T_{s,Random,a})
\]

\[
\theta_A=\frac{1}{10}\sum_{s=5}^{14}D_s
\]

The primary estimand is \(\theta_A\): mean DINO-minus-Random human active time required to complete the frozen q20 batch.

### Phase B primary estimand

At one frozen human-time budget \(T\), let \(Y_{s,k,r}(T)\) be development mAP50–95 for acquisition seed \(s\), strategy \(k\), training seed \(r\).

First average the three training seeds within acquisition seed:

\[
\bar{Y}_{s,k}(T)=\frac{1}{3}\sum_rY_{s,k,r}(T)
\]

Then calculate:

\[
\theta_B=\frac{1}{10}\sum_s
[\bar{Y}_{s,DINO}(T)-\bar{Y}_{s,Random}(T)]
\]

The 60 learner runs are not \(n=60\). Acquisition seed is the paired learner decision unit; training seed measures stochastic learner variation.

## 4. Primary overlap-aware uncertainty analysis

### A. Image-and-annotator cluster bootstrap

This is the recommended Phase A primary interval because the same DINO image can appear in many acquisition batches.

For each bootstrap replicate:

1. Draw annotator clusters with replacement from the frozen annotator set.
2. Draw multinomial cluster weights for the 233 unique Phase A images.
3. Apply one shared image weight to every occurrence of that image in every seed/strategy batch. This preserves the cross-batch dependence caused by repeated images.
4. For each q20 seed/strategy batch and resampled annotator, calculate a cluster-weighted batch total and normalize the image weights inside the batch to a total weight of 20:

   \[
   T^*_{s,k,a}=
   20\frac{\sum_i M_{s,k,i}w_i t_{i,a}}
   {\sum_i M_{s,k,i}w_i}
   \]

5. Reject and redraw a replicate only if any batch denominator is zero.
6. Average the resampled annotators inside each seed, calculate the 10 paired seed contrasts, and average them to obtain \(\theta_A^*\).
7. Use the predeclared bootstrap replicate count and percentile interval.

The normalization retains the q20 scale while the shared image weight keeps an image's repeated appearances dependent. The point estimate remains the unweighted frozen-batch estimate.

### What this interval does not mean

- It does not turn ten overlapping fixed-pool seeds into ten independent factory lots.
- With two annotators, it does not estimate a broad population of industrial annotators.
- It does not generalize beyond the GC10 acquisition pool or current annotation tool.

Annotator-specific point estimates and differences must therefore be reported even when the pooled bootstrap interval is shown.

## 5. Secondary sensitivity analysis

### B. Seed-level paired bootstrap

1. Calculate \(D_s\) for seeds 5–14.
2. Resample the ten paired seed contrasts with replacement.
3. Report the percentile CI, median contrast, and positive-seed count.

This matches the existing repository's acquisition-seed analysis style but ignores dependence from repeated images. It is secondary only.

### Appendix overlap-minimized seed design

`overlap_minimized_seed_sensitivity.csv` was constructed using sample IDs only:

- first seed 5;
- next seed minimizes the maximum same-strategy Jaccard against already selected seeds;
- ties use the smaller seed;
- no class, score, learner metric, or annotation outcome is used.

It is a design sensitivity proposal and does not authorize new annotation or replacement of primary seeds.

## 6. Phase A secondary analyses

- per-image active time, overall and by annotator
- active time per submitted box
- create/move/resize/delete/class-change event counts
- box precision and recall
- false-negative and image-level missed-defect rate
- matched-box mean/median IoU
- class accuracy
- duplicate count
- small-defect recall
- rare pooled recall and class 8/9/10 recall
- adjudication correction count and active time
- annotator×strategy interaction, clearly labeled low-powered when annotators are few

Per-image analyses are explanatory. The primary operational estimand remains q20 batch total.

## 7. Phase A authorization

The numerical burden materiality margin, quality non-inferiority margin, idle threshold, log completeness tolerance, and annotator count are unresolved. They must be entered in the signed config after Stage 0 measurement validation and before Phase A arm outcomes are viewed.

Stage 0 may estimate only instrumentation repeatability, log completeness, clock reconciliation, and idle behavior. It may not compare strategy effects or calibrate thresholds to make DINO or Random pass.

Phase B is not authorized when:

- the logger fails;
- Phase A does not meet the pre-frozen materiality rule;
- human-label rare/small safety fails;
- common-support time budget \(T\) cannot be constructed;
- human-participation approval is absent.

## 8. Phase B safety

Required safety family:

- rare macro AP50–95
- rare recall
- class 8 individual guard
- class 9 individual guard
- class 10 individual guard
- overall recall
- frequent-class non-inferiority

The final numerical margins require researcher sign-off. Known earlier GC10 results must not be used to weaken a future rare-safety requirement.

## 9. Missingness and technical failures

- Technical failures are reported by blinded reason.
- No task is removed because its duration or quality is unfavorable.
- Repeated technical attempts remain linked and auditable.
- Missing active time is not imputed from arm mean.
- A seed with an undefined subgroup metric is `NA`; it is not counted as safety PASS.
- Any exclusion rule must be frozen before Phase A.

## 10. Multiplicity and interpretation

There is one primary endpoint in Phase A and one in Phase B. Quality and rare-class measures are safety/diagnostic endpoints and must all be reported without selecting a favorable subset. The study is a bounded empirical audit, not a universal selector ranking.

## 11. Reporting

Minimum outputs:

- seed-level paired batch-time table
- annotator-specific time and quality table
- image-frequency/overlap figure
- primary overlap-aware CI
- seed-bootstrap sensitivity CI
- rare/small quality table
- fixed-time learner table with training-seed averages
- cost ledger separating human, adjudication, selection compute, learner compute, and setup
- explicit PASS/FAIL/STOP result under the signed thresholds

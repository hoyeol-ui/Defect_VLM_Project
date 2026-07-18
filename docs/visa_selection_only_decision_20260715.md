# VisA Selection-Only Decision

작성일: 2026-07-15  
상태: discovery signal confirmed / detector-training gate failed  
Detector training performed: False  
Final test used: False

## 1. Protocol integrity

- Official VisA source: 10,821 images, 1,200 anomaly images
- Valid v2 split: acquisition 8,650 / development 1,085 / final locked 1,086
- Acquisition anomaly prevalence: 960/8,650 = 11.10%
- Exact SHA/path leakage: 0
- DINO input: blind acquisition manifest only
- DINO GT/mask/bbox access: False
- Selection seeds: 200
- Shared initial/query: 20/20

The first pHash-hard-grouping split was invalidated before selection because aligned industrial images formed transitive groups as large as 1,069 samples. v2 uses exact SHA only as a hard identity group and retains pHash as audit-only metadata.

## 2. Primary discovery result

| Metric | Random | Frozen DINO | Difference |
|---|---:|---:|---:|
| Query anomaly images / 20 | 2.155 | 16.635 | +14.480 |
| Query anomaly rate | 10.78% | 83.18% | +72.40 pp |
| Annotations to first anomaly | 8.555 | 1.620 | -6.935 |
| Unique anomaly types | 2.095 | 9.340 | +7.245 |
| Object category coverage | 9.820 | 5.710 | -4.110 |

Anomaly-image yield difference:

- wins/losses/ties: 200/0/0
- bootstrap 95% CI: [+14.195, +14.770]
- Monte-Carlo paired sign-flip p ≈ 0.000005

This is strong evidence that DINO distance from a small random initial set acts as an anomaly/outlier discovery signal in the anomaly-sparse VisA acquisition pool.

## 3. Why the detector-training gate failed

The pre-registered category-coverage condition failed.

- Frozen DINO selected 12.17/20 images from `pipe_fryum` on average.
- `pipe_fryum` occupied 60.85% of the DINO query.
- 10.685/20 selected images were `pipe_fryum` anomalies on average.
- Removing `pipe_fryum` post hoc reduces the anomaly-yield advantage from +14.48 to +3.995.
- Every one of the 200 seeds had lower object-category coverage under DINO than under Random.

Therefore the result supports anomaly discovery but not balanced annotation triage across product lines. The gate must not be relaxed after observing this pattern.

## 4. Decision

- Do not train YOLO on the VisA selections from this run.
- Do not access the VisA final locked split.
- Do not add category balancing and claim it as confirmatory on the same acquisition pool.
- Preserve the current result as a discovery study.

## 5. New independent hypothesis

> A frozen hierarchical selector that first allocates annotation quota across known product/object categories and then applies DINO farthest-first within each category can retain VisA's anomaly-yield advantage without category collapse.

This hypothesis is motivated by the discovery result but was not pre-registered for the current VisA run. It requires an independent dataset or an explicitly new confirmatory benchmark. Reusing the VisA acquisition labels to validate it would be post-hoc tuning.

## 6. Recommended next gate

Use a separate mask-annotated industrial dataset for selection-only confirmation before any detector training.

All conditions must pass:

1. Random anomaly yield remains low.
2. Hierarchical frozen DINO improves anomaly yield by a pre-specified margin.
3. Object/category coverage is non-inferior to Random.
4. The gain survives leave-one-category-out analysis.
5. No final split is evaluated.

Only after an independent selection-only pass should a paired detector experiment be considered.

# MPDD Independent Hierarchical-DINO Selection-Only Protocol

Date: 2026-07-15

## Purpose

This experiment independently tests the hypothesis suggested by the failed VisA
unconstrained-DINO audit: frozen visual distance may help cold-start annotation
triage, but only when known product categories prevent a single visual domain
from consuming the query budget.

This is not a detector benchmark and it does not introduce a new tuned selector.
The hierarchy is the frozen combination already motivated by prior work:
product-category balancing followed by DINOv2 farthest-first selection.

## Safety and data roles

- Detector training before the selection-only gate: **prohibited**.
- Final split use during selection or gate analysis: **prohibited**.
- Anomaly labels, anomaly types, masks, boxes, and official train/test origin are
  hidden from all selectors.
- Product category is allowed as non-GT production metadata.
- Source paths are excluded from the selector manifest because MPDD paths reveal
  labels such as `test/scratches`.
- A private loader map may be used only to turn blind sample IDs into pixels for
  frozen embedding extraction.
- GT and mask-derived quantities are joined only after every selection is fixed.

## Dataset construction

All 1,346 MPDD inspection images are treated as a fresh annotation-triage
population; the 282 ground-truth mask PNGs are annotations, not pool images.
Byte-identical images are hard grouped by SHA-256. pHash is audit-only because
aligned industrial images may be legitimately near-identical.

A deterministic 80/10/10 acquisition/development/final split is made within
`product category x official origin x anomaly type` strata. Official origin is
used only to preserve the source composition and is not exposed to selectors.
The final manifest is immediately locked by SHA-256.

## Frozen selection experiment

- Acquisition seeds: 0--199.
- Shared random initial labeled set: 20 images.
- One cold-start query: 20 images.
- Frozen image representation: `facebook/dinov2-small`, L2-normalized.

Strategies:

1. `GTFreeRandom`: uniform sampling without replacement.
2. `CategoryBalancedRandom`: iteratively choose the currently least represented
   product category in initial-plus-query counts, then choose a random remaining
   image within it.
3. `FrozenCategoryBalancedDINO`: use the identical category allocation rule, then
   choose the candidate farthest in cosine distance from already labeled/selected
   images of that category. If a category has no reference image, use all current
   references. Ties are resolved by stable sample-ID order.

This three-way design separates category-quota benefit from visual-distance
benefit. No weights, thresholds, or category-specific rules may be changed after
observing MPDD results.

## Post-hoc metrics

Primary: anomaly images found in the 20-image query.

Secondary: annotations to first anomaly, product-category coverage, unique
anomaly types, mask connected components, mask area, within-query visual
redundancy, and distance from the initial set. Category and anomaly-type yields
are also reported. All are audit-only and cannot affect selection.

## Pre-registered detector-training gate

Every check must pass:

1. Plain Random mean anomaly yield is at most 5.0 of 20.
2. Hierarchical DINO minus plain Random mean anomaly yield is at least +2.0.
3. The paired 95% bootstrap CI lower bound for that difference is above zero.
4. Its strict paired win rate is at least 0.65.
5. Mean product-category coverage is no worse than Random by more than 0.25.
6. The minimum leave-one-product-category-out anomaly-yield difference versus
   Random is above zero.
7. Hierarchical DINO minus Category-Balanced Random mean anomaly yield is at
   least +1.0.
8. The paired 95% bootstrap CI lower bound for that decomposition difference is
   above zero.
9. Its strict paired win rate is at least 0.60.

If any check fails, no detector training is authorized. Passing only authorizes
a separately specified development-only detector experiment; it is not evidence
of detector performance by itself.


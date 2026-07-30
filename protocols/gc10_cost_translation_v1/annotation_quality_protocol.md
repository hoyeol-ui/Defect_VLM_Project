# GC10 Human Annotation Quality Protocol

Protocol version: `gc10-cost-translation-v1-candidate`

Status: candidate; thresholds other than the frozen geometry rule require researcher approval before Phase A.

## 1. Purpose

Annotation time alone is not a valid cost measure if a faster annotator misses defects, assigns the wrong class, or draws boxes that require extensive review. This protocol therefore evaluates time and quality separately and records adjudication as an additional cost.

## 2. Reference data and blinding

- The reference is the existing GC10 acquisition annotation used only after the human submission is frozen.
- Annotators must not see reference boxes, class IDs, rare/small flags, strategy, seed, query rank, or selection frequency.
- The quality evaluator joins human and reference annotations by opaque `sample_id` only after both are exported and hashed.
- The reference annotation is not assumed infallible. Suspected reference errors must be flagged, reviewed blind to strategy, and reported separately; they must not be silently edited.

## 3. Frozen definitions

- Geometry-match IoU threshold: **0.50**
- Assignment: Hungarian one-to-one assignment maximizing total IoU
- Rare classes: **8, 9, 10**
- Small defect: reference bbox area ratio **≤ 0.004831**
- Small-threshold source:
  `runs/gc10_taxonomy_protocol/gc10_protocol_20260715/gc10_acquisition_bbox_gt_audit.csv`
- Exact acquisition-pool lower quartile checked by the manifest builder:
  `0.0048310546875`
- Threshold implementation:
  `scripts/08_gc10_cost_translation/build_protocol_manifests.py`,
  constant `SMALL_BOX_AREA_RATIO_THRESHOLD`

## 4. Matching algorithm

For each image:

1. Construct an IoU matrix between all frozen human boxes and reference boxes.
2. Apply Hungarian one-to-one assignment to maximize total IoU without using class in the assignment cost.
3. Retain an assigned pair as a geometry match only when IoU ≥ 0.50.
4. Geometry match plus identical class is a class-aware true positive.
5. Geometry match with a different class is a classification error. It contributes one class-aware false positive for the human class and one class-aware false negative for the reference class.
6. An unmatched reference box is a false negative.
7. An unmatched human box is a false positive.
8. A human box that substantially duplicates another human box for the same reference object is counted separately as a duplicate. The duplicate rule must be implemented as a deterministic post-assignment rule and frozen before Phase A.
9. No greedy rematching is allowed after viewing arm results.

The Hungarian assignment is geometry-first so classification quality can be measured independently. A class-aware assignment would hide some wrong-class localization matches.

## 5. Required metrics

### Detection and omission

- box precision
- box recall
- false-negative count
- image-level missed-defect rate
- false-positive count

### Localization and classification

- matched-box mean IoU
- matched-box median IoU
- class accuracy on geometry-matched pairs
- duplicate-box count

### Safety subgroups

- small-defect recall
- rare-class recall, pooled classes 8–10
- per-class recall for classes 8, 9, and 10
- small-and-rare intersection recall when denominator is nonzero

### Review burden

- adjudication correction count
- boxes added, deleted, moved/resized, or relabeled during adjudication
- adjudication active time
- final label provenance: first annotator, consensus, or adjudicated

## 6. Zero-denominator rules

- If a metric denominator is zero for an image, store `null`, not zero.
- Aggregation denominators must be reported.
- Rare/small metrics are aggregated from counts before division; per-image percentages are secondary.
- An acquisition seed with no reference example for a safety subgroup is `NA` for that subgroup and is not silently treated as a pass.

## 7. Adjudication

The exact adjudication policy is unresolved. Permitted candidates are listed in `threshold_option_matrix.md`. Whichever policy is selected must:

- be blind to strategy and acquisition seed;
- preserve both original annotations;
- record disagreement reason;
- record correction events and active time;
- produce a frozen label artifact with SHA-256;
- define which label is used by the Phase B primary learner.

Review/adjudication time is part of human cost. It cannot be excluded because an arm produced more difficult images.

## 8. Quality analysis boundary

This quality audit measures human annotation behavior on the fixed GC10 acquisition pool. It does not validate GC10 ground truth, industrial deployment quality, or annotator performance at another factory, lot, product, camera, or annotation tool.

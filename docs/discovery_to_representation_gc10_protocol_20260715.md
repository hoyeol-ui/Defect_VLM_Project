# Discovery-to-Representation (D2R) development protocol

Date frozen: 2026-07-15

## Decision being tested

The previous GC10 result showed that frozen DINO was useful for cold-start discovery but did not reliably translate rare-class exposure into rare-class AP. The most concrete failure mode was repeated selection of a small set of global prototypes across acquisition seeds. This development experiment tests one narrow repair:

> Can a sequential `discovery -> local representation -> random guard` query preserve DINO's taxonomy-discovery advantage while increasing within-class and cross-seed variation enough to justify a detector follow-up?

This is not a new VLM-score experiment. Qwen2-VL, Qwen2.5-VL, and Qwen3-VL signals remain excluded after their frozen paired-compliance gates failed.

## Scope and claim status

- Dataset: GC10-DET acquisition pool only.
- Evaluation used by the selection audit: acquisition XML, joined after each permitted annotation return or after selection for post-hoc metrics.
- Development detector evaluation, if authorized: existing development split only.
- Final locked test: never opened or evaluated.
- Status: mechanism-development evidence only. GC10 already motivated this algorithm, so it cannot provide an independent confirmatory claim.
- Detector backbone: unchanged YOLOv8n for causal comparability. A backbone change is a later, separate robustness question and is not mixed into this selector test.

## Frozen budget and strategies

- Acquisition seeds: 0-199 for the selection-only audit.
- Initial labeled set: 20 images, identical reconstruction to the frozen Random-vs-DINO audit.
- Query budget: 20 images.
- Comparators:
  - `GTFreeRandom`: 20 random images.
  - `FrozenDINOVisualDiversity`: 20 global DINO farthest-first images.
  - `D2RDiscoveryRepresentationGuard`: 10 discovery + 5 representation + 5 guard images.

The frozen Random and pure-DINO selections must replay exactly against the prior 200-seed records before D2R is evaluated.

## D2R algorithm

### Stage A: discovery (10 images)

Select 10 images with the already frozen DINOv2-small farthest-first rule. No XML, class, bbox, path, filename, or source-folder information is available to this stage.

### Annotation return boundary

The initial 20 and newly selected discovery 10 are treated as annotated. Only their returned class IDs are made available. This is standard sequential active-learning feedback, not access to labels in the unselected pool. Every feedback access is written to `gc10_d2r_feedback_access_log.csv`.

### Stage B: representation repair (5 one-image micro-rounds)

For each micro-round:

1. Count labeled images for each class observed so far.
2. Target the currently least represented observed class; deterministic usage count and class ID break ties.
3. Form a normalized DINO prototype from labeled images containing that class.
4. Find the 32 unselected candidates closest to that prototype.
5. Within this local neighborhood, choose the candidate farthest from the entire currently labeled set.
6. Exclude exact-SHA duplicates of any already labeled/selected image.
7. Annotate that one image, log the returned class IDs, and update the next micro-round.

The selector never asks whether an unselected candidate has the target class. Target-hit rate is evaluated only after the candidate has been selected.

### Stage C: random guard (5 images)

Select 5 images uniformly at random from the remaining pool. This protects against DINO representation misspecification and gives every remaining sample non-zero acquisition probability.

## Selection-only outputs

- `gc10_d2r_selection_records_posthoc.csv`: all three query sets, ranks, D2R stages, and post-hoc labels.
- `gc10_d2r_feedback_access_log.csv`: exact label-feedback boundary for D2R.
- `gc10_d2r_seed_strategy_metrics.csv`: coverage, rare yield, bbox/instance yield, initial distance, redundancy, and within-class distance.
- `gc10_d2r_paired_metric_summary.csv`: paired D2R-minus-Random and D2R-minus-DINO estimates with acquisition-seed bootstrap intervals.
- `gc10_d2r_per_class_mean_yield.csv`: per-class selected images and instances.
- `gc10_d2r_cross_seed_stability.csv`: unique images, top-10 concentration, and mean query overlap across 200 seeds.
- `gc10_d2r_stage_diagnostics.csv`: discovery/representation/guard yield and representation target-hit diagnostics.
- `gc10_d2r_selection_gate.csv` and `gc10_d2r_selection_summary.md`.

## Frozen selection gate

All checks must pass before any new detector is trained.

Discovery value versus Random:

- Combined unique-class mean difference >= +0.50 and paired bootstrap lower bound > 0.
- Rare-class query-image mean difference >= +1.00 and paired bootstrap lower bound > 0.
- Unique rare-class mean difference >= +0.25 and paired bootstrap lower bound > 0.
- Query-instance mean difference >= -1.00.

Retention and representation repair:

- Combined unique classes versus pure DINO >= -0.25.
- Rare-class images versus pure DINO >= -1.00.
- Class 9 and class 10 image yield versus Random each >= -0.10.
- Unique query images across seeds >= 1.50 times pure DINO.
- Top-10 query-slot concentration <= 0.75 times pure DINO.
- Mean cross-seed query overlap at least 2 images lower than pure DINO.
- Rare-class within-class DINO distance is not below pure DINO.
- Representation target-hit rate >= 0.50.

This gate is deliberately mechanism-oriented. A FAIL ends the branch without YOLO training.

## Conditional detector follow-up

Only after selection PASS:

- Acquisition seeds: 0-4.
- Training seeds: 42, 43, 44.
- Labeled images: 40.
- Backbone/training: the frozen YOLOv8n, 75 epochs, 640 px, same settings as the existing 30-model run.
- Reuse: the 15 Random and 15 pure-DINO results are reused only after exact query-set replay and configuration/hash checks.
- New training: 15 D2R models only.
- Evaluation: development split only; final test remains locked regardless of result.

The detector gate asks D2R to keep the known aggregate benefit and repair the rare-class regression:

- mAP50-95 gain over Random >= +0.010, positive acquisition-seed bootstrap lower bound, and wins in at least 4/5 acquisition seeds.
- Recall gain over Random >= +0.020.
- Rare macro AP50-95 gain over Random >= 0 and gain over pure DINO >= +0.015.
- Frequent macro AP50-95 non-inferiority to Random >= -0.020.
- D2R aggregate mAP50-95 non-inferiority to pure DINO >= -0.005.
- No individual class AP50-95 difference versus Random below -0.030.

Even a detector PASS is development evidence. The next valid confirmatory step would require freezing D2R unchanged and using an independent mixed normal/defect industrial dataset.

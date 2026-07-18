# GC10 D2R label-aware retrieval diagnostic protocol

Frozen before execution: 2026-07-15

## Question

The frozen D2R representation rule failed because max-novelty selected the intended class only 29.9% of the time. Class 8 dominated the failure: centroid-nearest hit rate was 11.7% and centroid top-32 purity was 5.0%.

This post-hoc diagnostic asks whether class 8 information is recoverable from the labels and DINO embeddings that were legitimately available at each of the existing 1,000 micro-rounds.

## Fixed trajectory and feedback boundary

- Reuse the exact 1,000 frozen micro-round states.
- At micro-round `r`, use only the initial 20, discovery 10, and the `r-1` representation labels already returned in the frozen trajectory.
- Do not update later states with hypothetical alternative selections. The result is therefore an off-policy one-step diagnostic, not a new selector experiment.
- Exclude unselected GT from scoring. XML is joined only after ranking to calculate diagnostic hit/purity.
- Do not open or evaluate the final split.
- Do not train a detector.

## Fixed retrieval scores

All scores rank the complete remaining non-duplicate acquisition pool.

1. `centroid_similarity`: mean cosine similarity to all labeled target-class exemplars. This is the centroid-nearest reference.
2. `nearest_target_exemplar`: maximum cosine similarity to any labeled target-class exemplar.
3. `contrastive_margin`: maximum target-exemplar similarity minus maximum non-target labeled-exemplar similarity.
4. `top3_target_similarity`: mean of the three highest target-exemplar similarities, or all available target exemplars when fewer than three exist.
5. `top5_target_similarity`: mean of the five highest target-exemplar similarities, or all available target exemplars when fewer than five exist.

Ties are resolved by blind `sample_id`. Paths, filenames, source folders, production groups, XML, and bbox information are unavailable to the scores.

## Primary class-8 recovery gate

Each alternative method is evaluated independently. A method clearly recovers class 8 only if all checks pass:

- Class-8 top-1 target hit rate >= 0.50.
- Acquisition-seed bootstrap 95% CI lower bound for class-8 hit rate >= 0.40.
- Class-8 precision@5 >= 0.40.
- At least one class-8 image appears in the top 32 for >= 0.80 of class-8 rounds.
- Class-8 top-1 hit improvement over centroid similarity >= +0.25.
- Unique selected-image ratio across class-8 rounds >= 0.25.
- Overall target-class top-1 hit rate across all 1,000 rounds >= 0.55.

The fixed method priority is:

1. `contrastive_margin`
2. `nearest_target_exemplar`
3. `top3_target_similarity`
4. `top5_target_similarity`

If more than one passes, the first method in this list is the only method eligible for D2R-v2 preregistration.

## Decision

- `RECOVERABLE`: at least one alternative passes every gate. This authorizes writing a new D2R-v2 200-seed selection-only protocol. It does not authorize detector training.
- `NOT_RECOVERABLE`: no alternative passes every gate. Close the D2R representation branch on frozen DINOv2-small for GC10.

Any later D2R-v2 result remains GC10 development/tuning evidence and requires independent-dataset confirmation.

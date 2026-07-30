# DCAL-XAI V2.3: Detector Signal Validity Audit

Date frozen: 2026-07-18

## Motivation

V2.2 showed that K40 global-DINO cluster coverage reduced missing-class risk
but failed the downstream rare-AP and recall gates. The remaining defensible
path to detector-coupled AL is to verify that a detector-native, label-free
uncertainty score is associated with actual detection error before performing
another acquisition or training run.

## Inputs and safety

- Existing 15 `Random140 x YOLOv8n` checkpoints only
- Five acquisition realizations: 20000-20004
- Three training seeds: 42, 43, 44
- Existing 232-image development split only
- No new training
- No acquisition-pool selection
- Locked final test is never read or evaluated

Predictions are generated and sealed without XML/GT. Development GT is joined
only in the post-hoc audit stage.

## Frozen prediction settings

- Image size: 640
- NMS inference floor: confidence 0.001, IoU 0.70, max detections 300
- Operational error threshold: confidence 0.25
- Class-aware matching IoU: 0.50

## Label-free signals

For each acquisition realization, the three training-seed models produce:

- mean maximum-confidence deficit
- no-detection fraction
- predicted-count disagreement
- maximum-confidence disagreement
- class-presence entropy

Every component is percentile-ranked within the 232 development images. The
predeclared primary signal is the mean of the five ranks:
`ensemble_combined_uncertainty`.

Confidence-only and disagreement-only signals are secondary diagnostics. They
cannot rescue a failed primary gate.

## Post-hoc GT outcomes

- class-aware false positives and false negatives at IoU 0.50
- total error count
- false-negative count and rate
- majority-model error indicator
- rare-class false-negative count for classes 8, 9, 10

## Frozen primary gate

Across the five acquisition realizations, all checks must pass:

- top-20% total-error enrichment mean >= 1.50
- bootstrap 95% CI lower bound for error enrichment > 1.00
- mean AUROC for majority-model error >= 0.65
- AUROC > 0.50 in at least 4/5 acquisition realizations
- mean Spearman correlation with total error >= 0.20
- positive Spearman correlation in at least 4/5 realizations
- mean pairwise training-seed confidence-rank correlation >= 0.50

A PASS authorizes only a prospectively frozen selection-only query audit from
`Random140`. A FAIL closes detector uncertainty as an AL acquisition signal in
the present GC10 protocol. Neither outcome authorizes final-test access.

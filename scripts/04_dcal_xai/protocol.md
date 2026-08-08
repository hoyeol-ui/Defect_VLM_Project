# DCAL-XAI GC10 Round-1 Protocol

Date frozen: 2026-07-18

## Research question

Can a learner-coupled detector difficulty signal, combined with frozen visual
diversity, improve a 20-image cold-start query over Random without damaging
rare-class coverage, and can every selected image be accompanied by an
explanation grounded only in detector evidence?

This is a new hypothesis. It is not a rescue or continuation of the failed
GT-free VLM-consistency selector.

## Roles

- YOLOv8n produces the acquisition difficulty signal.
- Frozen DINOv2 embeddings protect batch diversity inside a detector-difficulty
  shortlist.
- The explanation layer verbalizes numeric detector evidence only.
- A VLM is not used for selection. Any later VLM may only paraphrase a frozen
  evidence packet and must pass a separate faithfulness gate.

## Data and split

- Dataset: GC10-DET
- Existing duplicate-safe protocol: `gc10_protocol_20260715`
- Acquisition: 1,836 images
- Development: 232 images
- Final: 224 images, locked and never opened by this runner
- Acquisition seeds: 0-4
- Shared initial set per acquisition seed: 20 images
- Query: 20 images

XML/class/bbox data are available for the initial labeled set and become
available for query images only after a strategy has committed its selection.
They are never inputs to detector scoring or diversity selection.

## Strategies

1. `Random`
2. `DetectorDifficulty`
3. `DetectorDifficultyDiversity` (primary)

The initial detector is shared by all three strategies within an acquisition
seed. Therefore strategy differences arise only from acquisition.

## Detector difficulty

The detector predicts each unlabeled image in its original and horizontally
flipped forms. Flipped boxes are mapped back to original coordinates.

The frozen score is:

```
0.40 * confidence_deficit
+ 0.30 * localization_instability
+ 0.15 * class_instability
+ 0.15 * count_instability
```

GC10 is an all-defect pool. No detections in both views are therefore treated
as potential missed detections, not confident normal predictions.

This score is called `difficulty`, not entropy: post-NMS YOLO output does not
expose a calibrated full class distribution.

## Diversity

The hybrid method first keeps the top 100 difficulty candidates. It then
greedily combines:

- 0.65 detector-difficulty rank
- 0.35 cosine distance from the initial set and already selected query images

DINO uses images only. No labels, paths, source folders, XML, or final data are
visible to selection.

## Stages and stopping rules

### Stage 0: audit

Verify protocol flags, manifest alignment, embedding shape, model hash, initial
sets, and output isolation. No training and no inference.

### Stage 1: acquisition

Train one shared warm-start detector per acquisition seed, score the unlabeled
pool, commit all strategies, then join GT for post-hoc selection audit.

All selection-gate checks must pass before detector confirmation:

- primary combined class coverage vs Random >= -0.25
- rare-image yield vs Random >= -0.50
- instance yield vs Random >= -1.00
- selected difficulty vs Random >= +0.05
- pairwise similarity vs DetectorDifficulty <= 0
- cross-seed top-10 concentration <= 0.40

### Stage 2: development detector confirmation

Train each 40-image set from scratch with three training seeds and evaluate the
existing development split only.

Primary `DetectorDifficultyDiversity - Random` gate:

- mean mAP50-95 >= +0.010
- at least 4/5 acquisition-seed wins
- acquisition-seed bootstrap CI lower bound > 0
- rare macro AP difference >= 0
- recall difference >= -0.020
- worst per-class AP difference >= -0.050

All checks must pass. Failure closes this branch; thresholds are not weakened.

## Claims and non-claims

Permitted if supported:

- detector-coupled difficulty changes which samples are annotated;
- diversity changes redundancy within a difficult shortlist;
- grounded evidence packets expose the numerical reason for each selection;
- development detector utility is conditional on the frozen gate.

Not permitted:

- VLM consistency is a valid acquisition uncertainty;
- the explanation reflects ground truth or human reasoning;
- human trust or acceptance improved without a user study;
- final-test performance;
- universal superiority across datasets or detector families.


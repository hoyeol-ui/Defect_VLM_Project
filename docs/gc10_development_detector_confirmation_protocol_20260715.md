# GC10-DET Development-Only Detector Confirmation Protocol

Date: 2026-07-15

## Authorization and scope

The pre-registered GC10 selection-only gate passed all 11 checks. This permits
one detector confirmation on the development split. It does not permit final
test access. The final 224-image manifest remains locked and must not be read by
this experiment.

## Frozen design

- Detector: local pretrained `yolov8n.pt`.
- Acquisition seeds: 0, 1, 2, 3, 4 (the first five seeds; not chosen from
  seed-level detector outcomes).
- Strategies: `GTFreeRandom` and `FrozenDINOVisualDiversity`.
- Labeled set per run: shared initial 20 plus the frozen strategy query 20.
- Training seeds: 42, 43, 44.
- Total models: 5 acquisition seeds x 2 strategies x 3 training seeds = 30.
- Training: 75 fixed epochs, image size 640, batch 16, deterministic mode,
  pretrained weights, standard Ultralytics augmentation, no validation during
  fitting, and evaluation of `last.pt` only.
- Evaluation: the locked development split of 232 images.
- No hyperparameter, seed, budget, selector, or checkpoint choice may change
  after viewing detector results.

Training-seed results are first averaged within each acquisition seed. The five
acquisition seeds are the independent selection units for paired inference.

## Metrics

- Primary: development mAP50-95.
- Secondary: mAP50, precision, recall, per-class AP50-95.
- Rare macro AP50-95: classes 8, 9, 10.
- Frequent macro AP50-95: classes 1--7.
- Selection-set class and instance counts are retained for interpretation.

## Pre-registered development confirmation gate

All checks must pass:

1. Mean paired DINO-minus-Random mAP50-95 is at least +0.010.
2. At least 3 of 5 acquisition-seed averages have positive mAP50-95 difference.
3. The paired 95% bootstrap CI lower bound across the five acquisition-seed
   averages is above zero.
4. Mean rare macro AP50-95 difference is at least +0.020.
5. Mean class-8 AP50-95 difference is above zero.
6. Mean class-9 AP50-95 difference is above zero.
7. Mean recall difference is at least -0.020.
8. Mean frequent-class macro AP50-95 difference is at least -0.020.

A failure closes detector escalation for this branch. A pass supports a
development-level cold-start triage result but does not automatically authorize
the final test; final access requires a separate explicit decision after the
complete development analysis is frozen.

## Robustness note

Across the selection audit, DINO used 143 unique images for 4,000 query slots;
the most frequent image appeared in 194/200 seeds and the top ten images filled
40% of all slots. The three most frequent prototypes were visually inspected
and were valid industrial images, not corrupt frames. This concentration means
the experiment tests whether a stable set of visual prototypes improves a
detector, not whether 200 selection seeds provide 200 independent DINO sets.


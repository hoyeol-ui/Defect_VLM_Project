# DCAL-XAI V2.1: Initial Budget Extension

Date frozen: 2026-07-18

## Trigger

The frozen V2 audit found no passing policy at budgets 20-120. The closest
candidate was `DINOClusterCoverageK40` at budget 120:

- design all-class/all-rare rate: 0.945/0.945
- design at-least-two-images-per-class rate: 0.700 (required 0.750)
- holdout at-least-two-images-per-class rate: 0.730

The V2 protocol explicitly required increasing initial annotation scope rather
than weakening a threshold when no candidate passed.

## Frozen extension

- Budgets: 140, 160, 180, 200
- Policies: unchanged (`Random`, DINO cluster coverage K=10/20/40)
- Design seeds: unchanged, 0-199
- Holdout seeds: unchanged, 10000-10199
- Metrics and gates: unchanged
- Selection inputs: blind IDs and frozen DINO embeddings only
- GT/XML: post-hoc audit only
- Detector training: prohibited
- Final-test access: prohibited

Choose the smallest passing budget, then the highest two-images-per-class rate,
then all-class rate, then policy name. The design-selected candidate must pass
the same gate on holdout seeds.


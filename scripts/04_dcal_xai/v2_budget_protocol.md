# DCAL-XAI V2: Data and Training-Scope Gate

Date frozen: 2026-07-18

## Motivation

The earlier experiments repeatedly used extremely small initial labeled sets:
15-20 images for 6-10 detection classes. At that scale:

- Random can already be strong when the pool is balanced;
- DINO can discover rare/outlying images but lose within-class representation;
- a warm-start detector can be too unstable to provide useful uncertainty;
- horizontal-flip inconsistency can be dominated by acquisition geometry;
- aggregate class coverage can hide too few images per class for learning.

V2 therefore determines a stable initial labeled scope before defining another
detector acquisition signal.

## Question

What is the smallest GT-free initial sampling budget that reliably provides
all-class, rare-class, instance, and production-group coverage on the GC10
acquisition pool?

This is a selection-only design audit. It performs no detector training and
does not open the final manifest.

## Candidate budgets and policies

Budgets: 20, 40, 60, 80, 100, 120 images.

Policies:

1. `Random`
2. `DINOClusterCoverageK10`
3. `DINOClusterCoverageK20`
4. `DINOClusterCoverageK40`

DINO clusters use the already frozen 384-dimensional embeddings. Cluster
coverage selects one random item from as many distinct clusters as possible,
then fills the remaining budget uniformly from the remaining pool. It does
not use class, XML, bbox, path, source folder, or production-group fields.

## Seed separation

- Design Monte Carlo seeds: 0-199
- Holdout Monte Carlo seeds: 10000-10199

The design seeds choose the policy/budget. The same frozen candidate must pass
again on the holdout seeds. This is seed generalization, not independent
dataset generalization.

## Minimum initial-set gate

For a candidate to pass on the design seeds:

- probability of all 10 classes present >= 0.90
- probability of all rare classes 8/9/10 present >= 0.90
- probability of at least two images per class >= 0.75
- mean production-group count vs Random at the same budget >= -1.0
- mean instance count vs Random at the same budget >= -2.0
- mean DINO pairwise similarity vs Random at the same budget <= +0.02

Choose the smallest passing budget. At that budget, choose the policy with the
highest two-images-per-class rate, then all-class rate, then the simpler name
as a deterministic tie-break.

The selected candidate must pass the identical checks on holdout seeds.

## Consequence

- If no candidate at budget <=120 passes, do not build another cold-start AL
  selector. Increase the initial annotation scope or change the dataset/task.
- If a candidate passes, use that fixed initial policy and budget for the next
  backbone-stability stage.
- Detector uncertainty is introduced only after the chosen initial scope
  yields a stable learner and its score correlates with actual detector error.

GT is joined only after every sampled set is committed in memory. It is used
to audit and choose an operating regime, not by any sampling policy.


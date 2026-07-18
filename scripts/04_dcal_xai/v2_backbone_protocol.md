# DCAL-XAI V2.2: Initial Data Policy x Backbone Stability

Date frozen: 2026-07-18

## Prerequisite

V2.1 passed with `DINOClusterCoverageK40` at an initial budget of 140 on both
design and holdout Monte Carlo seeds. Detector uncertainty remains prohibited
until a stable learner is identified.

## Question

Do the initial data policy and learner capacity independently or interactively
change development detector performance and training stability at 140 labels?

## Factorial design

- Data policy: `Random140`, `ClusterK40_140`
- Backbone: `YOLOv8n`, `YOLOv8s`
- New acquisition-set seeds: 20000-20004
- Training seeds: 42, 43, 44
- Models in main experiment: 2 x 2 x 5 x 3 = 60
- Epochs: 75
- Evaluation: existing development split only
- Final: locked

The acquisition policies see blind IDs and frozen DINO embeddings only. GT is
joined after the 140 IDs are committed to create training labels and post-hoc
composition reports.

## Staging

1. `audit`: materialize selection IDs and composition report; no training.
2. `smoke`: seed 20000, training seed 42, four factorial models.
3. `screen`: all five acquisition seeds and three training seeds for
   Random/K40 with YOLOv8n only (30 models).
4. `main`: expand to YOLOv8s only if the YOLOv8n screen passes every frozen
   stable-learner threshold. Existing YOLOv8n results are recovered by signed
   cache, so the complete factorial contains 60 models.

Smoke results never authorize a final claim.

This staged amendment was frozen after the technical smoke and before any
multi-seed main result. A YOLOv8n screen failure terminates the experiment and
prevents the remaining 30 YOLOv8s fits. The thresholds are unchanged from the
stable-learner decision below.

## Analysis

Acquisition set is the inferential unit. Three training seeds are averaged
inside each acquisition realization.

Report:

- K40 - Random within each backbone
- v8s - v8n within each data policy
- difference-in-differences interaction
- mAP50-95, recall, precision
- frequent and rare macro AP
- per-class AP
- mean within-set training-seed standard deviation

## Stable-learner decision

For each backbone, K40 is eligible for the next signal-validity stage only if:

- K40 - Random mAP50-95 >= -0.005
- K40 - Random rare macro AP >= -0.010
- worst per-class K40 - Random AP >= -0.050
- mean within-acquisition training-seed mAP standard deviation <= 0.020
- recall difference >= -0.020

If both backbones are eligible, select v8s only when its K40 mAP exceeds v8n
by at least 0.005; otherwise select v8n for the lower operational cost.

Passing this stage authorizes only a detector error-association audit. It does
not authorize an AL query, VLM acquisition, or final-test access.

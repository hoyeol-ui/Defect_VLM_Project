# Controlled Random–Frozen DINO Protocol

## Authority and scope

The normative base is `confirmatory_v1/confirmatory_protocol_v1.yaml`; v1.1 adds the locked-final outcome source and execution lifecycle in `confirmatory_v1_1/confirmatory_protocol_v1_1.yaml`, associated specifications, full-run registry, and seals. The base freeze commit is `56369eaa0108e10d6d9a0f84a21a7e2c54fc4cfc`. Later final-analysis files report outcomes but do not rewrite the training design.

## Dataset freeze

| Dataset | raw rows inspected | curated pixel-unique records | acquisition | development | locked final | budgets |
|---|---:|---:|---:|---:|---:|---|
| GC10-DET | 2,312 | 2,292 | 1,605 | 229 | 458 | 161, 241, 321, 401, 481, 561 |
| NEU-DET | 1,800 | 1,798 | 1,259 | 180 | 359 | 126, 189, 252, 315, 378, 441 |

GC10 exclusions were 6 missing XMLs, 2 invalid ground truths, and 12 collapsed same-pixel/equivalent-label duplicates. NEU excluded two same-pixel identities with conflicting ground truth. The split is deterministic rare-label-first iterative multilabel stratification at 70/10/20, seed `20260822`, with largest-remainder size rounding and canonical pixel-identity groups as the split unit. There are no observed split/class support cells with zero support.

The acquisition universe alone is visible to the selector. Development and final membership, labels, and outcomes are denied. The split is prospectively locked for this study, not described as an untouched historical test set.

## Identity and leakage guard

Each sample ID is the SHA-256 of canonical RGB uint8 pixels plus dimensions; filename, directory, class, and box are excluded. Identical-pixel/equivalent-GT images collapse to the minimum raw SHA; identical pixels with conflicting GT exclude the full identity group. Selector manifests expose only sample IDs and membership needed for acquisition.

Allowed selector information is raw image content inside the DINO adapter, sealed DINO embeddings, class-neutral sample IDs, current labeled/unlabeled membership, acquisition seed/round, and picks already made in that query. GT class/box, rarity, anomaly labels, bounding-box size, filenames, development/final metrics, future rounds, and learner seeds are forbidden.

## Initial set and budgets

For each dataset/acquisition seed, Round0 is a uniform 10% draw from the fixed acquisition universe, rounded once with Decimal half-up. Each of five subsequent queries is 5% of the original fixed universe, not the remaining pool. Round0 is shared by Random and DINO for a paired comparison. All selections are cumulative and sealed by SHA-256 of sorted sample IDs.

## Random-PCG64-v1

Random samples uniformly without replacement from the remaining pool ordered by ascending sample ID. It uses NumPy `2.4.6`, `Generator(PCG64)`, and a deterministic seed sequence derived from protocol namespace, dataset, acquisition seed, round, and purpose. Learner seed is not an input. A trajectory is generated once per dataset/acquisition seed and reused for learner A/B.

## FrozenDINO-FF-v1

- model: `facebook/dinov2-small`
- recovered revision: `ed25f3a31f01632728cabb09d1542f84ab7b0056`
- representation: `outputs.pooler_output`, 384 dimensions, float32, row-wise L2 normalized
- preprocessing: RGB, shortest edge 256, bicubic, center crop 224, ImageNet mean/std
- inference: eval/inference mode, FP32, no autocast/TF32, deterministic algorithms
- selection score: `d(x,R) = 1 − max cosine(z_x,z_r)`
- update: greedy farthest-first; immediately add each chosen point to the reference set
- tie-break: ascending sample ID for absolute score difference at most `1e-12`

The complete acquisition universe is embedded before selection. Missing, duplicate, non-finite, zero-norm, wrong-dimension, or extra embeddings are hard failures. This is a prospectively cleaned implementation with historical lineage; it is not claimed to be a bit-exact reproduction of every historical DINO script.

## Seeds and pairing

There are ten SHA-derived acquisition seeds per dataset. Each acquisition has learner slots A and B, also SHA-derived without outcome screening. Random and DINO use the same A/B learner seeds within an acquisition. This produces paired strategy contrasts while keeping learner variation nested inside acquisition variation. Exact seed values are in the protocol YAML and `confirmatory_v1/data_freeze/seed_manifest.csv`.

## YOLO lifecycle

- model family: YOLOv8n; sealed base checkpoint SHA-256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`
- Ultralytics `8.4.90`; PyTorch `2.12.1+cu126`
- 100 epochs, image size 640, batch 8, workers 4
- SGD, `lr0=0.01`, `lrf=0.01`, momentum `0.937`, weight decay `0.0005`
- deterministic mode, AMP enabled, standard recorded augmentation settings
- independent cold start from the same sealed checkpoint at every budget; no warm-start and no resume
- development split enabled during training; `best.pt` chosen by Ultralytics fitness
- locked-final evaluation uses `best.pt`, `conf=0.001`, `iou=0.7`, `max_det=300`

Round0 is physically shared only when dataset, acquisition, learner, manifest, and configuration hashes match.

## Why exactly 440 detector jobs?

Without sharing Round0, the conceptual design contains:

`2 datasets × 2 strategies × 10 acquisitions × 2 learners × 6 budgets = 480 strategy-budget rows`.

Random and DINO have an identical selected set at Round0. Training it twice per pair would create 40 redundant jobs:

`2 datasets × 10 acquisitions × 2 learners = 40 shared Round0 jobs`.

The executed lifecycle is therefore:

| component | count |
|---|---:|
| shared Round0 | 40 |
| Random R1–R5 | 2 × 10 × 2 × 5 = 200 |
| DINO R1–R5 | 2 × 10 × 2 × 5 = 200 |
| **total** | **440** |

For analysis, each shared Round0 outcome is attached to both conceptual curves, restoring 480 conceptual rows and 80 six-point learner curves. Sharing saves computation without changing a trajectory.

## Primary estimand and AULC

For learner curve values `m_i` at labeled fractions `f_i`,

`AULC = [Σ(i=0..4) ((m_i+m_(i+1))/2)(f_(i+1)−f_i)] / (f_5−f_0)`.

All six locked-final mAP50-95 values are required; interpolation and development substitution are forbidden. Within dataset/acquisition/strategy, A/B learner AULCs are averaged. The paired effect is then mean-learner DINO AULC minus mean-learner Random AULC. Ten effects are summarized separately for each dataset; pooling GC10 and NEU as `n=20` is forbidden.

## Statistical plan

- point estimate: arithmetic mean of ten paired acquisition effects
- interval: 100,000-replicate acquisition-cluster percentile bootstrap with frozen dataset-specific PCG64 seeds
- test: two-sided exact sign-flip over all `2^10` assignments
- multiplicity: Holm adjustment across the two primary dataset p-values
- no minimum-effect threshold; estimation first
- secondary/subgroup intervals are descriptive and cannot rescue the primary endpoint

An interval entirely above zero is a positive difference under this design; an interval containing zero means no clear advantage established, not equivalence.

## Secondary and subgroup endpoints

Secondary endpoints include final mAP50-95, recall trajectory/final recall, class AP50-95, GC10 rare macro, NEU macro, NEU worst class, and post-seal discovery/composition. GC10 rare classes were frozen from bottom-three image frequency in the curated pre-split universe: rolled pit, crease, and waist folding. NEU has no artificial rare group; all six-class macro and worst-class endpoints are reported.

Ground truth is joined only after a query/cumulative manifest hash is written, and post-seal composition results cannot feed acquisition.

## Failure, retry, and final lock

Retries were allowed only for infrastructure/process/transient-I/O failures with identical protocol, code, manifests, models, seeds, and environment. They do not add replicates. Outcome-based seed replacement, silent cell deletion, incomplete-curve AULC, or selector leakage are forbidden.

Final evaluation remained hard-locked until 440 training cells closed, training integrity passed, frozen hashes matched, and explicit authorization was recorded. After final evaluation, retraining, retuning, threshold changes, new primary tests, and development substitution were forbidden. Training and final audits both passed.

## Deviations and data note

No scientific protocol deviation was recorded. One exact duplicate bounding box in NEU final `crazing_120.xml` was consistently removed by Ultralytics in every evaluation; the source annotation and all outputs were preserved, and the duplicate audit is public. Final evaluation runtime was operationally separate from training and did not alter the design.

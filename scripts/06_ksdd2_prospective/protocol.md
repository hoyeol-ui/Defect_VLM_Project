# KSDD2 PatchCore Prospective G1-G4 Protocol

Prepared: 2026-07-26  
Protocol ID: `KSDD2_PATCHCORE_G1_G4_V1`  
Status before execution: `READY_TO_FREEZE`

## Research role

This is a prospective authorization case for the validity-gated workflow. It
is not a search for a winning selector. The candidate is the unmodified
PatchCore image anomaly score implemented by `anomalib==2.6.0`. Random is the
only comparator.

The branch asks whether a local patch-memory anomaly score:

1. measures defect-bearing-image and bbox-count variation on a target-blind
   development split (`G1`);
2. enriches defect-bearing images at query budget 100 (`G2`);
3. avoids material collapse of small-defect and defect-size composition
   (`G3`); and
4. repeats its paired discovery effect across ten target-blind acquisition
   perturbations (`G4`).

Passing G1-G4 grants only `BOUNDED_SCREEN`. It does not establish downstream
detector utility.

## Dataset and roles

- Dataset: KolektorSDD2, CC BY-NC-SA 4.0.
- Official archive SHA-256:
  `edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8`.
- Official train: 2,331 unique images after excluding two archive duplicate
  entries containing `(copy)`.
- Target-blind development: exactly 466 train images selected by a salted
  SHA-256 ordering of sample IDs.
- Acquisition: remaining 1,865 official-train images.
- Official test: 1,004 images listed only in `final_locked.csv`. Test image
  content and test masks are not read by G1-G4 code.

Every official-train image has a `_GT.png` mask. The pre-score blind manifest
records mask paths but never parses mask pixels. Only the initial labeled set
is revealed before PatchCore fitting, because confirmed normal initial images
form the nominal memory bank. Development and query masks are parsed only
after their score/selection CSVs are SHA-256 sealed.

## Frozen acquisition design

- Initial labeled set: 100 images.
- Query: 100 images.
- Acquisition seeds: 0-9.
- G1 reference seed: 0.
- Random query uses a disjoint deterministic RNG stream.
- Candidate selection: descending raw PatchCore score, then ascending
  `sample_id`.

PatchCore parameters are the `anomalib==2.6.0` defaults explicitly written in
`config.yaml`: Wide-ResNet-50-2, layer2/layer3, 0.10 coreset ratio, 9 nearest
neighbors, and 256x256 input. No backbone, layer, coreset, neighbor, resize, or
score ensemble search is permitted. The official freeze loads the actual
pretrained feature extractor and records a sorted state-dict SHA-256; every
prospective scoring run must reproduce that parameter fingerprint.
The pinned source is `timm/wide_resnet50_2.racm_in1k`, with expected extractor
state hash `67569d0ff3ed2e26eb2add3e28437ebfc218713bd1052e0b91418207883d677f`.

## Sequential execution

### Stage 0 - blind preparation and freeze

1. Verify the official ZIP hash and dataset counts.
2. Build target-blind development/acquisition/final manifests.
3. Generate ten target-blind initial sets.
4. Freeze config, protocol, code, dataset manifests, Git commit, and
   environment versions before any development or query mask is parsed.

Official execution refuses a dirty Git worktree. `--allow-dirty` exists only
for synthetic tests and local dry-run development.

### G1 - signal validity

1. Reveal masks for seed-0 initial images only.
2. Fit PatchCore on confirmed-normal seed-0 initial images.
3. Score the target-blind development images without loading their masks.
4. Seal `g1_scores_seed00.csv`.
5. Only then parse development masks and calculate:
   - defect-image AUROC;
   - Spearman correlation with connected-component bbox count;
   - top-100 defect-image enrichment over development prevalence.

All frozen G1 checks in `config.yaml` must pass. Otherwise the branch returns
`STOP`; acquisition scoring is unauthorized.

### G2-G4 - selection-only audit

For each acquisition seed:

1. reveal only that seed's initial labels;
2. fit the same frozen PatchCore on confirmed-normal initial images;
3. score the remaining acquisition pool and seal the score CSV;
4. commit candidate top-100 and Random-100 CSVs without reading query masks;
5. seal both selection CSVs;
6. parse train masks only after all ten selections are sealed.

G2 requires positive-image discovery over Random with a paired bootstrap lower
bound above zero, mean enrichment at least 1.25, and positive deltas in at
least 80% of seeds. Per-seed enrichment uses the frozen Haldane correction
`(PatchCore positives + 0.5) / (Random positives + 0.5)` so a zero Random
count cannot create an infinite or post-hoc undefined ratio.

KSDD2 has one semantic defect class and no documented production-lot field.
G3 is therefore task-specific, not category/source safety. It checks:

- small-defect-bearing-image yield;
- three-bin connected-component size coverage;
- Jensen-Shannon divergence of the selected size distribution from the
  acquisition pool relative to Random; and
- concentration of positive candidate excess in one size bin.

G4 is acquisition-seed perturbation reproducibility on one official train
pool. It is not independent production-pool generalization.

## Authorization

| Condition | Output |
|---|---|
| G1 FAIL/NA | `STOP` |
| G2 FAIL | `STOP` |
| G2 PASS and G3 FAIL | `DISCOVERY_ONLY_STOP` |
| G4 FAIL | `STOP` |
| G1-G4 all PASS | `BOUNDED_SCREEN` |

`BOUNDED_SCREEN` authorizes a separately frozen learner protocol only. It does
not authorize score rescue, a second candidate, hyperparameter tuning,
additional backbones, or official-test access.

## Supported claim if executed as frozen

The protocol can support a statement about prospective gate conformance and,
conditional on the results, KSDD2 fixed-pool signal validity, discovery,
task-specific composition safety, and acquisition-perturbation
reproducibility.

## Prohibited claims

- universal PatchCore selector superiority;
- independent production-pool generalization;
- detector learning utility before a separate G5 bounded screen;
- annotation-time, GPU-hour, or monetary savings;
- workflow predictive accuracy, false-stop rate, or good-selector sensitivity;
- category/source safety, because KSDD2 supplies neither multiple semantic
  defect classes nor a documented production-lot identifier;
- any official-test result before separate final authorization.

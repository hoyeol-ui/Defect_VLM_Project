# VLM Explanation Consistency and Groundedness Validity Protocol

Date: 2026-07-15  
Status: pre-training validity stage  
Final test: locked and prohibited  
Detector training: prohibited in this stage

## 1. Decision being made

The previous experiments do not support the broad claim that a frozen VLM or
DINO acquisition score reliably beats Random for detector learning. This
protocol returns to the original research question before another selector is
developed:

> Do prompt-sensitive VLM responses and their visual groundedness predict
> detector errors or human annotation difficulty on industrial defect images?

The purpose is signal validation, not selector optimization. A positive result
only authorizes a separately pre-registered active-learning experiment. It does
not authorize final-test use.

## 2. Hypotheses

### H1. Legacy consistency validity

For the existing 99-image Qwen2-VL pilot, higher explanation consistency should
be associated with higher GT-oracle location/scale groundedness.

This is a retrospective diagnostic only. The legacy prompts asked different
questions (location, scale, appearance), so their SBERT similarity is not a
clean self-consistency measure.

### H2. Structured field consistency

Across five semantically equivalent prompt variants, the following field-level
instability measures should predict detector or annotation errors:

- defect-presence disagreement;
- defect-class disagreement;
- bounding-box IoU disagreement;
- location-zone disagreement;
- scale disagreement;
- abstention rate.

Free-text semantic similarity is secondary and cannot replace the structured
measures.

### H3. Groundedness adds information

Groundedness or box consistency should add predictive value beyond text-only
semantic consistency. A consistently hallucinated explanation is expected to
have high text consistency but low groundedness.

## 3. Data boundary

### 3.1 Retrospective legacy audit

- Source: `outputs/grounded_experiment_20260601/pilot_grounded_consistency_results.json`
- Expected source count: 99 images
- Any filename present in a locked final-test manifest is excluded before
  analysis.
- GT/XML is used only after VLM generation for post-hoc validity auditing.

### 3.2 Prospective structured pilot

- Development-only images: 200-400 recommended.
- Sampling strata must be frozen before VLM generation.
- Allowed roles: acquisition pool, development evaluation, or an explicitly
  named pilot split.
- Forbidden roles: final test, final locked, test-only confirmation.
- The manifest must contain `image_id`, `image_path`, `dataset`, and
  `split_role`.

## 4. Frozen prompt and model policy

- Default VLM: `Qwen/Qwen2-VL-2B-Instruct`, retained for continuity with the
  original proposal.
- Five prompts ask the same structured question with controlled paraphrases.
- Decoding is deterministic for the prompt-sensitivity study.
- Required JSON fields:
  `defect_present`, `defect_type`, `bbox_norm`, `location_zone`, `scale`,
  `appearance`, `visual_evidence`, `abstain`, and `confidence`.
- Prompt text, prompt hash, model ID, decoding configuration, and input manifest
  hash are logged.
- No class folder, source folder, XML, mask, or GT label is shown to the model.

## 5. Outcomes

### 5.1 Legacy primary outcomes

- Spearman correlation between consistency and normalized oracle groundedness.
- AUROC of `1 - consistency` for severe groundedness failure (`score = 0`).
- Mean groundedness difference between the lowest and highest consistency
  quartiles.
- Directional consistency across NEU-DET and GC10-DET.

### 5.2 Prospective primary outcomes

- AUROC/AUPRC for image-level detector error.
- AUROC/AUPRC for localization failure.
- Correlation with per-image detector loss or maximum matching error.
- If human annotations are collected: annotation time, inter-rater agreement,
  correction rate, and confidence calibration.

### 5.3 Required baselines

- detector confidence or entropy;
- augmentation/prediction stability when available;
- frozen DINO distance;
- object/box count and image-size covariates;
- text-only semantic consistency.

## 6. Gates

### 6.1 Legacy diagnostic screen

All checks are descriptive because the data and prompts already exist:

1. Spearman rho between consistency and groundedness is at least 0.20.
2. The bootstrap 95% lower bound for rho is above 0.
3. Severe-failure AUROC using inconsistency is at least 0.60.
4. Its bootstrap 95% lower bound is above 0.50.
5. The lowest-consistency quartile has groundedness at least 0.10 lower than
   the highest-consistency quartile.
6. The correlation direction is non-negative in both datasets.

Passing this screen justifies a prospective validity pilot. Failing it rejects
the legacy SBERT score as a useful validity signal.

### 6.2 Prospective signal-validity gate

Before any new detector training, all of the following are required:

1. The frozen combined structured signal improves error AUROC by at least 0.03
   over detector confidence alone.
2. The paired bootstrap lower bound of the AUROC improvement is above 0.
3. Groundedness/box agreement adds positive incremental value over text-only
   consistency.
4. The direction holds in every frozen dataset stratum and is not driven by a
   source-folder or official-split confound.
5. Missing/invalid VLM responses are reported and do not exceed 10%.

If the gate fails, the VLM signal may still be reported as an explanation audit
but must not be promoted to an acquisition score.

## 7. Optional active-learning extension

Only after the prospective gate passes:

1. detector uncertainty forms a candidate pool;
2. VLM structured inconsistency and groundedness stratify the candidate pool;
3. DINO neighborhoods provide within-pattern representative medoids;
4. Random, detector uncertainty, and the full reranker are compared using
   repeated acquisition and training seeds;
5. mAP50-95, recall, localization AP, rare-class AP, and AULC are binding;
6. the final test stays locked until a separate development gate passes.

This is a detector-first reranker study, not another global farthest-first or
standalone VLM selector contest.

## 8. Expected artifacts

- `legacy_validity_rows.csv`
- `legacy_validity_metrics.csv`
- `legacy_validity_gate.csv`
- `legacy_validity_summary.md`
- `structured_vlm_responses.jsonl`
- `structured_vlm_signal_scores.csv`
- prompt/model/manifest hashes and an exclusion audit


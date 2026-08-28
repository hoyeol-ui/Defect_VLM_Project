# Authorized Final Analysis

This directory is a post-freeze analysis layer. It does not modify the frozen
training code, acquisition manifests, seeds, checkpoints, learner configuration,
or statistical estimand.

## Order of operations

1. `src/authorize_final_evaluation.py`
   - re-hashes all 440 selected `best.pt` files and manifests;
   - validates the training-integrity audit and retry closure;
   - writes the external authorization record and canonical cell registry.
2. `src/run_final_evaluation.py`
   - validates the authorization through `confirmatory_v1_1.final_lock`;
   - materializes the prospectively locked final data after authorization;
   - evaluates each sealed checkpoint once and preserves one raw JSON row per
     scientific cell.
3. `src/analyze_final_results.py`
   - materializes the 480 conceptual strategy-budget rows by sharing Round0;
   - computes 80 six-point curves and 20 acquisition-level paired primary
     contrasts;
   - runs the frozen 100,000-resample acquisition bootstrap, exact sign-flip
     tests and two-dataset Holm adjustment;
   - reports secondary, subgroup, class-wise and selection-composition results.
4. `src/development_final_diagnostic.py`
   - compares development and locked-final effects as a supplementary
     diagnostic only. It never substitutes development values into primary.
5. `src/plot_final_results.py`, `src/build_final_report.py`, and
   `src/build_raw_confirmatory_master.py`
   - generate publication figures, the interpretation report, and the raw
   result canon from the canonical tables.

## Human interpretation

- `FINAL_EXPERIMENT_ACADEMIC_EVALUATION_20260828.md` gives the bounded success
  decision, effect-size interpretation, subgroup risk, literature position, and
  allowed/forbidden claims.
- `FINAL_GT_DUPLICATE_ANNOTATION_AUDIT.md` preserves the one source-level NEU
  duplicate annotation warning without altering the locked final data.

## Non-negotiable boundaries

- No post-final retraining, retuning, seed replacement, rescue analysis, or
  development substitution.
- GC10-DET and NEU-DET remain dataset-specific primary analyses; naive `n=20`
  pooling is forbidden.
- Secondary and subgroup results cannot rescue an unsupported primary claim.
- Historical evidence is discussed but is never added to the confirmatory
  sample size.

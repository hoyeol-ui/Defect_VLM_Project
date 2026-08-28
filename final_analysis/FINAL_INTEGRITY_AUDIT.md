# Final Integrity Audit

Status: **PASS**

| Check | Result |
|---|---|
| planned training cells | 440 |
| completed training cells | 440 |
| missing/corrupt/duplicate training cells | 0 / 0 / 0 |
| unresolved retries or seed replacement | 0 / 0 |
| locked-final evaluations | 440/440 |
| missing/duplicate final evaluations | 0 / 0 |
| development substitutions | 0 |
| post-final retraining or retuning | 0 |
| complete six-point learner curves | 80 |
| conceptual strategy-budget rows | 480 |
| acquisition-level primary contrasts | 20 |
| missing or interpolated primary points | 0 |
| cross-dataset pooling | false |

Authoritative machine-readable records:

- `confirmatory_v1_1/full_run/runtime/FULL_TRAINING_INTEGRITY_AUDIT.json`
- `confirmatory_v1_1/final_analysis/final_evaluation/FINAL_EVALUATION_INTEGRITY_AUDIT.json`
- `confirmatory_v1_1/final_analysis/results/CONFIRMATORY_ANALYSIS_INTEGRITY_AND_SUMMARY.json`

The training audit was created before final-evaluation authorization, so its `actual_final_authorization_present: false` and `actual_final_evaluations: 0` fields describe the end of the training phase, not a failed final evaluation. The separate final-evaluation audit records the later authorized 440/440 evaluation and is the authority for that phase.

One exact duplicated bounding-box annotation was found in NEU final image `crazing_120.xml`. Ultralytics removed the same duplicate deterministically in every relevant evaluation. The raw annotation was not altered; scope and impact are recorded in `confirmatory_v1_1/final_analysis/FINAL_GT_DUPLICATE_ANNOTATION_AUDIT.md`.

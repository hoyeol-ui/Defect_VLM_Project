# GC10 Cost-Translation Protocol Package Report

Date: 2026-07-30

Protocol version: `gc10-cost-translation-v1-candidate`

## Final Status

**`READY_FOR_RESEARCHER_DECISIONS`**

The candidate package is internally consistent and can be reviewed by the researcher and supervisor. It is **not authorized for human annotation or learner training** because nine researcher/institution decisions remain open.

## Scope Completed

- Primary acquisition seeds fixed to 5–14 by chronological rule
- q20 selection membership rebuilt from the existing 200-seed CSV
- per-seed/strategy query rank, box count, rare/multilabel/small flags generated
- pairwise within- and cross-strategy overlap generated
- image selection frequency and seed membership generated
- Phase A unique union generated
- sample-ID-only overlap-minimized sensitivity seed design generated
- Phase A-disjoint 24-image Stage 0 stress manifest generated
- blinded Stage 0 and Phase A annotator exports generated
- event logger schema and export validation specification written
- Hungarian quality protocol and metric schema written
- overlap-aware statistical analysis plan written
- threshold decision matrix and human-participation documents written
- package validator implemented and executed

No annotation UI, human annotation, learner training, threshold selection, Git commit/tag, or official/final learner evaluation was performed.

## Generated Files

### Protocol package

- `protocols/gc10_cost_translation_v1/stage0_instrumentation_manifest.csv`
- `protocols/gc10_cost_translation_v1/phase_a_unique_image_manifest.csv`
- `protocols/gc10_cost_translation_v1/phase_a_seed_batch_manifest.csv`
- `protocols/gc10_cost_translation_v1/phase_b_frozen_candidate_manifest_template.csv`
- `protocols/gc10_cost_translation_v1/acquisition_overlap_matrix.csv`
- `protocols/gc10_cost_translation_v1/image_seed_membership.csv`
- `protocols/gc10_cost_translation_v1/overlap_minimized_seed_sensitivity.csv`
- `protocols/gc10_cost_translation_v1/stage0_annotator_blinded_manifest.csv`
- `protocols/gc10_cost_translation_v1/phase_a_annotator_blinded_manifest.csv`
- `protocols/gc10_cost_translation_v1/manifest_hashes.sha256`
- `protocols/gc10_cost_translation_v1/source_evidence_hashes.sha256`
- `protocols/gc10_cost_translation_v1/manifest_build_metadata.json`
- `protocols/gc10_cost_translation_v1/annotation_event_schema.json`
- `protocols/gc10_cost_translation_v1/annotation_event_examples.jsonl`
- `protocols/gc10_cost_translation_v1/annotation_logger_requirements.md`
- `protocols/gc10_cost_translation_v1/annotation_export_validation_spec.md`
- `protocols/gc10_cost_translation_v1/annotation_quality_protocol.md`
- `protocols/gc10_cost_translation_v1/annotation_quality_metric_schema.json`
- `protocols/gc10_cost_translation_v1/statistical_analysis_plan.md`
- `protocols/gc10_cost_translation_v1/statistical_analysis_config.template.yaml`
- `protocols/gc10_cost_translation_v1/unresolved_researcher_decisions.md`
- `protocols/gc10_cost_translation_v1/threshold_option_matrix.md`
- `protocols/gc10_cost_translation_v1/human_participation_checklist.md`
- `protocols/gc10_cost_translation_v1/annotator_information_sheet_draft.md`
- `protocols/gc10_cost_translation_v1/annotation_guide_outline.md`
- `protocols/gc10_cost_translation_v1/package_validation_report.json`

### Builder and validator

- `scripts/08_gc10_cost_translation/build_protocol_manifests.py`
- `scripts/08_gc10_cost_translation/validate_protocol_package.py`

### Documents

- `docs/gc10_cost_translation_protocol_v1_candidate_20260730.md`
- `docs/gc10_cost_translation_protocol_package_report_20260730.md`
- `docs/gc10_cost_translation_easy_value_explanation_20260730.md`

## Manifest Results

| Item | Result |
|---|---:|
| Stage 0 images | 24 |
| Phase A unique images | 233 |
| Phase A seed-batch rows | 400 |
| Primary acquisition seeds | 10 |
| Seed IDs | 5–14 |
| Strategies | 2 |
| Query ranks per seed/strategy | 1–20 |
| Stage 0–Phase A overlap | 0 |
| Duplicate blinded task IDs | 0 |
| Non-acquisition rows | 0 |
| Missing image paths | 0 |

Primary seed-set unique counts:

- Random: 187/200 slots
- DINO: 51/200 slots
- two-arm unique union: 233
- cross-strategy unique overlap: 5

Appendix overlap-minimized sensitivity seeds:

`5, 198, 95, 100, 15, 75, 24, 37, 55, 87`

This set is not authorized to replace primary seeds.

## Overlap-Aware Analysis

Implementation status: **FEASIBLE**

The necessary mapping exists in:

- `phase_a_seed_batch_manifest.csv`
- `phase_a_unique_image_manifest.csv`
- `image_seed_membership.csv`
- `acquisition_overlap_matrix.csv`

The primary Phase A interval is designed as an image-and-annotator cluster bootstrap. A shared image weight is propagated to every seed/strategy membership of that image, then each batch is normalized to the q20 scale. This retains the dependence caused by DINO's repeated image selection.

The seed-level paired bootstrap is secondary because it treats the ten seed contrasts as exchangeable and does not model repeated images.

## Validation Result

`validate_protocol_package.py` passed all 21 checks:

- required files and columns
- Stage 0/Phase A disjointness
- seed 5–14 cardinality
- complete q20 ranks
- blinded-field leakage
- image path existence
- acquisition-only split
- manifest/source hash match
- event schema fields and event enum
- event-example JSON, field coverage, and checksum validity
- quality metric schema
- protocol-version consistency

Machine-readable result:

`protocols/gc10_cost_translation_v1/package_validation_report.json`

## Unresolved Decisions

Count: **9**

1. annotator count: 2 or 4
2. primary time budget \(T\)
3. idle threshold
4. Phase A burden materiality margin
5. Phase A quality non-inferiority margin
6. Phase B minimum mAP50–95 effect
7. rare AP/recall and class 8/9/10 safety guards
8. disagreement/adjudication rule
9. institutional approval, consent, compensation, and retention

Until these are signed, `statistical_analysis_config.template.yaml` remains an incomplete template.

## Human Participation

Formal institutional confirmation is required. The package does not decide whether the work requires full review, exemption, or a not-human-subjects determination. Recruitment, annotation timing, or event collection must not begin before the institution's written decision and the approved information/consent process.

Human annotation status: **BLOCKED PENDING D09**

## Official/Final Lock

- Package manifests contain 0 non-acquisition rows.
- Builder and validator contain no code path that opens the final manifest.
- Phase B remains development-only.
- No final image or final learner metric was used in manifest construction.

Process transparency note: during preliminary filesystem inspection, a shell preview displayed the header and first record of the existing `gc10_final_test_locked.csv`. That row was not used in any selection, manifest, statistic, threshold, or learner evaluation, and the permanent builder/validator never reads that file. This should be recorded as a process-boundary inspection incident and acknowledged by the researcher/supervisor before formal freeze.

## Commit Decision

- Candidate review commit: **permissible after researcher review**
- Frozen execution commit/tag: **not yet permissible**
- Human annotation: **not yet permissible**
- Phase B learner: **not authorized**

The candidate may be committed as an explicitly unresolved review artifact. It must not be tagged or described as the frozen execution protocol until D01–D09, Stage 0, UI validation, and the final package validator have all passed.

## Final Decision

**`READY_FOR_RESEARCHER_DECISIONS`**

This status means the package is structurally ready for the researcher and supervisor to make and document the remaining choices. It does not mean the human study is ready to run.

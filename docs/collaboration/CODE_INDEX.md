# Code Index

## Status vocabulary

- **CANONICAL** — authoritative for the final controlled study or its public result audit. Do not silently edit; create a new protocol/version.
- **HISTORICAL-RELEVANT** — needed to audit how the thesis question developed, but not authoritative for the final 440 result.
- **SUPPORTING** — utility, audit, test, or documentation-generation code that supports an authoritative artifact.
- **DEPRECATED** — superseded implementation retained for provenance; do not use for new results.
- **SCRATCH** — exploratory/local experiment code without a frozen evidential role.

Status applies to the role in this thesis, not general software quality. Unlisted legacy Python should be assumed **SCRATCH or UNCLASSIFIED**, never silently promoted to canonical.

## Final controlled study — canonical

| Path | Purpose / thesis role | Input → output | Main entry points | Status / related evidence |
|---|---|---|---|---|
| `confirmatory_v1/confirmatory_protocol_v1.yaml` | Normative scientific design: data curation, split, seeds, budgets, selectors, YOLO, endpoints, statistics, leakage and failure rules. | audited source registry → frozen specification | declarative YAML | **CANONICAL**; all 440 controlled results |
| `confirmatory_v1_1/confirmatory_protocol_v1_1.yaml` | v1.1 locked-final source delta and execution authority. | parent v1 protocol → final evaluation policy | declarative YAML | **CANONICAL** with v1 parent |
| `confirmatory_v1/src/build_split_and_seed_freeze.py` | Builds deterministic pixel-identity-aware curation, 70/10/20 split, seed manifest, and blind acquisition universes. | local datasets → `data_freeze/*` | command-line `main` and split builders | **CANONICAL**; dataset/split freeze |
| `confirmatory_v1/src/confirmatory_v1/protocol_core.py` | Random selector, seed derivation, budgets, seals, sample identity, and GT-free manifest rules. | protocol + blind IDs → deterministic selections/hashes | selector/core functions | **CANONICAL**; Random and shared R0 |
| `confirmatory_v1/src/confirmatory_v1/frozen_dino_ff_v1.py` | Exact cleaned DINOv2 preprocessing, cache checks, cosine farthest-first acquisition, deterministic ties. | images/cache + current labeled IDs → query IDs | cache builder/loader and farthest-first selector | **CANONICAL**; FrozenDINO-FF-v1 |
| `confirmatory_v1_1/full_run/src/build_full_registry.py` | Materializes the 440-cell registry and reuse relationships. | protocol, seeds, manifest seals → `planned_cells.jsonl` and seal | `main` | **CANONICAL**; 440 derivation |
| `confirmatory_v1_1/full_run/src/prepare_full_yolo_data.py` | Creates YOLO datasets from sealed cumulative manifests without changing membership. | cell + split/manifests + local images/labels → YOLO data tree/YAML | `main` | **CANONICAL**; training input materialization |
| `confirmatory_v1_1/full_run/src/run_full_cell.py` | Executes one frozen cold-start 100-epoch YOLO cell and records operational/private artifacts. | registry cell + data + sealed base weight → best/last checkpoints and logs | `main` | **CANONICAL**; one detector job |
| `confirmatory_v1_1/full_run/src/run_full_440_orchestrator.py` | Resumable, idempotent sequential orchestrator with lock/status/audit closure. | 440 registry → completed cell tree and training audit | `main` | **CANONICAL**; full execution |
| `confirmatory_v1_1/final_analysis/src/authorize_final_evaluation.py` | Verifies preconditions and seals the external final-evaluation authorization. | freeze and integrity records → authorization record | `main` | **CANONICAL**; final lock |
| `confirmatory_v1_1/final_analysis/src/run_final_evaluation.py` | Evaluates every development-selected `best.pt` on the locked final split under one fixed configuration. | 440 checkpoints + locked manifests → raw final metrics/audit | `main` | **CANONICAL**; locked outcome generation; site-local paths must be configured for reproduction |
| `confirmatory_v1_1/final_analysis/src/final_common.py` | Shared parsing, hashing, metric, and path helpers for final analysis. | registries/tables → normalized records | helper functions | **CANONICAL** |
| `confirmatory_v1_1/final_analysis/src/analyze_final_results.py` | Constructs 480 conceptual rows, 80 curves, 20 paired effects, bootstrap/sign-flip/Holm, secondary and composition tables. | raw final metrics + manifests → `results/*.csv/json` | `main`, AULC/statistical helpers | **CANONICAL**; primary/secondary result |
| `confirmatory_v1_1/final_analysis/src/development_final_diagnostic.py` | Supplementary development-versus-final comparison after locked analysis; never substitutes development outcomes. | development + final summaries → diagnostic tables/audit | `main` | **SUPPORTING**; robustness diagnostic |
| `confirmatory_v1_1/final_analysis/src/plot_final_results.py` | Generates figures only from canonical final tables. | result CSVs → PNG/PDF/SVG | `main` | **CANONICAL** for released figures |
| `confirmatory_v1_1/final_analysis/src/build_final_report.py` | Generates the detailed controlled report from canonical outputs. | CSV/JSON → Markdown report | `main` | **SUPPORTING** |
| `confirmatory_v1_1/final_analysis/src/build_raw_confirmatory_master.py` | Preserves result-level raw/acquisition evidence in a readable master. | canonical tables → raw master Markdown | `main` | **SUPPORTING** |
| `scripts/verify_public_results.py` | CPU-only public audit of released hashes, counts, paired means, exact sign-flip, and Holm values. | `results/thesis_final/tables` → PASS/FAIL | `main` | **CANONICAL** for public table verification; does not retrain |

## Canonical manifests, tests, and audit artifacts

| Path | Role | Status |
|---|---|---|
| `confirmatory_v1/data_freeze/` | Pixel-identity hashes, frozen split assignments, blind universes, seeds, and class support. No image bytes. | **CANONICAL data index** |
| `confirmatory_v1_1/manifest_seals/` | Cumulative/query sample-ID lists for every dataset/acquisition/strategy/round. | **CANONICAL selection evidence** |
| `confirmatory_v1_1/full_run/planned_cells.jsonl` and `planned_registry_seal.json` | Exact planned jobs and registry hash. | **CANONICAL execution evidence** |
| `confirmatory_v1/tests/` and `confirmatory_v1_1/tests/test_v1_1_preflight.py` | Protocol, leakage, determinism, lifecycle, and preflight checks. | **SUPPORTING tests** |
| `confirmatory_v1_1/full_run/runtime/FULL_TRAINING_INTEGRITY_AUDIT.json` | 440 training closure. | **CANONICAL audit** |
| `confirmatory_v1_1/final_analysis/final_evaluation/FINAL_EVALUATION_INTEGRITY_AUDIT.json` | 440 final-evaluation closure. | **CANONICAL audit** |
| `confirmatory_v1_1/final_analysis/results/CONFIRMATORY_ANALYSIS_INTEGRITY_AND_SUMMARY.json` | Analysis row counts and table hashes. | **CANONICAL audit** |

## Historical evidence code — relevant, not final-controlled

| Path / family | Purpose | Typical input → output | Thesis role / result | Status |
|---|---|---|---|---|
| `scripts/01_score_generation/generate_structured_vlm_validity_responses.py`, `score_structured_vlm_validity.py` | Structured VLM candidate-validity probing. | images/prompts/responses → validity scores | Early VLM signal audit | **HISTORICAL-RELEVANT** |
| `scripts/03_analysis/audit_legacy_vlm_consistency_validity.py` | Tests whether legacy consistency/error scores predict severe errors. | historical VLM outputs → correlations/AUROC | Weak/contrary validity evidence | **HISTORICAL-RELEVANT** |
| `scripts/01_score_generation/compute_pseudo_groundedness.py` | Computes GT-free pseudo-groundedness from localized model outputs. | structured responses/regions → scores | Groundedness stage | **HISTORICAL-RELEVANT** |
| `scripts/grounded_prompt_experiment.py` | Prompt-grounding experiment runner. | images/prompts → VLM outputs | Early exploratory method | **HISTORICAL-RELEVANT**, not final selector |
| `scripts/03_analysis/prepare_oracle_crop_vlm_diagnostic.py`, `audit_oracle_crop_vlm_diagnostic.py` | Oracle-crop diagnostic of semantic/localization bottlenecks. | GT-derived diagnostic crops + VLM → audit | Diagnostic only; not GT-free acquisition | **HISTORICAL-RELEVANT** |
| `scripts/01_preprocessing/build_gc10_dinov2_embeddings.py` | Historical GC10 DINO cache. | GC10 images → embeddings | DINO diversity lineage | **HISTORICAL-RELEVANT**; superseded for controlled use |
| `scripts/01_preprocessing/build_mpdd_dinov2_embeddings.py` | MPDD embedding cache. | MPDD images → embeddings | MPDD discovery/composition | **HISTORICAL-RELEVANT** |
| `scripts/01_preprocessing/build_visa_dinov2_embeddings.py` | VisA embedding cache. | VisA images → embeddings | VisA discovery/category collapse | **HISTORICAL-RELEVANT** |
| `scripts/03_analysis/run_mpdd_selection_only_audit.py` | Paired Random/DINO MPDD selection audit. | fixed pool/embeddings → discovery and composition | anomaly +6.245, category −0.325 | **HISTORICAL-RELEVANT** |
| `scripts/03_analysis/run_visa_selection_only_audit.py` | Paired VisA selection audit. | fixed pool/embeddings → anomaly/category metrics | anomaly +14.480, categories −4.110 | **HISTORICAL-RELEVANT** |
| `scripts/03_analysis/run_gc10_discovery_representation_audit.py`, `run_gc10_taxonomy_selection_audit.py` | GC10 rare discovery and taxonomy/composition audits. | GC10 records/embeddings → selection summaries | Discovery versus rare/overall translation | **HISTORICAL-RELEVANT** |
| `scripts/02_active_learning/run_gc10_development_detector_confirmation.py` | Historical GC10 detector confirmation. | historical selected sets → detector metrics | Overall/rare conflict | **HISTORICAL-RELEVANT** |
| `scripts/03_analysis/audit_random_baseline_v10c24.py` | Re-evaluates Random strength against candidate selectors. | historical runs → paired baseline audit | Motivated selector-search stop | **HISTORICAL-RELEVANT** |
| `scripts/06_ksdd2_prospective/` | Frozen KSDD2 prospective anomaly protocol and PatchCore analysis. | KSDD2 → gate/audit | Prospective gate evidence | **HISTORICAL-RELEVANT** |
| `scripts/07_ksdd2_g5_bounded_learner/` and `scripts/08_ksdd2_g5_posthoc/` | Bounded U-Net learner and posthoc integrity. | frozen KSDD2 manifests → Dice/gate | +0.037565 below +0.05 | **HISTORICAL-RELEVANT** |
| `scripts/05_deeppcb_prospective/` | Prospective reference-residual gate and STOP audit. | DeepPCB pairs → enrichment/mechanism decision | enrichment 1.107693 < 1.20; STOP | **HISTORICAL-RELEVANT** |
| `scripts/04_dcal_xai/build_evidence_freeze_v2.py`, `scripts/05_deeppcb_prospective/build_evidence_freeze_v3.py` | Build immutable historical evidence ledgers. | branch outputs → ledgers/matrices | Preserves null/negative decisions | **SUPPORTING** |

## Deprecated and scratch boundaries

| Path / family | Status | Reason / replacement |
|---|---|---|
| `scripts/old/` | **DEPRECATED** | Earliest monolithic VLM experiments; preserved for provenance, not reproducibility of final claims. |
| `scripts/02_active_learning/launch_v8_neu_only_seed42.py` and related v8/v9/v10 probes | **HISTORICAL-RELEVANT or DEPRECATED by experiment** | Useful for lineage; superseded by the frozen controlled selector/lifecycle. Consult historical decision document before use. |
| repository-root `main.py`, `test_0329.py`, `test_model.py` | **SCRATCH** | Development entry points without final-study authority. |
| `실행결과 0511.txt` | **SCRATCH/legacy record** | Unstructured early output; not a canonical evidence table. |
| `docs/results/v7_*`, `v10b_*`, old `runs/*` | **HISTORICAL-RELEVANT artifacts** | May be authoritative only for their named historical branch; never for final 440. |
| `package/` and `notion_import_package_20260808.zip` | **DEPRECATED manuscript snapshot** | Pre-final thesis packaging retained unchanged; final controlled results supersede its current-state narrative. |

## Practical rule for collaborators

For a final number, begin in `results/thesis_final/` and trace to `confirmatory_v1_1/final_analysis/`. For a final method, begin in the two protocol YAMLs and the canonical code table. For a historical claim, require a named decision/evidence ledger and do not import its code or number into the controlled analysis without a new protocol.

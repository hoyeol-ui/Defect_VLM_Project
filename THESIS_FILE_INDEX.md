# 연구 결과 파일 안내

## Start Here — completed development analysis, 2026-09-13

1. [README](README.md) — 현재 질문·완료 상태·대표 결과.
2. [실험 결과 기록](results/development_20260913/EXPERIMENT_RESULT_BRIEF.md).
3. [그림 5종 PNG/SVG/PDF](results/development_20260913/figures/).
4. [Seed/budget/selection 원표](results/development_20260913/tables/).
5. [분석 검증 보고](results/development_20260913/ANALYSIS_VALIDATION_REPORT.md).
6. [Historical locked-final](results/thesis_final/README.md) — 신규 development와 별도 근거.
7. [Controlled protocol](docs/collaboration/CONTROLLED_440_PROTOCOL.md) / [Entropy protocol](entropy_extension_v1/official_release_v1/protocol.json).
8. [Public reproduction](scripts/analysis/development_20260913/README.md) / [Public release audit](results/development_20260913/audit/PUBLIC_RELEASE_AUDIT.md).

새 locked-final: NOT EXECUTED / PENDING SEPARATE FREEZE. 추가 AL training 종료와 최종평가 완료를 구분한다.

## Historical / Previous-stage navigation

아래는 기존 단계의 파일 안내다. 날짜 당시의 완료 수와 evidence 범위를 현재 상태로 오인하지 않는다.

### Previous file index

## A. Start here

- `README.md` — full project map and final result.
- `docs/collaboration/COLLABORATOR_BRIEF.md` — send this to a new collaborator.
- `docs/collaboration/RESEARCH_HISTORY.md` — hypothesis evolution and branch closures.

## B. Final result and integrity

- `final_analysis/FINAL_RESULT_BRIEF.md` — 1–3 page advisor summary.
- `final_analysis/FINAL_CONTROLLED_ANALYSIS.md` — integrated scientific interpretation.
- `final_analysis/FINAL_INTEGRITY_AUDIT.md` — phase/cardinality closure.
- `confirmatory_v1_1/final_analysis/CONFIRMATORY_FINAL_ANALYSIS_REPORT.md` — generated detailed report.
- `confirmatory_v1_1/final_analysis/RAW_CONFIRMATORY_RESULTS_MASTER_FINAL_20260828.md` — result-level readable master.
- `confirmatory_v1_1/final_analysis/FINAL_EXPERIMENT_ACADEMIC_EVALUATION_20260828.md` — independent academic assessment.

## C. Public tables and figures

- `results/thesis_final/README.md` — raw-to-summary map.
- `results/thesis_final/tables/locked_final_conceptual_480.csv` — full conceptual rows.
- `results/thesis_final/tables/curve_level_aulc.csv` — 80 learner curves.
- `results/thesis_final/tables/acquisition_paired_effects.csv` — 20 primary paired effects.
- `results/thesis_final/tables/locked_final_primary.csv` — final two-row primary summary.
- `results/thesis_final/tables/secondary_subgroup.csv` and `classwise_final.csv` — safety/subgroups.
- `results/thesis_final/figures/` — primary and classwise PNG/PDF/SVG.

## D. Controlled experiment

- `docs/collaboration/CONTROLLED_440_PROTOCOL.md` — readable protocol and 440 derivation.
- `confirmatory_v1/confirmatory_protocol_v1.yaml` — normative base freeze.
- `confirmatory_v1_1/confirmatory_protocol_v1_1.yaml` — locked-final v1.1 delta.
- `confirmatory_v1/src/build_split_and_seed_freeze.py` — split/seed builder.
- `confirmatory_v1/src/confirmatory_v1/` — canonical selectors/core.
- `confirmatory_v1_1/full_run/src/` — registry, data materialization, cell, orchestrator.
- `confirmatory_v1_1/manifest_seals/` — selection membership evidence.
- `confirmatory_v1_1/final_analysis/src/` — evaluation, analysis, reporting, figures.

## E. Historical evidence

- `runs/evidence_freeze_v3_20260718/research_evidence_ledger.csv` — historical evidence ledger.
- `docs/condition_map_master_20260718.csv` — comparable condition/result map.
- `docs/hypothesis_transition_matrix_v3_20260718.csv` — hypothesis changes.
- `docs/research_evolution_and_evidence_freeze_v3_20260718.md` — historical synthesis.
- `scripts/01_score_generation/`, `scripts/02_active_learning/`, `scripts/03_analysis/` — VLM/grounding/AL/audit lineage.
- `scripts/05_deeppcb_prospective/`, `06_ksdd2_prospective/`, `07_ksdd2_g5_bounded_learner/` — prospective STOP/gate branches.

Use `docs/collaboration/CODE_INDEX.md` to determine authority before executing historical code.

## F. Data and attribution

- `DATASETS.md` — upstream facts, exact project counts/use, license and redistribution status.
- `CITATIONS_DATASETS.bib` — verified dataset citations and explicit gaps.
- `docs/collaboration/DATASET_REDISTRIBUTION_AUDIT.md` — public raw-data audit.
- `docs/collaboration/MODEL_SOFTWARE_ATTRIBUTION.md` — external model/library records.

## G. Reproducibility and public release

- `docs/collaboration/REPRODUCIBILITY.md` — public audit and full reproduction levels.
- `requirements-thesis.txt` — observed key environment versions.
- `scripts/verify_public_results.py` — CPU-only table verifier.
- `docs/collaboration/CLAIM_EVIDENCE_MATRIX.md` — manuscript wording controls.
- `docs/collaboration/PUBLIC_RELEASE_AUDIT.md` — security/size/copyright audit.
- `docs/collaboration/FIRST_IMPRESSION_TEST.md` — 15-question onboarding check.

## H. Thesis/manuscript snapshots

- `package/manuscript/` and `notion_import_package_20260808.zip` — **deprecated pre-final manuscript snapshots**, retained for provenance.
- `docs/mini_paper_*` — historical mini-paper drafts, not final-result authority.
- `docs/figures/` — historical framework figures.

## I. Deprecated and scratch

- `scripts/old/` — deprecated earliest experiments.
- root `main.py`, `test_0329.py`, `test_model.py`, and `실행결과 0511.txt` — scratch/legacy, not canonical.
- old named run directories are authoritative only for their own historical branch.

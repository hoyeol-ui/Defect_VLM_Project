# Research History

## Reading rule

This chronology reconstructs why the final question changed. Historical experiments have different tasks, pools, budgets, endpoints, and evidential status. Their numerical results must not be pooled with the final controlled study.

## 1. VLM consistency and candidate validity

The project initially treated VLM semantic consistency/error signals as potential annotation-utility proxies. Structured validity and groundedness audits challenged that assumption: one canonical validity audit recorded Spearman `−0.1812` and severe-error AUROC `0.3734`. The lesson was not “VLMs are useless”; it was that an intuitive score needs a direct validity check before it can govern acquisition.

Relevant code: `scripts/01_score_generation/`, `scripts/03_analysis/audit_legacy_vlm_consistency_validity.py`, `audit_structured_vlm_gc10_groundedness.py`, and the paired model-gate scripts.

## 2. Groundedness and GT-free pseudo grounding

The next phase tried to make semantic signals spatially accountable using structured responses, pseudo-groundedness, oracle-crop diagnostics, and OWL-ViT. These experiments separated “the model says the image is relevant” from “the response is localized to a plausible defect region.” They improved diagnostics but did not by themselves establish downstream annotation utility.

Relevant code: `scripts/01_score_generation/compute_pseudo_groundedness.py`, `scripts/grounded_prompt_experiment.py`, `scripts/03_analysis/prepare_oracle_crop_vlm_diagnostic.py`, and `audit_oracle_crop_vlm_diagnostic.py`.

## 3. Detector-level and DINO diversity experiments

The project moved from selector-internal scores to detector outcomes and DINOv2 visual diversity. NEU, GC10, and hybrid/detector-aware experiments explored selection geometry, uncertainty, representation failure, and cold-start training. Random remained competitive, and detector-aware complexity did not provide a stable general solution.

Relevant code: `scripts/02_active_learning/`, `scripts/03_analysis/analyze_gc10_detector_translation_failure.py`, `analyze_gc10_d2r_representation_failure.py`, and `audit_random_baseline_v10c24.py`.

## 4. Reproducibility separation

A historical NEU fixed selected set showed mAP50-95 `+0.016236` and 5/5 positive learner seeds. Repeating acquisition itself across ten new seeds produced `+0.007019`, 95% interval `[-0.005211,+0.019678]`, 7/10 positive, exact p `0.322266`, failing its frozen gate. This established a central distinction: training stability on one selected set is not acquisition-set reproducibility.

Relevant evidence: `runs/seed45_fixed_set_stability/`, `runs/active_learning_v8_cold_start_confirmation/`, and `docs/v8_cold_start_visual_confirmation_protocol_20260714.md`.

## 5. Discovery and composition across datasets

Selection-only audits showed why target discovery cannot stand alone:

- GC10 rare-image discovery increased, yet one q20 downstream result combined overall mAP50-95 `+0.017378` with rare macro AP `−0.019877`. A later K40/140 gate recorded mAP `−0.001678`, rare AP `−0.018290`, and recall `−0.021871`.
- MPDD anomaly discovery increased by `+6.245` images per 20-image query, but product-category coverage changed `−0.325`, missing its safety margin; source composition explained a substantial part of the gain.
- VisA anomaly discovery increased `+14.480`, while category coverage changed `−4.110` and concentration increased, a category-collapse pattern.

These are selection/discovery or historical downstream case results, not final controlled effects. Relevant code is in `scripts/01_preprocessing/build_*_dinov2_embeddings.py`, `scripts/03_analysis/run_*_selection_only_audit.py`, and the decision documents for each branch.

## 6. Prospectively gated KSDD2 and DeepPCB branches

KSDD2 prospective work tested anomaly/segmentation and a bounded learner. A positive Dice estimate (`+0.037565`) remained below its frozen `+0.05` criterion, so “positive direction” and “gate PASS” were kept separate.

DeepPCB then served as an external prospective STOP case. Its primary enrichment was `1.107693`, below the frozen `1.20` threshold. The branch stopped before detector training/final-test use, preserving a negative decision rather than searching until success.

Relevant code: `scripts/06_ksdd2_prospective/`, `scripts/07_ksdd2_g5_bounded_learner/`, `scripts/08_ksdd2_g5_posthoc/`, and `scripts/05_deeppcb_prospective/`.

## 7. Selector-search stop and question refocus

At this point, additional selector invention risked outcome-guided wandering. The accumulating evidence supported a stronger methodological question: which apparent selection benefits survive validity, discovery, composition, acquisition replication, learner replication, and downstream testing? Random and a cleaned Frozen DINO candidate were chosen for a controlled two-arm study. The purpose was confidence in the evidence chain, not manufacturing novelty through 440 runs.

## 8. Prospective Random–Frozen DINO validation

The protocol fixed GC10/NEU splits, budgets, seed derivation, GT-free boundary, DINO snapshot and representation, deterministic farthest-first rule, YOLO lifecycle, final lock, AULC, and statistics before final outcomes. DINO caches used the full acquisition universes; operational smoke measured real 100-epoch jobs; then 440 training cells ran to integrity completion.

Locked-final analysis found a small positive primary AULC difference on both datasets but a repeated NEU worst-class deterioration. This final result vindicates the research refocus: the correct conclusion requires both overall learning utility and composition/class safety, not a selector leaderboard alone.

## Canonical transition

The final thesis story is:

`VLM candidate → groundedness → GT-free selection → detector translation → Random audit → discovery/composition trade-offs → reproducibility separation → prospective gates/STOP → controlled Random–DINO full curves → bounded overall gain plus subgroup failure`.

The historical master sources are the evidence ledgers and decision matrices under `docs/` and `runs/evidence_freeze_v3_20260718/`. The final controlled master is `confirmatory_v1_1/final_analysis/RAW_CONFIRMATORY_RESULTS_MASTER_FINAL_20260828.md`.

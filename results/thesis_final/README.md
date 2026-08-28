# Thesis Final Result Package

This directory is the compact public-facing copy of the canonical controlled tables and figures. It contains no raw dataset images, model weights, or private Ultralytics run directories.

## Provenance

The authoritative source package is `confirmatory_v1_1/final_analysis/`. Files here are byte-identical copies with collaborator-friendly names. Hashes are checked by `scripts/verify_public_results.py` against `tables/analysis_integrity.json`.

## Raw-to-summary trace

1. `locked_final_conceptual_480.csv` — 480 conceptual strategy/budget rows; shared Round0 appears on both trajectories.
2. `curve_level_aulc.csv` — 80 complete six-point learner curves.
3. `acquisition_paired_effects.csv` — 20 dataset/acquisition contrasts after averaging two learner seeds.
4. `locked_final_primary.csv` — two dataset-level primary summaries.

Selection composition follows `selection_composition_raw.csv` (240 states) → `selection_composition_summary.csv`. Secondary effects follow `secondary_acquisition_effects.csv` → `secondary_subgroup.csv`. `classwise_final.csv` contains final-budget class effects.

## Table index

| File | Rows | Meaning |
|---|---:|---|
| `locked_final_conceptual_480.csv` | 480 | per conceptual strategy/budget/learner locked-final metrics and hashes |
| `curve_level_aulc.csv` | 80 | one row per dataset × strategy × acquisition × learner curve |
| `acquisition_paired_effects.csv` | 20 | primary DINO-minus-Random effect per acquisition |
| `locked_final_primary.csv` | 2 | mean/median/CI/sign-flip/Holm by dataset |
| `budget_level.csv` | 12 | six rounds per dataset, paired budget-level summaries |
| `secondary_acquisition_effects.csv` | multiple endpoints × acquisitions | raw secondary paired effects |
| `secondary_subgroup.csv` | 12 | secondary/subgroup summaries |
| `classwise_final.csv` | 16 | ten GC10 and six NEU final class effects |
| `selection_composition_raw.csv` | 240 | post-seal selected-set states |
| `selection_composition_summary.csv` | summaries | paired composition effects |
| `training_seed_sensitivity.csv` | 4 | within-acquisition learner A/B difference |
| `analysis_integrity.json` | 1 | cardinalities, primary summary, canonical hashes |

## Figures

- `confirmatory_primary_results_600dpi.png` plus PDF/SVG: learning curves and acquisition-level primary effects.
- `confirmatory_classwise_results_600dpi.png` plus PDF/SVG: final class-level differences.

## Verify

```powershell
python scripts\verify_public_results.py
```

The script requires only the Python standard library.

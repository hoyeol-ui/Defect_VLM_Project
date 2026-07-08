# Curated Experiment Results

This directory is intended for lightweight, GitHub-trackable experiment summaries.

Raw YOLO training folders under `runs/` and generated score folders under `outputs/`
remain ignored because they can contain model artifacts, logs, and large intermediate
files. For lab meetings and README evidence, copy only selected CSV/PNG/MD files here.

## Recommended Result Sets

### `20260706_auxiliary_calibration_8seed/`

Source run:

```text
runs/active_learning_ablation_v3_minimal/al_ablation_v3_minimal_20260706_214805/
```

Main comparison:

- `Random`
- `CombinedSoftPenalty`
- `CombinedSuppressNoPseudo`

Interpretation:

- Random remains the strongest final detector reference.
- `CombinedSuppressNoPseudo` reduces excessive `no_pseudo_box` selection and improves or stabilizes AULC relative to naive combined scoring.
- This run diagnoses auxiliary pseudo-groundedness calibration; it does not directly test the core prompt-consistency hypothesis.

### `20260707_consistency_core_8seed/`

Source run:

```text
runs/active_learning_ablation_v3_minimal/al_ablation_v3_minimal_20260707_230706/
```

Main comparison:

- `ConsistencyOnly`
- `GroundednessOnlySoft`
- `LowPrioritySoft`

Interpretation:

- `ConsistencyOnly` is the core hypothesis test: expert prompt-family inconsistency as a GT-free acquisition signal.
- `ConsistencyOnly` is substantially stronger than `GroundednessOnlySoft`, suggesting that semantic consistency is more informative than pseudo-groundedness alone.
- `LowPrioritySoft` remains a strong direction-control result, indicating that score direction and calibration are still open issues.

## Suggested Files to Copy Per Run

```text
aggregate_strategy_metric_summary.csv
seed_strategy_metric_summary.csv
config.json
al_ablation_v3_minimal_summary.md
aggregate_learning_curve_map50.png
aggregate_learning_curve_map5095.png
final_map50_mean_std.png
final_map5095_mean_std.png
aulc_map50_mean_std.png
aulc_map5095_mean_std.png
selection_reason_distribution.png
selection_class_hint_distribution.png
selection_dataset_distribution.png
```


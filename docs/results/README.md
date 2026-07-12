# Curated Experiment Results

This directory is intended for lightweight, GitHub-trackable experiment summaries.

Raw YOLO training folders under `runs/` and generated score folders under `outputs/`
remain ignored because they can contain model artifacts, logs, and large intermediate
files. For lab meetings and README evidence, copy only selected CSV/PNG/MD files here.

## Recommended Result Sets

### `../analysis/latest_20260712/`

Source runs:

```text
runs/active_learning_ablation_v7_full_curve/v7_full_curve_20260712_053052/
runs/active_learning_ablation_v8_neu_only/v8_neu_only_20260712_105644/
```

Main comparison:

- `Random`
- `DINO Visual`
- `Consistency`

Interpretation:

- V7 mixed protocol was confounded by extreme GC10 pool skew and mixed-domain evaluation composition.
- V8 NEU-only substantially improved absolute YOLO performance for every strategy.
- `DINO Visual` was stronger than `Consistency`, especially for label-efficiency on mAP@50.
- `Random` remained the strongest or tied strongest baseline on final performance and mAP@50-95.
- Therefore, these results should be reported as failure analysis and pivot evidence, not as proof that visual diversity is a standalone winning acquisition strategy.

Curated files:

```text
docs/analysis/latest_20260712/README.md
docs/analysis/latest_20260712/*.png
docs/analysis/latest_20260712/*.csv
docs/v8_neu_only_5seed_result_log_20260712.md
docs/final_detector_aware_pivot_protocol_20260712.md
```

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


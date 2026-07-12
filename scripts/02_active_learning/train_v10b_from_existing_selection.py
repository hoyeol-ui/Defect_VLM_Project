"""
Train exactly one YOLO detector on:
    existing V10 shared initial 60
    + existing V10b selected query 30

This runner reuses completed artifacts from:
- V10 NEU large-pool smoke run
- V10b selection-only probe

It DOES NOT:
- regenerate the acquisition pool
- resample the initial set
- run Random or V9b training again
- regenerate DINO embeddings
- run detector inference for acquisition
- read or evaluate final_test_v10
- train more than one detector

The development split from the source V10 run is reused unchanged.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402
import run_al_yolo_ablation_v7_full_curve as v7  # noqa: E402
from audit_detection_pipeline_v7 import compute_file_sha256, parse_bool_env  # noqa: E402
from run_al_yolo_ablation_v7_full_curve import (  # noqa: E402
    extract_per_class_metrics,
    file_sha256,
    git_commit,
    git_dirty,
    train_eval,
)

PROJECT_ROOT = v6.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "v10b_single_training"
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "v10b_single_training"

DEFAULT_SOURCE_RUN = (
    PROJECT_ROOT
    / "runs"
    / "active_learning_ablation_v10_neu_large_pool"
    / "v10_neu_large_pool_smoke_20260712_185923"
)

DEFAULT_SELECTION_PROBE = (
    PROJECT_ROOT
    / "runs"
    / "v10b_selection_probe"
    / "v10b_selection_probe_20260712_194239"
)

V10B_STRATEGY = "DetectorUncertaintyDINOInstanceReducedV10b"


def table_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def resolve_path(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    if not value:
        return default.resolve()
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty CSV: {path}")
    return pd.read_csv(path)


def read_json_required(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def recover_initial_60(source_run: Path, acquisition_seed: int) -> pd.DataFrame:
    selected = read_csv_required(source_run / "all_selected_samples_by_round.csv")
    initial = selected[
        pd.to_numeric(selected["acquisition_seed"], errors="coerce").eq(acquisition_seed)
        & pd.to_numeric(selected["round"], errors="coerce").eq(0)
    ].copy()
    initial = initial.drop_duplicates("sample_id", keep="first").reset_index(drop=True)
    if len(initial) != 60:
        raise ValueError(f"Expected exactly 60 initial samples, got {len(initial)}")
    return initial


def recover_v10b_query(selection_probe: Path) -> pd.DataFrame:
    query = read_csv_required(selection_probe / "v10b_selected_samples.csv")
    query = query.drop_duplicates("sample_id", keep="first").reset_index(drop=True)
    if len(query) != 30:
        raise ValueError(f"Expected exactly 30 V10b query samples, got {len(query)}")
    return query


def validate_selection(
    initial: pd.DataFrame,
    query: pd.DataFrame,
    development_eval: pd.DataFrame,
) -> pd.DataFrame:
    for name, df in [
        ("initial", initial),
        ("query", query),
        ("development_eval", development_eval),
    ]:
        if "sample_id" not in df.columns:
            raise ValueError(f"{name} is missing sample_id")

    initial_ids = set(initial["sample_id"].astype(str))
    query_ids = set(query["sample_id"].astype(str))
    dev_ids = set(development_eval["sample_id"].astype(str))

    checks = pd.DataFrame(
        [
            {
                "check": "initial_size",
                "value": len(initial_ids),
                "expected": 60,
                "pass": len(initial_ids) == 60,
            },
            {
                "check": "query_size",
                "value": len(query_ids),
                "expected": 30,
                "pass": len(query_ids) == 30,
            },
            {
                "check": "cumulative_size",
                "value": len(initial_ids | query_ids),
                "expected": 90,
                "pass": len(initial_ids | query_ids) == 90,
            },
            {
                "check": "initial_query_overlap",
                "value": len(initial_ids & query_ids),
                "expected": 0,
                "pass": len(initial_ids & query_ids) == 0,
            },
            {
                "check": "train_dev_overlap",
                "value": len((initial_ids | query_ids) & dev_ids),
                "expected": 0,
                "pass": len((initial_ids | query_ids) & dev_ids) == 0,
            },
        ]
    )
    if not checks["pass"].all():
        raise ValueError(
            "Selection validation failed:\n" + checks.to_string(index=False)
        )
    return checks


def add_selection_metadata(
    df: pd.DataFrame,
    *,
    acquisition_seed: int,
    strategy: str,
    round_idx: int,
    selection_type: str,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    for col in [
        "acquisition_seed",
        "strategy",
        "round",
        "selection_type",
        "rank_in_selection",
    ]:
        if col in out.columns:
            out = out.drop(columns=[col])
    out.insert(0, "acquisition_seed", acquisition_seed)
    out.insert(1, "strategy", strategy)
    out.insert(2, "round", round_idx)
    out.insert(3, "selection_type", selection_type)
    out.insert(4, "rank_in_selection", range(1, len(out) + 1))
    return out


def write_summary(
    save_dir: Path,
    config: dict[str, Any],
    result_df: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    per_class: pd.DataFrame,
    actual_stats: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    lines = [
        "# V10b single detector training",
        "",
        "Exactly one YOLO detector was trained on existing initial-60 + V10b query-30.",
        "Final test was not read or evaluated.",
        "",
        "## Validation checks",
        "",
        table_md(checks),
        "",
        "## V10b result",
        "",
        table_md(result_df),
        "",
        "## Comparison with existing V10 baselines",
        "",
        table_md(baseline_comparison),
        "",
        "## Post-hoc actual instance statistics",
        "",
        table_md(actual_stats),
        "",
        "## Per-class development metrics",
        "",
        table_md(per_class),
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        "```",
    ]
    (save_dir / "v10b_single_training_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    source_run = resolve_path("AL_V10_SOURCE_RUN", DEFAULT_SOURCE_RUN)
    selection_probe = resolve_path(
        "AL_V10B_SELECTION_PROBE", DEFAULT_SELECTION_PROBE
    )

    source_config = read_json_required(source_run / "config.json")
    probe_config = read_json_required(selection_probe / "config.json")

    if bool(source_config.get("final_test_used", False)):
        raise RuntimeError("Source V10 run says final_test_used=True. Refusing.")
    if bool(probe_config.get("final_test_used", False)):
        raise RuntimeError("Selection probe says final_test_used=True. Refusing.")
    if bool(probe_config.get("yolo_training_run", False)):
        raise RuntimeError("Selection probe unexpectedly reports YOLO training.")

    acquisition_seed = int(
        os.environ.get(
            "AL_ACQUISITION_SEED",
            str(source_config.get("acquisition_seed", 42)),
        )
    )
    training_seed = int(
        os.environ.get(
            "AL_TRAINING_SEED",
            str(source_config.get("training_seed", 1000 + acquisition_seed)),
        )
    )

    initial = recover_initial_60(source_run, acquisition_seed)
    query = recover_v10b_query(selection_probe)
    development_eval = read_csv_required(source_run / "development_eval_v10.csv")
    checks = validate_selection(initial, query, development_eval)

    cumulative = (
        pd.concat([initial, query], ignore_index=True, sort=False)
        .drop_duplicates("sample_id", keep="first")
        .reset_index(drop=True)
    )
    if len(cumulative) != 90:
        raise ValueError(f"Expected cumulative budget 90, got {len(cumulative)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"v10b_single_training_{timestamp}"
    dataset_root = DATASETS_ROOT / save_dir.name
    save_dir.mkdir(parents=True, exist_ok=False)
    dataset_root.mkdir(parents=True, exist_ok=False)

    # Persist the exact selected manifests before any training starts.
    initial.to_csv(
        save_dir / "source_initial_60.csv", index=False, encoding="utf-8-sig"
    )
    query.to_csv(
        save_dir / "source_v10b_query_30.csv", index=False, encoding="utf-8-sig"
    )
    cumulative.to_csv(
        save_dir / "cumulative_v10b_90.csv", index=False, encoding="utf-8-sig"
    )
    checks.to_csv(
        save_dir / "pretraining_validation_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    yaml_path, build_log = v6.build_yolo_dataset(
        cumulative,
        development_eval,
        dataset_root / V10B_STRATEGY / "round_1",
    )
    build_log.insert(0, "strategy", V10B_STRATEGY)
    build_log.insert(1, "round", 1)
    build_log.to_csv(
        save_dir / "dataset_build_log.csv", index=False, encoding="utf-8-sig"
    )

    # This is the only train_eval call in this runner.
    result, runtime = train_eval(
        yaml_path,
        save_dir,
        V10B_STRATEGY,
        acquisition_seed,
        training_seed,
        1,
    )

    result_row = {
        "acquisition_seed": acquisition_seed,
        "training_seed": training_seed,
        "strategy": V10B_STRATEGY,
        "round": 1,
        "labeled_budget": 90,
        "development_eval_size": len(development_eval),
        "yaml_path": str(yaml_path),
        **result,
        "retry_count": 0,
    }
    result_df = pd.DataFrame([result_row])
    result_df.to_csv(
        save_dir / "all_round_results.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([runtime]).to_csv(
        save_dir / "runtime_profile.csv", index=False, encoding="utf-8-sig"
    )

    per_class = extract_per_class_metrics(
        result_row,
        V10B_STRATEGY,
        acquisition_seed,
        training_seed,
        1,
    )
    per_class.to_csv(
        save_dir / "per_class_metrics_by_round.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cumulative_meta = add_selection_metadata(
        cumulative,
        acquisition_seed=acquisition_seed,
        strategy=V10B_STRATEGY,
        round_idx=1,
        selection_type="cumulative_labeled",
    )
    actual_stats, actual_class = v7.actual_stats_by_round(cumulative_meta)
    actual_stats.to_csv(
        save_dir / "actual_instance_statistics_by_round.csv",
        index=False,
        encoding="utf-8-sig",
    )
    actual_class.to_csv(
        save_dir / "actual_class_distribution_by_round.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Reuse existing source results for comparison only; no baseline training.
    source_results = read_csv_required(source_run / "all_round_results.csv")
    source_view = source_results[
        [
            c
            for c in [
                "strategy",
                "round",
                "labeled_budget",
                "map50",
                "map5095",
                "precision",
                "recall",
                "train_status",
            ]
            if c in source_results.columns
        ]
    ].copy()

    v10b_view = result_df[
        [
            c
            for c in [
                "strategy",
                "round",
                "labeled_budget",
                "map50",
                "map5095",
                "precision",
                "recall",
                "train_status",
            ]
            if c in result_df.columns
        ]
    ].copy()

    baseline_comparison = pd.concat(
        [source_view, v10b_view], ignore_index=True, sort=False
    )
    baseline_comparison.to_csv(
        save_dir / "comparison_with_existing_v10_baselines.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "experiment_id": save_dir.name,
        "experiment_label": "V10b single training from frozen selection",
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "save_dir": str(save_dir),
        "dataset_root": str(dataset_root),
        "source_v10_run": str(source_run),
        "source_v10_config_sha256": compute_file_sha256(
            source_run / "config.json"
        ),
        "source_selection_probe": str(selection_probe),
        "source_v10b_selection_sha256": compute_file_sha256(
            selection_probe / "v10b_selected_samples.csv"
        ),
        "final_test_used": False,
        "final_test_read": False,
        "dino_regenerated": False,
        "detector_rescoring_run": False,
        "number_of_yolo_trainings": 1,
        "acquisition_seed": acquisition_seed,
        "training_seed": training_seed,
        "initial_size": len(initial),
        "query_size": len(query),
        "labeled_budget": len(cumulative),
        "development_eval_size": len(development_eval),
        "strategy": V10B_STRATEGY,
        "v10b_weights": probe_config.get("v10b_weights", {}),
        "constraints": probe_config.get("constraints", {}),
        "model": os.environ.get(
            "AL_YOLO_MODEL_NAME",
            str(source_config.get("model", "yolov8n.pt")),
        ),
        "epochs": int(
            os.environ.get(
                "AL_EPOCHS_PER_ROUND",
                str(source_config.get("epochs", 100)),
            )
        ),
        "patience": int(
            os.environ.get(
                "AL_YOLO_PATIENCE",
                str(source_config.get("patience", 100)),
            )
        ),
        "batch": int(
            os.environ.get(
                "AL_BATCH_SIZE",
                str(source_config.get("batch", 8)),
            )
        ),
        "workers": int(
            os.environ.get(
                "AL_WORKERS",
                str(source_config.get("workers", 4)),
            )
        ),
        "cache": os.environ.get(
            "AL_YOLO_CACHE",
            str(source_config.get("cache", "false")),
        ),
        "dry_run": parse_bool_env("AL_DRY_RUN_ONLY", False),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    (save_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    write_summary(
        save_dir,
        config,
        result_df,
        baseline_comparison,
        per_class,
        actual_stats,
        checks,
    )

    print("=" * 100)
    print("[DONE] V10b single detector training finished")
    print(f"Output dir : {save_dir}")
    print("YOLO trainings executed: 1")
    print("DINO regenerated: False")
    print("Detector rescoring executed: False")
    print("Final test used: False")
    print("=" * 100)


if __name__ == "__main__":
    main()

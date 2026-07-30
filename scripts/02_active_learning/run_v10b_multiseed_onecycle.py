"""V10b multiseed independent one-cycle validation runner.

Purpose:
    Validate the frozen V10b selector against Random on independent acquisition
    seeds 43-46.  This is intentionally *not* a full learning curve and it does
    not run V9b.

Per acquisition seed:
    1. Build an independent NEU protocol split.
    2. Sample the initial labeled set of 60 images.
    3. Train/evaluate shared Round0.
    4. Score the unlabeled pool with the Round0 detector.
    5. Select Random query 30.
    6. Select frozen V10b query 30.
    7. Train/evaluate Random budget 90.
    8. Train/evaluate V10b budget 90.
    9. Recover development-only NEU6 per-class metrics from best.pt.

Final test is saved for protocol bookkeeping only and is never evaluated.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("AL_EMBEDDING_BACKEND", "dinov2")
os.environ.setdefault("AL_ALLOW_MODEL_DOWNLOAD", "0")

import probe_v10b_selection_from_existing_v10 as v10bprobe  # noqa: E402
import probe_v9_detector_aware_selection as detector_probe  # noqa: E402
import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402
import run_al_yolo_ablation_v7_full_curve as v7  # noqa: E402
import run_v10_neu_large_pool_smoke as v10smoke  # noqa: E402
import run_v9_stage1_seed42_round1 as stage1  # noqa: E402
from audit_detection_pipeline_v7 import parse_bool_env  # noqa: E402
from canonical_sampling_v7 import sample_initial_labeled_set  # noqa: E402
from run_al_yolo_ablation_v7_full_curve import (  # noqa: E402
    file_sha256,
    git_commit,
    git_dirty,
    train_eval,
)
from ultralytics import YOLO  # noqa: E402


PROJECT_ROOT = v6.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "active_learning_v10b_multiseed_onecycle"
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "active_learning_v10b_multiseed_onecycle"

ROUND0_STRATEGY = "__SHARED_ROUND0__"
RANDOM_STRATEGY = "GTFreeRandom"
V10B_STRATEGY = "DetectorUncertaintyDINOInstanceReducedV10b"

NEU6 = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

FROZEN_V10B_WEIGHTS = {
    "detector_uncertainty": 0.25,
    "dino_visual_distance": 0.35,
    "predicted_class_deficit": 0.15,
    "pseudo_instance_count": 0.25,
}


def table_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def parse_seed_list() -> list[int]:
    raw = os.environ.get("AL_ACQUISITION_SEEDS", "43,44,45,46")
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise ValueError("AL_ACQUISITION_SEEDS is empty.")
    if 42 in seeds:
        raise ValueError("Seed42 is the development/tuning seed and is forbidden in this runner.")
    return seeds


def read_frozen_v10b_weights() -> dict[str, float]:
    weights = {
        "detector_uncertainty": float_env("AL_V10B_W_UNCERTAINTY", FROZEN_V10B_WEIGHTS["detector_uncertainty"]),
        "dino_visual_distance": float_env("AL_V10B_W_DINO", FROZEN_V10B_WEIGHTS["dino_visual_distance"]),
        "predicted_class_deficit": float_env("AL_V10B_W_BALANCE", FROZEN_V10B_WEIGHTS["predicted_class_deficit"]),
        "pseudo_instance_count": float_env("AL_V10B_W_INSTANCE", FROZEN_V10B_WEIGHTS["pseudo_instance_count"]),
    }
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Frozen V10b weights must sum to 1.0, got {total}: {weights}")
    for key, expected in FROZEN_V10B_WEIGHTS.items():
        if abs(weights[key] - expected) > 1e-9:
            raise ValueError(
                "V10b weights are frozen for this validation. "
                f"{key}={weights[key]} but expected {expected}."
            )
    return weights


def f1_score(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall) or precision + recall == 0:
        return np.nan
    return float(2 * precision * recall / (precision + recall))


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def normalize_class_name(name: Any) -> str:
    return str(name).replace("rolled-in-scale", "rolled-in_scale")


def add_f1(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "precision" in out.columns and "recall" in out.columns:
        out["f1"] = [
            f1_score(safe_float(p), safe_float(r))
            for p, r in zip(out["precision"], out["recall"])
        ]
    return out


def add_selection_metadata(
    df: pd.DataFrame,
    *,
    acquisition_seed: int,
    training_seed: int,
    strategy: str,
    round_idx: int,
    selection_type: str,
    dry_run_placeholder: bool = False,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    for col in [
        "acquisition_seed",
        "training_seed",
        "strategy",
        "round",
        "selection_type",
        "rank_in_selection",
        "dry_run_placeholder",
    ]:
        if col in out.columns:
            out = out.drop(columns=[col])
    out.insert(0, "acquisition_seed", acquisition_seed)
    out.insert(1, "training_seed", training_seed)
    out.insert(2, "strategy", strategy)
    out.insert(3, "round", round_idx)
    out.insert(4, "selection_type", selection_type)
    out.insert(5, "rank_in_selection", range(1, len(out) + 1))
    out.insert(6, "dry_run_placeholder", dry_run_placeholder)
    return out


def train_row(
    *,
    yaml_path: Path,
    save_dir: Path,
    strategy: str,
    acquisition_seed: int,
    training_seed: int,
    round_idx: int,
    labeled_budget: int,
    dev_eval_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result, runtime = train_eval(yaml_path, save_dir, strategy, acquisition_seed, training_seed, round_idx)
    row = {
        "acquisition_seed": acquisition_seed,
        "training_seed": training_seed,
        "strategy": strategy,
        "round": round_idx,
        "labeled_budget": labeled_budget,
        "development_eval_size": dev_eval_size,
        "yaml_path": str(yaml_path),
        **result,
        "retry_count": 0,
    }
    row["f1"] = f1_score(safe_float(row.get("precision")), safe_float(row.get("recall")))
    return row, runtime


def validate_split(pool_df: pd.DataFrame, dev_df: pd.DataFrame, final_df: pd.DataFrame) -> dict[str, int]:
    checks = {
        "pool_dev_overlap": int(len(set(pool_df["sample_id"]) & set(dev_df["sample_id"]))),
        "pool_final_overlap": int(len(set(pool_df["sample_id"]) & set(final_df["sample_id"]))),
        "dev_final_overlap": int(len(set(dev_df["sample_id"]) & set(final_df["sample_id"]))),
    }
    if any(v != 0 for v in checks.values()):
        raise ValueError(f"Protocol split overlap detected: {checks}")
    return checks


def validate_labeled_sets(
    initial_df: pd.DataFrame,
    random_query: pd.DataFrame,
    v10b_query: pd.DataFrame,
    dev_df: pd.DataFrame,
    *,
    dry_run: bool,
) -> pd.DataFrame:
    initial_ids = set(initial_df["sample_id"].astype(str))
    random_ids = set(random_query["sample_id"].astype(str))
    v10b_ids = set(v10b_query["sample_id"].astype(str))
    dev_ids = set(dev_df["sample_id"].astype(str))
    rows = [
        {"check": "initial_size", "value": len(initial_ids), "expected": 60, "pass": len(initial_ids) == 60},
        {"check": "random_query_size", "value": len(random_ids), "expected": 30, "pass": len(random_ids) == 30},
        {"check": "random_initial_query_overlap", "value": len(initial_ids & random_ids), "expected": 0, "pass": len(initial_ids & random_ids) == 0},
        {"check": "random_train_dev_overlap", "value": len((initial_ids | random_ids) & dev_ids), "expected": 0, "pass": len((initial_ids | random_ids) & dev_ids) == 0},
        {"check": "v10b_query_size", "value": len(v10b_ids), "expected": 30, "pass": len(v10b_ids) == 30},
        {"check": "v10b_initial_query_overlap", "value": len(initial_ids & v10b_ids), "expected": 0, "pass": len(initial_ids & v10b_ids) == 0},
        {"check": "v10b_train_dev_overlap", "value": len((initial_ids | v10b_ids) & dev_ids), "expected": 0, "pass": len((initial_ids | v10b_ids) & dev_ids) == 0},
    ]
    checks = pd.DataFrame(rows)
    if not checks["pass"].all():
        raise ValueError("Labeled-set validation failed:\n" + checks.to_string(index=False))
    if dry_run:
        checks["note"] = np.where(
            checks["check"].str.startswith("v10b_"),
            "dry-run V10b query is a structural placeholder; real V10b query requires Round0 best.pt.",
            "",
        )
    return checks


def random_v10b_overlap(random_query: pd.DataFrame, v10b_query: pd.DataFrame, seed: int) -> pd.DataFrame:
    random_ids = set(random_query["sample_id"].astype(str))
    v10b_ids = set(v10b_query["sample_id"].astype(str))
    union = random_ids | v10b_ids
    return pd.DataFrame(
        [
            {
                "acquisition_seed": seed,
                "random_size": len(random_ids),
                "v10b_size": len(v10b_ids),
                "overlap_count": len(random_ids & v10b_ids),
                "random_only_count": len(random_ids - v10b_ids),
                "v10b_only_count": len(v10b_ids - random_ids),
                "jaccard": len(random_ids & v10b_ids) / len(union) if union else np.nan,
            }
        ]
    )


def recover_per_class_metrics(
    *,
    result_row: dict[str, Any],
    strategy_label: str,
    acquisition_seed: int,
    training_seed: int,
    round_idx: int,
    save_dir: Path,
) -> pd.DataFrame:
    train_run_dir = result_row.get("train_run_dir")
    yaml_path = result_row.get("yaml_path")
    if not train_run_dir or not yaml_path or result_row.get("train_status") != "success":
        return pd.DataFrame(
            [
                {
                    "acquisition_seed": acquisition_seed,
                    "training_seed": training_seed,
                    "strategy": strategy_label,
                    "round": round_idx,
                    "status": "skipped",
                    "note": "no successful checkpoint available",
                }
            ]
        )
    weights = Path(str(train_run_dir)) / "weights" / "best.pt"
    if not weights.exists():
        return pd.DataFrame(
            [
                {
                    "acquisition_seed": acquisition_seed,
                    "training_seed": training_seed,
                    "strategy": strategy_label,
                    "round": round_idx,
                    "status": "failed",
                    "note": f"missing checkpoint: {weights}",
                }
            ]
        )

    try:
        metrics = YOLO(str(weights)).val(
            data=str(yaml_path),
            imgsz=int_env("AL_IMGSZ", 640),
            batch=int_env("AL_BATCH_SIZE", 8),
            workers=0,
            device=os.environ.get("AL_YOLO_DEVICE", "0"),
            split="val",
            verbose=False,
            plots=False,
        )
        box = metrics.box
        names = getattr(metrics, "names", {})
        if isinstance(names, dict):
            class_names = [names.get(i, str(i)) for i in range(len(names))]
        else:
            class_names = list(names)

        rows = []
        for class_id, class_name in enumerate(class_names):
            normalized = normalize_class_name(class_name)
            if normalized not in NEU6:
                continue
            rows.append(
                {
                    "acquisition_seed": acquisition_seed,
                    "training_seed": training_seed,
                    "strategy": strategy_label,
                    "round": round_idx,
                    "class_id": class_id,
                    "class_name": normalized,
                    "ap50": arr_value(getattr(box, "ap50", None), class_id),
                    "ap5095": arr_value(getattr(box, "ap", None), class_id),
                    "precision": arr_value(getattr(box, "p", None), class_id),
                    "recall": arr_value(getattr(box, "r", None), class_id),
                    "validation_instance_count": int_value(getattr(box, "nt_per_class", None), class_id),
                    "aggregate_map50_recovered": safe_float(getattr(box, "map50", np.nan)),
                    "aggregate_map5095_recovered": safe_float(getattr(box, "map", np.nan)),
                    "recorded_map50": safe_float(result_row.get("map50")),
                    "recorded_map5095": safe_float(result_row.get("map5095")),
                    "aggregate_abs_diff_map50": abs(safe_float(getattr(box, "map50", np.nan)) - safe_float(result_row.get("map50"))),
                    "aggregate_abs_diff_map5095": abs(safe_float(getattr(box, "map", np.nan)) - safe_float(result_row.get("map5095"))),
                    "status": "success",
                    "checkpoint": str(weights),
                }
            )
        return pd.DataFrame(rows)
    except Exception as exc:
        traceback.print_exc()
        return pd.DataFrame(
            [
                {
                    "acquisition_seed": acquisition_seed,
                    "training_seed": training_seed,
                    "strategy": strategy_label,
                    "round": round_idx,
                    "status": "failed",
                    "note": repr(exc),
                    "checkpoint": str(weights),
                }
            ]
        )


def arr_value(arr: Any, idx: int) -> float:
    try:
        if arr is None or idx >= len(arr):
            return np.nan
        return float(arr[idx])
    except Exception:
        return np.nan


def int_value(arr: Any, idx: int) -> int | None:
    try:
        if arr is None or idx >= len(arr):
            return None
        return int(arr[idx])
    except Exception:
        return None


def compute_selection_diagnostics(
    *,
    seed: int,
    initial_df: pd.DataFrame,
    random_query: pd.DataFrame,
    v10b_query: pd.DataFrame,
    scored_pool: pd.DataFrame,
    embedding_lookup: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selections = {
        RANDOM_STRATEGY: random_query.merge(
            scored_pool.drop_duplicates("sample_id", keep="first"),
            on="sample_id",
            how="left",
            suffixes=("", "_score"),
        ),
        V10B_STRATEGY: v10b_query,
    }
    static_scores = v10bprobe.static_component_diagnostics(scored_pool, initial_df, embedding_lookup)
    component = v10bprobe.selection_component_summary(static_scores, selections)
    geometry = v10bprobe.selection_geometry_summary(initial_df, selections, embedding_lookup)
    for df in [component, geometry]:
        if len(df):
            df.insert(0, "acquisition_seed", seed)
    return component, geometry


def bootstrap_ci(values: np.ndarray, *, n_boot: int = 5000, seed: int = 20260712) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(finite, size=len(finite), replace=True)
        means[i] = float(np.mean(sample))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_results(results_df: pd.DataFrame, per_class_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if results_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    results = add_f1(results_df)
    round1 = results[results["strategy"].isin([RANDOM_STRATEGY, V10B_STRATEGY])].copy()

    agg_rows = []
    for strategy, sub in round1.groupby("strategy", dropna=False):
        row = {"strategy": strategy, "num_seeds": int(sub["acquisition_seed"].nunique())}
        for metric in ["map50", "map5095", "precision", "recall", "f1"]:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(dtype=float)
            lo, hi = bootstrap_ci(vals) if len(vals) else (np.nan, np.nan)
            row[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        agg_rows.append(row)
    aggregate = pd.DataFrame(agg_rows)

    comparison_rows = []
    for seed, sub in round1.groupby("acquisition_seed", dropna=False):
        random = sub[sub["strategy"] == RANDOM_STRATEGY]
        v10b = sub[sub["strategy"] == V10B_STRATEGY]
        if random.empty or v10b.empty:
            continue
        r = random.iloc[0]
        v = v10b.iloc[0]
        row = {"acquisition_seed": int(seed), "training_seed": int(v["training_seed"])}
        for metric in ["map50", "map5095", "precision", "recall", "f1"]:
            row[f"random_{metric}"] = safe_float(r[metric])
            row[f"v10b_{metric}"] = safe_float(v[metric])
            row[f"v10b_minus_random_{metric}"] = safe_float(v[metric]) - safe_float(r[metric])
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)

    diff_rows = []
    for metric in ["map50", "map5095", "precision", "recall", "f1"]:
        col = f"v10b_minus_random_{metric}"
        vals = pd.to_numeric(comparison.get(col, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        lo, hi = bootstrap_ci(vals) if len(vals) else (np.nan, np.nan)
        wins = int((vals > 1e-12).sum()) if len(vals) else 0
        losses = int((vals < -1e-12).sum()) if len(vals) else 0
        ties = int((np.abs(vals) <= 1e-12).sum()) if len(vals) else 0
        std = float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan
        diff_rows.append(
            {
                "metric": metric,
                "n": int(len(vals)),
                "paired_mean_difference": float(np.mean(vals)) if len(vals) else np.nan,
                "paired_median_difference": float(np.median(vals)) if len(vals) else np.nan,
                "paired_std_difference": std,
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "paired_standardized_mean_difference": float(np.mean(vals) / std) if len(vals) >= 2 and std > 0 else np.nan,
                "note": "n=4; treat inferential statistics as descriptive only.",
            }
        )
    paired = pd.DataFrame(diff_rows)

    if not per_class_df.empty and {"strategy", "class_name", "ap5095"}.issubset(per_class_df.columns):
        class_rows = []
        for cls, sub in per_class_df.groupby("class_name", dropna=False):
            for seed, seed_sub in sub.groupby("acquisition_seed", dropna=False):
                random = seed_sub[seed_sub["strategy"] == RANDOM_STRATEGY]
                v10b = seed_sub[seed_sub["strategy"] == V10B_STRATEGY]
                if random.empty or v10b.empty:
                    continue
                class_rows.append(
                    {
                        "class_name": cls,
                        "acquisition_seed": int(seed),
                        "v10b_minus_random_ap5095": safe_float(v10b.iloc[0]["ap5095"]) - safe_float(random.iloc[0]["ap5095"]),
                    }
                )
        class_diff = pd.DataFrame(class_rows)
        if len(class_diff):
            class_summary = (
                class_diff.assign(
                    win=class_diff["v10b_minus_random_ap5095"] > 1e-12,
                    loss=class_diff["v10b_minus_random_ap5095"] < -1e-12,
                    tie=class_diff["v10b_minus_random_ap5095"].abs() <= 1e-12,
                )
                .groupby("class_name", dropna=False)
                .agg(
                    mean_v10b_minus_random_ap5095=("v10b_minus_random_ap5095", "mean"),
                    wins=("win", "sum"),
                    losses=("loss", "sum"),
                    ties=("tie", "sum"),
                )
                .reset_index()
            )
            paired = pd.concat(
                [
                    paired,
                    pd.DataFrame(
                        [
                            {
                                "metric": f"class_ap5095::{r['class_name']}",
                                "n": int(r["wins"] + r["losses"] + r["ties"]),
                                "paired_mean_difference": float(r["mean_v10b_minus_random_ap5095"]),
                                "wins": int(r["wins"]),
                                "losses": int(r["losses"]),
                                "ties": int(r["ties"]),
                                "note": "Class-wise AP50-95 V10b-Random.",
                            }
                            for _, r in class_summary.iterrows()
                        ]
                    ),
                ],
                ignore_index=True,
                sort=False,
            )

    return aggregate, comparison, paired


def write_outputs(
    *,
    save_dir: Path,
    config: dict[str, Any],
    seed_registry: list[dict[str, Any]],
    results_rows: list[dict[str, Any]],
    selected_rows: list[pd.DataFrame],
    cumulative_rows: list[pd.DataFrame],
    build_logs: list[pd.DataFrame],
    runtime_rows: list[dict[str, Any]],
    per_class_rows: list[pd.DataFrame],
    split_logs: list[pd.DataFrame],
    selection_components: list[pd.DataFrame],
    selection_geometry: list[pd.DataFrame],
    overlap_rows: list[pd.DataFrame],
) -> None:
    results_df = add_f1(pd.DataFrame(results_rows))
    selected_df = pd.concat(selected_rows, ignore_index=True, sort=False) if selected_rows else pd.DataFrame()
    cumulative_df = pd.concat(cumulative_rows, ignore_index=True, sort=False) if cumulative_rows else pd.DataFrame()
    build_df = pd.concat(build_logs, ignore_index=True, sort=False) if build_logs else pd.DataFrame()
    per_class_df = pd.concat(per_class_rows, ignore_index=True, sort=False) if per_class_rows else pd.DataFrame()
    split_df = pd.concat(split_logs, ignore_index=True, sort=False) if split_logs else pd.DataFrame()
    components_df = pd.concat(selection_components, ignore_index=True, sort=False) if selection_components else pd.DataFrame()
    geometry_df = pd.concat(selection_geometry, ignore_index=True, sort=False) if selection_geometry else pd.DataFrame()
    overlap_df = pd.concat(overlap_rows, ignore_index=True, sort=False) if overlap_rows else pd.DataFrame()

    actual_stats_df, actual_class_df = v7.actual_stats_by_round(cumulative_df) if len(cumulative_df) else (pd.DataFrame(), pd.DataFrame())
    aggregate, comparison, paired = summarize_results(results_df, per_class_df)

    (save_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(seed_registry).to_csv(save_dir / "seed_registry.csv", index=False, encoding="utf-8-sig")
    results_df.to_csv(save_dir / "all_round_results.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(save_dir / "all_selected_samples_by_round.csv", index=False, encoding="utf-8-sig")
    cumulative_df.to_csv(save_dir / "cumulative_labeled_sets_by_round.csv", index=False, encoding="utf-8-sig")
    actual_stats_df.to_csv(save_dir / "actual_instance_statistics_by_round.csv", index=False, encoding="utf-8-sig")
    actual_class_df.to_csv(save_dir / "actual_class_distribution_by_round.csv", index=False, encoding="utf-8-sig")
    per_class_df.to_csv(save_dir / "per_class_metrics_by_round.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runtime_rows).to_csv(save_dir / "runtime_profile.csv", index=False, encoding="utf-8-sig")
    build_df.to_csv(save_dir / "dataset_build_log.csv", index=False, encoding="utf-8-sig")
    split_df.to_csv(save_dir / "protocol_split_registry.csv", index=False, encoding="utf-8-sig")
    components_df.to_csv(save_dir / "selection_component_summary_by_seed.csv", index=False, encoding="utf-8-sig")
    geometry_df.to_csv(save_dir / "selection_geometry_by_seed.csv", index=False, encoding="utf-8-sig")
    overlap_df.to_csv(save_dir / "random_v10b_overlap_by_seed.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(save_dir / "seedwise_random_v10b_comparison.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(save_dir / "aggregate_random_v10b_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(save_dir / "paired_difference_summary.csv", index=False, encoding="utf-8-sig")
    write_summary(save_dir, config, pd.DataFrame(seed_registry), results_df, actual_stats_df, aggregate, comparison, paired)


def write_summary(
    save_dir: Path,
    config: dict[str, Any],
    seed_registry: pd.DataFrame,
    results_df: pd.DataFrame,
    actual_stats_df: pd.DataFrame,
    aggregate: pd.DataFrame,
    comparison: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    lines = [
        "# V10b multiseed one-cycle validation",
        "",
        "Frozen V10b vs Random on independent acquisition seeds. Final test is locked and unused.",
        "",
        "## Protocol",
        "",
        f"- Acquisition seeds: {config['acquisition_seeds']}",
        "- Seed42 is forbidden in this runner.",
        f"- Initial budget: {config['initial_seed_size']}",
        f"- Query size: {config['query_size']}",
        "- Strategies: shared Round0, GTFreeRandom, DetectorUncertaintyDINOInstanceReducedV10b",
        f"- Dry run: {config['dry_run']}",
        f"- V10b weights frozen: {config['method_weights_frozen']}",
        f"- Final test used: {config['final_test_used']}",
        "",
        "## Seed registry",
        "",
        table_md(seed_registry),
        "",
        "## Round results",
        "",
        table_md(results_df[[c for c in ["acquisition_seed", "training_seed", "strategy", "round", "labeled_budget", "map50", "map5095", "precision", "recall", "f1", "train_status", "error"] if c in results_df.columns]]),
        "",
        "## Random vs V10b seedwise comparison",
        "",
        table_md(comparison),
        "",
        "## Aggregate summary",
        "",
        table_md(aggregate),
        "",
        "## Paired difference summary",
        "",
        table_md(paired),
        "",
        "## Post-hoc cumulative XML statistics",
        "",
        table_md(actual_stats_df),
        "",
        "## Interpretation guardrails",
        "",
        "- n=4 is small; paired differences are descriptive until reviewed.",
        "- Do not tune V10b weights from these results before declaring the validation set exhausted.",
        "- Do not use final test until the method/protocol is frozen.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        "```",
    ]
    (save_dir / "v10b_multiseed_onecycle_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_one_seed(
    *,
    acquisition_seed: int,
    save_dir: Path,
    dataset_root: Path,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    accum: dict[str, list[Any]],
) -> None:
    training_seed = 1000 + acquisition_seed
    dry_run = bool(config["dry_run"])
    seed_t0 = time.perf_counter()
    seed_status = "started"
    seed_error = None

    try:
        print("=" * 100)
        print(f"[SEED {acquisition_seed}] training_seed={training_seed} dry_run={dry_run}")

        pool_df, dev_df, final_df, unused_df = v10smoke.build_v10_protocol_split(
            manifest,
            seed=acquisition_seed,
            pool_per_class=int(config["pool_per_class"]),
            dev_per_class=int(config["dev_per_class"]),
            final_per_class=int(config["final_per_class"]),
        )
        overlap_checks = validate_split(pool_df, dev_df, final_df)
        split_df = pd.DataFrame(
            [
                {
                    "acquisition_seed": acquisition_seed,
                    "split": "pool",
                    "num_images": len(pool_df),
                    **overlap_checks,
                },
                {"acquisition_seed": acquisition_seed, "split": "development_eval", "num_images": len(dev_df), **overlap_checks},
                {"acquisition_seed": acquisition_seed, "split": "final_test_LOCKED_UNUSED", "num_images": len(final_df), **overlap_checks},
                {"acquisition_seed": acquisition_seed, "split": "unused_reserve", "num_images": len(unused_df), **overlap_checks},
            ]
        )
        accum["split_logs"].append(split_df)
        pool_df.to_csv(save_dir / f"seed{acquisition_seed}_acquisition_pool_v10.csv", index=False, encoding="utf-8-sig")
        dev_df.to_csv(save_dir / f"seed{acquisition_seed}_development_eval_v10.csv", index=False, encoding="utf-8-sig")
        final_df.to_csv(save_dir / f"seed{acquisition_seed}_final_test_v10_LOCKED_UNUSED.csv", index=False, encoding="utf-8-sig")
        unused_df.to_csv(save_dir / f"seed{acquisition_seed}_unused_reserve_v10.csv", index=False, encoding="utf-8-sig")

        seed_artifact_dir = save_dir / f"seed{acquisition_seed}_artifacts"
        seed_artifact_dir.mkdir(parents=True, exist_ok=True)
        embedding_dir, embedding_lookup, embedding_config = v10smoke.load_or_build_embeddings(pool_df, seed_artifact_dir)

        initial_df = sample_initial_labeled_set(
            pool_df,
            initial_seed_size=int(config["initial_seed_size"]),
            acquisition_seed=acquisition_seed,
        ).reset_index(drop=True)
        current_pool = pool_df[~pool_df["sample_id"].isin(initial_df["sample_id"])].copy().reset_index(drop=True)
        random_query = v10smoke.select_random(current_pool, int(config["query_size"]), acquisition_seed)

        selected_frames = [
            add_selection_metadata(
                initial_df,
                acquisition_seed=acquisition_seed,
                training_seed=training_seed,
                strategy=ROUND0_STRATEGY,
                round_idx=0,
                selection_type="shared_initial_random",
            ),
            add_selection_metadata(
                random_query,
                acquisition_seed=acquisition_seed,
                training_seed=training_seed,
                strategy=RANDOM_STRATEGY,
                round_idx=1,
                selection_type=RANDOM_STRATEGY,
            ),
        ]

        cumulative_random = pd.concat([initial_df, random_query], ignore_index=True, sort=False).drop_duplicates("sample_id", keep="first")

        seed_dataset_root = dataset_root / f"seed{acquisition_seed}"
        build_logs: list[pd.DataFrame] = []

        yaml_path, build_log = v6.build_yolo_dataset(initial_df, dev_df, seed_dataset_root / ROUND0_STRATEGY)
        build_log.insert(0, "acquisition_seed", acquisition_seed)
        build_log.insert(1, "strategy", ROUND0_STRATEGY)
        build_log.insert(2, "round", 0)
        build_logs.append(build_log)
        row0, runtime0 = train_row(
            yaml_path=yaml_path,
            save_dir=save_dir,
            strategy=ROUND0_STRATEGY,
            acquisition_seed=acquisition_seed,
            training_seed=training_seed,
            round_idx=0,
            labeled_budget=len(initial_df),
            dev_eval_size=len(dev_df),
        )
        accum["results_rows"].append(row0)
        accum["runtime_rows"].append(runtime0)
        accum["per_class_rows"].append(
            recover_per_class_metrics(
                result_row=row0,
                strategy_label=ROUND0_STRATEGY,
                acquisition_seed=acquisition_seed,
                training_seed=training_seed,
                round_idx=0,
                save_dir=save_dir,
            )
        )

        round0_weight = Path(str(row0.get("train_run_dir", ""))) / "weights" / "best.pt"
        if dry_run:
            scored_pool = current_pool.copy()
            v10b_query = current_pool.sort_values("sample_id", kind="mergesort").head(int(config["query_size"])).copy().reset_index(drop=True)
            v10b_query["dry_run_selector_note"] = "placeholder; real V10b query requires Round0 detector scores"
        else:
            if row0.get("train_status") != "success" or not round0_weight.exists():
                raise FileNotFoundError(f"Round0 best.pt missing or failed for seed {acquisition_seed}: {round0_weight}")
            model = detector_probe.YOLO(str(round0_weight))
            detector_scores = detector_probe.prediction_rows(
                model,
                current_pool,
                device=str(config["device"]),
                imgsz=int(config["imgsz"]),
                conf=float(config["predict_conf"]),
                iou=float(config["predict_iou"]),
            )
            scored_pool = current_pool.merge(detector_scores, on="sample_id", how="left")
            scored_pool.to_csv(save_dir / f"seed{acquisition_seed}_round1_detector_scores.csv", index=False, encoding="utf-8-sig")
            v10b_query = stage1.select_instance_rich_dino_balanced(
                scored_pool,
                initial_df,
                embedding_lookup,
                query_size=int(config["query_size"]),
                candidate_fraction=float(config["candidate_fraction"]),
                weights=dict(config["v10b_weights"]),
                max_no_box=int(config["constraints"]["max_no_box"]),
                min_pseudo_boxes=int(config["constraints"]["min_pseudo_boxes"]),
                max_per_pred_class=int(config["constraints"]["max_per_pred_class"]),
            )
            if len(v10b_query) != int(config["query_size"]):
                raise ValueError(f"V10b selected {len(v10b_query)} samples for seed {acquisition_seed}; expected {config['query_size']}")
            component, geometry = compute_selection_diagnostics(
                seed=acquisition_seed,
                initial_df=initial_df,
                random_query=random_query,
                v10b_query=v10b_query,
                scored_pool=scored_pool,
                embedding_lookup=embedding_lookup,
            )
            accum["selection_components"].append(component)
            accum["selection_geometry"].append(geometry)

        selected_frames.append(
            add_selection_metadata(
                v10b_query,
                acquisition_seed=acquisition_seed,
                training_seed=training_seed,
                strategy=V10B_STRATEGY,
                round_idx=1,
                selection_type=V10B_STRATEGY,
                dry_run_placeholder=dry_run,
            )
        )
        checks = validate_labeled_sets(initial_df, random_query, v10b_query, dev_df, dry_run=dry_run)
        checks.insert(0, "acquisition_seed", acquisition_seed)
        checks.to_csv(save_dir / f"seed{acquisition_seed}_labeled_set_checks.csv", index=False, encoding="utf-8-sig")

        accum["overlap_rows"].append(random_v10b_overlap(random_query, v10b_query, acquisition_seed))

        cumulative_v10b = pd.concat([initial_df, v10b_query], ignore_index=True, sort=False).drop_duplicates("sample_id", keep="first")

        yaml_path, build_log = v6.build_yolo_dataset(cumulative_random, dev_df, seed_dataset_root / RANDOM_STRATEGY / "round_1")
        build_log.insert(0, "acquisition_seed", acquisition_seed)
        build_log.insert(1, "strategy", RANDOM_STRATEGY)
        build_log.insert(2, "round", 1)
        build_logs.append(build_log)
        row_random, runtime_random = train_row(
            yaml_path=yaml_path,
            save_dir=save_dir,
            strategy=RANDOM_STRATEGY,
            acquisition_seed=acquisition_seed,
            training_seed=training_seed,
            round_idx=1,
            labeled_budget=len(cumulative_random),
            dev_eval_size=len(dev_df),
        )
        accum["results_rows"].append(row_random)
        accum["runtime_rows"].append(runtime_random)
        accum["per_class_rows"].append(
            recover_per_class_metrics(
                result_row=row_random,
                strategy_label=RANDOM_STRATEGY,
                acquisition_seed=acquisition_seed,
                training_seed=training_seed,
                round_idx=1,
                save_dir=save_dir,
            )
        )

        yaml_path, build_log = v6.build_yolo_dataset(cumulative_v10b, dev_df, seed_dataset_root / V10B_STRATEGY / "round_1")
        build_log.insert(0, "acquisition_seed", acquisition_seed)
        build_log.insert(1, "strategy", V10B_STRATEGY)
        build_log.insert(2, "round", 1)
        build_logs.append(build_log)
        row_v10b, runtime_v10b = train_row(
            yaml_path=yaml_path,
            save_dir=save_dir,
            strategy=V10B_STRATEGY,
            acquisition_seed=acquisition_seed,
            training_seed=training_seed,
            round_idx=1,
            labeled_budget=len(cumulative_v10b),
            dev_eval_size=len(dev_df),
        )
        accum["results_rows"].append(row_v10b)
        accum["runtime_rows"].append(runtime_v10b)
        accum["per_class_rows"].append(
            recover_per_class_metrics(
                result_row=row_v10b,
                strategy_label=V10B_STRATEGY,
                acquisition_seed=acquisition_seed,
                training_seed=training_seed,
                round_idx=1,
                save_dir=save_dir,
            )
        )

        accum["selected_rows"].extend(selected_frames)
        accum["cumulative_rows"].extend(
            [
                add_selection_metadata(
                    initial_df,
                    acquisition_seed=acquisition_seed,
                    training_seed=training_seed,
                    strategy=ROUND0_STRATEGY,
                    round_idx=0,
                    selection_type="cumulative_labeled",
                ),
                add_selection_metadata(
                    cumulative_random,
                    acquisition_seed=acquisition_seed,
                    training_seed=training_seed,
                    strategy=RANDOM_STRATEGY,
                    round_idx=1,
                    selection_type="cumulative_labeled",
                ),
                add_selection_metadata(
                    cumulative_v10b,
                    acquisition_seed=acquisition_seed,
                    training_seed=training_seed,
                    strategy=V10B_STRATEGY,
                    round_idx=1,
                    selection_type="cumulative_labeled",
                    dry_run_placeholder=dry_run,
                ),
            ]
        )
        accum["build_logs"].extend(build_logs)
        seed_status = "success"
        seed_error = None
        accum["seed_registry"].append(
            {
                "acquisition_seed": acquisition_seed,
                "training_seed": training_seed,
                "status": seed_status,
                "error": seed_error,
                "elapsed_sec": time.perf_counter() - seed_t0,
                "embedding_dir": str(embedding_dir),
                "embedding_backend": embedding_config.get("backend"),
                "dry_run": dry_run,
                "final_test_used": False,
                "expected_yolo_trainings": 3,
                "v9b_executed": False,
            }
        )
    except Exception as exc:
        traceback.print_exc()
        seed_status = "failed"
        seed_error = repr(exc)
        accum["seed_registry"].append(
            {
                "acquisition_seed": acquisition_seed,
                "training_seed": training_seed,
                "status": seed_status,
                "error": seed_error,
                "elapsed_sec": time.perf_counter() - seed_t0,
                "dry_run": dry_run,
                "final_test_used": False,
                "expected_yolo_trainings": 3,
                "v9b_executed": False,
            }
        )
    finally:
        write_outputs(
            save_dir=save_dir,
            config=config,
            seed_registry=accum["seed_registry"],
            results_rows=accum["results_rows"],
            selected_rows=accum["selected_rows"],
            cumulative_rows=accum["cumulative_rows"],
            build_logs=accum["build_logs"],
            runtime_rows=accum["runtime_rows"],
            per_class_rows=accum["per_class_rows"],
            split_logs=accum["split_logs"],
            selection_components=accum["selection_components"],
            selection_geometry=accum["selection_geometry"],
            overlap_rows=accum["overlap_rows"],
        )
        print(f"[SEED {acquisition_seed}] status={seed_status} elapsed={time.perf_counter() - seed_t0:.1f}s")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"v10b_multiseed_onecycle_{timestamp}"
    dataset_root = DATASETS_ROOT / save_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    seeds = parse_seed_list()
    weights = read_frozen_v10b_weights()
    dry_run = parse_bool_env("AL_DRY_RUN_ONLY", True)
    manifest = v10smoke.ensure_neu_manifest()

    config: dict[str, Any] = {
        "experiment_id": save_dir.name,
        "experiment_label": "V10b frozen multiseed one-cycle validation",
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "save_dir": str(save_dir),
        "dataset_root": str(dataset_root),
        "acquisition_seeds": seeds,
        "training_seed_rule": "training_seed = 1000 + acquisition_seed",
        "method_weights_frozen": True,
        "seed42_forbidden": True,
        "v9b_executed": False,
        "full_curve": False,
        "final_test_used": False,
        "neu_manifest_size": int(len(manifest)),
        "pool_per_class": int_env("AL_V10_POOL_PER_CLASS", 150),
        "dev_per_class": int_env("AL_V10_DEV_PER_CLASS", 50),
        "final_per_class": int_env("AL_V10_FINAL_PER_CLASS", 50),
        "initial_seed_size": int_env("AL_INITIAL_SEED_SIZE", 60),
        "query_size": int_env("AL_QUERY_SIZE", 30),
        "rounds": 1,
        "strategies": [RANDOM_STRATEGY, V10B_STRATEGY],
        "expected_trainings_per_seed": 3,
        "expected_total_trainings": 3 * len(seeds),
        "v10b_weights": weights,
        "constraints": {
            "max_no_box": int_env("AL_V9B_MAX_NO_BOX", 0),
            "min_pseudo_boxes": int_env("AL_V9B_MIN_PSEUDO_BOXES", 2),
            "max_per_pred_class": int_env("AL_V9B_MAX_PER_PRED_CLASS", 2),
        },
        "candidate_fraction": float_env("AL_V9_CANDIDATE_FRACTION", 1.0),
        "predict_conf": float_env("AL_V9_PREDICT_CONF", 0.05),
        "predict_iou": float_env("AL_V9_PREDICT_IOU", 0.70),
        "model": os.environ.get("AL_YOLO_MODEL_NAME", "yolov8n.pt"),
        "epochs": int_env("AL_EPOCHS_PER_ROUND", 100),
        "patience": int_env("AL_YOLO_PATIENCE", int_env("AL_EPOCHS_PER_ROUND", 100)),
        "imgsz": int_env("AL_IMGSZ", 640),
        "batch": int_env("AL_BATCH_SIZE", 8),
        "workers": int_env("AL_WORKERS", 4),
        "cache": os.environ.get("AL_YOLO_CACHE", "false"),
        "plots": os.environ.get("AL_YOLO_PLOTS", "false"),
        "device": os.environ.get("AL_YOLO_DEVICE", "0"),
        "embedding_backend": os.environ.get("AL_EMBEDDING_BACKEND", "dinov2"),
        "allow_model_download": parse_bool_env("AL_ALLOW_MODEL_DOWNLOAD", False),
        "dry_run": dry_run,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }

    accum: dict[str, list[Any]] = {
        "seed_registry": [],
        "results_rows": [],
        "selected_rows": [],
        "cumulative_rows": [],
        "build_logs": [],
        "runtime_rows": [],
        "per_class_rows": [],
        "split_logs": [],
        "selection_components": [],
        "selection_geometry": [],
        "overlap_rows": [],
    }

    print("=" * 100)
    print("[V10b multiseed one-cycle validation]")
    print(f"Output dir: {save_dir}")
    print(f"Seeds     : {seeds}")
    print(f"Dry run   : {dry_run}")
    print("V9b       : excluded")
    print("Final test: locked / not evaluated")
    print("=" * 100)

    for seed in seeds:
        run_one_seed(
            acquisition_seed=seed,
            save_dir=save_dir,
            dataset_root=dataset_root,
            manifest=manifest,
            config=config,
            accum=accum,
        )

    seed_registry = pd.DataFrame(accum["seed_registry"])
    comparison_path = save_dir / "seedwise_random_v10b_comparison.csv"
    paired_path = save_dir / "paired_difference_summary.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() and comparison_path.stat().st_size else pd.DataFrame()
    paired = pd.read_csv(paired_path) if paired_path.exists() and paired_path.stat().st_size else pd.DataFrame()

    completed = seed_registry[seed_registry["status"].astype(str).eq("success")]["acquisition_seed"].tolist() if len(seed_registry) else []
    failed = seed_registry[~seed_registry["status"].astype(str).eq("success")]["acquisition_seed"].tolist() if len(seed_registry) else []

    map5095_diff = paired[paired["metric"].astype(str).eq("map5095")] if len(paired) and "metric" in paired.columns else pd.DataFrame()
    precision_diff = paired[paired["metric"].astype(str).eq("precision")] if len(paired) and "metric" in paired.columns else pd.DataFrame()
    recall_diff = paired[paired["metric"].astype(str).eq("recall")] if len(paired) and "metric" in paired.columns else pd.DataFrame()

    print("=" * 100)
    print("[DONE] V10b multiseed one-cycle runner finished")
    print(f"Output dir: {save_dir}")
    print(f"Completed seeds: {completed}")
    print(f"Failed seeds   : {failed}")
    if len(map5095_diff):
        row = map5095_diff.iloc[0]
        print(f"Paired mean mAP50-95 diff: {safe_float(row.get('paired_mean_difference')):.6f}")
        print(f"V10b wins/losses/ties    : {int(row.get('wins', 0))}/{int(row.get('losses', 0))}/{int(row.get('ties', 0))}")
    if len(precision_diff):
        print(f"Precision mean diff      : {safe_float(precision_diff.iloc[0].get('paired_mean_difference')):.6f}")
    if len(recall_diff):
        print(f"Recall mean diff         : {safe_float(recall_diff.iloc[0].get('paired_mean_difference')):.6f}")
    if len(comparison) and {"random_map5095", "v10b_map5095"}.issubset(comparison.columns):
        print(f"Random mean mAP50-95     : {pd.to_numeric(comparison['random_map5095'], errors='coerce').mean():.6f}")
        print(f"V10b mean mAP50-95       : {pd.to_numeric(comparison['v10b_map5095'], errors='coerce').mean():.6f}")
    print("Final test used=False")
    print("Method weights frozen=True")
    print("=" * 100)


if __name__ == "__main__":
    main()

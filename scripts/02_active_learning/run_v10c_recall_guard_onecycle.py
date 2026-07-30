"""V10c recall-guard one-cycle runner.

Purpose:
    Test a new GT-free detector-aware selector designed specifically to fix the
    V10b failure mode observed on 2026-07-12: precision improved, but recall
    dropped.  V10c keeps the instance-rich/DINO core, then reserves a small
    recall-guard quota for low-coverage / no-box / low-confidence samples.

Default validation seeds are 47-50 because seeds 43-46 were already inspected
when diagnosing V10b.  Final test remains locked and is never evaluated.

Per acquisition seed:
    1. Build independent NEU protocol split.
    2. Sample shared initial 60.
    3. Train/evaluate shared Round0.
    4. Score unlabeled pool with Round0 detector.
    5. Select Random query 30.
    6. Select V10c query 30 = core + recall guard.
    7. Train/evaluate Random budget 90.
    8. Train/evaluate V10c budget 90.
    9. Recover development-only NEU6 per-class metrics from best.pt.

No final-test evaluation. No GT oracle acquisition.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import Counter
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
import run_v10b_multiseed_onecycle as base  # noqa: E402


PROJECT_ROOT = base.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "active_learning_v10c_recall_guard_onecycle"
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "active_learning_v10c_recall_guard_onecycle"

ROUND0_STRATEGY = base.ROUND0_STRATEGY
RANDOM_STRATEGY = base.RANDOM_STRATEGY
V10C_STRATEGY = "DetectorRecallGuardDINOInstanceV10c"

DEFAULT_V10C_WEIGHTS = {
    "detector_uncertainty": 0.30,
    "dino_visual_distance": 0.35,
    "predicted_class_deficit": 0.20,
    "pseudo_instance_count": 0.15,
}

DEFAULT_RECALL_GUARD_WEIGHTS = {
    "detector_uncertainty": 0.25,
    "dino_visual_distance": 0.30,
    "predicted_class_deficit": 0.20,
    "low_coverage": 0.25,
}

PREVIOUSLY_INSPECTED_SEEDS = {42, 43, 44, 45, 46}


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


def parse_bool_env(name: str, default: bool = False) -> bool:
    return base.parse_bool_env(name, default)


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        raise ValueError(f"Invalid weights: {weights}")
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def read_v10c_weights() -> dict[str, float]:
    return normalize_weights(
        {
            "detector_uncertainty": float_env("AL_V10C_W_UNCERTAINTY", DEFAULT_V10C_WEIGHTS["detector_uncertainty"]),
            "dino_visual_distance": float_env("AL_V10C_W_DINO", DEFAULT_V10C_WEIGHTS["dino_visual_distance"]),
            "predicted_class_deficit": float_env("AL_V10C_W_BALANCE", DEFAULT_V10C_WEIGHTS["predicted_class_deficit"]),
            "pseudo_instance_count": float_env("AL_V10C_W_INSTANCE", DEFAULT_V10C_WEIGHTS["pseudo_instance_count"]),
        }
    )


def read_guard_weights() -> dict[str, float]:
    return normalize_weights(
        {
            "detector_uncertainty": float_env("AL_V10C_GUARD_W_UNCERTAINTY", DEFAULT_RECALL_GUARD_WEIGHTS["detector_uncertainty"]),
            "dino_visual_distance": float_env("AL_V10C_GUARD_W_DINO", DEFAULT_RECALL_GUARD_WEIGHTS["dino_visual_distance"]),
            "predicted_class_deficit": float_env("AL_V10C_GUARD_W_BALANCE", DEFAULT_RECALL_GUARD_WEIGHTS["predicted_class_deficit"]),
            "low_coverage": float_env("AL_V10C_GUARD_W_LOW_COVERAGE", DEFAULT_RECALL_GUARD_WEIGHTS["low_coverage"]),
        }
    )


def parse_seed_list() -> list[int]:
    raw = os.environ.get("AL_ACQUISITION_SEEDS", "47,48,49,50")
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise ValueError("AL_ACQUISITION_SEEDS is empty.")
    return seeds


def table_md(df: pd.DataFrame) -> str:
    return base.table_md(df)


def sample_ids(df: pd.DataFrame) -> set[str]:
    if df.empty or "sample_id" not in df.columns:
        return set()
    return set(df["sample_id"].astype(str))


def safe_float(value: Any) -> float:
    return base.safe_float(value)


def class_universe_from_scored(scored_pool: pd.DataFrame) -> list[str]:
    class_universe = sorted(str(v) for v in getattr(base.v6, "NEU_CLASSES", []) if str(v))
    if class_universe:
        return class_universe
    return sorted(
        c
        for c in scored_pool.get("detector_pred_class", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        if c != "__no_box__"
    )


def add_v10c_component_scores(
    candidates: pd.DataFrame,
    *,
    initial_df: pd.DataFrame,
    selected_ids: list[str],
    selected_pred_counts: Counter,
    embedding_lookup: dict[str, np.ndarray],
    weights: dict[str, float],
) -> pd.DataFrame:
    out = candidates.copy()
    reference_ids = initial_df["sample_id"].astype(str).tolist() + selected_ids
    class_universe = class_universe_from_scored(out)
    labeled_counts = Counter(initial_df["class_hint"].dropna().astype(str))
    out["detector_pseudo_box_count"] = pd.to_numeric(out.get("detector_pseudo_box_count", 0), errors="coerce").fillna(0)
    out["detector_no_box"] = pd.to_numeric(out.get("detector_no_box", 0), errors="coerce").fillna(0)
    out["detector_max_conf"] = pd.to_numeric(out.get("detector_max_conf", 0), errors="coerce").fillna(0.0)
    out["_visual_distance_raw"] = [
        base.detector_probe.min_cosine_distance(str(sid), reference_ids, embedding_lookup)
        for sid in out["sample_id"].astype(str)
    ]
    out["_balance_deficit_raw"] = [
        base.detector_probe.class_deficit_score(str(cls), labeled_counts, selected_pred_counts, class_universe)
        if str(cls) != "__no_box__"
        else 0.5
        for cls in out["detector_pred_class"].astype(str)
    ]
    out["_instance_log_raw"] = np.log1p(out["detector_pseudo_box_count"].clip(lower=0))
    out["_usable_uncertainty_raw"] = pd.to_numeric(out.get("detector_uncertainty", 0), errors="coerce").fillna(0.0)
    out["_usable_uncertainty_raw"] = out["_usable_uncertainty_raw"] * (1.0 - 0.50 * out["detector_no_box"].clip(lower=0, upper=1))
    out["_detector_norm"] = base.detector_probe.normalize(out["_usable_uncertainty_raw"])
    out["_visual_norm"] = base.detector_probe.normalize(out["_visual_distance_raw"])
    out["_balance_norm"] = base.detector_probe.normalize(out["_balance_deficit_raw"])
    out["_instance_norm"] = base.detector_probe.normalize(out["_instance_log_raw"])
    out["_v10c_score"] = (
        weights["detector_uncertainty"] * out["_detector_norm"]
        + weights["dino_visual_distance"] * out["_visual_norm"]
        + weights["predicted_class_deficit"] * out["_balance_norm"]
        + weights["pseudo_instance_count"] * out["_instance_norm"]
    )
    return out


def add_guard_scores(
    candidates: pd.DataFrame,
    *,
    initial_df: pd.DataFrame,
    selected_ids: list[str],
    selected_pred_counts: Counter,
    embedding_lookup: dict[str, np.ndarray],
    weights: dict[str, float],
) -> pd.DataFrame:
    out = candidates.copy()
    reference_ids = initial_df["sample_id"].astype(str).tolist() + selected_ids
    class_universe = class_universe_from_scored(out)
    labeled_counts = Counter(initial_df["class_hint"].dropna().astype(str))
    out["detector_pseudo_box_count"] = pd.to_numeric(out.get("detector_pseudo_box_count", 0), errors="coerce").fillna(0)
    out["detector_no_box"] = pd.to_numeric(out.get("detector_no_box", 0), errors="coerce").fillna(0)
    out["detector_max_conf"] = pd.to_numeric(out.get("detector_max_conf", 0), errors="coerce").fillna(0.0)
    out["_visual_distance_raw"] = [
        base.detector_probe.min_cosine_distance(str(sid), reference_ids, embedding_lookup)
        for sid in out["sample_id"].astype(str)
    ]
    out["_balance_deficit_raw"] = [
        base.detector_probe.class_deficit_score(str(cls), labeled_counts, selected_pred_counts, class_universe)
        if str(cls) != "__no_box__"
        else 0.5
        for cls in out["detector_pred_class"].astype(str)
    ]
    low_instance = 1.0 - base.detector_probe.normalize(np.log1p(out["detector_pseudo_box_count"].clip(lower=0)))
    no_box_bonus = out["detector_no_box"].clip(lower=0, upper=1)
    low_conf_bonus = 1.0 - base.detector_probe.normalize(out["detector_max_conf"])
    out["_low_coverage_raw"] = 0.50 * low_instance + 0.30 * no_box_bonus + 0.20 * low_conf_bonus
    out["_detector_norm"] = base.detector_probe.normalize(pd.to_numeric(out.get("detector_uncertainty", 0), errors="coerce").fillna(0.0))
    out["_visual_norm"] = base.detector_probe.normalize(out["_visual_distance_raw"])
    out["_balance_norm"] = base.detector_probe.normalize(out["_balance_deficit_raw"])
    out["_low_coverage_norm"] = base.detector_probe.normalize(out["_low_coverage_raw"])
    out["_v10c_guard_score"] = (
        weights["detector_uncertainty"] * out["_detector_norm"]
        + weights["dino_visual_distance"] * out["_visual_norm"]
        + weights["predicted_class_deficit"] * out["_balance_norm"]
        + weights["low_coverage"] * out["_low_coverage_norm"]
    )
    return out


def select_v10c_recall_guard(
    scored_pool: pd.DataFrame,
    initial_df: pd.DataFrame,
    embedding_lookup: dict[str, np.ndarray],
    *,
    query_size: int,
    candidate_fraction: float,
    core_size: int,
    recall_guard_size: int,
    v10c_weights: dict[str, float],
    guard_weights: dict[str, float],
    core_max_no_box: int,
    core_min_pseudo_boxes: int,
    core_max_per_pred_class: int,
    guard_max_no_box: int,
    guard_max_per_pred_class: int,
) -> pd.DataFrame:
    """Select V10c query without using ground-truth labels from the pool."""
    if query_size <= 0:
        return scored_pool.iloc[0:0].copy()
    core_size = max(0, min(query_size, core_size))
    recall_guard_size = max(0, min(query_size - core_size, recall_guard_size))

    scored = scored_pool.copy()
    if "detector_no_box" not in scored.columns:
        scored["detector_no_box"] = scored["detector_pred_class"].astype(str).eq("__no_box__").astype(float)
    scored["detector_pseudo_box_count"] = pd.to_numeric(scored.get("detector_pseudo_box_count", 0), errors="coerce").fillna(0)
    scored["detector_max_conf"] = pd.to_numeric(scored.get("detector_max_conf", 0), errors="coerce").fillna(0.0)

    selected_parts: list[pd.DataFrame] = []
    selected_ids: list[str] = []
    selected_pred_counts: Counter = Counter()
    no_box_selected = 0

    if core_size > 0:
        core = base.stage1.select_instance_rich_dino_balanced(
            scored,
            initial_df,
            embedding_lookup,
            query_size=core_size,
            candidate_fraction=candidate_fraction,
            weights=v10c_weights,
            max_no_box=core_max_no_box,
            min_pseudo_boxes=core_min_pseudo_boxes,
            max_per_pred_class=core_max_per_pred_class,
        ).copy()
        if len(core):
            core["v10c_phase"] = "core_instance_dino"
            selected_parts.append(core)
            for _, row in core.iterrows():
                sid = str(row["sample_id"])
                selected_ids.append(sid)
                pred_cls = str(row.get("detector_pred_class", ""))
                if pred_cls == "__no_box__":
                    no_box_selected += 1
                elif pred_cls:
                    selected_pred_counts[pred_cls] += 1

    remaining = scored[~scored["sample_id"].astype(str).isin(set(selected_ids))].copy()
    low_conf_threshold = remaining["detector_max_conf"].quantile(0.35) if len(remaining) else 0.0
    guard_pool = remaining[
        remaining["detector_pred_class"].astype(str).eq("__no_box__")
        | remaining["detector_pseudo_box_count"].le(1)
        | remaining["detector_max_conf"].le(low_conf_threshold)
    ].copy()
    if len(guard_pool) < recall_guard_size:
        guard_pool = remaining.copy()

    for _ in range(min(recall_guard_size, len(guard_pool))):
        scored_guard = add_guard_scores(
            guard_pool,
            initial_df=initial_df,
            selected_ids=selected_ids,
            selected_pred_counts=selected_pred_counts,
            embedding_lookup=embedding_lookup,
            weights=guard_weights,
        )
        feasible = scored_guard.copy()
        if no_box_selected >= guard_max_no_box:
            feasible = feasible[feasible["detector_pred_class"].astype(str).ne("__no_box__")]
        if guard_max_per_pred_class > 0:
            feasible = feasible[
                feasible["detector_pred_class"].astype(str).map(
                    lambda c: c == "__no_box__" or selected_pred_counts.get(c, 0) < guard_max_per_pred_class
                )
            ]
        if feasible.empty:
            feasible = scored_guard.copy()
        pick_idx = feasible.sort_values(
            ["_v10c_guard_score", "_visual_distance_raw", "_low_coverage_raw", "sample_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).index[0]
        picked = scored_guard.loc[[pick_idx]].copy()
        picked["v10c_phase"] = "recall_guard"
        selected_parts.append(picked)
        sid = str(picked.iloc[0]["sample_id"])
        selected_ids.append(sid)
        pred_cls = str(picked.iloc[0].get("detector_pred_class", ""))
        if pred_cls == "__no_box__":
            no_box_selected += 1
        elif pred_cls:
            selected_pred_counts[pred_cls] += 1
        guard_pool = guard_pool.drop(index=pick_idx)
        remaining = remaining.drop(index=pick_idx, errors="ignore")

    selected = pd.concat(selected_parts, ignore_index=True, sort=False) if selected_parts else scored.iloc[0:0].copy()
    if len(selected) < query_size:
        fill_n = query_size - len(selected)
        fill_pool = scored[~scored["sample_id"].astype(str).isin(sample_ids(selected))].copy()
        for _ in range(min(fill_n, len(fill_pool))):
            scored_fill = add_v10c_component_scores(
                fill_pool,
                initial_df=initial_df,
                selected_ids=selected["sample_id"].astype(str).tolist(),
                selected_pred_counts=selected_pred_counts,
                embedding_lookup=embedding_lookup,
                weights=v10c_weights,
            )
            pick_idx = scored_fill.sort_values(
                ["_v10c_score", "_visual_distance_raw", "detector_max_conf", "sample_id"],
                ascending=[False, False, False, True],
                kind="mergesort",
            ).index[0]
            picked = scored_fill.loc[[pick_idx]].copy()
            picked["v10c_phase"] = "fill_v10c_score"
            selected = pd.concat([selected, picked], ignore_index=True, sort=False)
            pred_cls = str(picked.iloc[0].get("detector_pred_class", ""))
            if pred_cls and pred_cls != "__no_box__":
                selected_pred_counts[pred_cls] += 1
            fill_pool = fill_pool.drop(index=pick_idx)

    return selected.drop_duplicates("sample_id", keep="first").head(query_size).reset_index(drop=True)


def compute_selection_diagnostics(
    *,
    seed: int,
    initial_df: pd.DataFrame,
    random_query: pd.DataFrame,
    v10c_query: pd.DataFrame,
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
        V10C_STRATEGY: v10c_query,
    }
    static_scores = v10bprobe.static_component_diagnostics(scored_pool, initial_df, embedding_lookup)
    component = v10bprobe.selection_component_summary(static_scores, selections)
    geometry = v10bprobe.selection_geometry_summary(initial_df, selections, embedding_lookup)
    for df in [component, geometry]:
        if len(df):
            df.insert(0, "acquisition_seed", seed)
    return component, geometry


def random_v10c_overlap(random_query: pd.DataFrame, v10c_query: pd.DataFrame, seed: int) -> pd.DataFrame:
    random_ids = sample_ids(random_query)
    v10c_ids = sample_ids(v10c_query)
    union = random_ids | v10c_ids
    return pd.DataFrame(
        [
            {
                "acquisition_seed": seed,
                "random_size": len(random_ids),
                "v10c_size": len(v10c_ids),
                "overlap_count": len(random_ids & v10c_ids),
                "random_only_count": len(random_ids - v10c_ids),
                "v10c_only_count": len(v10c_ids - random_ids),
                "jaccard": len(random_ids & v10c_ids) / len(union) if union else np.nan,
            }
        ]
    )


def summarize_results(results_df: pd.DataFrame, per_class_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if results_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    results = base.add_f1(results_df)
    round1 = results[results["strategy"].isin([RANDOM_STRATEGY, V10C_STRATEGY])].copy()
    metric_cols = ["map50", "map5095", "precision", "recall", "f1"]

    aggregate_rows = []
    for strategy, sub in round1.groupby("strategy", dropna=False):
        row: dict[str, Any] = {"strategy": strategy, "num_seeds": int(sub["acquisition_seed"].nunique())}
        for metric in metric_cols:
            values = pd.to_numeric(sub.get(metric), errors="coerce")
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)

    paired_rows = []
    comparison_rows = []
    for seed, sub in round1.groupby("acquisition_seed", dropna=False):
        random = sub[sub["strategy"] == RANDOM_STRATEGY]
        v10c = sub[sub["strategy"] == V10C_STRATEGY]
        if random.empty or v10c.empty:
            continue
        random_row = random.iloc[0]
        v10c_row = v10c.iloc[0]
        comp = {"acquisition_seed": seed}
        for metric in metric_cols:
            r = safe_float(random_row.get(metric))
            v = safe_float(v10c_row.get(metric))
            comp[f"random_{metric}"] = r
            comp[f"v10c_{metric}"] = v
            comp[f"diff_{metric}"] = v - r
            paired_rows.append({"acquisition_seed": seed, "metric": metric, "difference_v10c_minus_random": v - r})
        comparison_rows.append(comp)
    comparison = pd.DataFrame(comparison_rows)

    paired_summary_rows = []
    paired_df = pd.DataFrame(paired_rows)
    for metric in metric_cols:
        vals = pd.to_numeric(paired_df[paired_df["metric"] == metric]["difference_v10c_minus_random"], errors="coerce").dropna()
        low, high = base.bootstrap_ci(vals.to_numpy(), seed=20260713) if len(vals) else (np.nan, np.nan)
        paired_summary_rows.append(
            {
                "metric": metric,
                "num_pairs": int(len(vals)),
                "paired_mean_difference": float(vals.mean()) if len(vals) else np.nan,
                "paired_median_difference": float(vals.median()) if len(vals) else np.nan,
                "wins": int((vals > 0).sum()) if len(vals) else 0,
                "losses": int((vals < 0).sum()) if len(vals) else 0,
                "ties": int((vals == 0).sum()) if len(vals) else 0,
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
            }
        )
    paired_summary = pd.DataFrame(paired_summary_rows)

    per_class_summary = pd.DataFrame()
    if not per_class_df.empty and {"strategy", "class_name", "ap5095"}.issubset(per_class_df.columns):
        rows = []
        good = per_class_df[per_class_df.get("status", "success").astype(str).eq("success")].copy()
        for seed, seed_sub in good.groupby("acquisition_seed", dropna=False):
            for cls, cls_sub in seed_sub.groupby("class_name", dropna=False):
                random = cls_sub[cls_sub["strategy"] == RANDOM_STRATEGY]
                v10c = cls_sub[cls_sub["strategy"] == V10C_STRATEGY]
                if random.empty or v10c.empty:
                    continue
                rows.append(
                    {
                        "acquisition_seed": seed,
                        "class_name": cls,
                        "random_ap5095": safe_float(random.iloc[0].get("ap5095")),
                        "v10c_ap5095": safe_float(v10c.iloc[0].get("ap5095")),
                        "diff_ap5095": safe_float(v10c.iloc[0].get("ap5095")) - safe_float(random.iloc[0].get("ap5095")),
                    }
                )
        per_class_summary = pd.DataFrame(rows)
    return aggregate, paired_summary, comparison, per_class_summary


def write_outputs(save_dir: Path, config: dict[str, Any], accum: dict[str, list[Any]]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    frames: dict[str, pd.DataFrame] = {
        "seed_registry.csv": pd.DataFrame(accum["seed_registry"]),
        "v10c_onecycle_results.csv": pd.DataFrame(accum["results_rows"]),
        "v10c_selected_samples.csv": pd.concat(accum["selected_rows"], ignore_index=True, sort=False) if accum["selected_rows"] else pd.DataFrame(),
        "v10c_cumulative_labeled_sets.csv": pd.concat(accum["cumulative_rows"], ignore_index=True, sort=False) if accum["cumulative_rows"] else pd.DataFrame(),
        "v10c_build_logs.csv": pd.concat(accum["build_logs"], ignore_index=True, sort=False) if accum["build_logs"] else pd.DataFrame(),
        "v10c_runtime.csv": pd.DataFrame(accum["runtime_rows"]),
        "v10c_per_class_metrics.csv": pd.concat(accum["per_class_rows"], ignore_index=True, sort=False) if accum["per_class_rows"] else pd.DataFrame(),
        "v10c_split_logs.csv": pd.concat(accum["split_logs"], ignore_index=True, sort=False) if accum["split_logs"] else pd.DataFrame(),
        "v10c_selection_component_summary.csv": pd.concat(accum["selection_components"], ignore_index=True, sort=False) if accum["selection_components"] else pd.DataFrame(),
        "v10c_selection_geometry_summary.csv": pd.concat(accum["selection_geometry"], ignore_index=True, sort=False) if accum["selection_geometry"] else pd.DataFrame(),
        "v10c_random_overlap.csv": pd.concat(accum["overlap_rows"], ignore_index=True, sort=False) if accum["overlap_rows"] else pd.DataFrame(),
    }
    for name, df in frames.items():
        df.to_csv(save_dir / name, index=False, encoding="utf-8-sig")

    results_df = frames["v10c_onecycle_results.csv"]
    per_class_df = frames["v10c_per_class_metrics.csv"]
    aggregate, paired, comparison, per_class_summary = summarize_results(results_df, per_class_df)
    aggregate.to_csv(save_dir / "aggregate_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(save_dir / "paired_difference_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(save_dir / "seedwise_random_v10c_comparison.csv", index=False, encoding="utf-8-sig")
    per_class_summary.to_csv(save_dir / "per_class_v10c_minus_random.csv", index=False, encoding="utf-8-sig")
    write_summary_md(save_dir, config, frames, aggregate, paired, comparison, per_class_summary)


def write_summary_md(
    save_dir: Path,
    config: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    comparison: pd.DataFrame,
    per_class_summary: pd.DataFrame,
) -> None:
    results_df = frames["v10c_onecycle_results.csv"]
    selected_df = frames["v10c_selected_samples.csv"]
    phase_summary = pd.DataFrame()
    if not selected_df.empty and "v10c_phase" in selected_df.columns:
        phase_summary = (
            selected_df[selected_df["strategy"].astype(str).eq(V10C_STRATEGY)]
            .groupby(["acquisition_seed", "v10c_phase"], dropna=False)
            .size()
            .reset_index(name="count")
        )

    lines = [
        "# V10c recall-guard one-cycle summary",
        "",
        f"- Experiment: `{config['experiment_id']}`",
        f"- Seeds: {config['acquisition_seeds']}",
        f"- Reused inspected seeds warning: {config['reused_inspected_seeds']}",
        f"- Dry run: {config['dry_run']}",
        f"- Final test used: {config['final_test_used']}",
        f"- Strategy: `{V10C_STRATEGY}`",
        "",
        "## Intent",
        "",
        "V10b reached near-parity with Random but showed a precision gain / recall loss trade-off. "
        "V10c reserves a small recall-guard quota for low-coverage samples while keeping the DINO/instance-rich core.",
        "",
        "## Aggregate results",
        "",
        table_md(aggregate),
        "",
        "## Paired V10c - Random differences",
        "",
        table_md(paired),
        "",
        "## Seedwise comparison",
        "",
        table_md(comparison),
        "",
        "## V10c phase summary",
        "",
        table_md(phase_summary),
        "",
        "## Per-class V10c - Random AP50-95",
        "",
        table_md(per_class_summary),
        "",
        "## Raw result rows",
        "",
        table_md(results_df[[c for c in ["acquisition_seed", "training_seed", "strategy", "round", "labeled_budget", "map50", "map5095", "precision", "recall", "f1", "train_status", "error"] if c in results_df.columns]]),
        "",
        "## Guardrails",
        "",
        "- This runner does not evaluate final test.",
        "- V10c uses no GT oracle signal during acquisition.",
        "- Seeds 47-50 are the default because seeds 42-46 have already influenced the V10c design.",
        "- If seeds 42-46 are reused, results should be described as development/tuning evidence only.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        "```",
    ]
    (save_dir / "v10c_recall_guard_onecycle_summary.md").write_text("\n".join(lines), encoding="utf-8")


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
        print(f"[SEED {acquisition_seed}] V10c training_seed={training_seed} dry_run={dry_run}")

        pool_df, dev_df, final_df, unused_df = base.v10smoke.build_v10_protocol_split(
            manifest,
            seed=acquisition_seed,
            pool_per_class=int(config["pool_per_class"]),
            dev_per_class=int(config["dev_per_class"]),
            final_per_class=int(config["final_per_class"]),
        )
        overlap_checks = base.validate_split(pool_df, dev_df, final_df)
        split_df = pd.DataFrame(
            [
                {"acquisition_seed": acquisition_seed, "split": "pool", "num_images": len(pool_df), **overlap_checks},
                {"acquisition_seed": acquisition_seed, "split": "development_eval", "num_images": len(dev_df), **overlap_checks},
                {"acquisition_seed": acquisition_seed, "split": "final_test_LOCKED_UNUSED", "num_images": len(final_df), **overlap_checks},
                {"acquisition_seed": acquisition_seed, "split": "unused_reserve", "num_images": len(unused_df), **overlap_checks},
            ]
        )
        accum["split_logs"].append(split_df)
        pool_df.to_csv(save_dir / f"seed{acquisition_seed}_acquisition_pool_v10c.csv", index=False, encoding="utf-8-sig")
        dev_df.to_csv(save_dir / f"seed{acquisition_seed}_development_eval_v10c.csv", index=False, encoding="utf-8-sig")
        final_df.to_csv(save_dir / f"seed{acquisition_seed}_final_test_v10c_LOCKED_UNUSED.csv", index=False, encoding="utf-8-sig")
        unused_df.to_csv(save_dir / f"seed{acquisition_seed}_unused_reserve_v10c.csv", index=False, encoding="utf-8-sig")

        seed_artifact_dir = save_dir / f"seed{acquisition_seed}_artifacts"
        seed_artifact_dir.mkdir(parents=True, exist_ok=True)
        embedding_dir, embedding_lookup, embedding_config = base.v10smoke.load_or_build_embeddings(pool_df, seed_artifact_dir)

        initial_df = base.sample_initial_labeled_set(
            pool_df,
            initial_seed_size=int(config["initial_seed_size"]),
            acquisition_seed=acquisition_seed,
        ).reset_index(drop=True)
        current_pool = pool_df[~pool_df["sample_id"].isin(initial_df["sample_id"])].copy().reset_index(drop=True)
        random_query = base.v10smoke.select_random(current_pool, int(config["query_size"]), acquisition_seed)

        selected_frames = [
            base.add_selection_metadata(
                initial_df,
                acquisition_seed=acquisition_seed,
                training_seed=training_seed,
                strategy=ROUND0_STRATEGY,
                round_idx=0,
                selection_type="shared_initial_random",
            ),
            base.add_selection_metadata(
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

        yaml_path, build_log = base.v6.build_yolo_dataset(initial_df, dev_df, seed_dataset_root / ROUND0_STRATEGY)
        build_log.insert(0, "acquisition_seed", acquisition_seed)
        build_log.insert(1, "strategy", ROUND0_STRATEGY)
        build_log.insert(2, "round", 0)
        build_logs.append(build_log)
        row0, runtime0 = base.train_row(
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
            base.recover_per_class_metrics(
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
            v10c_query = current_pool.sort_values("sample_id", kind="mergesort").head(int(config["query_size"])).copy().reset_index(drop=True)
            v10c_query["v10c_phase"] = "dry_run_placeholder"
            v10c_query["dry_run_selector_note"] = "placeholder; real V10c query requires Round0 detector scores"
        else:
            if row0.get("train_status") != "success" or not round0_weight.exists():
                raise FileNotFoundError(f"Round0 best.pt missing or failed for seed {acquisition_seed}: {round0_weight}")
            model = base.detector_probe.YOLO(str(round0_weight))
            detector_scores = base.detector_probe.prediction_rows(
                model,
                current_pool,
                device=str(config["device"]),
                imgsz=int(config["imgsz"]),
                conf=float(config["predict_conf"]),
                iou=float(config["predict_iou"]),
            )
            scored_pool = current_pool.merge(detector_scores, on="sample_id", how="left")
            scored_pool.to_csv(save_dir / f"seed{acquisition_seed}_round1_detector_scores_v10c.csv", index=False, encoding="utf-8-sig")
            v10c_query = select_v10c_recall_guard(
                scored_pool,
                initial_df,
                embedding_lookup,
                query_size=int(config["query_size"]),
                candidate_fraction=float(config["candidate_fraction"]),
                core_size=int(config["v10c_core_size"]),
                recall_guard_size=int(config["v10c_recall_guard_size"]),
                v10c_weights=dict(config["v10c_weights"]),
                guard_weights=dict(config["recall_guard_weights"]),
                core_max_no_box=int(config["core_constraints"]["max_no_box"]),
                core_min_pseudo_boxes=int(config["core_constraints"]["min_pseudo_boxes"]),
                core_max_per_pred_class=int(config["core_constraints"]["max_per_pred_class"]),
                guard_max_no_box=int(config["guard_constraints"]["max_no_box"]),
                guard_max_per_pred_class=int(config["guard_constraints"]["max_per_pred_class"]),
            )
            if len(v10c_query) != int(config["query_size"]):
                raise ValueError(f"V10c selected {len(v10c_query)} samples for seed {acquisition_seed}; expected {config['query_size']}")
            component, geometry = compute_selection_diagnostics(
                seed=acquisition_seed,
                initial_df=initial_df,
                random_query=random_query,
                v10c_query=v10c_query,
                scored_pool=scored_pool,
                embedding_lookup=embedding_lookup,
            )
            accum["selection_components"].append(component)
            accum["selection_geometry"].append(geometry)

        selected_frames.append(
            base.add_selection_metadata(
                v10c_query,
                acquisition_seed=acquisition_seed,
                training_seed=training_seed,
                strategy=V10C_STRATEGY,
                round_idx=1,
                selection_type=V10C_STRATEGY,
                dry_run_placeholder=dry_run,
            )
        )
        checks = base.validate_labeled_sets(initial_df, random_query, v10c_query, dev_df, dry_run=dry_run)
        checks.insert(0, "acquisition_seed", acquisition_seed)
        checks.to_csv(save_dir / f"seed{acquisition_seed}_labeled_set_checks_v10c.csv", index=False, encoding="utf-8-sig")

        accum["overlap_rows"].append(random_v10c_overlap(random_query, v10c_query, acquisition_seed))
        cumulative_v10c = pd.concat([initial_df, v10c_query], ignore_index=True, sort=False).drop_duplicates("sample_id", keep="first")

        yaml_path, build_log = base.v6.build_yolo_dataset(cumulative_random, dev_df, seed_dataset_root / RANDOM_STRATEGY / "round_1")
        build_log.insert(0, "acquisition_seed", acquisition_seed)
        build_log.insert(1, "strategy", RANDOM_STRATEGY)
        build_log.insert(2, "round", 1)
        build_logs.append(build_log)
        row_random, runtime_random = base.train_row(
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
            base.recover_per_class_metrics(
                result_row=row_random,
                strategy_label=RANDOM_STRATEGY,
                acquisition_seed=acquisition_seed,
                training_seed=training_seed,
                round_idx=1,
                save_dir=save_dir,
            )
        )

        yaml_path, build_log = base.v6.build_yolo_dataset(cumulative_v10c, dev_df, seed_dataset_root / V10C_STRATEGY / "round_1")
        build_log.insert(0, "acquisition_seed", acquisition_seed)
        build_log.insert(1, "strategy", V10C_STRATEGY)
        build_log.insert(2, "round", 1)
        build_logs.append(build_log)
        row_v10c, runtime_v10c = base.train_row(
            yaml_path=yaml_path,
            save_dir=save_dir,
            strategy=V10C_STRATEGY,
            acquisition_seed=acquisition_seed,
            training_seed=training_seed,
            round_idx=1,
            labeled_budget=len(cumulative_v10c),
            dev_eval_size=len(dev_df),
        )
        accum["results_rows"].append(row_v10c)
        accum["runtime_rows"].append(runtime_v10c)
        accum["per_class_rows"].append(
            base.recover_per_class_metrics(
                result_row=row_v10c,
                strategy_label=V10C_STRATEGY,
                acquisition_seed=acquisition_seed,
                training_seed=training_seed,
                round_idx=1,
                save_dir=save_dir,
            )
        )

        accum["selected_rows"].extend(selected_frames)
        accum["cumulative_rows"].extend(
            [
                base.add_selection_metadata(
                    initial_df,
                    acquisition_seed=acquisition_seed,
                    training_seed=training_seed,
                    strategy=ROUND0_STRATEGY,
                    round_idx=0,
                    selection_type="cumulative_labeled",
                ),
                base.add_selection_metadata(
                    cumulative_random,
                    acquisition_seed=acquisition_seed,
                    training_seed=training_seed,
                    strategy=RANDOM_STRATEGY,
                    round_idx=1,
                    selection_type="cumulative_labeled",
                ),
                base.add_selection_metadata(
                    cumulative_v10c,
                    acquisition_seed=acquisition_seed,
                    training_seed=training_seed,
                    strategy=V10C_STRATEGY,
                    round_idx=1,
                    selection_type="cumulative_labeled",
                    dry_run_placeholder=dry_run,
                ),
            ]
        )
        accum["build_logs"].extend(build_logs)
        seed_status = "success"
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
                "v10b_executed": False,
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
                "v10b_executed": False,
            }
        )
    finally:
        write_outputs(save_dir, config, accum)
        print(f"[SEED {acquisition_seed}] status={seed_status} elapsed={time.perf_counter() - seed_t0:.1f}s")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"v10c_recall_guard_onecycle_{timestamp}"
    dataset_root = DATASETS_ROOT / save_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    seeds = parse_seed_list()
    reused = sorted(set(seeds) & PREVIOUSLY_INSPECTED_SEEDS)
    query_size = int_env("AL_QUERY_SIZE", 30)
    recall_guard_size = int_env("AL_V10C_RECALL_GUARD_SIZE", 6)
    core_size = int_env("AL_V10C_CORE_SIZE", max(0, query_size - recall_guard_size))
    manifest = base.v10smoke.ensure_neu_manifest()
    dry_run = parse_bool_env("AL_DRY_RUN_ONLY", True)

    config: dict[str, Any] = {
        "experiment_id": save_dir.name,
        "experiment_label": "V10c recall-guard one-cycle validation",
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "save_dir": str(save_dir),
        "dataset_root": str(dataset_root),
        "acquisition_seeds": seeds,
        "previously_inspected_seeds": sorted(PREVIOUSLY_INSPECTED_SEEDS),
        "reused_inspected_seeds": reused,
        "training_seed_rule": "training_seed = 1000 + acquisition_seed",
        "method_weights_frozen_for_this_runner": True,
        "v9b_executed": False,
        "v10b_executed": False,
        "full_curve": False,
        "final_test_used": False,
        "neu_manifest_size": int(len(manifest)),
        "pool_per_class": int_env("AL_V10_POOL_PER_CLASS", 150),
        "dev_per_class": int_env("AL_V10_DEV_PER_CLASS", 50),
        "final_per_class": int_env("AL_V10_FINAL_PER_CLASS", 50),
        "initial_seed_size": int_env("AL_INITIAL_SEED_SIZE", 60),
        "query_size": query_size,
        "rounds": 1,
        "strategies": [RANDOM_STRATEGY, V10C_STRATEGY],
        "expected_trainings_per_seed": 3,
        "expected_total_trainings": 3 * len(seeds),
        "v10c_weights": read_v10c_weights(),
        "recall_guard_weights": read_guard_weights(),
        "v10c_core_size": core_size,
        "v10c_recall_guard_size": recall_guard_size,
        "core_constraints": {
            "max_no_box": int_env("AL_V10C_CORE_MAX_NO_BOX", 0),
            "min_pseudo_boxes": int_env("AL_V10C_CORE_MIN_PSEUDO_BOXES", 2),
            "max_per_pred_class": int_env("AL_V10C_CORE_MAX_PER_PRED_CLASS", 2),
        },
        "guard_constraints": {
            "max_no_box": int_env("AL_V10C_GUARD_MAX_NO_BOX", 2),
            "max_per_pred_class": int_env("AL_V10C_GUARD_MAX_PER_PRED_CLASS", 3),
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
        "git_commit": base.git_commit(),
        "git_dirty": base.git_dirty(),
        "runner_sha256": base.file_sha256(Path(__file__).resolve()),
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
    print("[V10c recall-guard one-cycle validation]")
    print(f"Output dir : {save_dir}")
    print(f"Seeds      : {seeds}")
    print(f"Dry run    : {dry_run}")
    print(f"Core/guard : {core_size}/{recall_guard_size}")
    print(f"Reused inspected seeds: {reused}")
    print("V9b/V10b   : excluded")
    print("Final test : locked / not evaluated")
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
    paired_path = save_dir / "paired_difference_summary.csv"
    paired = pd.read_csv(paired_path) if paired_path.exists() and paired_path.stat().st_size else pd.DataFrame()
    completed = seed_registry[seed_registry["status"].astype(str).eq("success")]["acquisition_seed"].tolist() if len(seed_registry) else []
    failed = seed_registry[~seed_registry["status"].astype(str).eq("success")]["acquisition_seed"].tolist() if len(seed_registry) else []
    mapdiff = paired[paired["metric"].astype(str).eq("map5095")] if len(paired) and "metric" in paired.columns else pd.DataFrame()
    precision_diff = paired[paired["metric"].astype(str).eq("precision")] if len(paired) and "metric" in paired.columns else pd.DataFrame()
    recall_diff = paired[paired["metric"].astype(str).eq("recall")] if len(paired) and "metric" in paired.columns else pd.DataFrame()

    print("=" * 100)
    print("[DONE] V10c recall-guard one-cycle runner finished")
    print(f"Output dir: {save_dir}")
    print(f"Completed seeds: {completed}")
    print(f"Failed seeds   : {failed}")
    if len(mapdiff):
        row = mapdiff.iloc[0]
        print(f"Paired mean mAP50-95 diff: {safe_float(row.get('paired_mean_difference')):.6f}")
        print(f"V10c wins/losses/ties    : {int(row.get('wins', 0))}/{int(row.get('losses', 0))}/{int(row.get('ties', 0))}")
    if len(precision_diff):
        print(f"Precision mean diff      : {safe_float(precision_diff.iloc[0].get('paired_mean_difference')):.6f}")
    if len(recall_diff):
        print(f"Recall mean diff         : {safe_float(recall_diff.iloc[0].get('paired_mean_difference')):.6f}")
    print("Final test used=False")
    print("Method weights frozen=True")
    print("=" * 100)


if __name__ == "__main__":
    main()

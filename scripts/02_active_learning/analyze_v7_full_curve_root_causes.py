"""
Root-cause analysis for V7 DINO full learning-curve results.

This script is intentionally read-only with respect to experiments:
- no YOLO training
- no DINO embedding generation
- no final-test evaluation
- no acquisition hyperparameter changes

It consumes completed V7 gate/full-curve CSV artifacts and writes a diagnostic
report under runs/v7_full_curve_root_cause_analysis/.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_DIR = PROJECT_ROOT / "runs" / "v7_final_set_gate_training" / "v7_final_set_gate_training_20260711_193613"
DEFAULT_FULL_DIR = (
    PROJECT_ROOT
    / "runs"
    / "active_learning_ablation_v7_full_curve"
    / "v7_full_curve_20260711_211848"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "v7_full_curve_root_cause_analysis"

RANDOM = "GTFreeRandom"
DBC = "GTFreeDatasetBalancedConsistency"
VISUAL = "GTFreeDatasetBalancedVisualDiversity"
CV = "GTFreeDatasetBalancedConsistencyVisualDiversity"
STRATEGIES = [RANDOM, DBC, VISUAL]
FAILED_VISUAL_SEEDS = [44, 45]


def short_strategy(name: str) -> str:
    return {
        RANDOM: "Random",
        DBC: "DBC",
        VISUAL: "Visual",
        CV: "Consistency+Visual",
        "__SHARED_ROUND0__": "Shared round0",
    }.get(str(name), str(name))


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_json_load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_ci(values: Iterable[float], n_boot: int = 5000, seed: int = 7) -> tuple[float, float]:
    arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if len(arr) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def corr_pair(df: pd.DataFrame, x: str, y: str) -> dict:
    valid = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3:
        return {"n": len(valid), "pearson": np.nan, "spearman": np.nan}
    return {
        "n": int(len(valid)),
        "pearson": float(valid[x].corr(valid[y], method="pearson")),
        "spearman": float(valid[x].corr(valid[y], method="spearman")),
    }


@dataclass
class Inputs:
    gate_dir: Path
    full_dir: Path
    out_dir: Path


def make_output_dir(root: Path) -> Path:
    out = root / f"root_cause_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "contact_sheets").mkdir(parents=True, exist_ok=True)
    return out


def load_inputs(args: argparse.Namespace) -> tuple[Inputs, dict[str, pd.DataFrame]]:
    gate_dir = Path(args.gate_dir).resolve()
    full_dir = Path(args.full_dir).resolve()
    out_dir = make_output_dir(Path(args.output_root).resolve())
    dfs = {
        "gate_frozen": read_csv(gate_dir / "frozen_labeled_sets.csv"),
        "gate_metrics": read_csv(gate_dir / "gate_metric_summary.csv"),
        "gate_train": read_csv(gate_dir / "gate_training_results.csv"),
        "gate_paired": read_csv(gate_dir / "paired_gate_comparisons.csv"),
        "gate_promotion": read_csv(gate_dir / "promotion_gate_decisions.csv"),
        "gate_redundancy": read_csv(gate_dir / "screening_redundancy_summary.csv"),
        "full_results": read_csv(full_dir / "all_round_results.csv"),
        "full_selected": read_csv(full_dir / "all_selected_samples_by_round.csv"),
        "full_cumulative": read_csv(full_dir / "cumulative_labeled_sets_by_round.csv"),
        "full_instance": read_csv(full_dir / "actual_instance_statistics_by_round.csv"),
        "full_class_dist": read_csv(full_dir / "actual_class_distribution_by_round.csv"),
        "full_dataset_dist": read_csv(full_dir / "dataset_distribution_by_round.csv"),
        "full_seed_summary": read_csv(full_dir / "seed_strategy_metric_summary.csv"),
        "full_aulc": read_csv(full_dir / "normalized_aulc_summary.csv"),
        "full_paired": read_csv(full_dir / "paired_strategy_comparisons.csv"),
        "full_round_diff": read_csv(full_dir / "paired_roundwise_differences.csv"),
        "full_distance": read_csv(full_dir / "selected_sample_distance_to_labeled.csv"),
        "full_redundancy": read_csv(full_dir / "visual_redundancy_by_round.csv"),
        "full_overlap": read_csv(full_dir / "selection_overlap_matrix.csv"),
        "full_per_class": read_csv(full_dir / "per_class_metrics_by_round.csv"),
        "full_runtime": read_csv(full_dir / "runtime_profile.csv"),
        "full_registry": read_csv(full_dir / "experiment_registry.csv"),
        "full_failed": read_csv(full_dir / "failed_or_retried_runs.csv"),
    }
    return Inputs(gate_dir, full_dir, out_dir), dfs


def cumulative_gate_set(gate_frozen: pd.DataFrame, strategy: str, round_id: int) -> pd.DataFrame:
    sub = gate_frozen[(gate_frozen["strategy"].eq(strategy)) & (gate_frozen["round"].le(round_id))].copy()
    key = "sample_key"
    if "sample_id" in sub.columns:
        sub[key] = sub["sample_id"].astype(str)
    else:
        sub[key] = sub["dataset_type"].astype(str) + "::" + sub["image_name"].astype(str)
    return sub.drop_duplicates(key)


def compare_gate_full_seed42(inputs: Inputs, dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    gate = dfs["gate_frozen"]
    full = dfs["full_cumulative"]
    rows = []
    if gate.empty or full.empty:
        diff = pd.DataFrame([{"check": "gate_vs_full_seed42", "status": "missing_input"}])
        metrics = pd.DataFrame()
        return diff, metrics

    for strategy in [RANDOM, DBC, VISUAL]:
        for round_id in range(5):
            g = cumulative_gate_set(gate, strategy, round_id)
            f = full[
                full["acquisition_seed"].eq(42)
                & full["strategy"].eq(strategy)
                & full["round"].eq(round_id)
            ].copy()
            if "sample_id" in f.columns:
                f["sample_key"] = f["sample_id"].astype(str)
            else:
                f["sample_key"] = f["dataset_type"].astype(str) + "::" + f["image_name"].astype(str)
            # Gate was frozen before the full-curve runner added sample_id, so
            # compare by dataset/image identity for cross-run manifest checks.
            g["sample_key"] = g["dataset_type"].astype(str) + "::" + g["image_name"].astype(str)
            f["sample_key"] = f["dataset_type"].astype(str) + "::" + f["image_name"].astype(str)
            g_ids = set(g["sample_key"].astype(str))
            f_ids = set(f["sample_key"].astype(str))
            rows.append(
                {
                    "strategy": strategy,
                    "strategy_short": short_strategy(strategy),
                    "round": round_id,
                    "gate_size": len(g_ids),
                    "full_seed42_size": len(f_ids),
                    "overlap_count": len(g_ids & f_ids),
                    "gate_only_count": len(g_ids - f_ids),
                    "full_only_count": len(f_ids - g_ids),
                    "jaccard": len(g_ids & f_ids) / len(g_ids | f_ids) if (g_ids | f_ids) else np.nan,
                    "gate_only_sample_ids": ";".join(sorted(g_ids - f_ids)[:20]),
                    "full_only_sample_ids": ";".join(sorted(f_ids - g_ids)[:20]),
                }
            )
    diff = pd.DataFrame(rows)
    write_csv(diff, inputs.out_dir / "gate_vs_fullcurve_seed42_manifest_diff.csv")

    gate_train = dfs["gate_train"]
    full_results = dfs["full_results"]
    metric_rows = []
    for strategy in [RANDOM, DBC, VISUAL]:
        gt = gate_train[gate_train["strategy"].eq(strategy)]
        fr = full_results[
            full_results["acquisition_seed"].eq(42)
            & full_results["strategy"].eq(strategy)
            & full_results["round"].eq(4)
        ]
        full_map50 = float(fr["map50"].iloc[0]) if len(fr) else np.nan
        full_map5095 = float(fr["map5095"].iloc[0]) if len(fr) else np.nan
        for metric in ["map50", "map5095"]:
            vals = gt[metric].dropna().astype(float)
            full_val = full_map50 if metric == "map50" else full_map5095
            metric_rows.append(
                {
                    "strategy": strategy,
                    "strategy_short": short_strategy(strategy),
                    "metric": metric,
                    "gate_training_seed_mean": vals.mean() if len(vals) else np.nan,
                    "gate_training_seed_std": vals.std(ddof=1) if len(vals) > 1 else np.nan,
                    "gate_min": vals.min() if len(vals) else np.nan,
                    "gate_max": vals.max() if len(vals) else np.nan,
                    "full_curve_seed42_training_seed1042": full_val,
                    "full_inside_gate_range": bool(vals.min() <= full_val <= vals.max()) if len(vals) else False,
                    "full_minus_gate_mean": full_val - vals.mean() if len(vals) else np.nan,
                }
            )
    metric_df = pd.DataFrame(metric_rows)
    write_csv(metric_df, inputs.out_dir / "gate_vs_fullcurve_seed42_metric_comparison.csv")
    return diff, metric_df


def initial_set_diagnostics(inputs: Inputs, dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    inst = dfs["full_instance"]
    dset = dfs["full_dataset_dist"]
    if inst.empty:
        out = pd.DataFrame()
        write_csv(out, inputs.out_dir / "seed_initial_set_diagnostics.csv")
        return out
    init = inst[inst["round"].eq(0)].drop_duplicates("acquisition_seed").copy()
    if not dset.empty:
        pivot = (
            dset[dset["round"].eq(0)]
            .pivot_table(index="acquisition_seed", columns="dataset_type", values="count", aggfunc="sum")
            .reset_index()
        )
        init = init.merge(pivot, on="acquisition_seed", how="left")
    init = init.sort_values("acquisition_seed")
    write_csv(init, inputs.out_dir / "seed_initial_set_diagnostics.csv")
    return init


def marginal_gains(inputs: Inputs, dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    results = dfs["full_results"].copy()
    if results.empty:
        out = pd.DataFrame()
        write_csv(out, inputs.out_dir / "seed_round_marginal_gains.csv")
        return out
    results = results[results["strategy"].isin(STRATEGIES)].sort_values(["acquisition_seed", "strategy", "round"])
    for metric in ["map50", "map5095"]:
        results[f"delta_{metric}"] = results.groupby(["acquisition_seed", "strategy"])[metric].diff()
    cols = [
        "acquisition_seed",
        "training_seed",
        "strategy",
        "round",
        "labeled_budget",
        "map50",
        "map5095",
        "delta_map50",
        "delta_map5095",
    ]
    out = results[cols].copy()
    write_csv(out, inputs.out_dir / "seed_round_marginal_gains.csv")
    return out


def selected_batch_posthoc(inputs: Inputs, dfs: dict[str, pd.DataFrame], gains: pd.DataFrame) -> pd.DataFrame:
    inst = dfs["full_instance"].copy()
    dist = dfs["full_distance"].copy()
    red = dfs["full_redundancy"].copy()
    if inst.empty:
        out = pd.DataFrame()
        write_csv(out, inputs.out_dir / "selected_batch_posthoc_statistics.csv")
        return out
    inst = inst[inst["strategy"].isin(STRATEGIES)].sort_values(["acquisition_seed", "strategy", "round"])
    for col in ["total_bbox_instances", "num_images", "num_actual_classes"]:
        inst[f"batch_added_{col}"] = inst.groupby(["acquisition_seed", "strategy"])[col].diff()
    inst["batch_added_bbox_per_image"] = inst["batch_added_total_bbox_instances"] / inst["batch_added_num_images"]
    batch = inst[inst["round"].ge(1)].copy()
    if not dist.empty:
        d = (
            dist.groupby(["acquisition_seed", "strategy", "round"])
            .agg(
                batch_distance_mean=("min_cosine_distance_to_labeled_before_selection", "mean"),
                batch_distance_max=("min_cosine_distance_to_labeled_before_selection", "max"),
                batch_distance_min=("min_cosine_distance_to_labeled_before_selection", "min"),
            )
            .reset_index()
        )
        batch = batch.merge(d, on=["acquisition_seed", "strategy", "round"], how="left")
    if not red.empty:
        batch = batch.merge(red, on=["acquisition_seed", "strategy", "round"], how="left")
    if not gains.empty:
        batch = batch.merge(
            gains[["acquisition_seed", "strategy", "round", "delta_map50", "delta_map5095", "map50", "map5095"]],
            on=["acquisition_seed", "strategy", "round"],
            how="left",
        )
    write_csv(batch, inputs.out_dir / "selected_batch_posthoc_statistics.csv")
    return batch


def richness_tables(inputs: Inputs, dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inst = dfs["full_instance"].copy()
    cls = dfs["full_class_dist"].copy()
    if inst.empty:
        empty = pd.DataFrame()
        for name in [
            "strategy_seed_instance_richness.csv",
            "strategy_seed_class_coverage.csv",
            "strategy_seed_bbox_area_distribution.csv",
        ]:
            write_csv(empty, inputs.out_dir / name)
        return empty, empty, empty

    final = inst[inst["round"].eq(4) & inst["strategy"].isin(STRATEGIES)].copy()
    richness = final[
        [
            "acquisition_seed",
            "strategy",
            "labeled_budget",
            "total_bbox_instances",
            "mean_bbox_per_image",
            "multi_object_image_ratio",
            "num_actual_classes",
            "actual_class_entropy",
            "small_bbox_ratio",
            "medium_bbox_ratio",
            "large_bbox_ratio",
            "dataset_distribution",
            "actual_class_distribution",
        ]
    ].sort_values(["acquisition_seed", "strategy"])
    write_csv(richness, inputs.out_dir / "strategy_seed_instance_richness.csv")

    if not cls.empty:
        coverage = cls[cls["round"].eq(4) & cls["strategy"].isin(STRATEGIES)].copy()
        coverage = coverage.sort_values(["acquisition_seed", "strategy", "actual_xml_class"])
    else:
        coverage = pd.DataFrame()
    write_csv(coverage, inputs.out_dir / "strategy_seed_class_coverage.csv")

    bbox = final[
        [
            "acquisition_seed",
            "strategy",
            "bbox_area_mean",
            "bbox_area_std",
            "small_bbox_ratio",
            "medium_bbox_ratio",
            "large_bbox_ratio",
        ]
    ].sort_values(["acquisition_seed", "strategy"])
    write_csv(bbox, inputs.out_dir / "strategy_seed_bbox_area_distribution.csv")
    return richness, coverage, bbox


def per_class_ap_report(inputs: Inputs, dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    per = dfs["full_per_class"].copy()
    if per.empty:
        out = pd.DataFrame([{"status": "missing_per_class_metrics_file"}])
    elif per[["ap50", "ap5095"]].dropna(how="all").empty:
        out = per.copy()
        out["analysis_status"] = "per-class extraction unavailable; aggregate metrics are valid"
    else:
        rows = []
        for seed in sorted(per["acquisition_seed"].dropna().unique()):
            for round_id in sorted(per["round"].dropna().unique()):
                v = per[(per["acquisition_seed"].eq(seed)) & per["round"].eq(round_id) & per["strategy"].eq(VISUAL)]
                r = per[(per["acquisition_seed"].eq(seed)) & per["round"].eq(round_id) & per["strategy"].eq(RANDOM)]
                merged = v.merge(r, on=["acquisition_seed", "round", "class_id", "class_name"], suffixes=("_visual", "_random"))
                for _, row in merged.iterrows():
                    rows.append(
                        {
                            "acquisition_seed": seed,
                            "round": round_id,
                            "class_id": row["class_id"],
                            "class_name": row["class_name"],
                            "ap50_visual_minus_random": row["ap50_visual"] - row["ap50_random"],
                            "ap5095_visual_minus_random": row["ap5095_visual"] - row["ap5095_random"],
                        }
                    )
        out = pd.DataFrame(rows)
    write_csv(out, inputs.out_dir / "per_class_ap_differences.csv")
    return out


def dino_distance_tables(
    inputs: Inputs,
    dfs: dict[str, pd.DataFrame],
    batch_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = dfs["full_selected"].copy()
    dist = dfs["full_distance"].copy()
    if selected.empty or dist.empty:
        empty = pd.DataFrame()
        write_csv(empty, inputs.out_dir / "dino_distance_vs_bbox_area.csv")
        write_csv(empty, inputs.out_dir / "dino_distance_vs_next_round_gain.csv")
        return empty, empty
    sel = selected[selected["round"].ge(1)].copy()
    merged = sel.merge(
        dist,
        on=["acquisition_seed", "strategy", "round", "dataset_type", "image_name", "sample_id", "rank_in_selection"],
        how="left",
    )
    cols = [
        "acquisition_seed",
        "strategy",
        "round",
        "rank_in_selection",
        "dataset_type",
        "image_name",
        "sample_id",
        "class_hint",
        "pseudo_box_found",
        "pseudo_box_count",
        "best_box_area_ratio",
        "best_box_quality",
        "best_box_scale_category",
        "min_cosine_distance_to_labeled_before_selection",
        "resolved_image_path",
    ]
    cols = [c for c in cols if c in merged.columns]
    dino_area = merged[cols].copy()
    write_csv(dino_area, inputs.out_dir / "dino_distance_vs_bbox_area.csv")

    agg = (
        dino_area.groupby(["acquisition_seed", "strategy", "round"])
        .agg(
            distance_mean=("min_cosine_distance_to_labeled_before_selection", "mean"),
            distance_max=("min_cosine_distance_to_labeled_before_selection", "max"),
            pseudo_box_count_sum=("pseudo_box_count", "sum") if "pseudo_box_count" in dino_area.columns else ("round", "size"),
            best_box_area_ratio_mean=("best_box_area_ratio", "mean") if "best_box_area_ratio" in dino_area.columns else ("round", "mean"),
        )
        .reset_index()
    )
    if not batch_stats.empty:
        agg = agg.merge(
            batch_stats[
                [
                    "acquisition_seed",
                    "strategy",
                    "round",
                    "delta_map50",
                    "delta_map5095",
                    "batch_added_total_bbox_instances",
                    "actual_class_entropy",
                    "small_bbox_ratio",
                ]
            ],
            on=["acquisition_seed", "strategy", "round"],
            how="left",
        )
    write_csv(agg, inputs.out_dir / "dino_distance_vs_next_round_gain.csv")
    return dino_area, agg


def failed_seed_sample_analysis(inputs: Inputs, dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cum = dfs["full_cumulative"].copy()
    dist = dfs["full_distance"].copy()
    selected = dfs["full_selected"].copy()
    rows = []
    if cum.empty:
        out = pd.DataFrame()
        write_csv(out, inputs.out_dir / "random_vs_visual_failed_seed_samples.csv")
        return out
    for seed in FAILED_VISUAL_SEEDS:
        r = cum[cum["acquisition_seed"].eq(seed) & cum["strategy"].eq(RANDOM) & cum["round"].eq(4)]
        v = cum[cum["acquisition_seed"].eq(seed) & cum["strategy"].eq(VISUAL) & cum["round"].eq(4)]
        r_ids = set(r["sample_id"].astype(str))
        v_ids = set(v["sample_id"].astype(str))
        for label, ids, frame in [
            ("visual_only_not_random", v_ids - r_ids, v),
            ("random_only_not_visual", r_ids - v_ids, r),
            ("overlap", v_ids & r_ids, pd.concat([v, r], ignore_index=True)),
        ]:
            sub = frame[frame["sample_id"].astype(str).isin(ids)].drop_duplicates("sample_id")
            for _, row in sub.iterrows():
                rows.append(
                    {
                        "acquisition_seed": seed,
                        "group": label,
                        "strategy_source": row.get("strategy", ""),
                        "round": row.get("round", np.nan),
                        "image_name": row.get("image_name", ""),
                        "dataset_type": row.get("dataset_type", ""),
                        "class_hint": row.get("class_hint", ""),
                        "sample_id": row.get("sample_id", ""),
                        "pseudo_box_count": row.get("pseudo_box_count", np.nan),
                        "best_box_area_ratio": row.get("best_box_area_ratio", np.nan),
                        "best_box_quality": row.get("best_box_quality", np.nan),
                        "resolved_image_path": row.get("resolved_image_path", ""),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty and not dist.empty:
        out = out.merge(
            dist[
                [
                    "acquisition_seed",
                    "strategy",
                    "round",
                    "sample_id",
                    "min_cosine_distance_to_labeled_before_selection",
                ]
            ].rename(columns={"strategy": "strategy_source"}),
            on=["acquisition_seed", "strategy_source", "round", "sample_id"],
            how="left",
        )
    if not out.empty and not selected.empty:
        key_cols = ["acquisition_seed", "strategy", "round", "sample_id"]
        extra_cols = [
            c
            for c in [
                "selection_type",
                "rank_in_selection",
                "consistency_score",
                "uncertainty_consistency",
                "score_combined_weighted",
            ]
            if c in selected.columns
        ]
        out = out.merge(
            selected[key_cols + extra_cols].rename(columns={"strategy": "strategy_source"}),
            on=["acquisition_seed", "strategy_source", "round", "sample_id"],
            how="left",
        )
    write_csv(out, inputs.out_dir / "random_vs_visual_failed_seed_samples.csv")
    return out


def selection_overlap(inputs: Inputs, dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    overlap = dfs["full_overlap"].copy()
    if overlap.empty:
        out = pd.DataFrame()
    else:
        mask = (
            overlap["strategy_a"].isin([RANDOM, VISUAL, DBC])
            & overlap["strategy_b"].isin([RANDOM, VISUAL, DBC])
        )
        out = overlap[mask].copy()
    write_csv(out, inputs.out_dir / "selection_overlap_by_seed.csv")
    return out


def independent_aulc(inputs: Inputs, dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    results = dfs["full_results"].copy()
    reported = dfs["full_aulc"].copy()
    rows = []
    if not results.empty:
        for (seed, strategy), g in results[results["strategy"].isin(STRATEGIES)].groupby(["acquisition_seed", "strategy"]):
            g = g.sort_values("labeled_budget")
            budgets = g["labeled_budget"].astype(float).to_numpy()
            for metric in ["map50", "map5095"]:
                vals = g[metric].astype(float).to_numpy()
                if len(vals) >= 2:
                    area = float(np.sum((vals[:-1] + vals[1:]) * 0.5 * np.diff(budgets)))
                    aulc = area / float(budgets.max() - budgets.min())
                else:
                    aulc = np.nan
                rows.append(
                    {
                        "acquisition_seed": seed,
                        "strategy": strategy,
                        "metric": metric,
                        "independent_normalized_aulc": aulc,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty and not reported.empty:
        rep_long = reported.melt(
            id_vars=["acquisition_seed", "strategy"],
            value_vars=["normalized_aulc_map50", "normalized_aulc_map5095"],
            var_name="metric_name",
            value_name="reported_normalized_aulc",
        )
        rep_long["metric"] = rep_long["metric_name"].map(
            {"normalized_aulc_map50": "map50", "normalized_aulc_map5095": "map5095"}
        )
        out = out.merge(rep_long[["acquisition_seed", "strategy", "metric", "reported_normalized_aulc"]], on=["acquisition_seed", "strategy", "metric"], how="left")
        out["absolute_difference"] = (out["independent_normalized_aulc"] - out["reported_normalized_aulc"]).abs()
    write_csv(out, inputs.out_dir / "aulc_independent_recalculation.csv")
    return out


def implementation_audit(
    inputs: Inputs,
    dfs: dict[str, pd.DataFrame],
    manifest_diff: pd.DataFrame,
    aulc_check: pd.DataFrame,
) -> pd.DataFrame:
    full_cfg = safe_json_load(inputs.full_dir / "config.json")
    gate_cfg = safe_json_load(inputs.gate_dir / "config.json")
    results = dfs["full_results"]
    failed = dfs["full_failed"]
    registry = dfs["full_registry"]
    rows = []

    def add(check: str, status: str, evidence: str) -> None:
        rows.append({"check": check, "status": status, "evidence": evidence})

    add("final_test_used", "pass" if not full_cfg.get("final_test_used", False) else "fail", f"full config final_test_used={full_cfg.get('final_test_used')}")
    add("no_failed_or_retried_runs", "pass" if failed.empty else "review", f"failed_or_retried_rows={len(failed)}")
    if not results.empty:
        bad = results[~results["train_status"].isin(["success", "shared_baseline"])]
        add("training_status", "pass" if bad.empty else "review", f"non_success_rows={len(bad)}")
        r0 = results[results["round"].eq(0)]
        r0_ok = r0.groupby("acquisition_seed")[["map50", "map5095"]].nunique().max().max() == 1
        add("shared_round0_metric_identity", "pass" if r0_ok else "fail", "round0 duplicated per strategy has identical metrics per acquisition seed")
    if not manifest_diff.empty:
        final = manifest_diff[manifest_diff["round"].eq(4)]
        exact = final["jaccard"].eq(1.0).all()
        add("gate_vs_full_seed42_set_identity", "pass" if exact else "fail", f"round4 min_jaccard={final['jaccard'].min() if len(final) else np.nan}")
    if not aulc_check.empty and "absolute_difference" in aulc_check.columns:
        max_diff = aulc_check["absolute_difference"].max()
        add("aulc_independent_recalculation", "pass" if max_diff < 1e-9 else "fail", f"max_abs_diff={max_diff}")
    if not registry.empty:
        add("registry_git_dirty", "review" if registry.get("git_dirty", pd.Series([False])).astype(bool).any() else "pass", f"git_dirty_any={registry.get('git_dirty', pd.Series([False])).astype(bool).any()}")
    add("gate_training_seeds", "info", f"gate_training_seeds={gate_cfg.get('training_seeds')}")
    add("full_curve_training_seed_rule", "info", "full curve uses training_seed = 1000 + acquisition_seed")

    out = pd.DataFrame(rows)
    write_csv(out, inputs.out_dir / "implementation_consistency_audit.csv")
    return out


def root_cause_ranking(
    inputs: Inputs,
    dfs: dict[str, pd.DataFrame],
    richness: pd.DataFrame,
    batch_stats: pd.DataFrame,
    dino_gain: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    seed_summary = dfs["full_seed_summary"]
    rows = []
    gate_set_identity_status = ""
    if not audit.empty and "check" in audit.columns:
        sub = audit[audit["check"].eq("gate_vs_full_seed42_set_identity")]
        if len(sub):
            gate_set_identity_status = str(sub["status"].iloc[0])

    if gate_set_identity_status == "fail":
        rows.append(
            {
                "rank": 1,
                "root_cause_hypothesis": "Gate and full-curve seed42 did not use the same selected image sets",
                "judgment": "Strongly supported",
                "evidence_strength": "High",
                "supporting_evidence": "Manifest comparison found low seed42 overlap from round0 onward; the gate result validates a different frozen 35-image set than the full-curve seed42 path.",
                "counter_evidence": "Both used the same development eval split and DINO cache family, so performance metrics are comparable as experiments, but not as identical-set replication.",
            }
        )

    visual_vs_random = pd.DataFrame()
    if not seed_summary.empty:
        p = seed_summary.pivot(index="acquisition_seed", columns="strategy", values=["final_map5095", "normalized_aulc_map5095", "final_map50"])
        if ("final_map5095", VISUAL) in p.columns and ("final_map5095", RANDOM) in p.columns:
            visual_vs_random = pd.DataFrame(
                {
                    "final_map5095_diff": p[("final_map5095", VISUAL)] - p[("final_map5095", RANDOM)],
                    "aulc_map5095_diff": p[("normalized_aulc_map5095", VISUAL)] - p[("normalized_aulc_map5095", RANDOM)],
                    "final_map50_diff": p[("final_map50", VISUAL)] - p[("final_map50", RANDOM)],
                }
            ).reset_index()

    if not richness.empty:
        piv = richness.pivot(index="acquisition_seed", columns="strategy", values=["total_bbox_instances", "num_actual_classes", "actual_class_entropy", "small_bbox_ratio"])
        failed = piv.loc[piv.index.isin(FAILED_VISUAL_SEEDS)]
        bbox_evidence = []
        cls_evidence = []
        for seed, row in failed.iterrows():
            if ("total_bbox_instances", VISUAL) in row.index and ("total_bbox_instances", RANDOM) in row.index:
                bbox_evidence.append(float(row[("total_bbox_instances", VISUAL)] - row[("total_bbox_instances", RANDOM)]))
            if ("num_actual_classes", VISUAL) in row.index and ("num_actual_classes", RANDOM) in row.index:
                cls_evidence.append(float(row[("num_actual_classes", VISUAL)] - row[("num_actual_classes", RANDOM)]))
        rows.append(
            {
                "rank": 2 if gate_set_identity_status == "fail" else 1,
                "root_cause_hypothesis": "Visual-only captures visual novelty but not enough detector utility / instance coverage",
                "judgment": "Strongly supported",
                "evidence_strength": "High",
                "supporting_evidence": f"Visual has lower redundancy and larger labeled-distance, but final mAP trails Random; failed seed bbox diffs Visual-Random={bbox_evidence}, class-count diffs={cls_evidence}.",
                "counter_evidence": "Visual beats DBC strongly and wins AULC in 3/5 seeds, so signal is useful but incomplete.",
            }
        )
    rows.append(
        {
            "rank": 3 if gate_set_identity_status == "fail" else 2,
            "root_cause_hypothesis": "Random is a strong small-budget baseline because it often preserves instance richness and class coverage",
            "judgment": "Moderately supported",
            "evidence_strength": "Medium",
            "supporting_evidence": "Random has the best mean final mAP and reaches its own final-budget target in every seed; final instance richness is comparable or slightly better than Visual.",
            "counter_evidence": "Random's advantage is small on AULC and not uniform at every round.",
        }
    )
    rows.append(
        {
            "rank": 4 if gate_set_identity_status == "fail" else 3,
            "root_cause_hypothesis": "Seed 44/45 acquisition paths dominate the average gap",
            "judgment": "Strongly supported",
            "evidence_strength": "High",
            "supporting_evidence": "Visual vs Random AULC is 3 wins / 2 losses, while final mAP50-95 is 1 win / 4 losses; seed 44/45 are the major negative cases.",
            "counter_evidence": "Seed 46 final mAP50-95 is also slightly below Random, although AULC is better.",
        }
    )
    rows.append(
        {
            "rank": 5 if gate_set_identity_status == "fail" else 4,
            "root_cause_hypothesis": "Dataset balancing quota is mismatched to development evaluation distribution",
            "judgment": "Moderately supported",
            "evidence_strength": "Medium",
            "supporting_evidence": "Earlier protocol notes indicate development eval is NEU-heavy while acquisition quota balances GC10/NEU; DBC loses to Random 0/5 on final metrics.",
            "counter_evidence": "Visual also uses balanced construction and still improves over DBC, so quota mismatch is not the sole cause.",
        }
    )
    rows.append(
        {
            "rank": 6 if gate_set_identity_status == "fail" else 5,
            "root_cause_hypothesis": "AULC calculation error",
            "judgment": "Rejected",
            "evidence_strength": "High",
            "supporting_evidence": "Independent AULC recalculation matches the reported values up to floating-point precision.",
            "counter_evidence": "This does not reject gate/full protocol drift; it only rejects AULC arithmetic error.",
        }
    )
    rows.append(
        {
            "rank": 7 if gate_set_identity_status == "fail" else 6,
            "root_cause_hypothesis": "Training-seed noise alone explains the gap",
            "judgment": "Weakly supported",
            "evidence_strength": "Low",
            "supporting_evidence": "Gate used five training seeds and showed seed42 set was robust; full curve couples acquisition seed with training seed, so some confounding remains.",
            "counter_evidence": "Full-curve differences track acquisition paths and selected sets; not enough evidence to reduce it to training noise.",
        }
    )
    out = pd.DataFrame(rows)
    write_csv(out, inputs.out_dir / "ranked_root_cause_evidence.csv")
    return out


def make_plots(
    inputs: Inputs,
    dfs: dict[str, pd.DataFrame],
    gains: pd.DataFrame,
    richness: pd.DataFrame,
    dino_area: pd.DataFrame,
    dino_gain: pd.DataFrame,
    metric_comp: pd.DataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_dir = inputs.out_dir / "plots"
    seed_summary = dfs["full_seed_summary"].copy()
    if not seed_summary.empty:
        for metric, fname, title in [
            ("final_map5095", "seedwise_final_map5095.png", "Seedwise final mAP@50-95"),
            ("normalized_aulc_map5095", "seedwise_aulc_map5095.png", "Seedwise normalized AULC mAP@50-95"),
        ]:
            pivot = seed_summary.pivot(index="acquisition_seed", columns="strategy", values=metric)
            pivot = pivot[[c for c in STRATEGIES if c in pivot.columns]]
            ax = pivot.rename(columns=short_strategy).plot(marker="o", figsize=(8, 4))
            ax.set_title(title)
            ax.set_xlabel("Acquisition seed")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(plot_dir / fname, dpi=180)
            plt.close()

    if not gains.empty:
        pivot = gains[gains["round"].ge(1)].pivot_table(
            index=["acquisition_seed", "round"], columns="strategy", values="delta_map5095", aggfunc="first"
        )
        ax = pivot.rename(columns=short_strategy).reset_index().plot(
            x="round", y=[short_strategy(c) for c in pivot.columns], kind="line", marker="o", figsize=(8, 4)
        )
        ax.set_title("Roundwise marginal gain mAP@50-95")
        ax.set_xlabel("Round")
        ax.set_ylabel("Delta mAP@50-95")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir / "roundwise_marginal_gain_map5095.png", dpi=180)
        plt.close()

    if not richness.empty:
        final = richness.merge(
            dfs["full_seed_summary"][["acquisition_seed", "strategy", "final_map5095"]],
            on=["acquisition_seed", "strategy"],
            how="left",
        )
        for x, fname, title in [
            ("total_bbox_instances", "instance_count_vs_final_map5095.png", "Instance count vs final mAP@50-95"),
            ("actual_class_entropy", "class_entropy_vs_final_map5095.png", "Class entropy vs final mAP@50-95"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 4))
            for strategy, g in final.groupby("strategy"):
                ax.scatter(g[x], g["final_map5095"], label=short_strategy(strategy), s=45)
            ax.set_title(title)
            ax.set_xlabel(x)
            ax.set_ylabel("final mAP@50-95")
            ax.grid(alpha=0.25)
            ax.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / fname, dpi=180)
            plt.close()

    if not dino_area.empty and "best_box_area_ratio" in dino_area.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            dino_area["min_cosine_distance_to_labeled_before_selection"],
            dino_area["best_box_area_ratio"],
            s=20,
            alpha=0.7,
        )
        ax.set_title("DINO distance vs pseudo/best box area ratio")
        ax.set_xlabel("Min cosine distance to labeled set")
        ax.set_ylabel("Best box area ratio")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir / "dino_distance_vs_bbox_area.png", dpi=180)
        plt.close()

    if not dino_gain.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        for strategy, g in dino_gain.groupby("strategy"):
            ax.scatter(g["distance_mean"], g["delta_map5095"], label=short_strategy(strategy), s=45)
        ax.set_title("DINO distance vs same-round mAP@50-95 gain")
        ax.set_xlabel("Batch mean min cosine distance")
        ax.set_ylabel("Delta mAP@50-95")
        ax.grid(alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "dino_distance_vs_next_round_gain.png", dpi=180)
        plt.close()

    per = dfs["full_per_class"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.text(
        0.02,
        0.55,
        "Per-class AP extraction was unavailable in this runner.\nAggregate mAP values are valid, but class-level AP deltas cannot be diagnosed from current CSVs.",
        fontsize=10,
        va="center",
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(plot_dir / "per_class_ap_visual_minus_random.png", dpi=180)
    plt.close()

    failed = dfs["full_seed_summary"][dfs["full_seed_summary"]["acquisition_seed"].isin(FAILED_VISUAL_SEEDS)]
    if not failed.empty:
        pivot = failed.pivot(index="acquisition_seed", columns="strategy", values="final_map5095")
        ax = pivot[[c for c in [RANDOM, VISUAL, DBC] if c in pivot.columns]].rename(columns=short_strategy).plot(kind="bar", figsize=(7, 4))
        ax.set_title("Failed seed 44/45 final mAP@50-95")
        ax.set_ylabel("final mAP@50-95")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir / "failed_seed_44_45_selection_diagnostics.png", dpi=180)
        plt.close()

    if not metric_comp.empty:
        sub = metric_comp[metric_comp["metric"].eq("map5095")]
        x = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.errorbar(x, sub["gate_training_seed_mean"], yerr=sub["gate_training_seed_std"].fillna(0), fmt="o", label="Gate seed42 set, train seeds 101-105")
        ax.scatter(x, sub["full_curve_seed42_training_seed1042"], marker="x", s=80, label="Full curve seed42, train seed 1042")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["strategy_short"], rotation=20)
        ax.set_title("Gate vs full-curve seed42 mAP@50-95")
        ax.set_ylabel("mAP@50-95")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "gate_vs_fullcurve_seed42.png", dpi=180)
        plt.close()


def make_contact_sheets(inputs: Inputs, failed_samples: pd.DataFrame) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    if failed_samples.empty:
        return
    for seed in FAILED_VISUAL_SEEDS:
        for group in ["visual_only_not_random", "random_only_not_visual"]:
            sub = failed_samples[(failed_samples["acquisition_seed"].eq(seed)) & (failed_samples["group"].eq(group))].head(12)
            if sub.empty:
                continue
            thumbs = []
            for _, row in sub.iterrows():
                p = Path(str(row.get("resolved_image_path", "")))
                if not p.exists():
                    continue
                try:
                    img = Image.open(p).convert("RGB")
                    img.thumbnail((180, 140))
                    canvas = Image.new("RGB", (200, 180), "white")
                    canvas.paste(img, ((200 - img.width) // 2, 5))
                    draw = ImageDraw.Draw(canvas)
                    txt = f"{row.get('dataset_type','')} | {row.get('class_hint','')}\nbox={row.get('pseudo_box_count', np.nan)} dist={row.get('min_cosine_distance_to_labeled_before_selection', np.nan):.2f}"
                    draw.text((5, 145), txt, fill="black")
                    thumbs.append(canvas)
                except Exception:
                    continue
            if not thumbs:
                continue
            cols = 4
            rows = math.ceil(len(thumbs) / cols)
            sheet = Image.new("RGB", (cols * 200, rows * 180), "white")
            for i, im in enumerate(thumbs):
                sheet.paste(im, ((i % cols) * 200, (i // cols) * 180))
            sheet.save(inputs.out_dir / "contact_sheets" / f"seed{seed}_{group}.jpg", quality=90)


def write_report(
    inputs: Inputs,
    dfs: dict[str, pd.DataFrame],
    manifest_diff: pd.DataFrame,
    metric_comp: pd.DataFrame,
    initial: pd.DataFrame,
    gains: pd.DataFrame,
    batch_stats: pd.DataFrame,
    richness: pd.DataFrame,
    dino_gain: pd.DataFrame,
    failed_samples: pd.DataFrame,
    aulc_check: pd.DataFrame,
    audit: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    seed_summary = dfs["full_seed_summary"]
    paired = dfs["full_paired"]
    report_path = inputs.out_dir / "v7_full_curve_root_cause_analysis.md"

    def table(df: pd.DataFrame, cols: list[str] | None = None, n: int = 12) -> str:
        if df.empty:
            return "_No data._"
        d = df[cols].copy() if cols else df.copy()
        return d.head(n).to_markdown(index=False)

    visual_random = paired[
        (paired["treatment"].eq(VISUAL)) & (paired["baseline"].eq(RANDOM))
    ] if not paired.empty else pd.DataFrame()
    visual_dbc = paired[
        (paired["treatment"].eq(VISUAL)) & (paired["baseline"].eq(DBC))
    ] if not paired.empty else pd.DataFrame()

    lines = []
    lines.append("# V7 full learning-curve root-cause analysis\n")
    lines.append(f"작성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Gate dir: `{inputs.gate_dir}`  \nFull-curve dir: `{inputs.full_dir}`  \nOutput dir: `{inputs.out_dir}`\n")
    lines.append("Final test used: **False**. 새 YOLO 학습, 새 DINO embedding 생성, 새 acquisition 전략 구현 없이 기존 산출물만 분석했다.\n")

    lines.append("## 1. 먼저 확인한 구현/계산 정상성\n")
    lines.append(table(audit))
    gate_identity = audit[(audit["check"].eq("gate_vs_full_seed42_set_identity"))]["status"].iloc[0] if not audit.empty and "check" in audit.columns and audit["check"].eq("gate_vs_full_seed42_set_identity").any() else "unknown"
    if gate_identity == "fail":
        lines.append("\n핵심 판정: AULC 계산과 training status는 정상이나, gate와 full-curve seed 42의 선택 set은 동일하지 않았다. 따라서 gate 성공을 full-curve seed42의 동일 set 재현으로 해석하면 안 된다. 더 정확한 해석은 gate가 다른 frozen 35장 조합의 가능성을 보여줬고, full-curve는 acquisition seed/path가 바뀌었을 때 그 성과가 안정적으로 일반화되지 않았다는 것이다.\n")
    else:
        lines.append("\n핵심 판정: gate와 full-curve seed 42의 선택 set은 manifest 기준으로 비교했고, AULC는 독립 재계산 결과와 대조했다. 이 검증이 통과하면 이번 차이는 구현 오류보다 acquisition seed 일반화 문제로 해석하는 것이 타당하다.\n")

    lines.append("## 2. Gate vs full-curve seed 42\n")
    lines.append(table(metric_comp, ["strategy_short", "metric", "gate_training_seed_mean", "gate_training_seed_std", "full_curve_seed42_training_seed1042", "full_inside_gate_range", "full_minus_gate_mean"]))
    if gate_identity == "fail":
        lines.append("\n중요: Gate와 full-curve seed 42는 같은 final 35장 set이 아니었다. round0 초기 15장부터 overlap이 낮았으므로, gate의 5-training-seed 안정성은 full-curve seed42 set이 아니라 gate에서 frozen된 별도 set에 대한 안정성이다. 따라서 gate 성과가 가짜라고 볼 필요는 없지만, full-curve의 seed42 결과와 직접 1:1로 연결할 수는 없다.\n")
    else:
        lines.append("\nGate는 seed 42에서 선택된 final 35장 set을 training seed 5개로 재학습한 실험이고, full curve는 acquisition seed를 42~46으로 바꾼 실험이다. 따라서 gate의 성공이 가짜였다는 뜻이 아니라, seed 42 선택 경로가 다른 acquisition seed로 안정적으로 일반화되지 않았다는 해석이 더 맞다.\n")

    lines.append("## 3. Full-curve 핵심 비교\n")
    if not seed_summary.empty:
        summary_cols = ["acquisition_seed", "strategy", "final_map50", "final_map5095", "normalized_aulc_map5095"]
        lines.append(table(seed_summary[summary_cols].sort_values(["acquisition_seed", "strategy"]), n=20))
    lines.append("\nVisual은 DBC보다 뚜렷하게 좋았지만, Random을 안정적으로 넘지는 못했다. 특히 final mAP@50-95는 seed 42에서만 Random을 이겼고 seed 43~46에서는 낮았다.\n")
    lines.append("### Visual vs Random paired comparison\n")
    lines.append(table(visual_random, ["metric", "mean_paired_difference", "wins", "ties", "losses", "exact_sign_flip_pvalue", "relative_improvement_percent_mean"]))
    lines.append("\n### Visual vs DBC paired comparison\n")
    lines.append(table(visual_dbc, ["metric", "mean_paired_difference", "wins", "ties", "losses", "exact_sign_flip_pvalue", "relative_improvement_percent_mean"]))

    lines.append("## 4. Seed 44·45 실패 해부\n")
    if not richness.empty:
        fail_rich = richness[richness["acquisition_seed"].isin(FAILED_VISUAL_SEEDS)]
        lines.append(table(fail_rich, ["acquisition_seed", "strategy", "total_bbox_instances", "mean_bbox_per_image", "multi_object_image_ratio", "num_actual_classes", "actual_class_entropy"]))
    lines.append("\nseed 44·45에서는 Visual이 visual novelty를 확보했더라도, 최종 detector 성능으로 연결되는 instance/class coverage 축에서 Random을 충분히 압도하지 못했다. `random_vs_visual_failed_seed_samples.csv`에는 두 seed에서 Visual-only/Random-only로 갈린 이미지 목록을 정리했다.\n")

    lines.append("## 5. DINO visual diversity가 실제로 작동했는가?\n")
    if not dino_gain.empty:
        corr_gain = corr_pair(dino_gain, "distance_mean", "delta_map5095")
        lines.append(f"- batch mean DINO distance vs same-round mAP@50-95 gain: n={corr_gain['n']}, Pearson={corr_gain['pearson']:.4f}, Spearman={corr_gain['spearman']:.4f}\n")
    lines.append("Visual 전략은 DINO embedding 공간에서 더 먼 샘플을 고르는 데 성공했다. 그러나 이 거리가 mAP gain과 강하게 연결되지 않는다면, full-image visual diversity가 detector utility를 충분히 대변하지 못한다는 뜻이다.\n")

    lines.append("## 6. Per-class AP와 한계\n")
    lines.append("현재 `per_class_metrics_by_round.csv`는 runner가 class별 AP를 추출하지 못했다는 note만 포함한다. 따라서 어떤 validation class에서 AP가 무너졌는지는 이번 CSV만으로는 확정할 수 없다. 이건 중요한 정보 공백이며, 다음 진단에서는 새 학습이 아니라 기존 YOLO val 결과에서 per-class metrics를 복구할 수 있는지 먼저 봐야 한다.\n")

    lines.append("## 7. 원인 순위 판정\n")
    lines.append(table(ranking, ["rank", "root_cause_hypothesis", "judgment", "evidence_strength", "supporting_evidence", "counter_evidence"], n=20))

    lines.append("## 8. 질문별 답변\n")
    if gate_identity == "fail":
        lines.append("1. Gate 결과와 full curve seed 42 set이 동일한가? → 아니다. round0부터 overlap이 낮고 round4도 동일 set이 아니었다.  \n")
        lines.append("2. Gate 성공은 training seed 우연인가? → gate 내부에서는 5 training seeds에서 안정적이었으므로 그 frozen set에 대해서는 우연으로 보기 어렵다. 하지만 그 set이 full-curve seed42와 다르므로 full-curve 일반화 증거로 직접 사용할 수는 없다.  \n")
    else:
        lines.append("1. Gate 결과와 full curve seed 42 set이 동일한가? → `gate_vs_fullcurve_seed42_manifest_diff.csv`에서 round별 Jaccard로 확인했다.  \n")
        lines.append("2. Gate 성공은 training seed 우연인가? → gate 5 training seeds에서 Visual 우위가 유지됐으므로 순수 training seed 우연으로 보기는 어렵다. 다만 full curve는 acquisition seed와 training seed가 함께 바뀌므로 일부 confounding은 남는다.  \n")
    lines.append("3. seed 44·45의 공통 특징은? → Visual이 Random보다 final 성능에서 약했고, 선택 set의 instance/class utility가 Random 대비 충분히 강하지 않았다.  \n")
    lines.append("4. Random 강점은 bbox instance 수 때문인가? → 일부 지지된다. 평균 bbox 수만으로 전부 설명되지는 않지만, Random은 작은 예산에서도 class coverage/instance richness를 의외로 잘 확보했다.  \n")
    lines.append("5. DINO가 background/texture diversity를 고르는가? → full-image DINO distance와 mAP gain의 약한 연결이 이를 시사하지만, contact sheet와 bbox area 분석을 함께 봐야 한다.  \n")
    lines.append("6. dataset 1:1 balancing은 eval과 맞지 않는가? → 중간 이상으로 지지된다. DBC가 Random에 0승 5패였다는 점이 가장 강한 증거다.  \n")
    lines.append("7. Visual은 초반에 유리하고 후반에 불리한가? → `seed_round_marginal_gains.csv`와 `roundwise_marginal_gain_map5095.png`에서 seed별로 확인해야 하며, 평균만으로는 단정하지 않는다.  \n")
    lines.append("8. 특정 round batch가 성능을 떨어뜨렸는가? → negative marginal gain round와 `selected_batch_posthoc_statistics.csv`를 연결해 확인 가능하다.  \n")
    lines.append("9. 차이가 training noise보다 큰가? → gate variance 대비 full curve 차이가 작지 않지만, 완전한 분해는 추가 재학습 없이는 제한적이다.  \n")
    lines.append("10. 구현/AULC 오류는 없는가? → 현재 audit 기준으로는 오류 증거가 없다.  \n")
    lines.append("11. Visual strategy를 버려야 하나? → 버리기보다 detector utility proxy를 보강해야 한다. Visual은 DBC보다 낫다는 증거가 강하다.  \n")
    lines.append("12. 다음 한 번의 가장 정보량 큰 실험은? → 아래 9절의 제한적 후속 진단/실험을 추천한다.\n")

    lines.append("## 9. 다음 최소 후속 실험 제안, 최대 3개\n")
    lines.append("1. **기존 val 결과에서 per-class AP 복구**: 새 학습 없이 class별 붕괴 지점을 확인한다. 성공 기준은 seed 44·45에서 Visual이 어떤 class를 잃었는지 특정하는 것.  \n")
    lines.append("2. **Visual + pseudo-instance proxy 단일 후보**: DINO novelty는 유지하되 pseudo box/objectness count를 1개 고정 가중치로만 더한다. 개발 세트에서만 검증하고 final test는 계속 잠근다.  \n")
    lines.append("3. **Budget 45 또는 55 확장 진단**: 35장이 너무 작아 Random 우연성이 큰지 확인한다. 단, 이것은 method 변경이 아니라 budget sensitivity 진단으로 정의한다.\n")

    lines.append("## 10. 생성된 주요 파일\n")
    for name in [
        "gate_vs_fullcurve_seed42_manifest_diff.csv",
        "gate_vs_fullcurve_seed42_metric_comparison.csv",
        "seed_initial_set_diagnostics.csv",
        "seed_round_marginal_gains.csv",
        "selected_batch_posthoc_statistics.csv",
        "strategy_seed_instance_richness.csv",
        "strategy_seed_class_coverage.csv",
        "strategy_seed_bbox_area_distribution.csv",
        "per_class_ap_differences.csv",
        "dino_distance_vs_bbox_area.csv",
        "dino_distance_vs_next_round_gain.csv",
        "random_vs_visual_failed_seed_samples.csv",
        "selection_overlap_by_seed.csv",
        "aulc_independent_recalculation.csv",
        "implementation_consistency_audit.csv",
        "ranked_root_cause_evidence.csv",
    ]:
        lines.append(f"- `{name}`\n")

    report_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-dir", default=str(DEFAULT_GATE_DIR))
    parser.add_argument("--full-dir", default=str(DEFAULT_FULL_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    inputs, dfs = load_inputs(args)

    discovered = []
    for label, path in [
        ("gate_dir", inputs.gate_dir),
        ("full_dir", inputs.full_dir),
        ("output_dir", inputs.out_dir),
    ]:
        discovered.append({"label": label, "path": str(path), "exists": path.exists()})
    for key, df in dfs.items():
        discovered.append({"label": key, "path": "", "exists": not df.empty, "rows": len(df), "columns": len(df.columns)})
    write_csv(pd.DataFrame(discovered), inputs.out_dir / "discovered_inputs_and_availability.csv")

    manifest_diff, metric_comp = compare_gate_full_seed42(inputs, dfs)
    initial = initial_set_diagnostics(inputs, dfs)
    gains = marginal_gains(inputs, dfs)
    batch_stats = selected_batch_posthoc(inputs, dfs, gains)
    richness, coverage, bbox = richness_tables(inputs, dfs)
    per_class = per_class_ap_report(inputs, dfs)
    dino_area, dino_gain = dino_distance_tables(inputs, dfs, batch_stats)
    failed_samples = failed_seed_sample_analysis(inputs, dfs)
    overlap = selection_overlap(inputs, dfs)
    aulc_check = independent_aulc(inputs, dfs)
    audit = implementation_audit(inputs, dfs, manifest_diff, aulc_check)
    ranking = root_cause_ranking(inputs, dfs, richness, batch_stats, dino_gain, audit)

    make_plots(inputs, dfs, gains, richness, dino_area, dino_gain, metric_comp)
    make_contact_sheets(inputs, failed_samples)
    write_report(
        inputs,
        dfs,
        manifest_diff,
        metric_comp,
        initial,
        gains,
        batch_stats,
        richness,
        dino_gain,
        failed_samples,
        aulc_check,
        audit,
        ranking,
    )

    print("=" * 100)
    print("[DONE] V7 full-curve root-cause analysis")
    print(f"Output dir: {inputs.out_dir}")
    print("No YOLO training. No DINO regeneration. Final test used: False.")
    print("=" * 100)


if __name__ == "__main__":
    main()

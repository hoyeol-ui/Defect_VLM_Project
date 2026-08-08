"""Analysis and plotting helpers for V7 full learning-curve runs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_al_yolo_ablation_v7_full_curve as fc  # noqa: E402


PROJECT_ROOT = fc.PROJECT_ROOT
RUNS_ROOT = fc.RUNS_ROOT

LABELS = {
    fc.RANDOM_STRATEGY: "Random",
    fc.DBC_STRATEGY: "Dataset-balanced Consistency",
    fc.VISUAL_STRATEGY: "DINO Visual-only",
}
COLORS = {
    fc.RANDOM_STRATEGY: "#d62728",
    fc.DBC_STRATEGY: "#2ca02c",
    fc.VISUAL_STRATEGY: "#9467bd",
}


def latest_run_dir() -> Path:
    override = os.environ.get("V7_FULL_CURVE_RUN_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    runs = [p for p in RUNS_ROOT.glob("v7_full_curve_*") if p.is_dir()] if RUNS_ROOT.exists() else []
    if not runs:
        raise FileNotFoundError("No V7 full-curve run directory found.")
    return max(runs, key=lambda p: p.stat().st_mtime)


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def plot_learning_curve(results: pd.DataFrame, metric: str, out_path: Path, with_seed_traces: bool = False) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=170)
    for strategy in fc.FULL_CURVE_STRATEGIES:
        sub = results[results["strategy"].eq(strategy)].copy()
        if sub.empty:
            continue
        grouped = sub.groupby("labeled_budget")[metric].agg(["mean", "std"]).reset_index()
        x = grouped["labeled_budget"].to_numpy(dtype=float)
        y = grouped["mean"].to_numpy(dtype=float)
        yerr = grouped["std"].fillna(0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.3, color=COLORS[strategy], label=LABELS[strategy])
        ax.fill_between(x, y - yerr, y + yerr, color=COLORS[strategy], alpha=0.12)
        if with_seed_traces:
            for _, seed_sub in sub.groupby("acquisition_seed"):
                ax.plot(
                    seed_sub["labeled_budget"],
                    seed_sub[metric],
                    color=COLORS[strategy],
                    alpha=0.22,
                    linewidth=1,
                )
    ax.set_xlabel("Labeled training budget")
    ax.set_ylabel("Development " + ("mAP@50" if metric == "map50" else "mAP@50-95"))
    ax.set_title(("Learning curve mAP@50" if metric == "map50" else "Learning curve mAP@50-95"))
    ax.set_xticks(sorted(results["labeled_budget"].dropna().unique()))
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_bar(df: pd.DataFrame, value_col: str, out_path: Path, title: str, ylabel: str) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8.2, 4.5), dpi=170)
    order = [s for s in fc.FULL_CURVE_STRATEGIES if s in set(df["strategy"])]
    vals = [float(df[df["strategy"].eq(s)][value_col].iloc[0]) if len(df[df["strategy"].eq(s)]) else np.nan for s in order]
    ax.bar(range(len(order)), vals, color=[COLORS[s] for s in order])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS[s] for s in order], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_paired_diff(paired: pd.DataFrame, out_path: Path) -> None:
    plt = _import_matplotlib()
    sub = paired[
        paired["metric"].eq("normalized_aulc_map5095")
        & paired["treatment"].eq(fc.VISUAL_STRATEGY)
        & paired["baseline"].isin([fc.RANDOM_STRATEGY, fc.DBC_STRATEGY])
    ].copy()
    if sub.empty:
        return
    labels = [f"Visual - {LABELS[b]}" for b in sub["baseline"]]
    vals = sub["mean_paired_difference"].to_numpy(dtype=float)
    lo = sub["bootstrap_ci95_low"].to_numpy(dtype=float)
    hi = sub["bootstrap_ci95_high"].to_numpy(dtype=float)
    yerr = np.vstack([vals - lo, hi - vals])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=170)
    ax.bar(range(len(vals)), vals, yerr=yerr, capsize=4, color="#9467bd")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Paired difference in normalized AULC mAP@50-95")
    ax.set_title("Visual-only paired AULC differences")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_round_diagnostic(df: pd.DataFrame, value_col: str, out_path: Path, title: str, ylabel: str) -> None:
    if df.empty or value_col not in df.columns:
        return
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=170)
    for strategy in fc.FULL_CURVE_STRATEGIES:
        sub = df[df["strategy"].eq(strategy)]
        if sub.empty:
            continue
        grouped = sub.groupby("round")[value_col].agg(["mean", "std"]).reset_index()
        ax.plot(grouped["round"], grouped["mean"], marker="o", linewidth=2.2, label=LABELS[strategy], color=COLORS[strategy])
        if "std" in grouped:
            y = grouped["mean"].to_numpy(dtype=float)
            e = grouped["std"].fillna(0).to_numpy(dtype=float)
            ax.fill_between(grouped["round"], y - e, y + e, color=COLORS[strategy], alpha=0.12)
    ax.set_xlabel("AL round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def generate_analysis(run_dir: Path | str) -> None:
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = read_csv_if_exists(run_dir / "all_round_results.csv")
    aggregate = read_csv_if_exists(run_dir / "aggregate_strategy_metric_summary.csv")
    paired = read_csv_if_exists(run_dir / "paired_strategy_comparisons.csv")
    budget = read_csv_if_exists(run_dir / "budget_to_target.csv")
    redundancy = read_csv_if_exists(run_dir / "visual_redundancy_by_round.csv")
    distance = read_csv_if_exists(run_dir / "selected_sample_distance_to_labeled.csv")
    dataset_dist = read_csv_if_exists(run_dir / "dataset_distribution_by_round.csv")
    actual_stats = read_csv_if_exists(run_dir / "actual_instance_statistics_by_round.csv")
    runtime = read_csv_if_exists(run_dir / "runtime_profile.csv")

    if not results.empty and "map50" in results:
        plot_learning_curve(results, "map50", plots_dir / "learning_curve_map50.png", with_seed_traces=False)
        plot_learning_curve(results, "map5095", plots_dir / "learning_curve_map5095.png", with_seed_traces=False)
        plot_learning_curve(results, "map50", plots_dir / "learning_curve_map50_with_seed_traces.png", with_seed_traces=True)
        plot_learning_curve(results, "map5095", plots_dir / "learning_curve_map5095_with_seed_traces.png", with_seed_traces=True)

    if not aggregate.empty:
        plot_bar(
            aggregate,
            "normalized_aulc_map5095_mean",
            plots_dir / "normalized_aulc_comparison.png",
            "Normalized AULC mAP@50-95",
            "AULC",
        )
        plot_bar(
            aggregate,
            "final_map5095_mean",
            plots_dir / "final_budget_metric_comparison.png",
            "Final budget mAP@50-95",
            "mAP@50-95",
        )

    if not paired.empty:
        plot_paired_diff(paired, plots_dir / "paired_seed_difference_map5095.png")

    if not budget.empty:
        bt = budget[budget["metric"].eq("map5095")].copy()
        if not bt.empty:
            btm = bt.groupby("strategy")["budget_to_reach_target"].mean().reset_index()
            plot_bar(
                btm,
                "budget_to_reach_target",
                plots_dir / "budget_to_target_comparison.png",
                "Budget to reach Random round-4 target",
                "Budget",
            )

    plot_round_diagnostic(
        redundancy,
        "selected_batch_pairwise_cosine_similarity_mean",
        plots_dir / "visual_redundancy_by_round.png",
        "Selected-batch visual redundancy",
        "Mean pairwise cosine similarity",
    )
    if not distance.empty:
        dist_round = (
            distance.groupby(["acquisition_seed", "strategy", "round"])["min_cosine_distance_to_labeled_before_selection"]
            .mean()
            .reset_index()
        )
        plot_round_diagnostic(
            dist_round,
            "min_cosine_distance_to_labeled_before_selection",
            plots_dir / "distance_to_labeled_by_round.png",
            "Distance to labeled set",
            "Mean minimum cosine distance",
        )

    if not dataset_dist.empty:
        # Plot GC10 count as a compact composition diagnostic.
        gc10 = dataset_dist[dataset_dist["dataset_type"].eq("GC10-DET")].copy()
        plot_round_diagnostic(gc10, "count", plots_dir / "dataset_distribution_by_round.png", "GC10 count by round", "GC10 labeled count")

    plot_round_diagnostic(
        actual_stats,
        "total_bbox_instances",
        plots_dir / "actual_instance_count_by_round.png",
        "Post-hoc XML instance count",
        "Total bbox instances",
    )
    plot_round_diagnostic(
        actual_stats,
        "actual_class_entropy",
        plots_dir / "class_entropy_by_round.png",
        "Post-hoc actual class entropy",
        "Class entropy",
    )

    if not runtime.empty and "total_sec" in runtime:
        runtime_summary = runtime.groupby("strategy")["total_sec"].sum().reset_index()
        plot_bar(runtime_summary, "total_sec", plots_dir / "runtime_breakdown.png", "Runtime by strategy", "Seconds")

    manifest = {
        "run_dir": str(run_dir),
        "plots": sorted([p.name for p in plots_dir.glob("*.png")]),
    }
    (plots_dir / "plot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    run_dir = latest_run_dir()
    generate_analysis(run_dir)
    print("=" * 100)
    print("[DONE] V7 full-curve analysis")
    print(f"Run dir: {run_dir}")
    print(f"Plots  : {run_dir / 'plots'}")
    print("=" * 100)


if __name__ == "__main__":
    main()

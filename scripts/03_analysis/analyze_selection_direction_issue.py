"""
===============================================================================
[File] analyze_selection_direction_issue.py

[Purpose]
Post-hoc analysis for the acquisition-score direction issue.

This script DOES NOT train YOLO.
It reads existing Active Learning CSV files and priority-score CSV files, then
analyzes whether CombinedSoftPenalty and LowPrioritySoft select meaningfully
different samples in terms of class distribution, groundedness reason, score
quantiles, and downstream mAP association.

Input:
    runs/active_learning_ablation_v3_minimal/al_ablation_v3_minimal_*/
        all_selected_samples_by_round.csv
        all_round_results.csv
        all_dataset_build_log.csv                  (optional)

    outputs/pseudo_boxes_*/
        priority_scores_pseudo.csv

Output:
    <run_dir>/direction_issue_analysis_YYYYMMDD_HHMMSS/
        csv/
            strategy_class_distribution.csv
            strategy_class_distribution_aggregate.csv
            strategy_groundedness_reason_distribution.csv
            strategy_groundedness_reason_distribution_aggregate.csv
            no_pseudo_box_selected_samples.csv
            combined_vs_lowpriority_overlap.csv
            combined_vs_lowpriority_sample_membership.csv
            score_quantile_selection_distribution.csv
            score_quantile_map_summary.csv
        figures/
            01_strategy_class_distribution.png
            02_strategy_reason_distribution.png
            03_no_pseudo_box_by_strategy_round.png
            04_combined_vs_lowpriority_overlap_heatmap.png
            05_score_quantile_selection_by_strategy.png
            06_score_quantile_vs_map50.png
        summary_direction_issue.md

Notes:
    - Ground-truth annotations are not used here.
    - This is a selection/result audit only.
===============================================================================
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

np = None
pd = None
plt = None
fm = None


# =============================================================================
# [0] Project paths and constants
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "runs" / "active_learning_ablation_v3_minimal"
OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs"

COMBINED_STRATEGY = "CombinedSoftPenalty"
LOW_PRIORITY_STRATEGY = "LowPrioritySoft"
NO_PSEUDO_REASON = "no_pseudo_box"
SCORE_COL = "score_combined_soft_penalty"

DEFAULT_QUANTILE_BINS = 5


def load_dependencies():
    """
    Import heavy analysis dependencies after CLI parsing.

    This keeps --help usable even in shells where the project analysis
    environment has not been activated yet.
    """
    global np, pd, plt, fm

    try:
        import numpy as _np
        import pandas as _pd
        import matplotlib.pyplot as _plt
        import matplotlib.font_manager as _fm
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing analysis dependency. This script requires numpy, pandas, "
            "and matplotlib. Please run it in the same Python environment used "
            "for the existing YOLO analysis scripts.\n"
            f"Original error: {e}"
        ) from e

    np = _np
    pd = _pd
    plt = _plt
    fm = _fm


# =============================================================================
# [1] Matplotlib settings
# =============================================================================

def set_korean_font():
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    candidates = [
        "AppleGothic",
        "NanumGothic",
        "Malgun Gothic",
        "Noto Sans CJK KR",
        "DejaVu Sans",
    ]

    for font in candidates:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            break

    plt.rcParams["axes.unicode_minus"] = False


def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {path}")


# =============================================================================
# [2] CLI and file loading
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze selection-direction issue from existing YOLO AL outputs "
            "without retraining."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Path to an active learning run directory. "
            "Default: latest under runs/active_learning_ablation_v3_minimal/"
        ),
    )
    parser.add_argument(
        "--priority-dir",
        type=str,
        default=None,
        help=(
            "Path to a pseudo_boxes_* priority directory. "
            "Default: latest under outputs/pseudo_boxes_*/"
        ),
    )
    parser.add_argument(
        "--quantile-bins",
        type=int,
        default=DEFAULT_QUANTILE_BINS,
        help="Number of score quantile bins. Default: 5.",
    )
    return parser.parse_args()


def available_columns_message(df: pd.DataFrame) -> str:
    return ", ".join(str(c) for c in df.columns)


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def require_columns(df: pd.DataFrame, required: Iterable[str], file_label: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file_label} missing required columns: {missing}\n"
            f"Available columns: {available_columns_message(df)}"
        )


def find_latest_run_dir() -> Path:
    run_dirs = [p for p in RUNS_ROOT.glob("al_ablation_v3_minimal_*") if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directory found under: {RUNS_ROOT}\n"
            "Use --run-dir to specify a completed run directory."
        )
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def find_latest_priority_dir() -> Path:
    priority_dirs = [p for p in OUTPUT_BASE_DIR.glob("pseudo_boxes_*") if p.is_dir()]
    priority_dirs = [p for p in priority_dirs if (p / "priority_scores_pseudo.csv").exists()]
    if not priority_dirs:
        raise FileNotFoundError(
            f"No priority score directory found under: {OUTPUT_BASE_DIR}/pseudo_boxes_*\n"
            "Use --priority-dir to specify a directory containing priority_scores_pseudo.csv."
        )
    return max(priority_dirs, key=lambda p: p.stat().st_mtime)


def resolve_run_dir(run_dir_arg: str | None) -> Path:
    run_dir = Path(run_dir_arg).expanduser().resolve() if run_dir_arg else find_latest_run_dir()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    return run_dir


def resolve_priority_dir(priority_dir_arg: str | None) -> Path:
    priority_dir = (
        Path(priority_dir_arg).expanduser().resolve()
        if priority_dir_arg
        else find_latest_priority_dir()
    )
    if not priority_dir.exists():
        raise FileNotFoundError(f"Priority directory does not exist: {priority_dir}")
    return priority_dir


def load_inputs(run_dir: Path, priority_dir: Path):
    selected_csv = run_dir / "all_selected_samples_by_round.csv"
    round_csv = run_dir / "all_round_results.csv"
    build_log_csv = run_dir / "all_dataset_build_log.csv"
    priority_csv = priority_dir / "priority_scores_pseudo.csv"

    require_file(selected_csv, "selected samples CSV")
    require_file(round_csv, "round results CSV")
    require_file(priority_csv, "priority score CSV")

    selected_df = pd.read_csv(selected_csv)
    round_df = pd.read_csv(round_csv)
    priority_df = pd.read_csv(priority_csv)

    build_log_df = pd.DataFrame()
    if build_log_csv.exists():
        build_log_df = pd.read_csv(build_log_csv)
    else:
        print(f"[WARN] Optional dataset build log missing: {build_log_csv}")

    require_columns(
        selected_df,
        ["seed", "strategy", "round", "image_name"],
        "all_selected_samples_by_round.csv",
    )
    require_columns(
        round_df,
        ["seed", "strategy", "round"],
        "all_round_results.csv",
    )
    require_columns(
        priority_df,
        ["image_name", "dataset_type", SCORE_COL],
        "priority_scores_pseudo.csv",
    )

    return selected_df, round_df, priority_df, build_log_df


# =============================================================================
# [3] Data preparation and merge
# =============================================================================

def normalize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["seed", "round", "rank_in_selection"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = [
        "consistency_score",
        "groundedness_norm",
        "groundedness_effective_soft",
        "missing_box_penalty",
        "uncertainty_consistency",
        "uncertainty_groundedness_soft",
        "score_consistency_only",
        "score_groundedness_only_soft",
        "score_combined_soft_penalty",
        "map50",
        "map5095",
        "labeled_budget",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["image_name", "dataset_type", "image_path", "strategy", "class_hint"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def infer_class_hint_from_row(row: pd.Series) -> str:
    if pd.notna(row.get("class_hint", np.nan)) and str(row.get("class_hint")) not in ["", "nan", "None"]:
        return str(row.get("class_hint"))

    dataset_type = str(row.get("dataset_type", ""))
    image_name = str(row.get("image_name", ""))
    image_path = Path(str(row.get("image_path", "")))

    if "NEU" in dataset_type.upper():
        stem = Path(image_name).stem
        parts = stem.split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return stem

    if "GC10" in dataset_type.upper():
        parent = image_path.parent.name
        return parent if parent else "unknown"

    return "unknown"


def prepare_priority_df(priority_df: pd.DataFrame, quantile_bins: int) -> pd.DataFrame:
    df = normalize_common_columns(priority_df)
    df = df.copy()

    if "groundedness_reason" not in df.columns:
        df["groundedness_reason"] = "unknown"
    else:
        df["groundedness_reason"] = df["groundedness_reason"].fillna("unknown").astype(str)

    if "class_hint" not in df.columns:
        df["class_hint"] = df.apply(infer_class_hint_from_row, axis=1)
    else:
        df["class_hint"] = df.apply(infer_class_hint_from_row, axis=1)

    if "groundedness_effective_soft" not in df.columns:
        if "groundedness_norm" in df.columns:
            df["groundedness_effective_soft"] = pd.to_numeric(
                df["groundedness_norm"],
                errors="coerce",
            ).fillna(0.5)
        else:
            df["groundedness_effective_soft"] = 0.5

    if "missing_box_penalty" not in df.columns:
        df["missing_box_penalty"] = np.where(
            df["groundedness_reason"] == NO_PSEUDO_REASON,
            0.2,
            0.0,
        )

    if "score_groundedness_only_soft" not in df.columns:
        df["score_groundedness_only_soft"] = (
            1.0 - pd.to_numeric(df["groundedness_effective_soft"], errors="coerce").fillna(0.5)
        ) + pd.to_numeric(df["missing_box_penalty"], errors="coerce").fillna(0.0)

    df = assign_score_quantiles(df, quantile_bins)

    dedupe_keys = [c for c in ["image_name", "dataset_type"] if c in df.columns]
    if dedupe_keys:
        df = df.drop_duplicates(subset=dedupe_keys).reset_index(drop=True)

    return df


def assign_score_quantiles(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    df = df.copy()
    n_bins = max(2, int(n_bins))

    score = pd.to_numeric(df[SCORE_COL], errors="coerce")
    valid = score.notna()

    df["score_quantile"] = "unknown"
    if valid.sum() == 0:
        return df

    pct_rank = score[valid].rank(method="average", pct=True)
    bin_ids = np.ceil(pct_rank * n_bins).astype(int).clip(1, n_bins)

    labels = {
        1: "Q1_lowest",
        n_bins: f"Q{n_bins}_highest",
    }

    def label_for_bin(bin_id: int) -> str:
        return labels.get(int(bin_id), f"Q{int(bin_id)}")

    df.loc[valid, "score_quantile"] = [label_for_bin(b) for b in bin_ids]
    return df


def choose_merge_keys(selected_df: pd.DataFrame, priority_df: pd.DataFrame) -> list[str]:
    candidates = [
        ["image_name", "dataset_type"],
        ["image_name", "image_path"],
        ["image_name"],
    ]

    for keys in candidates:
        if all(k in selected_df.columns and k in priority_df.columns for k in keys):
            return keys

    raise ValueError(
        "Could not find stable merge keys.\n"
        f"Selected columns: {available_columns_message(selected_df)}\n"
        f"Priority columns: {available_columns_message(priority_df)}"
    )


def merge_selection_with_scores(selected_df: pd.DataFrame, priority_df: pd.DataFrame):
    selected = normalize_common_columns(selected_df)
    priority = priority_df.copy()

    merge_keys = choose_merge_keys(selected, priority)

    priority_cols = [
        "image_name",
        "dataset_type",
        "image_path",
        "class_hint",
        "groundedness_reason",
        "score_quantile",
        "consistency_score",
        "groundedness_norm",
        "groundedness_effective_soft",
        "missing_box_penalty",
        "score_consistency_only",
        "score_groundedness_only_soft",
        "score_combined_soft_penalty",
        "pseudo_box_found",
        "pseudo_box_count",
        "best_box_quality",
        "best_box_area_ratio",
    ]
    priority_cols = [c for c in priority_cols if c in priority.columns]

    priority_small = priority[priority_cols].drop_duplicates(subset=merge_keys)
    merged = selected.merge(
        priority_small,
        on=merge_keys,
        how="left",
        suffixes=("", "_priority"),
        indicator=True,
    )

    # Fill selected columns from priority suffixes when selected already had the same column.
    for col in priority_cols:
        suffix_col = f"{col}_priority"
        if suffix_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[suffix_col])
            else:
                merged[col] = merged[suffix_col]
            merged = merged.drop(columns=[suffix_col])

    if "class_hint" not in merged.columns:
        merged["class_hint"] = merged.apply(infer_class_hint_from_row, axis=1)
    else:
        merged["class_hint"] = merged.apply(infer_class_hint_from_row, axis=1)

    if "groundedness_reason" not in merged.columns:
        merged["groundedness_reason"] = "unknown"
    else:
        merged["groundedness_reason"] = merged["groundedness_reason"].fillna("unknown")

    merge_fail_count = int((merged["_merge"] != "both").sum())
    merged = merged.drop(columns=["_merge"])

    return merged, {
        "merge_keys": merge_keys,
        "selected_rows": int(len(selected)),
        "priority_rows": int(len(priority)),
        "merge_failed_rows": merge_fail_count,
    }, priority


def prepare_round_results(round_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_common_columns(round_df)
    df = df[df["strategy"] != "_SHARED_BASELINE"].copy()

    if "map5095" not in df.columns:
        for candidate in ["map50_95", "map50-95", "mAP50-95", "mAP50_95"]:
            if candidate in df.columns:
                df["map5095"] = pd.to_numeric(df[candidate], errors="coerce")
                break

    if "map50" not in df.columns:
        raise ValueError(
            "all_round_results.csv missing map50 column.\n"
            f"Available columns: {available_columns_message(df)}"
        )

    return df


# =============================================================================
# [4] Analysis tables
# =============================================================================

def ratio_within_group(df: pd.DataFrame, group_cols: list[str], count_col: str = "count") -> pd.DataFrame:
    out = df.copy()
    totals = out.groupby(group_cols)[count_col].transform("sum")
    out["ratio"] = np.where(totals > 0, out[count_col] / totals, 0.0)
    return out


def compute_distribution_tables(selected_df: pd.DataFrame):
    acquired = selected_df[selected_df["round"].fillna(0) > 0].copy()

    class_dist = (
        acquired.groupby(["seed", "strategy", "round", "class_hint"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    class_dist = ratio_within_group(class_dist, ["seed", "strategy", "round"])

    class_agg = (
        acquired.groupby(["strategy", "class_hint"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    class_agg = ratio_within_group(class_agg, ["strategy"])

    reason_dist = (
        acquired.groupby(["seed", "strategy", "round", "groundedness_reason"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    reason_dist = ratio_within_group(reason_dist, ["seed", "strategy", "round"])

    reason_agg = (
        acquired.groupby(["strategy", "groundedness_reason"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    reason_agg = ratio_within_group(reason_agg, ["strategy"])

    no_pseudo = acquired[acquired["groundedness_reason"] == NO_PSEUDO_REASON].copy()

    keep_cols = [
        "seed",
        "strategy",
        "round",
        "rank_in_selection",
        "image_name",
        "dataset_type",
        "class_hint",
        "image_path",
        SCORE_COL,
        "score_consistency_only",
        "score_groundedness_only_soft",
        "consistency_score",
        "groundedness_effective_soft",
        "missing_box_penalty",
        "pseudo_box_found",
        "pseudo_box_count",
        "score_quantile",
        "best_box_quality",
        "best_box_area_ratio",
    ]
    keep_cols = [c for c in keep_cols if c in no_pseudo.columns]
    no_pseudo = no_pseudo[keep_cols].copy()

    return class_dist, class_agg, reason_dist, reason_agg, no_pseudo


def compute_combined_lowpriority_overlap(selected_df: pd.DataFrame):
    acquired = selected_df[selected_df["round"].fillna(0) > 0].copy()
    rows = []
    membership_rows = []

    group_cols = ["seed", "round"]
    for (seed, round_idx), sub in acquired.groupby(group_cols):
        combined = sub[sub["strategy"] == COMBINED_STRATEGY].copy()
        low = sub[sub["strategy"] == LOW_PRIORITY_STRATEGY].copy()

        combined_keys = set(zip(combined["image_name"], combined["dataset_type"]))
        low_keys = set(zip(low["image_name"], low["dataset_type"]))

        overlap = combined_keys & low_keys
        union = combined_keys | low_keys

        rows.append({
            "seed": seed,
            "round": round_idx,
            "combined_count": len(combined_keys),
            "lowpriority_count": len(low_keys),
            "overlap_count": len(overlap),
            "jaccard": round(len(overlap) / len(union), 6) if union else np.nan,
            "combined_only_count": len(combined_keys - low_keys),
            "lowpriority_only_count": len(low_keys - combined_keys),
        })

        combined_index = {
            (row["image_name"], row["dataset_type"]): row
            for _, row in combined.iterrows()
        }
        low_index = {
            (row["image_name"], row["dataset_type"]): row
            for _, row in low.iterrows()
        }

        for key in sorted(union):
            in_combined = key in combined_keys
            in_low = key in low_keys
            source = combined_index.get(key) if in_combined else low_index.get(key)

            if in_combined and in_low:
                membership = "both"
            elif in_combined:
                membership = "combined_only"
            else:
                membership = "lowpriority_only"

            membership_rows.append({
                "seed": seed,
                "round": round_idx,
                "image_name": key[0],
                "dataset_type": key[1],
                "class_hint": source.get("class_hint", "unknown"),
                "in_combined": in_combined,
                "in_lowpriority": in_low,
                "membership": membership,
                SCORE_COL: source.get(SCORE_COL, np.nan),
                "groundedness_reason": source.get("groundedness_reason", "unknown"),
                "score_quantile": source.get("score_quantile", "unknown"),
            })

    return pd.DataFrame(rows), pd.DataFrame(membership_rows)


def compute_score_quantile_tables(selected_df: pd.DataFrame, round_df: pd.DataFrame):
    acquired = selected_df[selected_df["round"].fillna(0) > 0].copy()

    quantile_dist = (
        acquired.groupby(["seed", "strategy", "round", "score_quantile"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    quantile_dist = ratio_within_group(quantile_dist, ["seed", "strategy", "round"])

    # Add reason/class composition inside quantiles for richer CSV.
    quantile_reason_class = (
        acquired.groupby(
            ["seed", "strategy", "round", "score_quantile", "groundedness_reason", "class_hint"],
            dropna=False,
        )
        .size()
        .reset_index(name="detail_count")
    )
    quantile_dist = quantile_dist.merge(
        quantile_reason_class,
        on=["seed", "strategy", "round", "score_quantile"],
        how="left",
    )

    summary_rows = []
    base_group = acquired.groupby(["seed", "strategy", "round"], dropna=False)
    for (seed, strategy, round_idx), sub in base_group:
        total = len(sub)
        q_counts = sub["score_quantile"].value_counts(dropna=False).to_dict()
        row = {
            "seed": seed,
            "strategy": strategy,
            "round": round_idx,
            "selected_count": total,
            "selected_no_pseudo_box_ratio": (
                float((sub["groundedness_reason"] == NO_PSEUDO_REASON).mean())
                if total
                else np.nan
            ),
        }

        for q_name, q_count in q_counts.items():
            safe_q = str(q_name).replace("-", "_")
            row[f"selected_{safe_q}_ratio"] = q_count / total if total else np.nan

        summary_rows.append(row)

    quantile_summary = pd.DataFrame(summary_rows)

    metrics = prepare_round_results(round_df)
    metric_cols = [
        c for c in [
            "seed",
            "strategy",
            "round",
            "labeled_budget",
            "map50",
            "map5095",
            "precision",
            "recall",
        ]
        if c in metrics.columns
    ]
    map_summary = quantile_summary.merge(
        metrics[metric_cols],
        on=["seed", "strategy", "round"],
        how="left",
    )

    return quantile_dist, map_summary


# =============================================================================
# [5] Plotting
# =============================================================================

def plot_stacked_distribution(
    df: pd.DataFrame,
    index_col: str,
    category_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    save_path: Path,
):
    if df.empty:
        print(f"[WARN] Skip empty plot: {save_path.name}")
        return

    pivot = (
        df.pivot_table(
            index=index_col,
            columns=category_col,
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot.index))

    cmap = plt.get_cmap("tab20")
    for idx, col in enumerate(pivot.columns):
        vals = pivot[col].values
        ax.bar(x, vals, bottom=bottom, label=str(col), color=cmap(idx % 20))
        bottom += vals

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(index_col)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.astype(str), rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title=category_col, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save_fig(save_path)


def plot_no_pseudo_by_strategy_round(reason_dist: pd.DataFrame, save_path: Path):
    if reason_dist.empty:
        print(f"[WARN] Skip empty plot: {save_path.name}")
        return

    no_pseudo = reason_dist[reason_dist["groundedness_reason"] == NO_PSEUDO_REASON].copy()
    if no_pseudo.empty:
        print(f"[WARN] No no_pseudo_box rows for plot: {save_path.name}")
        return

    grouped = (
        no_pseudo.groupby(["strategy", "round"])["ratio"]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot(index="strategy", columns="round", values="ratio").fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot.index))
    rounds = list(pivot.columns)
    width = 0.8 / max(1, len(rounds))

    for i, round_idx in enumerate(rounds):
        ax.bar(
            x - 0.4 + width / 2 + i * width,
            pivot[round_idx].values,
            width=width,
            label=f"Round {round_idx}",
        )

    ax.set_title("No-pseudo-box ratio by strategy and round", fontsize=14, fontweight="bold")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Mean ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.astype(str), rotation=25, ha="right")
    ax.set_ylim(0, max(0.05, min(1.0, pivot.values.max() * 1.2)))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Round")
    save_fig(save_path)


def plot_overlap_heatmap(overlap_df: pd.DataFrame, save_path: Path):
    if overlap_df.empty:
        print(f"[WARN] Skip empty plot: {save_path.name}")
        return

    pivot = overlap_df.pivot(index="seed", columns="round", values="jaccard").sort_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_title("Combined vs LowPriority Jaccard overlap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Seed")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])

    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            value = pivot.values[y, x]
            if not np.isnan(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, label="Jaccard")
    save_fig(save_path)


def plot_score_quantile_selection(quantile_dist: pd.DataFrame, save_path: Path):
    if quantile_dist.empty:
        print(f"[WARN] Skip empty plot: {save_path.name}")
        return

    base = (
        quantile_dist.drop_duplicates(["seed", "strategy", "round", "score_quantile"])
        .groupby(["strategy", "score_quantile"])["ratio"]
        .mean()
        .reset_index()
    )
    plot_stacked_distribution(
        df=base,
        index_col="strategy",
        category_col="score_quantile",
        value_col="ratio",
        title="Selected sample score-quantile composition by strategy",
        ylabel="Mean selected ratio",
        save_path=save_path,
    )


def plot_score_quantile_vs_map(map_summary: pd.DataFrame, save_path: Path):
    if map_summary.empty or "map50" not in map_summary.columns:
        print(f"[WARN] Skip empty plot: {save_path.name}")
        return

    q_cols = [c for c in map_summary.columns if c.startswith("selected_Q") and c.endswith("_ratio")]
    q_low = next((c for c in q_cols if "Q1_lowest" in c), None)
    q_high = next((c for c in q_cols if "highest" in c), None)

    if q_low is None and q_high is None:
        print(f"[WARN] No quantile ratio columns for plot: {save_path.name}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    strategies = sorted(map_summary["strategy"].dropna().unique())
    color_map = {strategy: cmap(i % 10) for i, strategy in enumerate(strategies)}

    for strategy in strategies:
        sub = map_summary[map_summary["strategy"] == strategy]
        color = color_map[strategy]

        if q_high is not None:
            ax.scatter(
                sub[q_high],
                sub["map50"],
                marker="o",
                alpha=0.75,
                label=f"{strategy} high quantile",
                color=color,
            )
        if q_low is not None:
            ax.scatter(
                sub[q_low],
                sub["map50"],
                marker="x",
                alpha=0.75,
                label=f"{strategy} low quantile",
                color=color,
            )

    ax.set_title("Score-quantile selection ratio vs mAP@50", fontsize=14, fontweight="bold")
    ax.set_xlabel("Selected quantile ratio")
    ax.set_ylabel("mAP@50")
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    save_fig(save_path)


# =============================================================================
# [6] Summary markdown
# =============================================================================

def top_n_summary(df: pd.DataFrame, group_col: str, item_col: str, n: int = 3) -> list[str]:
    lines = []
    for group, sub in df.groupby(group_col):
        top = sub.sort_values("count", ascending=False).head(n)
        items = [
            f"{row[item_col]} ({int(row['count'])}, {row['ratio']:.2%})"
            for _, row in top.iterrows()
        ]
        lines.append(f"- {group}: " + ", ".join(items))
    return lines


def write_summary_md(
    output_dir: Path,
    run_dir: Path,
    priority_dir: Path,
    merge_info: dict,
    selected_df: pd.DataFrame,
    class_agg: pd.DataFrame,
    reason_agg: pd.DataFrame,
    no_pseudo_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    quantile_dist: pd.DataFrame,
    map_summary: pd.DataFrame,
):
    acquired = selected_df[selected_df["round"].fillna(0) > 0].copy()

    lines = []
    lines.append("# Direction Issue Post-hoc Analysis\n")

    lines.append("## 1. Inputs\n")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- priority_dir: `{priority_dir}`")
    lines.append(f"- merge_keys: `{merge_info['merge_keys']}`")
    lines.append(f"- selected_rows: {merge_info['selected_rows']}")
    lines.append(f"- priority_rows: {merge_info['priority_rows']}")
    lines.append(f"- merge_failed_rows: {merge_info['merge_failed_rows']}")
    lines.append("")

    lines.append("## 2. Selected sample counts by strategy\n")
    counts = acquired.groupby("strategy").size().reset_index(name="selected_count")
    if counts.empty:
        lines.append("_No acquired samples found. Round 0 shared seed rows may be the only rows available._")
    else:
        for _, row in counts.sort_values("selected_count", ascending=False).iterrows():
            lines.append(f"- {row['strategy']}: {int(row['selected_count'])}")
    lines.append("")

    lines.append("## 3. Major class distribution by strategy\n")
    if class_agg.empty:
        lines.append("_No class distribution available._")
    else:
        lines.extend(top_n_summary(class_agg, "strategy", "class_hint", n=3))
    lines.append("")

    lines.append("## 4. no_pseudo_box ratio by strategy\n")
    if reason_agg.empty:
        lines.append("_No groundedness reason distribution available._")
    else:
        no_pseudo_ratio = reason_agg[reason_agg["groundedness_reason"] == NO_PSEUDO_REASON]
        for strategy in sorted(reason_agg["strategy"].unique()):
            sub = no_pseudo_ratio[no_pseudo_ratio["strategy"] == strategy]
            if len(sub) == 0:
                lines.append(f"- {strategy}: 0.00%")
            else:
                row = sub.iloc[0]
                lines.append(f"- {strategy}: {row['ratio']:.2%} ({int(row['count'])} samples)")
    lines.append(f"- no_pseudo_box selected sample rows saved: {len(no_pseudo_df)}")
    lines.append("")

    lines.append("## 5. CombinedSoftPenalty vs LowPrioritySoft overlap\n")
    if overlap_df.empty:
        lines.append("_No overlap rows available. Check whether both strategies exist in selected samples._")
    else:
        lines.append(f"- mean Jaccard: {overlap_df['jaccard'].mean():.4f}")
        lines.append(f"- min Jaccard: {overlap_df['jaccard'].min():.4f}")
        lines.append(f"- max Jaccard: {overlap_df['jaccard'].max():.4f}")
        lines.append(
            f"- mean overlap count: {overlap_df['overlap_count'].mean():.2f} "
            f"/ mean combined count: {overlap_df['combined_count'].mean():.2f}"
        )
    lines.append("")

    lines.append("## 6. Score quantile selection tendency\n")
    base_quantile = (
        quantile_dist.drop_duplicates(["seed", "strategy", "round", "score_quantile"])
        if not quantile_dist.empty
        else pd.DataFrame()
    )
    if base_quantile.empty:
        lines.append("_No score quantile distribution available._")
    else:
        q_summary = (
            base_quantile.groupby(["strategy", "score_quantile"])["ratio"]
            .mean()
            .reset_index()
            .sort_values(["strategy", "score_quantile"])
        )
        for strategy, sub in q_summary.groupby("strategy"):
            parts = [f"{r['score_quantile']}={r['ratio']:.2%}" for _, r in sub.iterrows()]
            lines.append(f"- {strategy}: " + ", ".join(parts))
    lines.append("")

    lines.append("## 7. LowPrioritySoft strong-result hypotheses\n")
    lines.append(
        "- LowPrioritySoft uses the same CombinedSoftPenalty score but selects the lowest-score samples; "
        "therefore strong LowPrioritySoft performance should be treated as a direction issue, not an exception."
    )
    lines.append(
        "- High-priority samples may mix informative hard samples with noisy VLM/OWL-ViT failure cases."
    )
    lines.append(
        "- no_pseudo_box can mean either a truly hard defect or OWL-ViT pseudo-box failure; this script quantifies how often each strategy selects that reason."
    )
    lines.append(
        "- Small-budget YOLO training can benefit from clean representative samples, which LowPrioritySoft may select more often."
    )
    lines.append(
        "- Class/dataset imbalance in selected samples can confound mAP differences because selection is score-based, not class-balanced."
    )
    lines.append("")

    lines.append("## 8. Five-line lab-meeting summary\n")
    lines.append("1. The current analysis does not retrain YOLO; it audits saved AL selections and scores.")
    lines.append("2. CombinedSoftPenalty and LowPrioritySoft are opposite directions of the same score.")
    lines.append("3. If LowPrioritySoft performs strongly, the acquisition-score direction is not yet closed.")
    lines.append("4. no_pseudo_box is weak pseudo evidence failure, not automatically a true hard sample.")
    lines.append("5. Next steps are class/reason audit, pseudo-box quality audit, penalty sensitivity, and score calibration.")
    lines.append("")

    lines.append("## 9. Generated CSV/Figure outputs\n")
    lines.append("- CSV files are saved under `csv/`.")
    lines.append("- PNG figures are saved under `figures/`.")
    lines.append("")

    summary_path = output_dir / "summary_direction_issue.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] {summary_path}")


# =============================================================================
# [7] Main
# =============================================================================

def main():
    args = parse_args()
    load_dependencies()
    set_korean_font()

    run_dir = resolve_run_dir(args.run_dir)
    priority_dir = resolve_priority_dir(args.priority_dir)

    print("=" * 100)
    print("[DIRECTION ISSUE POST-HOC ANALYSIS]")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Run dir      : {run_dir}")
    print(f"Priority dir : {priority_dir}")
    print("=" * 100)

    selected_df, round_df, priority_df, build_log_df = load_inputs(run_dir, priority_dir)

    # Re-prepare priority with user-selected quantile bins.
    priority_prepared = prepare_priority_df(priority_df, args.quantile_bins)
    selected_merged, merge_info, _ = merge_selection_with_scores(selected_df, priority_prepared)

    output_dir = run_dir / f"direction_issue_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    csv_dir = output_dir / "csv"
    fig_dir = output_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Merge keys: {merge_info['merge_keys']}")
    print(f"[INFO] Merge failed rows: {merge_info['merge_failed_rows']} / {merge_info['selected_rows']}")
    if merge_info["merge_failed_rows"] > 0:
        print("[WARN] Some selected rows were not matched to priority scores.")
        print(f"Selected columns: {available_columns_message(selected_df)}")
        print(f"Priority columns: {available_columns_message(priority_df)}")

    class_dist, class_agg, reason_dist, reason_agg, no_pseudo = compute_distribution_tables(selected_merged)
    overlap_df, membership_df = compute_combined_lowpriority_overlap(selected_merged)
    quantile_dist, map_summary = compute_score_quantile_tables(selected_merged, round_df)

    # -------------------------------------------------------------------------
    # Save CSVs
    # -------------------------------------------------------------------------
    csv_outputs = {
        "strategy_class_distribution.csv": class_dist,
        "strategy_class_distribution_aggregate.csv": class_agg,
        "strategy_groundedness_reason_distribution.csv": reason_dist,
        "strategy_groundedness_reason_distribution_aggregate.csv": reason_agg,
        "no_pseudo_box_selected_samples.csv": no_pseudo,
        "combined_vs_lowpriority_overlap.csv": overlap_df,
        "combined_vs_lowpriority_sample_membership.csv": membership_df,
        "score_quantile_selection_distribution.csv": quantile_dist,
        "score_quantile_map_summary.csv": map_summary,
    }

    for filename, df in csv_outputs.items():
        path = csv_dir / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {path}")

    # Optional debug/context files.
    selected_merged.to_csv(csv_dir / "selected_samples_merged_with_scores.csv", index=False, encoding="utf-8-sig")
    if not build_log_df.empty:
        build_log_df.to_csv(csv_dir / "dataset_build_log_copy.csv", index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------------------
    # Save figures
    # -------------------------------------------------------------------------
    plot_stacked_distribution(
        df=class_agg,
        index_col="strategy",
        category_col="class_hint",
        value_col="ratio",
        title="Selected sample class distribution by strategy",
        ylabel="Ratio",
        save_path=fig_dir / "01_strategy_class_distribution.png",
    )

    plot_stacked_distribution(
        df=reason_agg,
        index_col="strategy",
        category_col="groundedness_reason",
        value_col="ratio",
        title="Selected sample groundedness-reason distribution by strategy",
        ylabel="Ratio",
        save_path=fig_dir / "02_strategy_reason_distribution.png",
    )

    plot_no_pseudo_by_strategy_round(
        reason_dist=reason_dist,
        save_path=fig_dir / "03_no_pseudo_box_by_strategy_round.png",
    )

    plot_overlap_heatmap(
        overlap_df=overlap_df,
        save_path=fig_dir / "04_combined_vs_lowpriority_overlap_heatmap.png",
    )

    plot_score_quantile_selection(
        quantile_dist=quantile_dist,
        save_path=fig_dir / "05_score_quantile_selection_by_strategy.png",
    )

    plot_score_quantile_vs_map(
        map_summary=map_summary,
        save_path=fig_dir / "06_score_quantile_vs_map50.png",
    )

    write_summary_md(
        output_dir=output_dir,
        run_dir=run_dir,
        priority_dir=priority_dir,
        merge_info=merge_info,
        selected_df=selected_merged,
        class_agg=class_agg,
        reason_agg=reason_agg,
        no_pseudo_df=no_pseudo,
        overlap_df=overlap_df,
        quantile_dist=quantile_dist,
        map_summary=map_summary,
    )

    print("\n" + "=" * 100)
    print("[DONE] Direction issue analysis completed")
    print(f"Output dir: {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()

"""
===============================================================================
[File] recover_v3_minimal_results.py

[Purpose]
Recover analysis outputs from an already completed YOLO Active Learning V3
minimal experiment.

This script DOES NOT train YOLO again.
It only reads saved CSV files from an existing run directory and regenerates:

1. Seed-level metric summary
2. Aggregate strategy metric summary
3. AULC results
4. Learning curve plots
5. Final performance comparison plots
6. Selection distribution plots
7. Notion-ready markdown summary

Input:
    runs/active_learning_ablation_v3_minimal/al_ablation_v3_minimal_*/
        all_round_results.csv
        all_selected_samples_by_round.csv
        all_dataset_build_log.csv

Output:
    runs/active_learning_ablation_v3_minimal/al_ablation_v3_minimal_*/
        recovered_analysis_YYYYMMDD_HHMMSS/
            seed_strategy_metric_summary.csv
            aggregate_strategy_metric_summary.csv
            aggregate_learning_curve_map50.png
            aggregate_learning_curve_map5095.png
            final_map50_mean_std.png
            final_map5095_mean_std.png
            aulc_map50_mean_std.png
            aulc_map5095_mean_std.png
            selection_reason_distribution.png
            selection_dataset_distribution.png
            selection_class_hint_distribution.png
            notion_recovered_summary.md

Why this script exists:
    The original YOLO experiment completed training and saved raw CSV files,
    but crashed during AULC calculation because np.trapz was unavailable
    in the current NumPy version.

Important:
    This script uses a manual trapezoidal AULC implementation.
    It does not rely on np.trapz or np.trapezoid.
===============================================================================
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# =============================================================================
# [0] Project paths
# =============================================================================

PROJECT_ROOT = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project")

RUNS_ROOT = PROJECT_ROOT / "runs" / "active_learning_ablation_v3_minimal"

# 직접 특정 폴더를 지정하고 싶으면 아래에 Path를 넣으면 됨.
# 예:
# TARGET_RUN_DIR = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project/runs/active_learning_ablation_v3_minimal/al_ablation_v3_minimal_20260622_222607")
TARGET_RUN_DIR = Path(
    "/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project/runs/active_learning_direction_check/al_ablation_v3_minimal_20260623_124817"
)

# =============================================================================
# [1] Matplotlib settings
# =============================================================================

def set_korean_font():
    """
    Set Korean font if available.
    """
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


# =============================================================================
# [2] File loading
# =============================================================================

def find_latest_run_dir() -> Path:
    """
    Find latest al_ablation_v3_minimal_* run directory.
    """
    run_dirs = [
        p for p in RUNS_ROOT.glob("al_ablation_v3_minimal_*")
        if p.is_dir()
    ]

    if not run_dirs:
        raise FileNotFoundError(
            f"No run directory found under: {RUNS_ROOT}"
        )

    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def resolve_target_run_dir() -> Path:
    """
    Use manually specified TARGET_RUN_DIR if provided,
    otherwise use latest run directory.
    """
    if TARGET_RUN_DIR is not None:
        target = Path(TARGET_RUN_DIR)
        if not target.exists():
            raise FileNotFoundError(f"TARGET_RUN_DIR does not exist: {target}")
        return target

    return find_latest_run_dir()


def load_required_csvs(run_dir: Path):
    """
    Load all saved CSV files from the completed experiment.
    """
    round_csv = run_dir / "all_round_results.csv"
    selected_csv = run_dir / "all_selected_samples_by_round.csv"
    build_log_csv = run_dir / "all_dataset_build_log.csv"

    if not round_csv.exists():
        raise FileNotFoundError(f"Missing file: {round_csv}")

    if not selected_csv.exists():
        print(f"[WARN] Missing selected samples CSV: {selected_csv}")

    if not build_log_csv.exists():
        print(f"[WARN] Missing dataset build log CSV: {build_log_csv}")

    results_df = pd.read_csv(round_csv)

    selected_df = (
        pd.read_csv(selected_csv)
        if selected_csv.exists()
        else pd.DataFrame()
    )

    build_log_df = (
        pd.read_csv(build_log_csv)
        if build_log_csv.exists()
        else pd.DataFrame()
    )

    return round_csv, selected_csv, build_log_csv, results_df, selected_df, build_log_df


# =============================================================================
# [3] Data cleaning
# =============================================================================

def normalize_results_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and data types.
    """
    df = df.copy()

    # metric column fallback
    if "map5095" not in df.columns:
        candidates = ["map50_95", "map50-95", "mAP50-95", "mAP50_95"]
        for c in candidates:
            if c in df.columns:
                df["map5095"] = df[c]
                break

    required_cols = [
        "seed",
        "strategy",
        "round",
        "labeled_budget",
        "map50",
        "map5095",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"all_round_results.csv missing columns: {missing}")

    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    df["labeled_budget"] = pd.to_numeric(df["labeled_budget"], errors="coerce")

    for c in ["map50", "map5095", "precision", "recall"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remove internal shared baseline row from strategy comparison if it exists.
    # Round 0 is already copied to each strategy in the v3 minimal script.
    df = df[df["strategy"] != "_SHARED_BASELINE"].copy()

    df = df.sort_values(
        ["seed", "strategy", "round", "labeled_budget"]
    ).reset_index(drop=True)

    return df


def normalize_selected_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize selected samples dataframe.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()

    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")

    if "round" in df.columns:
        df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")

    for col in ["strategy", "groundedness_reason", "dataset_type", "class_hint"]:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)

    return df


# =============================================================================
# [4] AULC and summaries
# =============================================================================

def compute_aulc(x_values, y_values):
    """
    Manual trapezoidal AULC.

    This avoids np.trapz / np.trapezoid compatibility problems.
    """
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    dx = x[1:] - x[:-1]
    avg_height = (y[1:] + y[:-1]) / 2.0

    return float(np.sum(dx * avg_height))


def make_seed_strategy_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create seed-level strategy summary.

    One row per seed-strategy.
    """
    rows = []

    for (seed, strategy), sub in results_df.groupby(["seed", "strategy"]):
        sub = sub.sort_values("labeled_budget").copy()

        initial_row = sub.iloc[0]
        final_row = sub.iloc[-1]

        row = {
            "seed": int(seed),
            "strategy": strategy,
            "num_rounds": int(sub["round"].nunique()),
            "initial_budget": int(initial_row["labeled_budget"]),
            "final_budget": int(final_row["labeled_budget"]),

            "initial_map50": float(initial_row["map50"]),
            "final_map50": float(final_row["map50"]),
            "max_map50": float(sub["map50"].max()),
            "delta_final_map50": float(final_row["map50"] - initial_row["map50"]),

            "initial_map5095": float(initial_row["map5095"]),
            "final_map5095": float(final_row["map5095"]),
            "max_map5095": float(sub["map5095"].max()),
            "delta_final_map5095": float(final_row["map5095"] - initial_row["map5095"]),

            "aulc_map50": compute_aulc(sub["labeled_budget"], sub["map50"]),
            "aulc_map5095": compute_aulc(sub["labeled_budget"], sub["map5095"]),
        }

        if "precision" in sub.columns:
            row["final_precision"] = float(final_row["precision"]) if pd.notna(final_row["precision"]) else np.nan

        if "recall" in sub.columns:
            row["final_recall"] = float(final_row["recall"]) if pd.notna(final_row["recall"]) else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(
            ["seed", "final_map50"],
            ascending=[True, False]
        ).reset_index(drop=True)

    return out


def make_aggregate_strategy_summary(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across seeds.
    """
    metric_cols = [
        "final_map50",
        "final_map5095",
        "max_map50",
        "max_map5095",
        "delta_final_map50",
        "delta_final_map5095",
        "aulc_map50",
        "aulc_map5095",
    ]

    optional_cols = ["final_precision", "final_recall"]
    for c in optional_cols:
        if c in seed_summary_df.columns:
            metric_cols.append(c)

    rows = []

    for strategy, sub in seed_summary_df.groupby("strategy"):
        row = {
            "strategy": strategy,
            "num_seeds": int(sub["seed"].nunique()),
            "final_budget_mean": float(sub["final_budget"].mean()),
        }

        for metric in metric_cols:
            if metric not in sub.columns:
                continue

            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std()) if len(sub) > 1 else 0.0
            row[f"{metric}_min"] = float(sub[metric].min())
            row[f"{metric}_max"] = float(sub[metric].max())

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return out

    # Random 대비 차이 계산
    if "Random" in out["strategy"].values:
        random_row = out[out["strategy"] == "Random"].iloc[0]

        for metric in [
            "final_map50_mean",
            "final_map5095_mean",
            "aulc_map50_mean",
            "aulc_map5095_mean",
        ]:
            if metric in out.columns:
                out[f"{metric}_minus_random"] = out[metric] - random_row[metric]

    # LowPrioritySoft 대비 차이 계산
    if "LowPrioritySoft" in out["strategy"].values:
        low_row = out[out["strategy"] == "LowPrioritySoft"].iloc[0]

        for metric in [
            "final_map50_mean",
            "final_map5095_mean",
            "aulc_map50_mean",
            "aulc_map5095_mean",
        ]:
            if metric in out.columns:
                out[f"{metric}_minus_lowpriority"] = out[metric] - low_row[metric]

    if "final_map50_mean" in out.columns:
        out = out.sort_values("final_map50_mean", ascending=False)

    return out.reset_index(drop=True)


def make_learning_curve_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean/std learning curve by strategy and budget.
    """
    rows = []

    for (strategy, budget), sub in results_df.groupby(["strategy", "labeled_budget"]):
        row = {
            "strategy": strategy,
            "labeled_budget": int(budget),
            "num_seeds": int(sub["seed"].nunique()),
            "map50_mean": float(sub["map50"].mean()),
            "map50_std": float(sub["map50"].std()) if len(sub) > 1 else 0.0,
            "map5095_mean": float(sub["map5095"].mean()),
            "map5095_std": float(sub["map5095"].std()) if len(sub) > 1 else 0.0,
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(["strategy", "labeled_budget"]).reset_index(drop=True)

    return out


# =============================================================================
# [5] Plotting helpers
# =============================================================================

def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {path}")


def plot_learning_curve(results_df: pd.DataFrame, metric: str, save_dir: Path):
    """
    Plot mean ± std learning curve across seeds.
    """
    if metric not in results_df.columns:
        print(f"[SKIP] metric not found: {metric}")
        return

    if results_df[metric].isna().all():
        print(f"[SKIP] all NaN metric: {metric}")
        return

    plt.figure(figsize=(9, 6))

    for strategy, sub in results_df.groupby("strategy"):
        curve = (
            sub.groupby("labeled_budget")[metric]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("labeled_budget")
        )

        x = curve["labeled_budget"].values
        y = curve["mean"].values
        std = curve["std"].fillna(0.0).values

        plt.plot(x, y, marker="o", linewidth=2, label=strategy)
        plt.fill_between(x, y - std, y + std, alpha=0.15)

    title_metric = "mAP@50" if metric == "map50" else "mAP@50-95"

    plt.title(
        f"YOLOv8 Active Learning V3 Minimal: {title_metric}",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Number of labeled training samples")
    plt.ylabel(title_metric)
    plt.grid(alpha=0.3)
    plt.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")

    save_fig(save_dir / f"aggregate_learning_curve_{metric}.png")


def plot_bar_mean_std(
    aggregate_df: pd.DataFrame,
    metric_mean_col: str,
    metric_std_col: str,
    title: str,
    ylabel: str,
    save_path: Path,
):
    """
    Plot strategy mean ± std bar chart.
    """
    if metric_mean_col not in aggregate_df.columns:
        print(f"[SKIP] column not found: {metric_mean_col}")
        return

    plot_df = aggregate_df.copy()
    plot_df = plot_df.sort_values(metric_mean_col, ascending=False)

    x = plot_df["strategy"].values
    y = plot_df[metric_mean_col].values
    yerr = plot_df[metric_std_col].fillna(0.0).values if metric_std_col in plot_df.columns else None

    plt.figure(figsize=(9, 6))
    bars = plt.bar(x, y, yerr=yerr, capsize=4)

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Strategy")
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")

    for bar, value in zip(bars, y):
        if pd.notna(value):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    save_fig(save_path)


def plot_selection_distribution(
    selected_df: pd.DataFrame,
    group_col: str,
    save_path: Path,
):
    """
    Plot stacked selected sample distribution by strategy.
    """
    if selected_df is None or len(selected_df) == 0:
        print(f"[SKIP] selected_df empty: {group_col}")
        return

    if group_col not in selected_df.columns:
        print(f"[SKIP] column not found in selected_df: {group_col}")
        return

    acquired = selected_df.copy()

    if "round" in acquired.columns:
        acquired = acquired[acquired["round"] > 0].copy()

    if len(acquired) == 0:
        print(f"[SKIP] no acquired samples after round > 0: {group_col}")
        return

    pivot = (
        acquired.groupby(["strategy", group_col])
        .size()
        .unstack(fill_value=0)
    )

    # Strategy order: keep common interpretation order if available.
    preferred_order = [
        "Random",
        "ConsistencyOnly",
        "GroundednessOnlySoft",
        "CombinedSoftPenalty",
        "LowPrioritySoft",
    ]

    ordered_index = [s for s in preferred_order if s in pivot.index]
    ordered_index += [s for s in pivot.index if s not in ordered_index]
    pivot = pivot.loc[ordered_index]

    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        width=0.75,
    )

    ax.set_title(
        f"Selected sample distribution by {group_col}",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Number of selected samples across all seeds / rounds")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=25, ha="right")

    # Add labels
    for container in ax.containers:
        ax.bar_label(container, label_type="center", fontsize=8)

    save_fig(save_path)


# =============================================================================
# [6] Markdown summary
# =============================================================================

def dataframe_to_markdown_safe(df: pd.DataFrame, max_rows: int = None) -> str:
    """
    Convert dataframe to markdown without tabulate dependency.
    """
    if df is None or len(df) == 0:
        return "_No data available._"

    show_df = df.copy()

    if max_rows is not None:
        show_df = show_df.head(max_rows)

    # Round floats for readability
    for col in show_df.columns:
        if pd.api.types.is_float_dtype(show_df[col]):
            show_df[col] = show_df[col].round(6)

    show_df = show_df.astype(str)

    header = "| " + " | ".join(show_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(show_df.columns)) + " |"

    rows = []
    for _, row in show_df.iterrows():
        values = [str(v).replace("\n", " ") for v in row.values.tolist()]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator] + rows)


def get_value_or_none(df: pd.DataFrame, strategy: str, col: str):
    if df is None or len(df) == 0:
        return None

    if "strategy" not in df.columns or col not in df.columns:
        return None

    sub = df[df["strategy"] == strategy]
    if len(sub) == 0:
        return None

    value = sub.iloc[0][col]

    if pd.isna(value):
        return None

    return float(value)


def make_interpretation_text(aggregate_df: pd.DataFrame) -> list:
    """
    Generate simple interpretation bullets.
    """
    lines = []

    if aggregate_df is None or len(aggregate_df) == 0:
        return ["- 결과 요약을 생성할 수 없습니다."]

    if "final_map50_mean" in aggregate_df.columns:
        best_final_map50 = aggregate_df.sort_values(
            "final_map50_mean",
            ascending=False,
        ).iloc[0]

        lines.append(
            f"- 최종 mAP@50 평균 기준 1위 전략은 "
            f"`{best_final_map50['strategy']}` "
            f"({best_final_map50['final_map50_mean']:.4f})입니다."
        )

    if "final_map5095_mean" in aggregate_df.columns:
        best_final_map5095 = aggregate_df.sort_values(
            "final_map5095_mean",
            ascending=False,
        ).iloc[0]

        lines.append(
            f"- 최종 mAP@50-95 평균 기준 1위 전략은 "
            f"`{best_final_map5095['strategy']}` "
            f"({best_final_map5095['final_map5095_mean']:.4f})입니다."
        )

    if "aulc_map50_mean" in aggregate_df.columns:
        best_aulc_map50 = aggregate_df.sort_values(
            "aulc_map50_mean",
            ascending=False,
        ).iloc[0]

        lines.append(
            f"- AULC-mAP@50 평균 기준 1위 전략은 "
            f"`{best_aulc_map50['strategy']}` "
            f"({best_aulc_map50['aulc_map50_mean']:.4f})입니다."
        )

    combined_final = get_value_or_none(
        aggregate_df,
        "CombinedSoftPenalty",
        "final_map50_mean_minus_random",
    )

    low_final = get_value_or_none(
        aggregate_df,
        "LowPrioritySoft",
        "final_map50_mean",
    )

    combined_vs_low = get_value_or_none(
        aggregate_df,
        "CombinedSoftPenalty",
        "final_map50_mean_minus_lowpriority",
    )

    if combined_final is not None:
        if combined_final > 0:
            lines.append(
                f"- `CombinedSoftPenalty`는 최종 mAP@50 평균에서 Random보다 "
                f"{combined_final:.4f} 높습니다."
            )
        else:
            lines.append(
                f"- `CombinedSoftPenalty`는 최종 mAP@50 평균에서 Random보다 "
                f"{abs(combined_final):.4f} 낮습니다."
            )

    if combined_vs_low is not None:
        if combined_vs_low > 0:
            lines.append(
                f"- `CombinedSoftPenalty`는 `LowPrioritySoft`보다 최종 mAP@50 평균이 "
                f"{combined_vs_low:.4f} 높아, priority score 방향성 검증에 긍정적 신호가 있습니다."
            )
        else:
            lines.append(
                f"- `CombinedSoftPenalty`는 `LowPrioritySoft`보다 최종 mAP@50 평균이 "
                f"{abs(combined_vs_low):.4f} 낮아, priority score 방향성 검증은 아직 조심스럽게 해석해야 합니다."
            )

    if low_final is not None:
        lines.append(
            f"- `LowPrioritySoft`의 최종 성능은 negative control로 해석할 수 있으며, "
            f"high-priority 전략과 Random 사이의 방향성 판단에 사용됩니다."
        )

    lines.append(
        "- 단, validation set이 작기 때문에 단일 수치보다 seed 평균, 표준편차, AULC를 함께 해석해야 합니다."
    )

    return lines


def write_summary_md(
    save_dir: Path,
    run_dir: Path,
    round_csv: Path,
    selected_csv: Path,
    build_log_csv: Path,
    results_df: pd.DataFrame,
    seed_summary_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    learning_curve_df: pd.DataFrame,
):
    """
    Write Notion-ready markdown summary.
    """
    lines = []

    lines.append("# YOLO Active Learning V3 Minimal — Recovered Analysis\n")

    lines.append("## 1. 복구 목적\n")
    lines.append(
        "본 문서는 YOLO 학습이 완료된 이후 AULC 계산 단계에서 중단된 "
        "V3 minimal active learning 실험 결과를 복구 분석한 것이다. "
        "이 스크립트는 YOLO를 재학습하지 않고, 저장된 CSV 파일만 읽어 "
        "AULC, seed별 요약, 전략별 평균/표준편차, 그래프를 재생성한다.\n"
    )

    lines.append("## 2. 입력 파일\n")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Round results: `{round_csv}`")
    lines.append(f"- Selected samples: `{selected_csv}`")
    lines.append(f"- Dataset build log: `{build_log_csv}`")
    lines.append("")

    lines.append("## 3. 실험 구성 요약\n")

    if len(results_df) > 0:
        strategies = sorted(results_df["strategy"].dropna().unique().tolist())
        seeds = sorted(results_df["seed"].dropna().unique().tolist())
        budgets = sorted(results_df["labeled_budget"].dropna().unique().tolist())

        lines.append(f"- Strategies: `{', '.join(map(str, strategies))}`")
        lines.append(f"- Seeds: `{', '.join(map(str, seeds))}`")
        lines.append(f"- Labeled budgets: `{', '.join(map(lambda x: str(int(x)), budgets))}`")
        lines.append(f"- Total round records: `{len(results_df)}`")
        lines.append("")

    lines.append("## 4. 핵심 해석\n")
    lines.extend(make_interpretation_text(aggregate_df))
    lines.append("")

    lines.append("## 5. 전략별 평균 성능 요약\n")
    compact_cols = [
        "strategy",
        "num_seeds",
        "final_map50_mean",
        "final_map50_std",
        "final_map5095_mean",
        "final_map5095_std",
        "aulc_map50_mean",
        "aulc_map50_std",
        "aulc_map5095_mean",
        "aulc_map5095_std",
        "final_map50_mean_minus_random",
        "final_map50_mean_minus_lowpriority",
    ]

    available_cols = [c for c in compact_cols if c in aggregate_df.columns]
    lines.append(dataframe_to_markdown_safe(aggregate_df[available_cols]))
    lines.append("")

    lines.append("## 6. Seed별 결과 요약\n")
    seed_cols = [
        "seed",
        "strategy",
        "initial_budget",
        "final_budget",
        "initial_map50",
        "final_map50",
        "delta_final_map50",
        "initial_map5095",
        "final_map5095",
        "delta_final_map5095",
        "aulc_map50",
        "aulc_map5095",
    ]
    available_seed_cols = [c for c in seed_cols if c in seed_summary_df.columns]
    lines.append(dataframe_to_markdown_safe(seed_summary_df[available_seed_cols]))
    lines.append("")

    lines.append("## 7. 생성된 그래프 파일\n")
    expected_figs = [
        "aggregate_learning_curve_map50.png",
        "aggregate_learning_curve_map5095.png",
        "final_map50_mean_std.png",
        "final_map5095_mean_std.png",
        "aulc_map50_mean_std.png",
        "aulc_map5095_mean_std.png",
        "selection_reason_distribution.png",
        "selection_dataset_distribution.png",
        "selection_class_hint_distribution.png",
    ]

    for fig in expected_figs:
        fig_path = save_dir / fig
        if fig_path.exists():
            lines.append(f"- `{fig_path}`")

    lines.append("")

    lines.append("## 8. 보고 시 주의점\n")
    lines.append(
        "- Active Learning 성능은 최종 mAP 한 점만이 아니라 AULC를 함께 봐야 한다."
    )
    lines.append(
        "- `Random`은 같은 budget에서 비교하는 가장 중요한 baseline이다."
    )
    lines.append(
        "- `LowPrioritySoft`는 priority score 방향성이 맞는지 확인하기 위한 negative control이다."
    )
    lines.append(
        "- `ConsistencyOnly`, `GroundednessOnlySoft`, `CombinedSoftPenalty` 비교는 교수님 피드백의 ablation 요구에 대응한다."
    )
    lines.append(
        "- validation set이 작기 때문에 seed별 편차가 크면 평균 성능을 단정적으로 주장하면 안 된다."
    )

    summary_path = save_dir / "notion_recovered_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] {summary_path}")


# =============================================================================
# [7] Main
# =============================================================================

def main():
    set_korean_font()

    run_dir = resolve_target_run_dir()

    print("=" * 100)
    print("[RECOVER V3 MINIMAL RESULTS]")
    print(f"Run dir: {run_dir}")
    print("=" * 100)

    (
        round_csv,
        selected_csv,
        build_log_csv,
        results_df,
        selected_df,
        build_log_df,
    ) = load_required_csvs(run_dir)

    results_df = normalize_results_df(results_df)
    selected_df = normalize_selected_df(selected_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = run_dir / f"recovered_analysis_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Round results rows   : {len(results_df)}")
    print(f"Selected sample rows : {len(selected_df)}")
    print(f"Build log rows       : {len(build_log_df)}")
    print(f"Save dir             : {save_dir}")

    # -------------------------------------------------------------------------
    # Save cleaned raw data
    # -------------------------------------------------------------------------
    cleaned_results_path = save_dir / "cleaned_all_round_results.csv"
    cleaned_selected_path = save_dir / "cleaned_all_selected_samples_by_round.csv"

    results_df.to_csv(cleaned_results_path, index=False, encoding="utf-8-sig")
    selected_df.to_csv(cleaned_selected_path, index=False, encoding="utf-8-sig")

    print(f"[SAVE] {cleaned_results_path}")
    print(f"[SAVE] {cleaned_selected_path}")

    # -------------------------------------------------------------------------
    # Summary tables
    # -------------------------------------------------------------------------
    seed_summary_df = make_seed_strategy_summary(results_df)
    aggregate_df = make_aggregate_strategy_summary(seed_summary_df)
    learning_curve_df = make_learning_curve_summary(results_df)

    seed_summary_path = save_dir / "seed_strategy_metric_summary.csv"
    aggregate_path = save_dir / "aggregate_strategy_metric_summary.csv"
    learning_curve_path = save_dir / "aggregate_learning_curve_summary.csv"

    seed_summary_df.to_csv(seed_summary_path, index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    learning_curve_df.to_csv(learning_curve_path, index=False, encoding="utf-8-sig")

    print(f"[SAVE] {seed_summary_path}")
    print(f"[SAVE] {aggregate_path}")
    print(f"[SAVE] {learning_curve_path}")

    # -------------------------------------------------------------------------
    # JSON summary
    # -------------------------------------------------------------------------
    json_summary = {
        "run_dir": str(run_dir),
        "round_csv": str(round_csv),
        "selected_csv": str(selected_csv),
        "build_log_csv": str(build_log_csv),
        "num_rows_results": int(len(results_df)),
        "num_rows_selected": int(len(selected_df)),
        "strategies": sorted(results_df["strategy"].dropna().unique().tolist()),
        "seeds": [int(x) for x in sorted(results_df["seed"].dropna().unique().tolist())],
        "budgets": [int(x) for x in sorted(results_df["labeled_budget"].dropna().unique().tolist())],
    }

    if len(aggregate_df) > 0:
        if "final_map50_mean" in aggregate_df.columns:
            json_summary["best_final_map50_strategy"] = aggregate_df.sort_values(
                "final_map50_mean",
                ascending=False,
            ).iloc[0]["strategy"]

        if "aulc_map50_mean" in aggregate_df.columns:
            json_summary["best_aulc_map50_strategy"] = aggregate_df.sort_values(
                "aulc_map50_mean",
                ascending=False,
            ).iloc[0]["strategy"]

    json_path = save_dir / "recovered_analysis_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, ensure_ascii=False, indent=4)

    print(f"[SAVE] {json_path}")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    plot_learning_curve(results_df, "map50", save_dir)
    plot_learning_curve(results_df, "map5095", save_dir)

    plot_bar_mean_std(
        aggregate_df=aggregate_df,
        metric_mean_col="final_map50_mean",
        metric_std_col="final_map50_std",
        title="Final mAP@50 mean ± std by strategy",
        ylabel="Final mAP@50",
        save_path=save_dir / "final_map50_mean_std.png",
    )

    plot_bar_mean_std(
        aggregate_df=aggregate_df,
        metric_mean_col="final_map5095_mean",
        metric_std_col="final_map5095_std",
        title="Final mAP@50-95 mean ± std by strategy",
        ylabel="Final mAP@50-95",
        save_path=save_dir / "final_map5095_mean_std.png",
    )

    plot_bar_mean_std(
        aggregate_df=aggregate_df,
        metric_mean_col="aulc_map50_mean",
        metric_std_col="aulc_map50_std",
        title="AULC-mAP@50 mean ± std by strategy",
        ylabel="AULC-mAP@50",
        save_path=save_dir / "aulc_map50_mean_std.png",
    )

    plot_bar_mean_std(
        aggregate_df=aggregate_df,
        metric_mean_col="aulc_map5095_mean",
        metric_std_col="aulc_map5095_std",
        title="AULC-mAP@50-95 mean ± std by strategy",
        ylabel="AULC-mAP@50-95",
        save_path=save_dir / "aulc_map5095_mean_std.png",
    )

    if len(selected_df) > 0:
        plot_selection_distribution(
            selected_df=selected_df,
            group_col="groundedness_reason",
            save_path=save_dir / "selection_reason_distribution.png",
        )

        plot_selection_distribution(
            selected_df=selected_df,
            group_col="dataset_type",
            save_path=save_dir / "selection_dataset_distribution.png",
        )

        plot_selection_distribution(
            selected_df=selected_df,
            group_col="class_hint",
            save_path=save_dir / "selection_class_hint_distribution.png",
        )

    # -------------------------------------------------------------------------
    # Markdown
    # -------------------------------------------------------------------------
    write_summary_md(
        save_dir=save_dir,
        run_dir=run_dir,
        round_csv=round_csv,
        selected_csv=selected_csv,
        build_log_csv=build_log_csv,
        results_df=results_df,
        seed_summary_df=seed_summary_df,
        aggregate_df=aggregate_df,
        learning_curve_df=learning_curve_df,
    )

    print("\n" + "=" * 100)
    print("[완료] V3 minimal 결과 복구 분석 완료")
    print(f"Recovered analysis dir: {save_dir}")
    print("=" * 100)

    print("\n[먼저 확인할 파일]")
    print(f"1. {save_dir / 'aggregate_strategy_metric_summary.csv'}")
    print(f"2. {save_dir / 'notion_recovered_summary.md'}")
    print(f"3. {save_dir / 'aggregate_learning_curve_map50.png'}")
    print(f"4. {save_dir / 'aggregate_learning_curve_map5095.png'}")
    print(f"5. {save_dir / 'aulc_map50_mean_std.png'}")
    print(f"6. {save_dir / 'selection_reason_distribution.png'}")


if __name__ == "__main__":
    main()
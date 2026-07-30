"""Analyze and visualize V10c-PDF recall-guard one-cycle results.

Usage:
    .\.python311\python.exe scripts\03_analysis\analyze_v10c_pdf_recall_guard_results.py `
      --run-dir runs\active_learning_v10c_pdf_recall_guard_onecycle\<RUN_ID>

The script is read-only with respect to experiment data.  It writes figures and
summary tables under:

    <RUN_DIR>\analysis_v10c_pdf\
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["map5095", "map50", "precision", "recall", "f1"]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def infer_method_label(results: pd.DataFrame) -> str:
    if results.empty or "strategy" not in results.columns:
        return "DetectorPdfRecallGuardV10c"
    candidates = [
        s
        for s in results["strategy"].dropna().astype(str).unique().tolist()
        if s not in {"GTFreeRandom", "__SHARED_ROUND0__"}
    ]
    return candidates[0] if candidates else "DetectorPdfRecallGuardV10c"


def add_f1(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    if {"precision", "recall"}.issubset(out.columns):
        p = pd.to_numeric(out["precision"], errors="coerce")
        r = pd.to_numeric(out["recall"], errors="coerce")
        out["f1"] = np.where((p + r) > 0, 2 * p * r / (p + r), np.nan)
    return out


def ensure_comparison(results: pd.DataFrame, method_label: str) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    results = add_f1(results)
    rows = []
    round1 = results[results["strategy"].isin(["GTFreeRandom", method_label])].copy()
    for seed, sub in round1.groupby("acquisition_seed", dropna=False):
        random = sub[sub["strategy"] == "GTFreeRandom"]
        method = sub[sub["strategy"] == method_label]
        if random.empty or method.empty:
            continue
        rr = random.iloc[0]
        mm = method.iloc[0]
        row = {"acquisition_seed": seed}
        for metric in METRICS:
            row[f"random_{metric}"] = safe_float(rr.get(metric))
            row[f"method_{metric}"] = safe_float(mm.get(metric))
            row[f"diff_{metric}"] = row[f"method_{metric}"] - row[f"random_{metric}"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_diffs(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        col = f"diff_{metric}"
        if col not in comparison.columns:
            continue
        values = pd.to_numeric(comparison[col], errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "num_pairs": int(len(values)),
                "mean_diff": float(values.mean()) if len(values) else np.nan,
                "median_diff": float(values.median()) if len(values) else np.nan,
                "wins": int((values > 0).sum()) if len(values) else 0,
                "losses": int((values < 0).sum()) if len(values) else 0,
                "ties": int((values == 0).sum()) if len(values) else 0,
            }
        )
    return pd.DataFrame(rows)


def plot_seedwise_diffs(comparison: pd.DataFrame, out_dir: Path) -> None:
    if comparison.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plot_metrics = ["map5095", "precision", "recall", "f1"]
    seeds = comparison["acquisition_seed"].astype(str).tolist()
    for ax, metric in zip(axes.ravel(), plot_metrics):
        col = f"diff_{metric}"
        vals = pd.to_numeric(comparison.get(col), errors="coerce")
        colors = ["#2ca02c" if v > 0 else "#d62728" if v < 0 else "#7f7f7f" for v in vals]
        ax.bar(seeds, vals, color=colors)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"Method - Random: {metric}")
        ax.set_xlabel("acquisition seed")
        ax.set_ylabel("difference")
    fig.suptitle("Seedwise paired differences", fontsize=14)
    fig.savefig(out_dir / "fig01_seedwise_paired_differences.png", dpi=180)
    plt.close(fig)


def plot_mean_diffs(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    values = pd.to_numeric(summary["mean_diff"], errors="coerce")
    colors = ["#2ca02c" if v > 0 else "#d62728" if v < 0 else "#7f7f7f" for v in values]
    ax.bar(summary["metric"].astype(str), values, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    for idx, row in summary.iterrows():
        y = safe_float(row["mean_diff"])
        label = f"{y:+.4f}\n{int(row['wins'])}/{int(row['losses'])}/{int(row['ties'])}"
        ax.text(idx, y, label, ha="center", va="bottom" if y >= 0 else "top", fontsize=9)
    ax.set_title("Mean paired differences with win/loss/tie")
    ax.set_ylabel("Method - Random")
    fig.savefig(out_dir / "fig02_mean_differences.png", dpi=180)
    plt.close(fig)


def plot_selection_phases(selected: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if selected.empty or "strategy" not in selected.columns:
        return pd.DataFrame()
    method = selected[~selected["strategy"].astype(str).isin(["GTFreeRandom", "__SHARED_ROUND0__"])].copy()
    if method.empty:
        return pd.DataFrame()
    if "v10c_phase" not in method.columns:
        method["v10c_phase"] = "unknown"
    phase = method.groupby(["acquisition_seed", "v10c_phase"], dropna=False).size().reset_index(name="count")
    pivot = phase.pivot_table(index="acquisition_seed", columns="v10c_phase", values="count", fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("V10c-PDF selection phase composition")
    ax.set_xlabel("acquisition seed")
    ax.set_ylabel("selected images")
    ax.legend(title="phase", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(out_dir / "fig03_selection_phase_composition.png", dpi=180)
    plt.close(fig)
    return phase


def plot_pseudo_box_composition(selected: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if selected.empty or "strategy" not in selected.columns:
        return pd.DataFrame()
    method = selected[~selected["strategy"].astype(str).isin(["GTFreeRandom", "__SHARED_ROUND0__"])].copy()
    if method.empty:
        return pd.DataFrame()
    counts = pd.to_numeric(method.get("detector_pseudo_box_count"), errors="coerce").fillna(-1)
    method["pseudo_box_bucket"] = np.select(
        [counts.eq(0), counts.eq(1), counts.ge(2)],
        ["0 boxes", "1 box", "2+ boxes"],
        default="missing",
    )
    bucket = method.groupby(["acquisition_seed", "pseudo_box_bucket"], dropna=False).size().reset_index(name="count")
    pivot = bucket.pivot_table(index="acquisition_seed", columns="pseudo_box_bucket", values="count", fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#d62728", "#ff7f0e", "#2ca02c", "#7f7f7f"])
    ax.set_title("Selected pseudo-box composition")
    ax.set_xlabel("acquisition seed")
    ax.set_ylabel("selected images")
    ax.legend(title="bucket", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(out_dir / "fig04_pseudo_box_composition.png", dpi=180)
    plt.close(fig)
    return bucket


def plot_per_class(per_class: pd.DataFrame, out_dir: Path) -> None:
    if per_class.empty or "diff_ap5095" not in per_class.columns:
        return
    summary = (
        per_class.groupby("class_name", dropna=False)["diff_ap5095"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    vals = pd.to_numeric(summary["mean"], errors="coerce")
    colors = ["#2ca02c" if v > 0 else "#d62728" if v < 0 else "#7f7f7f" for v in vals]
    ax.bar(summary["class_name"].astype(str), vals, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Per-class AP50-95 delta: Method - Random")
    ax.set_xlabel("class")
    ax.set_ylabel("mean diff")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(out_dir / "fig05_per_class_ap5095_delta.png", dpi=180)
    plt.close(fig)


def write_summary(
    *,
    run_dir: Path,
    out_dir: Path,
    method_label: str,
    comparison: pd.DataFrame,
    diff_summary: pd.DataFrame,
    phase_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        try:
            return df.to_markdown(index=False)
        except Exception:
            return "```text\n" + df.to_string(index=False) + "\n```"

    lines = [
        "# V10c-PDF recall-guard analysis",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Method label: `{method_label}`",
        "",
        "## Paired difference summary",
        "",
        md_table(diff_summary),
        "",
        "## Seedwise comparison",
        "",
        md_table(comparison),
        "",
        "## Selection phase composition",
        "",
        md_table(phase_summary),
        "",
        "## Pseudo-box composition",
        "",
        md_table(bucket_summary),
        "",
        "## Figures",
        "",
        "- `fig01_seedwise_paired_differences.png`",
        "- `fig02_mean_differences.png`",
        "- `fig03_selection_phase_composition.png`",
        "- `fig04_pseudo_box_composition.png`",
        "- `fig05_per_class_ap5095_delta.png`",
        "",
        "## Interpretation guardrail",
        "",
        "Positive mAP50-95 with recall recovery is meaningful. Precision-only gains with recall loss should not be treated as a breakthrough.",
    ]
    (out_dir / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="V10c-PDF run output directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "analysis_v10c_pdf"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    results = read_csv(run_dir / "v10c_onecycle_results.csv")
    selected = read_csv(run_dir / "v10c_selected_samples.csv")
    comparison = read_csv(run_dir / "seedwise_random_v10c_comparison.csv")
    per_class = read_csv(run_dir / "per_class_v10c_minus_random.csv")

    method_label = infer_method_label(results)
    if comparison.empty:
        comparison = ensure_comparison(results, method_label)
    else:
        rename_map = {}
        for col in comparison.columns:
            if col.startswith("v10c_"):
                rename_map[col] = col.replace("v10c_", "method_", 1)
        comparison = comparison.rename(columns=rename_map)
    diff_summary = summarize_diffs(comparison)

    comparison.to_csv(out_dir / "analysis_seedwise_comparison.csv", index=False, encoding="utf-8-sig")
    diff_summary.to_csv(out_dir / "analysis_diff_summary.csv", index=False, encoding="utf-8-sig")

    plot_seedwise_diffs(comparison, out_dir)
    plot_mean_diffs(diff_summary, out_dir)
    phase_summary = plot_selection_phases(selected, out_dir)
    bucket_summary = plot_pseudo_box_composition(selected, out_dir)
    plot_per_class(per_class, out_dir)

    if config:
        (out_dir / "analysis_input_config_snapshot.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    write_summary(
        run_dir=run_dir,
        out_dir=out_dir,
        method_label=method_label,
        comparison=comparison,
        diff_summary=diff_summary,
        phase_summary=phase_summary,
        bucket_summary=bucket_summary,
    )
    print(f"[DONE] V10c-PDF analysis written to: {out_dir}")


if __name__ == "__main__":
    main()


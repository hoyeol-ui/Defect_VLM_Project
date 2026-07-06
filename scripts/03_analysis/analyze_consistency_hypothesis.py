"""
Analyze active-learning results around the core VLM consistency hypothesis.

This script is intentionally post-hoc: it does not retrain YOLO and does not
modify existing run outputs. It reframes strategy comparisons around:

1. ConsistencyOnly as the core hypothesis test.
2. Combined strategies as auxiliary pseudo-groundedness extensions.
3. LowPrioritySoft and no_pseudo_box behavior as direction/calibration
   diagnostics.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "active_learning_ablation_v3_minimal"
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_DIR = SCRIPT_ROOT / "02_active_learning"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

try:
    from strategy_metadata import add_strategy_metadata_columns, get_strategy_metadata
except Exception:
    def get_strategy_metadata(strategy: str) -> dict[str, str]:
        return {"display_name": strategy, "family": "Uncategorized", "role": ""}

    def add_strategy_metadata_columns(df: pd.DataFrame, strategy_col: str = "strategy") -> pd.DataFrame:
        return df


METRICS = [
    "final_map50",
    "final_map5095",
    "best_map50",
    "best_map5095",
    "aulc_map50",
    "aulc_map5095",
]


def find_latest_run_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Run root not found: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("al_ablation_v3_minimal_")]
    if not candidates:
        raise FileNotFoundError(f"No al_ablation_v3_minimal_* directories found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def metric_mean_from_seed(seed_df: pd.DataFrame, strategy: str, metric: str) -> float | None:
    sub = seed_df[seed_df["strategy"] == strategy]
    if sub.empty or metric not in sub.columns:
        return None
    return float(sub[metric].mean())


def compare_pair_by_seed(seed_df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    rows = []
    left_df = seed_df[seed_df["strategy"] == left].set_index("seed")
    right_df = seed_df[seed_df["strategy"] == right].set_index("seed")
    common_seeds = sorted(set(left_df.index).intersection(set(right_df.index)))
    for seed in common_seeds:
        row = {"seed": seed, "left_strategy": left, "right_strategy": right}
        for metric in METRICS:
            if metric in left_df.columns and metric in right_df.columns:
                left_value = left_df.loc[seed, metric]
                right_value = right_df.loc[seed, metric]
                row[f"{metric}_left"] = left_value
                row[f"{metric}_right"] = right_value
                row[f"{metric}_delta_left_minus_right"] = left_value - right_value
        rows.append(row)
    return pd.DataFrame(rows)


def make_pair_summary(seed_df: pd.DataFrame, left: str, right: str, question: str) -> dict:
    pair_df = compare_pair_by_seed(seed_df, left, right)
    row = {
        "question": question,
        "left_strategy": left,
        "right_strategy": right,
        "num_common_seeds": int(pair_df["seed"].nunique()) if not pair_df.empty else 0,
    }
    for metric in METRICS:
        delta_col = f"{metric}_delta_left_minus_right"
        if delta_col in pair_df.columns:
            deltas = pair_df[delta_col].dropna()
            row[f"{metric}_mean_delta"] = deltas.mean()
            row[f"{metric}_wins"] = int((deltas > 0).sum())
            row[f"{metric}_losses"] = int((deltas < 0).sum())
            row[f"{metric}_ties"] = int((deltas == 0).sum())
    return row


def make_auxiliary_gain_table(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("CombinedSoftPenalty", "ConsistencyOnly", "Naive auxiliary gain over core consistency"),
        ("CombinedSuppressNoPseudo", "ConsistencyOnly", "Calibrated auxiliary gain over core consistency"),
        ("CombinedSuppressNoPseudo", "CombinedSoftPenalty", "Calibration gain over naive auxiliary"),
        ("GroundednessOnlySoft", "ConsistencyOnly", "Auxiliary-only signal versus core consistency"),
        ("LowPrioritySoft", "CombinedSoftPenalty", "Reverse-direction diagnostic versus high-priority combined"),
    ]
    for left, right, question in comparisons:
        if left in set(seed_df["strategy"]) and right in set(seed_df["strategy"]):
            rows.append(make_pair_summary(seed_df, left, right, question))
    return pd.DataFrame(rows)


def make_consistency_vs_random(seed_df: pd.DataFrame) -> pd.DataFrame:
    if "ConsistencyOnly" not in set(seed_df["strategy"]) or "Random" not in set(seed_df["strategy"]):
        return pd.DataFrame()
    return compare_pair_by_seed(seed_df, "ConsistencyOnly", "Random")


def summarize_selection_reasons(selected_df: pd.DataFrame) -> pd.DataFrame:
    if selected_df.empty or "groundedness_reason" not in selected_df.columns:
        return pd.DataFrame()
    acquired = selected_df[selected_df["round"] > 0].copy() if "round" in selected_df.columns else selected_df.copy()
    if acquired.empty:
        return pd.DataFrame()
    out = acquired.groupby(["strategy", "groundedness_reason"]).size().reset_index(name="count")
    totals = out.groupby("strategy")["count"].transform("sum")
    out["ratio"] = out["count"] / totals
    return add_strategy_metadata_columns(out)


def write_summary(
    out_dir: Path,
    run_dir: Path,
    seed_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    hypothesis_df: pd.DataFrame,
    reason_df: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Consistency Hypothesis Analysis\n")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`\n")

    lines.append("## 1. Framing\n")
    lines.append("- `ConsistencyOnly` is the core hypothesis test, not merely a baseline.")
    lines.append("- Combined strategies are auxiliary extensions around the core consistency signal.")
    lines.append("- Pseudo groundedness is not the core novelty; it is weak auxiliary visual evidence.")
    lines.append("- `no_pseudo_box` behavior should be interpreted as a calibration/failure-analysis issue.")
    lines.append("- Do not claim the method is proven superior; frame the current state as promising but not closed.\n")

    lines.append("## 2. Strategy Roles\n")
    for strategy in sorted(seed_df["strategy"].dropna().unique()):
        meta = get_strategy_metadata(str(strategy))
        lines.append(
            f"- `{strategy}`: {meta['display_name']} | {meta['family']} | {meta['role']}"
        )
    lines.append("")

    if not hypothesis_df.empty:
        lines.append("## 3. Hypothesis Comparisons\n")
        lines.append(hypothesis_df.to_markdown(index=False))
        lines.append("")

    if not aggregate_df.empty:
        lines.append("## 4. Aggregate Metrics With Strategy Metadata\n")
        display_cols = [
            c
            for c in [
                "strategy",
                "strategy_display_name",
                "strategy_family",
                "num_seeds",
                "final_map50_mean",
                "final_map5095_mean",
                "aulc_map50_mean",
                "aulc_map5095_mean",
            ]
            if c in aggregate_df.columns
        ]
        lines.append(aggregate_df[display_cols].to_markdown(index=False))
        lines.append("")

    if not reason_df.empty:
        lines.append("## 5. no_pseudo_box / groundedness_reason Diagnostic\n")
        no_pseudo = reason_df[reason_df["groundedness_reason"] == "no_pseudo_box"]
        if not no_pseudo.empty:
            cols = ["strategy", "strategy_display_name", "count", "ratio"]
            lines.append(no_pseudo[[c for c in cols if c in no_pseudo.columns]].to_markdown(index=False))
        else:
            lines.append("- No `no_pseudo_box` rows found in selected sample logs.")
        lines.append("")

    lines.append("## 6. Lab-meeting Takeaway\n")
    lines.append(
        "The research should be presented as expert-designed prompt-family consistency "
        "for GT-free active learning. OWL-ViT pseudo groundedness is an auxiliary "
        "signal whose calibration is under diagnosis, especially through "
        "`no_pseudo_box` and reverse-direction controls."
    )

    (out_dir / "summary_consistency_hypothesis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir(DEFAULT_RUN_ROOT)
    out_dir = run_dir / f"consistency_hypothesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_df = read_required_csv(run_dir / "seed_strategy_metric_summary.csv")
    aggregate_df = read_required_csv(run_dir / "aggregate_strategy_metric_summary.csv")
    selected_df = read_required_csv(run_dir / "all_selected_samples_by_round.csv")

    seed_df = add_strategy_metadata_columns(seed_df)
    aggregate_df = add_strategy_metadata_columns(aggregate_df)

    consistency_vs_random = make_consistency_vs_random(seed_df)
    auxiliary_gain = make_auxiliary_gain_table(seed_df)
    reason_df = summarize_selection_reasons(selected_df)

    seed_df.to_csv(out_dir / "consistency_hypothesis_summary.csv", index=False, encoding="utf-8-sig")
    consistency_vs_random.to_csv(out_dir / "consistency_vs_random_by_seed.csv", index=False, encoding="utf-8-sig")
    auxiliary_gain.to_csv(out_dir / "auxiliary_gain_over_consistency.csv", index=False, encoding="utf-8-sig")
    reason_df.to_csv(out_dir / "selection_reason_consistency_framing.csv", index=False, encoding="utf-8-sig")

    write_summary(
        out_dir=out_dir,
        run_dir=run_dir,
        seed_df=seed_df,
        aggregate_df=aggregate_df,
        hypothesis_df=auxiliary_gain,
        reason_df=reason_df,
    )

    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()

"""Confirmatory paired analysis for the frozen V8 cold-start experiment.

This script only reads development-evaluation results. It never trains a
model, performs acquisition, or accesses the final test set.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM = "GTFreeRandom"
VISUAL = "GTFreeDatasetBalancedVisualDiversity"
EXPECTED_SEEDS = list(range(52, 62))
METRICS = ["map5095", "map50", "precision", "recall", "f1"]


def exact_sign_flip_pvalue(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        means.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(means) >= observed - 1e-15))


def bootstrap_mean_ci(values: np.ndarray, seed: int = 20260714) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def bool_text(value: bool) -> str:
    return "PASS" if value else "FAIL"


def analyze(run_dir: Path) -> Path:
    config_path = run_dir / "config.json"
    results_path = run_dir / "all_round_results.csv"
    if not config_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"Missing config/results under {run_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if bool(config.get("dry_run", True)) or bool(config.get("selection_only", False)):
        raise RuntimeError("Confirmatory analysis requires completed training results, not dry-run/selection-only output.")
    development_eval_path = Path(str(config.get("development_eval_path", "")))
    if development_eval_path.stem != "development_eval_v7":
        raise RuntimeError(f"Unexpected evaluation split: {development_eval_path}")
    if list(config.get("acquisition_seeds", [])) != EXPECTED_SEEDS:
        raise RuntimeError(f"Unexpected seeds: {config.get('acquisition_seeds')}")
    if list(config.get("strategies", [])) != [RANDOM, VISUAL]:
        raise RuntimeError(f"Unexpected strategies: {config.get('strategies')}")
    if int(config.get("rounds", -1)) != 1 or list(config.get("budgets", [])) != [15, 20]:
        raise RuntimeError("Expected frozen one-round budgets [15, 20].")

    results = pd.read_csv(results_path)
    required = {"acquisition_seed", "strategy", "round", "labeled_budget", "train_status", "map5095", "map50", "precision", "recall"}
    missing = required.difference(results.columns)
    if missing:
        raise RuntimeError(f"Missing result columns: {sorted(missing)}")
    round1 = results[(results["round"] == 1) & (results["labeled_budget"] == 20)].copy()
    round1 = round1[round1["train_status"].astype(str).eq("success")]
    round1["acquisition_seed"] = pd.to_numeric(round1["acquisition_seed"], errors="raise").astype(int)
    for col in ["map5095", "map50", "precision", "recall"]:
        round1[col] = pd.to_numeric(round1[col], errors="raise")
    round1["f1"] = 2.0 * round1["precision"] * round1["recall"] / (round1["precision"] + round1["recall"]).replace(0, np.nan)

    counts = round1.groupby(["acquisition_seed", "strategy"]).size()
    expected_pairs = pd.MultiIndex.from_product([EXPECTED_SEEDS, [RANDOM, VISUAL]])
    if not counts.reindex(expected_pairs, fill_value=0).eq(1).all() or len(round1) != 20:
        raise RuntimeError("Expected exactly one successful round-1 result for each of 10 seeds and 2 strategies.")

    paired_rows: list[dict[str, float | int]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    pivots: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        pivot = round1.pivot(index="acquisition_seed", columns="strategy", values=metric).loc[EXPECTED_SEEDS]
        pivot["difference_visual_minus_random"] = pivot[VISUAL] - pivot[RANDOM]
        pivots[metric] = pivot
        values = pivot["difference_visual_minus_random"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_mean_ci(values)
        loo_means = np.asarray([np.delete(values, i).mean() for i in range(len(values))])
        summary_rows.append(
            {
                "metric": metric,
                "n_pairs": len(values),
                "random_mean": float(pivot[RANDOM].mean()),
                "visual_mean": float(pivot[VISUAL].mean()),
                "mean_difference": float(values.mean()),
                "median_difference": float(np.median(values)),
                "std_difference": float(values.std(ddof=1)),
                "bootstrap_ci95_low_descriptive": ci_low,
                "bootstrap_ci95_high_descriptive": ci_high,
                "exact_sign_flip_p_two_sided": exact_sign_flip_pvalue(values),
                "wins": int((values > 0).sum()),
                "losses": int((values < 0).sum()),
                "ties": int((values == 0).sum()),
                "leave_one_out_min_mean": float(loo_means.min()),
                "leave_one_out_max_mean": float(loo_means.max()),
            }
        )
    for seed in EXPECTED_SEEDS:
        row: dict[str, float | int] = {"acquisition_seed": seed}
        for metric in METRICS:
            row[f"random_{metric}"] = float(pivots[metric].loc[seed, RANDOM])
            row[f"visual_{metric}"] = float(pivots[metric].loc[seed, VISUAL])
            row[f"delta_{metric}"] = float(pivots[metric].loc[seed, "difference_visual_minus_random"])
        paired_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    map_row = summary.set_index("metric").loc["map5095"]
    recall_row = summary.set_index("metric").loc["recall"]
    gates = [
        ("mean_map5095_difference_at_least_0p01", float(map_row["mean_difference"]) >= 0.010),
        ("map5095_wins_at_least_7_of_10", int(map_row["wins"]) >= 7),
        ("map5095_exact_sign_flip_p_at_most_0p05", float(map_row["exact_sign_flip_p_two_sided"]) <= 0.05),
        ("mean_recall_difference_nonnegative", float(recall_row["mean_difference"]) >= 0.0),
        ("map5095_leave_one_out_min_mean_positive", float(map_row["leave_one_out_min_mean"]) > 0.0),
    ]
    gate_df = pd.DataFrame([{"check": name, "passed": passed} for name, passed in gates])
    overall = all(passed for _, passed in gates)

    paired.to_csv(run_dir / "confirmatory_paired_differences.csv", index=False)
    summary.to_csv(run_dir / "confirmatory_metric_summary.csv", index=False)
    gate_df.to_csv(run_dir / "confirmatory_gate.csv", index=False)

    lines = [
        "# Frozen V8 Visual Cold-Start Confirmation",
        "",
        "- Analysis: pre-registered paired acquisition-seed comparison",
        "- Evaluation: `development_eval_v7` only",
        "- Final test used: **False**",
        f"- Overall gate: **{bool_text(overall)}**",
        "",
        "## Primary result",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The bootstrap interval is descriptive. The exact paired sign-flip p-value is the pre-registered inferential check.",
        "",
        "## Per-seed paired results",
        "",
        paired.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Pre-registered gate",
        "",
        gate_df.assign(result=gate_df["passed"].map(bool_text)).drop(columns="passed").to_markdown(index=False),
        "",
        "All five checks must pass. A failure does not authorize selector tuning, YOLO replacement, round-2 expansion, or final-test access.",
    ]
    output = run_dir / "v8_cold_start_visual_confirmation_analysis.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ANALYSIS] gate={bool_text(overall)}")
    print(f"[SUMMARY] {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_dir.expanduser().resolve())


if __name__ == "__main__":
    main()

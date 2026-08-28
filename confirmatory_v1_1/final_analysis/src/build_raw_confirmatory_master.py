"""Build a human-readable canonical index over all raw confirmatory result tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_common import FINAL_EVALUATION, FINAL_LOCK, ROOT, sha256_file


RESULTS = ROOT / "final_analysis" / "results"
MASTER = ROOT / "final_analysis" / "RAW_CONFIRMATORY_RESULTS_MASTER_FINAL_20260828.md"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> list[str]:
    output = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        output.append("| " + " | ".join(str(row[key]) for key, _ in columns) + " |")
    return output


def main() -> None:
    final_audit = json.loads((FINAL_EVALUATION / "FINAL_EVALUATION_INTEGRITY_AUDIT.json").read_text(encoding="utf-8"))
    analysis_audit = json.loads((RESULTS / "CONFIRMATORY_ANALYSIS_INTEGRITY_AND_SUMMARY.json").read_text(encoding="utf-8"))
    authorization = json.loads((FINAL_LOCK / "FINAL_EVALUATION_AUTHORIZATION.json").read_text(encoding="utf-8"))
    primary = read_csv("primary_summary_2.csv")
    deltas = read_csv("primary_acquisition_deltas_20.csv")
    curves = read_csv("budget_curve_summary.csv")
    secondary = read_csv("secondary_summary.csv")
    database = ROOT / "full_run" / "runtime" / "full_run.sqlite"
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        runtime = connection.execute(
            "SELECT COUNT(*), SUM(train_runtime_seconds), SUM(development_eval_runtime_seconds), "
            "AVG(train_runtime_seconds), MIN(train_runtime_seconds), MAX(train_runtime_seconds) FROM cells"
        ).fetchone()
        attempt_counts = connection.execute("SELECT status, COUNT(*) FROM attempts GROUP BY status ORDER BY status").fetchall()
    finally:
        connection.close()

    lines = [
        "# Raw Confirmatory Results Master — Final 2026-08-28",
        "",
        "This file is the human-readable final confirmatory result canon. It does not mutate or mix with the historical `RAW_EXPERIMENT_RESULTS_MASTER_20260822.md` or the Phase 2B operational master.",
        "",
        "## Authorization and integrity",
        "",
        f"- authorization status: `{authorization['status']}`",
        f"- protocol SHA-256: `{authorization['protocol_sha256']}`",
        f"- freeze commit: `{authorization['freeze_commit_sha']}`",
        f"- cell registry SHA-256: `{authorization['cell_registry_sha256']}`",
        f"- attempt registry SHA-256: `{authorization['attempt_registry_sha256']}`",
        f"- final evaluation integrity: `{final_audit['status']}`",
        f"- completed locked-final evaluations: `{final_audit['completed_evaluations']}/{final_audit['planned_evaluations']}`",
        f"- raw final metric registry SHA-256: `{final_audit['raw_final_metric_registry_sha256']}`",
        f"- analysis integrity: `{analysis_audit['status']}`",
        f"- development substitutions / interpolated primary points: `{analysis_audit['development_metric_substitutions']}/{analysis_audit['missing_or_interpolated_primary_points']}`",
        "",
        "## Execution facts",
        "",
        f"- scientific training cells: `{runtime[0]}`",
        f"- cumulative training runtime: `{runtime[1] / 3600:.3f}` hours",
        f"- cumulative development-evaluation runtime: `{runtime[2] / 3600:.3f}` hours",
        f"- mean/min/max training runtime: `{runtime[3]:.3f}/{runtime[4]:.3f}/{runtime[5]:.3f}` seconds",
        f"- attempt states: `{dict(attempt_counts)}`",
        "- infrastructure retries are not additional scientific replicates.",
        "",
        "## Primary dataset summaries",
        "",
    ]
    lines += table(primary, [
        ("dataset", "dataset"), ("mean_dino_minus_random", "mean delta"),
        ("ci95_low", "CI low"), ("ci95_high", "CI high"),
        ("wins", "wins"), ("losses", "losses"),
        ("exact_sign_flip_p_raw", "raw p"), ("holm_adjusted_p", "Holm p"),
    ])
    lines += ["", "## All 20 primary acquisition-seed contrasts", ""]
    lines += table(deltas, [
        ("dataset", "dataset"), ("acquisition_index", "acquisition"),
        ("acquisition_seed", "seed"),
        ("random_two_training_seed_mean", "Random AULC"),
        ("dino_two_training_seed_mean", "DINO AULC"),
        ("dino_minus_random", "DINO-Random"),
    ])
    lines += ["", "## Full budget-curve summary", ""]
    lines += table(curves, [
        ("dataset", "dataset"), ("round", "round"), ("budget_fraction", "fraction"),
        ("random_mean_map50_95", "Random"), ("dino_mean_map50_95", "DINO"),
        ("mean_dino_minus_random", "delta"), ("delta_ci95_low", "CI low"),
        ("delta_ci95_high", "CI high"),
    ])
    lines += ["", "## Secondary/subgroup summaries", ""]
    lines += table(secondary, [
        ("dataset", "dataset"), ("metric", "metric"),
        ("mean_dino_minus_random", "mean delta"),
        ("ci95_low", "CI low"), ("ci95_high", "CI high"),
        ("wins", "wins"), ("losses", "losses"),
    ])
    lines += [
        "",
        "## Canonical raw files",
        "",
        "The tables below remain separate because flattening 440 checkpoint evaluations, 480 conceptual rows, 80 curves, and 240 selection states into this Markdown would make independent audit harder.",
        "",
    ]
    for path in sorted(RESULTS.glob("*")):
        if path.is_file():
            lines.append(f"- `{path.name}` — SHA-256 `{sha256_file(path)}`")
    lines += ["", "## Preservation rules", "", "- Null, negative, subgroup-harm and failed-attempt rows are retained.", "- Raw locked-final values are never replaced by development metrics.", "- GC10 and NEU are not pooled as naive `n=20`.", "- Historical evidence is not added to the confirmatory sample size.", ""]
    MASTER.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(str(MASTER))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Check that the curated MacBook handoff package is present.

This script is intentionally dependency-free. It does not train models, read raw
datasets, or touch final-test data. It only verifies committed documentation
artifacts and prints a compact continuation summary.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    REPO_ROOT
    / "docs"
    / "results"
    / "v10b_seed42_documentation_20260712_215841"
)

REQUIRED_FILES = [
    "docs/README.md",
    "docs/macbook_handoff_guide_20260712.md",
    "docs/research_context_handoff_20260712.md",
    "docs/continuation_playbook_20260712.md",
    "docs/preview.html",
    "docs/vlm_gt_free_al_workflow.html",
    "docs/final_detector_aware_pivot_protocol_20260712.md",
    "docs/v8_neu_only_5seed_result_log_20260712.md",
    "docs/v9_detector_aware_reimplementation_plan_20260712.md",
    "docs/results/v10b_seed42_documentation_20260712_215841/V10b_Seed42_Development_Gate_Updated_Report.docx",
    "docs/results/v10b_seed42_documentation_20260712_215841/V10b_Seed42_Development_Gate_Updated_Report.md",
    "docs/results/v10b_seed42_documentation_20260712_215841/report_recalculated_metrics.csv",
    "docs/results/v10b_seed42_documentation_20260712_215841/tables/table01_aggregate_detector_performance.csv",
    "docs/results/v10b_seed42_documentation_20260712_215841/tables/table03_per_class_v10b_minus_random.csv",
    "scripts/docs/macbook_open_docs.sh",
]

REQUIRED_DIRS = [
    "docs/results/v10b_seed42_documentation_20260712_215841/figures",
    "docs/results/v10b_seed42_documentation_20260712_215841/tables",
    "scripts/02_active_learning",
]


def _status_line(ok: bool, label: str) -> str:
    return f"[{'OK' if ok else 'MISSING'}] {label}"


def check_paths(paths: Iterable[str], expect_dir: bool = False) -> list[str]:
    missing: list[str] = []
    for rel in paths:
        path = REPO_ROOT / rel
        ok = path.is_dir() if expect_dir else path.is_file()
        print(_status_line(ok, rel))
        if not ok:
            missing.append(rel)
    return missing


def print_seed42_metrics() -> None:
    metrics_path = REPORT_DIR / "report_recalculated_metrics.csv"
    if not metrics_path.is_file():
        print("\nSeed42 metric summary unavailable: report_recalculated_metrics.csv missing")
        return

    wanted = {
        ("GTFreeRandom", "map5095"): "Random mAP50-95",
        ("DetectorInstanceRichDINOBalanced", "map5095"): "V9b mAP50-95",
        ("DetectorUncertaintyDINOInstanceReducedV10b", "map5095"): "V10b mAP50-95",
        ("DetectorUncertaintyDINOInstanceReducedV10b", "precision"): "V10b precision",
        ("DetectorUncertaintyDINOInstanceReducedV10b", "recall"): "V10b recall",
    }
    found: dict[str, str] = {}

    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("strategy", ""), row.get("metric", ""))
            if key in wanted:
                found[wanted[key]] = row.get("audit_value", row.get("source_value", ""))

    print("\nSeed42 development-gate summary:")
    for label in [
        "Random mAP50-95",
        "V9b mAP50-95",
        "V10b mAP50-95",
        "V10b precision",
        "V10b recall",
    ]:
        value = found.get(label, "n/a")
        print(f"  - {label}: {value}")


def print_continuation_notes() -> None:
    print("\nLatest independent-seed interpretation:")
    print("  - V10b seed43~46 paired mean mAP50-95 diff vs Random: +0.000949")
    print("  - wins/losses/ties: 3/1/0")
    print("  - precision mean diff: +0.023781")
    print("  - recall mean diff: -0.026170")
    print("  - final test used: False")
    print("  - conclusion: promising detector-aware direction, not a proven Random replacement")

    print("\nOpen these first on MacBook:")
    print("  - docs/preview.html")
    print("  - docs/vlm_gt_free_al_workflow.html")
    print("  - docs/macbook_handoff_guide_20260712.md")
    print("  - docs/research_context_handoff_20260712.md")
    print("  - docs/results/v10b_seed42_documentation_20260712_215841/V10b_Seed42_Development_Gate_Updated_Report.md")


def main() -> int:
    print(f"Repository root: {REPO_ROOT}")
    print("\nChecking required handoff files:")
    missing_files = check_paths(REQUIRED_FILES, expect_dir=False)

    print("\nChecking required handoff directories:")
    missing_dirs = check_paths(REQUIRED_DIRS, expect_dir=True)

    print_seed42_metrics()
    print_continuation_notes()

    if missing_files or missing_dirs:
        print("\nHandoff package status: INCOMPLETE")
        return 1

    print("\nHandoff package status: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Integrity tests for the prospective DeepPCB selection-only branch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs" / "deeppcb_reference_residual_gate" / "prospective_main_20260718"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    decision = json.loads((run / "gate_decision.json").read_text(encoding="utf-8"))
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    selected = pd.read_csv(run / "frozen_selected_images.csv")
    scores = pd.read_csv(run / "gt_free_reference_residual_scores.csv")
    groups = pd.read_csv(run / "posthoc_group_metrics.csv")
    checks = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, bool(ok), detail))

    required = [
        "trainval_manifest_no_gt.csv", "gt_free_reference_residual_scores.csv",
        "frozen_selected_images.csv", "frozen_selection_sha256.txt",
        "posthoc_group_metrics.csv", "posthoc_class_distribution.csv",
        "posthoc_box_audit.csv", "gate_decision.json", "config.json",
        "deeppcb_reference_residual_gate_summary.md",
    ]
    check("required_outputs", all((run / x).exists() for x in required), f"files={len(required)}")
    check("selection_hash", sha256(run / "frozen_selected_images.csv") == decision["selection_sha256"], "selection immutable")
    check("official_test_locked", decision["split_lock"]["official_test_content_read"] is False and decision["split_lock"]["official_test_annotation_read"] is False, "official test not read")
    check("reserved_development", decision["split_lock"]["reserved_development_group"] == "group92000", "group92000 reserved")
    check("eligible_groups", set(groups["group"]) == set(decision["split_lock"]["eligible_groups"]), f"groups={len(groups)}")
    check("no_gt_in_score_file", not any("class" in c.lower() or "instance" in c.lower() or "annotation" in c.lower() for c in scores.columns), f"columns={list(scores.columns)}")
    check("selection_subset", set(selected["canonical_path"]).issubset(set(scores["canonical_path"])), f"selected={len(selected)}")
    check("query_fraction", all(abs(len(g[g['canonical_path'].isin(set(selected['canonical_path']))]) - __import__('math').ceil(0.2 * len(g))) == 0 for _, g in scores.groupby('group')), "ceil(20%) per group")
    check("decision_consistency", (decision["decision"] == "PASS_SELECTION_ONLY") == all(decision["checks"].values()), decision["decision"])
    check("training_false", config["training_performed"] is False and decision["training_performed"] is False, "training=false")
    check("inference_false", config["detector_inference_performed"] is False and decision["detector_inference_performed"] is False, "detector inference=false")
    check("final_false", config["official_test_used"] is False and config["final_test_used"] is False, "official/final=false")
    failed = [x for x in checks if not x[1]]
    lines = [f"passed={len(checks)-len(failed)}", f"failed={len(failed)}"]
    lines.extend(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}" for name, ok, detail in checks)
    (run / "test_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS" if not failed else "FAIL",
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "decision": decision["decision"],
        "training_performed": False,
        "detector_inference_performed": False,
        "official_test_used": False,
        "final_test_used": False,
    }, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


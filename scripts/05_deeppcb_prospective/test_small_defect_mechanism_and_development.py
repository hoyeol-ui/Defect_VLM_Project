#!/usr/bin/env python3
"""Integrity tests for the DeepPCB mechanism and development-only audits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "runs" / "deeppcb_reference_residual_gate" / "prospective_main_20260718"
MECHANISM = ROOT / "runs" / "deeppcb_small_defect_mechanism_audit"
DEVELOPMENT = ROOT / "runs" / "deeppcb_reference_residual_development"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    original = load_json(ORIGINAL / "gate_decision.json")
    phase_a = load_json(MECHANISM / "phase_a_decision.json")
    candidate = load_json(DEVELOPMENT / "frozen_candidate.json")
    mechanism_effects = pd.read_csv(MECHANISM / "macro_micro_effects.csv", encoding="utf-8-sig")
    grounding = pd.read_csv(MECHANISM / "spatial_grounding.csv", encoding="utf-8-sig")
    comparison = pd.read_csv(DEVELOPMENT / "method_comparison.csv", encoding="utf-8-sig")
    tests = {
        "original_decision_remains_fail_stop": original["decision"] == "FAIL_STOP",
        "original_authorization_remains_stop": original["authorization"] == "STOP",
        "phase_a_does_not_modify_parent": phase_a["original_decision"] == "FAIL_STOP" and phase_a["original_decision_unchanged"],
        "phase_a_has_one_valid_decision": phase_a["phase_a_decision"] in {"A1_MECHANISM_SUPPORTED", "A2_MECHANISM_AMBIGUOUS", "A3_ARTIFACT_DOMINATED_STOP"},
        "six_mechanism_groups": len(mechanism_effects) == 6 and mechanism_effects["group"].nunique() == 6,
        "grounding_has_small_and_non_small": set(grounding["size_label"]) == {"small", "non_small"},
        "exactly_three_development_methods": set(comparison["method"]) == {"M1_ABSDIFF", "M2_SSIM", "M3_FUSED"},
        "development_result_scoped": candidate["result"] in {"FROZEN_CANDIDATE_FOR_EXTERNAL_CONFIRMATION", "NO_CANDIDATE"},
        "development_not_confirmatory": not candidate.get("confirmatory_evidence", False),
        "detector_screen_not_allowed": not candidate.get("detector_screen_allowed", False),
        "mechanism_no_training": not phase_a["training_performed"],
        "mechanism_no_detector_inference": not phase_a["detector_inference_performed"],
        "mechanism_no_official_test": not phase_a["official_test_used"],
        "mechanism_no_final_test": not phase_a["final_test_used"],
        "all_development_points_at_or_below_one_fppi": bool((comparison["false_positives_per_image"] <= 1.0 + 1e-12).all()),
        "mechanism_document_exists": (ROOT / "docs" / "deeppcb_frozen_small_defect_mechanism_audit.md").exists(),
        "development_document_exists": (ROOT / "docs" / "deeppcb_reference_residual_development_benchmark.md").exists(),
        "feasibility_document_exists": (ROOT / "docs" / "paired_reference_external_confirmation_feasibility.md").exists(),
    }
    output = "\n".join(f"{'PASS' if value else 'FAIL'} {name}" for name, value in tests.items()) + "\n"
    (DEVELOPMENT / "combined_test_results.txt").write_text(output, encoding="utf-8")
    print(output, end="")
    print(f"[DONE] passed={sum(tests.values())} failed={len(tests) - sum(tests.values())}")
    if not all(tests.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

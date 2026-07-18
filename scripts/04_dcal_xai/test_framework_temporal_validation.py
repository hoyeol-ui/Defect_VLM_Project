from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
OUT = ROOT / "runs" / "framework_temporal_validation_20260718"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def one(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    matches = [item for item in rows if item[key] == value]
    assert len(matches) == 1, f"Expected one row where {key}={value}; got {len(matches)}"
    return matches[0]


def main() -> None:
    timeline = read_csv(DOCS / "framework_branch_timeline_20260718.csv")
    holdout = read_csv(DOCS / "framework_holdout_confusion_matrix_20260718.csv")
    costs = read_csv(DOCS / "framework_cost_avoidance_summary_20260718.csv")
    loo = read_csv(DOCS / "framework_leave_one_branch_out_20260718.csv")
    sources = read_csv(OUT / "source_registry.csv")
    decision = (DOCS / "framework_temporal_validation_decision_20260718.md").read_text(encoding="utf-8")
    config = json.loads((OUT / "audit_config.json").read_text(encoding="utf-8"))

    tests: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        tests.append((name, bool(condition), detail))

    # 1. Branch, not image or selection seed, is the inferential unit.
    check("branch_is_inferential_unit", len(timeline) == 15 and config["inferential_unit"] == "research_branch", f"branches={len(timeline)}")

    # 2-3. The split is deterministic chronology, not random.
    ordered = sorted(timeline, key=lambda item: datetime.strptime(item["outcome_time"], "%Y-%m-%d %H:%M:%S"))
    expected_dev = {item["branch_id"] for item in ordered[:9]}
    actual_dev = {item["branch_id"] for item in timeline if item["chronology_partition"] == "development_chronology"}
    actual_holdout = {item["branch_id"] for item in timeline if item["chronology_partition"] == "temporal_holdout"}
    check("chronological_not_random_split", expected_dev == actual_dev, f"development={sorted(actual_dev)}")
    check("split_is_65_35_floor", len(actual_dev) == 9 and len(actual_holdout) == 6, f"{len(actual_dev)}/{len(actual_holdout)}")

    # 4. Generic policy was not retroactively marked as pre-frozen.
    check("generic_policy_not_prefrozen", all(item["generic_policy_precommitted"] == "False" for item in timeline), "all 15=False")

    # 5. Local gate artifacts precede their recorded summary/outcome artifacts.
    local_order_ok = all(datetime.strptime(item["protocol_time"], "%Y-%m-%d %H:%M:%S") < datetime.strptime(item["outcome_time"], "%Y-%m-%d %H:%M:%S") for item in timeline)
    check("local_gate_artifact_precedes_recorded_outcome", local_order_ok, "local timestamps only, not independent preregistration")
    explicit = [item for item in timeline if item["protocol_evidence_status"] == "EXPLICIT_PROTOCOL_BEFORE_OUTCOME"]
    script_only = [item for item in timeline if item["protocol_evidence_status"] == "LOCAL_SCRIPT_BEFORE_SUMMARY_ONLY"]
    check("protocol_evidence_not_overstated", len(explicit) == 11 and len(script_only) == 4 and config["explicit_protocol_branches"] == 11 and config["local_script_only_branches"] == 4 and "11/15" in decision, f"explicit={len(explicit)}, script_only={len(script_only)}")

    # 6-7. No pseudo-confusion matrix or downstream leakage.
    predictive_fields = ("correct_early_stop", "false_advance", "possible_false_stop")
    check("predictive_metrics_remain_na", len(holdout) == 6 and all(item["requested_predicted_decision"] == "NOT_IDENTIFIABLE" and all(item[field] == "NA" for field in predictive_fields) for item in holdout), f"holdout={len(holdout)}")
    check("downstream_outcome_not_gate_input", all(item["threshold"] and item["actual_later_outcome"] not in item["threshold"] for item in timeline), "threshold field is separate from later outcome")

    # 8. No result-driven gate promotion.
    failed = [item for item in timeline if item["local_gate"] == "FAIL"]
    check("failed_branches_not_upgraded", bool(failed) and all(item["workflow_action"].startswith("STOP") for item in failed), f"failed={len(failed)}")

    # 9. Selection PASS authorizes a bounded screen and is not labeled a false advance.
    b06 = one(timeline, "branch_id", "B06")
    b13 = one(timeline, "branch_id", "B13")
    h13 = one(holdout, "branch_id", "B13")
    check("selection_pass_means_bounded_screen", all(item["workflow_action"] == "ADVANCE_TO_BOUNDED_SCREEN" for item in (b06, b13)), "B06+B13")
    check("bounded_screen_not_false_advance", h13["false_advance"] == "NA" and "false advance" in decision.lower(), h13["false_advance"])

    # 10. Stopped branches have no invented counterfactual downstream truth.
    stopped = [item for item in timeline if item["workflow_action"].startswith("STOP") and item["actual_outcome_class"] == "COUNTERFACTUAL_UNOBSERVED"]
    check("stopped_counterfactual_unobserved", len(stopped) >= 8 and all("not observed" in item["actual_later_outcome"] for item in stopped), f"unobserved={len(stopped)}")

    # 11-13. Cost accounting is explicit, non-overlapping, and final-test-safe.
    components = [item for item in costs if item["metric"] == "prevented_detector_model_runs"]
    total = one(costs, "metric", "prevented_detector_model_runs_total")
    check("prevented_runs_not_double_counted", sum(int(item["value"]) for item in components) == int(total["value"]) == 45 and len(components) == 2, "15+30=45")
    final_actual = one(costs, "metric", "actual_locked_final_test_uses")
    final_counterfactual = one(costs, "metric", "counterfactual_final_test_uses_avoided")
    check("actual_final_test_zero", final_actual["value"] == "0" and config["final_test_used"] is False, final_actual["value"])
    check("final_uses_avoided_not_estimated", final_counterfactual["value"] == "NA" and final_counterfactual["identifiability"] == "NOT_IDENTIFIABLE", final_counterfactual["value"])

    # 14. Positive-selector sensitivity and false-stop performance are not exaggerated.
    check("positive_selector_sensitivity_not_identifiable", all(item["possible_false_stop"] == "NA" for item in holdout) and "false-stop performance | NA" in decision, "NA retained")

    # 15. Every branch has protocol and outcome source locators; registry entries exist.
    check("all_branches_have_source_locators", all(item["protocol_source_file"] and item["source_file"] for item in timeline), f"branches={len(timeline)}")
    missing_sources = [item["path"] for item in sources if item["exists"] != "True"]
    check("registered_sources_exist", not missing_sources, ", ".join(missing_sources) if missing_sources else f"sources={len(sources)}")

    # 16. Leave-one-branch-out sensitivity is claim-specific and not universally promoted.
    unstable = [item for item in loo if item["stable_after_removal"] == "False"]
    check("loo_reports_fragile_claims", any(item["removed_branch_id"] == "B15" and item["claim"] == "signal_ranking_not_operational_utility" for item in unstable) and any(item["removed_branch_id"] in {"B02", "B03"} and item["claim"] == "training_stability_not_acquisition_generalization" for item in unstable), f"unstable={len(unstable)}")

    # 17-18. This audit does not overwrite the thesis-reframe decision or claim predictive validation.
    check("decision_is_retrospective_only", config["decision"] == "C_RETROSPECTIVE_AUDIT_ONLY" and "C. RETROSPECTIVE_AUDIT_ONLY" in decision, config["decision"])
    check("evidence_freeze_decision_unchanged", config["evidence_freeze_v2_decision_unchanged"] == "A_THESIS_REFRAME_STRONGLY_SUPPORTED", config["evidence_freeze_v2_decision_unchanged"])

    # 19. Figures and protected execution state.
    figures = [DOCS / "figures" / "framework_temporal_validation.png", DOCS / "figures" / "framework_advance_stop_timeline.png", DOCS / "figures" / "framework_cost_avoidance.png"]
    check("figures_exist", all(path.exists() and path.stat().st_size > 10_000 for path in figures), ", ".join(f"{path.name}:{path.stat().st_size if path.exists() else 0}" for path in figures))
    protected = not any(config[key] for key in ("training_performed", "inference_performed", "vlm_calls_performed", "embedding_extraction_performed", "selector_implementation_performed", "fn_screen_performed", "final_test_used"))
    check("no_new_compute_or_final", protected, "all protected flags=False")

    failures = [item for item in tests if not item[1]]
    lines = [f"Framework temporal validation tests: {len(tests)-len(failures)}/{len(tests)} PASS", ""]
    lines.extend(f"{'PASS' if passed else 'FAIL'}\t{name}\t{detail}" for name, passed, detail in tests)
    lines.extend(["", "TRAINING=False", "INFERENCE=False", "VLM_CALLS=False", "EMBEDDING_EXTRACTION=False", "SELECTOR_IMPLEMENTATION=False", "FN_SCREEN=False", "FINAL_TEST_USED=False"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

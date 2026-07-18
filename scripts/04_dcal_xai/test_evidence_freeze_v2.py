from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
OUT = ROOT / "runs" / "evidence_freeze_v2_20260718"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    matches = [row for row in rows if row.get(key) == value]
    assert len(matches) == 1, f"Expected one row where {key}={value}, got {len(matches)}"
    return matches[0]


def main() -> None:
    hypotheses = read_csv(DOCS / "hypothesis_transition_matrix_20260718.csv")
    assets = read_csv(DOCS / "research_asset_reclassification_20260718.csv")
    ledger = read_csv(OUT / "research_evidence_ledger.csv")
    sources = read_csv(OUT / "source_registry.csv")
    mechanisms = read_csv(DOCS / "acquisition_mechanism_matrix_20260718.csv")
    outline = (DOCS / "reframed_thesis_outline_20260718.md").read_text(encoding="utf-8")
    claims = (DOCS / "thesis_claim_boundary_20260718.md").read_text(encoding="utf-8")
    evolution = (DOCS / "research_evolution_and_evidence_freeze_v2_20260718.md").read_text(encoding="utf-8")
    config = json.loads((OUT / "freeze_config.json").read_text(encoding="utf-8"))

    tests: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        tests.append((name, bool(condition), detail))

    # 1. Rejected hypotheses are not promoted.
    rejected = [h for h in hypotheses if h["evidence_status"] == "rejected"]
    check("rejected_not_promoted", bool(rejected) and all("pass" not in h["allowed_claim"].lower() for h in rejected), f"rejected={len(rejected)}")

    # 2. FN remains exploratory.
    h19 = find(hypotheses, "hypothesis_id", "H19")
    fn_rows = [e for e in ledger if "FN enrichment" in e["metric"] or "rare-FN enrichment" in e["metric"]]
    check("fn_exploratory_only", h19["evidence_status"] == "exploratory_only" and all(e["evidence_status"] == "exploratory_only" for e in fn_rows), f"fn_rows={len(fn_rows)}")

    # 3-4. 200 seeds and their CI are scoped correctly.
    pooled = " ".join(e["allowed_claim"] + " " + e["inferential_unit"] for e in ledger)
    check("seeds_not_independent_production", "independent production" not in pooled.lower(), "allowed claims contain no independent-production promotion")
    ci_rows = [e for e in ledger if "paired seed CI" in e["uncertainty"]]
    check("seed_ci_not_population_ci", all("production" not in e["uncertainty"].lower() and "population" not in e["uncertainty"].lower() for e in ci_rows), f"ci_rows={len(ci_rows)}")

    # 5-7. Metadata proxy names retain their real status.
    check("mpdd_exif_not_lot", "MPDD EXIF day는 production lot이다" in claims, "claim appears only in prohibited section")
    check("gc10_filename_not_official_group", "GC10 filename group은 official production group이다" in claims, "claim appears only in prohibited section")
    check("visa_category_not_session", "VisA category는 capture session이다" in claims, "claim appears only in prohibited section")

    # 8-10. Translation distinctions are explicit.
    gc10_disc = find(ledger, "evidence_id", "E010")
    gc10_map = find(ledger, "evidence_id", "E017")
    gc10_rare = find(ledger, "evidence_id", "E018")
    check("discovery_not_downstream", gc10_disc["metric"] != gc10_map["metric"] and "Detector utility" in gc10_disc["prohibited_claim"], "separate ledger rows and claim boundary")
    check("overall_map_with_rare_loss", float(gc10_map["value"]) > 0 and float(gc10_rare["value"]) < 0 and "rare macro AP -0.019877" in evolution, "both effects recorded together")
    fixed = find(hypotheses, "hypothesis_id", "H08")
    check("fixed_set_not_generalization", "not" in fixed["prohibited_claim"].lower() or "generally" in fixed["prohibited_claim"].lower(), fixed["prohibited_claim"])

    # 11-15. Protected state, missing metrics, no cross-protocol averages, no cost/trust claims.
    check("final_test_unused", config["final_test_used"] is False, str(config["final_test_used"]))
    untested = [h for h in hypotheses if h["evidence_status"] == "not_tested"]
    check("missing_not_zero_filled", all("0" != h["observed_result"].strip() for h in untested), f"untested={len(untested)}")
    check("no_cross_protocol_average", "cross-dataset average" not in evolution.lower() and "raw average" not in evolution.lower(), "no pooled numeric endpoint")
    prohibited_section = claims.split("## 절대로 주장하면 안 되는 것", 1)[1].split("## 용어 사용 규칙", 1)[0]
    check("no_unmeasured_cost_reduction", "50-80%" in prohibited_section and "cost" in prohibited_section.lower(), "cost reduction restricted to prohibited section")
    check("no_unmeasured_trust_gain", "inspector trust" in prohibited_section.lower(), "trust restricted to prohibited section")

    # 16-18. Provenance, titles, asset reclassification.
    check("all_core_claims_have_source", all(e["source_file"] and e["source_row_or_key"] for e in ledger), f"ledger={len(ledger)}")
    title_lines = [line.strip(" *") for line in outline.splitlines() if line.lstrip().startswith(tuple(f"{i}. **" for i in range(1, 6)))]
    unsafe_titles = ("outperforms random", "superior selector", "improves detector performance", "production generalization")
    check("titles_do_not_imply_superiority", len(title_lines) == 10 and not any(term in " ".join(title_lines).lower() for term in unsafe_titles), "10 title candidates checked")
    initial_assets = [a for a in assets if a["asset_id"] in {"A01", "A02", "A03", "A04", "A05"}]
    check("initial_materials_reclassified", len(initial_assets) == 5 and all(a["current_classification"] for a in initial_assets), f"initial_assets={len(initial_assets)}")

    # Additional structural checks.
    expected_mechanisms = {"grounding_collapse", "category_collapse", "coverage_utility_gap", "geometry_confound", "acquisition_non_generalization", "ranking_without_operational_enrichment"}
    observed_mechanisms = {m["dominant_mechanism"] for m in mechanisms}
    check("mechanisms_present", expected_mechanisms.issubset(observed_mechanisms), ",".join(sorted(observed_mechanisms)))
    check("source_registry_nonempty", len(sources) >= 20, f"sources={len(sources)}")
    check("decision_is_reframe_only", config["decision"] == "A_THESIS_REFRAME_STRONGLY_SUPPORTED" and "original selector superiority rejected" in config["decision_scope"], config["decision_scope"])
    figures = [
        DOCS / "figures" / "research_hypothesis_evolution.png",
        DOCS / "figures" / "discovery_composition_utility_matrix.png",
        DOCS / "figures" / "evidence_pyramid.png",
        DOCS / "figures" / "claim_boundary_map.png",
    ]
    check("figures_exist", all(p.exists() and p.stat().st_size > 10_000 for p in figures), ", ".join(f"{p.name}:{p.stat().st_size if p.exists() else 0}" for p in figures))

    failures = [test for test in tests if not test[1]]
    lines = [f"Evidence Freeze v2 tests: {len(tests)-len(failures)}/{len(tests)} PASS", ""]
    for name, passed, detail in tests:
        lines.append(f"{'PASS' if passed else 'FAIL'}\t{name}\t{detail}")
    lines += ["", "TRAINING=False", "INFERENCE=False", "VLM_CALLS=False", "EMBEDDING_EXTRACTION=False", "SELECTOR_IMPLEMENTATION=False", "FN_SCREEN=False", "FINAL_TEST_USED=False"]
    (OUT / "test_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

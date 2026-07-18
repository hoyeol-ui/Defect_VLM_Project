from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
OUT = ROOT / "runs" / "evidence_freeze_v3_20260718"
PARENT = ROOT / "runs" / "deeppcb_reference_residual_gate" / "prospective_main_20260718"
PHASE_A = ROOT / "runs" / "deeppcb_small_defect_mechanism_audit"
PHASE_B = ROOT / "runs" / "deeppcb_reference_residual_development"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(name: str, condition: bool, details: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: {details}")
    print(f"[PASS] {name}")


def main() -> None:
    parent = read_json(PARENT / "gate_decision.json")
    phase_a = read_json(PHASE_A / "phase_a_decision.json")
    candidate = read_json(PHASE_B / "frozen_candidate.json")
    config = read_json(OUT / "freeze_config.json")
    ledger = read_csv(OUT / "research_evidence_ledger.csv")
    new_rows = {row["evidence_id"]: row for row in ledger if row["evidence_id"] in {f"E{i:03d}" for i in range(35, 47)}}

    require("original DeepPCB FAIL_STOP retained", parent["decision"] == "FAIL_STOP")
    require("total enrichment retained", math.isclose(parent["aggregate"]["instance_enrichment_vs_random"]["mean"], 1.107693, rel_tol=0, abs_tol=5e-7))
    require("Phase A A2 retained", phase_a["phase_a_decision"] == "A2_MECHANISM_AMBIGUOUS")
    require("Phase B NO_CANDIDATE retained", candidate["result"] == "NO_CANDIDATE")
    require("detector authorization false", parent["authorization"] == "STOP" and not config["detector_screen_authorized"])
    require("external confirmation false", not config["external_confirmation_authorized"])
    require("training and inference zero", not parent["training_performed"] and not parent["detector_inference_performed"] and not config["training_performed"] and not config["detector_inference_performed"])
    require("official/final use false", not parent["official_test_used"] and not parent["final_test_used"] and not config["official_test_used"] and not config["final_test_used"])
    require("small signal remains exploratory", new_rows["E038"]["result"] == "EXPLORATORY" and new_rows["E038"]["evidence_status"] == "exploratory_only")
    require("dominant group 45.79 percent retained", math.isclose(float(new_rows["E040"]["value"]), 0.4579125, rel_tol=0, abs_tol=5e-7))
    require("577-1024 interpretation retained", "577-1024" in new_rows["E041"]["metric"] and "tiny" in new_rows["E041"]["prohibited_claim"].lower())
    require("no score or threshold rescue", not config["score_threshold_rescue_authorized"] and config["deeppcb_branch_closed"])
    require("no invented prevented-run increment", config["documented_prevented_detector_runs_lower_bound"] == 45 and config["deeppcb_prevented_run_count_added"] == 0)
    require("v3 decision A", config["decision"] == "A_THESIS_CORE_STRENGTHENED_BY_PROSPECTIVE_STOP")

    require("all E035-E046 rows present", len(new_rows) == 12)
    for evidence_id, row in sorted(new_rows.items()):
        source = ROOT / row["source_file"]
        require(f"{evidence_id} source locator", bool(row["source_file"].strip()) and bool(row["source_row_or_key"].strip()) and source.exists(), str(source))

    paper = (DOCS / "mini_paper_validity_gated_industrial_al_v2_20260718.md").read_text(encoding="utf-8")
    closure = (DOCS / "deeppcb_branch_closure_20260718.md").read_text(encoding="utf-8")
    boundary = (DOCS / "thesis_claim_boundary_v3_20260718.md").read_text(encoding="utf-8")
    readiness = (DOCS / "thesis_defense_readiness_v3_20260718.md").read_text(encoding="utf-8")
    require("paper DeepPCB subsection", "## 6.4 Prospective External Authorization Case: DeepPCB" in paper)
    require("paper defense non-regression subsection", "## 8.5 학위논문 방어를 위한 수용 조건과 보완책" in paper)
    require("closure forbids further DeepPCB experiment", "Prohibited next action" in closure and "threshold 32" in closure and "추가 candidate search" in closure)
    require("claim boundary is three-level", all(word in boundary for word in ("## Supported", "## Exploratory only", "## Rejected or closed")))
    require("readiness requires advisor contribution agreement", "지도교수" in readiness and "원점 회귀" in readiness)

    required = [
        DOCS / "research_evolution_and_evidence_freeze_v3_20260718.md",
        DOCS / "hypothesis_transition_matrix_v3_20260718.csv",
        DOCS / "acquisition_mechanism_matrix_v3_20260718.csv",
        DOCS / "thesis_claim_boundary_v3_20260718.md",
        DOCS / "mini_paper_validity_gated_industrial_al_v2_20260718.md",
        DOCS / "mini_paper_validity_gated_industrial_al_v2_20260718.docx",
        DOCS / "mini_paper_validity_gated_industrial_al_v2_20260718.pdf",
        DOCS / "advisor_decision_brief_deeppcb_closure_20260718.md",
        DOCS / "deeppcb_branch_closure_20260718.md",
        DOCS / "thesis_defense_readiness_v3_20260718.md",
        DOCS / "figures" / "deeppcb_prospective_stop_case.png",
        DOCS / "figures" / "full_page_validity_gated_architecture_v2.html",
        DOCS / "figures" / "full_page_validity_gated_architecture_v2.svg",
        DOCS / "figures" / "full_page_validity_gated_architecture_v2.png",
        DOCS / "figures" / "full_page_validity_gated_architecture_v2.pdf",
    ]
    require("all required artifacts exist", all(path.exists() and path.stat().st_size > 0 for path in required))
    with Image.open(DOCS / "figures" / "full_page_validity_gated_architecture_v2.png") as image:
        require("architecture is full-screen 16:9", image.size == (1920, 1080))

    print("[DONE] Evidence Freeze v3 integrity checks passed")
    print("[TRAINING] False")
    print("[INFERENCE] False")
    print("[OFFICIAL/FINAL TEST USED] False")


if __name__ == "__main__":
    main()

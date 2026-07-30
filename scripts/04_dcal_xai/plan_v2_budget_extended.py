"""Run the frozen V2.1 GC10 initial-budget extension without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import plan_v2_budget as base  # noqa: E402


EXTENDED_BUDGETS = [140, 160, 180, 200]
DEFAULT_OUT = ROOT / "runs" / "dcal_xai" / "v2_budget_extended"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=base.DEFAULT_PROTOCOL)
    parser.add_argument("--embedding-dir", type=Path, default=base.DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    protocol = args.protocol_dir.expanduser().resolve()
    embedding_dir = args.embedding_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    protocol_config_path = protocol / "gc10_protocol_config.json"
    embedding_config_path = embedding_dir / "embedding_config.json"
    protocol_config = json.loads(protocol_config_path.read_text(encoding="utf-8"))
    embedding_config = json.loads(embedding_config_path.read_text(encoding="utf-8"))
    if bool(protocol_config.get("final_test_evaluated", True)) or bool(embedding_config.get("final_test_used", True)):
        raise RuntimeError("Final-test safety flag failed")

    blind = pd.read_csv(protocol / "gc10_acquisition_pool_blind.csv").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    gt = pd.read_csv(protocol / "gc10_acquisition_pool_gt_audit.csv").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    boxes = pd.read_csv(protocol / "gc10_acquisition_bbox_gt_audit.csv")
    embedding_manifest = pd.read_csv(embedding_dir / "embedding_manifest.csv")
    embeddings = base.np.load(embedding_dir / "embeddings.npy")
    if blind["sample_id"].tolist() != gt["sample_id"].tolist() or blind["sample_id"].tolist() != embedding_manifest["sample_id"].tolist():
        raise RuntimeError("Manifest alignment failure")

    labels = base.fit_clusters(embeddings)
    original_budgets = base.BUDGETS
    base.BUDGETS = EXTENDED_BUDGETS
    try:
        design = base.evaluate_split(
            split_name="design", seeds=base.DESIGN_SEEDS, gt=gt, boxes=boxes,
            embeddings=embeddings, cluster_labels=labels,
        )
        holdout = base.evaluate_split(
            split_name="holdout", seeds=base.HOLDOUT_SEEDS, gt=gt, boxes=boxes,
            embeddings=embeddings, cluster_labels=labels,
        )
    finally:
        base.BUDGETS = original_budgets

    metrics = pd.concat([design, holdout], ignore_index=True)
    summary = base.add_random_differences(base.summarize(metrics))
    design_gate = base.gate_candidates(summary, "design")
    candidate = base.choose_design_candidate(design_gate)
    holdout_gate = base.gate_candidates(summary, "holdout")
    if candidate is None:
        chosen = None
        overall = False
    else:
        chosen = {"budget": int(candidate["budget"]), "policy": str(candidate["policy"])}
        match = holdout_gate[(holdout_gate["budget"] == chosen["budget"]) & holdout_gate["policy"].eq(chosen["policy"])]
        overall = len(match) == 1 and bool(match.iloc[0]["gate_pass"])

    metrics.to_csv(out / "budget_seed_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "budget_summary.csv", index=False, encoding="utf-8-sig")
    design_gate.to_csv(out / "design_gate.csv", index=False, encoding="utf-8-sig")
    holdout_gate.to_csv(out / "holdout_gate.csv", index=False, encoding="utf-8-sig")
    decision = {
        "status": "PASS" if overall else "FAIL",
        "chosen": chosen,
        "holdout_gate_pass": bool(overall),
        "thresholds": base.THRESHOLDS,
        "budgets": EXTENDED_BUDGETS,
        "policies": ["Random", *labels.keys()],
        "design_seeds": [base.DESIGN_SEEDS[0], base.DESIGN_SEEDS[-1]],
        "holdout_seeds": [base.HOLDOUT_SEEDS[0], base.HOLDOUT_SEEDS[-1]],
        "base_script_sha256": digest(HERE / "plan_v2_budget.py"),
        "extension_script_sha256": digest(Path(__file__).resolve()),
        "selector_used_gt": False,
        "gt_joined_posthoc": True,
        "detector_training_performed": False,
        "final_test_used": False,
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    columns = [
        "split", "budget", "policy", "all_classes_rate", "all_rare_classes_rate",
        "min_two_images_per_class_rate", "instances_mean",
        "unique_production_groups_mean", "pairwise_cosine_similarity_mean", "gate_pass",
    ]
    gates = pd.concat([design_gate, holdout_gate], ignore_index=True)
    chosen_text = "None" if chosen is None else f"{chosen['policy']} at budget {chosen['budget']}"
    report = [
        "# DCAL-XAI V2.1 Extended Initial Budget Decision", "",
        f"- Decision: **{'PASS' if overall else 'FAIL'}**",
        f"- Design-chosen candidate: **{chosen_text}**",
        f"- Holdout gate: **{'PASS' if overall else 'FAIL'}**",
        "- Detector training performed: **False**",
        "- Selector used GT/XML: **False**",
        "- GT/XML joined post-hoc: **True**",
        "- Final test used: **False**", "",
        "## Candidate gates", "", gates[columns].to_markdown(index=False, floatfmt=".4f"), "",
        "A PASS authorizes only a separately frozen backbone-stability experiment.",
    ]
    path = out / "budget_decision.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[DONE] {path}")
    print(f"[GATE] {'PASS' if overall else 'FAIL'}")
    print(f"[CHOSEN] {chosen_text}")
    print("Training performed: False")
    print("Final test used: False")


if __name__ == "__main__":
    main()


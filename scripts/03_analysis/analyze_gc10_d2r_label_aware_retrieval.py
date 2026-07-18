"""Evaluate label-aware DINO retrieval on the frozen 1,000 D2R states.

This is an off-policy, one-step post-hoc diagnostic. Hypothetical selections do
not alter later states. GT is joined only after ranking. No detector or final
split is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715"
DEFAULT_EMBEDDINGS = ROOT / "outputs" / "gc10_visual_embeddings" / "dinov2_small_protocol_20260715"
DEFAULT_D2R = ROOT / "runs" / "gc10_discovery_representation_audit" / "gc10_d2r_200seed_20260715"
METHOD_PRIORITY = [
    "contrastive_margin",
    "nearest_target_exemplar",
    "top3_target_similarity",
    "top5_target_similarity",
]
ALL_METHODS = ["centroid_similarity", *METHOD_PRIORITY]
TARGET_CLASS = 8
TOP_K_AUDIT = 32


def class_set(value: Any) -> set[int]:
    return {int(item) for item in str(value).split("|") if item and item != "nan"}


def bootstrap_ci(values: np.ndarray, seed: int = 20260715) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def top_mean(similarities: np.ndarray, k: int) -> np.ndarray:
    effective_k = min(k, similarities.shape[1])
    if effective_k == similarities.shape[1]:
        return similarities.mean(axis=1)
    partitioned = np.partition(similarities, similarities.shape[1] - effective_k, axis=1)
    return partitioned[:, -effective_k:].mean(axis=1)


def load_inputs(args: argparse.Namespace) -> tuple[Path, pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    protocol = args.protocol_dir.expanduser().resolve()
    embedding_dir = args.embedding_dir.expanduser().resolve()
    d2r_dir = args.d2r_dir.expanduser().resolve()
    protocol_config = json.loads((protocol / "gc10_protocol_config.json").read_text(encoding="utf-8"))
    embedding_config = json.loads((embedding_dir / "embedding_config.json").read_text(encoding="utf-8"))
    d2r_config = json.loads((d2r_dir / "config.json").read_text(encoding="utf-8"))
    if bool(protocol_config.get("final_test_evaluated", True)):
        raise RuntimeError("Protocol final-test safety flag failed")
    if bool(embedding_config.get("final_test_used", True)) or bool(d2r_config.get("final_test_used", True)):
        raise RuntimeError("Embedding/D2R final-test safety flag failed")
    if bool(d2r_config.get("overall_gate_pass", True)):
        raise RuntimeError("Expected the frozen D2R selection gate to be FAIL")

    blind = pd.read_csv(protocol / "gc10_acquisition_pool_blind.csv").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if blind.columns.tolist() != ["sample_id", "image_sha256", "phash64"]:
        raise RuntimeError("Unexpected selector-visible schema")
    gt_raw = pd.read_csv(protocol / "gc10_acquisition_pool_gt_audit.csv")
    gt = blind[["sample_id"]].merge(gt_raw, on="sample_id", how="left", validate="one_to_one")
    manifest = pd.read_csv(embedding_dir / "embedding_manifest.csv")
    embeddings = np.load(embedding_dir / "embeddings.npy").astype(np.float32, copy=False)
    if blind["sample_id"].astype(str).tolist() != manifest["sample_id"].astype(str).tolist():
        raise RuntimeError("Embedding order mismatch")
    if embeddings.shape != (len(blind), 384) or not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4):
        raise RuntimeError("Unexpected embedding shape or normalization")
    feedback = pd.read_csv(d2r_dir / "gc10_d2r_feedback_access_log.csv").sort_values(["acquisition_seed", "feedback_step"])
    rounds = pd.read_csv(d2r_dir / "gc10_d2r_representation_micro_rounds.csv").sort_values(["acquisition_seed", "micro_round"])
    if len(feedback) != 7000 or len(rounds) != 1000:
        raise RuntimeError("Frozen trajectory cardinality mismatch")
    return d2r_dir, blind, embeddings, gt, feedback, rounds


def evaluate(
    blind: pd.DataFrame, embeddings: np.ndarray, gt: pd.DataFrame,
    feedback: pd.DataFrame, rounds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = blind["sample_id"].astype(str).to_numpy()
    image_hashes = blind["image_sha256"].astype(str).to_numpy()
    id_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    gt_classes = [class_set(value) for value in gt["class_ids"]]
    all_indices = np.arange(len(blind), dtype=int)
    feedback_by_seed = {
        int(seed): [id_to_index[str(value)] for value in frame.sort_values("feedback_step")["sample_id"]]
        for seed, frame in feedback.groupby("acquisition_seed")
    }
    similarity_matrix = embeddings @ embeddings.T
    round_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []

    for ordinal, row in enumerate(rounds.itertuples(index=False), start=1):
        seed = int(row.acquisition_seed)
        micro_round = int(row.micro_round)
        target_class = int(row.target_class)
        known_indices = feedback_by_seed[seed][: 30 + micro_round - 1]
        if len(known_indices) != 30 + micro_round - 1:
            raise RuntimeError(f"Known-feedback replay failed seed={seed}, round={micro_round}")
        target_members = [index for index in known_indices if target_class in gt_classes[index]]
        non_target_members = [index for index in known_indices if target_class not in gt_classes[index]]
        if not target_members or not non_target_members:
            raise RuntimeError(f"Target/non-target feedback unavailable seed={seed}, round={micro_round}")
        known_set = set(known_indices)
        selected_hashes = set(image_hashes[known_indices])
        candidates = np.asarray([
            int(index) for index in all_indices
            if int(index) not in known_set and image_hashes[int(index)] not in selected_hashes
        ], dtype=int)
        target_similarities = similarity_matrix[np.ix_(candidates, np.asarray(target_members, dtype=int))]
        non_target_similarities = similarity_matrix[np.ix_(candidates, np.asarray(non_target_members, dtype=int))]
        max_target = target_similarities.max(axis=1)
        scores = {
            "centroid_similarity": target_similarities.mean(axis=1),
            "nearest_target_exemplar": max_target,
            "contrastive_margin": max_target - non_target_similarities.max(axis=1),
            "top3_target_similarity": top_mean(target_similarities, 3),
            "top5_target_similarity": top_mean(target_similarities, 5),
        }
        max_similarity_to_known = similarity_matrix[np.ix_(candidates, np.asarray(known_indices, dtype=int))].max(axis=1)
        novelty = 1.0 - max_similarity_to_known
        candidate_ids = sample_ids[candidates]

        for method in ALL_METHODS:
            order = np.lexsort((candidate_ids, -scores[method]))
            top_positions = order[:TOP_K_AUDIT]
            top_indices = candidates[top_positions]
            hits = np.asarray([target_class in gt_classes[int(index)] for index in top_indices], dtype=bool)
            chosen_position = int(top_positions[0])
            chosen_index = int(candidates[chosen_position])
            round_rows.append({
                "acquisition_seed": seed,
                "micro_round": micro_round,
                "target_class": target_class,
                "method": method,
                "target_exemplars_available": len(target_members),
                "non_target_exemplars_available": len(non_target_members),
                "selected_sample_id_diagnostic": sample_ids[chosen_index],
                "selected_class_ids_posthoc": "|".join(str(value) for value in sorted(gt_classes[chosen_index])),
                "top1_target_hit_posthoc": bool(hits[0]),
                "precision_at_5_posthoc": float(hits[:5].mean()),
                "precision_at_10_posthoc": float(hits[:10].mean()),
                "precision_at_32_posthoc": float(hits.mean()),
                "target_present_in_top32_posthoc": bool(hits.any()),
                "selected_score": float(scores[method][chosen_position]),
                "selected_max_target_similarity": float(max_target[chosen_position]),
                "selected_distance_to_known": float(novelty[chosen_position]),
                "diagnostic_only_not_new_selection": True,
            })
            for rank, (position, index) in enumerate(zip(top_positions, top_indices), start=1):
                rank_rows.append({
                    "acquisition_seed": seed,
                    "micro_round": micro_round,
                    "target_class": target_class,
                    "method": method,
                    "rank": rank,
                    "sample_id": sample_ids[int(index)],
                    "class_ids_posthoc": "|".join(str(value) for value in sorted(gt_classes[int(index)])),
                    "target_hit_posthoc": target_class in gt_classes[int(index)],
                    "score": float(scores[method][int(position)]),
                    "max_target_similarity": float(max_target[int(position)]),
                    "distance_to_known": float(novelty[int(position)]),
                })
        if ordinal % 100 == 0:
            print(f"[LABEL-AWARE DIAGNOSTIC] rounds={ordinal}/{len(rounds)}", flush=True)
    return pd.DataFrame(round_rows), pd.DataFrame(rank_rows)


def summarize(d2r_dir: Path, out: Path, rounds: pd.DataFrame, ranked: pd.DataFrame) -> tuple[str, str | None, Path]:
    method_summary = rounds.groupby("method").agg(
        rounds=("micro_round", "size"),
        top1_hit_rate=("top1_target_hit_posthoc", "mean"),
        precision_at_5=("precision_at_5_posthoc", "mean"),
        precision_at_10=("precision_at_10_posthoc", "mean"),
        precision_at_32=("precision_at_32_posthoc", "mean"),
        target_present_in_top32_rate=("target_present_in_top32_posthoc", "mean"),
        unique_selected_images=("selected_sample_id_diagnostic", "nunique"),
        selected_distance_to_known_mean=("selected_distance_to_known", "mean"),
    ).reset_index()
    target_summary = rounds.groupby(["method", "target_class"]).agg(
        rounds=("micro_round", "size"),
        top1_hit_rate=("top1_target_hit_posthoc", "mean"),
        precision_at_5=("precision_at_5_posthoc", "mean"),
        precision_at_10=("precision_at_10_posthoc", "mean"),
        precision_at_32=("precision_at_32_posthoc", "mean"),
        target_present_in_top32_rate=("target_present_in_top32_posthoc", "mean"),
        unique_selected_images=("selected_sample_id_diagnostic", "nunique"),
        selected_distance_to_known_mean=("selected_distance_to_known", "mean"),
    ).reset_index()
    class8 = target_summary[target_summary["target_class"].eq(TARGET_CLASS)].copy()
    class8["unique_selected_image_ratio"] = class8["unique_selected_images"] / class8["rounds"]
    ci_rows = []
    class8_rounds = rounds[rounds["target_class"].eq(TARGET_CLASS)]
    for method, frame in class8_rounds.groupby("method"):
        seed_rates = frame.groupby("acquisition_seed")["top1_target_hit_posthoc"].mean().to_numpy(float)
        low, high = bootstrap_ci(seed_rates)
        ci_rows.append({
            "method": method,
            "class8_seed_bootstrap_ci95_low": low,
            "class8_seed_bootstrap_ci95_high": high,
            "class8_acquisition_seeds": len(seed_rates),
        })
    class8 = class8.merge(pd.DataFrame(ci_rows), on="method")
    centroid_hit = float(class8.set_index("method").loc["centroid_similarity", "top1_hit_rate"])
    class8["top1_hit_improvement_over_centroid"] = class8["top1_hit_rate"] - centroid_hit
    overall_hit = method_summary.set_index("method")["top1_hit_rate"]

    gate_rows = []
    method_passes: dict[str, bool] = {}
    for method in METHOD_PRIORITY:
        row = class8.set_index("method").loc[method]
        checks = [
            ("class8_top1_hit_rate", float(row["top1_hit_rate"]), ">= 0.50", float(row["top1_hit_rate"]) >= 0.50),
            ("class8_seed_bootstrap_ci95_low", float(row["class8_seed_bootstrap_ci95_low"]), ">= 0.40", float(row["class8_seed_bootstrap_ci95_low"]) >= 0.40),
            ("class8_precision_at_5", float(row["precision_at_5"]), ">= 0.40", float(row["precision_at_5"]) >= 0.40),
            ("class8_target_present_in_top32_rate", float(row["target_present_in_top32_rate"]), ">= 0.80", float(row["target_present_in_top32_rate"]) >= 0.80),
            ("class8_hit_improvement_over_centroid", float(row["top1_hit_improvement_over_centroid"]), ">= 0.25", float(row["top1_hit_improvement_over_centroid"]) >= 0.25),
            ("class8_unique_selected_image_ratio", float(row["unique_selected_image_ratio"]), ">= 0.25", float(row["unique_selected_image_ratio"]) >= 0.25),
            ("overall_top1_hit_rate", float(overall_hit.loc[method]), ">= 0.55", float(overall_hit.loc[method]) >= 0.55),
        ]
        method_passes[method] = all(passed for _, _, _, passed in checks)
        gate_rows.extend({
            "method": method, "check": check, "observed": observed, "threshold": threshold, "passed": passed
        } for check, observed, threshold, passed in checks)
    gate = pd.DataFrame(gate_rows)
    eligible_method = next((method for method in METHOD_PRIORITY if method_passes[method]), None)
    decision = "RECOVERABLE" if eligible_method else "NOT_RECOVERABLE"

    confusion_rows = []
    for row in rounds.itertuples(index=False):
        for actual_class in sorted(class_set(row.selected_class_ids_posthoc)):
            confusion_rows.append({
                "method": row.method,
                "target_class": int(row.target_class),
                "actual_class_posthoc": actual_class,
                "selections": 1,
                "target_hit": int(row.target_class) in class_set(row.selected_class_ids_posthoc),
            })
    confusion = pd.DataFrame(confusion_rows).groupby(
        ["method", "target_class", "actual_class_posthoc"], as_index=False
    ).agg(selections=("selections", "sum"), hit_rows=("target_hit", "sum"))

    rounds.to_csv(out / "label_aware_round_diagnostics.csv", index=False, encoding="utf-8-sig")
    ranked.to_csv(out / "label_aware_top32_candidates_posthoc.csv", index=False, encoding="utf-8-sig")
    method_summary.to_csv(out / "label_aware_method_summary.csv", index=False, encoding="utf-8-sig")
    target_summary.to_csv(out / "label_aware_target_class_summary.csv", index=False, encoding="utf-8-sig")
    class8.to_csv(out / "label_aware_class8_summary.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(out / "label_aware_class8_recovery_gate.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(out / "label_aware_target_actual_confusion.csv", index=False, encoding="utf-8-sig")

    report = [
        "# GC10 D2R Label-Aware Retrieval Diagnostic", "",
        "- Frozen states evaluated: **1,000**",
        "- Alternative selections fed into later states: **False**",
        "- New selection experiment performed: **False**",
        "- Detector training performed: **False**",
        "- Final test used: **False**",
        f"- Class-8 recovery decision: **{decision}**",
        f"- Eligible D2R-v2 method: **{eligible_method or 'None'}**", "",
        "## Overall retrieval", "", method_summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## Class 8", "", class8.to_markdown(index=False, floatfmt=".6f"), "",
        "## Frozen recovery gate", "", gate.assign(result=gate["passed"].map({True: "PASS", False: "FAIL"})).drop(columns="passed").to_markdown(index=False, floatfmt=".6f"), "",
        "## All target classes", "", target_summary.to_markdown(index=False, floatfmt=".6f"), "",
        "A RECOVERABLE result authorizes only D2R-v2 preregistration and a new 200-seed selection-only audit. It never authorizes detector training.",
    ]
    summary_path = out / "label_aware_retrieval_summary.md"
    summary_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps({
        "source_d2r_dir": str(d2r_dir),
        "methods": ALL_METHODS,
        "method_priority": METHOD_PRIORITY,
        "target_class": TARGET_CLASS,
        "decision": decision,
        "eligible_d2r_v2_method": eligible_method,
        "off_policy_one_step_diagnostic": True,
        "new_selection_experiment_performed": False,
        "detector_training_performed": False,
        "final_test_used": False,
    }, indent=2), encoding="utf-8")
    return decision, eligible_method, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--d2r-dir", type=Path, default=DEFAULT_D2R)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    d2r_dir, blind, embeddings, gt, feedback, frozen_rounds = load_inputs(args)
    out = args.output_dir.expanduser().resolve() if args.output_dir else d2r_dir / "label_aware_retrieval_audit"
    out.mkdir(parents=True, exist_ok=True)
    round_diagnostics, ranked = evaluate(blind, embeddings, gt, feedback, frozen_rounds)
    decision, method, summary = summarize(d2r_dir, out, round_diagnostics, ranked)
    print("=" * 100)
    print("[DONE] GC10 D2R label-aware retrieval diagnostic")
    print(f"[DECISION] {decision}")
    print(f"[ELIGIBLE METHOD] {method or 'None'}")
    print(f"[SUMMARY] {summary}")
    print("New selection experiment performed: False")
    print("Detector training performed: False")
    print("Final test used: False")
    print("=" * 100)


if __name__ == "__main__":
    main()

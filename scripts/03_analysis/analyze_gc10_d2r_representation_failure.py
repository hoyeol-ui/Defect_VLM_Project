"""Post-hoc diagnostic of the failed GC10 D2R representation stage.

This script replays each of the 1,000 representation micro-rounds, reconstructs
the frozen 32-image DINO neighborhood, and joins GT only for diagnostic metrics.
It performs no selection experiment, detector training, or final-test access.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715"
DEFAULT_EMBEDDINGS = ROOT / "outputs" / "gc10_visual_embeddings" / "dinov2_small_protocol_20260715"
DEFAULT_D2R = ROOT / "runs" / "gc10_discovery_representation_audit" / "gc10_d2r_200seed_20260715"
LOCAL_K = 32
POLICIES = {
    "nearest_prototype": 0.0,
    "similarity_plus_0p25_novelty": 0.25,
    "similarity_plus_0p50_novelty": 0.50,
    "similarity_plus_1p00_novelty": 1.00,
}


def class_set(value: Any) -> set[int]:
    return {int(item) for item in str(value).split("|") if item and item != "nan"}


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise RuntimeError("Zero prototype encountered")
    return vector / norm


def bootstrap_ci(values: np.ndarray, seed: int = 20260715) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


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
        raise RuntimeError("This failure diagnostic expects the frozen D2R gate to be FAIL")

    blind = pd.read_csv(protocol / "gc10_acquisition_pool_blind.csv").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if blind.columns.tolist() != ["sample_id", "image_sha256", "phash64"]:
        raise RuntimeError("Unexpected selector-visible schema")
    gt_raw = pd.read_csv(protocol / "gc10_acquisition_pool_gt_audit.csv")
    gt = blind[["sample_id"]].merge(gt_raw, on="sample_id", how="left", validate="one_to_one")
    manifest = pd.read_csv(embedding_dir / "embedding_manifest.csv")
    embeddings = np.load(embedding_dir / "embeddings.npy")
    if blind["sample_id"].astype(str).tolist() != manifest["sample_id"].astype(str).tolist():
        raise RuntimeError("Embedding order mismatch")
    if embeddings.shape != (len(blind), 384) or not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4):
        raise RuntimeError("Unexpected DINO embedding shape or normalization")
    feedback = pd.read_csv(d2r_dir / "gc10_d2r_feedback_access_log.csv")
    rounds = pd.read_csv(d2r_dir / "gc10_d2r_representation_micro_rounds.csv")
    selections = pd.read_csv(d2r_dir / "gc10_d2r_selection_records_posthoc.csv")
    if len(feedback) != 7000 or len(rounds) != 1000 or len(selections) != 12000:
        raise RuntimeError("Frozen D2R output cardinality mismatch")
    return d2r_dir, blind, embeddings, gt, feedback, rounds


def replay_neighborhoods(
    blind: pd.DataFrame, embeddings: np.ndarray, gt: pd.DataFrame,
    feedback: pd.DataFrame, rounds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_to_index = {sample_id: index for index, sample_id in enumerate(blind["sample_id"].astype(str))}
    gt_classes = {index: class_set(value) for index, value in enumerate(gt["class_ids"])}
    all_indices = np.arange(len(blind), dtype=int)
    candidate_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []

    for ordinal, round_row in enumerate(rounds.sort_values(["acquisition_seed", "micro_round"]).itertuples(index=False), start=1):
        seed = int(round_row.acquisition_seed)
        micro_round = int(round_row.micro_round)
        target_class = int(round_row.target_class)
        selected_index = id_to_index[str(round_row.sample_id)]
        seed_feedback = feedback[
            (feedback["acquisition_seed"] == seed)
            & (
                feedback["reason"].isin(["initial_existing_label", "discovery_annotation_return"])
                | ((feedback["reason"] == "representation_annotation_return") & (feedback["feedback_step"] < int(30 + micro_round)))
            )
        ].sort_values("feedback_step")
        # Initial 20 + discovery 10 + representation selections from earlier micro-rounds.
        expected_known = 30 + micro_round - 1
        if len(seed_feedback) != expected_known:
            raise RuntimeError(f"Feedback replay mismatch seed={seed}, round={micro_round}: {len(seed_feedback)} != {expected_known}")
        known_indices = [id_to_index[str(value)] for value in seed_feedback["sample_id"]]
        target_members = [index for index in known_indices if target_class in gt_classes[index]]
        if not target_members:
            raise RuntimeError(f"Target class absent from known labels seed={seed}, round={micro_round}")
        prototype = normalize(embeddings[np.asarray(target_members)].mean(axis=0))
        selected_hashes = set(blind.iloc[known_indices]["image_sha256"].astype(str))
        known_set = set(known_indices)
        candidates = np.asarray([
            int(index) for index in all_indices
            if int(index) not in known_set
            and str(blind.iloc[int(index)]["image_sha256"]) not in selected_hashes
        ], dtype=int)
        similarities = embeddings[candidates] @ prototype
        candidate_ids = blind.iloc[candidates]["sample_id"].astype(str).to_numpy()
        similarity_order = np.lexsort((candidate_ids, -similarities))
        local_positions = similarity_order[:LOCAL_K]
        local_indices = candidates[local_positions]
        local_similarity = similarities[local_positions]
        max_similarity_to_labeled = (embeddings[local_indices] @ embeddings[np.asarray(known_indices)].T).max(axis=1)
        novelty = 1.0 - max_similarity_to_labeled
        local_ids = blind.iloc[local_indices]["sample_id"].astype(str).to_numpy()
        novelty_order = np.lexsort((local_ids, -novelty))
        replay_selected = int(local_indices[int(novelty_order[0])])
        if replay_selected != selected_index:
            raise RuntimeError(
                f"Frozen selection replay mismatch seed={seed}, round={micro_round}: "
                f"{blind.iloc[replay_selected]['sample_id']} != {round_row.sample_id}"
            )

        hit_vector = np.asarray([target_class in gt_classes[int(index)] for index in local_indices], dtype=bool)
        class_counts = Counter(
            class_id for index in local_indices for class_id in gt_classes[int(index)]
        )
        for position, index in enumerate(local_indices):
            candidate_rows.append({
                "acquisition_seed": seed,
                "micro_round": micro_round,
                "target_class": target_class,
                "candidate_rank_by_prototype_similarity": position + 1,
                "candidate_rank_by_novelty": int(np.flatnonzero(novelty_order == position)[0]) + 1,
                "sample_id": blind.iloc[int(index)]["sample_id"],
                "candidate_class_ids_posthoc": "|".join(str(value) for value in sorted(gt_classes[int(index)])),
                "target_hit_posthoc": bool(hit_vector[position]),
                "prototype_cosine_similarity": float(local_similarity[position]),
                "distance_to_labeled_set": float(novelty[position]),
                "was_frozen_selected": int(index) == selected_index,
                "neighborhood_target_purity": float(hit_vector.mean()),
                "neighborhood_has_target": bool(hit_vector.any()),
                "dominant_neighborhood_class_posthoc": int(class_counts.most_common(1)[0][0]),
            })

        policy_positions: dict[str, int] = {
            name: int(np.lexsort((local_ids, -(local_similarity + weight * novelty)))[0])
            for name, weight in POLICIES.items()
        }
        policy_positions["frozen_max_novelty"] = int(novelty_order[0])
        for policy, position in policy_positions.items():
            index = int(local_indices[position])
            policy_rows.append({
                "acquisition_seed": seed,
                "micro_round": micro_round,
                "target_class": target_class,
                "policy": policy,
                "sample_id": blind.iloc[index]["sample_id"],
                "candidate_class_ids_posthoc": "|".join(str(value) for value in sorted(gt_classes[index])),
                "target_hit_posthoc": target_class in gt_classes[index],
                "prototype_cosine_similarity": float(local_similarity[position]),
                "distance_to_labeled_set": float(novelty[position]),
                "diagnostic_only_not_new_selection": True,
            })
        if ordinal % 100 == 0:
            print(f"[REPLAY] micro-rounds={ordinal}/{len(rounds)}", flush=True)
    return pd.DataFrame(candidate_rows), pd.DataFrame(policy_rows)


def confusion_table(rounds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in rounds.itertuples(index=False):
        returned = class_set(row.returned_class_ids)
        for actual_class in sorted(returned):
            rows.append({
                "target_class": int(row.target_class),
                "actual_class_posthoc": actual_class,
                "micro_round_selections": 1,
                "target_hit": int(row.target_class) in returned,
            })
    frame = pd.DataFrame(rows)
    return frame.groupby(["target_class", "actual_class_posthoc"], as_index=False).agg(
        micro_round_selections=("micro_round_selections", "sum"),
        hit_rows=("target_hit", "sum"),
    )


def summarize(
    d2r_dir: Path, out: Path, gt: pd.DataFrame, rounds: pd.DataFrame,
    candidates: pd.DataFrame, policies: pd.DataFrame,
) -> tuple[str, Path]:
    round_neighborhood = candidates.groupby(["acquisition_seed", "micro_round", "target_class"], as_index=False).agg(
        neighborhood_target_purity=("neighborhood_target_purity", "first"),
        neighborhood_has_target=("neighborhood_has_target", "first"),
        dominant_neighborhood_class_posthoc=("dominant_neighborhood_class_posthoc", "first"),
    )
    frozen = policies[policies["policy"].eq("frozen_max_novelty")]
    target_summary = round_neighborhood.groupby("target_class").agg(
        micro_rounds=("micro_round", "size"),
        mean_neighborhood_target_purity=("neighborhood_target_purity", "mean"),
        neighborhoods_with_any_target_rate=("neighborhood_has_target", "mean"),
    ).reset_index()
    frozen_target = frozen.groupby("target_class").agg(
        frozen_target_hit_rate=("target_hit_posthoc", "mean"),
        frozen_similarity_mean=("prototype_cosine_similarity", "mean"),
        frozen_distance_to_labeled_mean=("distance_to_labeled_set", "mean"),
    ).reset_index()
    nearest_target = policies[policies["policy"].eq("nearest_prototype")].groupby("target_class").agg(
        nearest_target_hit_rate=("target_hit_posthoc", "mean"),
    ).reset_index()
    target_summary = target_summary.merge(frozen_target, on="target_class").merge(nearest_target, on="target_class")

    policy_summary = policies.groupby("policy").agg(
        target_hit_rate=("target_hit_posthoc", "mean"),
        prototype_similarity_mean=("prototype_cosine_similarity", "mean"),
        distance_to_labeled_mean=("distance_to_labeled_set", "mean"),
        unique_selected_images=("sample_id", "nunique"),
    ).reset_index()
    policy_ci_rows = []
    for policy, frame in policies.groupby("policy"):
        seed_rates = frame.groupby("acquisition_seed")["target_hit_posthoc"].mean().to_numpy(float)
        low, high = bootstrap_ci(seed_rates)
        policy_ci_rows.append({"policy": policy, "seed_bootstrap_ci95_low": low, "seed_bootstrap_ci95_high": high})
    policy_summary = policy_summary.merge(pd.DataFrame(policy_ci_rows), on="policy")

    micro_summary = policies.groupby(["policy", "micro_round"]).agg(
        target_hit_rate=("target_hit_posthoc", "mean"),
        prototype_similarity_mean=("prototype_cosine_similarity", "mean"),
        distance_to_labeled_mean=("distance_to_labeled_set", "mean"),
    ).reset_index()
    repeated = frozen.groupby("sample_id").agg(
        selection_count=("sample_id", "size"),
        target_hit_rate=("target_hit_posthoc", "mean"),
        target_classes=("target_class", lambda values: "|".join(str(value) for value in sorted(set(values)))),
    ).reset_index().sort_values(["selection_count", "sample_id"], ascending=[False, True])
    repeated = repeated.merge(gt[["sample_id", "class_ids", "production_group_raw", "hard_group_id"]], on="sample_id", how="left")

    frozen_with_bins = frozen.copy()
    frozen_with_bins["similarity_quartile"] = pd.qcut(
        frozen_with_bins["prototype_cosine_similarity"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"]
    )
    similarity_bins = frozen_with_bins.groupby("similarity_quartile", observed=True).agg(
        selections=("sample_id", "size"),
        similarity_mean=("prototype_cosine_similarity", "mean"),
        target_hit_rate=("target_hit_posthoc", "mean"),
        distance_to_labeled_mean=("distance_to_labeled_set", "mean"),
    ).reset_index()
    confusion = confusion_table(rounds)

    target_summary.to_csv(out / "representation_target_class_summary.csv", index=False, encoding="utf-8-sig")
    policy_summary.to_csv(out / "representation_policy_diagnostic.csv", index=False, encoding="utf-8-sig")
    micro_summary.to_csv(out / "representation_micro_round_summary.csv", index=False, encoding="utf-8-sig")
    round_neighborhood.to_csv(out / "representation_neighborhood_round_summary.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(out / "representation_neighborhood_candidates_posthoc.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(out / "representation_target_actual_confusion.csv", index=False, encoding="utf-8-sig")
    similarity_bins.to_csv(out / "representation_similarity_quartile_summary.csv", index=False, encoding="utf-8-sig")
    repeated.to_csv(out / "representation_repeated_image_summary.csv", index=False, encoding="utf-8-sig")

    policy_index = policy_summary.set_index("policy")
    frozen_hit = float(policy_index.loc["frozen_max_novelty", "target_hit_rate"])
    nearest_hit = float(policy_index.loc["nearest_prototype", "target_hit_rate"])
    best_policy = str(policy_summary.sort_values(["target_hit_rate", "policy"], ascending=[False, True]).iloc[0]["policy"])
    best_hit = float(policy_index.loc[best_policy, "target_hit_rate"])
    mean_purity = float(round_neighborhood["neighborhood_target_purity"].mean())
    availability = float(round_neighborhood["neighborhood_has_target"].mean())
    eligible_targets = target_summary[target_summary["micro_rounds"] >= 20]
    adequate_target_fraction = float((eligible_targets["nearest_target_hit_rate"] >= 0.50).mean())

    if nearest_hit >= 0.50 and adequate_target_fraction >= 0.70:
        decision = "RULE_FAILURE_WITH_LOCAL_STRUCTURE"
        interpretation = "The local DINO neighborhoods contain usable class structure, but max-novelty chooses the wrong boundary samples."
    elif adequate_target_fraction >= 0.70:
        decision = "CLASS_CONDITIONAL_STRUCTURE_WITH_DOMINANT_TARGET_COLLAPSE"
        interpretation = (
            "Most sufficiently sampled target classes have usable nearest-neighbor structure, "
            "but a frequently targeted class collapses strongly enough to invalidate the aggregate representation rule."
        )
    elif availability >= 0.80 and best_hit >= 0.50:
        decision = "WEAK_STRUCTURE_REQUIRES_NEW_REPRESENTATION_RULE"
        interpretation = "Targets often exist inside the neighborhood, but no simple frozen rule is uniformly reliable."
    else:
        decision = "DINO_LOCAL_CLASS_STRUCTURE_INADEQUATE"
        interpretation = "The frozen DINO neighborhood is not sufficiently class-aligned for label-feedback representation repair."

    report = [
        "# GC10 D2R Representation-Failure Audit", "",
        "- New selection performed: **False**",
        "- Detector training performed: **False**",
        "- Final test used: **False**",
        "- Replayed frozen representation micro-rounds: **1,000/1,000**",
        "- Post-hoc neighborhood candidates audited: **32,000**",
        f"- Diagnostic decision: **{decision}**", "",
        "## Core diagnosis", "",
        f"- Frozen max-novelty target-hit rate: **{frozen_hit:.3f}**",
        f"- Nearest-prototype target-hit rate: **{nearest_hit:.3f}**",
        f"- Best diagnostic policy: **{best_policy} ({best_hit:.3f})**",
        f"- Mean target purity inside top-{LOCAL_K}: **{mean_purity:.3f}**",
        f"- Neighborhoods containing at least one target-class image: **{availability:.3f}**",
        f"- Eligible target classes with nearest-hit >= 0.50: **{adequate_target_fraction:.3f}**", "",
        interpretation, "",
        "## Target-class breakdown", "", target_summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## Diagnostic policy comparison", "", policy_summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## Micro-round behavior", "", micro_summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## Similarity quartiles of the frozen selection", "", similarity_bins.to_markdown(index=False, floatfmt=".6f"), "",
        "These alternative policies are post-hoc diagnostics on GC10 and cannot authorize detector training or be reported as confirmatory selector results.",
    ]
    summary_path = out / "representation_failure_summary.md"
    summary_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps({
        "source_d2r_dir": str(d2r_dir),
        "replayed_micro_rounds": len(rounds),
        "audited_candidates": len(candidates),
        "diagnostic_decision": decision,
        "new_selection_performed": False,
        "detector_training_performed": False,
        "final_test_used": False,
    }, indent=2), encoding="utf-8")
    return decision, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--d2r-dir", type=Path, default=DEFAULT_D2R)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    d2r_dir, blind, embeddings, gt, feedback, rounds = load_inputs(args)
    out = args.output_dir.expanduser().resolve() if args.output_dir else d2r_dir / "representation_failure_audit"
    out.mkdir(parents=True, exist_ok=True)
    candidates, policies = replay_neighborhoods(blind, embeddings, gt, feedback, rounds)
    decision, summary = summarize(d2r_dir, out, gt, rounds, candidates, policies)
    print("=" * 100)
    print("[DONE] GC10 D2R representation-failure audit")
    print(f"[DECISION] {decision}")
    print(f"[SUMMARY] {summary}")
    print("New selection performed: False")
    print("Detector training performed: False")
    print("Final test used: False")
    print("=" * 100)


if __name__ == "__main__":
    main()

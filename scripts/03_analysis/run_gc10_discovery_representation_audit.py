"""Run the frozen GC10 D2R selection-only mechanism audit.

D2R = 10 global DINO discovery selections, 5 label-feedback-driven local
representation selections, and 5 random guard selections. Only labels of the
initial/already-selected images may be returned to the sequential selector.
The final split is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715"
DEFAULT_EMBEDDINGS = ROOT / "outputs" / "gc10_visual_embeddings" / "dinov2_small_protocol_20260715"
DEFAULT_BASELINE = ROOT / "runs" / "gc10_taxonomy_selection_audit" / "gc10_random_vs_dino_200seed_20260715"
DEFAULT_OUT = ROOT / "runs" / "gc10_discovery_representation_audit" / "gc10_d2r_200seed_20260715"

RANDOM = "GTFreeRandom"
DINO = "FrozenDINOVisualDiversity"
D2R = "D2RDiscoveryRepresentationGuard"
STRATEGIES = [RANDOM, DINO, D2R]
SEEDS = list(range(200))
INITIAL_SIZE = 20
QUERY_SIZE = 20
DISCOVERY_SIZE = 10
REPRESENTATION_SIZE = 5
GUARD_SIZE = 5
LOCAL_NEIGHBORHOOD_SIZE = 32
EXPECTED_ROWS = 1836
EXPECTED_DIM = 384
RARE_CLASSES = {8, 9, 10}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_set(value: Any) -> set[int]:
    return {int(item) for item in str(value).split("|") if item and item != "nan"}


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise RuntimeError("Cannot normalize a zero DINO prototype")
    return vector / norm


def bootstrap_ci(values: np.ndarray, seed: int = 20260715) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def initial_indices(all_indices: np.ndarray, seed: int) -> list[int]:
    return (
        pd.DataFrame({"idx": all_indices})
        .sample(n=INITIAL_SIZE, random_state=seed + 999, replace=False)["idx"]
        .astype(int)
        .tolist()
    )


def random_query(remaining: np.ndarray, seed: int, size: int = QUERY_SIZE, offset: int = 101) -> list[int]:
    return (
        pd.DataFrame({"idx": remaining})
        .sample(n=size, random_state=seed + offset, replace=False)["idx"]
        .astype(int)
        .tolist()
    )


def dino_query(remaining: np.ndarray, initial: list[int], embeddings: np.ndarray, size: int = QUERY_SIZE) -> list[int]:
    candidates = remaining.copy()
    max_similarity = (embeddings[candidates] @ embeddings[np.asarray(initial)].T).max(axis=1)
    selected: list[int] = []
    for _ in range(size):
        position = int(np.argmin(max_similarity))
        chosen = int(candidates[position])
        selected.append(chosen)
        similarity = embeddings[candidates] @ embeddings[chosen]
        max_similarity = np.maximum(max_similarity, similarity)
        max_similarity[position] = np.inf
    return selected


class SelectedFeedbackOracle:
    """Simulation adapter that refuses label access before an item is selected."""

    def __init__(self, gt_classes: dict[int, set[int]], sample_ids: list[str], acquisition_seed: int):
        self._gt_classes = gt_classes
        self._sample_ids = sample_ids
        self.acquisition_seed = acquisition_seed
        self.selected: set[int] = set()
        self.known: dict[int, set[int]] = {}
        self.rows: list[dict[str, Any]] = []
        self.feedback_step = 0

    def mark_selected(self, indices: list[int]) -> None:
        self.selected.update(int(index) for index in indices)

    def reveal(self, indices: list[int], reason: str, target_class: int | None = None) -> None:
        for index in indices:
            index = int(index)
            if index not in self.selected:
                raise RuntimeError(f"Unselected GT access blocked: seed={self.acquisition_seed}, idx={index}")
            classes = set(self._gt_classes[index])
            self.known[index] = classes
            self.feedback_step += 1
            self.rows.append({
                "acquisition_seed": self.acquisition_seed,
                "feedback_step": self.feedback_step,
                "reason": reason,
                "sample_id": self._sample_ids[index],
                "target_class": target_class,
                "returned_class_ids": "|".join(str(value) for value in sorted(classes)),
                "selected_before_feedback": True,
            })


def choose_representation_candidate(
    oracle: SelectedFeedbackOracle,
    blind: pd.DataFrame,
    embeddings: np.ndarray,
    all_indices: np.ndarray,
    target_uses: Counter[int],
) -> tuple[int, int, dict[str, float]]:
    class_counts: Counter[int] = Counter()
    for classes in oracle.known.values():
        class_counts.update(classes)
    if not class_counts:
        raise RuntimeError("No observed classes are available for representation repair")
    target_class = min(class_counts, key=lambda value: (class_counts[value], target_uses[value], value))
    target_members = [index for index, classes in oracle.known.items() if target_class in classes]
    prototype = normalize(embeddings[np.asarray(target_members)].mean(axis=0))

    selected_hashes = set(blind.iloc[sorted(oracle.selected)]["image_sha256"].astype(str))
    candidate_indices = np.asarray([
        int(index) for index in all_indices
        if int(index) not in oracle.selected and str(blind.iloc[int(index)]["image_sha256"]) not in selected_hashes
    ], dtype=int)
    if len(candidate_indices) < LOCAL_NEIGHBORHOOD_SIZE:
        raise RuntimeError("Too few non-duplicate representation candidates")
    prototype_similarity = embeddings[candidate_indices] @ prototype
    candidate_ids = blind.iloc[candidate_indices]["sample_id"].astype(str).to_numpy()
    similarity_order = np.lexsort((candidate_ids, -prototype_similarity))
    local_positions = similarity_order[:LOCAL_NEIGHBORHOOD_SIZE]
    local_indices = candidate_indices[local_positions]
    local_similarity = prototype_similarity[local_positions]

    known_indices = np.asarray(sorted(oracle.known), dtype=int)
    max_similarity_to_labeled = (embeddings[local_indices] @ embeddings[known_indices].T).max(axis=1)
    novelty = 1.0 - max_similarity_to_labeled
    local_ids = blind.iloc[local_indices]["sample_id"].astype(str).to_numpy()
    novelty_order = np.lexsort((local_ids, -novelty))
    chosen_position = int(novelty_order[0])
    chosen = int(local_indices[chosen_position])
    diagnostics = {
        "target_class_labeled_image_count_before": float(class_counts[target_class]),
        "target_prototype_cosine_similarity": float(local_similarity[chosen_position]),
        "distance_to_labeled_set": float(novelty[chosen_position]),
    }
    return chosen, int(target_class), diagnostics


def d2r_query(
    seed: int,
    initial: list[int],
    remaining: np.ndarray,
    blind: pd.DataFrame,
    embeddings: np.ndarray,
    gt_classes: dict[int, set[int]],
    all_indices: np.ndarray,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    sample_ids = blind["sample_id"].astype(str).tolist()
    oracle = SelectedFeedbackOracle(gt_classes, sample_ids, seed)
    oracle.mark_selected(initial)
    oracle.reveal(initial, "initial_existing_label")

    discovery = dino_query(remaining, initial, embeddings, size=DISCOVERY_SIZE)
    oracle.mark_selected(discovery)
    oracle.reveal(discovery, "discovery_annotation_return")

    representation: list[int] = []
    representation_diagnostics: list[dict[str, Any]] = []
    target_uses: Counter[int] = Counter()
    for micro_round in range(1, REPRESENTATION_SIZE + 1):
        chosen, target_class, diagnostics = choose_representation_candidate(
            oracle, blind, embeddings, all_indices, target_uses
        )
        oracle.mark_selected([chosen])
        oracle.reveal([chosen], "representation_annotation_return", target_class=target_class)
        target_uses[target_class] += 1
        representation.append(chosen)
        representation_diagnostics.append({
            "acquisition_seed": seed,
            "micro_round": micro_round,
            "sample_id": sample_ids[chosen],
            "target_class": target_class,
            "returned_class_ids": "|".join(str(value) for value in sorted(oracle.known[chosen])),
            "target_hit_posthoc": target_class in oracle.known[chosen],
            **diagnostics,
        })

    remaining_after_repair = np.asarray([index for index in all_indices if int(index) not in oracle.selected], dtype=int)
    guard = random_query(remaining_after_repair, seed, size=GUARD_SIZE, offset=30301)
    query = discovery + representation + guard
    if len(query) != QUERY_SIZE or len(set(query)) != QUERY_SIZE or set(query) & set(initial):
        raise RuntimeError(f"D2R cardinality/overlap failure for seed={seed}")
    return query, oracle.rows, representation_diagnostics


def union_classes(frame: pd.DataFrame) -> set[int]:
    output: set[int] = set()
    for value in frame["class_ids"]:
        output.update(class_set(value))
    return output


def within_class_distance(
    query_indices: list[int], gt_classes: dict[int, set[int]], embeddings: np.ndarray, allowed_classes: set[int]
) -> float:
    pairs: set[tuple[int, int]] = set()
    for class_id in allowed_classes:
        members = sorted(index for index in query_indices if class_id in gt_classes[index])
        pairs.update((left, right) for left, right in combinations(members, 2))
    if not pairs:
        return float("nan")
    distances = [1.0 - float(embeddings[left] @ embeddings[right]) for left, right in sorted(pairs)]
    return float(np.mean(distances))


def metrics_for_selection(
    query_indices: list[int], initial_indices_: list[int], gt: pd.DataFrame, embeddings: np.ndarray,
    gt_classes: dict[int, set[int]],
) -> dict[str, Any]:
    query = gt.iloc[query_indices]
    initial = gt.iloc[initial_indices_]
    initial_classes = union_classes(initial)
    query_classes = union_classes(query)
    query_rare_sets = query["class_ids"].map(class_set).map(lambda values: values & RARE_CLASSES)
    rare_flags = query_rare_sets.map(bool).to_numpy()
    rare_ranks = np.flatnonzero(rare_flags) + 1
    query_embeddings = embeddings[np.asarray(query_indices)]
    initial_embeddings = embeddings[np.asarray(initial_indices_)]
    pairwise = query_embeddings @ query_embeddings.T
    upper = pairwise[np.triu_indices(len(query_indices), k=1)]
    initial_similarity = query_embeddings @ initial_embeddings.T
    return {
        "query_unique_classes": len(query_classes),
        "combined_unique_classes": len(initial_classes | query_classes),
        "query_new_classes_vs_initial": len(query_classes - initial_classes),
        "query_images_with_rare_class": int(rare_flags.sum()),
        "query_unique_rare_classes": len(query_classes & RARE_CLASSES),
        "query_new_rare_classes_vs_initial": len((query_classes & RARE_CLASSES) - initial_classes),
        "annotations_to_first_rare_class": int(rare_ranks[0]) if len(rare_ranks) else QUERY_SIZE + 1,
        "query_instances": int(query["num_instances"].sum()),
        "query_multilabel_images": int((query["num_unique_classes"] > 1).sum()),
        "query_bbox_area_ratio_sum": float(query["bbox_area_ratio_sum"].sum()),
        "query_pairwise_cosine_similarity_mean": float(upper.mean()),
        "query_pairwise_cosine_similarity_max": float(upper.max()),
        "query_min_distance_to_initial_mean": float((1.0 - initial_similarity.max(axis=1)).mean()),
        "query_within_class_cosine_distance_mean": within_class_distance(
            query_indices, gt_classes, embeddings, set(range(1, 11))
        ),
        "rare_query_within_class_cosine_distance_mean": within_class_distance(
            query_indices, gt_classes, embeddings, RARE_CLASSES
        ),
    }


def load_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    protocol = args.protocol_dir.expanduser().resolve()
    embedding_dir = args.embedding_dir.expanduser().resolve()
    baseline_dir = args.frozen_baseline_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    protocol_config = json.loads((protocol / "gc10_protocol_config.json").read_text(encoding="utf-8"))
    embedding_config = json.loads((embedding_dir / "embedding_config.json").read_text(encoding="utf-8"))
    baseline_config = json.loads((baseline_dir / "config.json").read_text(encoding="utf-8"))
    if bool(protocol_config.get("final_test_evaluated", True)):
        raise RuntimeError("Protocol final-test safety flag failed")
    if bool(embedding_config.get("final_test_used", True)) or bool(baseline_config.get("final_test_used", True)):
        raise RuntimeError("Embedding/baseline final-test safety flag failed")
    if baseline_config.get("seeds") != SEEDS or int(baseline_config.get("initial_size", -1)) != INITIAL_SIZE:
        raise RuntimeError("Frozen baseline protocol does not match D2R protocol")

    blind = pd.read_csv(protocol / "gc10_acquisition_pool_blind.csv").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if blind.columns.tolist() != ["sample_id", "image_sha256", "phash64"]:
        raise RuntimeError(f"Unexpected selector-visible columns: {blind.columns.tolist()}")
    gt_raw = pd.read_csv(protocol / "gc10_acquisition_pool_gt_audit.csv")
    gt = blind[["sample_id"]].merge(gt_raw, on="sample_id", how="left", validate="one_to_one")
    boxes = pd.read_csv(protocol / "gc10_acquisition_bbox_gt_audit.csv")
    embedding_manifest = pd.read_csv(embedding_dir / "embedding_manifest.csv")
    embeddings = np.load(embedding_dir / "embeddings.npy")
    frozen_records = pd.read_csv(baseline_dir / "gc10_selection_records_posthoc.csv")
    if len(blind) != EXPECTED_ROWS or embeddings.shape != (EXPECTED_ROWS, EXPECTED_DIM):
        raise RuntimeError(f"Unexpected blind/embedding size: {len(blind)}/{embeddings.shape}")
    if blind["sample_id"].astype(str).tolist() != embedding_manifest["sample_id"].astype(str).tolist():
        raise RuntimeError("Embedding order does not match the blind acquisition pool")
    if gt["class_ids"].isna().any():
        raise RuntimeError("Acquisition GT join failed")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise RuntimeError("DINO embeddings are not L2-normalized")
    return protocol, embedding_dir, baseline_dir, out, blind, gt, boxes, embeddings, frozen_records


def assert_frozen_replay(seed: int, strategy: str, query_indices: list[int], blind: pd.DataFrame, records: pd.DataFrame) -> None:
    expected = (
        records[(records["acquisition_seed"] == seed) & records["strategy"].eq(strategy)]
        .sort_values("rank_in_query")["sample_id"].astype(str).tolist()
    )
    actual = blind.iloc[query_indices]["sample_id"].astype(str).tolist()
    if actual != expected:
        raise RuntimeError(f"Frozen baseline replay mismatch: seed={seed}, strategy={strategy}")


def strategy_stability(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        frame = selections[selections["strategy"].eq(strategy)]
        sets = {
            int(seed): set(group["sample_id"].astype(str))
            for seed, group in frame.groupby("acquisition_seed")
        }
        counts = Counter(frame["sample_id"].astype(str))
        overlaps = [len(sets[left] & sets[right]) for left, right in combinations(SEEDS, 2)]
        probabilities = np.asarray(list(counts.values()), dtype=float) / float(len(frame))
        rows.append({
            "strategy": strategy,
            "query_slots": len(frame),
            "unique_query_images_across_seeds": len(counts),
            "top10_query_slot_fraction": sum(value for _, value in counts.most_common(10)) / float(len(frame)),
            "effective_unique_images_inverse_simpson": float(1.0 / np.sum(probabilities ** 2)),
            "mean_pairwise_query_overlap": float(np.mean(overlaps)),
            "median_pairwise_query_overlap": float(np.median(overlaps)),
            "max_pairwise_query_overlap": int(np.max(overlaps)),
        })
    return pd.DataFrame(rows)


def paired_metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_names = [column for column in metrics.columns if column not in {"acquisition_seed", "strategy"}]
    for comparator in [RANDOM, DINO]:
        for metric in metric_names:
            pivot = metrics.pivot(index="acquisition_seed", columns="strategy", values=metric).loc[SEEDS]
            valid = np.isfinite(pivot[D2R].to_numpy(float)) & np.isfinite(pivot[comparator].to_numpy(float))
            delta = pivot.loc[valid, D2R].to_numpy(float) - pivot.loc[valid, comparator].to_numpy(float)
            low, high = bootstrap_ci(delta)
            rows.append({
                "comparator": comparator,
                "metric": metric,
                "valid_paired_seeds": int(valid.sum()),
                "comparator_mean": float(pivot.loc[valid, comparator].mean()),
                "d2r_mean": float(pivot.loc[valid, D2R].mean()),
                "mean_difference": float(np.mean(delta)),
                "median_difference": float(np.median(delta)),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "wins": int((delta > 0).sum()),
                "losses": int((delta < 0).sum()),
                "ties": int((delta == 0).sum()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--frozen-baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    protocol, embedding_dir, baseline_dir, out, blind, gt, boxes, embeddings, frozen_records = load_inputs(args)
    all_indices = np.arange(len(blind), dtype=int)
    gt_classes = {index: class_set(value) for index, value in enumerate(gt["class_ids"])}

    # A one-seed preflight catches ordering and algorithm failures without creating experiment results.
    preflight_initial = initial_indices(all_indices, 0)
    preflight_remaining = np.setdiff1d(all_indices, np.asarray(preflight_initial), assume_unique=True)
    preflight_random = random_query(preflight_remaining, 0)
    preflight_dino = dino_query(preflight_remaining, preflight_initial, embeddings)
    assert_frozen_replay(0, RANDOM, preflight_random, blind, frozen_records)
    assert_frozen_replay(0, DINO, preflight_dino, blind, frozen_records)
    preflight_d2r, preflight_feedback, _ = d2r_query(
        0, preflight_initial, preflight_remaining, blind, embeddings, gt_classes, all_indices
    )
    if len(preflight_feedback) != INITIAL_SIZE + DISCOVERY_SIZE + REPRESENTATION_SIZE or len(preflight_d2r) != QUERY_SIZE:
        raise RuntimeError("D2R preflight cardinality failed")
    if args.validate_only:
        print("[PREFLIGHT PASS] frozen baseline replay, blind schema, embedding alignment, and D2R feedback boundary")
        print("Experiment results written: False")
        print("Detector training performed: False")
        print("Final test used: False")
        return

    out.mkdir(parents=True, exist_ok=True)
    selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    class_yield_rows: list[dict[str, Any]] = []
    feedback_rows: list[dict[str, Any]] = []
    representation_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        initial = initial_indices(all_indices, seed)
        remaining = np.setdiff1d(all_indices, np.asarray(initial), assume_unique=True)
        random_indices = random_query(remaining, seed)
        dino_indices = dino_query(remaining, initial, embeddings)
        assert_frozen_replay(seed, RANDOM, random_indices, blind, frozen_records)
        assert_frozen_replay(seed, DINO, dino_indices, blind, frozen_records)
        d2r_indices, feedback, representation_diagnostics = d2r_query(
            seed, initial, remaining, blind, embeddings, gt_classes, all_indices
        )
        feedback_rows.extend(feedback)
        representation_rows.extend(representation_diagnostics)
        queries = {RANDOM: random_indices, DINO: dino_indices, D2R: d2r_indices}
        stages = {
            RANDOM: ["baseline_query"] * QUERY_SIZE,
            DINO: ["baseline_query"] * QUERY_SIZE,
            D2R: ["discovery"] * DISCOVERY_SIZE + ["representation"] * REPRESENTATION_SIZE + ["random_guard"] * GUARD_SIZE,
        }
        representation_by_id = {row["sample_id"]: row for row in representation_diagnostics}

        for strategy, query_indices in queries.items():
            metric_rows.append({
                "acquisition_seed": seed,
                "strategy": strategy,
                **metrics_for_selection(query_indices, initial, gt, embeddings, gt_classes),
            })
            query_gt = gt.iloc[query_indices]
            selected_ids = set(query_gt["sample_id"].astype(str))
            selected_boxes = boxes[boxes["sample_id"].astype(str).isin(selected_ids)]
            for rank, (index, stage) in enumerate(zip(query_indices, stages[strategy]), start=1):
                row = gt.iloc[int(index)]
                diag = representation_by_id.get(str(row["sample_id"]), {}) if strategy == D2R else {}
                target_class = diag.get("target_class")
                selection_rows.append({
                    "acquisition_seed": seed,
                    "strategy": strategy,
                    "rank_in_query": rank,
                    "selection_stage": stage,
                    "sample_id": row["sample_id"],
                    "target_class": target_class,
                    "target_hit_posthoc": bool(target_class in gt_classes[int(index)]) if target_class is not None else "",
                    "class_ids_posthoc": row["class_ids"],
                    "num_instances_posthoc": int(row["num_instances"]),
                    "num_unique_classes_posthoc": int(row["num_unique_classes"]),
                    "contains_rare_class_posthoc": bool(gt_classes[int(index)] & RARE_CLASSES),
                })
            for class_id in range(1, 11):
                images_with_class = sum(class_id in gt_classes[int(index)] for index in query_indices)
                class_yield_rows.append({
                    "acquisition_seed": seed,
                    "strategy": strategy,
                    "class_id": class_id,
                    "selected_images_with_class": int(images_with_class),
                    "selected_instances": int((selected_boxes["class_id"] == class_id).sum()),
                })
            if strategy == D2R:
                for stage in ["discovery", "representation", "random_guard"]:
                    indices = [index for index, value in zip(query_indices, stages[strategy]) if value == stage]
                    classes = set().union(*(gt_classes[int(index)] for index in indices))
                    stage_rows.append({
                        "acquisition_seed": seed,
                        "stage": stage,
                        "selected_images": len(indices),
                        "unique_classes": len(classes),
                        "images_with_rare_class": sum(bool(gt_classes[int(index)] & RARE_CLASSES) for index in indices),
                        "instances": int(gt.iloc[indices]["num_instances"].sum()),
                        "representation_target_hit_rate": (
                            float(np.mean([row["target_hit_posthoc"] for row in representation_diagnostics]))
                            if stage == "representation" else float("nan")
                        ),
                    })
        if (seed + 1) % 20 == 0:
            print(f"[D2R SELECTION] seeds={seed + 1}/{len(SEEDS)}", flush=True)

    selections = pd.DataFrame(selection_rows)
    metrics = pd.DataFrame(metric_rows)
    class_yields = pd.DataFrame(class_yield_rows)
    feedback = pd.DataFrame(feedback_rows)
    representation = pd.DataFrame(representation_rows)
    stages = pd.DataFrame(stage_rows)
    paired = paired_metric_summary(metrics)
    stability = strategy_stability(selections)
    per_class = class_yields.groupby(["strategy", "class_id"])[["selected_images_with_class", "selected_instances"]].mean().reset_index()

    if not feedback["selected_before_feedback"].all():
        raise RuntimeError("Feedback-access boundary audit failed")
    if len(feedback) != len(SEEDS) * (INITIAL_SIZE + DISCOVERY_SIZE + REPRESENTATION_SIZE):
        raise RuntimeError("Unexpected feedback-access log cardinality")

    selections.to_csv(out / "gc10_d2r_selection_records_posthoc.csv", index=False, encoding="utf-8-sig")
    feedback.to_csv(out / "gc10_d2r_feedback_access_log.csv", index=False, encoding="utf-8-sig")
    representation.to_csv(out / "gc10_d2r_representation_micro_rounds.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(out / "gc10_d2r_seed_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(out / "gc10_d2r_paired_metric_summary.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(out / "gc10_d2r_per_class_mean_yield.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(out / "gc10_d2r_cross_seed_stability.csv", index=False, encoding="utf-8-sig")
    stages.to_csv(out / "gc10_d2r_stage_diagnostics.csv", index=False, encoding="utf-8-sig")

    paired_index = paired.set_index(["comparator", "metric"])
    stability_index = stability.set_index("strategy")
    class_index = per_class.set_index(["strategy", "class_id"])

    def diff(comparator: str, metric: str) -> float:
        return float(paired_index.loc[(comparator, metric), "mean_difference"])

    def ci_low(comparator: str, metric: str) -> float:
        return float(paired_index.loc[(comparator, metric), "bootstrap_ci95_low"])

    unique_ratio = (
        float(stability_index.loc[D2R, "unique_query_images_across_seeds"])
        / float(stability_index.loc[DINO, "unique_query_images_across_seeds"])
    )
    concentration_ratio = (
        float(stability_index.loc[D2R, "top10_query_slot_fraction"])
        / float(stability_index.loc[DINO, "top10_query_slot_fraction"])
    )
    overlap_difference = (
        float(stability_index.loc[D2R, "mean_pairwise_query_overlap"])
        - float(stability_index.loc[DINO, "mean_pairwise_query_overlap"])
    )
    class9_difference = (
        float(class_index.loc[(D2R, 9), "selected_images_with_class"])
        - float(class_index.loc[(RANDOM, 9), "selected_images_with_class"])
    )
    class10_difference = (
        float(class_index.loc[(D2R, 10), "selected_images_with_class"])
        - float(class_index.loc[(RANDOM, 10), "selected_images_with_class"])
    )
    target_hit_rate = float(representation["target_hit_posthoc"].mean())
    gate_specs = [
        ("discovery", "combined_unique_classes_gain_vs_random", diff(RANDOM, "combined_unique_classes"), ">= 0.50", diff(RANDOM, "combined_unique_classes") >= 0.50),
        ("discovery", "combined_unique_classes_ci_low_vs_random", ci_low(RANDOM, "combined_unique_classes"), "> 0", ci_low(RANDOM, "combined_unique_classes") > 0.0),
        ("discovery", "rare_images_gain_vs_random", diff(RANDOM, "query_images_with_rare_class"), ">= 1.00", diff(RANDOM, "query_images_with_rare_class") >= 1.0),
        ("discovery", "rare_images_ci_low_vs_random", ci_low(RANDOM, "query_images_with_rare_class"), "> 0", ci_low(RANDOM, "query_images_with_rare_class") > 0.0),
        ("discovery", "unique_rare_classes_gain_vs_random", diff(RANDOM, "query_unique_rare_classes"), ">= 0.25", diff(RANDOM, "query_unique_rare_classes") >= 0.25),
        ("discovery", "unique_rare_classes_ci_low_vs_random", ci_low(RANDOM, "query_unique_rare_classes"), "> 0", ci_low(RANDOM, "query_unique_rare_classes") > 0.0),
        ("discovery", "query_instances_noninferiority_vs_random", diff(RANDOM, "query_instances"), ">= -1.00", diff(RANDOM, "query_instances") >= -1.0),
        ("repair", "combined_unique_classes_retention_vs_dino", diff(DINO, "combined_unique_classes"), ">= -0.25", diff(DINO, "combined_unique_classes") >= -0.25),
        ("repair", "rare_images_retention_vs_dino", diff(DINO, "query_images_with_rare_class"), ">= -1.00", diff(DINO, "query_images_with_rare_class") >= -1.0),
        ("repair", "class9_image_yield_vs_random", class9_difference, ">= -0.10", class9_difference >= -0.10),
        ("repair", "class10_image_yield_vs_random", class10_difference, ">= -0.10", class10_difference >= -0.10),
        ("repair", "cross_seed_unique_image_ratio_vs_dino", unique_ratio, ">= 1.50", unique_ratio >= 1.50),
        ("repair", "top10_concentration_ratio_vs_dino", concentration_ratio, "<= 0.75", concentration_ratio <= 0.75),
        ("repair", "mean_cross_seed_overlap_difference_vs_dino", overlap_difference, "<= -2.00", overlap_difference <= -2.0),
        ("repair", "rare_within_class_distance_difference_vs_dino", diff(DINO, "rare_query_within_class_cosine_distance_mean"), ">= 0", diff(DINO, "rare_query_within_class_cosine_distance_mean") >= 0.0),
        ("mechanism", "representation_target_hit_rate", target_hit_rate, ">= 0.50", target_hit_rate >= 0.50),
    ]
    gate = pd.DataFrame([
        {"role": role, "check": name, "observed": observed, "threshold": threshold, "passed": passed}
        for role, name, observed, threshold, passed in gate_specs
    ])
    overall = bool(gate["passed"].all())
    gate.to_csv(out / "gc10_d2r_selection_gate.csv", index=False, encoding="utf-8-sig")

    primary = paired[
        paired["metric"].isin([
            "combined_unique_classes", "query_images_with_rare_class", "query_unique_rare_classes",
            "query_instances", "query_pairwise_cosine_similarity_mean",
            "query_min_distance_to_initial_mean", "rare_query_within_class_cosine_distance_mean",
        ])
    ]
    report = [
        "# GC10 D2R Selection-Only Mechanism Audit", "",
        "- Detector training performed: **False**",
        "- Final test used: **False**",
        "- Unselected-pool GT/XML used by selector: **False**",
        "- Labels returned for initial/already-selected images: **True**",
        "- Frozen Random/pure-DINO replay: **PASS (200/200 seeds)**",
        f"- Acquisition seeds: **{len(SEEDS)}**",
        f"- Initial/query budget: **{INITIAL_SIZE}/{QUERY_SIZE}**",
        f"- D2R stages: **{DISCOVERY_SIZE} discovery + {REPRESENTATION_SIZE} representation + {GUARD_SIZE} random guard**",
        f"- Detector-authorization gate: **{'PASS' if overall else 'FAIL'}**", "",
        "## Paired primary metrics", "", primary.to_markdown(index=False, floatfmt=".6f"), "",
        "## Cross-seed stability", "", stability.to_markdown(index=False, floatfmt=".6f"), "",
        "## Per-class query yield", "", per_class.to_markdown(index=False, floatfmt=".6f"), "",
        "## D2R stage diagnostics", "", stages.groupby("stage", dropna=False).mean(numeric_only=True).reset_index().to_markdown(index=False, floatfmt=".6f"), "",
        "## Frozen gate", "", gate.assign(result=gate["passed"].map({True: "PASS", False: "FAIL"})).drop(columns="passed").to_markdown(index=False, floatfmt=".6f"), "",
        "A FAIL prohibits new YOLO training. A PASS authorizes only the pre-registered GC10 development follow-up; it does not authorize final-test use.",
    ]
    summary_path = out / "gc10_d2r_selection_summary.md"
    summary_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    config = {
        "protocol": "GC10 D2R discovery-to-representation mechanism audit",
        "seeds": SEEDS,
        "initial_size": INITIAL_SIZE,
        "query_size": QUERY_SIZE,
        "d2r_stages": {"discovery": DISCOVERY_SIZE, "representation": REPRESENTATION_SIZE, "random_guard": GUARD_SIZE},
        "local_neighborhood_size": LOCAL_NEIGHBORHOOD_SIZE,
        "strategies": STRATEGIES,
        "frozen_baseline_dir": str(baseline_dir),
        "frozen_baseline_replay_pass": True,
        "selector_uses_unselected_gt": False,
        "selected_label_feedback_used": True,
        "feedback_access_violations": 0,
        "detector_training_performed": False,
        "development_only": True,
        "final_test_used": False,
        "overall_gate_pass": overall,
        "selection_records_sha256": sha256(out / "gc10_d2r_selection_records_posthoc.csv"),
        "feedback_log_sha256": sha256(out / "gc10_d2r_feedback_access_log.csv"),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("=" * 100)
    print("[DONE] GC10 D2R selection-only mechanism audit")
    print(f"[GATE] {'PASS' if overall else 'FAIL'}")
    print(f"[SUMMARY] {summary_path}")
    print("Detector training performed: False")
    print("Final test used: False")
    print("=" * 100)


if __name__ == "__main__":
    main()

"""Select a stable GC10 initial annotation scope before detector-coupled AL.

No detector training or final-test access is performed. Sampling policies see
the blind sample IDs and frozen DINO embeddings only. GT is joined post-hoc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_PROTOCOL = ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715"
DEFAULT_EMBEDDINGS = ROOT / "outputs" / "gc10_visual_embeddings" / "dinov2_small_protocol_20260715"
DEFAULT_OUT = ROOT / "runs" / "dcal_xai" / "v2_budget_design"
BUDGETS = [20, 40, 60, 80, 100, 120]
CLUSTER_COUNTS = [10, 20, 40]
DESIGN_SEEDS = list(range(200))
HOLDOUT_SEEDS = list(range(10000, 10200))
RARE_CLASSES = {8, 9, 10}
THRESHOLDS = {
    "all_classes_rate_min": 0.90,
    "all_rare_classes_rate_min": 0.90,
    "min_two_images_per_class_rate_min": 0.75,
    "production_groups_difference_vs_random_min": -1.0,
    "instances_difference_vs_random_min": -2.0,
    "pairwise_similarity_difference_vs_random_max": 0.02,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_classes(value: Any) -> set[int]:
    return {int(part) for part in str(value).split("|") if str(part).strip()}


def random_selection(size: int, budget: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice(size, size=budget, replace=False).astype(int)


def cluster_coverage_selection(labels: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Cover clusters once, then uniformly fill from unselected candidates."""

    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    cluster_ids = np.unique(labels)
    by_cluster = {cluster: np.flatnonzero(labels == cluster) for cluster in cluster_ids}
    chosen: list[int] = []
    if budget >= len(cluster_ids):
        selected_clusters = rng.permutation(cluster_ids)
    else:
        sizes = np.asarray([len(by_cluster[value]) for value in cluster_ids], dtype=float)
        selected_clusters = rng.choice(cluster_ids, size=budget, replace=False, p=sizes / sizes.sum())
    for cluster in selected_clusters:
        members = by_cluster[int(cluster)]
        chosen.append(int(rng.choice(members)))
        if len(chosen) == budget:
            return np.asarray(chosen, dtype=int)
    remaining = np.setdiff1d(np.arange(len(labels), dtype=int), np.asarray(chosen, dtype=int), assume_unique=False)
    extra = rng.choice(remaining, size=budget - len(chosen), replace=False).astype(int)
    return np.asarray(chosen + extra.tolist(), dtype=int)


def compute_metrics(
    indices: np.ndarray,
    gt: pd.DataFrame,
    boxes: pd.DataFrame,
    embeddings: np.ndarray,
) -> dict[str, Any]:
    selected = gt.iloc[indices]
    classes_by_image = [parse_classes(value) for value in selected["class_ids"]]
    per_class_images = {class_id: sum(class_id in values for values in classes_by_image) for class_id in range(1, 11)}
    class_union = set().union(*classes_by_image)
    sample_ids = set(selected["sample_id"].astype(str))
    selected_boxes = boxes[boxes["sample_id"].isin(sample_ids)]
    vectors = np.asarray(embeddings[indices], dtype=np.float64)
    pairwise = vectors @ vectors.T
    upper = pairwise[np.triu_indices(len(indices), k=1)]
    production_counts = selected["production_group_raw"].astype(str).value_counts()
    return {
        "unique_classes": len(class_union),
        "all_classes": len(class_union) == 10,
        "unique_rare_classes": len(class_union & RARE_CLASSES),
        "all_rare_classes": RARE_CLASSES.issubset(class_union),
        "rare_images": sum(bool(values & RARE_CLASSES) for values in classes_by_image),
        "min_class_images": min(per_class_images.values()),
        "min_two_images_per_class": min(per_class_images.values()) >= 2,
        "min_three_images_per_class": min(per_class_images.values()) >= 3,
        "instances": int(len(selected_boxes)),
        "multilabel_images": sum(len(values) > 1 for values in classes_by_image),
        "unique_production_groups": int(selected["production_group_raw"].nunique()),
        "top_production_group_share": float(production_counts.iloc[0] / len(selected)),
        "unique_hard_groups": int(selected["hard_group_id"].nunique()),
        "bbox_area_ratio_sum": float(selected["bbox_area_ratio_sum"].sum()),
        "pairwise_cosine_similarity_mean": float(upper.mean()),
    }


def fit_clusters(embeddings: np.ndarray) -> dict[str, np.ndarray]:
    labels: dict[str, np.ndarray] = {}
    for clusters in CLUSTER_COUNTS:
        model = MiniBatchKMeans(
            n_clusters=clusters,
            random_state=20260718,
            n_init=10,
            max_iter=500,
            batch_size=256,
            reassignment_ratio=0.01,
        )
        labels[f"DINOClusterCoverageK{clusters}"] = model.fit_predict(embeddings).astype(int)
    return labels


def evaluate_split(
    *,
    split_name: str,
    seeds: list[int],
    gt: pd.DataFrame,
    boxes: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_labels: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds, start=1):
        for budget in BUDGETS:
            selections = {"Random": random_selection(len(gt), budget, seed + budget * 100_000)}
            for policy, labels in cluster_labels.items():
                selections[policy] = cluster_coverage_selection(labels, budget, seed + budget * 100_000)
            for policy, indices in selections.items():
                if len(indices) != budget or len(set(indices.tolist())) != budget:
                    raise RuntimeError("Selection cardinality failure")
                rows.append({
                    "split": split_name,
                    "seed": seed,
                    "budget": budget,
                    "policy": policy,
                    **compute_metrics(indices, gt, boxes, embeddings),
                    "selector_used_gt": False,
                    "final_test_used": False,
                })
        if seed_index % 25 == 0:
            print(f"[{split_name}] {seed_index}/{len(seeds)}", flush=True)
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(["split", "budget", "policy"], as_index=False).agg(
        seeds=("seed", "nunique"),
        unique_classes_mean=("unique_classes", "mean"),
        all_classes_rate=("all_classes", "mean"),
        unique_rare_classes_mean=("unique_rare_classes", "mean"),
        all_rare_classes_rate=("all_rare_classes", "mean"),
        rare_images_mean=("rare_images", "mean"),
        min_class_images_mean=("min_class_images", "mean"),
        min_two_images_per_class_rate=("min_two_images_per_class", "mean"),
        min_three_images_per_class_rate=("min_three_images_per_class", "mean"),
        instances_mean=("instances", "mean"),
        multilabel_images_mean=("multilabel_images", "mean"),
        unique_production_groups_mean=("unique_production_groups", "mean"),
        top_production_group_share_mean=("top_production_group_share", "mean"),
        unique_hard_groups_mean=("unique_hard_groups", "mean"),
        bbox_area_ratio_sum_mean=("bbox_area_ratio_sum", "mean"),
        pairwise_cosine_similarity_mean=("pairwise_cosine_similarity_mean", "mean"),
    )


def add_random_differences(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    reference = result[result["policy"].eq("Random")].set_index(["split", "budget"])
    for metric in ["unique_production_groups_mean", "instances_mean", "pairwise_cosine_similarity_mean"]:
        result[f"{metric}_difference_vs_random"] = [
            float(row[metric] - reference.loc[(row["split"], row["budget"]), metric])
            for _, row in result.iterrows()
        ]
    return result


def gate_candidates(summary: pd.DataFrame, split: str) -> pd.DataFrame:
    subset = summary[summary["split"].eq(split)].copy()
    checks = {
        "all_classes": subset["all_classes_rate"] >= THRESHOLDS["all_classes_rate_min"],
        "all_rare_classes": subset["all_rare_classes_rate"] >= THRESHOLDS["all_rare_classes_rate_min"],
        "min_two_images_per_class": subset["min_two_images_per_class_rate"] >= THRESHOLDS["min_two_images_per_class_rate_min"],
        "production_groups": subset["unique_production_groups_mean_difference_vs_random"] >= THRESHOLDS["production_groups_difference_vs_random_min"],
        "instances": subset["instances_mean_difference_vs_random"] >= THRESHOLDS["instances_difference_vs_random_min"],
        "pairwise_similarity": subset["pairwise_cosine_similarity_mean_difference_vs_random"] <= THRESHOLDS["pairwise_similarity_difference_vs_random_max"],
    }
    for name, values in checks.items():
        subset[f"check_{name}"] = values
    subset["gate_pass"] = np.logical_and.reduce([values.to_numpy(bool) for values in checks.values()])
    return subset


def choose_design_candidate(gate: pd.DataFrame) -> pd.Series | None:
    passed = gate[gate["gate_pass"]].copy()
    if passed.empty:
        return None
    passed = passed.sort_values(
        ["budget", "min_two_images_per_class_rate", "all_classes_rate", "policy"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    return passed.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
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
    embeddings = np.load(embedding_dir / "embeddings.npy")
    if blind["sample_id"].tolist() != gt["sample_id"].tolist() or blind["sample_id"].tolist() != embedding_manifest["sample_id"].tolist():
        raise RuntimeError("Manifest alignment failure")
    if embeddings.shape != (1836, 384):
        raise RuntimeError(f"Unexpected embeddings: {embeddings.shape}")

    cluster_labels = fit_clusters(embeddings)
    design = evaluate_split(
        split_name="design", seeds=DESIGN_SEEDS, gt=gt, boxes=boxes,
        embeddings=embeddings, cluster_labels=cluster_labels,
    )
    holdout = evaluate_split(
        split_name="holdout", seeds=HOLDOUT_SEEDS, gt=gt, boxes=boxes,
        embeddings=embeddings, cluster_labels=cluster_labels,
    )
    metrics = pd.concat([design, holdout], ignore_index=True)
    summary = add_random_differences(summarize(metrics))
    design_gate = gate_candidates(summary, "design")
    candidate = choose_design_candidate(design_gate)
    if candidate is None:
        chosen = None
        holdout_gate_row = None
        overall = False
    else:
        chosen = {"budget": int(candidate["budget"]), "policy": str(candidate["policy"])}
        holdout_gate = gate_candidates(summary, "holdout")
        match = holdout_gate[(holdout_gate["budget"] == chosen["budget"]) & holdout_gate["policy"].eq(chosen["policy"])]
        if len(match) != 1:
            raise RuntimeError("Chosen holdout candidate missing")
        holdout_gate_row = match.iloc[0]
        overall = bool(holdout_gate_row["gate_pass"])

    metrics.to_csv(out / "budget_seed_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "budget_summary.csv", index=False, encoding="utf-8-sig")
    design_gate.to_csv(out / "design_gate.csv", index=False, encoding="utf-8-sig")
    decision = {
        "status": "PASS" if overall else "FAIL",
        "chosen": chosen,
        "holdout_gate_pass": bool(overall),
        "thresholds": THRESHOLDS,
        "budgets": BUDGETS,
        "policies": ["Random", *cluster_labels.keys()],
        "design_seeds": [DESIGN_SEEDS[0], DESIGN_SEEDS[-1]],
        "holdout_seeds": [HOLDOUT_SEEDS[0], HOLDOUT_SEEDS[-1]],
        "protocol_config_sha256": sha256(protocol_config_path),
        "embedding_config_sha256": sha256(embedding_config_path),
        "script_sha256": sha256(Path(__file__).resolve()),
        "selector_used_gt": False,
        "gt_joined_posthoc": True,
        "detector_training_performed": False,
        "final_test_used": False,
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    display_columns = [
        "split", "budget", "policy", "all_classes_rate", "all_rare_classes_rate",
        "min_two_images_per_class_rate", "instances_mean",
        "unique_production_groups_mean", "pairwise_cosine_similarity_mean",
        "gate_pass",
    ]
    combined_gate = pd.concat([design_gate, gate_candidates(summary, "holdout")], ignore_index=True)
    chosen_text = "None" if chosen is None else f"{chosen['policy']} at budget {chosen['budget']}"
    report = [
        "# DCAL-XAI V2 Initial Budget Decision", "",
        f"- Decision: **{'PASS' if overall else 'FAIL'}**",
        f"- Design-chosen candidate: **{chosen_text}**",
        f"- Holdout gate: **{'PASS' if overall else 'FAIL'}**",
        "- Detector training performed: **False**",
        "- Selector used GT/XML: **False**",
        "- GT/XML joined post-hoc: **True**",
        "- Final test used: **False**", "",
        "## Candidate gates", "", combined_gate[display_columns].to_markdown(index=False, floatfmt=".4f"), "",
    ]
    if chosen is not None:
        chosen_rows = summary[(summary["budget"] == chosen["budget"]) & summary["policy"].eq(chosen["policy"])]
        report.extend(["## Chosen candidate details", "", chosen_rows.to_markdown(index=False, floatfmt=".4f"), ""])
    report.extend([
        "## Interpretation", "",
        "A PASS authorizes only a separately frozen backbone-stability experiment at the chosen initial scope.",
        "It does not authorize detector-based acquisition, downstream confirmation, or final-test access.",
    ])
    summary_path = out / "budget_decision.md"
    summary_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[DONE] {summary_path}")
    print(f"[GATE] {'PASS' if overall else 'FAIL'}")
    print(f"[CHOSEN] {chosen_text}")
    print("Training performed: False")
    print("Final test used: False")


if __name__ == "__main__":
    main()


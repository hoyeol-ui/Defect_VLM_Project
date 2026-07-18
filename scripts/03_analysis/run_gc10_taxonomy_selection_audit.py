"""Run the pre-registered GC10 Random-vs-frozen-DINO selection-only audit."""

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
DEFAULT_OUT = ROOT / "runs" / "gc10_taxonomy_selection_audit" / "gc10_random_vs_dino_200seed_20260715"
RANDOM = "GTFreeRandom"
DINO = "FrozenDINOVisualDiversity"
SEEDS = list(range(200))
INITIAL_SIZE = 20
QUERY_SIZE = 20
EXPECTED_ROWS = 1836
RARE_CLASSES = {8, 9, 10}


def class_set(value: str) -> set[int]:
    return {int(item) for item in str(value).split("|") if item and item != "nan"}


def bootstrap_ci(values: np.ndarray, seed: int = 20260715) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def sign_flip_p(values: np.ndarray, draws: int = 200_000) -> float:
    rng = np.random.default_rng(20260715)
    observed = abs(float(values.mean()))
    extreme = 0
    complete = 0
    while complete < draws:
        batch = min(5_000, draws - complete)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(batch, len(values)), replace=True)
        null = np.abs((signs * values).mean(axis=1))
        extreme += int(np.sum(null >= observed - 1e-15))
        complete += batch
    return float((extreme + 1) / (draws + 1))


def random_query(remaining: np.ndarray, seed: int) -> list[int]:
    return pd.DataFrame({"idx": remaining}).sample(n=QUERY_SIZE, random_state=seed + 101, replace=False)["idx"].astype(int).tolist()


def dino_query(remaining: np.ndarray, initial: list[int], embeddings: np.ndarray) -> list[int]:
    candidates = remaining.copy()
    max_similarity = (embeddings[candidates] @ embeddings[np.asarray(initial)].T).max(axis=1)
    selected: list[int] = []
    for _ in range(QUERY_SIZE):
        position = int(np.argmin(max_similarity))
        chosen = int(candidates[position])
        selected.append(chosen)
        similarity = embeddings[candidates] @ embeddings[chosen]
        max_similarity = np.maximum(max_similarity, similarity)
        max_similarity[position] = np.inf
    return selected


def union_classes(frame: pd.DataFrame) -> set[int]:
    output: set[int] = set()
    for value in frame["class_ids"]:
        output.update(class_set(value))
    return output


def metrics_for_selection(query: pd.DataFrame, initial: pd.DataFrame, query_embeddings: np.ndarray, initial_embeddings: np.ndarray) -> dict[str, Any]:
    initial_classes = union_classes(initial)
    query_classes = union_classes(query)
    query_rare_sets = query["class_ids"].map(class_set).map(lambda values: values & RARE_CLASSES)
    rare_flags = query_rare_sets.map(bool).to_numpy()
    rare_ranks = np.flatnonzero(rare_flags) + 1
    pairwise = query_embeddings @ query_embeddings.T
    upper = pairwise[np.triu_indices(len(query), k=1)]
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
    }


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
    protocol_config = json.loads((protocol / "gc10_protocol_config.json").read_text(encoding="utf-8"))
    embedding_config = json.loads((embedding_dir / "embedding_config.json").read_text(encoding="utf-8"))
    if bool(protocol_config.get("final_test_evaluated", True)) or bool(embedding_config.get("final_test_used", True)):
        raise RuntimeError("Final-test safety flag failed")
    blind = pd.read_csv(protocol / "gc10_acquisition_pool_blind.csv").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if blind.columns.tolist() != ["sample_id", "image_sha256", "phash64"]:
        raise RuntimeError(f"Unexpected selector columns: {blind.columns.tolist()}")
    gt = pd.read_csv(protocol / "gc10_acquisition_pool_gt_audit.csv")
    embedding_manifest = pd.read_csv(embedding_dir / "embedding_manifest.csv")
    embeddings = np.load(embedding_dir / "embeddings.npy")
    if len(blind) != EXPECTED_ROWS or embeddings.shape != (EXPECTED_ROWS, 384):
        raise RuntimeError("Unexpected acquisition/embedding size")
    if blind["sample_id"].tolist() != embedding_manifest["sample_id"].tolist():
        raise RuntimeError("Embedding order does not match blind pool")
    gt = blind[["sample_id"]].merge(gt, on="sample_id", how="left", validate="one_to_one")
    if gt["class_ids"].isna().any():
        raise RuntimeError("Post-hoc GT join failed")
    boxes = pd.read_csv(protocol / "gc10_acquisition_bbox_gt_audit.csv")
    all_indices = np.arange(len(blind), dtype=int)
    selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    class_yield_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        initial_indices = pd.DataFrame({"idx": all_indices}).sample(n=INITIAL_SIZE, random_state=seed + 999, replace=False)["idx"].astype(int).tolist()
        remaining = np.setdiff1d(all_indices, np.asarray(initial_indices), assume_unique=True)
        queries = {
            RANDOM: random_query(remaining, seed),
            DINO: dino_query(remaining, initial_indices, embeddings),
        }
        overlap_rows.append({"acquisition_seed": seed, "query_overlap": len(set(queries[RANDOM]) & set(queries[DINO]))})
        initial_gt = gt.iloc[initial_indices]
        for strategy, query_indices in queries.items():
            query_gt = gt.iloc[query_indices].copy()
            metric_rows.append({
                "acquisition_seed": seed,
                "strategy": strategy,
                **metrics_for_selection(query_gt, initial_gt, embeddings[np.asarray(query_indices)], embeddings[np.asarray(initial_indices)]),
            })
            for rank, (idx, row) in enumerate(zip(query_indices, query_gt.to_dict("records")), start=1):
                selection_rows.append({
                    "acquisition_seed": seed,
                    "strategy": strategy,
                    "rank_in_query": rank,
                    "sample_id": row["sample_id"],
                    "class_ids_posthoc": row["class_ids"],
                    "num_instances_posthoc": int(row["num_instances"]),
                    "num_unique_classes_posthoc": int(row["num_unique_classes"]),
                    "contains_rare_class_posthoc": bool(class_set(row["class_ids"]) & RARE_CLASSES),
                })
            selected_ids = set(query_gt["sample_id"])
            selected_boxes = boxes[boxes["sample_id"].isin(selected_ids)]
            for class_id in range(1, 11):
                images_with_class = query_gt["class_ids"].map(class_set).map(lambda values: class_id in values)
                class_yield_rows.append({
                    "acquisition_seed": seed,
                    "strategy": strategy,
                    "class_id": class_id,
                    "selected_images_with_class": int(images_with_class.sum()),
                    "selected_instances": int((selected_boxes["class_id"] == class_id).sum()),
                })
        if (seed + 1) % 20 == 0:
            print(f"[SELECTION] seeds={seed + 1}/{len(SEEDS)}", flush=True)

    selections = pd.DataFrame(selection_rows)
    metrics = pd.DataFrame(metric_rows)
    class_yields = pd.DataFrame(class_yield_rows)
    overlaps = pd.DataFrame(overlap_rows)
    selections.to_csv(out / "gc10_selection_records_posthoc.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(out / "gc10_seed_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    class_yields.to_csv(out / "gc10_class_yield_posthoc.csv", index=False, encoding="utf-8-sig")
    overlaps.to_csv(out / "gc10_query_overlap.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    paired_rows = []
    for metric in [column for column in metrics.columns if column not in {"acquisition_seed", "strategy"}]:
        pivot = metrics.pivot(index="acquisition_seed", columns="strategy", values=metric).loc[SEEDS]
        delta = (pivot[DINO] - pivot[RANDOM]).to_numpy(float)
        low, high = bootstrap_ci(delta)
        summary_rows.append({
            "metric": metric,
            "random_mean": float(pivot[RANDOM].mean()),
            "dino_mean": float(pivot[DINO].mean()),
            "mean_difference": float(delta.mean()),
            "median_difference": float(np.median(delta)),
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
            "wins": int((delta > 0).sum()),
            "losses": int((delta < 0).sum()),
            "ties": int((delta == 0).sum()),
            "win_rate": float((delta > 0).mean()),
            "loss_rate": float((delta < 0).mean()),
            "monte_carlo_sign_flip_p_two_sided": sign_flip_p(delta),
        })
        paired_rows.extend({
            "acquisition_seed": seed,
            "metric": metric,
            "random": pivot.loc[seed, RANDOM],
            "dino": pivot.loc[seed, DINO],
            "difference": pivot.loc[seed, DINO] - pivot.loc[seed, RANDOM],
        } for seed in SEEDS)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "gc10_metric_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired_rows).to_csv(out / "gc10_paired_seed_metrics.csv", index=False, encoding="utf-8-sig")

    per_class_summary = class_yields.groupby(["strategy", "class_id"])[["selected_images_with_class", "selected_instances"]].mean().reset_index()
    per_class_summary.to_csv(out / "gc10_per_class_mean_yield.csv", index=False, encoding="utf-8-sig")
    indexed = summary.set_index("metric")
    coverage = indexed.loc["combined_unique_classes"]
    rare_images = indexed.loc["query_images_with_rare_class"]
    rare_coverage = indexed.loc["query_unique_rare_classes"]
    instances = indexed.loc["query_instances"]
    redundancy = indexed.loc["query_pairwise_cosine_similarity_mean"]
    distance = indexed.loc["query_min_distance_to_initial_mean"]
    gates = [
        ("combined_unique_class_gain_at_least_0p25", float(coverage["mean_difference"]) >= 0.25),
        ("combined_unique_class_gain_ci_low_positive", float(coverage["bootstrap_ci95_low"]) > 0.0),
        ("combined_unique_class_loss_rate_at_most_0p15", float(coverage["loss_rate"]) <= 0.15),
        ("rare_class_image_yield_gain_at_least_0p75", float(rare_images["mean_difference"]) >= 0.75),
        ("rare_class_image_yield_ci_low_positive", float(rare_images["bootstrap_ci95_low"]) > 0.0),
        ("rare_class_image_yield_win_rate_at_least_0p60", float(rare_images["win_rate"]) >= 0.60),
        ("unique_rare_class_coverage_gain_at_least_0p25", float(rare_coverage["mean_difference"]) >= 0.25),
        ("unique_rare_class_coverage_ci_low_positive", float(rare_coverage["bootstrap_ci95_low"]) > 0.0),
        ("bbox_instance_yield_noninferiority_minus_1", float(instances["mean_difference"]) >= -1.0),
        ("dino_query_redundancy_lower_than_random", float(redundancy["mean_difference"]) < 0.0),
        ("dino_distance_to_initial_higher_than_random", float(distance["mean_difference"]) > 0.0),
    ]
    gate = pd.DataFrame([{"check": name, "passed": passed} for name, passed in gates])
    gate.to_csv(out / "gc10_selection_only_gate.csv", index=False, encoding="utf-8-sig")
    overall = all(passed for _, passed in gates)
    primary_metrics = summary[summary["metric"].isin([
        "query_unique_classes", "combined_unique_classes", "query_new_classes_vs_initial",
        "query_images_with_rare_class", "query_unique_rare_classes", "query_new_rare_classes_vs_initial",
        "annotations_to_first_rare_class", "query_instances", "query_multilabel_images",
        "query_pairwise_cosine_similarity_mean", "query_min_distance_to_initial_mean",
    ])]
    overlap_summary = overlaps["query_overlap"].agg(["mean", "median", "min", "max"]).to_frame("value").reset_index().rename(columns={"index": "statistic"})
    report = [
        "# GC10-DET Cold-Start Taxonomy Selection-Only Audit", "",
        "- Detector training performed: **False**",
        "- Final test used: **False**",
        "- GT/XML/bbox used by selectors: **False**",
        "- Source paths/folders/filenames visible to selectors: **False**",
        "- GT/XML/bbox joined after selection for audit: **True**",
        f"- Seeds: **{len(SEEDS)}**",
        f"- Initial/query budget: **{INITIAL_SIZE}/{QUERY_SIZE}**",
        "- Frozen rare classes: **8, 9, 10**",
        f"- Overall YOLO-authorization gate: **{'PASS' if overall else 'FAIL'}**", "",
        "## Primary and diagnostic metrics", "", primary_metrics.to_markdown(index=False, floatfmt=".6f"), "",
        "## Mean per-class query yield", "", per_class_summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## Query overlap", "", overlap_summary.to_markdown(index=False, floatfmt=".3f"), "",
        "## Pre-registered gate", "", gate.assign(result=gate["passed"].map({True: "PASS", False: "FAIL"})).drop(columns="passed").to_markdown(index=False), "",
        "A failed gate prohibits YOLO training. A passed gate only authorizes a separately pre-registered development-only confirmation.",
    ]
    summary_path = out / "gc10_taxonomy_selection_audit_summary.md"
    summary_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps({
        "seeds": SEEDS,
        "initial_size": INITIAL_SIZE,
        "query_size": QUERY_SIZE,
        "rare_classes": sorted(RARE_CLASSES),
        "strategies": [RANDOM, DINO],
        "detector_training_performed": False,
        "final_test_used": False,
        "overall_gate_pass": overall,
    }, indent=2), encoding="utf-8")
    print("=" * 100)
    print("[DONE] GC10 taxonomy selection-only audit")
    print(f"[GATE] {'PASS' if overall else 'FAIL'}")
    print(f"[SUMMARY] {summary_path}")
    print("Detector training performed: False")
    print("Final test used: False")
    print("=" * 100)


if __name__ == "__main__":
    main()

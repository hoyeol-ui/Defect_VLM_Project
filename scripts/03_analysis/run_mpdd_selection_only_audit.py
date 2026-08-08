"""Run the pre-registered MPDD three-way selection-only audit without training."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "runs" / "mpdd_annotation_triage_protocol" / "mpdd_protocol_20260715"
DEFAULT_EMBEDDINGS = ROOT / "outputs" / "mpdd_visual_embeddings" / "dinov2_small_protocol_20260715"
DEFAULT_OUT = ROOT / "runs" / "mpdd_selection_only_audit" / "mpdd_hierarchical_dino_200seed_20260715"
RANDOM = "GTFreeRandom"
BALANCED_RANDOM = "CategoryBalancedRandom"
BALANCED_DINO = "FrozenCategoryBalancedDINO"
STRATEGIES = [RANDOM, BALANCED_RANDOM, BALANCED_DINO]
SEEDS = list(range(200))
INITIAL_SIZE = 20
QUERY_SIZE = 20
EXPECTED_ROWS = 1056


def bootstrap_ci(values: np.ndarray, seed: int = 20260715) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


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


def category_schedule(initial: list[int], categories: np.ndarray, available: set[int], seed: int) -> list[str]:
    names = sorted(set(categories.tolist()))
    priority_order = np.random.default_rng(seed + 20260715).permutation(names).tolist()
    priority = {name: rank for rank, name in enumerate(priority_order)}
    counts = {name: int(np.sum(categories[np.asarray(initial)] == name)) for name in names}
    schedule: list[str] = []
    simulated_available = set(available)
    for _ in range(QUERY_SIZE):
        eligible = [name for name in names if any(categories[idx] == name for idx in simulated_available)]
        chosen = min(eligible, key=lambda name: (counts[name], priority[name]))
        schedule.append(chosen)
        counts[chosen] += 1
        placeholder = next(idx for idx in simulated_available if categories[idx] == chosen)
        simulated_available.remove(placeholder)
    return schedule


def balanced_random_query(remaining: np.ndarray, initial: list[int], categories: np.ndarray, seed: int) -> list[int]:
    available = set(int(x) for x in remaining)
    schedule = category_schedule(initial, categories, available, seed)
    rng = np.random.default_rng(seed + 303)
    random_scores = {idx: float(rng.random()) for idx in sorted(available)}
    selected: list[int] = []
    for category in schedule:
        candidates = [idx for idx in available if categories[idx] == category]
        chosen = min(candidates, key=lambda idx: (random_scores[idx], idx))
        selected.append(chosen)
        available.remove(chosen)
    return selected


def balanced_dino_query(remaining: np.ndarray, initial: list[int], categories: np.ndarray, embeddings: np.ndarray, seed: int) -> list[int]:
    available = set(int(x) for x in remaining)
    schedule = category_schedule(initial, categories, available, seed)
    references = list(initial)
    selected: list[int] = []
    for category in schedule:
        candidates = np.asarray(sorted(idx for idx in available if categories[idx] == category), dtype=int)
        category_refs = [idx for idx in references if categories[idx] == category]
        refs = np.asarray(category_refs if category_refs else references, dtype=int)
        max_similarity = (embeddings[candidates] @ embeddings[refs].T).max(axis=1)
        chosen = int(candidates[int(np.argmin(max_similarity))])
        selected.append(chosen)
        references.append(chosen)
        available.remove(chosen)
    return selected


def selection_metrics(query: pd.DataFrame, initial_embeddings: np.ndarray, query_embeddings: np.ndarray, bbox_counts: pd.Series) -> dict[str, Any]:
    anomaly = query["is_anomaly"].astype(bool).to_numpy()
    ranks = np.flatnonzero(anomaly) + 1
    pairwise = query_embeddings @ query_embeddings.T
    upper = pairwise[np.triu_indices(len(query), k=1)]
    initial_similarity = query_embeddings @ initial_embeddings.T
    anomaly_names = query.loc[query["is_anomaly"].astype(bool)].apply(
        lambda row: f"{row['product_category']}::{row['anomaly_type']}", axis=1
    )
    return {
        "query_anomaly_images": int(anomaly.sum()),
        "query_anomaly_rate": float(anomaly.mean()),
        "annotations_to_first_anomaly": int(ranks[0]) if len(ranks) else QUERY_SIZE + 1,
        "query_unique_product_categories": int(query["product_category"].nunique()),
        "query_unique_anomaly_types": int(anomaly_names.nunique()),
        "query_defect_components": int(bbox_counts.reindex(query["sample_id"], fill_value=0).sum()),
        "query_defect_area_ratio_sum": float(query["defect_pixel_area_ratio"].sum()),
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

    protocol_config = json.loads((protocol / "mpdd_protocol_config.json").read_text(encoding="utf-8"))
    embedding_config = json.loads((embedding_dir / "embedding_config.json").read_text(encoding="utf-8"))
    if bool(protocol_config.get("final_test_evaluated", True)) or bool(embedding_config.get("final_test_used", True)):
        raise RuntimeError("Final-test safety flag failed")
    blind = pd.read_csv(protocol / "mpdd_acquisition_pool_blind.csv").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    forbidden = {"is_anomaly", "anomaly_type", "official_split", "image_path", "mask_path"}.intersection(blind.columns)
    if forbidden:
        raise RuntimeError(f"Blind selector view leaks prohibited columns: {sorted(forbidden)}")
    gt = pd.read_csv(protocol / "mpdd_acquisition_pool_gt_audit.csv")
    embedding_manifest = pd.read_csv(embedding_dir / "embedding_manifest.csv")
    embeddings = np.load(embedding_dir / "embeddings.npy")
    if len(blind) != EXPECTED_ROWS or embeddings.shape != (EXPECTED_ROWS, 384):
        raise RuntimeError("Unexpected acquisition/embedding size")
    if blind["sample_id"].tolist() != embedding_manifest["sample_id"].tolist():
        raise RuntimeError("Embedding order does not match blind pool")
    gt = blind[["sample_id"]].merge(gt, on="sample_id", how="left", validate="one_to_one")
    if gt["is_anomaly"].isna().any():
        raise RuntimeError("Post-hoc GT join failed")
    boxes = pd.read_csv(protocol / "mpdd_acquisition_bbox_gt_audit.csv")
    bbox_counts = boxes.groupby("sample_id").size()
    categories = blind["product_category"].astype(str).to_numpy()
    category_names = sorted(set(categories.tolist()))
    all_indices = np.arange(len(blind), dtype=int)

    selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        initial = pd.DataFrame({"idx": all_indices}).sample(n=INITIAL_SIZE, random_state=seed + 999, replace=False)["idx"].astype(int).tolist()
        remaining = np.setdiff1d(all_indices, np.asarray(initial), assume_unique=True)
        queries = {
            RANDOM: random_query(remaining, seed),
            BALANCED_RANDOM: balanced_random_query(remaining, initial, categories, seed),
            BALANCED_DINO: balanced_dino_query(remaining, initial, categories, embeddings, seed),
        }
        for left, right in itertools.combinations(STRATEGIES, 2):
            overlap_rows.append({"acquisition_seed": seed, "strategy_left": left, "strategy_right": right, "query_overlap": len(set(queries[left]) & set(queries[right]))})
        for strategy, query_indices in queries.items():
            query = gt.iloc[query_indices].copy()
            metrics = selection_metrics(query, embeddings[np.asarray(initial)], embeddings[np.asarray(query_indices)], bbox_counts)
            metric_rows.append({"acquisition_seed": seed, "strategy": strategy, **metrics})
            for rank, (idx, row) in enumerate(zip(query_indices, query.to_dict("records")), start=1):
                selection_rows.append({
                    "acquisition_seed": seed,
                    "strategy": strategy,
                    "rank_in_query": rank,
                    "sample_id": row["sample_id"],
                    "is_anomaly_posthoc": bool(row["is_anomaly"]),
                    "product_category_posthoc": row["product_category"],
                    "anomaly_type_posthoc": row["anomaly_type"],
                    "num_defect_components_posthoc": int(bbox_counts.get(row["sample_id"], 0)),
                })
            for category in category_names:
                sub = query[query["product_category"].eq(category)]
                category_rows.append({
                    "acquisition_seed": seed,
                    "strategy": strategy,
                    "product_category": category,
                    "selected_images": len(sub),
                    "anomaly_images": int(sub["is_anomaly"].astype(bool).sum()),
                })
        if (seed + 1) % 20 == 0:
            print(f"[SELECTION] seeds={seed + 1}/{len(SEEDS)}", flush=True)

    selections = pd.DataFrame(selection_rows)
    metrics = pd.DataFrame(metric_rows)
    category_yield = pd.DataFrame(category_rows)
    overlaps = pd.DataFrame(overlap_rows)
    selections.to_csv(out / "mpdd_selection_records_posthoc.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(out / "mpdd_seed_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    category_yield.to_csv(out / "mpdd_category_yield_posthoc.csv", index=False, encoding="utf-8-sig")
    overlaps.to_csv(out / "mpdd_query_overlap.csv", index=False, encoding="utf-8-sig")

    comparisons = [(BALANCED_DINO, RANDOM), (BALANCED_DINO, BALANCED_RANDOM), (BALANCED_RANDOM, RANDOM)]
    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    metric_names = [column for column in metrics.columns if column not in {"acquisition_seed", "strategy"}]
    for method, baseline in comparisons:
        for metric in metric_names:
            pivot = metrics.pivot(index="acquisition_seed", columns="strategy", values=metric).loc[SEEDS]
            delta = (pivot[method] - pivot[baseline]).to_numpy(float)
            low, high = bootstrap_ci(delta)
            summary_rows.append({
                "comparison": f"{method}-minus-{baseline}",
                "metric": metric,
                "baseline_mean": float(pivot[baseline].mean()),
                "method_mean": float(pivot[method].mean()),
                "mean_difference": float(delta.mean()),
                "median_difference": float(np.median(delta)),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "wins": int((delta > 0).sum()),
                "losses": int((delta < 0).sum()),
                "ties": int((delta == 0).sum()),
                "win_rate": float((delta > 0).mean()),
                "monte_carlo_sign_flip_p_two_sided": sign_flip_p(delta),
            })
            paired_rows.extend({
                "acquisition_seed": seed,
                "comparison": f"{method}-minus-{baseline}",
                "metric": metric,
                "baseline": pivot.loc[seed, baseline],
                "method": pivot.loc[seed, method],
                "difference": pivot.loc[seed, method] - pivot.loc[seed, baseline],
            } for seed in SEEDS)
    summary = pd.DataFrame(summary_rows)
    pd.DataFrame(paired_rows).to_csv(out / "mpdd_paired_seed_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "mpdd_metric_summary.csv", index=False, encoding="utf-8-sig")

    pivot_category = category_yield.pivot_table(index=["acquisition_seed", "product_category"], columns="strategy", values="anomaly_images", aggfunc="sum", fill_value=0).reset_index()
    loo_rows = []
    for category in category_names:
        per_seed = pivot_category[~pivot_category["product_category"].eq(category)].groupby("acquisition_seed")[[RANDOM, BALANCED_DINO]].sum()
        delta = per_seed[BALANCED_DINO] - per_seed[RANDOM]
        loo_rows.append({"held_out_product_category": category, "mean_anomaly_yield_difference": float(delta.mean())})
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(out / "mpdd_leave_one_category_out.csv", index=False, encoding="utf-8-sig")

    indexed = summary.set_index(["comparison", "metric"])
    primary = indexed.loc[(f"{BALANCED_DINO}-minus-{RANDOM}", "query_anomaly_images")]
    coverage = indexed.loc[(f"{BALANCED_DINO}-minus-{RANDOM}", "query_unique_product_categories")]
    decomposition = indexed.loc[(f"{BALANCED_DINO}-minus-{BALANCED_RANDOM}", "query_anomaly_images")]
    gates = [
        ("random_mean_anomaly_yield_at_most_5_of_20", float(primary["baseline_mean"]) <= 5.0),
        ("hierarchical_dino_gain_vs_random_at_least_2_of_20", float(primary["mean_difference"]) >= 2.0),
        ("hierarchical_dino_vs_random_ci_low_positive", float(primary["bootstrap_ci95_low"]) > 0.0),
        ("hierarchical_dino_vs_random_win_rate_at_least_0p65", float(primary["win_rate"]) >= 0.65),
        ("product_category_coverage_noninferiority_minus_0p25", float(coverage["mean_difference"]) >= -0.25),
        ("leave_one_product_category_out_min_gain_positive", float(loo["mean_anomaly_yield_difference"].min()) > 0.0),
        ("dino_gain_beyond_category_balance_at_least_1_of_20", float(decomposition["mean_difference"]) >= 1.0),
        ("dino_beyond_category_balance_ci_low_positive", float(decomposition["bootstrap_ci95_low"]) > 0.0),
        ("dino_beyond_category_balance_win_rate_at_least_0p60", float(decomposition["win_rate"]) >= 0.60),
    ]
    gate = pd.DataFrame([{"check": name, "passed": passed} for name, passed in gates])
    gate.to_csv(out / "mpdd_selection_only_gate.csv", index=False, encoding="utf-8-sig")
    overall = all(passed for _, passed in gates)
    primary_metrics = summary[summary["metric"].isin(["query_anomaly_images", "query_unique_product_categories", "annotations_to_first_anomaly", "query_unique_anomaly_types", "query_pairwise_cosine_similarity_mean", "query_min_distance_to_initial_mean"])]
    overlap_summary = overlaps.groupby(["strategy_left", "strategy_right"])["query_overlap"].agg(["mean", "median", "min", "max"]).reset_index()
    report = [
        "# MPDD Hierarchical-DINO Selection-Only Audit", "",
        "- Detector training performed: **False**",
        "- Final test used: **False**",
        "- GT/mask/bbox used by selectors: **False**",
        "- Product category used by balanced selectors: **True (allowed metadata)**",
        "- GT/mask/bbox joined after selection for audit: **True**",
        f"- Seeds: **{len(SEEDS)}**",
        f"- Initial/query budget: **{INITIAL_SIZE}/{QUERY_SIZE}**",
        f"- Overall detector-training gate: **{'PASS' if overall else 'FAIL'}**", "",
        "## Primary and diagnostic metrics", "", primary_metrics.to_markdown(index=False, floatfmt=".6f"), "",
        "## Query overlap", "", overlap_summary.to_markdown(index=False, floatfmt=".3f"), "",
        "## Leave-one-product-category-out robustness", "", loo.to_markdown(index=False, floatfmt=".6f"), "",
        "## Pre-registered gate", "", gate.assign(result=gate["passed"].map({True: "PASS", False: "FAIL"})).drop(columns="passed").to_markdown(index=False), "",
        "A failed gate prohibits detector training. A passed gate only authorizes a separately specified development-only detector experiment.",
    ]
    summary_path = out / "mpdd_selection_only_audit_summary.md"
    summary_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps({
        "seeds": SEEDS,
        "initial_size": INITIAL_SIZE,
        "query_size": QUERY_SIZE,
        "strategies": STRATEGIES,
        "detector_training_performed": False,
        "final_test_used": False,
        "overall_gate_pass": overall,
    }, indent=2), encoding="utf-8")
    print("=" * 100)
    print("[DONE] MPDD three-way selection-only audit")
    print(f"[GATE] {'PASS' if overall else 'FAIL'}")
    print(f"[SUMMARY] {summary_path}")
    print("Detector training performed: False")
    print("Final test used: False")
    print("=" * 100)


if __name__ == "__main__":
    main()

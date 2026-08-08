"""Run the pre-registered VisA Random-vs-DINO selection-only audit.

No detector is loaded or trained. GT and mask-derived metadata are joined only
after both selectors have returned their image identities.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "runs" / "visa_annotation_triage_protocol" / "visa_protocol_v2_20260715"
DEFAULT_EMBEDDINGS = ROOT / "outputs" / "visa_visual_embeddings" / "dinov2_small_protocol_v2_20260715"
DEFAULT_OUT = ROOT / "runs" / "visa_selection_only_audit" / "visa_random_vs_visual_200seed_20260715"
RANDOM = "GTFreeRandom"
VISUAL = "FrozenDINOVisualDiversity"
SEEDS = list(range(200))
INITIAL_SIZE = 20
QUERY_SIZE = 20


def bootstrap_ci(values: np.ndarray, seed: int = 20260715) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def exact_sign_flip(values: np.ndarray, draws: int = 200_000) -> float:
    rng = np.random.default_rng(20260715)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(values)), replace=True)
    null = np.abs((signs * values).mean(axis=1))
    return float((np.sum(null >= abs(values.mean()) - 1e-15) + 1) / (draws + 1))


def random_query(remaining_indices: np.ndarray, seed: int) -> list[int]:
    frame = pd.DataFrame({"idx": remaining_indices})
    return frame.sample(n=QUERY_SIZE, random_state=seed + 101, replace=False)["idx"].astype(int).tolist()


def visual_query(remaining_indices: np.ndarray, initial_indices: list[int], embeddings: np.ndarray) -> list[int]:
    candidates = remaining_indices.copy()
    max_similarity = embeddings[candidates] @ embeddings[np.asarray(initial_indices)].T
    max_similarity = max_similarity.max(axis=1)
    selected: list[int] = []
    for _ in range(QUERY_SIZE):
        distances = 1.0 - max_similarity
        best_position = int(np.argmax(distances))
        best_idx = int(candidates[best_position])
        selected.append(best_idx)
        new_similarity = embeddings[candidates] @ embeddings[best_idx]
        max_similarity = np.maximum(max_similarity, new_similarity)
        max_similarity[best_position] = np.inf
    return selected


def selection_metrics(query: pd.DataFrame, initial_embeddings: np.ndarray, query_embeddings: np.ndarray, bbox_counts: pd.Series) -> dict[str, Any]:
    anomaly = query["is_anomaly"].astype(bool).to_numpy()
    anomaly_ranks = np.flatnonzero(anomaly) + 1
    similarity = query_embeddings @ query_embeddings.T
    upper = similarity[np.triu_indices(len(query), k=1)]
    initial_similarity = query_embeddings @ initial_embeddings.T
    return {
        "query_anomaly_images": int(anomaly.sum()),
        "query_anomaly_rate": float(anomaly.mean()),
        "annotations_to_first_anomaly": int(anomaly_ranks[0]) if len(anomaly_ranks) else QUERY_SIZE + 1,
        "query_unique_object_categories": int(query["category"].nunique()),
        "query_unique_anomaly_types": int(query.loc[query["is_anomaly"].astype(bool), "label_raw"].nunique()),
        "query_defect_components": int(bbox_counts.reindex(query["sample_id"], fill_value=0).sum()),
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

    protocol_config = json.loads((protocol / "visa_protocol_config.json").read_text(encoding="utf-8"))
    embedding_config = json.loads((embedding_dir / "embedding_config.json").read_text(encoding="utf-8"))
    if bool(protocol_config.get("final_test_evaluated", True)) or bool(embedding_config.get("final_test_used", True)):
        raise RuntimeError("Final-test safety flag failed.")
    blind = pd.read_csv(protocol / "visa_acquisition_pool_blind.csv").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    gt = pd.read_csv(protocol / "visa_acquisition_pool_gt_audit.csv")
    embedding_manifest = pd.read_csv(embedding_dir / "embedding_manifest.csv")
    embeddings = np.load(embedding_dir / "embeddings.npy")
    if len(blind) != 8650 or embeddings.shape != (8650, 384):
        raise RuntimeError("Unexpected acquisition/embedding size.")
    if blind["sample_id"].tolist() != embedding_manifest["sample_id"].tolist():
        raise RuntimeError("Embedding manifest order does not match blind pool.")
    forbidden = {"is_anomaly", "label_raw", "mask_path", "category"}.intersection(blind.columns)
    if forbidden:
        raise RuntimeError(f"Blind selector view leaked GT columns: {sorted(forbidden)}")
    gt = blind[["sample_id"]].merge(gt, on="sample_id", how="left", validate="one_to_one")
    if gt["is_anomaly"].isna().any():
        raise RuntimeError("GT audit join failed.")
    boxes = pd.read_csv(protocol / "visa_acquisition_bbox_gt_audit.csv")
    bbox_counts = boxes.groupby("sample_id").size()

    all_indices = np.arange(len(blind), dtype=int)
    selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    category_yield_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        canonical = pd.DataFrame({"idx": all_indices})
        initial = canonical.sample(n=INITIAL_SIZE, random_state=seed + 999, replace=False)["idx"].astype(int).tolist()
        remaining = np.setdiff1d(all_indices, np.asarray(initial), assume_unique=True)
        queries = {
            RANDOM: random_query(remaining, seed),
            VISUAL: visual_query(remaining, initial, embeddings),
        }
        for strategy, query_indices in queries.items():
            query = gt.iloc[query_indices].copy()
            query_embeddings = embeddings[np.asarray(query_indices)]
            metrics = selection_metrics(query, embeddings[np.asarray(initial)], query_embeddings, bbox_counts)
            metric_rows.append({"acquisition_seed": seed, "strategy": strategy, **metrics})
            for rank, (idx, row) in enumerate(zip(query_indices, query.to_dict("records")), start=1):
                selection_rows.append(
                    {
                        "acquisition_seed": seed,
                        "strategy": strategy,
                        "rank_in_query": rank,
                        "sample_id": row["sample_id"],
                        "is_anomaly_posthoc": bool(row["is_anomaly"]),
                        "category_posthoc": row["category"],
                        "label_raw_posthoc": row["label_raw"],
                        "num_defect_components_posthoc": int(bbox_counts.get(row["sample_id"], 0)),
                    }
                )
            for category, sub in query.groupby("category"):
                category_yield_rows.append(
                    {
                        "acquisition_seed": seed,
                        "strategy": strategy,
                        "category": category,
                        "selected_images": len(sub),
                        "anomaly_images": int(sub["is_anomaly"].astype(bool).sum()),
                    }
                )
        if (seed + 1) % 20 == 0:
            print(f"[SELECTION] seeds={seed + 1}/{len(SEEDS)}", flush=True)

    selections = pd.DataFrame(selection_rows)
    metrics = pd.DataFrame(metric_rows)
    categories = pd.DataFrame(category_yield_rows)
    selections.to_csv(out / "visa_selection_records_posthoc.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(out / "visa_seed_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    categories.to_csv(out / "visa_category_yield_posthoc.csv", index=False, encoding="utf-8-sig")

    paired_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    metric_names = [column for column in metrics.columns if column not in {"acquisition_seed", "strategy"}]
    for metric in metric_names:
        pivot = metrics.pivot(index="acquisition_seed", columns="strategy", values=metric).loc[SEEDS]
        delta = (pivot[VISUAL] - pivot[RANDOM]).to_numpy(float)
        low, high = bootstrap_ci(delta)
        paired_rows.extend(
            {"acquisition_seed": seed, "metric": metric, "random": pivot.loc[seed, RANDOM], "visual": pivot.loc[seed, VISUAL], "delta": pivot.loc[seed, VISUAL] - pivot.loc[seed, RANDOM]}
            for seed in SEEDS
        )
        summary_rows.append(
            {
                "metric": metric,
                "random_mean": float(pivot[RANDOM].mean()),
                "visual_mean": float(pivot[VISUAL].mean()),
                "mean_difference": float(delta.mean()),
                "median_difference": float(np.median(delta)),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "wins": int((delta > 0).sum()),
                "losses": int((delta < 0).sum()),
                "ties": int((delta == 0).sum()),
                "win_rate": float((delta > 0).mean()),
                "monte_carlo_sign_flip_p_two_sided": exact_sign_flip(delta),
            }
        )
    paired = pd.DataFrame(paired_rows)
    summary = pd.DataFrame(summary_rows)
    paired.to_csv(out / "visa_paired_seed_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "visa_metric_summary.csv", index=False, encoding="utf-8-sig")

    anomaly = summary.set_index("metric").loc["query_anomaly_images"]
    category_metric = summary.set_index("metric").loc["query_unique_object_categories"]
    category_pivot = categories.pivot_table(index=["acquisition_seed", "category"], columns="strategy", values="anomaly_images", aggfunc="sum", fill_value=0).reset_index()
    leave_one_category_out: list[dict[str, Any]] = []
    for category in sorted(category_pivot["category"].unique()):
        remaining_category = category_pivot[~category_pivot["category"].eq(category)]
        per_seed = remaining_category.groupby("acquisition_seed")[[RANDOM, VISUAL]].sum()
        leave_one_category_out.append({"held_out_category": category, "mean_anomaly_yield_difference": float((per_seed[VISUAL] - per_seed[RANDOM]).mean())})
    loo = pd.DataFrame(leave_one_category_out)
    loo.to_csv(out / "visa_leave_one_category_out.csv", index=False, encoding="utf-8-sig")
    gates = [
        ("random_mean_anomaly_yield_at_most_3_of_20", float(anomaly["random_mean"]) <= 3.0),
        ("visual_mean_anomaly_yield_gain_at_least_2_of_20", float(anomaly["mean_difference"]) >= 2.0),
        ("anomaly_yield_bootstrap_ci_low_positive", float(anomaly["bootstrap_ci95_low"]) > 0.0),
        ("anomaly_yield_win_rate_at_least_0p65", float(anomaly["win_rate"]) >= 0.65),
        ("object_category_coverage_not_worse", float(category_metric["mean_difference"]) >= 0.0),
        ("leave_one_category_out_min_gain_positive", float(loo["mean_anomaly_yield_difference"].min()) > 0.0),
    ]
    gate = pd.DataFrame([{"check": name, "passed": passed} for name, passed in gates])
    gate.to_csv(out / "visa_selection_only_gate.csv", index=False, encoding="utf-8-sig")
    overall = all(passed for _, passed in gates)
    report = [
        "# VisA Selection-Only Random vs Frozen DINO Audit",
        "",
        "- Detector training performed: **False**",
        "- Final test used: **False**",
        "- GT/mask/bbox used by selectors: **False**",
        "- GT/mask/bbox used after selection for audit: **True**",
        f"- Seeds: **{len(SEEDS)}**",
        f"- Initial/query budget: **{INITIAL_SIZE}/{QUERY_SIZE}**",
        f"- Overall detector-training gate: **{'PASS' if overall else 'FAIL'}**",
        "",
        "## Metric summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Pre-registered gate",
        "",
        gate.assign(result=gate["passed"].map({True: "PASS", False: "FAIL"})).drop(columns="passed").to_markdown(index=False),
        "",
        "A failed gate prohibits detector training and post-hoc selector tuning.",
    ]
    summary_path = out / "visa_selection_only_audit_summary.md"
    summary_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    run_config = {
        "seeds": SEEDS,
        "initial_size": INITIAL_SIZE,
        "query_size": QUERY_SIZE,
        "strategies": [RANDOM, VISUAL],
        "detector_training_performed": False,
        "final_test_used": False,
        "overall_gate_pass": overall,
    }
    (out / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print("=" * 100)
    print("[DONE] VisA selection-only audit")
    print(f"[GATE] {'PASS' if overall else 'FAIL'}")
    print(f"[SUMMARY] {summary_path}")
    print("Detector training performed: False")
    print("Final test used: False")
    print("=" * 100)


if __name__ == "__main__":
    main()

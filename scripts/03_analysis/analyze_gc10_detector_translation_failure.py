"""Explain why GC10 taxonomy discovery did not translate to rare-class AP.

Uses only frozen acquisition selections, acquisition annotations, and completed
development metrics. It performs no new selection, training, or final access.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715"
DEFAULT_SELECTION = ROOT / "runs" / "gc10_taxonomy_selection_audit" / "gc10_random_vs_dino_200seed_20260715"
DEFAULT_DETECTOR = ROOT / "runs" / "gc10_detector_confirmation" / "gc10_dev_confirm_5acq_3train_20260715"
SEEDS = [0, 1, 2, 3, 4]
STRATEGIES = ["GTFreeRandom", "FrozenDINOVisualDiversity"]


def reconstruct_initial(blind: pd.DataFrame, seed: int) -> list[str]:
    indices = np.arange(len(blind), dtype=int)
    chosen = pd.DataFrame({"idx": indices}).sample(n=20, random_state=seed + 999, replace=False)["idx"].astype(int)
    return blind.iloc[chosen.tolist()]["sample_id"].astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--detector-dir", type=Path, default=DEFAULT_DETECTOR)
    args = parser.parse_args()
    protocol = args.protocol_dir.expanduser().resolve()
    selection = args.selection_dir.expanduser().resolve()
    detector = args.detector_dir.expanduser().resolve()
    protocol_config = json.loads((protocol / "gc10_protocol_config.json").read_text(encoding="utf-8"))
    selection_config = json.loads((selection / "config.json").read_text(encoding="utf-8"))
    detector_config = json.loads((detector / "config.json").read_text(encoding="utf-8"))
    if any(bool(config.get("final_test_used", config.get("final_test_evaluated", True))) for config in [protocol_config, selection_config, detector_config]):
        raise RuntimeError("Final-test safety flag failed")

    blind = pd.read_csv(protocol / "gc10_acquisition_pool_blind.csv").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    boxes = pd.read_csv(protocol / "gc10_acquisition_bbox_gt_audit.csv")
    selections = pd.read_csv(selection / "gc10_selection_records_posthoc.csv")
    detector_class = pd.read_csv(detector / "detector_per_class_metrics.csv")
    detector_runs = pd.read_csv(detector / "detector_run_metrics.csv")
    exposure_rows = []
    set_ids: dict[tuple[int, str], set[str]] = {}
    query_ids: dict[tuple[int, str], set[str]] = {}
    for seed in SEEDS:
        initial = reconstruct_initial(blind, seed)
        for strategy in STRATEGIES:
            query = selections[(selections["acquisition_seed"] == seed) & selections["strategy"].eq(strategy)].sort_values("rank_in_query")
            query_set = set(query["sample_id"].astype(str))
            train_set = set(initial) | query_set
            if len(query_set) != 20 or len(train_set) != 40:
                raise RuntimeError("Frozen set cardinality failure")
            query_ids[(seed, strategy)] = query_set
            set_ids[(seed, strategy)] = train_set
            sub = boxes[boxes["sample_id"].isin(train_set)]
            query_boxes = boxes[boxes["sample_id"].isin(query_set)]
            for class_id in range(1, 11):
                class_sub = sub[sub["class_id"].eq(class_id)]
                class_query = query_boxes[query_boxes["class_id"].eq(class_id)]
                exposure_rows.append({
                    "acquisition_seed": seed,
                    "strategy": strategy,
                    "class_id": class_id,
                    "train_images_with_class": int(class_sub["sample_id"].nunique()),
                    "train_instances": len(class_sub),
                    "train_bbox_area_ratio_sum": float(class_sub["bbox_area_ratio"].sum()),
                    "train_bbox_area_ratio_mean": float(class_sub["bbox_area_ratio"].mean()) if len(class_sub) else np.nan,
                    "query_images_with_class": int(class_query["sample_id"].nunique()),
                    "query_instances": len(class_query),
                })
    exposure = pd.DataFrame(exposure_rows)
    ap = detector_class.groupby(["acquisition_seed", "strategy", "class_id", "class_name"])["ap5095"].mean().reset_index()
    joined = exposure.merge(ap, on=["acquisition_seed", "strategy", "class_id"], how="left", validate="one_to_one")
    if joined["ap5095"].isna().any():
        raise RuntimeError("Detector AP join failed")
    joined.to_csv(detector / "detector_class_exposure_vs_ap.csv", index=False, encoding="utf-8-sig")

    mean_rows = []
    for class_id in range(1, 11):
        for metric in ["train_images_with_class", "train_instances", "train_bbox_area_ratio_sum", "train_bbox_area_ratio_mean", "query_images_with_class", "query_instances", "ap5095"]:
            pivot = joined[joined["class_id"].eq(class_id)].pivot(index="acquisition_seed", columns="strategy", values=metric)
            mean_rows.append({
                "class_id": class_id,
                "metric": metric,
                "random_mean": float(pivot[STRATEGIES[0]].mean()),
                "dino_mean": float(pivot[STRATEGIES[1]].mean()),
                "mean_difference": float((pivot[STRATEGIES[1]] - pivot[STRATEGIES[0]]).mean()),
            })
    means = pd.DataFrame(mean_rows)
    means.to_csv(detector / "detector_class_exposure_difference_summary.csv", index=False, encoding="utf-8-sig")

    overlap_rows = []
    for strategy in STRATEGIES:
        for left, right in itertools.combinations(SEEDS, 2):
            overlap_rows.append({
                "strategy": strategy,
                "seed_left": left,
                "seed_right": right,
                "query_overlap": len(query_ids[(left, strategy)] & query_ids[(right, strategy)]),
                "train_set_overlap": len(set_ids[(left, strategy)] & set_ids[(right, strategy)]),
            })
    overlaps = pd.DataFrame(overlap_rows)
    overlaps.to_csv(detector / "detector_five_seed_set_overlap.csv", index=False, encoding="utf-8-sig")

    unique_rows = []
    for strategy in STRATEGIES:
        all_query = set().union(*(query_ids[(seed, strategy)] for seed in SEEDS))
        all_train = set().union(*(set_ids[(seed, strategy)] for seed in SEEDS))
        for class_id in range(1, 11):
            class_samples = set(boxes.loc[boxes["class_id"].eq(class_id), "sample_id"].astype(str))
            unique_rows.append({
                "strategy": strategy,
                "class_id": class_id,
                "unique_query_images_across_five_seeds": len(all_query & class_samples),
                "unique_train_images_across_five_seeds": len(all_train & class_samples),
            })
    unique = pd.DataFrame(unique_rows)
    unique.to_csv(detector / "detector_unique_class_examples_across_five_seeds.csv", index=False, encoding="utf-8-sig")

    map_pivot = detector_runs.groupby(["acquisition_seed", "strategy"])[["map5095", "recall", "precision"]].mean().reset_index().pivot(index="acquisition_seed", columns="strategy")
    acquisition_differences = pd.DataFrame({
        "acquisition_seed": SEEDS,
        "map5095_difference": (map_pivot["map5095"][STRATEGIES[1]] - map_pivot["map5095"][STRATEGIES[0]]).reindex(SEEDS).to_numpy(),
        "recall_difference": (map_pivot["recall"][STRATEGIES[1]] - map_pivot["recall"][STRATEGIES[0]]).reindex(SEEDS).to_numpy(),
        "precision_difference": (map_pivot["precision"][STRATEGIES[1]] - map_pivot["precision"][STRATEGIES[0]]).reindex(SEEDS).to_numpy(),
    })
    acquisition_differences.to_csv(detector / "detector_acquisition_seed_differences.csv", index=False, encoding="utf-8-sig")

    rare = means[means["class_id"].isin([8, 9, 10])].pivot(index="class_id", columns="metric", values=["random_mean", "dino_mean", "mean_difference"])
    rare.columns = [f"{left}_{right}" for left, right in rare.columns]
    rare = rare.reset_index()
    overlap_summary = overlaps.groupby("strategy")[["query_overlap", "train_set_overlap"]].mean().reset_index()
    report = [
        "# GC10 Detector Translation-Failure Audit", "",
        "- New selection performed: **False**",
        "- New detector training performed: **False**",
        "- Final test used: **False**", "",
        "## Acquisition-seed detector differences", "", acquisition_differences.to_markdown(index=False, floatfmt=".6f"), "",
        "## Rare-class exposure and AP", "", rare.to_markdown(index=False, floatfmt=".6f"), "",
        "## Cross-seed set overlap", "", overlap_summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## Unique rare examples across the five training sets", "", unique[unique["class_id"].isin([8, 9, 10])].to_markdown(index=False), "",
        "The audit is descriptive. More rare labels do not imply representative within-class coverage, and repeated global prototypes reduce the effective independence of acquisition seeds.",
    ]
    summary = detector / "gc10_detector_translation_failure_audit.md"
    summary.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("=" * 100)
    print("[DONE] GC10 detector translation-failure audit")
    print("New selection/training: False/False")
    print("Final test used: False")
    print(f"[SUMMARY] {summary}")
    print("=" * 100)


if __name__ == "__main__":
    main()

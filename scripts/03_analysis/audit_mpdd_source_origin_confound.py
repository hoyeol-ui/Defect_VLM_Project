"""Post-hoc audit of official train/test-origin confounding in MPDD selections.

This script never reselects images. It joins the already frozen selection records
to acquisition-only audit metadata and does not read the final manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "runs" / "mpdd_annotation_triage_protocol" / "mpdd_protocol_20260715"
DEFAULT_RUN = ROOT / "runs" / "mpdd_selection_only_audit" / "mpdd_hierarchical_dino_200seed_20260715"
RANDOM = "GTFreeRandom"
BALANCED_RANDOM = "CategoryBalancedRandom"
BALANCED_DINO = "FrozenCategoryBalancedDINO"


def bootstrap_ci(values: np.ndarray, seed: int = 20260715) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    protocol = args.protocol_dir.expanduser().resolve()
    run = args.run_dir.expanduser().resolve()
    config = json.loads((protocol / "mpdd_protocol_config.json").read_text(encoding="utf-8"))
    run_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if bool(config.get("final_test_evaluated", True)) or bool(run_config.get("final_test_used", True)):
        raise RuntimeError("Final-test safety flag failed")
    selections = pd.read_csv(run / "mpdd_selection_records_posthoc.csv")
    gt = pd.read_csv(protocol / "mpdd_acquisition_pool_gt_audit.csv")
    audit_columns = ["sample_id", "official_split", "is_anomaly", "product_category", "anomaly_type"]
    joined = selections.drop(columns=[column for column in ["is_anomaly_posthoc", "product_category_posthoc", "anomaly_type_posthoc"] if column in selections]).merge(
        gt[audit_columns], on="sample_id", how="left", validate="many_to_one"
    )
    if joined["official_split"].isna().any():
        raise RuntimeError("Acquisition-only audit join failed")
    joined["is_official_test"] = joined["official_split"].eq("test")
    joined["is_test_good"] = joined["is_official_test"] & ~joined["is_anomaly"].astype(bool)

    seed_metrics = joined.groupby(["acquisition_seed", "strategy"]).agg(
        query_images=("sample_id", "size"),
        official_test_images=("is_official_test", "sum"),
        official_train_images=("is_official_test", lambda values: int((~values).sum())),
        anomaly_images=("is_anomaly", "sum"),
        official_test_good_images=("is_test_good", "sum"),
    ).reset_index()
    seed_metrics["anomaly_rate_within_selected_test"] = seed_metrics["anomaly_images"] / seed_metrics["official_test_images"]
    seed_metrics.to_csv(run / "mpdd_source_origin_seed_metrics.csv", index=False, encoding="utf-8-sig")

    comparisons = [(BALANCED_DINO, RANDOM), (BALANCED_DINO, BALANCED_RANDOM), (BALANCED_RANDOM, RANDOM)]
    paired_rows = []
    for method, baseline in comparisons:
        for metric in ["official_test_images", "official_train_images", "official_test_good_images", "anomaly_images", "anomaly_rate_within_selected_test"]:
            pivot = seed_metrics.pivot(index="acquisition_seed", columns="strategy", values=metric)
            delta = (pivot[method] - pivot[baseline]).to_numpy(float)
            low, high = bootstrap_ci(delta)
            paired_rows.append({
                "comparison": f"{method}-minus-{baseline}",
                "metric": metric,
                "baseline_mean": float(pivot[baseline].mean()),
                "method_mean": float(pivot[method].mean()),
                "mean_difference": float(delta.mean()),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "wins": int((delta > 0).sum()),
                "losses": int((delta < 0).sum()),
                "ties": int((delta == 0).sum()),
            })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(run / "mpdd_source_origin_paired_summary.csv", index=False, encoding="utf-8-sig")

    composition_rows = []
    means = seed_metrics.groupby("strategy")[["official_test_images", "anomaly_images"]].mean()
    for method, baseline in [(BALANCED_DINO, RANDOM), (BALANCED_DINO, BALANCED_RANDOM)]:
        test_b = float(means.loc[baseline, "official_test_images"])
        test_m = float(means.loc[method, "official_test_images"])
        anomaly_b = float(means.loc[baseline, "anomaly_images"])
        anomaly_m = float(means.loc[method, "anomaly_images"])
        rate_b = anomaly_b / test_b
        rate_m = anomaly_m / test_m
        composition = (test_m - test_b) * rate_b
        within_test = test_m * (rate_m - rate_b)
        composition_rows.append({
            "comparison": f"{method}-minus-{baseline}",
            "observed_anomaly_yield_gain": anomaly_m - anomaly_b,
            "gain_explained_by_more_official_test_images": composition,
            "gain_explained_by_higher_anomaly_rate_within_test": within_test,
            "composition_fraction_of_observed_gain": composition / (anomaly_m - anomaly_b),
            "baseline_selected_test_anomaly_rate": rate_b,
            "method_selected_test_anomaly_rate": rate_m,
        })
    decomposition = pd.DataFrame(composition_rows)
    decomposition.to_csv(run / "mpdd_source_origin_gain_decomposition.csv", index=False, encoding="utf-8-sig")

    pool = gt.groupby(["product_category", "official_split"]).agg(
        pool_images=("sample_id", "size"), anomaly_images=("is_anomaly", "sum")
    ).reset_index()
    pool["anomaly_rate"] = pool["anomaly_images"] / pool["pool_images"]
    selected_category = joined.groupby(["strategy", "product_category"]).agg(
        selected_images=("sample_id", "size"),
        selected_test_images=("is_official_test", "sum"),
        selected_anomaly_images=("is_anomaly", "sum"),
    ).reset_index()
    selected_category[["selected_images", "selected_test_images", "selected_anomaly_images"]] = selected_category[["selected_images", "selected_test_images", "selected_anomaly_images"]] / 200.0
    selected_category.to_csv(run / "mpdd_source_origin_by_category.csv", index=False, encoding="utf-8-sig")

    primary = paired[paired["comparison"].eq(f"{BALANCED_DINO}-minus-{RANDOM}")]
    report = [
        "# MPDD Official-Source-Origin Confound Audit", "",
        "- New selection performed: **False**",
        "- Detector training performed: **False**",
        "- Final test used: **False**",
        "- Metadata joined: **acquisition-only, post-hoc**", "",
        "MPDD's official train partition contains only normal images, while all anomaly images occur in the official test partition. Therefore, selecting official-test-looking images can increase anomaly yield even without defect-specific recognition.", "",
        "## Hierarchical DINO versus Random", "", primary.to_markdown(index=False, floatfmt=".6f"), "",
        "## Mean-level anomaly-gain decomposition", "", decomposition.to_markdown(index=False, floatfmt=".6f"), "",
        "The decomposition is descriptive: it separates the observed mean anomaly gain into a source-composition term and a within-official-test anomaly-enrichment term. It is not causal proof.", "",
        "## Acquisition-pool source composition", "", pool.to_markdown(index=False, floatfmt=".6f"), "",
        "## Selected source composition by product category (means per 20-image query)", "", selected_category.to_markdown(index=False, floatfmt=".6f"), "",
        "This audit cannot rescue the pre-registered selection gate. It only determines how cautiously the large anomaly-yield effect must be interpreted.",
    ]
    summary = run / "mpdd_source_origin_confound_audit.md"
    summary.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("=" * 100)
    print("[DONE] MPDD source-origin confound audit")
    print("New selection performed: False")
    print("Detector training performed: False")
    print("Final test used: False")
    print(f"[SUMMARY] {summary}")
    print("=" * 100)


if __name__ == "__main__":
    main()

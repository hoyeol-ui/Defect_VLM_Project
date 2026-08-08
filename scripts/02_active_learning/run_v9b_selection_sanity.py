"""V9b selection-only sanity check across acquisition seeds.

This script does not train YOLO and does not evaluate the final test split.

Goal:
    Before paying for multi-seed full-curve training, verify that the locked
    V9b instance-rich detector-aware selector produces reasonable round-1
    batches across seeds 42-46.

Data protocol:
    - Acquisition pool: V8 NEU-only 50-image pool.
    - Initial labeled set: V8 shared round-0 set for each seed.
    - Detector used for scoring: V8 seed-specific round-0 detector.
    - Query size: 5.
    - Post-hoc XML statistics are diagnostics only, not acquisition inputs.
    - Final test: locked / not used.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("AL_POOL_DATASET_FILTER", "NEU-DET")
os.environ.setdefault("AL_DEV_EVAL_DATASET_FILTER", "NEU-DET")
os.environ.setdefault("AL_EVAL_SPLIT", "development_eval_v7")
os.environ.setdefault("AL_EMBEDDING_BACKEND", "dinov2")
os.environ.setdefault("AL_ALLOW_MODEL_DOWNLOAD", "0")
os.environ.setdefault("AL_V9_SELECTOR", "instance_rich")
os.environ.setdefault("AL_V9_STRATEGY", "DetectorInstanceRichDINOBalanced")

import probe_v9_detector_aware_selection as probe  # noqa: E402
import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402
import run_v9_stage1_seed42_round1 as stage1  # noqa: E402
from audit_detection_pipeline_v7 import build_image_index, build_xml_index, parse_int_list_env  # noqa: E402
from run_al_yolo_ablation_v7_full_curve import (  # noqa: E402
    file_sha256,
    git_commit,
    git_dirty,
    load_development_eval,
    load_dino_embeddings,
    load_priority_pool_with_identity,
)

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    raise ImportError("Install ultralytics before running V9b selection sanity.") from exc


PROJECT_ROOT = v6.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "v9b_selection_sanity"
SOURCE_RUN = stage1.DEFAULT_SOURCE_RUN
STRATEGY = os.environ.get("AL_V9_STRATEGY", "DetectorInstanceRichDINOBalanced")
BASELINE_STRATEGIES = [
    "GTFreeRandom",
    "GTFreeDatasetBalancedConsistency",
    "GTFreeDatasetBalancedVisualDiversity",
]


def resolve_source_run() -> Path:
    override = os.environ.get("AL_V9_SOURCE_RUN_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    return SOURCE_RUN


def add_source_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    out = df.copy()
    out["source_strategy"] = strategy
    return out


def load_v8_round1_baselines(source_run: Path, seed: int) -> pd.DataFrame:
    selected_path = source_run / "all_selected_samples_by_round.csv"
    selected = pd.read_csv(selected_path)
    sub = selected[
        pd.to_numeric(selected["acquisition_seed"], errors="coerce").eq(seed)
        & pd.to_numeric(selected["round"], errors="coerce").eq(1)
        & selected["selection_type"].astype(str).ne("cumulative_labeled")
        & selected["strategy"].astype(str).isin(BASELINE_STRATEGIES)
    ].copy()
    sub["source_strategy"] = sub["strategy"].astype(str)
    return sub


def actual_stats_by_seed(selected_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_index = build_image_index()
    xml_index = build_xml_index()
    instance_parts: list[pd.DataFrame] = []
    stat_parts: list[pd.DataFrame] = []
    for seed, sub in selected_df.groupby("acquisition_seed", dropna=False):
        inst, stats = probe.posthoc_stats(sub, image_index, xml_index)
        if len(inst):
            inst.insert(0, "acquisition_seed", int(seed))
            instance_parts.append(inst)
        if len(stats):
            stats.insert(0, "acquisition_seed", int(seed))
            stat_parts.append(stats)
    return (
        pd.concat(instance_parts, ignore_index=True, sort=False) if instance_parts else pd.DataFrame(),
        pd.concat(stat_parts, ignore_index=True, sort=False) if stat_parts else pd.DataFrame(),
    )


def summarize_relative_to_random(stats: pd.DataFrame, scope: str) -> pd.DataFrame:
    if stats.empty or "source_strategy" not in stats.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for seed, seed_df in stats.groupby("acquisition_seed", dropna=False):
        random_rows = seed_df[seed_df["source_strategy"].eq("GTFreeRandom")]
        v9_rows = seed_df[seed_df["source_strategy"].eq(STRATEGY)]
        if random_rows.empty or v9_rows.empty:
            continue
        rnd = random_rows.iloc[0]
        v9 = v9_rows.iloc[0]
        rows.append(
            {
                "scope": scope,
                "acquisition_seed": int(seed),
                "v9_bbox_minus_random": float(v9["total_bbox_instances"]) - float(rnd["total_bbox_instances"]),
                "v9_entropy_minus_random": float(v9["actual_class_entropy"]) - float(rnd["actual_class_entropy"]),
                "v9_multi_ratio_minus_random": float(v9["multi_object_image_ratio"]) - float(rnd["multi_object_image_ratio"]),
                "v9_classes_minus_random": float(v9["num_actual_classes"]) - float(rnd["num_actual_classes"]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_stats(stats: pd.DataFrame, scope: str) -> pd.DataFrame:
    if stats.empty:
        return pd.DataFrame()
    rows = []
    for strategy, sub in stats.groupby("source_strategy", dropna=False):
        rows.append(
            {
                "scope": scope,
                "source_strategy": strategy,
                "num_seeds": int(sub["acquisition_seed"].nunique()),
                "bbox_mean": float(pd.to_numeric(sub["total_bbox_instances"], errors="coerce").mean()),
                "bbox_min": float(pd.to_numeric(sub["total_bbox_instances"], errors="coerce").min()),
                "bbox_max": float(pd.to_numeric(sub["total_bbox_instances"], errors="coerce").max()),
                "entropy_mean": float(pd.to_numeric(sub["actual_class_entropy"], errors="coerce").mean()),
                "classes_mean": float(pd.to_numeric(sub["num_actual_classes"], errors="coerce").mean()),
                "multi_object_ratio_mean": float(pd.to_numeric(sub["multi_object_image_ratio"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["scope", "source_strategy"], kind="mergesort")


def table_md(df: pd.DataFrame) -> str:
    return probe.table_md(df) if len(df) else "_No rows._"


def write_summary(
    save_dir: Path,
    *,
    config: dict[str, Any],
    batch_agg: pd.DataFrame,
    cumulative_agg: pd.DataFrame,
    batch_delta: pd.DataFrame,
    cumulative_delta: pd.DataFrame,
) -> None:
    lines = [
        "# V9b 5-seed selection-only sanity",
        "",
        "No YOLO training. No final-test evaluation. XML statistics are post-hoc diagnostics only.",
        "",
        "## Data protocol",
        "",
        f"- Acquisition seeds: {config['acquisition_seeds']}",
        f"- Acquisition pool: NEU-DET only, {config['pool_size_after_filter']} images",
        f"- Initial labeled set per seed: {config['initial_size']} images",
        f"- Query size: {config['query_size']} images",
        f"- Selector: {config['selector']}",
        f"- Stage 1b weights: `{json.dumps(config['stage1b_instance_rich_weights'], ensure_ascii=False)}`",
        f"- Stage 1b constraints: `{json.dumps(config['v9b_constraints'], ensure_ascii=False)}`",
        "- Final test: locked / unused",
        "",
        "## Batch-level aggregate stats",
        "",
        table_md(batch_agg),
        "",
        "## Cumulative-after-round1 aggregate stats",
        "",
        table_md(cumulative_agg),
        "",
        "## V9b minus Random, batch level",
        "",
        table_md(batch_delta),
        "",
        "## V9b minus Random, cumulative level",
        "",
        table_md(cumulative_delta),
        "",
        "## Reading guide",
        "",
        "- Pass condition is not performance; this only checks whether V9b avoids composition collapse before full training.",
        "- Strong warning signs: lower bbox count than Random in most seeds, class entropy collapse, or repeated no-box/one-class batches.",
        "- If this passes, the next paid step is 5-seed full-curve training on the current 50-image pilot pool.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        "```",
    ]
    (save_dir / "v9b_selection_sanity_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"v9b_selection_sanity_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    source_run = resolve_source_run()
    seeds = parse_int_list_env("AL_ACQUISITION_SEEDS", [42, 43, 44, 45, 46])
    query_size = int(os.environ.get("AL_QUERY_SIZE", "5"))
    candidate_fraction = float(os.environ.get("AL_V9_CANDIDATE_FRACTION", "1.00"))
    imgsz = int(os.environ.get("AL_IMGSZ", "640"))
    device = os.environ.get("AL_YOLO_DEVICE", "0")
    conf = float(os.environ.get("AL_V9_PREDICT_CONF", "0.05"))
    iou = float(os.environ.get("AL_V9_PREDICT_IOU", "0.70"))

    priority_csv, full_pool, _ = load_priority_pool_with_identity()
    protocol_dir, dev_eval_path, dev_eval_df = load_development_eval()
    embedding_dir, _, _, embedding_lookup, dino_config = load_dino_embeddings(full_pool)
    weights = stage1.v9b_weight_config()
    max_no_box = stage1.int_env("AL_V9B_MAX_NO_BOX", 0)
    min_pseudo_boxes = stage1.int_env("AL_V9B_MIN_PSEUDO_BOXES", 2)
    max_per_pred_class = stage1.int_env("AL_V9B_MAX_PER_PRED_CLASS", 2)

    selected_parts: list[pd.DataFrame] = []
    cumulative_parts: list[pd.DataFrame] = []
    batch_for_stats_parts: list[pd.DataFrame] = []
    cumulative_for_stats_parts: list[pd.DataFrame] = []
    score_parts: list[pd.DataFrame] = []

    print("=" * 100)
    print("[V9b selection-only sanity]")
    print(f"Output dir : {save_dir}")
    print(f"Seeds      : {seeds}")
    print(f"Pool       : NEU-DET {len(full_pool)} images")
    print(f"Dev eval   : NEU-DET {len(dev_eval_df)} images (metadata only; no evaluation)")
    print("Final test : LOCKED / NOT USED")
    print("=" * 100)

    for seed in seeds:
        print(f"[seed {seed}] scoring with V8 round-0 detector...")
        initial_df, _ = probe.load_source_initial_and_baselines(source_run, seed)
        initial_df = initial_df.drop_duplicates("sample_id", keep="first").reset_index(drop=True)
        current_pool = full_pool[~full_pool["sample_id"].isin(initial_df["sample_id"])].copy().reset_index(drop=True)
        detector_weights = probe.resolve_weights(source_run, seed)
        if not detector_weights.exists():
            raise FileNotFoundError(f"Missing V8 round-0 detector weights for seed {seed}: {detector_weights}")
        model = YOLO(str(detector_weights))
        detector_scores = probe.prediction_rows(model, current_pool, device=device, imgsz=imgsz, conf=conf, iou=iou)
        scored_pool = current_pool.merge(detector_scores, on="sample_id", how="left")
        scored_pool.insert(0, "acquisition_seed", seed)
        score_parts.append(scored_pool)

        v9_selected = stage1.select_instance_rich_dino_balanced(
            scored_pool,
            initial_df,
            embedding_lookup,
            query_size=query_size,
            candidate_fraction=candidate_fraction,
            weights=weights,
            max_no_box=max_no_box,
            min_pseudo_boxes=min_pseudo_boxes,
            max_per_pred_class=max_per_pred_class,
        )
        v9_selected = stage1.set_front_metadata(
            v9_selected.reset_index(drop=True),
            {
                "acquisition_seed": seed,
                "strategy": STRATEGY,
                "round": 1,
                "selection_type": STRATEGY,
            },
        )
        if "rank_in_selection" in v9_selected.columns:
            v9_selected = v9_selected.drop(columns=["rank_in_selection"])
        v9_selected.insert(4, "rank_in_selection", range(1, len(v9_selected) + 1))

        baselines = load_v8_round1_baselines(source_run, seed)
        selected_parts.append(v9_selected)
        selected_parts.append(baselines)

        batch_for_stats_parts.append(add_source_strategy(v9_selected, STRATEGY))
        batch_for_stats_parts.append(baselines)

        v9_cumulative = add_source_strategy(
            pd.concat([initial_df, v9_selected], ignore_index=True, sort=False).drop_duplicates("sample_id", keep="first"),
            STRATEGY,
        )
        v9_cumulative = stage1.set_front_metadata(v9_cumulative, {"acquisition_seed": seed})
        cumulative_for_stats_parts.append(v9_cumulative)
        cumulative_parts.append(v9_cumulative)

        for strategy, sub in baselines.groupby("source_strategy", dropna=False):
            cum = add_source_strategy(
                pd.concat([initial_df, sub], ignore_index=True, sort=False).drop_duplicates("sample_id", keep="first"),
                str(strategy),
            )
            cum = stage1.set_front_metadata(cum, {"acquisition_seed": seed})
            cumulative_for_stats_parts.append(cum)

    selected_df = pd.concat(selected_parts, ignore_index=True, sort=False)
    cumulative_df = pd.concat(cumulative_parts, ignore_index=True, sort=False)
    score_df = pd.concat(score_parts, ignore_index=True, sort=False)
    batch_stats_input = pd.concat(batch_for_stats_parts, ignore_index=True, sort=False)
    cumulative_stats_input = pd.concat(cumulative_for_stats_parts, ignore_index=True, sort=False)

    batch_instances, batch_stats = actual_stats_by_seed(batch_stats_input)
    cumulative_instances, cumulative_stats = actual_stats_by_seed(cumulative_stats_input)

    batch_agg = aggregate_stats(batch_stats, "batch")
    cumulative_agg = aggregate_stats(cumulative_stats, "cumulative")
    batch_delta = summarize_relative_to_random(batch_stats, "batch")
    cumulative_delta = summarize_relative_to_random(cumulative_stats, "cumulative")

    config = {
        "experiment_id": save_dir.name,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "save_dir": str(save_dir),
        "source_run": str(source_run),
        "status": "selection_only",
        "final_test_used": False,
        "yolo_training_run": False,
        "selector": "instance_rich",
        "strategy": STRATEGY,
        "acquisition_seeds": seeds,
        "query_size": query_size,
        "initial_size": 15,
        "pool_dataset_filter": ["NEU-DET"],
        "pool_size_after_filter": int(len(full_pool)),
        "priority_csv": str(priority_csv),
        "priority_csv_sha256": file_sha256(priority_csv),
        "eval_protocol_dir": str(protocol_dir),
        "development_eval_path": str(dev_eval_path),
        "development_eval_sha256": file_sha256(dev_eval_path),
        "development_eval_size_after_filter": int(len(dev_eval_df)),
        "embedding_dir": str(embedding_dir),
        "DINO_manifest_sha256": file_sha256(embedding_dir / "embedding_manifest.csv"),
        "DINO_config": dino_config,
        "candidate_fraction": candidate_fraction,
        "predict_conf": conf,
        "predict_iou": iou,
        "v9b_constraints": {
            "max_no_box": max_no_box,
            "min_pseudo_boxes": min_pseudo_boxes,
            "max_per_pred_class": max_per_pred_class,
        },
        "stage1b_instance_rich_weights": weights,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }

    selected_df.to_csv(save_dir / "v9b_selected_samples_with_v8_baselines.csv", index=False, encoding="utf-8-sig")
    cumulative_df.to_csv(save_dir / "v9b_cumulative_labeled_sets.csv", index=False, encoding="utf-8-sig")
    score_df.to_csv(save_dir / "v9b_detector_scores_by_seed.csv", index=False, encoding="utf-8-sig")
    batch_instances.to_csv(save_dir / "v9b_batch_instances.csv", index=False, encoding="utf-8-sig")
    batch_stats.to_csv(save_dir / "v9b_batch_actual_instance_stats.csv", index=False, encoding="utf-8-sig")
    cumulative_instances.to_csv(save_dir / "v9b_cumulative_instances.csv", index=False, encoding="utf-8-sig")
    cumulative_stats.to_csv(save_dir / "v9b_cumulative_actual_instance_stats.csv", index=False, encoding="utf-8-sig")
    batch_agg.to_csv(save_dir / "v9b_batch_aggregate_stats.csv", index=False, encoding="utf-8-sig")
    cumulative_agg.to_csv(save_dir / "v9b_cumulative_aggregate_stats.csv", index=False, encoding="utf-8-sig")
    batch_delta.to_csv(save_dir / "v9b_batch_delta_vs_random.csv", index=False, encoding="utf-8-sig")
    cumulative_delta.to_csv(save_dir / "v9b_cumulative_delta_vs_random.csv", index=False, encoding="utf-8-sig")
    (save_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_summary(
        save_dir,
        config=config,
        batch_agg=batch_agg,
        cumulative_agg=cumulative_agg,
        batch_delta=batch_delta,
        cumulative_delta=cumulative_delta,
    )

    print("=" * 100)
    print("[DONE] V9b selection-only sanity finished")
    print(f"Output dir: {save_dir}")
    print("No YOLO training. Final test used: False")
    print("=" * 100)


if __name__ == "__main__":
    main()

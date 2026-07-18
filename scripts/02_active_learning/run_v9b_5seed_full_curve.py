"""V9b 5-seed pilot full-curve runner.

This is the paid pilot check after V9b passed selection-only sanity.

Protocol:
    - Domain/data: NEU-DET only, same 50-image pilot acquisition pool as V8.
    - Evaluation: NEU-only development_eval_v7, 177 images.
    - Final test: locked / not used.
    - Seeds: 42,43,44,45,46 by default.
    - Budgets: 15 -> 20 -> 25 -> 30 -> 35.
    - Round 0 detector: copied V8 seed-specific round-0 detector.
    - Rounds 1-4: V9b instance-rich detector-aware acquisition, with each
      next round scored by the previous V9b round detector.

Default is dry-run.  Set AL_DRY_RUN_ONLY=0 for the 20 YOLO trainings.
"""

from __future__ import annotations

import json
import os
import sys
import time
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
import run_al_yolo_ablation_v7_full_curve as v7  # noqa: E402
import run_v9_seed42_full_curve as single_curve  # noqa: E402
import run_v9_stage1_seed42_round1 as stage1  # noqa: E402
from audit_detection_pipeline_v7 import parse_bool_env, parse_int_list_env  # noqa: E402
from run_al_yolo_ablation_v7_full_curve import (  # noqa: E402
    append_run_registry,
    dataset_distribution_by_round,
    extract_per_class_metrics,
    file_sha256,
    git_commit,
    git_dirty,
    load_development_eval,
    load_dino_embeddings,
    load_priority_pool_with_identity,
    train_eval,
)


PROJECT_ROOT = v6.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "active_learning_ablation_v9_detector_aware"
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "active_learning_ablation_v9_detector_aware"
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
    return stage1.DEFAULT_SOURCE_RUN


def copy_v8_baseline_rows(source_run: Path, seeds: list[int]) -> pd.DataFrame:
    path = source_run / "all_round_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    keep = df[
        pd.to_numeric(df["acquisition_seed"], errors="coerce").isin(seeds)
        & df["strategy"].astype(str).isin(BASELINE_STRATEGIES)
    ].copy()
    keep["result_origin"] = "copied_from_v8_neu_only"
    return keep


def write_summary(
    save_dir: Path,
    *,
    config: dict[str, Any],
    results_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    seed_summary: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    actual_stats_df: pd.DataFrame,
) -> None:
    result_cols = [
        "result_origin",
        "acquisition_seed",
        "strategy",
        "round",
        "labeled_budget",
        "map50",
        "map5095",
        "precision",
        "recall",
        "train_status",
    ]
    summary_cols = [
        "acquisition_seed",
        "strategy",
        "final_map50",
        "final_map5095",
        "normalized_aulc_map50",
        "normalized_aulc_map5095",
        "best_map50",
        "best_map5095",
    ]
    lines = [
        "# V9b 5-seed pilot full curve",
        "",
        "Pilot full curve on the current 50-image NEU-only acquisition pool. Final test was not used.",
        "",
        "## Data protocol",
        "",
        f"- Acquisition seeds: {config['acquisition_seeds']}",
        f"- Acquisition pool: NEU-DET only, {config['pool_size_after_filter']} images",
        f"- Initial labeled set: {config['initial_seed_size']} images per seed",
        f"- Query size: {config['query_size']} images",
        f"- Budgets: {config['budgets']}",
        f"- Development evaluation: NEU-DET only, {config['development_eval_size_after_filter']} images",
        "- Final test: locked / unused",
        f"- Selector: {config['selector']}",
        f"- Stage 1b weights: `{json.dumps(config['stage1b_instance_rich_weights'], ensure_ascii=False)}`",
        f"- Stage 1b constraints: `{json.dumps(config['v9b_constraints'], ensure_ascii=False)}`",
        "",
        "## Aggregate metric summary",
        "",
        probe.table_md(aggregate_df),
        "",
        "## Seed-level metric summary",
        "",
        probe.table_md(seed_summary[[c for c in summary_cols if c in seed_summary.columns]]),
        "",
        "## V9b round results",
        "",
        probe.table_md(results_df[[c for c in result_cols if c in results_df.columns]]),
        "",
        "## V8 baseline rows copied for comparison",
        "",
        probe.table_md(baseline_df[[c for c in result_cols if c in baseline_df.columns]]),
        "",
        "## Post-hoc actual instance statistics",
        "",
        probe.table_md(actual_stats_df),
        "",
        "## Reading guide",
        "",
        "- This is still a pilot because the acquisition pool has only 50 NEU images.",
        "- If V9b is competitive on final mAP@50-95 and not badly behind on AULC mAP@50-95, expand to a large NEU pool.",
        "- Do not use final_test_v7 until the method and data protocol are locked.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        "```",
    ]
    (save_dir / "v9b_5seed_full_curve_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"v9b_5seed_full_curve_{timestamp}"
    dataset_root = DATASETS_ROOT / save_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    source_run = resolve_source_run()
    seeds = parse_int_list_env("AL_ACQUISITION_SEEDS", [42, 43, 44, 45, 46])
    initial_size = int(os.environ.get("AL_INITIAL_SEED_SIZE", "15"))
    rounds = int(os.environ.get("AL_ROUNDS", "4"))
    query_size = int(os.environ.get("AL_QUERY_SIZE", "5"))
    budgets = [initial_size + query_size * r for r in range(rounds + 1)]
    candidate_fraction = float(os.environ.get("AL_V9_CANDIDATE_FRACTION", "1.00"))
    imgsz = int(os.environ.get("AL_IMGSZ", "640"))
    device = os.environ.get("AL_YOLO_DEVICE", "0")
    conf = float(os.environ.get("AL_V9_PREDICT_CONF", "0.05"))
    iou = float(os.environ.get("AL_V9_PREDICT_IOU", "0.70"))
    dry_run = parse_bool_env("AL_DRY_RUN_ONLY", True)

    priority_csv, full_pool, _ = load_priority_pool_with_identity()
    protocol_dir, dev_eval_path, dev_eval_df = load_development_eval()
    embedding_dir, _, _, embedding_lookup, dino_config = load_dino_embeddings(full_pool)

    config = {
        "experiment_id": save_dir.name,
        "experiment_label": "V9b detector-aware 5-seed pilot full curve",
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "save_dir": str(save_dir),
        "dataset_root": str(dataset_root),
        "source_run": str(source_run),
        "stage": "5seed_pilot_full_curve",
        "selector": "instance_rich",
        "strategy": STRATEGY,
        "acquisition_seeds": seeds,
        "training_seed_rule": "training_seed = 1000 + acquisition_seed",
        "initial_seed_size": initial_size,
        "rounds": rounds,
        "query_size": query_size,
        "budgets": budgets,
        "pool_dataset_filter": ["NEU-DET"],
        "pool_size_after_filter": int(len(full_pool)),
        "priority_csv": str(priority_csv),
        "priority_csv_sha256": file_sha256(priority_csv),
        "development_eval_path": str(dev_eval_path),
        "development_eval_sha256": file_sha256(dev_eval_path),
        "development_eval_dataset_filter": ["NEU-DET"],
        "development_eval_size_after_filter": int(len(dev_eval_df)),
        "eval_protocol_dir": str(protocol_dir),
        "final_test_used": False,
        "embedding_dir": str(embedding_dir),
        "DINO_manifest_sha256": file_sha256(embedding_dir / "embedding_manifest.csv"),
        "DINO_config": dino_config,
        "candidate_fraction": candidate_fraction,
        "predict_conf": conf,
        "predict_iou": iou,
        "v9b_constraints": {
            "max_no_box": stage1.int_env("AL_V9B_MAX_NO_BOX", 0),
            "min_pseudo_boxes": stage1.int_env("AL_V9B_MIN_PSEUDO_BOXES", 2),
            "max_per_pred_class": stage1.int_env("AL_V9B_MAX_PER_PRED_CLASS", 2),
        },
        "stage1b_instance_rich_weights": stage1.v9b_weight_config(),
        "model": os.environ.get("AL_YOLO_MODEL_NAME", "yolov8n.pt"),
        "epochs": int(os.environ.get("AL_EPOCHS_PER_ROUND", "100")),
        "patience": int(os.environ.get("AL_YOLO_PATIENCE", os.environ.get("AL_EPOCHS_PER_ROUND", "100"))),
        "batch": int(os.environ.get("AL_BATCH_SIZE", "8")),
        "workers": int(os.environ.get("AL_WORKERS", "4")),
        "cache": os.environ.get("AL_YOLO_CACHE", "ram"),
        "dry_run": dry_run,
        "expected_trainings": int(len(seeds) * rounds),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    (save_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("[V9b 5-seed pilot full curve]")
    print(f"Output dir : {save_dir}")
    print(f"Seeds      : {seeds}")
    print(f"Pool       : NEU-DET {len(full_pool)} images")
    print(f"Dev eval   : NEU-DET {len(dev_eval_df)} images")
    print(f"Budgets    : {budgets}")
    print(f"Dry run    : {dry_run}")
    print(f"Expected YOLO trainings if enabled: {len(seeds) * rounds}")
    print("Final test : LOCKED / NOT USED")
    print("=" * 100)

    selected_logs: list[pd.DataFrame] = []
    cumulative_logs: list[pd.DataFrame] = []
    results_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    build_logs: list[pd.DataFrame] = []
    per_class_rows: list[pd.DataFrame] = []
    failed_rows: list[dict[str, Any]] = []

    for seed in seeds:
        print("-" * 100)
        print(f"[seed {seed}] starting V9b full curve")
        training_seed = 1000 + seed
        initial_df, _ = probe.load_source_initial_and_baselines(source_run, seed)
        initial_df = initial_df.drop_duplicates("sample_id", keep="first").reset_index(drop=True)
        if len(initial_df) != initial_size:
            raise ValueError(f"Seed {seed}: expected initial_size={initial_size}, got {len(initial_df)}")

        labeled = initial_df.copy()
        current_pool = full_pool[~full_pool["sample_id"].isin(labeled["sample_id"])].copy().reset_index(drop=True)
        selected_logs.append(
            single_curve.add_round_metadata(
                initial_df,
                acquisition_seed=seed,
                round_idx=0,
                selection_type="shared_initial_seed_random",
            )
        )
        cumulative_logs.append(
            single_curve.add_round_metadata(
                labeled,
                acquisition_seed=seed,
                round_idx=0,
                selection_type="cumulative_labeled",
            )
        )

        round0_row = single_curve.v8_round0_result(source_run, seed, initial_size, len(dev_eval_df))
        round0_row["result_origin"] = "copied_v8_shared_round0"
        round0_row["strategy"] = STRATEGY
        results_rows.append(round0_row)

        prev_weights = probe.resolve_weights(source_run, seed)
        if not prev_weights.exists():
            raise FileNotFoundError(f"Seed {seed}: missing V8 round-0 weights: {prev_weights}")

        for round_idx in range(1, rounds + 1):
            print(f"[seed {seed}] round {round_idx}/{rounds}: selecting query batch")
            seed_score_dir = save_dir / f"seed_{seed}"
            seed_score_dir.mkdir(parents=True, exist_ok=True)
            picked = single_curve.select_next_batch(
                detector_weights=prev_weights,
                current_pool=current_pool,
                labeled=labeled,
                embedding_lookup=embedding_lookup,
                acquisition_seed=seed,
                round_idx=round_idx,
                query_size=query_size,
                candidate_fraction=candidate_fraction,
                imgsz=imgsz,
                device=device,
                conf=conf,
                iou=iou,
                save_dir=seed_score_dir,
            )
            selected_logs.append(picked)
            labeled = pd.concat([labeled, picked], ignore_index=True, sort=False).drop_duplicates("sample_id", keep="first")
            current_pool = current_pool[~current_pool["sample_id"].isin(picked["sample_id"])].copy().reset_index(drop=True)
            cumulative_logs.append(
                single_curve.add_round_metadata(
                    labeled,
                    acquisition_seed=seed,
                    round_idx=round_idx,
                    selection_type="cumulative_labeled",
                )
            )

            dataset_dir = dataset_root / f"seed_{seed}" / STRATEGY / f"round_{round_idx}"
            build_t0 = time.perf_counter()
            yaml_path, build_log = v6.build_yolo_dataset(labeled, dev_eval_df, dataset_dir)
            dataset_build_sec = time.perf_counter() - build_t0
            build_log.insert(0, "strategy", STRATEGY)
            build_log.insert(1, "acquisition_seed", seed)
            build_log.insert(2, "training_seed", training_seed)
            build_log.insert(3, "round", round_idx)
            build_log["dataset_build_sec"] = dataset_build_sec
            build_logs.append(build_log)

            print(f"[seed {seed}] round {round_idx}/{rounds}: training/evaluating")
            result, runtime = train_eval(yaml_path, save_dir, STRATEGY, seed, training_seed, round_idx)
            runtime["dataset_build_sec"] = dataset_build_sec
            runtime["total_sec"] = dataset_build_sec + float(runtime.get("train_eval_sec", 0.0))
            runtime_rows.append(runtime)
            row = {
                "result_origin": "trained_v9b_5seed_full_curve",
                "acquisition_seed": seed,
                "training_seed": training_seed,
                "strategy": STRATEGY,
                "round": round_idx,
                "labeled_budget": len(labeled),
                "development_eval_size": len(dev_eval_df),
                "yaml_path": str(yaml_path),
                **result,
                "retry_count": 0,
            }
            results_rows.append(row)
            per_class_rows.append(extract_per_class_metrics(result, STRATEGY, seed, training_seed, round_idx))
            if result.get("train_status") == "failed":
                failed_rows.append(row)
            append_run_registry(save_dir / "experiment_registry.csv", row, config)

            pd.DataFrame(results_rows).to_csv(save_dir / "all_round_results.csv", index=False, encoding="utf-8-sig")

            if dry_run:
                print(f"[DRY RUN] seed {seed}: stopping after round {round_idx}; no V9b detector weights were trained.")
                break
            train_dir = Path(str(result.get("train_run_dir", "")))
            next_weights = train_dir / "weights" / "best.pt"
            if not next_weights.exists():
                raise FileNotFoundError(f"Seed {seed} round {round_idx}: missing trained best.pt: {next_weights}")
            prev_weights = next_weights

    selected_df = pd.concat(selected_logs, ignore_index=True, sort=False)
    cumulative_df = pd.concat(cumulative_logs, ignore_index=True, sort=False)
    results_df = pd.DataFrame(results_rows)
    baseline_df = copy_v8_baseline_rows(source_run, seeds)
    combined_df = pd.concat([baseline_df, results_df], ignore_index=True, sort=False)

    selected_df.to_csv(save_dir / "all_selected_samples_by_round.csv", index=False, encoding="utf-8-sig")
    cumulative_df.to_csv(save_dir / "cumulative_labeled_sets_by_round.csv", index=False, encoding="utf-8-sig")
    results_df.to_csv(save_dir / "all_round_results.csv", index=False, encoding="utf-8-sig")
    combined_df.to_csv(save_dir / "v9b_results_with_v8_baselines.csv", index=False, encoding="utf-8-sig")
    dataset_distribution_by_round(cumulative_df).to_csv(save_dir / "dataset_distribution_by_round.csv", index=False, encoding="utf-8-sig")
    actual_stats_df, actual_class_df = v7.actual_stats_by_round(cumulative_df)
    actual_stats_df.to_csv(save_dir / "actual_instance_statistics_by_round.csv", index=False, encoding="utf-8-sig")
    actual_class_df.to_csv(save_dir / "actual_class_distribution_by_round.csv", index=False, encoding="utf-8-sig")
    pd.concat(build_logs, ignore_index=True).to_csv(save_dir / "dataset_build_log.csv", index=False, encoding="utf-8-sig") if build_logs else pd.DataFrame().to_csv(save_dir / "dataset_build_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runtime_rows).to_csv(save_dir / "runtime_profile.csv", index=False, encoding="utf-8-sig")
    pd.concat(per_class_rows, ignore_index=True).to_csv(save_dir / "per_class_metrics_by_round.csv", index=False, encoding="utf-8-sig") if per_class_rows else pd.DataFrame().to_csv(save_dir / "per_class_metrics_by_round.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failed_rows).to_csv(save_dir / "failed_or_retried_runs.csv", index=False, encoding="utf-8-sig")

    seed_summary = v7.seed_strategy_summary(combined_df)
    aggregate_df = v7.aggregate_summary(seed_summary)
    seed_summary.to_csv(save_dir / "seed_strategy_metric_summary.csv", index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(save_dir / "aggregate_strategy_metric_summary.csv", index=False, encoding="utf-8-sig")
    try:
        v7.paired_strategy_comparisons(seed_summary).to_csv(save_dir / "paired_strategy_comparisons.csv", index=False, encoding="utf-8-sig")
        v7.paired_roundwise_differences(combined_df).to_csv(save_dir / "paired_roundwise_differences.csv", index=False, encoding="utf-8-sig")
    except Exception:
        # Non-essential comparison artifacts should not invalidate a completed training run.
        pass

    write_summary(
        save_dir,
        config=config,
        results_df=results_df,
        baseline_df=baseline_df,
        seed_summary=seed_summary,
        aggregate_df=aggregate_df,
        actual_stats_df=actual_stats_df,
    )

    print("=" * 100)
    print("[DONE] V9b 5-seed pilot full curve finished")
    print(f"Output dir: {save_dir}")
    print("Final test used: False")
    print("=" * 100)


if __name__ == "__main__":
    main()

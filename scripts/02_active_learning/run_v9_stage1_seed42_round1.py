"""V9 Stage 1: seed42 round-1 detector-aware smoke training.

This runner is intentionally narrow.  It checks whether the V9 detector-aware
selection that passed selection-only sanity can produce a reasonable YOLO model
at budget 20 for acquisition seed 42.

Data protocol:
    - Acquisition pool: NEU-DET only, same 50-image pool as V8.
    - Initial labeled set: same V8 seed42 round-0 15 images.
    - Query batch: 5 images selected GT-free by DetectorUncertaintyDINOBalanced.
    - Training set: 20 labeled images.
    - Evaluation set: development_eval_v7 filtered to NEU-DET, 177 images.
    - Final test: locked / not used.

Default behavior is dry-run.  Set AL_DRY_RUN_ONLY=0 to actually train YOLO.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Keep the protocol aligned with the V8 NEU-only run before importing helpers.
os.environ.setdefault("AL_POOL_DATASET_FILTER", "NEU-DET")
os.environ.setdefault("AL_DEV_EVAL_DATASET_FILTER", "NEU-DET")
os.environ.setdefault("AL_EVAL_SPLIT", "development_eval_v7")
os.environ.setdefault("AL_EMBEDDING_BACKEND", "dinov2")
os.environ.setdefault("AL_ALLOW_MODEL_DOWNLOAD", "0")

import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402
import run_al_yolo_ablation_v7_full_curve as v7  # noqa: E402
import probe_v9_detector_aware_selection as probe  # noqa: E402
from audit_detection_pipeline_v7 import build_image_index, build_xml_index, parse_bool_env  # noqa: E402
from run_al_yolo_ablation_v7_full_curve import (  # noqa: E402
    append_run_registry,
    extract_per_class_metrics,
    file_sha256,
    git_commit,
    git_dirty,
    load_development_eval,
    load_dino_embeddings,
    load_priority_pool_with_identity,
    train_eval,
)

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    raise ImportError("Install ultralytics before running V9 Stage 1.") from exc


PROJECT_ROOT = v6.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "active_learning_ablation_v9_detector_aware"
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "active_learning_ablation_v9_detector_aware"
DEFAULT_SOURCE_RUN = probe.DEFAULT_SOURCE_RUN
V9_STRATEGY = os.environ.get("AL_V9_STRATEGY", "DetectorUncertaintyDINOBalanced")


def set_front_metadata(df: pd.DataFrame, values: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col in values:
        if col in out.columns:
            out = out.drop(columns=[col])
    for idx, (col, value) in enumerate(values.items()):
        out.insert(idx, col, value)
    return out


def resolve_source_run() -> Path:
    override = os.environ.get("AL_V9_SOURCE_RUN_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    return DEFAULT_SOURCE_RUN


def copy_v8_baseline_rows(source_run: Path, acquisition_seed: int, round_idx: int) -> pd.DataFrame:
    path = source_run / "all_round_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    keep = df[
        pd.to_numeric(df["acquisition_seed"], errors="coerce").eq(acquisition_seed)
        & pd.to_numeric(df["round"], errors="coerce").eq(round_idx)
        & df["strategy"].astype(str).isin(
            [
                "GTFreeRandom",
                "GTFreeDatasetBalancedConsistency",
                "GTFreeDatasetBalancedVisualDiversity",
            ]
        )
    ].copy()
    keep["result_origin"] = "copied_from_v8_neu_only"
    return keep


def float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def v9b_weight_config() -> dict[str, float]:
    """Stage 1b defaults: prefer learnable, instance-rich samples over hardest samples."""
    weights = {
        "detector_uncertainty": float_env("AL_V9B_W_UNCERTAINTY", 0.20),
        "dino_visual_distance": float_env("AL_V9B_W_DINO", 0.25),
        "predicted_class_deficit": float_env("AL_V9B_W_BALANCE", 0.15),
        "pseudo_instance_count": float_env("AL_V9B_W_INSTANCE", 0.40),
    }
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0:
        raise ValueError(f"Invalid V9B weights: {weights}")
    return {k: max(0.0, v) / total for k, v in weights.items()}


def select_instance_rich_dino_balanced(
    scored_pool: pd.DataFrame,
    initial_df: pd.DataFrame,
    embedding_lookup: dict[str, np.ndarray],
    *,
    query_size: int,
    candidate_fraction: float,
    weights: dict[str, float],
    max_no_box: int,
    min_pseudo_boxes: int,
    max_per_pred_class: int,
) -> pd.DataFrame:
    """GT-free Stage 1b selector.

    Differences from the first V9 selector:
      - does not start from top-uncertainty-only samples unless requested;
      - removes no-box samples when enough detected samples exist;
      - prefers pseudo-instance-rich samples;
      - keeps predicted-class balance as a weak constraint/tie signal.
    """
    candidate_fraction = min(max(float(candidate_fraction), 0.0), 1.0)
    n_candidates = len(scored_pool) if candidate_fraction >= 1.0 else max(query_size, int(np.ceil(len(scored_pool) * candidate_fraction)))
    candidates = scored_pool.copy()
    if "detector_no_box" not in candidates.columns:
        candidates["detector_no_box"] = candidates["detector_pred_class"].astype(str).eq("__no_box__").astype(float)
    candidates["detector_pseudo_box_count"] = pd.to_numeric(
        candidates["detector_pseudo_box_count"], errors="coerce"
    ).fillna(0)
    candidates["detector_max_conf"] = pd.to_numeric(candidates["detector_max_conf"], errors="coerce").fillna(0.0)
    candidates["_pre_instance_log"] = np.log1p(candidates["detector_pseudo_box_count"].clip(lower=0))
    candidates["_pre_usable_uncertainty"] = pd.to_numeric(
        candidates["detector_uncertainty"], errors="coerce"
    ).fillna(0.0) * (1.0 - pd.to_numeric(candidates["detector_no_box"], errors="coerce").fillna(0.0))
    candidates["_pre_score"] = (
        0.65 * probe.normalize(candidates["_pre_instance_log"])
        + 0.25 * probe.normalize(candidates["_pre_usable_uncertainty"])
        + 0.10 * probe.normalize(candidates["detector_max_conf"])
    )
    candidates = candidates.sort_values(["_pre_score", "sample_id"], ascending=[False, True], kind="mergesort").head(n_candidates).copy()

    non_no_box = candidates[candidates["detector_pred_class"].astype(str).ne("__no_box__")].copy()
    if len(non_no_box) >= query_size:
        candidates = non_no_box

    enough_min_box = candidates[candidates["detector_pseudo_box_count"].ge(min_pseudo_boxes)].copy()
    if len(enough_min_box) >= query_size:
        candidates = enough_min_box

    class_universe = sorted(str(v) for v in getattr(v6, "NEU_CLASSES", []) if str(v))
    if not class_universe:
        class_universe = sorted(
            c
            for c in scored_pool["detector_pred_class"].dropna().astype(str).unique().tolist()
            if c != "__no_box__"
        )
    labeled_counts = Counter(initial_df["class_hint"].dropna().astype(str))
    selected_pred_counts: Counter = Counter()
    selected_parts: list[pd.DataFrame] = []
    selected_ids: list[str] = []
    reference_ids = initial_df["sample_id"].astype(str).tolist()
    remaining = candidates.copy()
    no_box_selected = 0

    for _ in range(min(query_size, len(remaining))):
        refs = reference_ids + selected_ids
        remaining = remaining.copy()
        remaining["_visual_distance_raw"] = [
            probe.min_cosine_distance(str(sid), refs, embedding_lookup)
            for sid in remaining["sample_id"].astype(str)
        ]
        remaining["_balance_deficit_raw"] = [
            probe.class_deficit_score(str(cls), labeled_counts, selected_pred_counts, class_universe)
            for cls in remaining["detector_pred_class"].astype(str)
        ]
        remaining["_instance_log_raw"] = np.log1p(
            pd.to_numeric(remaining["detector_pseudo_box_count"], errors="coerce").fillna(0).clip(lower=0)
        )
        remaining["_usable_uncertainty_raw"] = pd.to_numeric(
            remaining["detector_uncertainty"], errors="coerce"
        ).fillna(0.0) * (
            1.0 - pd.to_numeric(remaining["detector_no_box"], errors="coerce").fillna(0.0)
        )
        remaining["_detector_norm"] = probe.normalize(remaining["_usable_uncertainty_raw"])
        remaining["_visual_norm"] = probe.normalize(remaining["_visual_distance_raw"])
        remaining["_balance_norm"] = probe.normalize(remaining["_balance_deficit_raw"])
        remaining["_instance_norm"] = probe.normalize(remaining["_instance_log_raw"])
        remaining["_v9b_score"] = (
            weights["detector_uncertainty"] * remaining["_detector_norm"]
            + weights["dino_visual_distance"] * remaining["_visual_norm"]
            + weights["predicted_class_deficit"] * remaining["_balance_norm"]
            + weights["pseudo_instance_count"] * remaining["_instance_norm"]
        )

        feasible = remaining.copy()
        if no_box_selected >= max_no_box:
            feasible = feasible[feasible["detector_pred_class"].astype(str).ne("__no_box__")]
        if max_per_pred_class > 0:
            feasible = feasible[
                feasible["detector_pred_class"].astype(str).map(lambda c: selected_pred_counts.get(c, 0) < max_per_pred_class)
            ]
        if feasible.empty:
            feasible = remaining.copy()

        pick_idx = feasible.sort_values(
            ["_v9b_score", "_instance_log_raw", "_visual_distance_raw", "detector_max_conf", "sample_id"],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        ).index[0]
        picked = remaining.loc[[pick_idx]].copy()
        selected_parts.append(picked)
        selected_ids.append(str(picked.iloc[0]["sample_id"]))
        pred_cls = str(picked.iloc[0]["detector_pred_class"])
        if pred_cls == "__no_box__":
            no_box_selected += 1
        else:
            selected_pred_counts[pred_cls] += 1
        remaining = remaining.drop(index=pick_idx)

    return pd.concat(selected_parts, ignore_index=True) if selected_parts else scored_pool.iloc[0:0].copy()


def select_v9_round1_batch(
    *,
    source_run: Path,
    acquisition_seed: int,
    query_size: int,
    candidate_fraction: float,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    selector: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    priority_csv, full_pool, _ = load_priority_pool_with_identity()
    embedding_dir, _, _, embedding_lookup, dino_config = load_dino_embeddings(full_pool)
    initial_df, _ = probe.load_source_initial_and_baselines(source_run, acquisition_seed)
    current_pool = full_pool[~full_pool["sample_id"].isin(initial_df["sample_id"])].copy().reset_index(drop=True)
    if current_pool.empty:
        raise ValueError("Current unlabeled pool is empty after removing the initial set.")

    weights = probe.resolve_weights(source_run, acquisition_seed)
    if not weights.exists():
        raise FileNotFoundError(f"Missing V8 round-0 detector weights: {weights}")

    model = YOLO(str(weights))
    detector_scores = probe.prediction_rows(model, current_pool, device=device, imgsz=imgsz, conf=conf, iou=iou)
    scored_pool = current_pool.merge(detector_scores, on="sample_id", how="left")
    selector = selector.strip().lower()
    v9b_weights = v9b_weight_config()
    max_no_box = int_env("AL_V9B_MAX_NO_BOX", 0)
    min_pseudo_boxes = int_env("AL_V9B_MIN_PSEUDO_BOXES", 2)
    max_per_pred_class = int_env("AL_V9B_MAX_PER_PRED_CLASS", 2)
    if selector in {"instance_rich", "stage1b", "v9b"}:
        selected = select_instance_rich_dino_balanced(
            scored_pool,
            initial_df,
            embedding_lookup,
            query_size=query_size,
            candidate_fraction=candidate_fraction,
            weights=v9b_weights,
            max_no_box=max_no_box,
            min_pseudo_boxes=min_pseudo_boxes,
            max_per_pred_class=max_per_pred_class,
        )
    elif selector in {"balanced", "stage1", "v9"}:
        selected = probe.select_detector_uncertainty_dino_balanced(
            scored_pool,
            initial_df,
            embedding_lookup,
            query_size=query_size,
            candidate_fraction=candidate_fraction,
        )
    else:
        raise ValueError(f"Unknown AL_V9_SELECTOR={selector!r}; use balanced or instance_rich.")
    selected = set_front_metadata(
        selected.reset_index(drop=True),
        {
            "acquisition_seed": acquisition_seed,
            "strategy": V9_STRATEGY,
            "round": 1,
            "selection_type": V9_STRATEGY,
        },
    )
    selected.insert(4, "rank_in_selection", range(1, len(selected) + 1))

    initial_for_plan = set_front_metadata(
        initial_df.reset_index(drop=True),
        {
            "acquisition_seed": acquisition_seed,
            "strategy": V9_STRATEGY,
            "round": 0,
            "selection_type": "shared_initial_seed_random",
        },
    )
    if "rank_in_selection" in initial_for_plan.columns:
        initial_for_plan = initial_for_plan.drop(columns=["rank_in_selection"])
    initial_for_plan.insert(4, "rank_in_selection", range(1, len(initial_for_plan) + 1))

    labeled_round1 = pd.concat([initial_df, selected], ignore_index=True, sort=False)
    labeled_round1 = labeled_round1.drop_duplicates("sample_id", keep="first").reset_index(drop=True)
    if len(labeled_round1) != len(initial_df) + min(query_size, len(current_pool)):
        raise ValueError(f"Unexpected round-1 labeled size: {len(labeled_round1)}")

    metadata = {
        "priority_csv": str(priority_csv),
        "priority_csv_sha256": file_sha256(priority_csv),
        "embedding_dir": str(embedding_dir),
        "DINO_manifest_sha256": file_sha256(embedding_dir / "embedding_manifest.csv"),
        "DINO_config": dino_config,
        "detector_weights": str(weights),
        "pool_size_after_filter": int(len(full_pool)),
        "initial_size": int(len(initial_df)),
        "unlabeled_size_before_round1": int(len(current_pool)),
        "query_size": int(query_size),
        "candidate_fraction": float(candidate_fraction),
        "predict_conf": float(conf),
        "predict_iou": float(iou),
        "selector": selector,
        "v9b_constraints": {
            "max_no_box": int(max_no_box),
            "min_pseudo_boxes": int(min_pseudo_boxes),
            "max_per_pred_class": int(max_per_pred_class),
        },
    }
    score_config = {
        "fixed_balanced_weights": {
            "detector_uncertainty": 0.35,
            "dino_visual_distance": 0.35,
            "predicted_class_deficit": 0.20,
            "pseudo_instance_count": 0.10,
        },
        "stage1b_instance_rich_weights": v9b_weights,
    }
    return initial_for_plan, selected, labeled_round1, metadata, score_config


def posthoc_stats(selected_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_index = build_image_index()
    xml_index = build_xml_index()
    return probe.posthoc_stats(selected_df, image_index, xml_index)


def write_stage1_summary(
    save_dir: Path,
    *,
    config: dict[str, Any],
    selected: pd.DataFrame,
    actual_stats: pd.DataFrame,
    results_df: pd.DataFrame,
) -> None:
    selected_cols = [
        "image_name",
        "class_hint",
        "detector_pred_class",
        "detector_pseudo_box_count",
        "detector_uncertainty",
        "detector_max_conf",
        "pseudo_instance_score",
    ]
    result_cols = [
        "result_origin",
        "strategy",
        "round",
        "labeled_budget",
        "map50",
        "map5095",
        "precision",
        "recall",
        "train_status",
        "train_run_dir",
    ]
    lines = [
        "# V9 Stage 1 seed42 round-1 smoke training",
        "",
        "This run trains only the V9 detector-aware budget-20 set and compares it with copied V8 seed42 round-1 baselines.",
        "",
        "Final test is locked and was not used.",
        "",
        "## Data protocol",
        "",
        f"- Acquisition pool: NEU-DET only, {config['pool_size_after_filter']} images",
        f"- Selector: {config.get('selector')}",
        f"- Initial labeled set: {config['initial_size']} images from V8 seed42 round 0",
        f"- V9 query batch: {config['query_size']} images",
        f"- Training budget: {config['labeled_budget']} images",
        f"- Development evaluation: NEU-DET only, {config['development_eval_size_after_filter']} images",
        "- Final test: locked / unused",
        f"- Stage 1b weights: `{json.dumps(config.get('stage1b_instance_rich_weights', {}), ensure_ascii=False)}`",
        f"- Stage 1b constraints: `{json.dumps(config.get('v9b_constraints', {}), ensure_ascii=False)}`",
        "",
        "## V9 selected query batch",
        "",
        probe.table_md(selected[[c for c in selected_cols if c in selected.columns]]),
        "",
        "## Post-hoc actual XML statistics",
        "",
        probe.table_md(actual_stats),
        "",
        "## Round-1 result comparison",
        "",
        probe.table_md(results_df[[c for c in result_cols if c in results_df.columns]]),
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, indent=2, ensure_ascii=False, default=str),
        "```",
    ]
    (save_dir / "v9_stage1_seed42_round1_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"v9_stage1_seed42_round1_{timestamp}"
    dataset_root = DATASETS_ROOT / save_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    acquisition_seed = int(os.environ.get("AL_ACQUISITION_SEED", "42"))
    if acquisition_seed != 42:
        raise ValueError("Stage 1 is intentionally fixed to acquisition seed 42. Set AL_ACQUISITION_SEED=42.")
    training_seed = int(os.environ.get("AL_TRAINING_SEED", str(1000 + acquisition_seed)))
    query_size = int(os.environ.get("AL_QUERY_SIZE", "5"))
    selector = os.environ.get("AL_V9_SELECTOR", "balanced").strip().lower()
    default_candidate_fraction = "1.00" if selector in {"instance_rich", "stage1b", "v9b"} else "0.30"
    candidate_fraction = float(os.environ.get("AL_V9_CANDIDATE_FRACTION", default_candidate_fraction))
    imgsz = int(os.environ.get("AL_IMGSZ", "640"))
    device = os.environ.get("AL_YOLO_DEVICE", "0")
    conf = float(os.environ.get("AL_V9_PREDICT_CONF", "0.05"))
    iou = float(os.environ.get("AL_V9_PREDICT_IOU", "0.70"))
    dry_run = parse_bool_env("AL_DRY_RUN_ONLY", True)
    source_run = resolve_source_run()

    protocol_dir, dev_eval_path, dev_eval_df = load_development_eval()
    initial_plan, selected, labeled_round1, selection_meta, score_config = select_v9_round1_batch(
        source_run=source_run,
        acquisition_seed=acquisition_seed,
        query_size=query_size,
        candidate_fraction=candidate_fraction,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        selector=selector,
    )

    selected.to_csv(save_dir / "v9_round1_selected_samples.csv", index=False, encoding="utf-8-sig")
    pd.concat([initial_plan, selected], ignore_index=True, sort=False).to_csv(
        save_dir / "v9_round1_selection_plan.csv", index=False, encoding="utf-8-sig"
    )
    labeled_round1.to_csv(save_dir / "v9_round1_labeled_set.csv", index=False, encoding="utf-8-sig")
    _, actual_stats = posthoc_stats(probe.add_source_strategy(labeled_round1, V9_STRATEGY))
    actual_stats.to_csv(save_dir / "v9_round1_actual_instance_stats.csv", index=False, encoding="utf-8-sig")

    config = {
        "experiment_id": save_dir.name,
        "experiment_label": "V9 detector-aware Stage 1 seed42 round-1 smoke training",
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "save_dir": str(save_dir),
        "dataset_root": str(dataset_root),
        "source_run": str(source_run),
        "final_test_used": False,
        "stage": "stage1_seed42_round1_only",
        "selector": selector,
        "strategy": V9_STRATEGY,
        "acquisition_seed": acquisition_seed,
        "training_seed": training_seed,
        "round": 1,
        "initial_seed_size": int(selection_meta["initial_size"]),
        "query_size": query_size,
        "labeled_budget": int(len(labeled_round1)),
        "pool_dataset_filter": ["NEU-DET"],
        "pool_size_after_filter": int(selection_meta["pool_size_after_filter"]),
        "development_eval_path": str(dev_eval_path),
        "development_eval_sha256": file_sha256(dev_eval_path),
        "development_eval_dataset_filter": ["NEU-DET"],
        "development_eval_size_after_filter": int(len(dev_eval_df)),
        "eval_protocol_dir": str(protocol_dir),
        "model": os.environ.get("AL_YOLO_MODEL_NAME", "yolov8n.pt"),
        "epochs": int(os.environ.get("AL_EPOCHS_PER_ROUND", "100")),
        "patience": int(os.environ.get("AL_YOLO_PATIENCE", os.environ.get("AL_EPOCHS_PER_ROUND", "100"))),
        "batch": int(os.environ.get("AL_BATCH_SIZE", "8")),
        "workers": int(os.environ.get("AL_WORKERS", "4")),
        "cache": os.environ.get("AL_YOLO_CACHE", "ram"),
        "dry_run": dry_run,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        **selection_meta,
        **score_config,
    }
    (save_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("=" * 100)
    print("[V9 Stage 1] seed42 round-1 smoke training")
    print(f"Output dir : {save_dir}")
    print(f"Pool       : NEU-DET {config['pool_size_after_filter']} images")
    print(f"Selector   : {selector}")
    print(f"Train      : {len(labeled_round1)} labeled images")
    print(f"Dev eval   : NEU-DET {len(dev_eval_df)} images")
    print(f"Dry run    : {dry_run}")
    print("Final test : LOCKED / NOT USED")
    print("=" * 100)

    build_t0 = time.perf_counter()
    yaml_path, build_log = v6.build_yolo_dataset(labeled_round1, dev_eval_df, dataset_root / "seed_42" / V9_STRATEGY / "round_1")
    dataset_build_sec = time.perf_counter() - build_t0
    build_log.insert(0, "strategy", V9_STRATEGY)
    build_log.insert(1, "acquisition_seed", acquisition_seed)
    build_log.insert(2, "training_seed", training_seed)
    build_log.insert(3, "round", 1)
    build_log["dataset_build_sec"] = dataset_build_sec
    build_log.to_csv(save_dir / "dataset_build_log.csv", index=False, encoding="utf-8-sig")

    result, runtime = train_eval(yaml_path, save_dir, V9_STRATEGY, acquisition_seed, training_seed, 1)
    runtime["dataset_build_sec"] = dataset_build_sec
    runtime["total_sec"] = dataset_build_sec + float(runtime.get("train_eval_sec", 0.0))
    pd.DataFrame([runtime]).to_csv(save_dir / "runtime_profile.csv", index=False, encoding="utf-8-sig")

    v9_row = {
        "result_origin": "trained_v9_stage1",
        "acquisition_seed": acquisition_seed,
        "training_seed": training_seed,
        "strategy": V9_STRATEGY,
        "round": 1,
        "labeled_budget": len(labeled_round1),
        "development_eval_size": len(dev_eval_df),
        "yaml_path": str(yaml_path),
        **result,
        "retry_count": 0,
    }
    baseline_rows = copy_v8_baseline_rows(source_run, acquisition_seed, 1)
    results_df = pd.concat([baseline_rows, pd.DataFrame([v9_row])], ignore_index=True, sort=False)
    results_df.to_csv(save_dir / "stage1_round1_results_with_v8_baselines.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([v9_row]).to_csv(save_dir / "all_round_results.csv", index=False, encoding="utf-8-sig")
    extract_per_class_metrics(result, V9_STRATEGY, acquisition_seed, training_seed, 1).to_csv(
        save_dir / "per_class_metrics_by_round.csv", index=False, encoding="utf-8-sig"
    )
    append_run_registry(save_dir / "experiment_registry.csv", v9_row, config)
    write_stage1_summary(save_dir, config=config, selected=selected, actual_stats=actual_stats, results_df=results_df)

    print("=" * 100)
    print("[DONE] V9 Stage 1 seed42 round-1 runner finished")
    print(f"Output dir: {save_dir}")
    print(f"mAP50={v9_row.get('map50')} mAP50-95={v9_row.get('map5095')} status={v9_row.get('train_status')}")
    print("Final test used: False")
    print("=" * 100)


if __name__ == "__main__":
    main()

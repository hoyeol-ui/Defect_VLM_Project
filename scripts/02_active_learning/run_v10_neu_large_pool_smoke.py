"""V10 NEU large-pool one-cycle smoke test.

Purpose:
    Test whether the V9b instance-rich detector-aware selector still makes
    sense when the acquisition pool is large enough for Random to be less
    accidentally optimal.

Protocol:
    - Dataset: NEU-DET only, all 1,800 local images.
    - Stratified protocol split by class_hint:
        * acquisition pool: 900 images, 150/class
        * development eval: 300 images, 50/class
        * final test: 300 images, 50/class, saved but LOCKED/UNUSED
        * unused reserve: remaining 300 images, 50/class
    - Seed: 42 by default.
    - Initial labeled set: 60 random images from acquisition pool.
    - Query size: 30.
    - Rounds: 1 smoke cycle only.
    - Compared strategies:
        * GTFreeRandom
        * DetectorInstanceRichDINOBalanced
    - YOLO trainings:
        * shared round0 initial-60 detector
        * Random round1 budget-90 detector
        * V9b round1 budget-90 detector

Final test is never evaluated by this runner.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("AL_EMBEDDING_BACKEND", "dinov2")
os.environ.setdefault("AL_ALLOW_MODEL_DOWNLOAD", "0")
os.environ.setdefault("AL_V9_STRATEGY", "DetectorInstanceRichDINOBalanced")

import build_visual_embedding_cache_v7 as embcache  # noqa: E402
import probe_v9_detector_aware_selection as probe  # noqa: E402
import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402
import run_al_yolo_ablation_v7_full_curve as v7  # noqa: E402
import run_v9_stage1_seed42_round1 as stage1  # noqa: E402
import run_yolo_upper_bound_v7 as upper  # noqa: E402
from audit_detection_pipeline_v7 import (  # noqa: E402
    build_image_index,
    compute_file_sha256,
    parse_bool_env,
)
from canonical_sampling_v7 import canonicalize_pool_for_sampling, sample_initial_labeled_set  # noqa: E402
from run_al_yolo_ablation_v7_full_curve import (  # noqa: E402
    extract_per_class_metrics,
    file_sha256,
    git_commit,
    git_dirty,
    train_eval,
)


PROJECT_ROOT = v6.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "active_learning_ablation_v10_neu_large_pool"
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "active_learning_ablation_v10_neu_large_pool"
EMBEDDING_ROOT = PROJECT_ROOT / "outputs" / "visual_embeddings_v10"

RANDOM_STRATEGY = "GTFreeRandom"
V9B_STRATEGY = os.environ.get("AL_V9_STRATEGY", "DetectorInstanceRichDINOBalanced")
STRATEGIES = [RANDOM_STRATEGY, V9B_STRATEGY]


def table_md(df: pd.DataFrame) -> str:
    return probe.table_md(df) if len(df) else "_No rows._"


def ensure_neu_manifest() -> pd.DataFrame:
    raw = upper.enumerate_dataset_manifest("NEU-DET")
    if raw.empty:
        raise FileNotFoundError("No NEU-DET images found under data/NEU-DET.")
    image_index = build_image_index()
    manifest = canonicalize_pool_for_sampling(raw, image_index, set_sample_id=True)
    return manifest


def stratified_take(sub: pd.DataFrame, n: int, seed: int, offset: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(sub) < n:
        raise ValueError(f"Not enough rows in class group: need {n}, got {len(sub)}")
    picked = sub.sample(n=n, random_state=seed + offset, replace=False)
    rest = sub.drop(index=picked.index)
    return picked, rest


def build_v10_protocol_split(
    manifest: pd.DataFrame,
    *,
    seed: int,
    pool_per_class: int,
    dev_per_class: int,
    final_per_class: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool_parts = []
    dev_parts = []
    final_parts = []
    unused_parts = []
    for i, (cls, sub) in enumerate(manifest.groupby("class_hint", sort=True, dropna=False)):
        sub = sub.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
        pool, rest = stratified_take(sub, pool_per_class, seed, 1000 + i * 17)
        dev, rest = stratified_take(rest, dev_per_class, seed, 2000 + i * 17)
        final, rest = stratified_take(rest, final_per_class, seed, 3000 + i * 17)
        pool_parts.append(pool)
        dev_parts.append(dev)
        final_parts.append(final)
        unused_parts.append(rest)
    return (
        pd.concat(pool_parts, ignore_index=True).sort_values("sample_id", kind="mergesort").reset_index(drop=True),
        pd.concat(dev_parts, ignore_index=True).sort_values("sample_id", kind="mergesort").reset_index(drop=True),
        pd.concat(final_parts, ignore_index=True).sort_values("sample_id", kind="mergesort").reset_index(drop=True),
        pd.concat(unused_parts, ignore_index=True).sort_values("sample_id", kind="mergesort").reset_index(drop=True),
    )


def manifest_for_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        image_path = Path(str(row["resolved_image_path"]))
        rows.append(
            {
                "embedding_index": i,
                "source_row_index": i,
                "dataset_type": row["dataset_type"],
                "image_name": row["image_name"],
                "image_path": str(image_path),
                "sha256": row["image_sha256"],
                "ahash": embcache.average_hash(image_path) if hasattr(embcache, "average_hash") else "",
            }
        )
    return pd.DataFrame(rows)


def cache_manifest_matches(existing_manifest: pd.DataFrame, current_manifest: pd.DataFrame) -> bool:
    cols = ["dataset_type", "image_name", "sha256"]
    if any(c not in existing_manifest.columns for c in cols):
        return False
    left = existing_manifest[cols].astype(str).sort_values(cols).reset_index(drop=True)
    right = current_manifest[cols].astype(str).sort_values(cols).reset_index(drop=True)
    return left.equals(right)


def find_v10_embedding_cache(backend: str, current_manifest: pd.DataFrame, model_id: str) -> Path | None:
    if parse_bool_env("AL_FORCE_REBUILD_EMBEDDINGS", False):
        return None
    if not parse_bool_env("AL_REUSE_EXISTING_EMBEDDING_CACHE", True):
        return None
    if not EMBEDDING_ROOT.exists():
        return None
    for cache_dir in sorted(EMBEDDING_ROOT.glob(f"{backend}_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        config_path = cache_dir / "embedding_config.json"
        manifest_path = cache_dir / "embedding_manifest.csv"
        embeddings_path = cache_dir / "embeddings.npy"
        if not (config_path.exists() and manifest_path.exists() and embeddings_path.exists()):
            continue
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            if cfg.get("status") != "success" or str(cfg.get("backend")).lower() != backend:
                continue
            if backend == "dinov2" and cfg.get("model_id") != model_id:
                continue
            existing = pd.read_csv(manifest_path)
            if len(existing) == len(current_manifest) and cache_manifest_matches(existing, current_manifest):
                return cache_dir
        except Exception:
            continue
    return None


def load_or_build_embeddings(pool_df: pd.DataFrame, save_dir: Path) -> tuple[Path, dict[str, np.ndarray], dict[str, Any]]:
    backend = os.environ.get("AL_EMBEDDING_BACKEND", "dinov2").strip().lower()
    model_id = os.environ.get("AL_DINOV2_MODEL_ID", embcache.DEFAULT_DINOV2_MODEL_ID).strip() or embcache.DEFAULT_DINOV2_MODEL_ID
    current_manifest = manifest_for_embeddings(pool_df)
    reusable = find_v10_embedding_cache(backend, current_manifest, model_id)
    if reusable is not None:
        cache_dir = reusable
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_dir = EMBEDDING_ROOT / f"{backend}_{timestamp}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        if backend == "handcrafted":
            embeddings, built_manifest = embcache.build_handcrafted(current_manifest)
            extra = {"embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] else 0}
        elif backend == "dinov2":
            embeddings, built_manifest, extra = embcache.maybe_build_dinov2(current_manifest)
        else:
            raise ValueError(f"Unsupported AL_EMBEDDING_BACKEND={backend!r}")
        np.save(cache_dir / "embeddings.npy", embeddings)
        built_manifest.to_csv(cache_dir / "embedding_manifest.csv", index=False, encoding="utf-8-sig")
        cfg = {
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "backend": backend,
            "status": "success",
            "uses_gt_labels": False,
            "uses_class_hint": False,
            "uses_xml_bbox": False,
            "num_manifest_images": int(len(built_manifest)),
            "allow_model_download": parse_bool_env("AL_ALLOW_MODEL_DOWNLOAD", False),
            "reuse_existing_embedding_cache": parse_bool_env("AL_REUSE_EXISTING_EMBEDDING_CACHE", True),
            "force_rebuild_embeddings": parse_bool_env("AL_FORCE_REBUILD_EMBEDDINGS", False),
            "runtime_sec": time.perf_counter() - t0,
            **extra,
        }
        (cache_dir / "embedding_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"stage": "embedding_build", "backend": backend, "num_embeddings": len(built_manifest), "runtime_sec": cfg["runtime_sec"], "status": "success"}]).to_csv(
            cache_dir / "embedding_build_runtime.csv",
            index=False,
            encoding="utf-8-sig",
        )

    manifest = pd.read_csv(cache_dir / "embedding_manifest.csv")
    embeddings = np.load(cache_dir / "embeddings.npy")
    if len(manifest) != len(pool_df) or embeddings.shape[0] != len(pool_df):
        raise ValueError(f"Embedding cache shape mismatch: manifest={len(manifest)} embeddings={embeddings.shape} pool={len(pool_df)}")
    lookup = {
        str(pool_df.iloc[i]["sample_id"]): embeddings[int(manifest.iloc[i]["embedding_index"])]
        for i in range(len(pool_df))
    }
    cfg_path = cache_dir / "embedding_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {"backend": backend}
    (save_dir / "embedding_cache_used.txt").write_text(str(cache_dir), encoding="utf-8")
    return cache_dir, lookup, cfg


def select_random(current_pool: pd.DataFrame, query_size: int, seed: int) -> pd.DataFrame:
    return current_pool.sample(n=min(query_size, len(current_pool)), random_state=seed + 101).copy().reset_index(drop=True)


def add_selection_metadata(
    df: pd.DataFrame,
    strategy: str,
    round_idx: int,
    selection_type: str,
    *,
    acquisition_seed: int | None = None,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    for col in ["acquisition_seed", "strategy", "round", "selection_type", "rank_in_selection"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    insert_at = 0
    if acquisition_seed is not None:
        out.insert(insert_at, "acquisition_seed", acquisition_seed)
        insert_at += 1
    out.insert(insert_at, "strategy", strategy)
    out.insert(insert_at + 1, "round", round_idx)
    out.insert(insert_at + 2, "selection_type", selection_type)
    out.insert(insert_at + 3, "rank_in_selection", range(1, len(out) + 1))
    return out


def train_row(
    *,
    yaml_path: Path,
    save_dir: Path,
    strategy: str,
    acquisition_seed: int,
    training_seed: int,
    round_idx: int,
    labeled_budget: int,
    dev_eval_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result, runtime = train_eval(yaml_path, save_dir, strategy, acquisition_seed, training_seed, round_idx)
    row = {
        "acquisition_seed": acquisition_seed,
        "training_seed": training_seed,
        "strategy": strategy,
        "round": round_idx,
        "labeled_budget": labeled_budget,
        "development_eval_size": dev_eval_size,
        "yaml_path": str(yaml_path),
        **result,
        "retry_count": 0,
    }
    return row, runtime


def write_summary(save_dir: Path, config: dict[str, Any], results: pd.DataFrame, actual_stats: pd.DataFrame) -> None:
    cols = ["strategy", "round", "labeled_budget", "map50", "map5095", "precision", "recall", "train_status", "train_run_dir"]
    lines = [
        "# V10 NEU large-pool smoke",
        "",
        "One-cycle large-pool smoke test. Final test split is saved but not evaluated.",
        "",
        "## Protocol",
        "",
        f"- NEU manifest size: {config['neu_manifest_size']}",
        f"- Acquisition pool: {config['pool_size']} images",
        f"- Development eval: {config['development_eval_size']} images",
        f"- Final test: {config['final_test_size']} images, locked / unused",
        f"- Initial budget: {config['initial_seed_size']}",
        f"- Query size: {config['query_size']}",
        f"- Strategies: {config['strategies']}",
        f"- Expected trainings: {config['expected_trainings']}",
        "",
        "## Results",
        "",
        table_md(results[[c for c in cols if c in results.columns]]),
        "",
        "## Post-hoc actual instance statistics",
        "",
        table_md(actual_stats),
        "",
        "## Reading guide",
        "",
        "- This is not a final claim; it tests whether large-pool data geometry changes the Random-vs-V9b picture.",
        "- Pass signal: V9b is comparable to Random at budget 90 while selecting richer/balanced samples.",
        "- Final test remains locked until the method and protocol are frozen.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        "```",
    ]
    (save_dir / "v10_neu_large_pool_smoke_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"v10_neu_large_pool_smoke_{timestamp}"
    dataset_root = DATASETS_ROOT / save_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    acquisition_seed = int(os.environ.get("AL_ACQUISITION_SEED", "42"))
    training_seed = int(os.environ.get("AL_TRAINING_SEED", str(1000 + acquisition_seed)))
    pool_per_class = int(os.environ.get("AL_V10_POOL_PER_CLASS", "150"))
    dev_per_class = int(os.environ.get("AL_V10_DEV_PER_CLASS", "50"))
    final_per_class = int(os.environ.get("AL_V10_FINAL_PER_CLASS", "50"))
    initial_size = int(os.environ.get("AL_INITIAL_SEED_SIZE", "60"))
    query_size = int(os.environ.get("AL_QUERY_SIZE", "30"))
    imgsz = int(os.environ.get("AL_IMGSZ", "640"))
    device = os.environ.get("AL_YOLO_DEVICE", "0")
    conf = float(os.environ.get("AL_V9_PREDICT_CONF", "0.05"))
    iou = float(os.environ.get("AL_V9_PREDICT_IOU", "0.70"))
    candidate_fraction = float(os.environ.get("AL_V9_CANDIDATE_FRACTION", "1.00"))
    dry_run = parse_bool_env("AL_DRY_RUN_ONLY", True)

    manifest = ensure_neu_manifest()
    pool_df, dev_df, final_df, unused_df = build_v10_protocol_split(
        manifest,
        seed=acquisition_seed,
        pool_per_class=pool_per_class,
        dev_per_class=dev_per_class,
        final_per_class=final_per_class,
    )
    overlap_checks = {
        "pool_dev_overlap": int(len(set(pool_df["sample_id"]) & set(dev_df["sample_id"]))),
        "pool_final_overlap": int(len(set(pool_df["sample_id"]) & set(final_df["sample_id"]))),
        "dev_final_overlap": int(len(set(dev_df["sample_id"]) & set(final_df["sample_id"]))),
    }
    if any(v != 0 for v in overlap_checks.values()):
        raise ValueError(f"Protocol split overlap detected: {overlap_checks}")

    pool_df.to_csv(save_dir / "acquisition_pool_v10.csv", index=False, encoding="utf-8-sig")
    dev_df.to_csv(save_dir / "development_eval_v10.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(save_dir / "final_test_v10_LOCKED_UNUSED.csv", index=False, encoding="utf-8-sig")
    unused_df.to_csv(save_dir / "unused_reserve_v10.csv", index=False, encoding="utf-8-sig")

    embedding_dir, embedding_lookup, embedding_config = load_or_build_embeddings(pool_df, save_dir)

    initial_df = sample_initial_labeled_set(pool_df, initial_seed_size=initial_size, acquisition_seed=acquisition_seed).reset_index(drop=True)
    current_pool = pool_df[~pool_df["sample_id"].isin(initial_df["sample_id"])].copy().reset_index(drop=True)
    random_query = select_random(current_pool, query_size, acquisition_seed)

    config = {
        "experiment_id": save_dir.name,
        "experiment_label": "V10 NEU large-pool one-cycle smoke",
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "save_dir": str(save_dir),
        "dataset_root": str(dataset_root),
        "neu_manifest_size": int(len(manifest)),
        "pool_size": int(len(pool_df)),
        "development_eval_size": int(len(dev_df)),
        "final_test_size": int(len(final_df)),
        "unused_size": int(len(unused_df)),
        "final_test_used": False,
        "overlap_checks": overlap_checks,
        "acquisition_seed": acquisition_seed,
        "training_seed": training_seed,
        "initial_seed_size": initial_size,
        "query_size": query_size,
        "rounds": 1,
        "strategies": STRATEGIES,
        "expected_trainings": 3,
        "pool_per_class": pool_per_class,
        "dev_per_class": dev_per_class,
        "final_per_class": final_per_class,
        "embedding_dir": str(embedding_dir),
        "embedding_config": embedding_config,
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
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    (save_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("[V10 NEU large-pool smoke]")
    print(f"Output dir : {save_dir}")
    print(f"Pool       : {len(pool_df)} images ({pool_per_class}/class)")
    print(f"Dev eval   : {len(dev_df)} images ({dev_per_class}/class)")
    print(f"Final test : {len(final_df)} images LOCKED / NOT USED")
    print(f"Initial    : {len(initial_df)}")
    print(f"Query      : {query_size}")
    print(f"Dry run    : {dry_run}")
    print("=" * 100)

    selected_logs = [
        add_selection_metadata(initial_df, "__SHARED_ROUND0__", 0, "shared_initial_random", acquisition_seed=acquisition_seed),
        add_selection_metadata(random_query, RANDOM_STRATEGY, 1, RANDOM_STRATEGY, acquisition_seed=acquisition_seed),
    ]
    cumulative_random = pd.concat([initial_df, random_query], ignore_index=True, sort=False).drop_duplicates("sample_id", keep="first")

    results_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    build_logs: list[pd.DataFrame] = []
    per_class_rows: list[pd.DataFrame] = []

    # Shared round 0.
    yaml_path, build_log = v6.build_yolo_dataset(initial_df, dev_df, dataset_root / "__SHARED_ROUND0__")
    build_log.insert(0, "strategy", "__SHARED_ROUND0__")
    build_log.insert(1, "round", 0)
    build_logs.append(build_log)
    row0, runtime0 = train_row(
        yaml_path=yaml_path,
        save_dir=save_dir,
        strategy="__SHARED_ROUND0__",
        acquisition_seed=acquisition_seed,
        training_seed=training_seed,
        round_idx=0,
        labeled_budget=len(initial_df),
        dev_eval_size=len(dev_df),
    )
    results_rows.append(row0)
    runtime_rows.append(runtime0)

    round0_weight = Path(str(row0.get("train_run_dir", ""))) / "weights" / "best.pt"
    if dry_run:
        # Use no detector-dependent V9b query in dry-run if there is no trained detector;
        # use existing yolov8n only for structural testing would change the protocol.
        v9b_query = current_pool.iloc[0:0].copy()
    else:
        if not round0_weight.exists():
            raise FileNotFoundError(f"Round0 best.pt missing: {round0_weight}")
        model = probe.YOLO(str(round0_weight))
        detector_scores = probe.prediction_rows(model, current_pool, device=device, imgsz=imgsz, conf=conf, iou=iou)
        scored_pool = current_pool.merge(detector_scores, on="sample_id", how="left")
        scored_pool.to_csv(save_dir / "v9b_round1_detector_scores.csv", index=False, encoding="utf-8-sig")
        v9b_query = stage1.select_instance_rich_dino_balanced(
            scored_pool,
            initial_df,
            embedding_lookup,
            query_size=query_size,
            candidate_fraction=candidate_fraction,
            weights=stage1.v9b_weight_config(),
            max_no_box=stage1.int_env("AL_V9B_MAX_NO_BOX", 0),
            min_pseudo_boxes=stage1.int_env("AL_V9B_MIN_PSEUDO_BOXES", 2),
            max_per_pred_class=stage1.int_env("AL_V9B_MAX_PER_PRED_CLASS", 2),
        )
    selected_logs.append(add_selection_metadata(v9b_query, V9B_STRATEGY, 1, V9B_STRATEGY, acquisition_seed=acquisition_seed))
    cumulative_v9b = pd.concat([initial_df, v9b_query], ignore_index=True, sort=False).drop_duplicates("sample_id", keep="first")

    # Round 1 Random.
    yaml_path, build_log = v6.build_yolo_dataset(cumulative_random, dev_df, dataset_root / RANDOM_STRATEGY / "round_1")
    build_log.insert(0, "strategy", RANDOM_STRATEGY)
    build_log.insert(1, "round", 1)
    build_logs.append(build_log)
    row_random, runtime_random = train_row(
        yaml_path=yaml_path,
        save_dir=save_dir,
        strategy=RANDOM_STRATEGY,
        acquisition_seed=acquisition_seed,
        training_seed=training_seed,
        round_idx=1,
        labeled_budget=len(cumulative_random),
        dev_eval_size=len(dev_df),
    )
    results_rows.append(row_random)
    runtime_rows.append(runtime_random)
    per_class_rows.append(extract_per_class_metrics(row_random, RANDOM_STRATEGY, acquisition_seed, training_seed, 1))

    # Round 1 V9b.
    yaml_path, build_log = v6.build_yolo_dataset(cumulative_v9b, dev_df, dataset_root / V9B_STRATEGY / "round_1")
    build_log.insert(0, "strategy", V9B_STRATEGY)
    build_log.insert(1, "round", 1)
    build_logs.append(build_log)
    row_v9b, runtime_v9b = train_row(
        yaml_path=yaml_path,
        save_dir=save_dir,
        strategy=V9B_STRATEGY,
        acquisition_seed=acquisition_seed,
        training_seed=training_seed,
        round_idx=1,
        labeled_budget=len(cumulative_v9b),
        dev_eval_size=len(dev_df),
    )
    results_rows.append(row_v9b)
    runtime_rows.append(runtime_v9b)
    per_class_rows.append(extract_per_class_metrics(row_v9b, V9B_STRATEGY, acquisition_seed, training_seed, 1))

    selected_df = pd.concat(selected_logs, ignore_index=True, sort=False)
    cumulative_df = pd.concat(
        [
            add_selection_metadata(initial_df, "__SHARED_ROUND0__", 0, "cumulative_labeled", acquisition_seed=acquisition_seed),
            add_selection_metadata(cumulative_random, RANDOM_STRATEGY, 1, "cumulative_labeled", acquisition_seed=acquisition_seed),
            add_selection_metadata(cumulative_v9b, V9B_STRATEGY, 1, "cumulative_labeled", acquisition_seed=acquisition_seed),
        ],
        ignore_index=True,
        sort=False,
    )
    results_df = pd.DataFrame(results_rows)
    actual_stats_df, actual_class_df = v7.actual_stats_by_round(cumulative_df)

    selected_df.to_csv(save_dir / "all_selected_samples_by_round.csv", index=False, encoding="utf-8-sig")
    cumulative_df.to_csv(save_dir / "cumulative_labeled_sets_by_round.csv", index=False, encoding="utf-8-sig")
    results_df.to_csv(save_dir / "all_round_results.csv", index=False, encoding="utf-8-sig")
    actual_stats_df.to_csv(save_dir / "actual_instance_statistics_by_round.csv", index=False, encoding="utf-8-sig")
    actual_class_df.to_csv(save_dir / "actual_class_distribution_by_round.csv", index=False, encoding="utf-8-sig")
    pd.concat(build_logs, ignore_index=True).to_csv(save_dir / "dataset_build_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runtime_rows).to_csv(save_dir / "runtime_profile.csv", index=False, encoding="utf-8-sig")
    pd.concat(per_class_rows, ignore_index=True).to_csv(save_dir / "per_class_metrics_by_round.csv", index=False, encoding="utf-8-sig") if per_class_rows else pd.DataFrame().to_csv(save_dir / "per_class_metrics_by_round.csv", index=False, encoding="utf-8-sig")
    write_summary(save_dir, config, results_df, actual_stats_df)

    print("=" * 100)
    print("[DONE] V10 NEU large-pool smoke finished")
    print(f"Output dir: {save_dir}")
    print("Final test used: False")
    print("=" * 100)


if __name__ == "__main__":
    main()

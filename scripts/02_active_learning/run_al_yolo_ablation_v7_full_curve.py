"""V7 DINO visual-diversity full active-learning curve.

This runner freezes the post-gate method definition and evaluates whether
Dataset-Balanced DINO Visual Diversity improves label efficiency across
budgets 15 -> 20 -> 25 -> 30 -> 35.

It intentionally does not evaluate final_test_v7.  All training/evaluation is
development_eval_v7 only.

Default behavior is safe:
    AL_DRY_RUN_ONLY=1      -> build selections/results skeleton only
    AL_SELECTION_ONLY=1    -> selection diagnostics only, no YOLO dataset build

Set AL_DRY_RUN_ONLY=0 only when ready for the 65 YOLO trainings.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402
from analyze_instance_richness_v7 import parse_image_instances, strategy_stats  # noqa: E402
from audit_detection_pipeline_v7 import (  # noqa: E402
    build_image_index,
    build_xml_index,
    load_priority_scores,
    parse_bool_env,
    parse_int_list_env,
)
from canonical_sampling_v7 import canonicalize_pool_for_sampling, sample_initial_labeled_set  # noqa: E402
from run_al_yolo_ablation_v7_visual_instance import (  # noqa: E402
    PROHIBITED_GTFREE_COLUMNS,
    assert_gtfree_view,
    latest_embedding_dir,
    make_gtfree_view,
)
from run_confirmatory_training_v7 import read_best_epochs  # noqa: E402
from run_training_variance_v7 import bootstrap_ci  # noqa: E402

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    raise ImportError("Install ultralytics before running V7 full-curve experiments.") from exc


PROJECT_ROOT = v6.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / os.environ.get("AL_FULL_CURVE_RUNS_SUBDIR", "active_learning_ablation_v7_full_curve")
DATASETS_ROOT = PROJECT_ROOT / "datasets" / os.environ.get("AL_FULL_CURVE_DATASETS_SUBDIR", "active_learning_ablation_v7_full_curve")
RUN_NAME_PREFIX = os.environ.get("AL_FULL_CURVE_RUN_PREFIX", "v7_full_curve")
EXPERIMENT_LABEL = os.environ.get("AL_EXPERIMENT_LABEL", "V7 DINO full learning curve")
SUMMARY_FILENAME = os.environ.get("AL_SUMMARY_FILENAME", "v7_full_curve_summary.md")

RANDOM_STRATEGY = "GTFreeRandom"
DBC_STRATEGY = "GTFreeDatasetBalancedConsistency"
VISUAL_STRATEGY = "GTFreeDatasetBalancedVisualDiversity"
FULL_CURVE_STRATEGIES = [RANDOM_STRATEGY, DBC_STRATEGY, VISUAL_STRATEGY]
METRICS = ["map50", "map5095"]

REGISTRY_COLUMNS = [
    "experiment_id",
    "timestamp",
    "git_commit",
    "git_dirty",
    "runner_sha256",
    "priority_csv_sha256",
    "DINO_manifest_sha256",
    "development_eval_sha256",
    "strategy",
    "acquisition_seed",
    "training_seed",
    "round",
    "labeled_budget",
    "model",
    "epochs",
    "patience",
    "batch",
    "workers",
    "cache",
    "status",
    "retry_count",
    "train_run_dir",
    "result_path",
    "map50",
    "map5095",
    "best_epoch",
    "wall_clock_sec",
    "error",
]


def parse_csv_env(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return default
    return [v.strip() for v in value.split(",") if v.strip()]


def dataset_filter_values(name: str) -> list[str]:
    return parse_csv_env(name, [])


def apply_dataset_filter(df: pd.DataFrame, env_name: str) -> tuple[pd.DataFrame, list[str]]:
    allowed = dataset_filter_values(env_name)
    if not allowed:
        return df, []
    if "dataset_type" not in df.columns:
        raise ValueError(f"{env_name} was set, but dataframe has no dataset_type column.")
    before = len(df)
    out = df[df["dataset_type"].astype(str).isin(allowed)].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{env_name}={allowed} removed all rows from dataframe with {before} rows.")
    return out, allowed


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip())
    except Exception:
        return True


def latest_eval_protocol_dir() -> Path:
    override = os.environ.get("EVAL_PROTOCOL_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    root = PROJECT_ROOT / "runs" / "evaluation_protocol_v7"
    runs = [p for p in root.glob("eval_protocol_*") if p.is_dir()] if root.exists() else []
    if not runs:
        raise FileNotFoundError("No evaluation protocol directory found. Run prepare_evaluation_protocol_v7.py first.")
    return max(runs, key=lambda p: p.stat().st_mtime)


def load_development_eval() -> tuple[Path, Path, pd.DataFrame]:
    eval_split = os.environ.get("AL_EVAL_SPLIT", "development_eval_v7").strip()
    if eval_split == "final_test_v7":
        raise RuntimeError("Final test is locked during V7 full-curve development.")
    if eval_split != "development_eval_v7":
        raise ValueError(f"Unsupported AL_EVAL_SPLIT={eval_split!r}; use development_eval_v7.")
    protocol_dir = latest_eval_protocol_dir()
    dev_path = protocol_dir / "development_eval_v7.csv"
    if not dev_path.exists():
        raise FileNotFoundError(f"Missing development eval manifest: {dev_path}")
    dev_df, _ = apply_dataset_filter(pd.read_csv(dev_path), "AL_DEV_EVAL_DATASET_FILTER")
    return protocol_dir, dev_path, dev_df


def make_sample_id(dataset_type: str, image_name: str, resolved_path: str, sha256: str) -> str:
    payload = f"{dataset_type}|{image_name}|{resolved_path}|{sha256}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def add_sample_identity(df: pd.DataFrame, image_index: dict[tuple[str, str], list[Path]]) -> pd.DataFrame:
    return canonicalize_pool_for_sampling(df, image_index, set_sample_id=True)


def load_priority_pool_with_identity() -> tuple[Path, pd.DataFrame, dict[tuple[str, str], list[Path]]]:
    priority_csv, df = load_priority_scores()
    image_index = build_image_index()
    pool = add_sample_identity(df, image_index)
    pool, _ = apply_dataset_filter(pool, "AL_POOL_DATASET_FILTER")
    return priority_csv, pool, image_index


def load_dino_embeddings(full_pool: pd.DataFrame) -> tuple[Path, pd.DataFrame, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    os.environ.setdefault("AL_EMBEDDING_BACKEND", "dinov2")
    embedding_dir = latest_embedding_dir()
    if embedding_dir is None:
        raise FileNotFoundError("No DINO embedding cache found. Run build_visual_embedding_cache_v7.py first.")
    manifest_path = embedding_dir / "embedding_manifest.csv"
    embeddings_path = embedding_dir / "embeddings.npy"
    config_path = embedding_dir / "embedding_config.json"
    if not manifest_path.exists() or not embeddings_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Incomplete embedding cache: {embedding_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("backend") != "dinov2":
        raise ValueError(f"Expected dinov2 cache, got {config.get('backend')}")
    if int(config.get("embedding_dim", -1)) != 384:
        raise ValueError(f"Expected DINOv2-small 384-d embeddings, got {config.get('embedding_dim')}")
    manifest = pd.read_csv(manifest_path)
    embeddings = np.load(embeddings_path)
    if embeddings.shape[0] != len(manifest):
        raise ValueError(
            f"DINO cache sample count mismatch: manifest={len(manifest)}, embeddings={embeddings.shape[0]}"
        )

    pool_keys = set(zip(full_pool["dataset_type"].astype(str), full_pool["image_name"].astype(str), full_pool["image_sha256"].astype(str)))
    manifest_keys = set(zip(manifest["dataset_type"].astype(str), manifest["image_name"].astype(str), manifest["sha256"].astype(str)))
    missing = pool_keys - manifest_keys
    if missing:
        raise ValueError(f"DINO cache is missing {len(missing)} pool samples. First: {list(missing)[:5]}")

    sample_id_by_key = {
        (str(row["dataset_type"]), str(row["image_name"]), str(row["image_sha256"])): str(row["sample_id"])
        for _, row in full_pool.iterrows()
    }
    lookup: dict[str, np.ndarray] = {}
    for _, row in manifest.iterrows():
        key = (str(row["dataset_type"]), str(row["image_name"]), str(row["sha256"]))
        if key not in sample_id_by_key:
            continue
        sample_id = sample_id_by_key[key]
        lookup[sample_id] = embeddings[int(row["embedding_index"])]
    return embedding_dir, manifest, embeddings, lookup, config


def choose_dataset_by_deficit(current_pool: pd.DataFrame, labeled_df: pd.DataFrame, selected_df: pd.DataFrame) -> str:
    available = sorted(current_pool["dataset_type"].dropna().astype(str).unique())
    if not available:
        return str(current_pool.iloc[0]["dataset_type"])
    labeled_counts = labeled_df["dataset_type"].astype(str).value_counts().to_dict() if len(labeled_df) else {}
    selected_counts = selected_df["dataset_type"].astype(str).value_counts().to_dict() if len(selected_df) else {}
    total_after = sum(labeled_counts.get(ds, 0) + selected_counts.get(ds, 0) for ds in available) + 1
    target = total_after / max(1, len(available))
    return sorted(
        available,
        key=lambda ds: (-(target - (labeled_counts.get(ds, 0) + selected_counts.get(ds, 0))), ds),
    )[0]


def min_cosine_distance(sample_id: str, reference_sample_ids: list[str], embedding_lookup: dict[str, np.ndarray]) -> float:
    emb = embedding_lookup.get(sample_id)
    if emb is None:
        return 0.0
    refs = [embedding_lookup[sid] for sid in reference_sample_ids if sid in embedding_lookup]
    if not refs:
        return 1.0
    ref_mat = np.vstack(refs)
    return float(1.0 - np.max(ref_mat @ emb))


def select_random(current_pool: pd.DataFrame, sample_size: int, seed: int, round_idx: int) -> pd.DataFrame:
    return current_pool.sample(n=min(sample_size, len(current_pool)), random_state=seed + round_idx * 101)


def select_dataset_balanced_consistency(
    current_pool_full: pd.DataFrame,
    labeled_full: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    current_pool = make_gtfree_view(current_pool_full)
    labeled_df = make_gtfree_view(labeled_full)
    assert_gtfree_view(current_pool)
    assert_gtfree_view(labeled_df)
    remaining = current_pool.sort_values("sample_id", kind="mergesort").copy()
    selected = []
    selected_df = remaining.iloc[0:0].copy()
    for _ in range(min(sample_size, len(remaining))):
        ds = choose_dataset_by_deficit(remaining, labeled_df, selected_df)
        candidates = remaining[remaining["dataset_type"].astype(str).eq(ds)].copy()
        if candidates.empty:
            candidates = remaining.copy()
        candidates["_score"] = pd.to_numeric(candidates["score_consistency_only"], errors="coerce").fillna(0.0)
        picked_idx = candidates.sort_values(["_score", "sample_id"], ascending=[False, True], kind="mergesort").index[0]
        picked = remaining.loc[[picked_idx]].drop(columns=["_score"], errors="ignore")
        selected.append(picked)
        selected_df = pd.concat([selected_df, picked], ignore_index=False)
        remaining = remaining.drop(index=[picked_idx])
    picked_view = pd.concat(selected) if selected else current_pool.iloc[0:0].copy()
    return current_pool_full[current_pool_full["sample_id"].isin(picked_view["sample_id"])].copy()


def select_dataset_balanced_visual(
    current_pool_full: pd.DataFrame,
    labeled_full: pd.DataFrame,
    sample_size: int,
    embedding_lookup: dict[str, np.ndarray],
) -> pd.DataFrame:
    current_pool = make_gtfree_view(current_pool_full)
    labeled_df = make_gtfree_view(labeled_full)
    assert_gtfree_view(current_pool)
    assert_gtfree_view(labeled_df)
    remaining = current_pool.sort_values("sample_id", kind="mergesort").copy()
    selected = []
    selected_df = remaining.iloc[0:0].copy()
    for _ in range(min(sample_size, len(remaining))):
        ds = choose_dataset_by_deficit(remaining, labeled_df, selected_df)
        candidates = remaining[remaining["dataset_type"].astype(str).eq(ds)].copy()
        if candidates.empty:
            candidates = remaining.copy()
        reference_ids = labeled_df["sample_id"].astype(str).tolist() + selected_df["sample_id"].astype(str).tolist()
        candidates["_visual_distance"] = [
            min_cosine_distance(str(row["sample_id"]), reference_ids, embedding_lookup) for _, row in candidates.iterrows()
        ]
        picked_idx = candidates.sort_values(["_visual_distance", "sample_id"], ascending=[False, True], kind="mergesort").index[0]
        picked = remaining.loc[[picked_idx]].drop(columns=["_visual_distance"], errors="ignore")
        selected.append(picked)
        selected_df = pd.concat([selected_df, picked], ignore_index=False)
        remaining = remaining.drop(index=[picked_idx])
    picked_view = pd.concat(selected) if selected else current_pool.iloc[0:0].copy()
    return current_pool_full[current_pool_full["sample_id"].isin(picked_view["sample_id"])].copy()


def select_samples(
    strategy: str,
    current_pool: pd.DataFrame,
    labeled: pd.DataFrame,
    sample_size: int,
    seed: int,
    round_idx: int,
    embedding_lookup: dict[str, np.ndarray],
) -> pd.DataFrame:
    if strategy == RANDOM_STRATEGY:
        return select_random(current_pool, sample_size, seed, round_idx)
    if strategy == DBC_STRATEGY:
        return select_dataset_balanced_consistency(current_pool, labeled, sample_size)
    if strategy == VISUAL_STRATEGY:
        return select_dataset_balanced_visual(current_pool, labeled, sample_size, embedding_lookup)
    raise ValueError(f"Unsupported full-curve strategy: {strategy}")


def add_selection_metadata(selected: pd.DataFrame, seed: int, strategy: str, round_idx: int, selection_type: str) -> pd.DataFrame:
    out = selected.copy()
    out.insert(0, "acquisition_seed", seed)
    out.insert(1, "training_seed", 1000 + seed)
    out.insert(2, "strategy", strategy)
    out.insert(3, "round", round_idx)
    out.insert(4, "labeled_budget_after_round", 15 + 5 * round_idx)
    out.insert(5, "selection_type", selection_type)
    out.insert(6, "rank_in_selection", range(1, len(out) + 1))
    return out


def build_selection_plan(
    full_pool: pd.DataFrame,
    seeds: list[int],
    strategies: list[str],
    initial_size: int,
    rounds: int,
    query_size: int,
    embedding_lookup: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_logs = []
    cumulative_logs = []
    for seed in seeds:
        initial = sample_initial_labeled_set(
            full_pool,
            initial_seed_size=initial_size,
            acquisition_seed=seed,
        ).copy()
        for strategy in strategies:
            current_pool = full_pool[~full_pool["sample_id"].isin(initial["sample_id"])].copy()
            labeled = initial.copy()
            selected_logs.append(add_selection_metadata(initial, seed, strategy, 0, "shared_initial_seed_random"))
            cumulative = add_selection_metadata(labeled, seed, strategy, 0, "cumulative_labeled")
            cumulative_logs.append(cumulative)
            for round_idx in range(1, rounds + 1):
                picked = select_samples(strategy, current_pool, labeled, query_size, seed, round_idx, embedding_lookup)
                picked = picked.sort_values("sample_id", kind="mergesort").copy()
                selected_logs.append(add_selection_metadata(picked, seed, strategy, round_idx, strategy))
                labeled = pd.concat([labeled, picked], ignore_index=True)
                current_pool = current_pool[~current_pool["sample_id"].isin(picked["sample_id"])].copy()
                if labeled["sample_id"].duplicated().any():
                    raise AssertionError(f"Duplicate labeled sample in seed={seed}, strategy={strategy}, round={round_idx}")
                expected_budget = initial_size + query_size * round_idx
                if len(labeled) != expected_budget:
                    raise AssertionError(f"Expected budget={expected_budget}, got {len(labeled)}")
                cumulative_logs.append(add_selection_metadata(labeled, seed, strategy, round_idx, "cumulative_labeled"))
    selected_df = pd.concat(selected_logs, ignore_index=True) if selected_logs else pd.DataFrame()
    cumulative_df = pd.concat(cumulative_logs, ignore_index=True) if cumulative_logs else pd.DataFrame()
    return selected_df, cumulative_df


def selected_set_for(cumulative_df: pd.DataFrame, acquisition_seed: int, strategy: str, round_idx: int) -> pd.DataFrame:
    sub = cumulative_df[
        (cumulative_df["acquisition_seed"].eq(acquisition_seed))
        & (cumulative_df["strategy"].eq(strategy))
        & (cumulative_df["round"].eq(round_idx))
    ].copy()
    sub = sub.drop_duplicates(subset=["sample_id"], keep="first")
    return sub.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def get_device() -> str | int:
    override = os.environ.get("AL_YOLO_DEVICE")
    if override is not None and override.strip():
        raw = override.strip()
        return int(raw) if raw.isdigit() else raw
    return v6.get_device()


def train_eval(
    yaml_path: Path,
    save_dir: Path,
    strategy: str,
    acquisition_seed: int,
    training_seed: int,
    round_idx: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dry_run = parse_bool_env("AL_DRY_RUN_ONLY", True)
    epochs = int(os.environ.get("AL_EPOCHS_PER_ROUND", "100"))
    patience = int(os.environ.get("AL_YOLO_PATIENCE", str(epochs)))
    imgsz = int(os.environ.get("AL_IMGSZ", "640"))
    batch = int(os.environ.get("AL_BATCH_SIZE", "8"))
    workers = int(os.environ.get("AL_WORKERS", "4"))
    model_name = os.environ.get("AL_YOLO_MODEL_NAME", "yolov8n.pt")
    plots = parse_bool_env("AL_YOLO_PLOTS", False)
    cache_env = os.environ.get("AL_YOLO_CACHE", "ram").strip().lower()
    cache_value: bool | str = cache_env if cache_env in {"ram", "disk"} else parse_bool_env("AL_YOLO_CACHE", False)

    runtime = {
        "strategy": strategy,
        "acquisition_seed": acquisition_seed,
        "training_seed": training_seed,
        "round": round_idx,
        "epochs": epochs,
        "patience": patience,
        "batch": batch,
        "workers": workers,
        "cache": cache_value,
        "plots": plots,
        "amp": True,
    }
    if dry_run:
        return (
            {
                "map50": np.nan,
                "map5095": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "train_status": "dry_run",
                "train_run_dir": None,
                "error": None,
            },
            {**runtime, "train_eval_sec": 0.0, "total_sec": 0.0},
        )

    t0 = time.perf_counter()
    try:
        model = YOLO(model_name)
        train_results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=get_device(),
            project=str(save_dir / "yolo_train_runs"),
            name=f"seed{acquisition_seed}_{strategy}_R{round_idx}_trainseed{training_seed}",
            exist_ok=True,
            patience=patience,
            cache=cache_value,
            plots=plots,
            verbose=False,
            seed=training_seed,
            deterministic=True,
        )
        train_run_dir = Path(getattr(train_results, "save_dir", "")) if getattr(train_results, "save_dir", None) else None
        best_weight = train_run_dir / "weights" / "best.pt" if train_run_dir else None
        eval_model = YOLO(str(best_weight)) if best_weight and best_weight.exists() else model
        metrics = eval_model.val(
            data=str(yaml_path),
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=get_device(),
            split="val",
            verbose=False,
            plots=False,
        )
        elapsed = time.perf_counter() - t0
        return (
            {
                "map50": round(float(metrics.box.map50), 6),
                "map5095": round(float(metrics.box.map), 6),
                "precision": round(float(metrics.box.mp), 6),
                "recall": round(float(metrics.box.mr), 6),
                "train_status": "success",
                "train_run_dir": str(train_run_dir) if train_run_dir else None,
                "error": None,
                **read_best_epochs(train_run_dir),
            },
            {**runtime, "train_eval_sec": elapsed, "total_sec": elapsed},
        )
    except Exception as exc:
        traceback.print_exc()
        elapsed = time.perf_counter() - t0
        return (
            {
                "map50": np.nan,
                "map5095": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "train_status": "failed",
                "train_run_dir": None,
                "error": str(exc),
            },
            {**runtime, "train_eval_sec": elapsed, "total_sec": elapsed},
        )


def extract_per_class_metrics(result: dict[str, Any], strategy: str, acquisition_seed: int, training_seed: int, round_idx: int) -> pd.DataFrame:
    # Per-class extraction differs across Ultralytics versions.  Preserve a
    # stable placeholder rather than silently omitting the expected artifact.
    rows = []
    for class_id, class_name in enumerate(v6.CLASS_NAMES):
        rows.append(
            {
                "strategy": strategy,
                "acquisition_seed": acquisition_seed,
                "training_seed": training_seed,
                "round": round_idx,
                "class_id": class_id,
                "class_name": class_name,
                "ap50": np.nan,
                "ap5095": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "note": "per-class extraction unavailable in this runner; aggregate metrics are valid",
            }
        )
    return pd.DataFrame(rows)


def result_key(acquisition_seed: int, strategy: str, round_idx: int) -> tuple[int, str, int]:
    return int(acquisition_seed), str(strategy), int(round_idx)


def load_existing_results(save_dir: Path) -> pd.DataFrame:
    path = save_dir / "all_round_results.csv"
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def existing_success_map(existing: pd.DataFrame) -> dict[tuple[int, str, int], dict[str, Any]]:
    out = {}
    if existing.empty:
        return out
    good = existing[existing["train_status"].astype(str).isin(["success", "shared_baseline", "dry_run", "selection_only"])]
    for _, row in good.iterrows():
        out[result_key(row["acquisition_seed"], row["strategy"], row["round"])] = row.to_dict()
    return out


def append_run_registry(registry_path: Path, row: dict[str, Any], config: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists() and registry_path.stat().st_size > 0:
        df = pd.read_csv(registry_path)
    else:
        df = pd.DataFrame(columns=REGISTRY_COLUMNS)
    payload = {
        "experiment_id": config["experiment_id"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": config["git_commit"],
        "git_dirty": config["git_dirty"],
        "runner_sha256": config["runner_sha256"],
        "priority_csv_sha256": config["priority_csv_sha256"],
        "DINO_manifest_sha256": config["DINO_manifest_sha256"],
        "development_eval_sha256": config["development_eval_sha256"],
        "strategy": row.get("strategy"),
        "acquisition_seed": row.get("acquisition_seed"),
        "training_seed": row.get("training_seed"),
        "round": row.get("round"),
        "labeled_budget": row.get("labeled_budget"),
        "model": config["model"],
        "epochs": config["epochs"],
        "patience": config["patience"],
        "batch": config["batch"],
        "workers": config["workers"],
        "cache": config["cache"],
        "status": row.get("train_status"),
        "retry_count": row.get("retry_count", 0),
        "train_run_dir": row.get("train_run_dir"),
        "result_path": str(config["save_dir"]),
        "map50": row.get("map50"),
        "map5095": row.get("map5095"),
        "best_epoch": row.get("best_epoch_map5095", row.get("best_epoch_map50")),
        "wall_clock_sec": row.get("total_sec"),
        "error": row.get("error"),
    }
    df = pd.concat([df, pd.DataFrame([payload])], ignore_index=True)
    df = df.reindex(columns=REGISTRY_COLUMNS)
    df.to_csv(registry_path, index=False, encoding="utf-8-sig")


def compute_visual_redundancy(selected_df: pd.DataFrame, embedding_lookup: dict[str, np.ndarray]) -> pd.DataFrame:
    acquired = selected_df[pd.to_numeric(selected_df["round"], errors="coerce").fillna(0).gt(0)].copy()
    rows = []
    for (seed, strategy, round_idx), sub in acquired.groupby(["acquisition_seed", "strategy", "round"], dropna=False):
        embs = [embedding_lookup[sid] for sid in sub["sample_id"].astype(str) if sid in embedding_lookup]
        if len(embs) < 2:
            pair_mean = np.nan
            pair_max = np.nan
        else:
            mat = np.vstack(embs)
            sims = mat @ mat.T
            upper = sims[np.triu_indices(len(mat), k=1)]
            pair_mean = float(np.mean(upper))
            pair_max = float(np.max(upper))
        rows.append(
            {
                "acquisition_seed": seed,
                "strategy": strategy,
                "round": round_idx,
                "selected_batch_pairwise_cosine_similarity_mean": pair_mean,
                "selected_batch_pairwise_cosine_similarity_max": pair_max,
                "num_embedded_selected": len(embs),
            }
        )
    return pd.DataFrame(rows)


def compute_distance_to_labeled(selected_df: pd.DataFrame, embedding_lookup: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for (seed, strategy), sdf in selected_df.groupby(["acquisition_seed", "strategy"], dropna=False):
        labeled_before: list[str] = []
        for round_idx in sorted(pd.to_numeric(sdf["round"], errors="coerce").dropna().astype(int).unique()):
            round_df = sdf[sdf["round"].eq(round_idx)].copy()
            if round_idx == 0:
                labeled_before.extend(round_df["sample_id"].astype(str).tolist())
                continue
            for _, row in round_df.iterrows():
                rows.append(
                    {
                        "acquisition_seed": seed,
                        "strategy": strategy,
                        "round": round_idx,
                        "dataset_type": row["dataset_type"],
                        "image_name": row["image_name"],
                        "sample_id": row["sample_id"],
                        "rank_in_selection": row["rank_in_selection"],
                        "min_cosine_distance_to_labeled_before_selection": min_cosine_distance(
                            str(row["sample_id"]), labeled_before, embedding_lookup
                        ),
                        "num_reference_labeled_before_selection": len(labeled_before),
                    }
                )
            labeled_before.extend(round_df["sample_id"].astype(str).tolist())
    return pd.DataFrame(rows)


def dataset_distribution_by_round(cumulative_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, strategy, round_idx), sub in cumulative_df.groupby(["acquisition_seed", "strategy", "round"], dropna=False):
        for dataset_type, count in sub["dataset_type"].astype(str).value_counts().sort_index().items():
            rows.append(
                {
                    "acquisition_seed": seed,
                    "strategy": strategy,
                    "round": round_idx,
                    "labeled_budget": len(sub),
                    "dataset_type": dataset_type,
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows)


def selection_overlap_matrix(cumulative_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, round_idx), seed_round_df in cumulative_df.groupby(["acquisition_seed", "round"], dropna=False):
        sets = {
            strategy: set(sub["sample_id"].astype(str))
            for strategy, sub in seed_round_df.groupby("strategy", dropna=False)
        }
        for a, set_a in sets.items():
            for b, set_b in sets.items():
                rows.append(
                    {
                        "acquisition_seed": seed,
                        "round": round_idx,
                        "strategy_a": a,
                        "strategy_b": b,
                        "overlap_count": len(set_a & set_b),
                        "strategy_a_size": len(set_a),
                        "strategy_b_size": len(set_b),
                        "jaccard": len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def actual_stats_by_round(cumulative_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_index = build_image_index()
    xml_index = build_xml_index()
    stats_rows = []
    class_rows = []
    for (seed, strategy, round_idx), sub in cumulative_df.groupby(["acquisition_seed", "strategy", "round"], dropna=False):
        tagged = sub.copy()
        tagged["seed"] = seed
        tagged["source_strategy"] = strategy
        tagged["round"] = round_idx
        instances = []
        for _, row in tagged.iterrows():
            instances.extend(parse_image_instances(row, image_index, xml_index))
        inst_df = pd.DataFrame(instances)
        stats = strategy_stats(tagged, inst_df)
        if len(stats):
            row = stats.iloc[0].to_dict()
            row.update({"acquisition_seed": seed, "strategy": strategy, "round": round_idx, "labeled_budget": len(sub)})
            stats_rows.append(row)
        if len(inst_df):
            dist = inst_df.groupby("actual_xml_class").size().reset_index(name="bbox_instance_count")
            dist.insert(0, "labeled_budget", len(sub))
            dist.insert(0, "round", round_idx)
            dist.insert(0, "strategy", strategy)
            dist.insert(0, "acquisition_seed", seed)
            class_rows.append(dist)
    return pd.DataFrame(stats_rows), pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame()


def normalized_aulc(budgets: np.ndarray, values: np.ndarray) -> float:
    budgets = np.asarray(budgets, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(budgets) & np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    x = budgets[mask]
    y = values[mask]
    area = float(np.sum((y[1:] + y[:-1]) * 0.5 * (x[1:] - x[:-1])))
    return area / float(x.max() - x.min())


def seed_strategy_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, strategy), sub in results_df.groupby(["acquisition_seed", "strategy"], dropna=False):
        sub = sub.sort_values("labeled_budget")
        budgets = sub["labeled_budget"].to_numpy(dtype=float)
        row = {
            "acquisition_seed": seed,
            "training_seed": int(sub["training_seed"].dropna().iloc[0]) if len(sub["training_seed"].dropna()) else np.nan,
            "strategy": strategy,
            "final_budget": int(sub["labeled_budget"].max()) if len(sub) else np.nan,
        }
        for metric in METRICS:
            vals = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
            row[f"final_{metric}"] = float(sub.loc[sub["labeled_budget"].idxmax(), metric]) if len(sub) else np.nan
            row[f"best_{metric}"] = float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan
            row[f"best_round_{metric}"] = int(sub.iloc[int(np.nanargmax(vals))]["round"]) if np.isfinite(vals).any() else np.nan
            row[f"normalized_aulc_{metric}"] = normalized_aulc(budgets, vals)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "final_map50",
        "final_map5095",
        "best_map50",
        "best_map5095",
        "normalized_aulc_map50",
        "normalized_aulc_map5095",
    ]
    rows = []
    for strategy, sub in seed_summary.groupby("strategy", dropna=False):
        row = {"strategy": strategy, "num_acquisition_seeds": int(len(sub))}
        for col in metric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            lo, hi = bootstrap_ci(finite) if len(finite) else (np.nan, np.nan)
            row[f"{col}_mean"] = float(np.mean(finite)) if len(finite) else np.nan
            row[f"{col}_std"] = float(np.std(finite, ddof=1)) if len(finite) >= 2 else np.nan
            row[f"{col}_ci95_low"] = lo
            row[f"{col}_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def exact_sign_flip_pvalue(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return np.nan
    obs = abs(float(diff.mean()))
    means = []
    for mask in range(1 << len(diff)):
        signs = np.array([1.0 if mask & (1 << i) else -1.0 for i in range(len(diff))])
        means.append(abs(float((diff * signs).mean())))
    return float(np.mean(np.asarray(means) >= obs - 1e-12))


def paired_strategy_comparisons(seed_summary: pd.DataFrame) -> pd.DataFrame:
    comparisons = [(VISUAL_STRATEGY, RANDOM_STRATEGY), (VISUAL_STRATEGY, DBC_STRATEGY), (DBC_STRATEGY, RANDOM_STRATEGY)]
    metrics = ["normalized_aulc_map5095", "normalized_aulc_map50", "final_map5095", "final_map50"]
    rows = []
    for treatment, baseline in comparisons:
        left = seed_summary[seed_summary["strategy"].eq(treatment)].set_index("acquisition_seed")
        right = seed_summary[seed_summary["strategy"].eq(baseline)].set_index("acquisition_seed")
        common = left.index.intersection(right.index)
        for metric in metrics:
            diff = (left.loc[common, metric] - right.loc[common, metric]).to_numpy(dtype=float) if len(common) else np.array([])
            finite = diff[np.isfinite(diff)]
            lo, hi = bootstrap_ci(finite) if len(finite) else (np.nan, np.nan)
            base_vals = right.loc[common, metric].to_numpy(dtype=float) if len(common) else np.array([])
            rel = finite / base_vals[np.isfinite(diff)] * 100 if len(finite) and np.all(np.isfinite(base_vals[np.isfinite(diff)])) else np.array([])
            rows.append(
                {
                    "treatment": treatment,
                    "baseline": baseline,
                    "metric": metric,
                    "num_pairs": int(len(finite)),
                    "mean_paired_difference": float(np.mean(finite)) if len(finite) else np.nan,
                    "std_paired_difference": float(np.std(finite, ddof=1)) if len(finite) >= 2 else np.nan,
                    "wins": int(np.sum(finite > 0)),
                    "ties": int(np.sum(finite == 0)),
                    "losses": int(np.sum(finite < 0)),
                    "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi,
                    "exact_sign_flip_pvalue": exact_sign_flip_pvalue(finite),
                    "relative_improvement_percent_mean": float(np.nanmean(rel)) if len(rel) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def paired_roundwise_differences(results_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [(VISUAL_STRATEGY, RANDOM_STRATEGY), (VISUAL_STRATEGY, DBC_STRATEGY), (DBC_STRATEGY, RANDOM_STRATEGY)]
    rows = []
    for treatment, baseline in comparisons:
        left = results_df[results_df["strategy"].eq(treatment)].set_index(["acquisition_seed", "round"])
        right = results_df[results_df["strategy"].eq(baseline)].set_index(["acquisition_seed", "round"])
        common = left.index.intersection(right.index)
        for (seed, round_idx) in common:
            for metric in METRICS:
                rows.append(
                    {
                        "treatment": treatment,
                        "baseline": baseline,
                        "acquisition_seed": seed,
                        "round": round_idx,
                        "labeled_budget": int(left.loc[(seed, round_idx), "labeled_budget"]),
                        "metric": metric,
                        "difference": float(left.loc[(seed, round_idx), metric] - right.loc[(seed, round_idx), metric]),
                    }
                )
    return pd.DataFrame(rows)


def budget_to_target(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, seed_df in results_df.groupby("acquisition_seed", dropna=False):
        random_final = seed_df[(seed_df["strategy"].eq(RANDOM_STRATEGY)) & (seed_df["round"].eq(4))]
        if random_final.empty:
            continue
        for metric in METRICS:
            target = float(random_final.iloc[0][metric])
            for strategy, sub in seed_df.groupby("strategy", dropna=False):
                sub = sub.sort_values("labeled_budget")
                envelope = pd.to_numeric(sub[metric], errors="coerce").cummax()
                hit = sub[envelope >= target]
                rows.append(
                    {
                        "acquisition_seed": seed,
                        "strategy": strategy,
                        "metric": metric,
                        "random_round4_target": target,
                        "budget_to_reach_target": int(hit.iloc[0]["labeled_budget"]) if len(hit) else np.nan,
                        "reached": bool(len(hit)),
                        "uses_cumulative_max_envelope": True,
                    }
                )
    return pd.DataFrame(rows)


def normalized_aulc_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    return seed_summary[
        ["acquisition_seed", "strategy", "normalized_aulc_map50", "normalized_aulc_map5095", "final_map50", "final_map5095"]
    ].copy()


def write_summary(
    save_dir: Path,
    config: dict[str, Any],
    results_df: pd.DataFrame,
    seed_summary: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    budget_df: pd.DataFrame,
) -> None:
    success_count = int(results_df["train_status"].astype(str).isin(["success", "shared_baseline", "dry_run", "selection_only"]).sum()) if len(results_df) else 0
    failed_count = int(results_df["train_status"].astype(str).eq("failed").sum()) if len(results_df) else 0
    visual = aggregate_df[aggregate_df["strategy"].eq(VISUAL_STRATEGY)]
    lock_lines = []
    if len(visual):
        visual_aulc = float(visual.iloc[0].get("normalized_aulc_map5095_mean", np.nan))
        lock_lines.append(f"- Visual-only normalized AULC mAP50-95 mean: {visual_aulc:.6f}")
    lines = [
        f"# {EXPERIMENT_LABEL}",
        "",
        "Development full-curve result only. Final test was not used.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## Run status",
        "",
        f"- Successful or skipped/dry-run rows: {success_count}",
        f"- Failed rows: {failed_count}",
        "",
        "## Strategy aggregate summary",
        "",
        aggregate_df.to_markdown(index=False) if len(aggregate_df) else "_No rows._",
        "",
        "## Paired acquisition-seed comparisons",
        "",
        paired_df.to_markdown(index=False) if len(paired_df) else "_No rows._",
        "",
        "## Budget-to-target",
        "",
        budget_df.to_markdown(index=False) if len(budget_df) else "_No rows._",
        "",
        "## Method-lock notes",
        "",
        *lock_lines,
        "",
        "Do not run final_test_v7 from this runner. Lock decision should be made from development full-curve criteria first.",
    ]
    (save_dir / SUMMARY_FILENAME).write_text("\n".join(lines), encoding="utf-8")


def resolve_save_dir() -> tuple[Path, Path, bool]:
    explicit_save = os.environ.get("AL_EXPLICIT_SAVE_DIR")
    if explicit_save:
        save_dir = Path(explicit_save).expanduser()
        if not save_dir.is_absolute():
            save_dir = PROJECT_ROOT / save_dir
        config_path = save_dir / "config.json"
        existed = config_path.exists()
        if existed:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            requested_dry = parse_bool_env("AL_DRY_RUN_ONLY", True)
            requested_selection_only = parse_bool_env("AL_SELECTION_ONLY", False)
            if bool(cfg.get("dry_run", True)) != requested_dry:
                raise RuntimeError(
                    f"Refusing to mix dry-run and training in {save_dir}. "
                    "Use a different AL_EXPLICIT_SAVE_DIR."
                )
            if bool(cfg.get("selection_only", False)) != requested_selection_only:
                raise RuntimeError(
                    f"Refusing to mix selection-only and training in {save_dir}. "
                    "Use a different AL_EXPLICIT_SAVE_DIR."
                )
        dataset_root = DATASETS_ROOT / save_dir.name
        save_dir.mkdir(parents=True, exist_ok=True)
        dataset_root.mkdir(parents=True, exist_ok=True)
        return save_dir, dataset_root, existed

    resume = parse_bool_env("AL_RESUME_EXISTING_RUN", False)
    explicit = os.environ.get("AL_RESUME_RUN_DIR")
    if resume and explicit:
        save_dir = Path(explicit).expanduser()
        if not save_dir.is_absolute():
            save_dir = PROJECT_ROOT / save_dir
        dataset_root = DATASETS_ROOT / save_dir.name
        save_dir.mkdir(parents=True, exist_ok=True)
        dataset_root.mkdir(parents=True, exist_ok=True)
        return save_dir, dataset_root, True
    if resume:
        current_dry_run = parse_bool_env("AL_DRY_RUN_ONLY", True)
        current_selection_only = parse_bool_env("AL_SELECTION_ONLY", False)
        runs = sorted(
            [p for p in RUNS_ROOT.glob(f"{RUN_NAME_PREFIX}_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if RUNS_ROOT.exists() else []
        for candidate in runs:
            config_path = candidate / "config.json"
            if not config_path.exists():
                continue
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                # Do not accidentally resume a selection-only/dry-run folder
                # when the caller is launching real training.
                if bool(cfg.get("dry_run", True)) != current_dry_run:
                    continue
                if bool(cfg.get("selection_only", False)) != current_selection_only:
                    continue
                expected = len(cfg.get("acquisition_seeds", [])) * len(cfg.get("strategies", [])) * (int(cfg.get("rounds", 4)) + 1)
                results_path = candidate / "all_round_results.csv"
                if not results_path.exists() or results_path.stat().st_size == 0:
                    dataset_root = DATASETS_ROOT / candidate.name
                    dataset_root.mkdir(parents=True, exist_ok=True)
                    return candidate, dataset_root, True
                existing = pd.read_csv(results_path)
                completed = existing[existing["train_status"].astype(str).isin(["success", "shared_baseline", "dry_run", "selection_only"])]
                failed = existing[existing["train_status"].astype(str).eq("failed")]
                if len(completed) < expected or len(failed) > 0:
                    dataset_root = DATASETS_ROOT / candidate.name
                    dataset_root.mkdir(parents=True, exist_ok=True)
                    return candidate, dataset_root, True
            except Exception:
                continue
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"{RUN_NAME_PREFIX}_{timestamp}"
    dataset_root = DATASETS_ROOT / f"{RUN_NAME_PREFIX}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    return save_dir, dataset_root, False


def run_analysis_if_available(save_dir: Path) -> None:
    try:
        import analyze_v7_full_curve

        analyze_v7_full_curve.generate_analysis(save_dir)
    except Exception as exc:
        print(f"[WARN] Analysis/plots generation failed: {exc}")


def main() -> None:
    save_dir, dataset_root, resumed = resolve_save_dir()
    priority_csv, full_pool, _ = load_priority_pool_with_identity()
    protocol_dir, dev_eval_path, dev_eval_df = load_development_eval()
    embedding_dir, dino_manifest, _, embedding_lookup, dino_config = load_dino_embeddings(full_pool)

    seeds = parse_int_list_env("AL_ACQUISITION_SEEDS", [42, 43, 44, 45, 46])
    strategies = parse_csv_env("AL_STRATEGIES", FULL_CURVE_STRATEGIES)
    if strategies != FULL_CURVE_STRATEGIES:
        allow_subset = parse_bool_env("AL_ALLOW_FROZEN_STRATEGY_SUBSET", False)
        valid_subset = bool(strategies) and all(strategy in FULL_CURVE_STRATEGIES for strategy in strategies)
        if not allow_subset or not valid_subset:
            raise ValueError(
                f"V7 full-curve method is frozen. Use strategies exactly: {FULL_CURVE_STRATEGIES}, "
                "or explicitly set AL_ALLOW_FROZEN_STRATEGY_SUBSET=1 for a pre-registered subset comparison."
            )
    initial_size = int(os.environ.get("AL_INITIAL_SEED_SIZE", "15"))
    rounds = int(os.environ.get("AL_ROUNDS", "4"))
    query_size = int(os.environ.get("AL_QUERY_SIZE", "5"))
    budgets = [initial_size + query_size * r for r in range(rounds + 1)]
    selection_only = parse_bool_env("AL_SELECTION_ONLY", False)
    dry_run = parse_bool_env("AL_DRY_RUN_ONLY", True)

    config = {
        "experiment_id": save_dir.name,
        "experiment_label": EXPERIMENT_LABEL,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "save_dir": str(save_dir),
        "dataset_root": str(dataset_root),
        "resumed": resumed,
        "priority_csv": str(priority_csv),
        "priority_csv_sha256": file_sha256(priority_csv),
        "eval_protocol_dir": str(protocol_dir),
        "development_eval_path": str(dev_eval_path),
        "development_eval_sha256": file_sha256(dev_eval_path),
        "development_eval_dataset_filter": dataset_filter_values("AL_DEV_EVAL_DATASET_FILTER"),
        "development_eval_size_after_filter": len(dev_eval_df),
        "pool_dataset_filter": dataset_filter_values("AL_POOL_DATASET_FILTER"),
        "pool_size_after_filter": len(full_pool),
        "pool_dataset_distribution_after_filter": full_pool["dataset_type"].astype(str).value_counts().to_dict(),
        "final_test_used": False,
        "embedding_dir": str(embedding_dir),
        "DINO_manifest_sha256": file_sha256(embedding_dir / "embedding_manifest.csv"),
        "DINO_config": dino_config,
        "strategies": strategies,
        "acquisition_seeds": seeds,
        "training_seed_rule": "training_seed = 1000 + acquisition_seed",
        "initial_seed_size": initial_size,
        "rounds": rounds,
        "query_size": query_size,
        "budgets": budgets,
        "canonical_sampling": True,
        "canonical_sampling_key": "canonical_sample_id = resolved_image_path::image_sha256",
        "initial_sampling_random_state_rule": "acquisition_seed + 999",
        "model": os.environ.get("AL_YOLO_MODEL_NAME", "yolov8n.pt"),
        "epochs": int(os.environ.get("AL_EPOCHS_PER_ROUND", "100")),
        "patience": int(os.environ.get("AL_YOLO_PATIENCE", os.environ.get("AL_EPOCHS_PER_ROUND", "100"))),
        "batch": int(os.environ.get("AL_BATCH_SIZE", "8")),
        "workers": int(os.environ.get("AL_WORKERS", "4")),
        "cache": os.environ.get("AL_YOLO_CACHE", "ram"),
        "dry_run": dry_run,
        "selection_only": selection_only,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "gtfree_prohibited_columns": sorted(PROHIBITED_GTFREE_COLUMNS),
        "primary_metric": "normalized_aulc_map5095",
    }
    (save_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    expected_trainings = len(seeds) * (1 + len(strategies) * rounds)
    gate_sec_per_train = 237.0
    print("=" * 100)
    print(f"[{EXPERIMENT_LABEL}]")
    print(f"Output dir       : {save_dir}")
    print(f"Pool filter      : {dataset_filter_values('AL_POOL_DATASET_FILTER') or 'none'} ({len(full_pool)} images)")
    print(f"Dev eval filter  : {dataset_filter_values('AL_DEV_EVAL_DATASET_FILTER') or 'none'} ({len(dev_eval_df)} images)")
    print(f"Dry run          : {dry_run}")
    print(f"Selection only   : {selection_only}")
    print(f"Final test       : LOCKED / NOT USED")
    print(f"Expected trainings if enabled: {expected_trainings}")
    print(f"Estimated wall-clock from gate runtime: {expected_trainings * gate_sec_per_train / 3600:.2f} hours")
    print("=" * 100)

    selected_df, cumulative_df = build_selection_plan(full_pool, seeds, strategies, initial_size, rounds, query_size, embedding_lookup)
    selected_df.to_csv(save_dir / "all_selected_samples_by_round.csv", index=False, encoding="utf-8-sig")
    cumulative_df.to_csv(save_dir / "cumulative_labeled_sets_by_round.csv", index=False, encoding="utf-8-sig")
    compute_visual_redundancy(selected_df, embedding_lookup).to_csv(save_dir / "visual_redundancy_by_round.csv", index=False, encoding="utf-8-sig")
    compute_distance_to_labeled(selected_df, embedding_lookup).to_csv(save_dir / "selected_sample_distance_to_labeled.csv", index=False, encoding="utf-8-sig")
    dataset_distribution_by_round(cumulative_df).to_csv(save_dir / "dataset_distribution_by_round.csv", index=False, encoding="utf-8-sig")
    selection_overlap_matrix(cumulative_df).to_csv(save_dir / "selection_overlap_matrix.csv", index=False, encoding="utf-8-sig")
    actual_stats_df, actual_class_df = actual_stats_by_round(cumulative_df)
    actual_stats_df.to_csv(save_dir / "actual_instance_statistics_by_round.csv", index=False, encoding="utf-8-sig")
    actual_class_df.to_csv(save_dir / "actual_class_distribution_by_round.csv", index=False, encoding="utf-8-sig")

    existing = load_existing_results(save_dir)
    success_map = existing_success_map(existing)
    results_rows: list[dict[str, Any]] = existing.to_dict("records") if len(existing) else []
    runtime_rows = []
    build_logs = []
    per_class_rows = []
    failed_rows = []

    if selection_only:
        for seed in seeds:
            training_seed = 1000 + seed
            for strategy in strategies:
                for round_idx, budget in enumerate(budgets):
                    row = {
                        "acquisition_seed": seed,
                        "training_seed": training_seed,
                        "strategy": strategy,
                        "round": round_idx,
                        "labeled_budget": budget,
                        "development_eval_size": len(dev_eval_df),
                        "yaml_path": None,
                        "map50": np.nan,
                        "map5095": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "train_status": "selection_only",
                        "train_run_dir": None,
                        "error": None,
                    }
                    results_rows.append(row)
        results_df = pd.DataFrame(results_rows)
    else:
        for seed in seeds:
            training_seed = 1000 + seed
            shared_round0_result: dict[str, Any] | None = None
            shared_round0_runtime: dict[str, Any] | None = None
            shared_round0_yaml: Path | None = None
            shared_round0_build_log: pd.DataFrame | None = None

            # Shared round 0: train once for this acquisition seed.
            round0_key = result_key(seed, "__SHARED_ROUND0__", 0)
            round0_labeled = selected_set_for(cumulative_df, seed, RANDOM_STRATEGY, 0)
            round0_dataset_dir = dataset_root / f"seed_{seed}" / "__SHARED_ROUND0__" / "round_0"
            if round0_key in success_map:
                shared_round0_result = success_map[round0_key]
                shared_round0_yaml = Path(str(shared_round0_result.get("yaml_path"))) if shared_round0_result.get("yaml_path") else None
            else:
                build_t0 = time.perf_counter()
                yaml_path, build_log = v6.build_yolo_dataset(round0_labeled, dev_eval_df, round0_dataset_dir)
                dataset_build_sec = time.perf_counter() - build_t0
                build_log.insert(0, "strategy", "__SHARED_ROUND0__")
                build_log.insert(1, "acquisition_seed", seed)
                build_log.insert(2, "training_seed", training_seed)
                build_log.insert(3, "round", 0)
                build_log["dataset_build_sec"] = dataset_build_sec
                build_logs.append(build_log)
                shared_round0_yaml = yaml_path
                shared_round0_result, shared_round0_runtime = train_eval(
                    yaml_path, save_dir, "__SHARED_ROUND0__", seed, training_seed, 0
                )
                shared_round0_runtime["dataset_build_sec"] = dataset_build_sec
                shared_round0_runtime["total_sec"] = dataset_build_sec + float(shared_round0_runtime.get("train_eval_sec", 0.0))
                runtime_rows.append(shared_round0_runtime)
                shared_row = {
                    "acquisition_seed": seed,
                    "training_seed": training_seed,
                    "strategy": "__SHARED_ROUND0__",
                    "round": 0,
                    "labeled_budget": initial_size,
                    "development_eval_size": len(dev_eval_df),
                    "yaml_path": str(yaml_path),
                    **shared_round0_result,
                    "retry_count": 0,
                }
                results_rows.append(shared_row)
                append_run_registry(save_dir / "experiment_registry.csv", shared_row, config)

            for strategy in strategies:
                for round_idx, budget in enumerate(budgets):
                    key = result_key(seed, strategy, round_idx)
                    if key in success_map:
                        continue
                    if round_idx == 0:
                        row = {
                            "acquisition_seed": seed,
                            "training_seed": training_seed,
                            "strategy": strategy,
                            "round": 0,
                            "labeled_budget": initial_size,
                            "development_eval_size": len(dev_eval_df),
                            "yaml_path": str(shared_round0_yaml) if shared_round0_yaml else None,
                            **{k: shared_round0_result.get(k) for k in shared_round0_result.keys()},
                            "train_status": "shared_baseline" if shared_round0_result else "failed",
                            "retry_count": 0,
                        }
                        results_rows.append(row)
                        append_run_registry(save_dir / "experiment_registry.csv", row, config)
                        continue

                    labeled_df = selected_set_for(cumulative_df, seed, strategy, round_idx)
                    dataset_dir = dataset_root / f"seed_{seed}" / strategy / f"round_{round_idx}"
                    build_t0 = time.perf_counter()
                    yaml_path, build_log = v6.build_yolo_dataset(labeled_df, dev_eval_df, dataset_dir)
                    dataset_build_sec = time.perf_counter() - build_t0
                    build_log.insert(0, "strategy", strategy)
                    build_log.insert(1, "acquisition_seed", seed)
                    build_log.insert(2, "training_seed", training_seed)
                    build_log.insert(3, "round", round_idx)
                    build_log["dataset_build_sec"] = dataset_build_sec
                    build_logs.append(build_log)

                    result, runtime = train_eval(yaml_path, save_dir, strategy, seed, training_seed, round_idx)
                    runtime["dataset_build_sec"] = dataset_build_sec
                    runtime["total_sec"] = dataset_build_sec + float(runtime.get("train_eval_sec", 0.0))
                    runtime_rows.append(runtime)
                    row = {
                        "acquisition_seed": seed,
                        "training_seed": training_seed,
                        "strategy": strategy,
                        "round": round_idx,
                        "labeled_budget": budget,
                        "development_eval_size": len(dev_eval_df),
                        "yaml_path": str(yaml_path),
                        **result,
                        "retry_count": 0,
                    }
                    results_rows.append(row)
                    per_class_rows.append(extract_per_class_metrics(result, strategy, seed, training_seed, round_idx))
                    if result.get("train_status") == "failed":
                        failed_rows.append(row)
                    pd.DataFrame(results_rows).to_csv(save_dir / "all_round_results.csv", index=False, encoding="utf-8-sig")
                    append_run_registry(save_dir / "experiment_registry.csv", row, config)

        results_df = pd.DataFrame(results_rows)

    results_df = results_df[results_df["strategy"].isin(strategies)].copy()
    results_df = results_df.sort_values(["acquisition_seed", "strategy", "round"], kind="mergesort")
    seed_summary = seed_strategy_summary(results_df)
    aggregate_df = aggregate_summary(seed_summary)
    paired_df = paired_strategy_comparisons(seed_summary)
    roundwise_df = paired_roundwise_differences(results_df)
    aulc_df = normalized_aulc_summary(seed_summary)
    budget_df = budget_to_target(results_df)

    results_df.to_csv(save_dir / "all_round_results.csv", index=False, encoding="utf-8-sig")
    seed_summary.to_csv(save_dir / "seed_strategy_metric_summary.csv", index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(save_dir / "aggregate_strategy_metric_summary.csv", index=False, encoding="utf-8-sig")
    paired_df.to_csv(save_dir / "paired_strategy_comparisons.csv", index=False, encoding="utf-8-sig")
    roundwise_df.to_csv(save_dir / "paired_roundwise_differences.csv", index=False, encoding="utf-8-sig")
    aulc_df.to_csv(save_dir / "normalized_aulc_summary.csv", index=False, encoding="utf-8-sig")
    budget_df.to_csv(save_dir / "budget_to_target.csv", index=False, encoding="utf-8-sig")
    pd.concat(build_logs, ignore_index=True).to_csv(save_dir / "dataset_build_log.csv", index=False, encoding="utf-8-sig") if build_logs else pd.DataFrame().to_csv(save_dir / "dataset_build_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runtime_rows).to_csv(save_dir / "runtime_profile.csv", index=False, encoding="utf-8-sig")
    pd.concat(per_class_rows, ignore_index=True).to_csv(save_dir / "per_class_metrics_by_round.csv", index=False, encoding="utf-8-sig") if per_class_rows else pd.DataFrame().to_csv(save_dir / "per_class_metrics_by_round.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failed_rows).to_csv(save_dir / "failed_or_retried_runs.csv", index=False, encoding="utf-8-sig")

    write_summary(save_dir, config, results_df, seed_summary, aggregate_df, paired_df, budget_df)
    run_analysis_if_available(save_dir)

    print("=" * 100)
    print(f"[DONE] {EXPERIMENT_LABEL} runner finished")
    print(f"Output dir: {save_dir}")
    print(f"Final test used: {config['final_test_used']}")
    print("=" * 100)


if __name__ == "__main__":
    main()

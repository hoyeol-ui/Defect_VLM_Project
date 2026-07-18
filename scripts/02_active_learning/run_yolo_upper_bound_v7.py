"""
Phase-1 YOLO upper-bound experiments for the active-learning methodology audit.

This runner answers a narrow question:

    "Is low mAP caused by the tiny AL budget, or is the dataset/label pipeline
    itself unable to support high mAP?"

It defaults to dry-run mode for safety.  Set AL_DRY_RUN_ONLY=0 when you are
ready to spend GPU time.

Output:
    runs/yolo_upper_bound_v7/upper_bound_YYYYMMDD_HHMMSS/
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402
from audit_detection_pipeline_v7 import (  # noqa: E402
    IMAGE_EXTENSIONS,
    build_fixed_external_eval_df_fast,
    build_image_index,
    build_xml_index,
    load_priority_scores,
    parse_bool_env,
    parse_int_list_env,
)

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - import-time environment guard
    raise ImportError("Install ultralytics before running upper-bound experiments.") from exc


PROJECT_ROOT = v6.PROJECT_ROOT
DATA_ROOT = v6.DATA_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "yolo_upper_bound_v7"
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "yolo_upper_bound_v7"


def parse_csv_env(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return default
    return [v.strip() for v in value.split(",") if v.strip()]


def stratified_split(df: pd.DataFrame, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    val_parts = []
    df = v6.stable_sample_order(df)
    group_cols = ["dataset_type", "class_hint"] if "dataset_type" in df.columns else ["class_hint"]
    for _, sub in df.groupby(group_cols, sort=True, dropna=False):
        sub = v6.stable_sample_order(sub)
        if len(sub) <= 1:
            train_parts.append(sub)
            continue
        n_val = max(1, int(round(len(sub) * val_ratio)))
        n_val = min(n_val, len(sub) - 1)
        val = sub.sample(n=n_val, random_state=seed + len(val_parts) * 101)
        train = sub.drop(index=val.index)
        train_parts.append(train)
        val_parts.append(val)
    train_df = pd.concat(train_parts, ignore_index=False) if train_parts else df.iloc[0:0]
    val_df = pd.concat(val_parts, ignore_index=False) if val_parts else df.iloc[0:0]
    return v6.stable_sample_order(train_df).reset_index(drop=True), v6.stable_sample_order(val_df).reset_index(drop=True)


def enumerate_dataset_manifest(dataset_type: str) -> pd.DataFrame:
    root = DATA_ROOT / dataset_type
    rows = []
    if not root.exists():
        return pd.DataFrame(columns=["image_name", "dataset_type", "image_path", "class_hint"])
    for image_path in root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        row = {
            "image_name": image_path.name,
            "dataset_type": dataset_type,
            "image_path": str(image_path.resolve()),
        }
        row["class_hint"] = v6.infer_class_hint(pd.Series(row))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["image_name", "dataset_type", "image_path", "class_hint"])
    return v6.stable_sample_order(pd.DataFrame(rows)).reset_index(drop=True)


def get_device():
    override = os.environ.get("AL_YOLO_DEVICE")
    if override is not None and override.strip() != "":
        return override.strip()
    return v6.get_device()


def train_eval(
    yaml_path: Path,
    run_output_dir: Path,
    experiment: str,
    seed: int,
) -> dict:
    dry_run = parse_bool_env("AL_DRY_RUN_ONLY", True)
    epochs = int(os.environ.get("AL_UPPER_BOUND_EPOCHS", "100"))
    patience = int(os.environ.get("AL_UPPER_BOUND_PATIENCE", "20"))
    imgsz = int(os.environ.get("AL_IMGSZ", "640"))
    batch = os.environ.get("AL_BATCH_SIZE", "8")
    workers = int(os.environ.get("AL_WORKERS", "4"))
    model_name = os.environ.get("AL_YOLO_MODEL_NAME", "yolov8n.pt")
    plots = parse_bool_env("AL_YOLO_PLOTS", False)
    cache_env = os.environ.get("AL_YOLO_CACHE", "false").strip().lower()
    cache_value: bool | str = cache_env if cache_env in {"ram", "disk"} else parse_bool_env("AL_YOLO_CACHE", False)

    if dry_run:
        return {
            "map50": np.nan,
            "map5095": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "train_status": "dry_run",
            "train_run_dir": None,
            "error": None,
        }

    try:
        model = YOLO(model_name)
        train_results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=int(batch) if str(batch).isdigit() else batch,
            workers=workers,
            device=get_device(),
            project=str(run_output_dir / "yolo_train_runs"),
            name=f"{experiment}_seed{seed}",
            exist_ok=True,
            patience=patience,
            cache=cache_value,
            plots=plots,
            verbose=False,
            seed=seed,
        )
        train_run_dir = Path(getattr(train_results, "save_dir", "")) if getattr(train_results, "save_dir", None) else None
        best_weight = train_run_dir / "weights" / "best.pt" if train_run_dir else None
        eval_model = YOLO(str(best_weight)) if best_weight and best_weight.exists() else model
        metrics = eval_model.val(
            data=str(yaml_path),
            imgsz=imgsz,
            batch=int(batch) if str(batch).isdigit() else batch,
            workers=workers,
            device=get_device(),
            split="val",
            verbose=False,
        )
        return {
            "map50": round(float(metrics.box.map50), 6),
            "map5095": round(float(metrics.box.map), 6),
            "precision": round(float(metrics.box.mp), 6),
            "recall": round(float(metrics.box.mr), 6),
            "train_status": "success",
            "train_run_dir": str(train_run_dir) if train_run_dir else None,
            "error": None,
        }
    except Exception as exc:
        traceback.print_exc()
        return {
            "map50": np.nan,
            "map5095": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "train_status": "failed",
            "train_run_dir": None,
            "error": str(exc),
        }


def make_aggregate(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df
    return (
        results_df.groupby("experiment", dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            train_size_mean=("train_size", "mean"),
            val_size_mean=("val_size", "mean"),
            map50_mean=("map50", "mean"),
            map50_std=("map50", "std"),
            map5095_mean=("map5095", "mean"),
            map5095_std=("map5095", "std"),
        )
        .reset_index()
    )


def write_summary(save_dir: Path, config: dict, results_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> None:
    lines = [
        "# YOLO Upper Bound V7",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Aggregate results",
        "",
        aggregate_df.to_markdown(index=False) if len(aggregate_df) else "_No rows._",
        "",
        "## Interpretation rule",
        "",
        "- If full-data mAP50 is still below about 0.30, prioritize pipeline/data/class-mapping investigation.",
        "- If full-data mAP50 is materially higher while 35-sample AL remains low, interpret V6 mainly as a low-budget regime.",
        "- If NEU-only is high but mixed/current-pool is low, suspect domain mixing and GC10 pool skew.",
        "",
    ]
    (save_dir / "upper_bound_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"upper_bound_{timestamp}"
    dataset_root = DATASETS_ROOT / f"upper_bound_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list_env("AL_SEEDS", [42])
    experiments = parse_csv_env("AL_UPPER_BOUND_EXPERIMENTS", ["priority_pool_full"])
    val_ratio = float(os.environ.get("AL_UPPER_BOUND_VAL_RATIO", "0.20"))

    priority_csv, pool_df = load_priority_scores()
    image_index = build_image_index()
    xml_index = build_xml_index()
    fixed_eval_df = build_fixed_external_eval_df_fast(pool_df, image_index, xml_index)

    print("=" * 100)
    print("[PHASE 1] YOLO upper-bound V7")
    print(f"Output dir: {save_dir}")
    print(f"Dry run   : {parse_bool_env('AL_DRY_RUN_ONLY', True)}")
    print(f"Experiments: {experiments}")
    print("=" * 100)

    rows = []
    build_logs = []
    per_dataset_rows = []

    for seed in seeds:
        for experiment in experiments:
            if experiment == "priority_pool_full":
                train_df = pool_df.copy()
                val_df = fixed_eval_df.copy()
            elif experiment == "neu_full":
                manifest = enumerate_dataset_manifest("NEU-DET")
                train_df, val_df = stratified_split(manifest, val_ratio=val_ratio, seed=seed)
            elif experiment == "gc10_full":
                manifest = enumerate_dataset_manifest("GC10-DET")
                train_df, val_df = stratified_split(manifest, val_ratio=val_ratio, seed=seed)
            else:
                raise ValueError(f"Unknown upper-bound experiment: {experiment}")

            if train_df.empty or val_df.empty:
                rows.append(
                    {
                        "seed": seed,
                        "experiment": experiment,
                        "train_size": len(train_df),
                        "val_size": len(val_df),
                        "train_status": "skipped_empty_split",
                        "map50": np.nan,
                        "map5095": np.nan,
                    }
                )
                continue

            exp_dataset_dir = dataset_root / experiment / f"seed_{seed}"
            yaml_path, build_log = v6.build_yolo_dataset(train_df, val_df, exp_dataset_dir)
            build_log.insert(0, "seed", seed)
            build_log.insert(1, "experiment", experiment)
            build_logs.append(build_log)

            for split_name, split_df in [("train", train_df), ("val", val_df)]:
                dist = v6.class_distribution(split_df)
                dist.insert(0, "seed", seed)
                dist.insert(1, "experiment", experiment)
                dist.insert(2, "split", split_name)
                per_dataset_rows.append(dist)

            result = train_eval(yaml_path, save_dir, experiment=experiment, seed=seed)
            rows.append(
                {
                    "seed": seed,
                    "experiment": experiment,
                    "train_size": len(train_df),
                    "val_size": len(val_df),
                    "yaml_path": str(yaml_path),
                    **result,
                }
            )

            pd.DataFrame(rows).to_csv(save_dir / "all_upper_bound_results.csv", index=False, encoding="utf-8-sig")

    results_df = pd.DataFrame(rows)
    aggregate_df = make_aggregate(results_df)
    per_dataset_df = pd.concat(per_dataset_rows, ignore_index=True) if per_dataset_rows else pd.DataFrame()
    build_log_df = pd.concat(build_logs, ignore_index=True) if build_logs else pd.DataFrame()

    config = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "priority_csv": str(priority_csv),
        "seeds": seeds,
        "experiments": experiments,
        "dry_run": parse_bool_env("AL_DRY_RUN_ONLY", True),
        "epochs": int(os.environ.get("AL_UPPER_BOUND_EPOCHS", "100")),
        "patience": int(os.environ.get("AL_UPPER_BOUND_PATIENCE", "20")),
        "fixed_eval_size": len(fixed_eval_df),
    }
    (save_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    results_df.to_csv(save_dir / "all_upper_bound_results.csv", index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(save_dir / "aggregate_upper_bound_results.csv", index=False, encoding="utf-8-sig")
    per_dataset_df.to_csv(save_dir / "per_dataset_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["seed", "experiment", "class_name", "ap50", "ap5095"]).to_csv(
        save_dir / "per_class_ap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_log_df.to_csv(save_dir / "dataset_build_log.csv", index=False, encoding="utf-8-sig")
    write_summary(save_dir, config, results_df, aggregate_df)

    print("=" * 100)
    print(f"[DONE] Upper-bound output dir: {save_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()

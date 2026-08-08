"""
Phase-2 training variance experiment for V6 selected labeled sets.

This runner freezes the selected 35-image labeled sets from a completed V6 run
and retrains YOLO multiple times with different training seeds.  It estimates
whether acquisition-strategy differences are larger than ordinary YOLO training
noise.

It defaults to dry-run mode for safety.  Set AL_DRY_RUN_ONLY=0 when ready.

Output:
    runs/training_variance_v7/training_variance_YYYYMMDD_HHMMSS/
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
from audit_detection_pipeline_v7 import parse_bool_env, parse_int_list_env  # noqa: E402

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    raise ImportError("Install ultralytics before running training variance experiments.") from exc


PROJECT_ROOT = v6.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "training_variance_v7"
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "training_variance_v7"


def parse_csv_env(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return default
    return [v.strip() for v in value.split(",") if v.strip()]


def latest_v6_run_dir() -> Path:
    override = os.environ.get("V6_RUN_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    root = PROJECT_ROOT / "runs" / "active_learning_ablation_v6_deficit_diversity"
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("al_ablation_v6_deficit_diversity_")]
    if not runs:
        raise FileNotFoundError("No V6 run directory found. Set V6_RUN_DIR explicitly.")
    return max(runs, key=lambda p: p.stat().st_mtime)


def get_labeled_set(selected_df: pd.DataFrame, strategy: str, final_round: int) -> pd.DataFrame:
    sub = selected_df[
        (selected_df["strategy"].astype(str) == strategy)
        & (pd.to_numeric(selected_df["round"], errors="coerce") <= final_round)
    ].copy()
    if sub.empty:
        raise ValueError(f"No selected samples found for strategy={strategy}")
    sub = sub.drop_duplicates(subset=["dataset_type", "image_name"], keep="first")
    return v6.stable_sample_order(sub).reset_index(drop=True)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 123) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def get_device():
    override = os.environ.get("AL_YOLO_DEVICE")
    if override is not None and override.strip() != "":
        return override.strip()
    return v6.get_device()


def train_eval(
    yaml_path: Path,
    save_dir: Path,
    strategy: str,
    training_seed: int,
) -> dict:
    dry_run = parse_bool_env("AL_DRY_RUN_ONLY", True)
    epochs = int(os.environ.get("AL_VARIANCE_EPOCHS", "50"))
    patience_env = os.environ.get("AL_VARIANCE_PATIENCE", "-1")
    patience = int(patience_env)
    if patience < 0:
        patience = epochs
    imgsz = int(os.environ.get("AL_IMGSZ", "640"))
    batch = int(os.environ.get("AL_BATCH_SIZE", "8"))
    workers = int(os.environ.get("AL_WORKERS", "4"))
    model_name = os.environ.get("AL_YOLO_MODEL_NAME", "yolov8n.pt")
    plots = parse_bool_env("AL_YOLO_PLOTS", False)

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
            batch=batch,
            workers=workers,
            device=get_device(),
            project=str(save_dir / "yolo_train_runs"),
            name=f"{strategy}_trainseed{training_seed}",
            exist_ok=True,
            patience=patience,
            cache=False,
            plots=plots,
            verbose=False,
            seed=training_seed,
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


def make_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, sub in results_df.groupby("strategy", dropna=False):
        for metric in ["map50", "map5095"]:
            values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
            lo, hi = bootstrap_ci(values)
            rows.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "num_training_seeds": int(np.isfinite(values).sum()),
                    "mean": float(np.nanmean(values)) if np.isfinite(values).any() else np.nan,
                    "std": float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() >= 2 else np.nan,
                    "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi,
                }
            )
    return pd.DataFrame(rows)


def make_signal_to_noise(summary_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("GTFreeConsistency", "GTFreeRandom"),
        ("GTFreeDatasetBalancedConsistency", "GTFreeRandom"),
        ("GTFreeDatasetBalancedConsistency", "GTFreeConsistency"),
    ]
    rows = []
    for metric in ["map50", "map5095"]:
        metric_df = summary_df[summary_df["metric"].eq(metric)].set_index("strategy")
        for treatment, baseline in comparisons:
            if treatment not in metric_df.index or baseline not in metric_df.index:
                continue
            t = metric_df.loc[treatment]
            b = metric_df.loc[baseline]
            std_values = np.asarray([float(t["std"]), float(b["std"])], dtype=float)
            finite_std = std_values[np.isfinite(std_values)]
            pooled_std = float(np.sqrt(np.mean(finite_std ** 2))) if len(finite_std) else np.nan
            diff = float(t["mean"]) - float(b["mean"]) if pd.notna(t["mean"]) and pd.notna(b["mean"]) else np.nan
            signal_to_noise = abs(diff) / pooled_std if np.isfinite(diff) and np.isfinite(pooled_std) and pooled_std > 0 else np.nan
            rows.append(
                {
                    "metric": metric,
                    "treatment": treatment,
                    "baseline": baseline,
                    "mean_difference": diff,
                    "pooled_training_std": pooled_std,
                    "signal_to_training_noise": signal_to_noise,
                    "interpretation": (
                        "not_available_in_dry_run_or_single_seed"
                        if not np.isfinite(signal_to_noise)
                        else (
                            "training_noise_dominates"
                            if signal_to_noise < 1
                            else "strategy_signal_may_exceed_noise"
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_markdown(save_dir: Path, config: dict, summary_df: pd.DataFrame, signal_df: pd.DataFrame) -> None:
    lines = [
        "# Training Variance V7",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Per-strategy training variance",
        "",
        summary_df.to_markdown(index=False) if len(summary_df) else "_No rows._",
        "",
        "## Signal-to-training-noise",
        "",
        signal_df.to_markdown(index=False) if len(signal_df) else "_No rows._",
        "",
        "If `signal_to_training_noise < 1`, the acquisition difference is smaller than ordinary YOLO retraining variance.",
        "",
    ]
    (save_dir / "training_variance_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"training_variance_{timestamp}"
    dataset_root = DATASETS_ROOT / f"training_variance_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    v6_run_dir = latest_v6_run_dir()
    selected_path = v6_run_dir / "all_selected_samples_by_round.csv"
    eval_path = v6_run_dir / "fixed_external_evaluation_split.csv"
    if not selected_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"Missing V6 selected/eval CSVs under {v6_run_dir}")

    selected_df = pd.read_csv(selected_path)
    eval_df = pd.read_csv(eval_path)
    strategies = parse_csv_env(
        "AL_VARIANCE_STRATEGIES",
        [
            "GTFreeRandom",
            "GTFreeConsistency",
            "GTFreeDatasetBalancedConsistency",
        ],
    )
    training_seeds = parse_int_list_env("AL_TRAINING_SEEDS", [101, 102])
    final_round = int(os.environ.get("AL_VARIANCE_FINAL_ROUND", "4"))

    print("=" * 100)
    print("[PHASE 2] Training variance V7")
    print(f"V6 run dir: {v6_run_dir}")
    print(f"Output dir: {save_dir}")
    print(f"Dry run   : {parse_bool_env('AL_DRY_RUN_ONLY', True)}")
    print("=" * 100)

    result_rows = []
    build_logs = []
    labeled_set_rows = []
    for strategy in strategies:
        labeled_df = get_labeled_set(selected_df, strategy, final_round=final_round)
        labeled_dump = labeled_df.copy()
        labeled_dump.insert(0, "source_strategy", strategy)
        labeled_set_rows.append(labeled_dump)

        for training_seed in training_seeds:
            dataset_dir = dataset_root / strategy / f"train_seed_{training_seed}"
            yaml_path, build_log = v6.build_yolo_dataset(labeled_df, eval_df, dataset_dir)
            build_log.insert(0, "strategy", strategy)
            build_log.insert(1, "training_seed", training_seed)
            build_logs.append(build_log)

            result = train_eval(yaml_path, save_dir, strategy=strategy, training_seed=training_seed)
            result_rows.append(
                {
                    "strategy": strategy,
                    "training_seed": training_seed,
                    "labeled_size": len(labeled_df),
                    "val_size": len(eval_df),
                    "yaml_path": str(yaml_path),
                    **result,
                }
            )
            pd.DataFrame(result_rows).to_csv(save_dir / "training_variance_results.csv", index=False, encoding="utf-8-sig")

    results_df = pd.DataFrame(result_rows)
    summary_df = make_summary(results_df)
    signal_df = make_signal_to_noise(summary_df)
    build_log_df = pd.concat(build_logs, ignore_index=True) if build_logs else pd.DataFrame()
    labeled_sets_df = pd.concat(labeled_set_rows, ignore_index=True) if labeled_set_rows else pd.DataFrame()

    config = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "v6_run_dir": str(v6_run_dir),
        "strategies": strategies,
        "training_seeds": training_seeds,
        "final_round": final_round,
        "dry_run": parse_bool_env("AL_DRY_RUN_ONLY", True),
        "epochs": int(os.environ.get("AL_VARIANCE_EPOCHS", "50")),
        "patience": os.environ.get("AL_VARIANCE_PATIENCE", "-1"),
    }
    (save_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    results_df.to_csv(save_dir / "training_variance_results.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(save_dir / "training_variance_metric_summary.csv", index=False, encoding="utf-8-sig")
    signal_df.to_csv(save_dir / "signal_to_training_noise.csv", index=False, encoding="utf-8-sig")
    build_log_df.to_csv(save_dir / "dataset_build_log.csv", index=False, encoding="utf-8-sig")
    labeled_sets_df.to_csv(save_dir / "frozen_labeled_sets.csv", index=False, encoding="utf-8-sig")
    write_markdown(save_dir, config, summary_df, signal_df)

    print("=" * 100)
    print(f"[DONE] Training variance output dir: {save_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()

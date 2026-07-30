"""Confirmatory 100-epoch retraining of fixed V6 final labeled sets.

Purpose:
    Check whether the 20-epoch V6 ordering was an under-training artifact and
    whether strategy ranking is stable under longer training.

Default behavior is dry-run.  Set AL_DRY_RUN_ONLY=0 to actually train.
"""

from __future__ import annotations

import json
import os
import sys
import time
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
from experiment_registry_v7 import append_registry_row  # noqa: E402
from run_training_variance_v7 import (  # noqa: E402
    bootstrap_ci,
    get_labeled_set,
    latest_v6_run_dir,
)

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    raise ImportError("Install ultralytics before running confirmatory training.") from exc


PROJECT_ROOT = v6.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "confirmatory_training_v7"
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "confirmatory_training_v7"


def parse_csv_env(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return default
    return [v.strip() for v in value.split(",") if v.strip()]


def latest_eval_protocol_dir() -> Path | None:
    override = os.environ.get("EVAL_PROTOCOL_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    root = PROJECT_ROOT / "runs" / "evaluation_protocol_v7"
    runs = [p for p in root.glob("eval_protocol_*") if p.is_dir()] if root.exists() else []
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def load_development_eval() -> tuple[Path | None, pd.DataFrame]:
    protocol_dir = latest_eval_protocol_dir()
    if protocol_dir is not None:
        dev_path = protocol_dir / "development_eval_v7.csv"
        if dev_path.exists():
            return protocol_dir, pd.read_csv(dev_path)

    # Safe fallback: use latest V6 fixed eval as development eval only.
    v6_run = latest_v6_run_dir()
    dev_path = v6_run / "fixed_external_evaluation_split.csv"
    if not dev_path.exists():
        raise FileNotFoundError("development_eval_v7.csv not found and no V6 fixed eval fallback exists.")
    return None, pd.read_csv(dev_path)


def get_device():
    override = os.environ.get("AL_YOLO_DEVICE")
    if override is not None and override.strip():
        return override.strip()
    return v6.get_device()


def read_best_epochs(train_run_dir: Path | None) -> dict:
    if train_run_dir is None:
        return {}
    results_path = train_run_dir / "results.csv"
    if not results_path.exists():
        return {}
    try:
        df = pd.read_csv(results_path)
        df.columns = [c.strip() for c in df.columns]
        map50_col = [c for c in df.columns if "metrics/mAP50(B)" in c and "50-95" not in c]
        map95_col = [c for c in df.columns if "metrics/mAP50-95(B)" in c]
        out = {
            "actual_trained_epochs": len(df),
            "final_epoch": int(df["epoch"].iloc[-1]) if "epoch" in df.columns else len(df),
        }
        if map50_col:
            idx = df[map50_col[0]].idxmax()
            out["best_epoch_map50"] = int(df.loc[idx, "epoch"]) if "epoch" in df.columns else int(idx)
            out["best_results_map50"] = float(df.loc[idx, map50_col[0]])
            out["last_results_map50"] = float(df[map50_col[0]].iloc[-1])
        if map95_col:
            idx = df[map95_col[0]].idxmax()
            out["best_epoch_map5095"] = int(df.loc[idx, "epoch"]) if "epoch" in df.columns else int(idx)
            out["best_results_map5095"] = float(df.loc[idx, map95_col[0]])
            out["last_results_map5095"] = float(df[map95_col[0]].iloc[-1])
        return out
    except Exception as exc:
        return {"results_read_error": str(exc)}


def train_eval(yaml_path: Path, save_dir: Path, strategy: str, training_seed: int) -> tuple[dict, dict]:
    dry_run = parse_bool_env("AL_DRY_RUN_ONLY", True)
    epochs = int(os.environ.get("AL_CONFIRM_EPOCHS", "100"))
    patience = int(os.environ.get("AL_CONFIRM_PATIENCE", str(epochs)))
    imgsz = int(os.environ.get("AL_IMGSZ", "640"))
    batch = int(os.environ.get("AL_BATCH_SIZE", "8"))
    workers = int(os.environ.get("AL_WORKERS", "4"))
    model_name = os.environ.get("AL_YOLO_MODEL_NAME", "yolov8n.pt")
    plots = parse_bool_env("AL_YOLO_PLOTS", False)
    cache_env = os.environ.get("AL_YOLO_CACHE", "false").strip().lower()
    cache_value: bool | str = cache_env if cache_env in {"ram", "disk"} else parse_bool_env("AL_YOLO_CACHE", False)

    runtime = {
        "stage": "confirmatory_training",
        "strategy": strategy,
        "training_seed": training_seed,
        "batch": batch,
        "workers": workers,
        "cache": cache_value,
        "amp": True,
        "compile": False,
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
            {**runtime, "dataset_build_sec": np.nan, "train_eval_sec": 0.0, "total_sec": 0.0},
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
            name=f"{strategy}_trainseed{training_seed}",
            exist_ok=True,
            patience=patience,
            cache=cache_value,
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
        result = {
            "map50": round(float(metrics.box.map50), 6),
            "map5095": round(float(metrics.box.map), 6),
            "precision": round(float(metrics.box.mp), 6),
            "recall": round(float(metrics.box.mr), 6),
            "train_status": "success",
            "train_run_dir": str(train_run_dir) if train_run_dir else None,
            "error": None,
            **read_best_epochs(train_run_dir),
        }
        runtime.update({"train_eval_sec": time.perf_counter() - t0, "total_sec": time.perf_counter() - t0})
        return result, runtime
    except Exception as exc:
        traceback.print_exc()
        runtime.update({"train_eval_sec": time.perf_counter() - t0, "total_sec": time.perf_counter() - t0})
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
            runtime,
        )


def metric_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, sub in results_df.groupby("strategy", dropna=False):
        for metric in ["map50", "map5095"]:
            values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            lo, hi = bootstrap_ci(finite) if len(finite) else (np.nan, np.nan)
            rows.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "num_training_seeds": len(finite),
                    "mean": float(np.mean(finite)) if len(finite) else np.nan,
                    "std": float(np.std(finite, ddof=1)) if len(finite) >= 2 else np.nan,
                    "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi,
                }
            )
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


def paired_differences(results_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("GTFreeConsistency", "GTFreeRandom"),
        ("GTFreeDatasetBalancedConsistency", "GTFreeRandom"),
        ("GTFreeDatasetBalancedConsistency", "GTFreeConsistency"),
    ]
    rows = []
    for treatment, baseline in comparisons:
        left = results_df[results_df["strategy"].eq(treatment)].set_index("training_seed")
        right = results_df[results_df["strategy"].eq(baseline)].set_index("training_seed")
        common = left.index.intersection(right.index)
        for metric in ["map50", "map5095"]:
            diff = (left.loc[common, metric] - right.loc[common, metric]).to_numpy(dtype=float)
            finite = diff[np.isfinite(diff)]
            lo, hi = bootstrap_ci(finite) if len(finite) else (np.nan, np.nan)
            rows.append(
                {
                    "treatment": treatment,
                    "baseline": baseline,
                    "metric": metric,
                    "num_pairs": len(finite),
                    "mean_paired_difference": float(np.mean(finite)) if len(finite) else np.nan,
                    "std_paired_difference": float(np.std(finite, ddof=1)) if len(finite) >= 2 else np.nan,
                    "wins": int(np.sum(finite > 0)),
                    "ties": int(np.sum(finite == 0)),
                    "losses": int(np.sum(finite < 0)),
                    "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi,
                    "exact_sign_flip_pvalue": exact_sign_flip_pvalue(finite),
                }
            )
    return pd.DataFrame(rows)


def write_summary(save_dir: Path, config: dict, summary_df: pd.DataFrame, paired_df: pd.DataFrame) -> None:
    lines = [
        "# Confirmatory Training V7",
        "",
        "This is still development-set confirmation, not final-test evaluation.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Strategy means",
        "",
        summary_df.to_markdown(index=False) if len(summary_df) else "_No rows._",
        "",
        "## Paired training-seed differences",
        "",
        paired_df.to_markdown(index=False) if len(paired_df) else "_No rows._",
        "",
        "Interpret exact sign-flip p-values cautiously with five or fewer seeds.",
    ]
    (save_dir / "confirmatory_training_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"confirmatory_training_{timestamp}"
    dataset_root = DATASETS_ROOT / f"confirmatory_training_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    v6_run = latest_v6_run_dir()
    selected_df = pd.read_csv(v6_run / "all_selected_samples_by_round.csv")
    protocol_dir, dev_eval_df = load_development_eval()

    strategies = parse_csv_env(
        "AL_CONFIRM_STRATEGIES",
        ["GTFreeRandom", "GTFreeConsistency", "GTFreeDatasetBalancedConsistency"],
    )
    training_seeds = parse_int_list_env("AL_TRAINING_SEEDS", [101, 102, 103, 104, 105])
    final_round = int(os.environ.get("AL_CONFIRM_FINAL_ROUND", "4"))

    config = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "v6_run_dir": str(v6_run),
        "eval_protocol_dir": str(protocol_dir) if protocol_dir else "fallback_latest_v6_fixed_eval_as_development_only",
        "strategies": strategies,
        "training_seeds": training_seeds,
        "epochs": int(os.environ.get("AL_CONFIRM_EPOCHS", "100")),
        "patience": int(os.environ.get("AL_CONFIRM_PATIENCE", os.environ.get("AL_CONFIRM_EPOCHS", "100"))),
        "dry_run": parse_bool_env("AL_DRY_RUN_ONLY", True),
        "eval_split": "development_eval_v7",
        "final_test_used": False,
    }
    (save_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("[V7] Confirmatory fixed-set training")
    print(f"Output dir: {save_dir}")
    print(f"Dry run   : {config['dry_run']}")
    print("Final test: NOT USED")
    print("=" * 100)

    results = []
    runtime_rows = []
    build_logs = []
    labeled_rows = []

    for strategy in strategies:
        labeled_df = get_labeled_set(selected_df, strategy, final_round=final_round)
        labeled_dump = labeled_df.copy()
        labeled_dump.insert(0, "source_strategy", strategy)
        labeled_rows.append(labeled_dump)

        for training_seed in training_seeds:
            dataset_dir = dataset_root / strategy / f"train_seed_{training_seed}"
            build_t0 = time.perf_counter()
            yaml_path, build_log = v6.build_yolo_dataset(labeled_df, dev_eval_df, dataset_dir)
            dataset_build_sec = time.perf_counter() - build_t0
            build_log.insert(0, "strategy", strategy)
            build_log.insert(1, "training_seed", training_seed)
            build_logs.append(build_log)

            result, runtime = train_eval(yaml_path, save_dir, strategy, training_seed)
            runtime["dataset_build_sec"] = dataset_build_sec
            runtime["total_sec"] = dataset_build_sec + float(runtime.get("train_eval_sec", 0.0))
            runtime_rows.append(runtime)

            row = {
                "strategy": strategy,
                "training_seed": training_seed,
                "labeled_size": len(labeled_df),
                "development_eval_size": len(dev_eval_df),
                "yaml_path": str(yaml_path),
                **result,
            }
            results.append(row)
            pd.DataFrame(results).to_csv(save_dir / "confirmatory_training_results.csv", index=False, encoding="utf-8-sig")

            append_registry_row(
                save_dir / "experiment_registry.csv",
                project_root=PROJECT_ROOT,
                experiment_id=save_dir.name,
                stage="confirmatory_training",
                strategy=strategy,
                training_seed=training_seed,
                eval_split="development_eval_v7",
                status=row["train_status"],
                result_path=save_dir,
                hyperparameters=config,
            )

    results_df = pd.DataFrame(results)
    summary_df = metric_summary(results_df)
    paired_df = paired_differences(results_df)
    build_log_df = pd.concat(build_logs, ignore_index=True) if build_logs else pd.DataFrame()
    labeled_df_all = pd.concat(labeled_rows, ignore_index=True) if labeled_rows else pd.DataFrame()
    runtime_df = pd.DataFrame(runtime_rows)

    results_df.to_csv(save_dir / "confirmatory_training_results.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(save_dir / "confirmatory_metric_summary.csv", index=False, encoding="utf-8-sig")
    paired_df.to_csv(save_dir / "paired_training_seed_differences.csv", index=False, encoding="utf-8-sig")
    build_log_df.to_csv(save_dir / "dataset_build_log.csv", index=False, encoding="utf-8-sig")
    labeled_df_all.to_csv(save_dir / "frozen_labeled_sets.csv", index=False, encoding="utf-8-sig")
    runtime_df.to_csv(save_dir / "runtime_profile.csv", index=False, encoding="utf-8-sig")
    write_summary(save_dir, config, summary_df, paired_df)

    print("=" * 100)
    print(f"[DONE] Confirmatory training output dir: {save_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()

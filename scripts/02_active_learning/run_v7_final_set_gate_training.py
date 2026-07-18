"""V7 final-35 gate training for real visual embedding candidates.

This script is intentionally narrower than a full active-learning curve.

Purpose:
    1. Freeze the final labeled set from a V7 Stage-A screening run.
    2. Train only the selected final set on the development split.
    3. Decide whether a visual strategy deserves promotion to a full AL curve.

Default behavior is dry-run.  Set AL_DRY_RUN_ONLY=0 to run YOLO training.

Default gate:
    - acquisition seed: 42 from the screening run
    - final labeled size: 15 + 5 * 4 = 35
    - strategies:
        GTFreeRandom
        GTFreeDatasetBalancedConsistency
        GTFreeDatasetBalancedVisualDiversity
        GTFreeDatasetBalancedConsistencyVisualDiversity
    - training seeds: 101, 102
    - epochs/patience: 100/100
    - eval split: development_eval_v7 only

Final-test evaluation is never used here.
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
from run_confirmatory_training_v7 import read_best_epochs  # noqa: E402
from run_training_variance_v7 import bootstrap_ci, get_labeled_set  # noqa: E402

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    raise ImportError("Install ultralytics before running V7 gate training.") from exc


PROJECT_ROOT = v6.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "v7_final_set_gate_training"
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "v7_final_set_gate_training"

DEFAULT_GATE_STRATEGIES = [
    "GTFreeRandom",
    "GTFreeDatasetBalancedConsistency",
    "GTFreeDatasetBalancedVisualDiversity",
    "GTFreeDatasetBalancedConsistencyVisualDiversity",
]
VISUAL_STRATEGIES = [
    "GTFreeDatasetBalancedVisualDiversity",
    "GTFreeDatasetBalancedConsistencyVisualDiversity",
]
DBC_STRATEGY = "GTFreeDatasetBalancedConsistency"
RANDOM_STRATEGY = "GTFreeRandom"


def parse_csv_env(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return default
    return [v.strip() for v in value.split(",") if v.strip()]


def latest_screening_run_dir() -> Path:
    override = os.environ.get("V7_SCREENING_RUN_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    root = PROJECT_ROOT / "runs" / "active_learning_ablation_v7_visual_instance"
    runs = [p for p in root.glob("v7_visual_instance_screening_*") if p.is_dir()] if root.exists() else []
    if not runs:
        raise FileNotFoundError("No V7 screening run found. Set V7_SCREENING_RUN_DIR explicitly.")
    return max(runs, key=lambda p: p.stat().st_mtime)


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
    raise FileNotFoundError("development_eval_v7.csv not found. Run prepare_evaluation_protocol_v7.py first.")


def latest_confirmatory_run_dir() -> Path | None:
    override = os.environ.get("CONFIRMATORY_RUN_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    root = PROJECT_ROOT / "runs" / "confirmatory_training_v7"
    runs = [p for p in root.glob("confirmatory_training_*") if p.is_dir()] if root.exists() else []
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def read_confirmatory_consistency_map5095(default: float = 0.166402) -> float:
    override = os.environ.get("AL_CONFIRMATORY_CONSISTENCY_MAP5095")
    if override:
        return float(override)
    run_dir = latest_confirmatory_run_dir()
    if run_dir is None:
        return default
    summary_path = run_dir / "confirmatory_metric_summary.csv"
    if not summary_path.exists():
        return default
    summary = pd.read_csv(summary_path)
    sub = summary[
        summary["strategy"].astype(str).eq("GTFreeConsistency")
        & summary["metric"].astype(str).eq("map5095")
    ]
    if sub.empty or pd.isna(sub.iloc[0].get("mean")):
        return default
    return float(sub.iloc[0]["mean"])


def get_device() -> str:
    override = os.environ.get("AL_YOLO_DEVICE")
    if override is not None and override.strip():
        return override.strip()
    return v6.get_device()


def train_eval(yaml_path: Path, save_dir: Path, strategy: str, training_seed: int) -> tuple[dict, dict]:
    dry_run = parse_bool_env("AL_DRY_RUN_ONLY", True)
    epochs = int(os.environ.get("AL_GATE_EPOCHS", "100"))
    patience = int(os.environ.get("AL_GATE_PATIENCE", str(epochs)))
    imgsz = int(os.environ.get("AL_IMGSZ", "640"))
    batch = int(os.environ.get("AL_BATCH_SIZE", "8"))
    workers = int(os.environ.get("AL_WORKERS", "4"))
    model_name = os.environ.get("AL_YOLO_MODEL_NAME", "yolov8n.pt")
    plots = parse_bool_env("AL_YOLO_PLOTS", False)
    cache_env = os.environ.get("AL_YOLO_CACHE", "ram").strip().lower()
    cache_value: bool | str = cache_env if cache_env in {"ram", "disk"} else parse_bool_env("AL_YOLO_CACHE", False)

    runtime = {
        "stage": "v7_final_set_gate_training",
        "strategy": strategy,
        "training_seed": training_seed,
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
                    "num_training_seeds": int(len(finite)),
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


def paired_gate_comparisons(results_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        (DBC_STRATEGY, RANDOM_STRATEGY),
        ("GTFreeDatasetBalancedVisualDiversity", RANDOM_STRATEGY),
        ("GTFreeDatasetBalancedConsistencyVisualDiversity", RANDOM_STRATEGY),
        ("GTFreeDatasetBalancedVisualDiversity", DBC_STRATEGY),
        ("GTFreeDatasetBalancedConsistencyVisualDiversity", DBC_STRATEGY),
    ]
    rows = []
    for treatment, baseline in comparisons:
        left = results_df[results_df["strategy"].eq(treatment)].set_index("training_seed")
        right = results_df[results_df["strategy"].eq(baseline)].set_index("training_seed")
        common = left.index.intersection(right.index)
        for metric in ["map50", "map5095"]:
            if len(common) == 0:
                diff = np.asarray([], dtype=float)
            else:
                diff = (left.loc[common, metric] - right.loc[common, metric]).to_numpy(dtype=float)
            finite = diff[np.isfinite(diff)]
            lo, hi = bootstrap_ci(finite) if len(finite) else (np.nan, np.nan)
            rows.append(
                {
                    "treatment": treatment,
                    "baseline": baseline,
                    "metric": metric,
                    "num_pairs": int(len(finite)),
                    "mean_paired_difference": float(np.mean(finite)) if len(finite) else np.nan,
                    "wins": int(np.sum(finite > 0)),
                    "ties": int(np.sum(finite == 0)),
                    "losses": int(np.sum(finite < 0)),
                    "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi,
                    "exact_sign_flip_pvalue": exact_sign_flip_pvalue(finite),
                }
            )
    return pd.DataFrame(rows)


def screening_redundancy_summary(screening_dir: Path) -> pd.DataFrame:
    path = screening_dir / "visual_redundancy_by_round.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    col = "selected_batch_pairwise_cosine_similarity_mean"
    if col not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("strategy", dropna=False)[col]
        .mean()
        .reset_index()
        .rename(columns={col: "mean_selected_batch_pairwise_cosine_similarity"})
    )


def mean_metric(summary_df: pd.DataFrame, strategy: str, metric: str) -> float:
    sub = summary_df[summary_df["strategy"].astype(str).eq(strategy) & summary_df["metric"].astype(str).eq(metric)]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0]["mean"])


def redundancy_mean(redundancy_df: pd.DataFrame, strategy: str) -> float:
    if redundancy_df.empty:
        return np.nan
    sub = redundancy_df[redundancy_df["strategy"].astype(str).eq(strategy)]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0]["mean_selected_batch_pairwise_cosine_similarity"])


def promotion_gate_decisions(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
    old_consistency_map5095: float,
) -> pd.DataFrame:
    dbc_map95 = mean_metric(summary_df, DBC_STRATEGY, "map5095")
    random_rows = results_df[results_df["strategy"].eq(RANDOM_STRATEGY)].set_index("training_seed")
    dbc_redundancy = redundancy_mean(redundancy_df, DBC_STRATEGY)
    random_redundancy = redundancy_mean(redundancy_df, RANDOM_STRATEGY)
    rows = []
    for strategy in VISUAL_STRATEGIES:
        visual_map95 = mean_metric(summary_df, strategy, "map5095")
        visual_rows = results_df[results_df["strategy"].eq(strategy)].set_index("training_seed")
        common = visual_rows.index.intersection(random_rows.index)
        beats_random_seed_count = 0
        if len(common):
            diff = visual_rows.loc[common, "map5095"].to_numpy(dtype=float) - random_rows.loc[common, "map5095"].to_numpy(dtype=float)
            beats_random_seed_count = int(np.sum(diff > 0))
        visual_redundancy = redundancy_mean(redundancy_df, strategy)
        pass_mean_gt_dbc = bool(np.isfinite(visual_map95) and np.isfinite(dbc_map95) and visual_map95 > dbc_map95)
        pass_beats_random_seed = beats_random_seed_count >= 1
        pass_gt_old_consistency = bool(np.isfinite(visual_map95) and visual_map95 > old_consistency_map5095)
        pass_redundancy_vs_dbc = bool(
            np.isfinite(visual_redundancy) and np.isfinite(dbc_redundancy) and visual_redundancy < dbc_redundancy
        )
        pass_redundancy_vs_random = bool(
            np.isfinite(visual_redundancy) and np.isfinite(random_redundancy) and visual_redundancy < random_redundancy
        )
        passed = pass_mean_gt_dbc and pass_beats_random_seed and pass_gt_old_consistency and pass_redundancy_vs_dbc
        rows.append(
            {
                "strategy": strategy,
                "promote_to_full_al_curve": passed,
                "mean_map5095": visual_map95,
                "dbc_mean_map5095": dbc_map95,
                "old_confirmatory_consistency_map5095": old_consistency_map5095,
                "beats_random_training_seed_count": beats_random_seed_count,
                "visual_redundancy_mean": visual_redundancy,
                "dbc_redundancy_mean": dbc_redundancy,
                "random_redundancy_mean": random_redundancy,
                "pass_mean_map5095_gt_dbc": pass_mean_gt_dbc,
                "pass_beats_random_at_least_one_seed": pass_beats_random_seed,
                "pass_mean_gt_old_consistency_only": pass_gt_old_consistency,
                "pass_redundancy_reduced_vs_dbc": pass_redundancy_vs_dbc,
                "redundancy_reduced_vs_random_report_only": pass_redundancy_vs_random,
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    save_dir: Path,
    config: dict,
    metric_summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
) -> None:
    lines = [
        "# V7 Final-set Gate Training",
        "",
        "Development-set gate only. Final test was not used.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Strategy means",
        "",
        metric_summary_df.to_markdown(index=False) if len(metric_summary_df) else "_No metric summary rows._",
        "",
        "## Paired comparisons",
        "",
        paired_df.to_markdown(index=False) if len(paired_df) else "_No paired rows._",
        "",
        "## Screening visual redundancy summary",
        "",
        redundancy_df.to_markdown(index=False) if len(redundancy_df) else "_No redundancy rows._",
        "",
        "## Promotion gate decisions",
        "",
        gate_df.to_markdown(index=False) if len(gate_df) else "_No gate rows._",
        "",
        "If no visual strategy passes, keep DatasetBalancedConsistency as the main method and stop V7 promotion.",
    ]
    (save_dir / "v7_final_set_gate_training_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"v7_final_set_gate_training_{timestamp}"
    dataset_root = DATASETS_ROOT / f"v7_final_set_gate_training_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    screening_dir = latest_screening_run_dir()
    selected_path = screening_dir / "all_selected_samples_by_round.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected sample file: {selected_path}")
    selected_df = pd.read_csv(selected_path)
    protocol_dir, dev_eval_df = load_development_eval()

    strategies = parse_csv_env("AL_GATE_STRATEGIES", DEFAULT_GATE_STRATEGIES)
    training_seeds = parse_int_list_env("AL_TRAINING_SEEDS", [101, 102])
    final_round = int(os.environ.get("AL_GATE_FINAL_ROUND", "4"))
    old_consistency_map5095 = read_confirmatory_consistency_map5095()
    dry_run = parse_bool_env("AL_DRY_RUN_ONLY", True)

    config = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "screening_run_dir": str(screening_dir),
        "eval_protocol_dir": str(protocol_dir),
        "strategies": strategies,
        "training_seeds": training_seeds,
        "final_round": final_round,
        "expected_labeled_size": int(os.environ.get("AL_INITIAL_SEED_SIZE", "15")) + int(os.environ.get("AL_QUERY_SIZE", "5")) * final_round,
        "epochs": int(os.environ.get("AL_GATE_EPOCHS", "100")),
        "patience": int(os.environ.get("AL_GATE_PATIENCE", os.environ.get("AL_GATE_EPOCHS", "100"))),
        "dry_run": dry_run,
        "eval_split": "development_eval_v7",
        "final_test_used": False,
        "cache": os.environ.get("AL_YOLO_CACHE", "ram"),
        "batch": int(os.environ.get("AL_BATCH_SIZE", "8")),
        "workers": int(os.environ.get("AL_WORKERS", "4")),
        "old_confirmatory_consistency_map5095": old_consistency_map5095,
        "gate_rule": [
            "visual mean mAP50-95 > DatasetBalancedConsistency mean mAP50-95",
            "visual beats Random on at least one paired training seed",
            "visual mean mAP50-95 > old confirmatory Consistency-only mean",
            "visual selected-batch redundancy is lower than DatasetBalancedConsistency",
        ],
    }
    (save_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("[V7] Final-set gate training")
    print(f"Output dir : {save_dir}")
    print(f"Screening  : {screening_dir}")
    print(f"Dry run    : {dry_run}")
    print("Final test : NOT USED")
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

        dataset_dir = dataset_root / strategy
        build_t0 = time.perf_counter()
        yaml_path, build_log = v6.build_yolo_dataset(labeled_df, dev_eval_df, dataset_dir)
        dataset_build_sec = time.perf_counter() - build_t0
        build_log.insert(0, "strategy", strategy)
        build_log["dataset_build_sec"] = dataset_build_sec
        build_logs.append(build_log)

        for training_seed in training_seeds:
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
            pd.DataFrame(results).to_csv(save_dir / "gate_training_results.csv", index=False, encoding="utf-8-sig")
            append_registry_row(
                save_dir / "experiment_registry.csv",
                project_root=PROJECT_ROOT,
                experiment_id=save_dir.name,
                stage="v7_final_set_gate_training",
                strategy=strategy,
                training_seed=training_seed,
                eval_split="development_eval_v7",
                status=row["train_status"],
                result_path=save_dir,
                hyperparameters=config,
            )

    results_df = pd.DataFrame(results)
    metric_summary_df = metric_summary(results_df)
    paired_df = paired_gate_comparisons(results_df)
    redundancy_df = screening_redundancy_summary(screening_dir)
    gate_df = promotion_gate_decisions(results_df, metric_summary_df, redundancy_df, old_consistency_map5095)
    build_log_df = pd.concat(build_logs, ignore_index=True) if build_logs else pd.DataFrame()
    labeled_df_all = pd.concat(labeled_rows, ignore_index=True) if labeled_rows else pd.DataFrame()
    runtime_df = pd.DataFrame(runtime_rows)

    results_df.to_csv(save_dir / "gate_training_results.csv", index=False, encoding="utf-8-sig")
    metric_summary_df.to_csv(save_dir / "gate_metric_summary.csv", index=False, encoding="utf-8-sig")
    paired_df.to_csv(save_dir / "paired_gate_comparisons.csv", index=False, encoding="utf-8-sig")
    gate_df.to_csv(save_dir / "promotion_gate_decisions.csv", index=False, encoding="utf-8-sig")
    redundancy_df.to_csv(save_dir / "screening_redundancy_summary.csv", index=False, encoding="utf-8-sig")
    build_log_df.to_csv(save_dir / "dataset_build_log.csv", index=False, encoding="utf-8-sig")
    labeled_df_all.to_csv(save_dir / "frozen_labeled_sets.csv", index=False, encoding="utf-8-sig")
    runtime_df.to_csv(save_dir / "runtime_profile.csv", index=False, encoding="utf-8-sig")
    write_summary(save_dir, config, metric_summary_df, paired_df, gate_df, redundancy_df)

    print("=" * 100)
    print(f"[DONE] V7 final-set gate training output dir: {save_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()

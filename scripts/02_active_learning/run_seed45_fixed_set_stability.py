"""Seed45 fixed-labeled-set training-stability experiment.

Purpose
-------
Separate acquisition-set utility from ordinary YOLO training variance by
retraining four already-created seed45 datasets with paired training seeds:

* Initial15 (shared round0)
* Random20
* Visual20
* InstanceRich20

Safety / protocol rules
-----------------------
* No acquisition is performed.
* Only the existing development ``val`` split is evaluated.
* Any path or YAML containing ``final_test`` is rejected.
* The four validation manifests must be identical and contain 177 images.
* XML is not read by this runner.
* A completed row is skipped when the same output directory is reused.

Use ``--dry-run`` first. The dry run validates every dataset and writes the
frozen plan without importing or training an Ultralytics model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "runs" / "seed45_fixed_set_stability"
MODEL = ROOT / "yolov8n.pt"
V8_DATA = ROOT / "datasets" / "active_learning_ablation_v8_neu_only" / "v8_neu_only_20260712_105644" / "seed_45"
V9_DATA = ROOT / "datasets" / "active_learning_ablation_v9_detector_aware" / "v9b_5seed_full_curve_20260712_172305" / "seed_45"

NEU6 = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
DEFAULT_SEEDS = [2045, 2046, 2047, 2048, 2049]


@dataclass(frozen=True)
class FixedSet:
    key: str
    budget: int
    yaml_path: Path
    provenance: str


SETS = {
    "Initial15": FixedSet(
        "Initial15",
        15,
        V8_DATA / "__SHARED_ROUND0__" / "round_0" / "data.yaml",
        "V8 seed45 shared round0",
    ),
    "Random20": FixedSet(
        "Random20",
        20,
        V8_DATA / "GTFreeRandom" / "round_1" / "data.yaml",
        "V8 seed45 GTFreeRandom round1",
    ),
    "Visual20": FixedSet(
        "Visual20",
        20,
        V8_DATA / "GTFreeDatasetBalancedVisualDiversity" / "round_1" / "data.yaml",
        "V8 seed45 GTFreeDatasetBalancedVisualDiversity round1",
    ),
    "InstanceRich20": FixedSet(
        "InstanceRich20",
        20,
        V9_DATA / "DetectorInstanceRichDINOBalanced" / "round_1" / "data.yaml",
        "V9b seed45 DetectorInstanceRichDINOBalanced round1; V9b source is V8",
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def yaml_scalar(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip().strip("'\"")
    raise ValueError(f"Missing {key!r} in data YAML")


def dataset_dirs(yaml_path: Path) -> tuple[Path, Path]:
    text = yaml_path.read_text(encoding="utf-8")
    root_text = yaml_scalar(text, "path")
    dataset_root = Path(root_text)
    if not dataset_root.is_absolute():
        dataset_root = (yaml_path.parent / dataset_root).resolve()
    train = Path(yaml_scalar(text, "train"))
    val = Path(yaml_scalar(text, "val"))
    return (
        train if train.is_absolute() else dataset_root / train,
        val if val.is_absolute() else dataset_root / val,
    )


def safe_path_guard(path: Path) -> None:
    lowered = str(path.resolve()).lower().replace("-", "_")
    if "final_test" in lowered or "finaltest" in lowered:
        raise RuntimeError(f"Refusing final-test path: {path}")


def directory_manifest(directory: Path) -> tuple[int, str]:
    safe_path_guard(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    rows: list[str] = []
    for path in sorted((p for p in directory.iterdir() if p.is_file()), key=lambda p: p.name.lower()):
        rows.append(f"{path.name}\t{path.stat().st_size}\t{sha256_file(path)}")
    payload = "\n".join(rows).encode("utf-8")
    return len(rows), hashlib.sha256(payload).hexdigest()


def validate_fixed_sets(selected: list[FixedSet]) -> pd.DataFrame:
    rows = []
    for item in selected:
        safe_path_guard(item.yaml_path)
        if not item.yaml_path.exists():
            raise FileNotFoundError(item.yaml_path)
        text = item.yaml_path.read_text(encoding="utf-8")
        if "final_test" in text.lower().replace("-", "_"):
            raise RuntimeError(f"Refusing YAML that references final test: {item.yaml_path}")
        train_dir, val_dir = dataset_dirs(item.yaml_path)
        train_count, train_hash = directory_manifest(train_dir)
        val_count, val_hash = directory_manifest(val_dir)
        label_train = train_dir.parent.parent / "labels" / "train"
        label_val = val_dir.parent.parent / "labels" / "val"
        train_label_count, train_label_hash = directory_manifest(label_train)
        val_label_count, val_label_hash = directory_manifest(label_val)
        if train_count != item.budget or train_label_count != item.budget:
            raise RuntimeError(
                f"{item.key}: expected {item.budget} train images/labels, "
                f"found images={train_count}, labels={train_label_count}"
            )
        if val_count != 177 or val_label_count != 177:
            raise RuntimeError(
                f"{item.key}: expected 177 development images/labels, "
                f"found images={val_count}, labels={val_label_count}"
            )
        rows.append(
            {
                "set_key": item.key,
                "budget": item.budget,
                "provenance": item.provenance,
                "yaml_path": str(item.yaml_path.resolve()),
                "yaml_sha256": sha256_file(item.yaml_path),
                "train_image_count": train_count,
                "train_image_manifest_sha256": train_hash,
                "train_label_count": train_label_count,
                "train_label_manifest_sha256": train_label_hash,
                "development_image_count": val_count,
                "development_image_manifest_sha256": val_hash,
                "development_label_count": val_label_count,
                "development_label_manifest_sha256": val_label_hash,
                "final_test_used": False,
            }
        )
    result = pd.DataFrame(rows)
    if result["development_image_manifest_sha256"].nunique() != 1:
        raise RuntimeError("The fixed sets do not share an identical development image manifest")
    if result["development_label_manifest_sha256"].nunique() != 1:
        raise RuntimeError("The fixed sets do not share an identical development label manifest")
    return result


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan


def bootstrap_mean_ci(values: np.ndarray, seed: int = 20260714, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = sampled.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def package_environment() -> dict[str, Any]:
    import torch
    import ultralytics

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def recover_metrics(metrics: Any) -> tuple[dict[str, float], list[dict[str, Any]]]:
    results = getattr(metrics, "results_dict", {}) or {}
    precision = to_float(results.get("metrics/precision(B)", getattr(metrics.box, "mp", np.nan)))
    recall = to_float(results.get("metrics/recall(B)", getattr(metrics.box, "mr", np.nan)))
    aggregate = {
        "precision": precision,
        "recall": recall,
        "map50": to_float(results.get("metrics/mAP50(B)", getattr(metrics.box, "map50", np.nan))),
        "map5095": to_float(results.get("metrics/mAP50-95(B)", getattr(metrics.box, "map", np.nan))),
        "f1": f1_score(precision, recall),
    }
    per_class: list[dict[str, Any]] = []
    summary_method = getattr(metrics, "summary", None)
    if callable(summary_method):
        for item in summary_method():
            class_name = str(item.get("Class", "")).replace("rolled-in-scale", "rolled-in_scale")
            if class_name.lower() == "all" or not class_name:
                continue
            p = to_float(item.get("Box-P"))
            r = to_float(item.get("Box-R"))
            per_class.append(
                {
                    "class_name": class_name,
                    "val_images": to_float(item.get("Images")),
                    "val_instances": to_float(item.get("Instances")),
                    "precision": p,
                    "recall": r,
                    "f1": to_float(item.get("Box-F1", f1_score(p, r))),
                    "ap50": to_float(item.get("mAP50")),
                    "ap5095": to_float(item.get("mAP50-95")),
                }
            )
    return aggregate, per_class


def train_one(
    item: FixedSet,
    training_seed: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from ultralytics import YOLO

    run_name = f"{item.key}_trainseed{training_seed}"
    started = time.perf_counter()
    train_root = output_dir / "yolo_train_runs"
    print(f"[TRAIN] set={item.key} budget={item.budget} training_seed={training_seed}", flush=True)
    model = YOLO(str(args.model))
    train_result = model.train(
        data=str(item.yaml_path),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        cache=args.cache,
        device=args.device,
        project=str(train_root),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=False,
        seed=training_seed,
        deterministic=True,
        plots=False,
        val=True,
        split="val",
        save_json=False,
    )
    train_dir = Path(str(train_result.save_dir))
    best = train_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished without best.pt: {best}")

    print(f"[DEV VAL] set={item.key} training_seed={training_seed}", flush=True)
    eval_model = YOLO(str(best))
    metrics = eval_model.val(
        data=str(item.yaml_path),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        save_json=False,
        verbose=False,
        project=str(output_dir / "ultralytics_dev_val_runs"),
        name=run_name,
        exist_ok=True,
    )
    aggregate, per_class = recover_metrics(metrics)
    elapsed = time.perf_counter() - started
    common = {
        "set_key": item.key,
        "budget": item.budget,
        "training_seed": training_seed,
        "yaml_path": str(item.yaml_path.resolve()),
        "model_path": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "checkpoint_path": str(best.resolve()),
        "checkpoint_sha256": sha256_file(best),
        "train_run_dir": str(train_dir.resolve()),
        "dev_val_run_dir": str(Path(str(metrics.save_dir)).resolve()),
        "runtime_seconds": elapsed,
        "status": "success",
        "final_test_used": False,
    }
    return {**common, **aggregate}, [{**common, **row} for row in per_class]


def aggregate_results(results: pd.DataFrame, per_class: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    success = results[results["status"].eq("success")].copy()
    metric_rows = []
    for set_key, sub in success.groupby("set_key"):
        for metric in ["map50", "map5095", "precision", "recall", "f1"]:
            values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
            lo, hi = bootstrap_mean_ci(values)
            metric_rows.append(
                {
                    "set_key": set_key,
                    "budget": int(sub["budget"].iloc[0]),
                    "metric": metric,
                    "n_training_seeds": int(np.isfinite(values).sum()),
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else np.nan,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    aggregate = pd.DataFrame(metric_rows)

    paired_rows = []
    random = success[success["set_key"].eq("Random20")]
    for treatment in ["Visual20", "InstanceRich20"]:
        other = success[success["set_key"].eq(treatment)]
        merged = other.merge(random, on="training_seed", suffixes=("_treatment", "_random"))
        for metric in ["map50", "map5095", "precision", "recall", "f1"]:
            diff = pd.to_numeric(merged[f"{metric}_treatment"], errors="coerce") - pd.to_numeric(
                merged[f"{metric}_random"], errors="coerce"
            )
            lo, hi = bootstrap_mean_ci(diff.to_numpy(dtype=float))
            paired_rows.append(
                {
                    "treatment": treatment,
                    "baseline": "Random20",
                    "metric": metric,
                    "n_pairs": int(diff.notna().sum()),
                    "mean_difference": float(diff.mean()),
                    "std_difference": float(diff.std(ddof=1)) if diff.notna().sum() > 1 else np.nan,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "wins": int((diff > 0).sum()),
                    "losses": int((diff < 0).sum()),
                    "ties": int((diff == 0).sum()),
                }
            )
    paired = pd.DataFrame(paired_rows)

    class_deltas_rows = []
    if not per_class.empty:
        random_class = per_class[(per_class["set_key"].eq("Random20")) & (per_class["status"].eq("success"))]
        for treatment in ["Visual20", "InstanceRich20"]:
            treatment_class = per_class[(per_class["set_key"].eq(treatment)) & (per_class["status"].eq("success"))]
            merged = treatment_class.merge(
                random_class,
                on=["training_seed", "class_name"],
                suffixes=("_treatment", "_random"),
            )
            for class_name, sub in merged.groupby("class_name"):
                for metric in ["ap50", "ap5095", "precision", "recall", "f1"]:
                    diff = pd.to_numeric(sub[f"{metric}_treatment"], errors="coerce") - pd.to_numeric(
                        sub[f"{metric}_random"], errors="coerce"
                    )
                    lo, hi = bootstrap_mean_ci(diff.to_numpy(dtype=float))
                    class_deltas_rows.append(
                        {
                            "treatment": treatment,
                            "baseline": "Random20",
                            "class_name": class_name,
                            "metric": metric,
                            "n_pairs": int(diff.notna().sum()),
                            "mean_difference": float(diff.mean()),
                            "ci95_low": lo,
                            "ci95_high": hi,
                        }
                    )
    class_deltas = pd.DataFrame(
        class_deltas_rows,
        columns=[
            "treatment",
            "baseline",
            "class_name",
            "metric",
            "n_pairs",
            "mean_difference",
            "ci95_low",
            "ci95_high",
        ],
    )

    gate_rows = []
    for treatment in ["Visual20", "InstanceRich20"]:
        def paired_value(metric: str, column: str) -> float:
            row = paired[(paired["treatment"].eq(treatment)) & (paired["metric"].eq(metric))]
            return to_float(row.iloc[0][column]) if len(row) else np.nan

        map_diff = paired_value("map5095", "mean_difference")
        wins = paired_value("map5095", "wins")
        recall_diff = paired_value("recall", "mean_difference")
        class_map = class_deltas[
            (class_deltas["treatment"].eq(treatment))
            & (class_deltas["metric"].eq("ap5095"))
            & (class_deltas["class_name"].isin(NEU6))
        ]
        class_wins = int((class_map["mean_difference"] >= 0).sum()) if len(class_map) else 0
        worst_class = float(class_map["mean_difference"].min()) if len(class_map) else np.nan
        checks = {
            "mean_map5095_at_least_0p01": bool(np.isfinite(map_diff) and map_diff >= 0.01),
            "wins_at_least_4_of_5": bool(np.isfinite(wins) and wins >= 4),
            "mean_recall_not_below_minus_0p01": bool(np.isfinite(recall_diff) and recall_diff >= -0.01),
            "nonnegative_class_ap_at_least_4_of_6": bool(len(class_map) == 6 and class_wins >= 4),
            "no_class_ap_drop_below_minus_0p05": bool(len(class_map) == 6 and worst_class >= -0.05),
        }
        gate_rows.append(
            {
                "treatment": treatment,
                "mean_map5095_difference": map_diff,
                "map5095_wins": wins,
                "mean_recall_difference": recall_diff,
                "class_ap_nonnegative_count": class_wins,
                "worst_class_ap_difference": worst_class,
                **checks,
                "passed_checks": sum(checks.values()),
                "total_checks": len(checks),
                "gate_pass": all(checks.values()),
            }
        )
    return aggregate, paired, class_deltas, pd.DataFrame(gate_rows)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def write_summary(
    output_dir: Path,
    config: dict[str, Any],
    results: pd.DataFrame,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    class_deltas: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    map_summary = aggregate[aggregate["metric"].eq("map5095")]
    paired_primary = paired[paired["metric"].isin(["map5095", "recall", "f1"])]
    class_primary = class_deltas[class_deltas["metric"].eq("ap5095")]
    text = f"""# Seed45 fixed-set training stability

- Training performed: `{not config['dry_run']}`
- Acquisition performed: `False`
- Final test used: `False`
- Evaluation: existing 177-image development val only
- Model: `{config['model']}`
- Training seeds: `{config['training_seeds']}`

## Status

{md_table(results[['set_key','budget','training_seed','status','map5095','recall','f1','runtime_seconds']] if len(results) else results)}

## mAP50-95 stability

{md_table(map_summary)}

## Paired differences vs Random20

{md_table(paired_primary)}

## Per-class AP50-95 differences

{md_table(class_primary)}

## Pre-registered screen gate

{md_table(gates)}

The gate is a screening decision, not final evidence. A pass permits an independent acquisition-seed experiment; a fail stops cold-start selector development from this seed45 case.
"""
    (output_dir / "seed45_fixed_set_stability_summary.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate and write plan only; never import/train YOLO.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Use a fixed path to make interrupted runs resumable.")
    parser.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    parser.add_argument("--training-seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cache", choices=["ram", "disk", "false"], default="ram")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model = (args.model if args.model.is_absolute() else ROOT / args.model).resolve()
    safe_path_guard(args.model)
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if len(set(args.training_seeds)) != len(args.training_seeds):
        raise ValueError("training seeds must be unique")

    selected = [SETS[key] for key in args.sets]
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = OUT_ROOT / f"seed45_fixed_set_stability_{datetime.now():%Y%m%d_%H%M%S}"
    elif not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    safe_path_guard(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = validate_fixed_sets(selected)
    atomic_csv(manifests, output_dir / "fixed_dataset_manifests.csv")
    cache_value: str | bool = False if args.cache == "false" else args.cache
    args.cache = cache_value
    config = {
        "experiment": "seed45_fixed_set_training_stability",
        "created_at": datetime.now().isoformat(),
        "project_root": str(ROOT),
        "output_dir": str(output_dir),
        "dry_run": bool(args.dry_run),
        "acquisition_performed": False,
        "final_test_used": False,
        "evaluation_split": "existing development val; 177 images",
        "sets": [asdict(item) | {"yaml_path": str(item.yaml_path.resolve())} for item in selected],
        "training_seeds": args.training_seeds,
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "cache": args.cache,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    plan = pd.DataFrame(
        [{"set_key": item.key, "budget": item.budget, "training_seed": seed, "yaml_path": str(item.yaml_path)} for seed in args.training_seeds for item in selected]
    )
    atomic_csv(plan, output_dir / "training_plan.csv")
    print(f"[OUTPUT] {output_dir}", flush=True)

    if args.dry_run:
        print(f"[DRY RUN PASS] {len(selected)} fixed sets, {len(args.training_seeds)} seeds, {len(plan)} planned trainings")
        print("No YOLO import. No training. No final test.")
        return

    environment = package_environment()
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")
    if not environment["cuda_available"] and str(args.device) not in {"cpu", "-1"}:
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    results_path = output_dir / "all_results.csv"
    per_class_path = output_dir / "per_class_metrics.csv"
    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    per_class = pd.read_csv(per_class_path) if per_class_path.exists() else pd.DataFrame()
    completed = set()
    if not results.empty:
        completed = set(
            zip(
                results.loc[results["status"].eq("success"), "set_key"].astype(str),
                pd.to_numeric(results.loc[results["status"].eq("success"), "training_seed"]).astype(int),
            )
        )

    for seed in args.training_seeds:
        for item in selected:
            key = (item.key, seed)
            if key in completed:
                print(f"[SKIP COMPLETED] set={item.key} training_seed={seed}", flush=True)
                continue
            try:
                aggregate_row, class_rows = train_one(item, seed, output_dir, args)
            except Exception as exc:
                traceback.print_exc()
                aggregate_row = {
                    "set_key": item.key,
                    "budget": item.budget,
                    "training_seed": seed,
                    "yaml_path": str(item.yaml_path.resolve()),
                    "model_path": str(args.model),
                    "status": "failed",
                    "error": repr(exc),
                    "final_test_used": False,
                    "map50": np.nan,
                    "map5095": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "runtime_seconds": np.nan,
                }
                class_rows = []
            if not results.empty:
                results = results[~((results["set_key"].eq(item.key)) & pd.to_numeric(results["training_seed"]).eq(seed))]
            results = pd.concat([results, pd.DataFrame([aggregate_row])], ignore_index=True, sort=False)
            if class_rows:
                if not per_class.empty:
                    per_class = per_class[~((per_class["set_key"].eq(item.key)) & pd.to_numeric(per_class["training_seed"]).eq(seed))]
                per_class = pd.concat([per_class, pd.DataFrame(class_rows)], ignore_index=True, sort=False)
            atomic_csv(results, results_path)
            atomic_csv(per_class, per_class_path)
            aggregate, paired, class_deltas, gates = aggregate_results(results, per_class)
            atomic_csv(aggregate, output_dir / "aggregate_metrics.csv")
            atomic_csv(paired, output_dir / "paired_differences_vs_random.csv")
            atomic_csv(class_deltas, output_dir / "per_class_differences_vs_random.csv")
            atomic_csv(gates, output_dir / "screen_gate.csv")
            write_summary(output_dir, config, results, aggregate, paired, class_deltas, gates)

    # Always rebuild derived tables, including when every training is skipped
    # during a resume. This also makes post-run analysis fixes non-destructive.
    aggregate, paired, class_deltas, gates = aggregate_results(results, per_class)
    atomic_csv(aggregate, output_dir / "aggregate_metrics.csv")
    atomic_csv(paired, output_dir / "paired_differences_vs_random.csv")
    atomic_csv(class_deltas, output_dir / "per_class_differences_vs_random.csv")
    atomic_csv(gates, output_dir / "screen_gate.csv")
    write_summary(output_dir, config, results, aggregate, paired, class_deltas, gates)

    failed = results[results["status"].ne("success")]
    print(f"[DONE] successful={int(results['status'].eq('success').sum())} failed={len(failed)}", flush=True)
    print(f"[SUMMARY] {output_dir / 'seed45_fixed_set_stability_summary.md'}", flush=True)
    if len(failed):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

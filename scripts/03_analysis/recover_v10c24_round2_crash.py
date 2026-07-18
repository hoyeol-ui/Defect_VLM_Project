"""Recover V10c24 round2 scale-smoke outputs after a post-training crash.

The 2026-07-13 crash happened after all YOLO trainings completed, during
summary writing, because NumPy no longer exposed np.trapz in the local runtime.

This script:
  - performs no training
  - evaluates only the development val split from each existing data.yaml
  - never uses final test
  - rewrites result summary CSV/MD files in the existing run folder
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_02 = PROJECT_ROOT / "scripts" / "02_active_learning"
if str(SCRIPT_02) not in sys.path:
    sys.path.insert(0, str(SCRIPT_02))

import run_v10c24_round2_scale_smoke as runner  # noqa: E402
from ultralytics import YOLO  # noqa: E402


ROUND0_STRATEGY = "__SHARED_ROUND0__"
RANDOM_STRATEGY = "GTFreeRandom"
V10C_STRATEGY = "DetectorRecallGuardDINOInstanceV10c"
NEU6 = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def f1_score(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall) or precision + recall == 0:
        return np.nan
    return float(2 * precision * recall / (precision + recall))


def table_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def parse_train_dir_name(name: str) -> dict[str, Any]:
    m = re.match(r"seed(?P<seed>\d+)_(?P<strategy>.+)_R(?P<round>\d+)_trainseed(?P<training_seed>\d+)$", name)
    if not m:
        raise ValueError(f"Unrecognized train directory name: {name}")
    return {
        "acquisition_seed": int(m.group("seed")),
        "strategy": m.group("strategy"),
        "round": int(m.group("round")),
        "training_seed": int(m.group("training_seed")),
    }


def yaml_for(dataset_root: Path, meta: dict[str, Any]) -> Path:
    seed = meta["acquisition_seed"]
    strategy = meta["strategy"]
    round_idx = meta["round"]
    if strategy == ROUND0_STRATEGY:
        return dataset_root / f"seed{seed}" / strategy / "data.yaml"
    return dataset_root / f"seed{seed}" / strategy / f"round_{round_idx}" / "data.yaml"


def arr_value(arr: Any, idx: int) -> float:
    try:
        if arr is None or idx >= len(arr):
            return np.nan
        return float(arr[idx])
    except Exception:
        return np.nan


def int_value(arr: Any, idx: int) -> int | None:
    try:
        if arr is None or idx >= len(arr):
            return None
        return int(arr[idx])
    except Exception:
        return None


def normalize_class_name(value: Any) -> str:
    return str(value).replace("rolled-in-scale", "rolled-in_scale")


def eval_checkpoint(train_dir: Path, dataset_root: Path, *, device: str, batch: int, imgsz: int) -> tuple[dict[str, Any], pd.DataFrame]:
    meta = parse_train_dir_name(train_dir.name)
    yaml_path = yaml_for(dataset_root, meta)
    weights = train_dir / "weights" / "best.pt"
    if not yaml_path.exists():
        raise FileNotFoundError(yaml_path)
    if not weights.exists():
        raise FileNotFoundError(weights)

    res = YOLO(str(weights)).val(
        data=str(yaml_path),
        imgsz=imgsz,
        batch=batch,
        workers=0,
        device=device,
        split="val",
        verbose=False,
        plots=False,
        save_json=False,
    )
    box = res.box
    precision = safe_float(getattr(box, "mp", np.nan))
    recall = safe_float(getattr(box, "mr", np.nan))
    map50 = safe_float(getattr(box, "map50", np.nan))
    map5095 = safe_float(getattr(box, "map", np.nan))

    budget = 60 if meta["round"] == 0 else 60 + 30 * int(meta["round"])
    row = {
        **meta,
        "labeled_budget": budget,
        "development_eval_size": 300,
        "yaml_path": str(yaml_path),
        "map50": map50,
        "map5095": map5095,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "train_status": "success",
        "train_run_dir": str(train_dir),
        "error": "",
        "recovered_after_summary_crash": True,
    }

    names = getattr(res, "names", {})
    if isinstance(names, dict):
        class_names = [names.get(i, str(i)) for i in range(len(names))]
    else:
        class_names = list(names)
    per_rows = []
    for class_id, class_name in enumerate(class_names):
        normalized = normalize_class_name(class_name)
        if normalized not in NEU6:
            continue
        per_rows.append(
            {
                **meta,
                "class_id": class_id,
                "class_name": normalized,
                "ap50": arr_value(getattr(box, "ap50", None), class_id),
                "ap5095": arr_value(getattr(box, "ap", None), class_id),
                "precision": arr_value(getattr(box, "p", None), class_id),
                "recall": arr_value(getattr(box, "r", None), class_id),
                "validation_instance_count": int_value(getattr(box, "nt_per_class", None), class_id),
                "aggregate_map50_recovered": map50,
                "aggregate_map5095_recovered": map5095,
                "status": "success",
                "checkpoint": str(weights),
            }
        )
    return row, pd.DataFrame(per_rows)


def write_report(run_dir: Path, results: pd.DataFrame, comparison: pd.DataFrame, aulc: pd.DataFrame, class_delta: pd.DataFrame, gate: pd.DataFrame) -> None:
    round2 = comparison[pd.to_numeric(comparison.get("round"), errors="coerce").eq(2)].copy() if len(comparison) else pd.DataFrame()
    lines = [
        "# Recovered V10c24 round2 scale-smoke results",
        "",
        f"- Run dir: `{run_dir}`",
        "- Recovery reason: summary crash after training (`np.trapz` unavailable)",
        "- Training performed by recovery: `False`",
        "- Final test used: `False`",
        "- Validation split: existing development `val` split in each `data.yaml`",
        "",
        "## Round results",
        "",
        table_md(results[["acquisition_seed", "training_seed", "strategy", "round", "labeled_budget", "map50", "map5095", "precision", "recall", "f1", "train_status"]]),
        "",
        "## Random vs V10c24",
        "",
        table_md(comparison),
        "",
        "## AULC / nAULC",
        "",
        table_md(aulc),
        "",
        "## Round2 gate",
        "",
        table_md(gate),
        "",
        "## Round2 per-class V10c24 - Random",
        "",
        table_md(class_delta),
    ]
    if len(round2):
        r = round2.iloc[0]
        lines.extend(
            [
                "",
                "## Short verdict",
                "",
                f"- Round2 mAP50-95 diff: `{safe_float(r.get('v10c24_minus_random_map5095')):.6f}`",
                f"- Round2 recall diff: `{safe_float(r.get('v10c24_minus_random_recall')):.6f}`",
                f"- Round2 F1 diff: `{safe_float(r.get('v10c24_minus_random_f1')):.6f}`",
            ]
        )
    (run_dir / "recovered_v10c24_round2_scale_smoke_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    dataset_root = PROJECT_ROOT / "datasets" / "active_learning_v10c24_round2_scale_smoke" / run_dir.name
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    train_root = run_dir / "yolo_train_runs"
    train_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    rows = []
    per_class_frames = []
    registry_rows = []
    print("=" * 100)
    print("[Recover V10c24 round2 crash]")
    print(f"Run dir: {run_dir}")
    print(f"Checkpoints: {len(train_dirs)}")
    print("No training. Final test used=False.")
    print("=" * 100)
    for train_dir in train_dirs:
        try:
            print(f"[VAL] {train_dir.name}")
            row, per_class = eval_checkpoint(train_dir, dataset_root, device=str(args.device), batch=int(args.batch), imgsz=int(args.imgsz))
            rows.append(row)
            per_class_frames.append(per_class)
            registry_rows.append({"train_dir": str(train_dir), "status": "success", "error": ""})
        except Exception as exc:
            registry_rows.append({"train_dir": str(train_dir), "status": "failed", "error": repr(exc)})
            raise

    results = pd.DataFrame(rows).sort_values(["acquisition_seed", "round", "strategy"]).reset_index(drop=True)
    per_class_df = pd.concat(per_class_frames, ignore_index=True, sort=False) if per_class_frames else pd.DataFrame()
    comparison, aulc, class_delta = runner.summarize_outputs(results, per_class_df)
    gate = runner.evaluate_gate(comparison, class_delta)

    results.to_csv(run_dir / "all_round_results.csv", index=False, encoding="utf-8-sig")
    per_class_df.to_csv(run_dir / "per_class_metrics_by_round.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(run_dir / "roundwise_random_v10c24_comparison.csv", index=False, encoding="utf-8-sig")
    aulc.to_csv(run_dir / "aulc_summary.csv", index=False, encoding="utf-8-sig")
    class_delta.to_csv(run_dir / "round2_per_class_v10c24_minus_random.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(run_dir / "round2_scale_gate.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(registry_rows).to_csv(run_dir / "recovery_registry.csv", index=False, encoding="utf-8-sig")
    write_report(run_dir, results, comparison, aulc, class_delta, gate)

    r2 = comparison[pd.to_numeric(comparison.get("round"), errors="coerce").eq(2)].copy()
    print("=" * 100)
    print("[DONE] Recovered V10c24 round2 scale-smoke outputs")
    print(f"Run dir: {run_dir}")
    if len(r2):
        row = r2.iloc[0]
        print(f"Round2 mAP50-95 diff: {safe_float(row.get('v10c24_minus_random_map5095')):.6f}")
        print(f"Round2 recall diff   : {safe_float(row.get('v10c24_minus_random_recall')):.6f}")
        print(f"Round2 F1 diff       : {safe_float(row.get('v10c24_minus_random_f1')):.6f}")
    if len(gate):
        print(f"Round2 gate passed   : {int(gate['gate_pass'].sum())}/{len(gate)}")
    print("Training performed=False")
    print("Final test used=False")
    print("=" * 100)


if __name__ == "__main__":
    main()

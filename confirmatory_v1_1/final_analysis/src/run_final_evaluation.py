"""Evaluate every sealed best checkpoint once on the authorized locked-final split."""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
import ctypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_common import (
    EXPECTED_CELLS,
    FINAL_ANALYSIS,
    FINAL_EVALUATION,
    FINAL_LOCK,
    FREEZE_COMMIT_SHA,
    PROTOCOL_SHA256,
    WORKTREE,
    atomic_json,
    attempt_registry_sha256,
    build_verified_cell_registry,
    canonical_sha256,
    read_json,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from confirmatory_v1_1.final_lock import require_actual_final_access  # noqa: E402
from prepare_yolo_benchmark_data import NAMES, yolo_label  # noqa: E402


DATA_ROOT = Path("C:/Users/user/Desktop/vlm/Defect_VLM_Project/data")
SOURCE_REGISTRY = WORKTREE / "confirmatory_v1" / "data_freeze" / "private_source_registry.csv"
FINAL_MANIFESTS = {
    "GC10-DET": WORKTREE / "confirmatory_v1" / "data_freeze" / "gc10_final_locked_manifest.csv",
    "NEU-DET": WORKTREE / "confirmatory_v1" / "data_freeze" / "neu_final_locked_manifest.csv",
}
EXPECTED_FINAL_MANIFEST_SHA = {
    "GC10-DET": "8febd2c19c347473d259d7b59ec01cf7a5a9799bf1333e4e6cac6411f4c84b43",
    "NEU-DET": "7e649a5f3eee8d422b4bc0ddd8dacb96f4382256fe5ab834296db0153dfb48a0",
}
EXPECTED_FINAL_COUNTS = {"GC10-DET": 458, "NEU-DET": 359}
EXPECTED_SOURCE_REGISTRY_SHA = "688fe8bee640b8a8e3fbadc9d3efdb4d08638be59cff8eb372ae7d62d5768f3c"


def as_float(value: object) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def link_or_copy(source: Path, target: Path) -> str:
    if target.exists():
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def materialize_authorized_final_data() -> dict[str, Any]:
    if sha256_file(SOURCE_REGISTRY) != EXPECTED_SOURCE_REGISTRY_SHA:
        raise RuntimeError("private source registry SHA mismatch")
    source_rows = read_csv(SOURCE_REGISTRY)
    lookup = {(row["dataset"], row["sample_id"]): row for row in source_rows}
    summary: dict[str, Any] = {"status": "PASS", "datasets": {}}
    for dataset in ("GC10-DET", "NEU-DET"):
        manifest = FINAL_MANIFESTS[dataset]
        if sha256_file(manifest) != EXPECTED_FINAL_MANIFEST_SHA[dataset]:
            raise RuntimeError(f"final manifest SHA mismatch: {dataset}")
        final_ids = [row["sample_id"] for row in read_csv(manifest)]
        if len(final_ids) != EXPECTED_FINAL_COUNTS[dataset] or len(set(final_ids)) != len(final_ids):
            raise RuntimeError(f"final manifest cardinality mismatch: {dataset}")
        output = FINAL_EVALUATION / "data_cache" / dataset
        images = output / "images"
        labels = output / "labels"
        lists = output / "lists"
        for directory in (images, labels, lists):
            directory.mkdir(parents=True, exist_ok=True)
        methods = {"hardlink": 0, "copy": 0, "existing": 0}
        for sample_id in final_ids:
            row = lookup.get((dataset, sample_id))
            if row is None or row.get("split") != "final":
                raise RuntimeError(f"final source lookup failure: {dataset}/{sample_id}")
            image_source = DATA_ROOT / row["relative_image_path"]
            annotation_source = DATA_ROOT / row["relative_annotation_path"]
            if not image_source.is_file() or not annotation_source.is_file():
                raise FileNotFoundError(sample_id)
            if sha256_file(image_source) != row["raw_sha256"]:
                raise RuntimeError(f"final image SHA mismatch: {sample_id}")
            if sha256_file(annotation_source) != row["annotation_sha256"]:
                raise RuntimeError(f"final annotation SHA mismatch: {sample_id}")
            target_image = images / f"{sample_id}.jpg"
            methods[link_or_copy(image_source, target_image)] += 1
            target_label = labels / f"{sample_id}.txt"
            label_text = yolo_label(dataset, image_source, annotation_source)
            if target_label.exists():
                if target_label.read_text(encoding="ascii") != label_text:
                    raise RuntimeError(f"final YOLO label mismatch: {sample_id}")
            else:
                target_label.write_text(label_text, encoding="ascii", newline="\n")
        final_list = lists / "locked_final.txt"
        final_list.write_text(
            "\n".join(
                str((images / f"{sample_id}.jpg").resolve()).replace("\\", "/")
                for sample_id in final_ids
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        names_yaml = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(NAMES[dataset]))
        yaml_path = output / "locked_final.yaml"
        yaml_path.write_text(
            f"path: {str(output.resolve()).replace(chr(92), '/')}\n"
            "train: lists/locked_final.txt\n"
            "val: lists/locked_final.txt\n"
            f"names:\n{names_yaml}\n",
            encoding="utf-8",
            newline="\n",
        )
        summary["datasets"][dataset] = {
            "final_count": len(final_ids),
            "source_manifest_sha256": sha256_file(manifest),
            "final_list_sha256": sha256_file(final_list),
            "data_yaml_sha256": sha256_file(yaml_path),
            "materialization": methods,
            "data_yaml_path": str(yaml_path.resolve()),
        }
    atomic_json(FINAL_EVALUATION / "FINAL_DATA_MATERIALIZATION_AUDIT.json", summary)
    return summary


def write_status(complete: int, current: str | None, phase: str) -> None:
    atomic_json(
        FINAL_EVALUATION / "FINAL_EVALUATION_STATUS.json",
        {
            "phase": phase,
            "planned": EXPECTED_CELLS,
            "complete": complete,
            "pending": EXPECTED_CELLS - complete,
            "current_cell": current,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def validate_existing_metric(path: Path, record: dict[str, Any], yaml_sha: str) -> None:
    metric = read_json(path)
    required = {
        "cell_id": record["cell_id"],
        "metric_source": "locked_final",
        "best_pt_sha256": record["best_pt_sha256"],
        "final_data_yaml_sha256": yaml_sha,
        "status": "COMPLETE",
    }
    for key, expected in required.items():
        if metric.get(key) != expected:
            raise RuntimeError(f"existing final metric mismatch: {record['cell_id']} / {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize-final-evaluation", action="store_true")
    args = parser.parse_args()
    registry, records = build_verified_cell_registry()
    authorization_path = FINAL_LOCK / "FINAL_EVALUATION_AUTHORIZATION.json"
    require_actual_final_access(
        authorization_flag=args.authorize_final_evaluation,
        authorization_path=authorization_path,
        expected_protocol_sha256=PROTOCOL_SHA256,
        expected_freeze_commit_sha=FREEZE_COMMIT_SHA,
        cell_registry=registry,
        expected_attempt_registry_sha256=attempt_registry_sha256(),
    )
    if canonical_sha256(registry) != read_json(authorization_path)["cell_registry_sha256"]:
        raise RuntimeError("authorization registry changed")

    data_audit = materialize_authorized_final_data()
    raw_metrics = FINAL_EVALUATION / "raw_metrics"
    logs = FINAL_EVALUATION / "logs"
    ultralytics_runs = FINAL_EVALUATION / "ultralytics_runs"
    for directory in (raw_metrics, logs, ultralytics_runs):
        directory.mkdir(parents=True, exist_ok=True)
    completed = 0
    for record in records:
        target = raw_metrics / f"{record['cell_id']}.json"
        yaml_info = data_audit["datasets"][record["dataset"]]
        yaml_path = Path(yaml_info["data_yaml_path"])
        yaml_sha = str(yaml_info["data_yaml_sha256"])
        if target.exists():
            validate_existing_metric(target, record, yaml_sha)
            completed += 1
    write_status(completed, None, "AUTHORIZED_FINAL_EVALUATION_RUNNING")

    # Prevent unattended Windows sleep for the one-time evaluation batch.
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
    import torch
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != "8.4.90":
        raise RuntimeError("Ultralytics version mismatch")
    for index, record in enumerate(records, start=1):
        cell_id = str(record["cell_id"])
        target = raw_metrics / f"{cell_id}.json"
        yaml_info = data_audit["datasets"][record["dataset"]]
        yaml_path = Path(yaml_info["data_yaml_path"])
        yaml_sha = str(yaml_info["data_yaml_sha256"])
        if target.exists():
            continue
        write_status(completed, cell_id, "AUTHORIZED_FINAL_EVALUATION_RUNNING")
        log_path = logs / f"{cell_id}.log"
        attempt = 1 + len(list(ultralytics_runs.glob(f"{cell_id}_attempt*")))
        run_name = f"{cell_id}_attempt{attempt:02d}"
        started = time.time()
        print(f"FINAL_EVAL_START {index}/{EXPECTED_CELLS} {cell_id}", flush=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as log, redirect_stdout(log), redirect_stderr(log):
            torch.cuda.empty_cache()
            model = YOLO(str(record["best_pt_path"]), verbose=False)
            metrics = model.val(
                data=str(yaml_path),
                split="val",
                imgsz=640,
                batch=8,
                device=0,
                workers=4,
                conf=0.001,
                iou=0.7,
                max_det=300,
                plots=False,
                save_json=False,
                verbose=False,
                project=str(ultralytics_runs),
                name=run_name,
                exist_ok=False,
            )
        per_class = np.asarray(metrics.box.maps, dtype=np.float64).tolist()
        if len(per_class) != len(NAMES[str(record["dataset"])]):
            raise RuntimeError(f"per-class metric count mismatch: {cell_id}")
        values = [float(metrics.box.map), float(metrics.box.map50), as_float(metrics.box.mp), as_float(metrics.box.mr), *per_class]
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"non-finite final metric: {cell_id}")
        payload = {
            "status": "COMPLETE",
            "metric_source": "locked_final",
            "cell_id": cell_id,
            "dataset": record["dataset"],
            "acquisition_index": record["acquisition_index"],
            "acquisition_seed": record["acquisition_seed"],
            "strategy": record["strategy"],
            "strategy_token": record["strategy_token"],
            "training_seed_slot": record["training_seed_slot"],
            "training_seed": record["training_seed"],
            "round": record["round"],
            "labeled_count": record["labeled_count"],
            "manifest_sha256": record["manifest_sha256"],
            "best_pt_sha256": record["best_pt_sha256"],
            "final_data_yaml_sha256": yaml_sha,
            "final_manifest_sha256": yaml_info["source_manifest_sha256"],
            "map50_95": values[0],
            "map50": values[1],
            "precision": values[2],
            "recall": values[3],
            "per_class_map50_95": per_class,
            "class_names": NAMES[str(record["dataset"])],
            "results_dict": {str(key): float(value) for key, value in metrics.results_dict.items()},
            "evaluation_runtime_seconds": time.time() - started,
            "evaluation_attempt": attempt,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "ultralytics_version": ultralytics.__version__,
            "post_final_retraining_or_retuning": False,
        }
        atomic_json(target, payload)
        completed += 1
        write_status(completed, None, "AUTHORIZED_FINAL_EVALUATION_RUNNING")
        print(f"FINAL_EVAL_COMPLETE {completed}/{EXPECTED_CELLS} {cell_id}", flush=True)

    files = sorted(raw_metrics.glob("*.json"))
    if len(files) != EXPECTED_CELLS:
        raise RuntimeError("locked-final metric registry incomplete")
    for record in records:
        validate_existing_metric(
            raw_metrics / f"{record['cell_id']}.json",
            record,
            str(data_audit["datasets"][record["dataset"]]["data_yaml_sha256"]),
        )
    registry_sha = canonical_sha256([read_json(path) for path in files])
    audit = {
        "status": "PASS",
        "planned_evaluations": EXPECTED_CELLS,
        "completed_evaluations": EXPECTED_CELLS,
        "unique_cell_ids": EXPECTED_CELLS,
        "missing_evaluations": 0,
        "development_substitutions": 0,
        "post_final_retraining_or_retuning": 0,
        "raw_final_metric_registry_sha256": registry_sha,
        "authorization_sha256": sha256_file(authorization_path),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(FINAL_EVALUATION / "FINAL_EVALUATION_INTEGRITY_AUDIT.json", audit)
    write_status(EXPECTED_CELLS, None, "FINAL_EVALUATION_COMPLETE_AWAITING_ANALYSIS")
    print(json.dumps(audit, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

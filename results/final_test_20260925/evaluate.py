"""One frozen Ultralytics validation path for the dev gate and held-out test."""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from input_audit import METADATA, OUT, PACKAGE, RECOVERY, RUNS, check, read_json, sha

PARAMS = {"split": "val", "imgsz": 640, "batch": 8, "conf": 0.001,
          "iou": 0.70, "max_det": 300, "device": 0, "workers": 4,
          "quantize": 32, "rect": False, "seed": 2026091900,
          "deterministic": True, "plots": False, "save_json": False}
DRY_RUN_ID = "GC10_v2_a2026091901_DINO_F518_R5"
DRY_TOLERANCE_ABSOLUTE = 0.0001
os.environ.setdefault("YOLO_CONFIG_DIR", str(OUT / "runtime_settings"))


def write_json_x(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def prepare_dev_view(dataset: str):
    base = OUT / "datasets" / dataset
    idx = read_json(PACKAGE / "inputs" / dataset / "dev_images.json")
    patch = OUT.parent / "output/annotation_patch_v2_0_2_20260919/payload/annotation_overlays/ANNOTATION_PATCH_v2_0_2/labels" / dataset / "dev"
    for row in idx:
        source = PACKAGE / row["image"]
        check(source.is_file() and sha(source) == row["image_sha"], f"Dev image SHA mismatch: {row['id']}")
        target = base / "images/dev" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(source, target)
        check(sha(target) == row["image_sha"], f"Dev image view mismatch: {row['id']}")
        label = patch / (row["id"] + ".txt")
        check(label.is_file(), f"Missing patched development label: {row['id']}")
        target_label = base / "labels/dev" / label.name
        target_label.parent.mkdir(parents=True, exist_ok=True)
        if not target_label.exists():
            shutil.copyfile(label, target_label)
        check(sha(target_label) == sha(label), f"Dev label view mismatch: {row['id']}")
    check(len(list((base / "images/dev").iterdir())) == len(idx), "Dev view image count mismatch")
    check(len(list((base / "labels/dev").iterdir())) == len(idx), "Dev view label count mismatch")


def dataset_yaml(dataset: str, split: str) -> Path:
    import yaml
    base = OUT / "datasets" / dataset
    names = read_json(METADATA / "config/protocol.json")["classes"][dataset]
    check((base / "images" / split).is_dir() and (base / "labels" / split).is_dir(),
          f"Missing evaluation view: {dataset}/{split}")
    path = OUT / "configs" / (dataset + "_" + split + ".yaml")
    data = {"path": str(base), "train": str(base / "images/dev"),
            "val": str(base / "images" / split), "names": dict(enumerate(names))}
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        check(path.read_text(encoding="utf-8") == text, f"Evaluation YAML changed: {path}")
    else:
        with path.open("x", encoding="utf-8") as f:
            f.write(text)
    return path


def evaluate(run_id: str, split: str):
    check(split in ("dev", "test"), "Unknown evaluation split")
    gate = read_json(OUT / "preflight/preflight.json")
    check(gate["status"] == "PASS", "Preflight gate missing")
    checks = read_json(METADATA / "state_attempts/ANNOTATION_PATCH_v2_0_2/gates/final_integrity.json")["evidence"]["checks"]
    check(run_id in checks and "TECHNICAL" not in run_id, "Non-scientific run forbidden")
    if split == "dev":
        check(run_id == DRY_RUN_ID, "Dry-run checkpoint changed")
    else:
        dry = read_json(OUT / "preflight/dev_dry_run.json")
        check(dry["status"] == "PASS" and dry["run_id"] == DRY_RUN_ID, "Dev dry-run gate not passed")
    cell = read_json(RUNS / run_id / "cell.json")
    dataset = cell["dataset"]
    checkpoint = RECOVERY / run_id / "train/weights/best.pt"
    check(checkpoint.is_file() and sha(checkpoint) == checks[run_id]["best_sha"], "Checkpoint SHA mismatch")
    yaml_path = dataset_yaml(dataset, split)
    dest = OUT / ("dry_run" if split == "dev" else "raw/runs") / run_id
    check(not dest.exists(), f"Refusing to overwrite run output: {dest}")
    dest.mkdir(parents=True)
    config = {"run_id": run_id, "dataset": dataset, "acquisition_seed": cell["seed"],
              "method": cell["method"], "round": cell["round"],
              "cumulative_labeled_images": cell["labeled_count"],
              "cumulative_labeled_ratio": cell["labeled_count"] / read_json(METADATA / "config/protocol.json")["datasets"][dataset]["train"],
              "checkpoint_path": str(checkpoint), "checkpoint_sha256": checks[run_id]["best_sha"],
              "evaluation_dataset": dataset + "/" + split,
              "dataset_yaml": str(yaml_path), "dataset_yaml_sha256": sha(yaml_path),
              "parameters": PARAMS, "ultralytics_source_version": "8.4.90"}
    write_json_x(dest / "config.json", config)

    from ultralytics import YOLO
    import numpy as np
    from ultralytics.utils import LOGGER
    logger = logging.FileHandler(dest / "ultralytics_logger.log", mode="x", encoding="utf-8")
    logger.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(logger)
    start = time.time()
    try:
        with (dest / "console.log").open("x", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            print(f"EVALUATION_START {run_id} {split}", flush=True)
            metrics = YOLO(str(checkpoint)).val(data=str(yaml_path), project=str(dest), name="ultralytics",
                                                exist_ok=False, verbose=False, **PARAMS)
            print(f"EVALUATION_END {run_id} {split}", flush=True)
        class_ap = {int(k): float(np.mean(v)) for k, v in zip(metrics.box.ap_class_index, metrics.box.all_ap)}
        names = read_json(METADATA / "config/protocol.json")["classes"][dataset]
        per_class = {name: class_ap.get(i, 0.0) for i, name in enumerate(names)}
        overall = {k: float(v) for k, v in metrics.results_dict.items()}
        raw_ap = {str(int(k)): [float(x) for x in row] for k, row in zip(metrics.box.ap_class_index, metrics.box.all_ap)}
        result = {**config, "status": "SUCCESS", "runtime_seconds": time.time() - start,
                  "mAP50_95": overall["metrics/mAP50-95(B)"], "mAP50": overall["metrics/mAP50(B)"],
                  "per_class_AP50_95": per_class, "raw_results_dict": overall,
                  "raw_AP_by_class_and_IoU_threshold": raw_ap,
                  "present_class_indices": [int(i) for i in metrics.box.ap_class_index],
                  "missing_class_rule": "zero only when class has GT but no measured AP",
                  "console_log": str(dest / "console.log"), "ultralytics_log": str(dest / "ultralytics_logger.log")}
        write_json_x(dest / "result.json", result)
        return result
    except BaseException as exc:
        write_json_x(dest / "failure.json", {**config, "status": "FAILED", "runtime_seconds": time.time() - start,
                     "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        LOGGER.removeHandler(logger)
        logger.close()


def dry_run():
    check(not (OUT / "preflight/dev_dry_run.json").exists(), "Dry-run gate already exists")
    prepare_dev_view("GC10-DET")
    result = evaluate(DRY_RUN_ID, "dev")
    saved = read_json(RUNS / DRY_RUN_ID / "sealed_dev_metrics.json")
    differences = {"mAP50_95": result["mAP50_95"] - saved["overall"]["metrics/mAP50-95(B)"],
                   "mAP50": result["mAP50"] - saved["overall"]["metrics/mAP50(B)"]}
    differences["per_class_AP50_95"] = {k: result["per_class_AP50_95"][k] - v
                                        for k, v in saved["per_class_AP50_95"].items()}
    values = [differences["mAP50_95"], differences["mAP50"], *differences["per_class_AP50_95"].values()]
    max_abs = max(abs(x) for x in values)
    report = {"status": "PASS" if max_abs <= DRY_TOLERANCE_ABSOLUTE else "FAIL",
              "run_id": DRY_RUN_ID, "absolute_tolerance_prefixed": DRY_TOLERANCE_ABSOLUTE,
              "max_absolute_difference": max_abs, "differences": differences,
              "saved_development_metrics": saved, "new_development_metrics": result,
              "checkpoint_sha256": result["checkpoint_sha256"]}
    write_json_x(OUT / "preflight/dev_dry_run.json", report)
    print(json.dumps({k: report[k] for k in ("status", "run_id", "max_absolute_difference", "absolute_tolerance_prefixed")}), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("Development dry-run differs beyond frozen tolerance; test evaluation blocked")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["dry-run", "test-one"])
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.mode == "dry-run":
        dry_run()
    else:
        check(args.run_id is not None, "--run-id required")
        evaluate(args.run_id, "test")


"""Run one outcome-blinded YOLO lifecycle or exact 100-epoch feasibility cell."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import psutil


WORKTREE = Path(__file__).resolve().parents[2]
ROOT = WORKTREE / "confirmatory_v1_1"
CHECKPOINT = Path("C:/Users/user/Desktop/vlm/Defect_VLM_Project/yolov8n.pt")
EXPECTED_CHECKPOINT_SHA = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
TRAINING_SEEDS = {"GC10-DET": 1356694077, "NEU-DET": 838661549}
CLASS_COUNTS = {"GC10-DET": 10, "NEU-DET": 6}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Sampler:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.peak_rss = 0
        self.peak_nvidia_used_mib = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        process = psutil.Process(os.getpid())
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        while not self.stop.wait(0.5):
            try:
                rss = process.memory_info().rss + sum(child.memory_info().rss for child in process.children(recursive=True))
                self.peak_rss = max(self.peak_rss, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            try:
                value = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, creationflags=flags, timeout=3)
                self.peak_nvidia_used_mib = max(self.peak_nvidia_used_mib, int(value.strip().splitlines()[0]))
            except Exception:
                pass

    def __enter__(self) -> "Sampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(timeout=3)


def append_attempt(row: dict[str, object]) -> None:
    path = ROOT / "attempt_registry" / "phase2b_attempts.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cell_id", "attempt", "dataset", "round", "kind", "status", "started_unix", "ended_unix", "error_type", "protocol_sha256", "checkpoint_sha256", "data_yaml_sha256"]
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def as_float(value: object) -> float:
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(CLASS_COUNTS), required=True)
    parser.add_argument("--round", type=int, choices=[0, 3, 5], required=True)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    if sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("sealed YOLO checkpoint hash mismatch")

    kind = "tiny_lifecycle" if args.tiny else "scientific_feasibility"
    cell_id = f"tiny_{args.dataset}_R{args.round}" if args.tiny else f"{args.dataset}_a00_Random_A_R{args.round}"
    data_yaml = ROOT / "yolo_data_cache" / args.dataset / f"dataset_R{args.round}.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    registry_path = ROOT / "benchmark_registry" / "jobs" / f"{cell_id}.json"
    if registry_path.exists():
        raise FileExistsError(f"cell registry already exists: {registry_path}")
    outcome_root = ROOT / "outcomes_private" / kind / cell_id
    console_dir = ROOT / "outcomes_private" / "console"
    console_dir.mkdir(parents=True, exist_ok=True)
    outcome_root.parent.mkdir(parents=True, exist_ok=True)
    raw_log = console_dir / f"{cell_id}.log"
    started_unix = time.time()
    protocol_sha = sha(ROOT / "confirmatory_protocol_v1_1.yaml")
    attempt_base = {
        "cell_id": cell_id,
        "attempt": 1,
        "dataset": args.dataset,
        "round": args.round,
        "kind": kind,
        "started_unix": started_unix,
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "data_yaml_sha256": sha(data_yaml),
    }
    console = sys.stdout

    try:
        with raw_log.open("w", encoding="utf-8", newline="\n") as log, redirect_stdout(log), redirect_stderr(log):
            import torch
            import ultralytics
            from ultralytics import YOLO

            torch.cuda.set_device("cuda:0")
            torch.cuda.init()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats("cuda:0")
            model = YOLO(str(CHECKPOINT), verbose=False)
            train_started = time.perf_counter()

            def heartbeat(trainer: object) -> None:
                epoch = int(getattr(trainer, "epoch", -1)) + 1
                every = 1 if args.tiny else 5
                if epoch == 1 or epoch % every == 0 or epoch == (1 if args.tiny else 100):
                    print("PHASE2B_HEARTBEAT " + json.dumps({"cell_id": cell_id, "epoch": epoch, "elapsed_seconds": time.perf_counter() - train_started, "gpu_reserved_bytes": int(torch.cuda.memory_reserved(0))}, sort_keys=True), file=console, flush=True)

            model.add_callback("on_train_epoch_end", heartbeat)
            epochs = 1 if args.tiny else 100
            imgsz = 64 if args.tiny else 640
            with Sampler() as sampler:
                model.train(
                    data=str(data_yaml), epochs=epochs, patience=100, imgsz=imgsz, batch=8,
                    device=0, workers=4, cache=False, amp=True, deterministic=True,
                    optimizer="SGD", lr0=0.01, lrf=0.01, cos_lr=False, momentum=0.937,
                    weight_decay=0.0005, nbs=64, warmup_epochs=3.0,
                    warmup_momentum=0.8, warmup_bias_lr=0.1, close_mosaic=10,
                    rect=False, single_cls=False, fraction=1.0, hsv_h=0.015,
                    hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5,
                    shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0,
                    mosaic=1.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
                    seed=TRAINING_SEEDS[args.dataset], resume=False, val=True,
                    project=str(outcome_root.parent), name=outcome_root.name,
                    exist_ok=False, save=True, save_period=-1, plots=False,
                    verbose=False,
                )
                train_seconds = time.perf_counter() - train_started
                save_dir = Path(model.trainer.save_dir)
                best = save_dir / "weights" / "best.pt"
                last = save_dir / "weights" / "last.pt"
                if not best.is_file() or not last.is_file():
                    raise RuntimeError("best.pt/last.pt lifecycle failure")
                eval_started = time.perf_counter()
                evaluation_model = YOLO(str(best), verbose=False)
                metrics = evaluation_model.val(data=str(data_yaml), split="val", imgsz=imgsz, batch=8, device=0, workers=4, conf=0.001, iou=0.7, max_det=300, plots=False, save_json=False, verbose=False, project=str(save_dir), name="sealed_development_eval")
                eval_seconds = time.perf_counter() - eval_started
                per_class = np.asarray(metrics.box.maps, dtype=np.float64).tolist()
                if len(per_class) != CLASS_COUNTS[args.dataset]:
                    raise RuntimeError("per-class metric extraction count mismatch")
                private_metrics = {
                    "metric_source": "development",
                    "role": "phase2b_outcome_private_not_used_for_readiness",
                    "map50_95": float(metrics.box.map),
                    "map50": float(metrics.box.map50),
                    "precision": as_float(metrics.box.mp),
                    "recall": as_float(metrics.box.mr),
                    "per_class_map50_95": per_class,
                    "results_dict": {str(key): float(value) for key, value in metrics.results_dict.items()},
                }
                (save_dir / "OUTCOME_PRIVATE_DEVELOPMENT_METRICS.json").write_text(json.dumps(private_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            artifact_files = [path for path in save_dir.rglob("*") if path.is_file()]
            operation = {
                "cell_id": cell_id,
                "dataset": args.dataset,
                "round": args.round,
                "kind": kind,
                "status": "PASS",
                "epochs_completed": epochs,
                "train_runtime_seconds": train_seconds,
                "development_eval_runtime_seconds": eval_seconds,
                "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                "peak_nvidia_memory_used_mib": int(sampler.peak_nvidia_used_mib),
                "peak_host_rss_bytes": int(sampler.peak_rss),
                "artifact_count": len(artifact_files),
                "artifact_bytes": sum(path.stat().st_size for path in artifact_files),
                "best_pt_present": True,
                "last_pt_present": True,
                "per_class_extraction_count": len(per_class),
                "resume": False,
                "cold_start_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
                "protocol_sha256": protocol_sha,
                "data_yaml_sha256": sha(data_yaml),
                "best_pt_sha256": sha(best),
                "last_pt_sha256": sha(last),
                "ultralytics_version": ultralytics.__version__,
                "training_seed": TRAINING_SEEDS[args.dataset],
                "outcome_values_exposed_to_readiness": False,
                "retry_count": 0,
                "reuse_classification": "NON_SCIENTIFIC_FIXTURE" if args.tiny else "REUSABLE_SCIENTIFIC_CELL_CANDIDATE_PENDING_PHASE_PASS",
            }
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(operation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        append_attempt({**attempt_base, "status": "PASS", "ended_unix": time.time(), "error_type": ""})
        public = {key: operation[key] for key in ("cell_id", "dataset", "round", "kind", "status", "epochs_completed", "train_runtime_seconds", "development_eval_runtime_seconds", "peak_gpu_reserved_bytes", "peak_nvidia_memory_used_mib", "peak_host_rss_bytes", "artifact_count", "artifact_bytes", "best_pt_present", "last_pt_present", "per_class_extraction_count", "retry_count")}
        print("PHASE2B_OPERATIONAL_RESULT " + json.dumps(public, sort_keys=True), flush=True)
    except Exception as error:
        append_attempt({**attempt_base, "status": "FAIL", "ended_unix": time.time(), "error_type": type(error).__name__})
        print("PHASE2B_OPERATIONAL_FAILURE " + json.dumps({"cell_id": cell_id, "error_type": type(error).__name__, "message": str(error)}, sort_keys=True), file=console, flush=True)
        traceback.print_exc(file=console)
        raise


if __name__ == "__main__":
    main()

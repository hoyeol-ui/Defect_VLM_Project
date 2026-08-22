"""Execute one frozen training cell while keeping all scientific outcomes private."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import psutil


HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[3]
ROOT = WORKTREE / "confirmatory_v1_1"
FULL = ROOT / "full_run"
sys.path.insert(0, str(WORKTREE / "confirmatory_v1" / "src"))

from confirmatory_v1.protocol_core import manifest_sha256, validate_sample_ids  # noqa: E402


PROTOCOL_PATH = ROOT / "confirmatory_protocol_v1_1.yaml"
FINAL_AUTH = ROOT / "final_lock" / "FINAL_EVALUATION_AUTHORIZATION.json"
CHECKPOINT = Path("C:/Users/user/Desktop/vlm/Defect_VLM_Project/yolov8n.pt")
PROTOCOL_SHA = "6836f452d22c3de26dfa2680c45d0b5cb8f925b1a9274789c4ef81a1770f0b10"
CHECKPOINT_SHA = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
CLASS_COUNTS = {"GC10-DET": 10, "NEU-DET": 6}


class ProtocolGuardError(RuntimeError):
    pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def verify_environment() -> dict[str, str]:
    observed = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": package_version("numpy"),
        "pillow": package_version("pillow"),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "ultralytics": package_version("ultralytics"),
    }
    expected = {
        "python": "3.11.9",
        "numpy": "2.4.6",
        "pillow": "12.3.0",
        "torch": "2.12.1",
        "transformers": "5.13.0",
        "ultralytics": "8.4.90",
    }
    for key, value in expected.items():
        if not observed[key].startswith(value):
            raise ProtocolGuardError(f"environment mismatch: {key}")
    return observed


class Sampler:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.peak_rss = 0
        self.peak_nvidia_used_mib = 0
        self.peak_temperature_c = 0
        self.temperature_sum = 0
        self.temperature_count = 0
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
                raw = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
                    text=True,
                    creationflags=flags,
                    timeout=3,
                ).strip().splitlines()[0]
                memory, temperature = (int(token.strip()) for token in raw.split(","))
                self.peak_nvidia_used_mib = max(self.peak_nvidia_used_mib, memory)
                self.peak_temperature_c = max(self.peak_temperature_c, temperature)
                self.temperature_sum += temperature
                self.temperature_count += 1
            except Exception:
                pass

    def __enter__(self) -> "Sampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(timeout=3)

    @property
    def mean_temperature_c(self) -> float | None:
        return None if self.temperature_count == 0 else self.temperature_sum / self.temperature_count


def as_float(value: object) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def validate_spec(spec: dict[str, object]) -> tuple[Path, Path, dict[str, str]]:
    if FINAL_AUTH.exists():
        raise ProtocolGuardError("final authorization must remain absent during training")
    if sha(PROTOCOL_PATH) != PROTOCOL_SHA or spec.get("protocol_sha256") != PROTOCOL_SHA:
        raise ProtocolGuardError("protocol SHA mismatch")
    ledger = json.loads((FULL / "runtime" / "FULL_RUN_FREEZE_LEDGER.json").read_text(encoding="utf-8"))
    if sha(HERE) != ledger.get("runner_sha256") or spec.get("runner_sha256") != ledger.get("runner_sha256"):
        raise ProtocolGuardError("runner SHA mismatch")
    if sha(CHECKPOINT) != CHECKPOINT_SHA or spec.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise ProtocolGuardError("checkpoint SHA mismatch")
    manifest = Path(str(spec["manifest_path"]))
    ids = [line.strip() for line in manifest.read_text(encoding="ascii").splitlines() if line.strip()]
    validate_sample_ids(ids)
    if manifest_sha256(ids) != spec.get("manifest_sha256") or len(ids) != int(spec["labeled_count"]):
        raise ProtocolGuardError("manifest integrity mismatch")
    data_yaml = Path(str(spec["data_yaml_path"]))
    if sha(data_yaml) != spec.get("data_yaml_sha256"):
        raise ProtocolGuardError("data YAML SHA mismatch")
    environment = verify_environment()
    return manifest, data_yaml, environment


def failure_class(error: BaseException) -> str:
    if isinstance(error, ProtocolGuardError):
        return "INTEGRITY_FAILURE"
    lowered = f"{type(error).__name__}: {error}".lower()
    if "outofmemory" in lowered or "out of memory" in lowered:
        return "OOM"
    return "INFRA_FAILURE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    cell_id = str(spec["cell_id"])
    result_path = FULL / "runtime" / "cell_results" / f"{cell_id}_attempt{args.attempt:02d}.json"
    output = FULL / "outcomes_private" / cell_id / f"attempt_{args.attempt:02d}"
    console_dir = FULL / "outcomes_private" / "console"
    console_dir.mkdir(parents=True, exist_ok=True)
    raw_log = console_dir / f"{cell_id}_attempt{args.attempt:02d}.log"
    public_console = sys.stdout
    started_unix = time.time()
    operation_base = {
        "cell_id": cell_id,
        "attempt": args.attempt,
        "dataset": spec["dataset"],
        "acquisition_index": spec["acquisition_index"],
        "acquisition_seed": spec["acquisition_seed"],
        "strategy": spec["strategy"],
        "round": spec["round"],
        "training_seed_slot": spec["training_seed_slot"],
        "training_seed": spec["training_seed"],
        "manifest_sha256": spec["manifest_sha256"],
        "protocol_sha256": PROTOCOL_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "started_unix": started_unix,
        "outcome_values_exposed": False,
        "final_accessed": False,
    }
    try:
        _, data_yaml, environment = validate_spec(spec)
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
                if epoch == 1 or epoch % 5 == 0 or epoch == 100:
                    temperature = None
                    try:
                        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                        temperature = int(subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], text=True, creationflags=flags, timeout=3).strip().splitlines()[0])
                    except Exception:
                        pass
                    print(
                        "PHASE2C_HEARTBEAT " + json.dumps({
                            "cell_id": cell_id,
                            "attempt": args.attempt,
                            "epoch": epoch,
                            "elapsed_seconds": time.perf_counter() - train_started,
                            "gpu_reserved_bytes": int(torch.cuda.memory_reserved(0)),
                            "gpu_temperature_c": temperature,
                        }, sort_keys=True),
                        file=public_console,
                        flush=True,
                    )

            model.add_callback("on_train_epoch_end", heartbeat)
            with Sampler() as sampler:
                model.train(
                    data=str(data_yaml), epochs=100, patience=100, imgsz=640, batch=8,
                    device=0, workers=4, cache=False, amp=True, deterministic=True,
                    optimizer="SGD", lr0=0.01, lrf=0.01, cos_lr=False, momentum=0.937,
                    weight_decay=0.0005, nbs=64, warmup_epochs=3.0,
                    warmup_momentum=0.8, warmup_bias_lr=0.1, close_mosaic=10,
                    rect=False, single_cls=False, fraction=1.0, hsv_h=0.015,
                    hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5,
                    shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0,
                    mosaic=1.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
                    seed=int(spec["training_seed"]), resume=False, val=True,
                    project=str(output.parent), name=output.name, exist_ok=False,
                    save=True, save_period=-1, plots=False, verbose=False,
                )
                train_seconds = time.perf_counter() - train_started
                save_dir = Path(model.trainer.save_dir)
                best = save_dir / "weights" / "best.pt"
                last = save_dir / "weights" / "last.pt"
                if not best.is_file() or not last.is_file():
                    raise ProtocolGuardError("best.pt/last.pt lifecycle failure")
                eval_started = time.perf_counter()
                evaluation_model = YOLO(str(best), verbose=False)
                metrics = evaluation_model.val(
                    data=str(data_yaml), split="val", imgsz=640, batch=8, device=0,
                    workers=4, conf=0.001, iou=0.7, max_det=300, plots=False,
                    save_json=False, verbose=False, project=str(save_dir),
                    name="sealed_development_eval",
                )
                eval_seconds = time.perf_counter() - eval_started
                per_class = np.asarray(metrics.box.maps, dtype=np.float64).tolist()
                if len(per_class) != CLASS_COUNTS[str(spec["dataset"])]:
                    raise ProtocolGuardError("per-class extraction mismatch")
                private_metrics = {
                    "metric_source": "development",
                    "role": "outcome_private_not_used_during_full_training",
                    "map50_95": float(metrics.box.map),
                    "map50": float(metrics.box.map50),
                    "precision": as_float(metrics.box.mp),
                    "recall": as_float(metrics.box.mr),
                    "per_class_map50_95": per_class,
                    "results_dict": {str(key): float(value) for key, value in metrics.results_dict.items()},
                }
                (save_dir / "OUTCOME_PRIVATE_DEVELOPMENT_METRICS.json").write_text(json.dumps(private_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            artifact_files = [path for path in save_dir.rglob("*") if path.is_file()]
            free_disk = shutil.disk_usage(WORKTREE).free
            operation = {
                **operation_base,
                "status": "COMPLETE",
                "ended_unix": time.time(),
                "epochs_completed": 100,
                "train_runtime_seconds": train_seconds,
                "development_eval_runtime_seconds": eval_seconds,
                "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                "peak_nvidia_memory_used_mib": int(sampler.peak_nvidia_used_mib),
                "peak_temperature_c": int(sampler.peak_temperature_c),
                "mean_temperature_c": sampler.mean_temperature_c,
                "peak_host_rss_bytes": int(sampler.peak_rss),
                "artifact_count": len(artifact_files),
                "artifact_bytes": sum(path.stat().st_size for path in artifact_files),
                "free_disk_bytes_after": int(free_disk),
                "best_pt_present": True,
                "last_pt_present": True,
                "best_pt_sha256": sha(best),
                "last_pt_sha256": sha(last),
                "per_class_extraction_count": len(per_class),
                "resume": False,
                "cold_start": True,
                "environment": environment,
                "ultralytics_version": ultralytics.__version__,
                "output_dir": str(save_dir.resolve()).replace("\\", "/"),
                "raw_console_log": str(raw_log.resolve()).replace("\\", "/"),
            }
        atomic_json(result_path, operation)
        print("PHASE2C_CELL_COMPLETE " + json.dumps({
            "cell_id": cell_id,
            "attempt": args.attempt,
            "status": "COMPLETE",
            "train_runtime_seconds": operation["train_runtime_seconds"],
            "development_eval_runtime_seconds": operation["development_eval_runtime_seconds"],
            "peak_temperature_c": operation["peak_temperature_c"],
            "peak_nvidia_memory_used_mib": operation["peak_nvidia_memory_used_mib"],
            "free_disk_bytes_after": operation["free_disk_bytes_after"],
        }, sort_keys=True), flush=True)
    except BaseException as error:
        classification = failure_class(error)
        failure = {
            **operation_base,
            "status": classification,
            "ended_unix": time.time(),
            "error_type": type(error).__name__,
            "error_message_private": str(error),
            "traceback_private_log": str(raw_log.resolve()).replace("\\", "/"),
        }
        try:
            atomic_json(result_path, failure)
            with raw_log.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
        finally:
            print("PHASE2C_CELL_FAILURE " + json.dumps({"cell_id": cell_id, "attempt": args.attempt, "status": classification, "error_type": type(error).__name__}, sort_keys=True), file=public_console, flush=True)
        raise


if __name__ == "__main__":
    main()

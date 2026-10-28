"""One authorized, fixed test evaluation. Importing this module does not import ML."""
import argparse
import importlib.metadata
import logging
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *


def kwargs_for(model, data, out, record):
    settings = record["evaluation"]
    keys = ("imgsz", "batch", "rect", "conf", "iou", "max_det", "quantize", "workers", "device", "augment", "plots", "save_json", "deterministic")
    return {**{k: settings[k] for k in keys}, "data": data["yaml"], "split": "test", "seed": model["learner_seed"],
            "project": str(out), "name": "native", "exist_ok": False,
            "end2end": True, "cache": False, "compile": False, "dnn": False,
            "single_cls": False, "classes": None, "agnostic_nms": False,
            "save": False, "save_txt": False, "save_conf": False, "visualize": False}


def metrics_payload(metrics, names):
    """Validate every class/IoU cell; no missing-class or nonfinite substitution."""
    overall = {k: float(v) for k, v in metrics.results_dict.items()}
    required = ("metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)")
    require(all(k in overall for k in required), "Native overall metric missing")
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in overall.values()), "Nonfinite/out-of-range native metrics")
    indices = [int(x) for x in metrics.box.ap_class_index]
    require(indices == list(range(len(names))), "Missing/reordered AP class support; never fill zero")
    matrix = metrics.box.all_ap.tolist()
    require(len(matrix) == len(names) and all(len(row) == 10 for row in matrix), "Class/IoU matrix shape")
    require(all(math.isfinite(float(x)) and 0 <= float(x) <= 1 for row in matrix for x in row), "Invalid class AP matrix")
    require(abs(sum(sum(row) / 10 for row in matrix) / len(names) - overall[required[0]]) < 1e-10, "Overall/class mAP disagreement")
    require(abs(sum(row[0] for row in matrix) / len(names) - overall[required[1]]) < 1e-10, "Overall/class AP50 disagreement")
    classes = []
    for i, name in enumerate(names):
        classes.append({"class_id": i, "name": name, "AP50": float(matrix[i][0]), "AP50_95": sum(matrix[i]) / 10,
                        "AP_by_IoU": [float(x) for x in matrix[i]], "precision": float(metrics.box.p[i]), "recall": float(metrics.box.r[i])})
    require(all(math.isfinite(c[k]) and 0 <= c[k] <= 1 for c in classes for k in ("precision", "recall")), "Invalid per-class P/R")
    return {"overall": overall, "class_metrics": classes, "IoU_thresholds": [0.5 + 0.05 * i for i in range(10)],
            "precision_recall_interpretation": "Native maximum smoothed mean F1 summary; not a fixed conf0.25 operating point"}


def permission(out, parent, cell):
    deadline = time.monotonic() + 30
    while not (out / "PERMIT.json").exists():
        require(time.monotonic() < deadline, "Worker permit timeout; no ML loaded")
        time.sleep(.05)
    permit = read(out / "PERMIT.json")
    require(os.getppid() == parent == permit["controller_pid"] and os.getpid() == permit["worker_pid"], "Wrong parent/worker identity")
    require(permit["cell_id"] == cell and Path(permit["output"]).resolve() == out.resolve(), "Wrong model output permit")
    require(read(GPU_LOCK).get("controller_pid") == parent and read(MUTEX).get("controller_pid") == parent, "Controller does not own GPU/evaluation lock")
    check_hash(E / "EXECUTION_AUTHORIZATION.json", permit["execution_authorization_sha"])
    check_hash(E / "TEST_PACKAGE_LOCK.json", permit["package_lock_sha"])
    require(read(ROOT / "EXECUTION_RESERVED.json")["controller_pid"] == parent, "No owned execution reservation")


def evaluate(cell, parent):
    out = RUN / "models" / cell
    permission(out, parent, cell)
    # No module-level CUDA/ML imports: Windows loader children must not evaluate.
    os.environ.update(YOLO_CONFIG_DIR=str(out / "settings"), MPLCONFIGDIR=str(out / "mpl"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled", YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true", CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    verify_seal()
    model = next(m for m in read(E / "TEST_MODELS.json")["models"] if m["cell_id"] == cell)
    data = read(E / "TEST_INPUTS.json")["datasets"][model["dataset"]]
    record = read(E / "TEST_PLAN.json")
    check_hash(model["best_path"], model["best_sha"])
    for row in data["records"]:
        check_hash(row["image"], row["image_sha"])
        check_hash(row["label"], row["label_sha"])
    import torch
    import numpy as np
    import ultralytics
    import ultralytics.data.utils as du
    import ultralytics.utils as utils
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    expected = record["runtime"]
    require(sys.version.split()[0] == expected["python"] and str(torch.__version__) == expected["torch"] and ultralytics.__version__ == expected["ultralytics"] and np.__version__ == expected["numpy"] and importlib.metadata.version("torchvision") == expected["torchvision"], "Runtime version changed; no install/fallback")
    require(torch.cuda.is_available() and expected["gpu_contains"] in torch.cuda.get_device_name(0), "Wrong/missing CUDA device")
    utils.ONLINE = False
    du.check_font = lambda *args, **kwargs: None  # IO only; do not download fonts.
    torch.set_num_threads(4)
    init_seeds(model["learner_seed"], deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = record["backend"]["matmul_TF32"]
    torch.backends.cudnn.allow_tf32 = record["backend"]["cudnn_TF32"]
    torch.backends.cudnn.benchmark = record["backend"]["cudnn_benchmark"]
    require(torch.are_deterministic_algorithms_enabled(), "Deterministic policy missing")
    actual_kwargs = kwargs_for(model, data, out, record)
    put(out / "REQUEST.json", {"cell_id": cell, "checkpoint_sha": model["best_sha"], "input_manifest_sha": sha(E / "TEST_INPUTS.json"), "parameters": actual_kwargs, "partition": "test", "training": False, "DINO_reextraction": False, "time_utc": now()})
    yolo = YOLO(model["best_path"])
    require(bool(yolo.model.end2end), "Expected native one2one/end2end head")
    require([yolo.names[i] for i in range(len(yolo.names))] == data["names"], "Model class map mismatch")
    require(all(p.dtype == torch.float32 for p in yolo.model.parameters()), "Loaded checkpoint not FP32")
    captured = {}
    messages = []
    class Watch(logging.Handler):
        def emit(self, entry):
            message = entry.getMessage()
            if any(x in message.lower() for x in ("duplicate labels removed", "corrupt image/label", "jpeg restored and saved", "no labels found")):
                messages.append(message)
    watch = Watch()
    utils.LOGGER.addHandler(watch)

    def on_start(validator):
        require(not validator.training and validator.device.type == "cuda", "Not a CUDA validation-only run")
        for key, expected_value in actual_kwargs.items():
            if key not in ("project", "name", "exist_ok", "quantize"):
                require(getattr(validator.args, key) == expected_value, "Resolved evaluation argument drift: " + key)
        require(validator.args.quantize is None, "Native backend unexpectedly FP16")
        dataset = validator.dataloader.dataset
        files = [str(Path(x).resolve()) for x in dataset.im_files]
        expected_files = [str(Path(r["image"]).resolve()) for r in data["records"]]
        require(files == expected_files, "Native loader reordered/dropped/added test images")
        require(validator.dataloader.drop_last is False and not dataset.augment and not dataset.rect, "Wrong native loader policy")
        require(len(dataset.labels) == data["n"], "Native label loader cardinality")
        for native, row in zip(dataset.labels, data["records"]):
            require(tuple(native["shape"]) == (row["height"], row["width"]), "Cached/native image geometry differs from canonical metadata")
            raw = np.loadtxt(row["label"], dtype=np.float32, ndmin=2)
            require(raw.shape == (row["bbox_rows"], 5), "Prepared label shape differs")
            loaded = np.concatenate((native["cls"], native["bboxes"]), axis=1)
            require(np.array_equal(raw, loaded) and native["normalized"] and native["bbox_format"] == "xywh", "Native parser changed normalized test rows")
        require(not messages, "Native parser warning: " + repr(messages))
        captured["validator"] = validator
        put(out / "NATIVE_LOADER_VERIFIED.json", {"partition": "test", "ids": [r["id"] for r in data["records"]], "rows_exact": True, "FP32": True, "effective_quantize": validator.args.quantize, "resolved_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(validator.args).items()}, "time_utc": now()})

    yolo.add_callback("on_val_start", on_start)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        metrics = yolo.val(**actual_kwargs)
    finally:
        utils.LOGGER.removeHandler(watch)
    torch.cuda.synchronize()
    validator = captured["validator"]
    require(validator.seen == data["n"] and validator.end2end, "Not all test images evaluated using end2end")
    require(not messages, "Native parser/evaluation warning: " + repr(messages))
    payload = metrics_payload(metrics, data["names"])
    for row in data["records"]:
        check_hash(row["image"], row["image_sha"])
        check_hash(row["label"], row["label_sha"])
    check_hash(model["best_path"], model["best_sha"])
    put(out / "metrics.json", {"status": "PASS", "cell_id": cell, "partition": "test", "technical_only": False, "model_sha": model["best_sha"], "input_manifest_sha": sha(E / "TEST_INPUTS.json"), "images_seen": validator.seen, "model": model, **payload,
                               "seconds": time.perf_counter() - start, "speed_ms_per_image": {k: float(v) for k, v in metrics.speed.items()}, "peak_GPU_bytes": torch.cuda.max_memory_allocated(), "training_performed": False, "DINO_reextraction_performed": False, "time_utc": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--parent", required=True, type=int)
    args = parser.parse_args()
    # Cell IDs come only from the fixed manifest, not a filesystem path argument.
    require(args.cell in {m["cell_id"] for m in read(E / "TEST_MODELS.json")["models"]}, "Unknown scientific model")
    try:
        evaluate(args.cell, args.parent)
    except BaseException:
        out = RUN / "models" / args.cell
        if out.exists() and not (out / "FAILURE.json").exists():
            put(out / "FAILURE.json", {"time_utc": now(), "traceback": traceback.format_exc(), "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()

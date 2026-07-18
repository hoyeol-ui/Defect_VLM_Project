"""Launch the frozen V8 Visual-vs-Random cold-start confirmation.

This launcher changes no acquisition algorithm. It reuses the frozen V8
NEU-only implementation and limits evaluation to budget 15 -> 20 for new
acquisition seeds 52..61. Final test remains locked.

The underlying runner defaults to dry-run. Set AL_DRY_RUN_ONLY=0 explicitly
for training. Use a different AL_EXPLICIT_SAVE_DIR for dry-run and training.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".python311" / "python.exe"
RUNNER = ROOT / "scripts" / "02_active_learning" / "run_al_yolo_ablation_v7_full_curve.py"
ANALYZER = ROOT / "scripts" / "03_analysis" / "analyze_v8_cold_start_visual_confirmation.py"


def main() -> None:
    env = os.environ.copy()
    if not env.get("AL_EXPLICIT_SAVE_DIR"):
        raise RuntimeError("Set AL_EXPLICIT_SAVE_DIR explicitly; dry-run and training must use separate directories.")
    env["AL_PROJECT_ROOT"] = str(ROOT)
    env["AL_FULL_CURVE_RUNS_SUBDIR"] = "active_learning_v8_cold_start_confirmation"
    env["AL_FULL_CURVE_DATASETS_SUBDIR"] = "active_learning_v8_cold_start_confirmation"
    env["AL_FULL_CURVE_RUN_PREFIX"] = "v8_cold_start_visual_confirm"
    env["AL_EXPERIMENT_LABEL"] = "Frozen V8 Visual cold-start confirmation"
    env["AL_SUMMARY_FILENAME"] = "v8_cold_start_visual_confirmation_summary.md"

    env["AL_POOL_DATASET_FILTER"] = "NEU-DET"
    env["AL_DEV_EVAL_DATASET_FILTER"] = "NEU-DET"
    env["AL_EVAL_SPLIT"] = "development_eval_v7"
    env["EVAL_PROTOCOL_DIR"] = str(ROOT / "runs" / "evaluation_protocol_v7" / "eval_protocol_20260711_173723")

    env["AL_ALLOW_FROZEN_STRATEGY_SUBSET"] = "1"
    env["AL_STRATEGIES"] = "GTFreeRandom,GTFreeDatasetBalancedVisualDiversity"
    env["AL_ACQUISITION_SEEDS"] = "52,53,54,55,56,57,58,59,60,61"
    env["AL_INITIAL_SEED_SIZE"] = "15"
    env["AL_ROUNDS"] = "1"
    env["AL_QUERY_SIZE"] = "5"

    env["AL_EPOCHS_PER_ROUND"] = "100"
    env["AL_YOLO_PATIENCE"] = "100"
    env["AL_IMGSZ"] = "640"
    env["AL_BATCH_SIZE"] = "8"
    env["AL_WORKERS"] = "4"
    env["AL_YOLO_DEVICE"] = "0"
    env["AL_YOLO_CACHE"] = "ram"
    env["AL_YOLO_PLOTS"] = "0"
    env["AL_RESUME_EXISTING_RUN"] = "0"

    env["AL_EMBEDDING_BACKEND"] = "dinov2"
    env["AL_ALLOW_MODEL_DOWNLOAD"] = "0"
    env["AL_YOLO_MODEL_NAME"] = str(ROOT / "yolov8n.pt")

    subprocess.run([str(PYTHON), str(RUNNER)], cwd=ROOT, env=env, check=True)
    dry_run = env.get("AL_DRY_RUN_ONLY", "1").strip().lower() not in {"0", "false", "no", "off"}
    selection_only = env.get("AL_SELECTION_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not dry_run and not selection_only:
        run_dir = Path(env["AL_EXPLICIT_SAVE_DIR"])
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        subprocess.run([str(PYTHON), str(ANALYZER), "--run-dir", str(run_dir)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

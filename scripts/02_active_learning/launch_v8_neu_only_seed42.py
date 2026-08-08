"""Convenience launcher for V8 NEU-only one-seed learning-curve experiment.

This launcher reuses the V7 full-curve runner but fixes the experimental
protocol to NEU-DET only:

    - acquisition pool: NEU-DET only
    - development eval: NEU-DET only
    - acquisition seed: 42 by default
    - final_test_v7: locked / never evaluated

The runner defaults remain safe.  Set AL_DRY_RUN_ONLY=0 and AL_SELECTION_ONLY=0
when intentionally launching YOLO training.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".python311" / "python.exe"
RUNNER = PROJECT_ROOT / "scripts" / "02_active_learning" / "run_al_yolo_ablation_v7_full_curve.py"


def main() -> None:
    env = os.environ.copy()
    env.setdefault("AL_PROJECT_ROOT", str(PROJECT_ROOT))

    env.setdefault("AL_FULL_CURVE_RUNS_SUBDIR", "active_learning_ablation_v8_neu_only")
    env.setdefault("AL_FULL_CURVE_DATASETS_SUBDIR", "active_learning_ablation_v8_neu_only")
    env.setdefault("AL_FULL_CURVE_RUN_PREFIX", "v8_neu_only")
    env.setdefault("AL_EXPERIMENT_LABEL", "V8 NEU-only DINO full learning curve")
    env.setdefault("AL_SUMMARY_FILENAME", "v8_neu_only_summary.md")

    env.setdefault("AL_POOL_DATASET_FILTER", "NEU-DET")
    env.setdefault("AL_DEV_EVAL_DATASET_FILTER", "NEU-DET")
    env.setdefault("AL_EVAL_SPLIT", "development_eval_v7")

    env.setdefault("AL_STRATEGIES", "GTFreeRandom,GTFreeDatasetBalancedConsistency,GTFreeDatasetBalancedVisualDiversity")
    env.setdefault("AL_ACQUISITION_SEEDS", "42")
    env.setdefault("AL_INITIAL_SEED_SIZE", "15")
    env.setdefault("AL_ROUNDS", "4")
    env.setdefault("AL_QUERY_SIZE", "5")

    env.setdefault("AL_EPOCHS_PER_ROUND", "100")
    env.setdefault("AL_YOLO_PATIENCE", "100")
    env.setdefault("AL_BATCH_SIZE", "8")
    env.setdefault("AL_WORKERS", "4")
    env.setdefault("AL_YOLO_DEVICE", "0")
    env.setdefault("AL_YOLO_CACHE", "ram")
    env.setdefault("AL_YOLO_PLOTS", "0")
    env.setdefault("AL_RESUME_EXISTING_RUN", "0")

    env.setdefault("AL_EMBEDDING_BACKEND", "dinov2")
    env.setdefault("AL_ALLOW_MODEL_DOWNLOAD", "0")
    subprocess.run([str(PYTHON), str(RUNNER)], cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

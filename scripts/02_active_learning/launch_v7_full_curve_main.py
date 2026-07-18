"""Convenience launcher for V7 DINO full learning-curve experiment.

This file does not force training on.  The runner defaults to AL_DRY_RUN_ONLY=1
unless the caller explicitly sets AL_DRY_RUN_ONLY=0.
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
    env.setdefault("AL_STRATEGIES", "GTFreeRandom,GTFreeDatasetBalancedConsistency,GTFreeDatasetBalancedVisualDiversity")
    env.setdefault("AL_ACQUISITION_SEEDS", "42,43,44,45,46")
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
    env.setdefault("AL_RESUME_EXISTING_RUN", "1")
    env.setdefault("AL_EVAL_SPLIT", "development_eval_v7")
    env.setdefault("AL_EMBEDDING_BACKEND", "dinov2")
    subprocess.run([str(PYTHON), str(RUNNER)], cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

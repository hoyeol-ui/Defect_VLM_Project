"""Run a quick V5 sanity check on seed 42.

This launcher is intentionally small and foreground-friendly: it uses the
project-local .python311 runtime, runs one seed, and matches the fixed external
evaluation split to the dataset/class strata that actually exist in the scored
active-learning pool.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".python311" / "python.exe"
RUNNER = (
    PROJECT_ROOT
    / "scripts"
    / "02_active_learning"
    / "run_al_yolo_ablation_v3_windows_cuda_tuned.py"
)


def main() -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "YOLO_CONFIG_DIR": str(PROJECT_ROOT),
            "AL_PROJECT_ROOT": str(PROJECT_ROOT),
            "AL_PRIORITY_CSV": (
                "outputs\\priority_sensitivity_20260706_152020\\penalty_0\\"
                "priority_scores_pseudo.csv"
            ),
            "AL_DRY_RUN_ONLY": "0",
            "AL_SEEDS": "42",
            "AL_STRATEGIES": (
                "GTFreeRandom,"
                "GTFreeConsistency,"
                "OracleClassDatasetBalancedRandom,"
                "OracleClassDatasetBalancedConsistency"
            ),
            "AL_ROUNDS": "4",
            "AL_FIXED_EVAL_SIZE": "180",
            "AL_FIXED_EVAL_SEED": "20260709",
            "AL_EVAL_MATCH_POOL_CLASSES": "1",
            "AL_MIN_POOL_CLASS_COUNT_FOR_EVAL": "3",
            "AL_STRICT_POOL_CLASS_CHECK": "0",
            "AL_INITIAL_SEED_SIZE": "15",
            "AL_QUERY_SIZE": "5",
            "AL_EPOCHS_PER_ROUND": "20",
            "AL_BATCH_SIZE": "8",
            "AL_WORKERS": "4",
            "AL_PATIENCE": "3",
            "AL_YOLO_CACHE": "false",
            "AL_YOLO_PLOTS": "true",
            "AL_YOLO_DEVICE": "0",
        }
    )
    subprocess.run([str(PYTHON), str(RUNNER)], cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

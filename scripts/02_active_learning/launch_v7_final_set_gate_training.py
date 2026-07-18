"""Convenience launcher for V7 final-set gate training.

It respects AL_DRY_RUN_ONLY.  By default the runner is dry-run; set
AL_DRY_RUN_ONLY=0 in PowerShell when you are ready for the 8 YOLO runs.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".python311" / "python.exe"
RUNNER = PROJECT_ROOT / "scripts" / "02_active_learning" / "run_v7_final_set_gate_training.py"


def main() -> None:
    env = os.environ.copy()
    env.setdefault("AL_PROJECT_ROOT", str(PROJECT_ROOT))
    env.setdefault(
        "AL_GATE_STRATEGIES",
        ",".join(
            [
                "GTFreeRandom",
                "GTFreeDatasetBalancedConsistency",
                "GTFreeDatasetBalancedVisualDiversity",
                "GTFreeDatasetBalancedConsistencyVisualDiversity",
            ]
        ),
    )
    env.setdefault("AL_TRAINING_SEEDS", "101,102")
    env.setdefault("AL_GATE_EPOCHS", "100")
    env.setdefault("AL_GATE_PATIENCE", "100")
    env.setdefault("AL_BATCH_SIZE", "8")
    env.setdefault("AL_WORKERS", "4")
    env.setdefault("AL_YOLO_CACHE", "ram")
    env.setdefault("AL_YOLO_PLOTS", "0")
    subprocess.run([str(PYTHON), str(RUNNER)], cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

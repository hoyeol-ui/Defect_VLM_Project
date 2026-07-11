"""Convenience launcher for V7 Stage-A dry-run screening."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".python311" / "python.exe"
RUNNER = PROJECT_ROOT / "scripts" / "02_active_learning" / "run_al_yolo_ablation_v7_visual_instance.py"


def main() -> None:
    env = os.environ.copy()
    env.setdefault("AL_PROJECT_ROOT", str(PROJECT_ROOT))
    env.setdefault("AL_SEEDS", "42")
    env.setdefault("AL_INITIAL_SEED_SIZE", "15")
    env.setdefault("AL_ROUNDS", "4")
    env.setdefault("AL_QUERY_SIZE", "5")
    env.setdefault("AL_V7_HYBRID_CANDIDATE", "A")
    env.setdefault(
        "AL_STRATEGIES",
        ",".join(
            [
                "GTFreeRandom",
                "GTFreeDatasetBalancedConsistency",
                "GTFreeDatasetBalancedVisualDiversity",
                "GTFreeDatasetBalancedConsistencyVisualDiversity",
            ]
        ),
    )
    subprocess.run([str(PYTHON), str(RUNNER)], cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

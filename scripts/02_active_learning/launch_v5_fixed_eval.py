"""Launch the V5 fixed-evaluation experiment as a detached Windows process."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".python311" / "python.exe"
RUNNER = PROJECT_ROOT / "scripts" / "02_active_learning" / "run_al_yolo_ablation_v3_windows_cuda_tuned.py"
LOG = PROJECT_ROOT / "runs" / "v5_fixed_eval_live.log"
ERROR_LOG = PROJECT_ROOT / "runs" / "v5_fixed_eval_live.err.log"


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
            "AL_SEEDS": "42,43,44,45,46,47,48,49",
            "AL_ROUNDS": "4",
            "AL_FIXED_EVAL_SIZE": "300",
            "AL_FIXED_EVAL_SEED": "20260709",
            "AL_INITIAL_SEED_SIZE": "15",
            "AL_QUERY_SIZE": "5",
            "AL_EPOCHS_PER_ROUND": "30",
            "AL_BATCH_SIZE": "16",
            "AL_WORKERS": "4",
            "AL_PATIENCE": "0",
            "AL_YOLO_CACHE": "false",
            "AL_YOLO_PLOTS": "false",
        }
    )
    creation_flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NO_WINDOW
    )
    with LOG.open("w", encoding="utf-8") as stdout, ERROR_LOG.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            [str(PYTHON), str(RUNNER)],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
            close_fds=True,
        )
    print(f"pid={process.pid}")
    print(f"log={LOG}")
    print(f"error_log={ERROR_LOG}")


if __name__ == "__main__":
    main()

"""Lightweight registry helpers for V7 methodology experiments.

The registry is deliberately append-only.  Failed and rejected runs are part of
the research trace and should not be deleted just because they are inconvenient.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REGISTRY_COLUMNS = [
    "experiment_id",
    "timestamp",
    "git_commit",
    "stage",
    "strategy",
    "acquisition_seed",
    "training_seed",
    "hyperparameters",
    "eval_split",
    "status",
    "result_path",
    "promoted_or_rejected",
    "rejection_reason",
]


def get_git_commit(project_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def append_registry_row(
    registry_path: Path,
    *,
    project_root: Path,
    experiment_id: str,
    stage: str,
    strategy: str | None = None,
    acquisition_seed: int | None = None,
    training_seed: int | None = None,
    hyperparameters: dict[str, Any] | None = None,
    eval_split: str | None = None,
    status: str = "created",
    result_path: str | Path | None = None,
    promoted_or_rejected: str | None = None,
    rejection_reason: str | None = None,
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists() and registry_path.stat().st_size > 0:
        df = pd.read_csv(registry_path)
    else:
        df = pd.DataFrame(columns=REGISTRY_COLUMNS)

    row = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": get_git_commit(project_root),
        "stage": stage,
        "strategy": strategy,
        "acquisition_seed": acquisition_seed,
        "training_seed": training_seed,
        "hyperparameters": json.dumps(hyperparameters or {}, ensure_ascii=False, sort_keys=True),
        "eval_split": eval_split,
        "status": status,
        "result_path": str(result_path) if result_path is not None else None,
        "promoted_or_rejected": promoted_or_rejected,
        "rejection_reason": rejection_reason,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.reindex(columns=REGISTRY_COLUMNS)
    df.to_csv(registry_path, index=False, encoding="utf-8-sig")

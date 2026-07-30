"""Monitor the existing frozen G5 run, then finalize and audit it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_G5_WORKTREE = ROOT.parent / "Defect_VLM_Project_ksdd2_g5"
DEFAULT_G5_ROOT = (
    DEFAULT_G5_WORKTREE / "runs" / "ksdd2_g5_single_unet_v1"
)


def completed_runs(g5_root: Path) -> int:
    learner_root = g5_root / "learner_runs"
    return sum(
        1
        for path in learner_root.glob("*/run_result.json")
        if path.with_suffix(path.suffix + ".sha256").exists()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g5-worktree", type=Path, default=DEFAULT_G5_WORKTREE)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stall-minutes", type=int, default=45)
    args = parser.parse_args()
    worktree = args.g5_worktree.resolve()
    g5_root = worktree / "runs" / "ksdd2_g5_single_unet_v1"
    train_log = g5_root / "g5_train.stdout.log"
    decision = g5_root / "g5_authorization_decision.json"
    if not g5_root.exists():
        raise FileNotFoundError(f"G5 run root not found: {g5_root}")

    print("Existing frozen G5 monitor started.", flush=True)
    print(f"run_root={g5_root}", flush=True)
    last_completed = -1
    while True:
        count = completed_runs(g5_root)
        log_age = (
            time.time() - train_log.stat().st_mtime
            if train_log.exists()
            else float("inf")
        )
        if count != last_completed:
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"completed={count}/60 log_age={log_age / 60:.1f}m",
                flush=True,
            )
            last_completed = count
        if count >= 60:
            break
        if log_age > args.stall_minutes * 60:
            raise RuntimeError(
                f"G5 appears stalled: no log update for {log_age / 60:.1f} minutes. "
                "Check that no train_g5.py process is running before resuming."
            )
        time.sleep(args.poll_seconds)

    for _ in range(30):
        if decision.exists() and decision.with_suffix(
            decision.suffix + ".sha256"
        ).exists():
            break
        time.sleep(10)
    else:
        analyze = (
            worktree
            / "scripts"
            / "07_ksdd2_g5_bounded_learner"
            / "analyze_g5.py"
        )
        print("Frozen decision missing; running frozen analyzer.", flush=True)
        subprocess.run(
            [sys.executable, str(analyze)],
            cwd=worktree,
            check=True,
        )

    frozen_decision = json.loads(decision.read_text(encoding="utf-8"))
    print(
        f"Frozen decision: {frozen_decision['decision']}",
        flush=True,
    )
    audit = HERE / "audit_g5.py"
    subprocess.run(
        [
            sys.executable,
            str(audit),
            "--g5-root",
            str(g5_root),
        ],
        cwd=ROOT,
        check=True,
    )
    print(
        f"Post-hoc audit completed: "
        f"{g5_root / 'posthoc_skeptical_audit' / 'SKEPTICAL_OBJECTIVE_AUDIT_KO.md'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

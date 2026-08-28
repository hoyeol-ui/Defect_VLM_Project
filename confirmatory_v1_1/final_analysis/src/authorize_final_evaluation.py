"""Create the external final-evaluation authorization after a full fail-closed audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_common import (
    EXPECTED_CELLS,
    FINAL_LOCK,
    FREEZE_COMMIT_SHA,
    PROTOCOL_SHA256,
    atomic_json,
    attempt_registry_sha256,
    build_verified_cell_registry,
    canonical_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from confirmatory_v1_1.final_lock import validate_final_authorization_fixture  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize-final-evaluation", action="store_true")
    parser.add_argument("--authorization-basis", required=True)
    args = parser.parse_args()
    if not args.authorize_final_evaluation:
        raise SystemExit("explicit --authorize-final-evaluation is required")

    authorization_path = FINAL_LOCK / "FINAL_EVALUATION_AUTHORIZATION.json"
    if authorization_path.exists():
        raise SystemExit("authorization record already exists; refusing to replace it")
    registry, internal = build_verified_cell_registry()
    if len(registry) != EXPECTED_CELLS or len(internal) != EXPECTED_CELLS:
        raise SystemExit("verified registry is incomplete")

    attempt_sha = attempt_registry_sha256()
    registry_sha = canonical_sha256(registry)
    authorization = {
        "status": "AUTHORIZED",
        "authorization_scope": "ONE_LOCKED_FINAL_EVALUATION_PER_440_SEALED_BEST_CHECKPOINTS",
        "authorization_basis": args.authorization_basis,
        "authorizer_role": "repository_owner_researcher",
        "advisor_authorization_claimed": False,
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": PROTOCOL_SHA256,
        "freeze_commit_sha": FREEZE_COMMIT_SHA,
        "attempt_registry_sha256": attempt_sha,
        "cell_registry_sha256": registry_sha,
        "manifest_integrity": "PASS",
        "checkpoint_integrity": "PASS",
        "retry_closure": "PASS",
        "final_lock_audit": "PASS",
        "post_final_retraining_or_retuning": "FORBIDDEN",
        "development_metric_substitution": "FORBIDDEN",
    }
    validate_final_authorization_fixture(
        authorization,
        expected_protocol_sha256=PROTOCOL_SHA256,
        expected_freeze_commit_sha=FREEZE_COMMIT_SHA,
        cell_registry=registry,
        expected_attempt_registry_sha256=attempt_sha,
    )
    FINAL_LOCK.mkdir(parents=True, exist_ok=False)
    atomic_json(FINAL_LOCK / "CELL_REGISTRY.json", registry)
    atomic_json(FINAL_LOCK / "INTERNAL_FINAL_EVALUATION_INPUTS.json", internal)
    atomic_json(authorization_path, authorization)
    audit = {
        "status": "PASS",
        "verified_cells": len(registry),
        "unique_cell_ids": len({row["cell_id"] for row in registry}),
        "cell_registry_sha256": registry_sha,
        "attempt_registry_sha256": attempt_sha,
        "protocol_sha256": PROTOCOL_SHA256,
        "freeze_commit_sha": FREEZE_COMMIT_SHA,
        "authorization_record": str(authorization_path.resolve()),
        "actual_final_evaluations_at_authorization": 0,
    }
    atomic_json(FINAL_LOCK / "FINAL_UNLOCK_PREFLIGHT_AUDIT.json", audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()

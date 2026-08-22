"""Fail-closed final registry and primary-metric source guards."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


class FinalLockError(RuntimeError):
    pass


REQUIRED_CELL_COUNT = 440


def _canonical_sha(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_final_authorization_fixture(
    authorization: Mapping[str, object],
    *,
    expected_protocol_sha256: str,
    expected_freeze_commit_sha: str,
    cell_registry: Sequence[Mapping[str, object]],
    expected_attempt_registry_sha256: str,
) -> None:
    """Validate authorization logic; accepting an in-memory fixture never unlocks actual final data."""

    required = {
        "status": "AUTHORIZED",
        "protocol_sha256": expected_protocol_sha256,
        "freeze_commit_sha": expected_freeze_commit_sha,
        "attempt_registry_sha256": expected_attempt_registry_sha256,
        "manifest_integrity": "PASS",
        "checkpoint_integrity": "PASS",
        "retry_closure": "PASS",
        "final_lock_audit": "PASS",
    }
    for key, expected in required.items():
        if authorization.get(key) != expected:
            raise FinalLockError(f"authorization mismatch: {key}")
    if authorization.get("cell_registry_sha256") != _canonical_sha(list(cell_registry)):
        raise FinalLockError("cell registry SHA mismatch")
    if len(cell_registry) != REQUIRED_CELL_COUNT:
        raise FinalLockError(f"exactly {REQUIRED_CELL_COUNT} cells are required")
    ids = [str(row.get("cell_id", "")) for row in cell_registry]
    if len(set(ids)) != REQUIRED_CELL_COUNT or any(not value for value in ids):
        raise FinalLockError("cell IDs must be nonempty and unique")
    for row in cell_registry:
        if row.get("status") != "COMPLETE":
            raise FinalLockError("all cells must be COMPLETE")
        if row.get("protocol_sha256") != expected_protocol_sha256:
            raise FinalLockError("cell protocol SHA mismatch")
        for key in ("code_sha256", "manifest_sha256", "model_sha256", "environment_sha256"):
            if not row.get(key):
                raise FinalLockError(f"cell is missing {key}")


def require_actual_final_access(
    *,
    authorization_flag: bool,
    authorization_path: Path,
    expected_protocol_sha256: str,
    expected_freeze_commit_sha: str,
    cell_registry: Sequence[Mapping[str, object]],
    expected_attempt_registry_sha256: str,
) -> None:
    """Only a real external file plus explicit flag can unlock final access."""

    if not authorization_flag or not authorization_path.is_file():
        raise FinalLockError("actual final evaluation remains locked")
    payload = json.loads(authorization_path.read_text(encoding="utf-8"))
    validate_final_authorization_fixture(
        payload,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_freeze_commit_sha=expected_freeze_commit_sha,
        cell_registry=cell_registry,
        expected_attempt_registry_sha256=expected_attempt_registry_sha256,
    )


def development_access(split: str) -> None:
    if split != "development":
        raise FinalLockError("this entry point exposes development only")


@dataclass(frozen=True)
class PrimaryAULCResult:
    status: str
    value: float | None


def primary_locked_final_aulc(rows: Sequence[Mapping[str, object]]) -> PrimaryAULCResult:
    """Compute primary AULC only from six locked-final values; never interpolate."""

    if len(rows) != 6:
        return PrimaryAULCResult("INCOMPLETE_NOT_ESTIMATED", None)
    ordered = sorted(rows, key=lambda row: int(row.get("round", -1)))
    if [int(row.get("round", -1)) for row in ordered] != list(range(6)):
        return PrimaryAULCResult("INCOMPLETE_NOT_ESTIMATED", None)
    if any(row.get("metric_source") != "locked_final" for row in ordered):
        raise FinalLockError("primary AULC accepts locked_final metrics only")
    if any("development_map5095" in row for row in ordered):
        raise FinalLockError("development metrics cannot enter primary AULC")
    try:
        x = [float(row["budget_fraction"]) for row in ordered]
        y = [float(row["final_map5095"]) for row in ordered]
    except (KeyError, TypeError, ValueError):
        return PrimaryAULCResult("INCOMPLETE_NOT_ESTIMATED", None)
    if any(x[index + 1] <= x[index] for index in range(5)):
        raise FinalLockError("budget fractions must increase")
    span = x[-1] - x[0]
    value = sum((y[i] + y[i + 1]) * 0.5 * (x[i + 1] - x[i]) for i in range(5)) / span
    return PrimaryAULCResult("COMPLETE", float(value))


def fixture_registry(protocol_sha: str, count: int = REQUIRED_CELL_COUNT) -> list[dict[str, str]]:
    return [
        {
            "cell_id": f"fixture-{index:03d}",
            "status": "COMPLETE",
            "protocol_sha256": protocol_sha,
            "code_sha256": "c" * 64,
            "manifest_sha256": "m" * 64,
            "model_sha256": "y" * 64,
            "environment_sha256": "e" * 64,
        }
        for index in range(count)
    ]


def fixture_authorization(protocol_sha: str, freeze_sha: str, registry: Sequence[Mapping[str, object]], attempts_sha: str) -> dict[str, str]:
    return {
        "status": "AUTHORIZED",
        "protocol_sha256": protocol_sha,
        "freeze_commit_sha": freeze_sha,
        "attempt_registry_sha256": attempts_sha,
        "cell_registry_sha256": _canonical_sha(list(registry)),
        "manifest_integrity": "PASS",
        "checkpoint_integrity": "PASS",
        "retry_closure": "PASS",
        "final_lock_audit": "PASS",
    }


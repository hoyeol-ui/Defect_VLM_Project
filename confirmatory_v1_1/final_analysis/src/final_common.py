"""Shared fail-closed helpers for the authorized locked-final evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
WORKTREE = HERE.parents[3]
FULL = ROOT / "full_run"
RUNTIME = FULL / "runtime"
FINAL_LOCK = ROOT / "final_lock"
FINAL_ANALYSIS = ROOT / "final_analysis"
FINAL_EVALUATION = FINAL_ANALYSIS / "final_evaluation"
PROTOCOL = ROOT / "confirmatory_protocol_v1_1.yaml"

PROTOCOL_SHA256 = "6836f452d22c3de26dfa2680c45d0b5cb8f925b1a9274789c4ef81a1770f0b10"
FREEZE_COMMIT_SHA = "56369eaa0108e10d6d9a0f84a21a7e2c54fc4cfc"
EXPECTED_CELLS = 440


class FinalAnalysisGuardError(RuntimeError):
    """Raised before any final access when a frozen invariant is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalAnalysisGuardError(f"expected JSON object: {path}")
    return value


def _resolve_best_checkpoint(
    plan: dict[str, Any], result_path: str | None
) -> tuple[Path, str, dict[str, Any]]:
    reusable = plan.get("reusable_artifact_path")
    if reusable:
        metadata_path = Path(str(plan["reusable_registry_path"]))
        metadata = read_json(metadata_path)
        best_path = Path(str(reusable)) / "weights" / "best.pt"
        expected_sha = str(plan["reusable_best_sha256"])
        if metadata.get("status") != "PASS" or metadata.get("best_pt_sha256") != expected_sha:
            raise FinalAnalysisGuardError(f"reusable checkpoint registry mismatch: {plan['cell_id']}")
    else:
        if not result_path:
            raise FinalAnalysisGuardError(f"missing result path: {plan['cell_id']}")
        metadata = read_json(Path(result_path))
        if metadata.get("status") != "COMPLETE" or metadata.get("cell_id") != plan["cell_id"]:
            raise FinalAnalysisGuardError(f"cell result mismatch: {plan['cell_id']}")
        best_path = Path(str(metadata["output_dir"])) / "weights" / "best.pt"
        expected_sha = str(metadata["best_pt_sha256"])
    if not best_path.is_file() or sha256_file(best_path) != expected_sha:
        raise FinalAnalysisGuardError(f"best checkpoint integrity failure: {plan['cell_id']}")
    return best_path, expected_sha, metadata


def build_verified_cell_registry() -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Build the canonical 440-row registry and verify every selected checkpoint."""

    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise FinalAnalysisGuardError("protocol SHA mismatch")
    integrity = read_json(RUNTIME / "FULL_TRAINING_INTEGRITY_AUDIT.json")
    required_integrity = {
        "status": "PASS",
        "complete": EXPECTED_CELLS,
        "planned": EXPECTED_CELLS,
        "missing_cells": 0,
        "duplicate_cell_ids": 0,
        "corrupt_checkpoint_cells": 0,
        "manifest_mismatches": 0,
        "seed_replacements": 0,
        "unauthorized_config_changes": 0,
        "unresolved_retry_cells": 0,
        "boundary_violations": 0,
        "actual_final_evaluations": 0,
    }
    for key, expected in required_integrity.items():
        if integrity.get(key) != expected:
            raise FinalAnalysisGuardError(f"training integrity mismatch: {key}")

    ledger = read_json(RUNTIME / "FULL_RUN_FREEZE_LEDGER.json")
    if ledger.get("full_run_freeze_commit") != FREEZE_COMMIT_SHA:
        raise FinalAnalysisGuardError("freeze commit mismatch")
    if ledger.get("protocol_sha256") != PROTOCOL_SHA256:
        raise FinalAnalysisGuardError("freeze ledger protocol mismatch")
    code_sha = str(ledger["runner_sha256"])
    environment_sha = canonical_sha256(ledger["environment"])

    database = RUNTIME / "full_run.sqlite"
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT cell_id, ordinal, state, plan_json, result_path "
            "FROM cells ORDER BY ordinal"
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != EXPECTED_CELLS:
        raise FinalAnalysisGuardError("cell registry count mismatch")

    public_registry: list[dict[str, str]] = []
    internal_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cell_id, ordinal, state, plan_json, result_path in rows:
        plan = json.loads(plan_json)
        if cell_id in seen or state != "COMPLETE" or plan.get("cell_id") != cell_id:
            raise FinalAnalysisGuardError(f"invalid completed cell: {cell_id}")
        seen.add(cell_id)
        if int(plan.get("ordinal", -1)) != ordinal:
            raise FinalAnalysisGuardError(f"ordinal mismatch: {cell_id}")
        if plan.get("protocol_sha256") != PROTOCOL_SHA256:
            raise FinalAnalysisGuardError(f"protocol mismatch: {cell_id}")
        manifest = Path(str(plan["manifest_path"]))
        if not manifest.is_file() or sha256_file(manifest) != plan.get("manifest_sha256"):
            raise FinalAnalysisGuardError(f"manifest integrity failure: {cell_id}")
        best_path, best_sha, metadata = _resolve_best_checkpoint(plan, result_path)
        public = {
            "cell_id": str(cell_id),
            "status": "COMPLETE",
            "protocol_sha256": PROTOCOL_SHA256,
            "code_sha256": code_sha,
            "manifest_sha256": str(plan["manifest_sha256"]),
            "model_sha256": best_sha,
            "environment_sha256": environment_sha,
        }
        public_registry.append(public)
        internal_records.append(
            {
                **plan,
                "best_pt_path": str(best_path.resolve()),
                "best_pt_sha256": best_sha,
                "training_result_path": result_path,
                "training_metadata": metadata,
            }
        )
    return public_registry, internal_records


def attempt_registry_sha256() -> str:
    return sha256_file(RUNTIME / "FULL_TRAINING_ATTEMPTS.csv")


"""Durable, outcome-blind, resumable orchestration of the frozen 440-cell study."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any

import psutil


HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[3]
ROOT = WORKTREE / "confirmatory_v1_1"
FULL = ROOT / "full_run"
RUNTIME = FULL / "runtime"
PLAN_PATH = FULL / "planned_cells.jsonl"
CONFIG_PATH = FULL / "full_run_config.json"
LEDGER_PATH = RUNTIME / "FULL_RUN_FREEZE_LEDGER.json"
RUNNER = FULL / "src" / "run_full_cell.py"
PROTOCOL_PATH = ROOT / "confirmatory_protocol_v1_1.yaml"
FINAL_AUTH = ROOT / "final_lock" / "FINAL_EVALUATION_AUTHORIZATION.json"
DB_PATH = RUNTIME / "full_run.sqlite"
STATUS_PATH = RUNTIME / "STATUS.json"
LOCK_PATH = RUNTIME / "orchestrator.lock"
ATTEMPTS_CSV = RUNTIME / "FULL_TRAINING_ATTEMPTS.csv"
OPERATIONAL_LOG = RUNTIME / "FULL_RUN_OPERATIONAL.log"


class FullRunBlocked(RuntimeError):
    pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def log(event: str, payload: dict[str, Any] | None = None) -> None:
    message = {"time_unix": time.time(), "event": event, **(payload or {})}
    line = json.dumps(message, sort_keys=True)
    OPERATIONAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OPERATIONAL_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def gpu_status() -> dict[str, int | None]:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            creationflags=flags,
            timeout=5,
        ).strip().splitlines()[0]
        temperature, utilization, memory = (int(token.strip()) for token in raw.split(","))
        return {"temperature_c": temperature, "utilization_percent": utilization, "memory_used_mib": memory}
    except Exception:
        return {"temperature_c": None, "utilization_percent": None, "memory_used_mib": None}


def acquire_lock() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text(encoding="ascii").strip())
        except Exception:
            pid = -1
        if pid > 0 and psutil.pid_exists(pid):
            raise FullRunBlocked(f"orchestrator already active with PID {pid}")
        LOCK_PATH.unlink()
    descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(str(os.getpid()))


def prevent_sleep(enable: bool) -> None:
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    flags = es_continuous | es_system_required if enable else es_continuous
    result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
    if not result:
        raise FullRunBlocked("SetThreadExecutionState failed")


def load_plan() -> list[dict[str, Any]]:
    plan = [json.loads(line) for line in PLAN_PATH.read_text(encoding="utf-8").splitlines() if line]
    if len(plan) != 440 or len({row["cell_id"] for row in plan}) != 440:
        raise FullRunBlocked("planned registry is not exactly 440 unique cells")
    return plan


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def initialize_db(connection: sqlite3.Connection, plan: list[dict[str, Any]], plan_sha: str) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS cells(
            cell_id TEXT PRIMARY KEY,
            ordinal INTEGER UNIQUE NOT NULL,
            state TEXT NOT NULL,
            attempt_count INTEGER NOT NULL,
            plan_json TEXT NOT NULL,
            result_path TEXT,
            train_runtime_seconds REAL,
            development_eval_runtime_seconds REAL,
            peak_temperature_c REAL,
            artifact_bytes INTEGER,
            updated_unix REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attempts(
            cell_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            status TEXT NOT NULL,
            result_path TEXT,
            started_unix REAL,
            ended_unix REAL,
            PRIMARY KEY(cell_id, attempt)
        );
        """
    )
    observed = connection.execute("SELECT value FROM metadata WHERE key='plan_sha256'").fetchone()
    if observed and observed[0] != plan_sha:
        raise FullRunBlocked("existing database plan SHA mismatch")
    if not observed:
        connection.execute("INSERT INTO metadata(key,value) VALUES('plan_sha256',?)", (plan_sha,))
        connection.execute("INSERT INTO metadata(key,value) VALUES('created_unix',?)", (str(time.time()),))
        for row in plan:
            reusable = row["initial_state"] == "COMPLETE_REUSABLE"
            result_path = row["reusable_registry_path"] if reusable else None
            train_runtime = None
            eval_runtime = None
            artifact_bytes = None
            if reusable:
                operation = json.loads(Path(str(result_path)).read_text(encoding="utf-8"))
                train_runtime = operation["train_runtime_seconds"]
                eval_runtime = operation["development_eval_runtime_seconds"]
                artifact_bytes = operation["artifact_bytes"]
            connection.execute(
                "INSERT INTO cells VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row["cell_id"], row["ordinal"], "COMPLETE" if reusable else "PENDING",
                    1 if reusable else 0, json.dumps(row, sort_keys=True), result_path,
                    train_runtime, eval_runtime, None, artifact_bytes, time.time(),
                ),
            )
    interrupted = connection.execute("SELECT cell_id,attempt_count FROM cells WHERE state='RUNNING'").fetchall()
    for row in interrupted:
        connection.execute("UPDATE cells SET state='INFRA_FAILURE',updated_unix=? WHERE cell_id=?", (time.time(), row["cell_id"]))
        connection.execute(
            "INSERT OR REPLACE INTO attempts VALUES(?,?,?,?,?,?)",
            (row["cell_id"], row["attempt_count"], "INFRA_FAILURE_PROCESS_INTERRUPTED", None, None, time.time()),
        )
    connection.commit()


def append_attempt_csv(payload: dict[str, Any]) -> None:
    fields = ["cell_id", "attempt", "status", "result_path", "started_unix", "ended_unix"]
    new = not ATTEMPTS_CSV.exists()
    with ATTEMPTS_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow({key: payload.get(key, "") for key in fields})


def counts(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute("SELECT state,COUNT(*) AS n FROM cells GROUP BY state").fetchall()
    result = {row["state"]: row["n"] for row in rows}
    return {
        "planned": 440,
        "complete": result.get("COMPLETE", 0),
        "pending": result.get("PENDING", 0),
        "running": result.get("RUNNING", 0),
        "infra_failure": result.get("INFRA_FAILURE", 0),
        "blocked": result.get("BLOCKED", 0),
    }


def update_status(connection: sqlite3.Connection, *, phase: str, current: dict[str, Any] | None = None) -> None:
    runtime_rows = connection.execute("SELECT train_runtime_seconds FROM cells WHERE state='COMPLETE' AND train_runtime_seconds IS NOT NULL").fetchall()
    average = None if not runtime_rows else sum(row[0] for row in runtime_rows) / len(runtime_rows)
    attempts = connection.execute("SELECT COUNT(*) FROM attempts WHERE status!='COMPLETE'").fetchone()[0]
    payload = {
        "phase": phase,
        "updated_unix": time.time(),
        "counts": counts(connection),
        "noncomplete_attempt_records": attempts,
        "average_train_runtime_seconds": average,
        "gpu": gpu_status(),
        "free_disk_gb": shutil.disk_usage(WORKTREE).free / (1024 ** 3),
        "current": current,
        "scientific_outcomes_exposed": False,
        "actual_final_evaluation_count": 0,
    }
    atomic_json(STATUS_PATH, payload)


def state_id(plan: dict[str, Any]) -> str:
    return f"a{int(plan['acquisition_index']):02d}_{plan['strategy_token']}_R{int(plan['round'])}"


def make_spec(plan: dict[str, Any], ledger: dict[str, Any]) -> Path:
    dataset = str(plan["dataset"])
    key = state_id(plan)
    data_yaml = FULL / "yolo_data_cache" / dataset / "states" / f"{key}.yaml"
    if not data_yaml.is_file():
        raise FullRunBlocked(f"data YAML missing: {data_yaml}")
    spec = {
        **plan,
        "data_yaml_path": str(data_yaml.resolve()).replace("\\", "/"),
        "data_yaml_sha256": sha(data_yaml),
        "runner_sha256": ledger["runner_sha256"],
        "full_run_freeze_commit": ledger["full_run_freeze_commit"],
    }
    path = RUNTIME / "cell_specs" / f"{plan['cell_id']}.json"
    atomic_json(path, spec)
    return path


def pre_cell_gate(config: dict[str, Any], ledger: dict[str, Any]) -> None:
    if FINAL_AUTH.exists():
        raise FullRunBlocked("actual final authorization appeared during forbidden training phase")
    if sha(PROTOCOL_PATH) != config["protocol_sha256"]:
        raise FullRunBlocked("protocol SHA changed")
    if sha(PLAN_PATH) != ledger["plan_sha256"] or sha(RUNNER) != ledger["runner_sha256"] or sha(HERE) != ledger["orchestrator_sha256"]:
        raise FullRunBlocked("full-run code or plan SHA changed")
    if shutil.disk_usage(WORKTREE).free / (1024 ** 3) < float(config["minimum_free_disk_gb"]):
        raise FullRunBlocked("free disk below frozen operational floor")
    wait_at = int(config["gpu_temperature_wait_c"])
    resume_at = int(config["gpu_temperature_resume_c"])
    status = gpu_status()
    if status["temperature_c"] is not None and int(status["temperature_c"]) >= wait_at:
        log("GPU_COOLDOWN_BEGIN", status)
        while True:
            time.sleep(30)
            status = gpu_status()
            if status["temperature_c"] is None or int(status["temperature_c"]) <= resume_at:
                log("GPU_COOLDOWN_END", status)
                break


def run_cell(connection: sqlite3.Connection, row: sqlite3.Row, config: dict[str, Any], ledger: dict[str, Any]) -> None:
    plan = json.loads(row["plan_json"])
    attempt = int(row["attempt_count"]) + 1
    spec = make_spec(plan, ledger)
    connection.execute("UPDATE cells SET state='RUNNING',attempt_count=?,updated_unix=? WHERE cell_id=?", (attempt, time.time(), row["cell_id"]))
    connection.commit()
    started = time.time()
    connection.execute("INSERT OR REPLACE INTO attempts VALUES(?,?,?,?,?,?)", (row["cell_id"], attempt, "RUNNING", None, started, None))
    connection.commit()
    update_status(connection, phase="RUNNING", current={"cell_id": row["cell_id"], "attempt": attempt, "epoch": 0})
    log("CELL_START", {"cell_id": row["cell_id"], "attempt": attempt, "completed_before": counts(connection)["complete"]})
    command = [str(config["python_executable"]), str(RUNNER), "--spec", str(spec), "--attempt", str(attempt)]
    process = subprocess.Popen(
        command, cwd=WORKTREE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("PHASE2C_HEARTBEAT "):
            heartbeat = json.loads(line.split(" ", 1)[1])
            update_status(connection, phase="RUNNING", current=heartbeat)
            log("HEARTBEAT", heartbeat)
        elif line.startswith("PHASE2C_CELL_"):
            log("RUNNER_EVENT", {"line": line})
    return_code = process.wait()
    result_path = RUNTIME / "cell_results" / f"{row['cell_id']}_attempt{attempt:02d}.json"
    ended = time.time()
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        status = str(result.get("status", "INFRA_FAILURE"))
    else:
        result = {"status": "INFRA_FAILURE", "error_type": "MISSING_RESULT_AFTER_PROCESS_EXIT", "return_code": return_code}
        status = "INFRA_FAILURE"
    attempt_payload = {"cell_id": row["cell_id"], "attempt": attempt, "status": status, "result_path": str(result_path) if result_path.exists() else "", "started_unix": started, "ended_unix": ended}
    connection.execute("UPDATE attempts SET status=?,result_path=?,ended_unix=? WHERE cell_id=? AND attempt=?", (status, attempt_payload["result_path"], ended, row["cell_id"], attempt))
    append_attempt_csv(attempt_payload)
    if return_code == 0 and status == "COMPLETE":
        if result.get("protocol_sha256") != plan["protocol_sha256"] or result.get("checkpoint_sha256") != plan["checkpoint_sha256"] or result.get("manifest_sha256") != plan["manifest_sha256"] or int(result.get("training_seed", -1)) != int(plan["training_seed"]):
            raise FullRunBlocked(f"completed result seal mismatch: {row['cell_id']}")
        if result.get("outcome_values_exposed") or result.get("final_accessed"):
            raise FullRunBlocked(f"outcome/final boundary violation: {row['cell_id']}")
        connection.execute(
            "UPDATE cells SET state='COMPLETE',result_path=?,train_runtime_seconds=?,development_eval_runtime_seconds=?,peak_temperature_c=?,artifact_bytes=?,updated_unix=? WHERE cell_id=?",
            (str(result_path), result["train_runtime_seconds"], result["development_eval_runtime_seconds"], result["peak_temperature_c"], result["artifact_bytes"], time.time(), row["cell_id"]),
        )
        connection.commit()
        log("CELL_COMPLETE", {"cell_id": row["cell_id"], "attempt": attempt, "completed": counts(connection)["complete"], "train_runtime_seconds": result["train_runtime_seconds"], "peak_temperature_c": result["peak_temperature_c"], "free_disk_gb": result["free_disk_bytes_after"] / (1024 ** 3)})
        update_status(connection, phase="RUNNING", current=None)
        return
    if status in {"INTEGRITY_FAILURE", "OOM"}:
        connection.execute("UPDATE cells SET state='BLOCKED',result_path=?,updated_unix=? WHERE cell_id=?", (str(result_path) if result_path.exists() else None, time.time(), row["cell_id"]))
        connection.commit()
        raise FullRunBlocked(f"non-retriable {status} in {row['cell_id']}")
    if attempt >= int(config["max_infrastructure_attempts_per_cell"]):
        connection.execute("UPDATE cells SET state='BLOCKED',result_path=?,updated_unix=? WHERE cell_id=?", (str(result_path) if result_path.exists() else None, time.time(), row["cell_id"]))
        connection.commit()
        raise FullRunBlocked(f"retry limit reached for {row['cell_id']}")
    connection.execute("UPDATE cells SET state='INFRA_FAILURE',result_path=?,updated_unix=? WHERE cell_id=?", (str(result_path) if result_path.exists() else None, time.time(), row["cell_id"]))
    connection.commit()
    log("SAME_CELL_RETRY_SCHEDULED", {"cell_id": row["cell_id"], "failed_attempt": attempt})
    time.sleep(10)


def verify_completed_cell(plan: dict[str, Any], result_path: str) -> list[str]:
    errors: list[str] = []
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    reusable = plan["initial_state"] == "COMPLETE_REUSABLE"
    expected_status = "PASS" if reusable else "COMPLETE"
    if result.get("status") != expected_status:
        errors.append("status")
    if reusable:
        if result.get("protocol_sha256") != plan["protocol_sha256"]:
            errors.append("protocol_sha256")
        if result.get("cold_start_checkpoint_sha256") != plan["checkpoint_sha256"]:
            errors.append("checkpoint_sha256")
        if str(result.get("training_seed")) != str(plan["training_seed"]):
            errors.append("training_seed")
        if result.get("data_yaml_sha256") != plan.get("reusable_data_yaml_sha256"):
            errors.append("manifest_sha256")
        output = Path(str(plan["reusable_artifact_path"]))
        best_expected = plan["reusable_best_sha256"]
        last_expected = plan["reusable_last_sha256"]
    else:
        for key in ("protocol_sha256", "checkpoint_sha256", "manifest_sha256", "training_seed"):
            if str(result.get(key)) != str(plan.get(key)):
                errors.append(key)
        output = Path(str(result.get("output_dir", "")))
        best_expected = result.get("best_pt_sha256")
        last_expected = result.get("last_pt_sha256")
    best = output / "weights" / "best.pt"
    last = output / "weights" / "last.pt"
    if not best.is_file() or sha(best) != best_expected:
        errors.append("best_pt")
    if not last.is_file() or sha(last) != last_expected:
        errors.append("last_pt")
    if not reusable and (result.get("outcome_values_exposed") or result.get("final_accessed")):
        errors.append("boundary")
    return errors


def final_training_integrity(connection: sqlite3.Connection, plan: list[dict[str, Any]]) -> None:
    rows = connection.execute("SELECT * FROM cells ORDER BY ordinal").fetchall()
    cell_errors: dict[str, list[str]] = {}
    for row, expected in zip(rows, plan, strict=True):
        if row["state"] != "COMPLETE" or not row["result_path"]:
            cell_errors[row["cell_id"]] = ["not_complete"]
            continue
        errors = verify_completed_cell(expected, row["result_path"])
        if errors:
            cell_errors[row["cell_id"]] = errors
    audit = {
        "planned": 440,
        "complete": sum(row["state"] == "COMPLETE" for row in rows),
        "duplicate_cell_ids": len(rows) - len({row["cell_id"] for row in rows}),
        "missing_cells": 440 - len(rows),
        "unauthorized_config_changes": sum("protocol_sha256" in errors or "checkpoint_sha256" in errors for errors in cell_errors.values()),
        "seed_replacements": sum("training_seed" in errors for errors in cell_errors.values()),
        "manifest_mismatches": sum("manifest_sha256" in errors for errors in cell_errors.values()),
        "corrupt_checkpoint_cells": sum("best_pt" in errors or "last_pt" in errors for errors in cell_errors.values()),
        "unresolved_retry_cells": sum(row["state"] in {"INFRA_FAILURE", "RUNNING", "BLOCKED"} for row in rows),
        "boundary_violations": sum("boundary" in errors for errors in cell_errors.values()),
        "actual_final_evaluations": 0,
        "actual_final_authorization_present": FINAL_AUTH.exists(),
        "cell_errors": cell_errors,
    }
    audit["status"] = "PASS" if all(
        audit[key] == 0 for key in (
            "duplicate_cell_ids", "missing_cells", "unauthorized_config_changes",
            "seed_replacements", "manifest_mismatches", "corrupt_checkpoint_cells",
            "unresolved_retry_cells", "boundary_violations", "actual_final_evaluations",
        )
    ) and audit["complete"] == 440 and not audit["actual_final_authorization_present"] and not cell_errors else "FAIL"
    atomic_json(RUNTIME / "FULL_TRAINING_INTEGRITY_AUDIT.json", audit)
    if audit["status"] != "PASS":
        raise FullRunBlocked("440-cell integrity audit failed")
    update_status(connection, phase="TRAINING_COMPLETE_AWAITING_FINAL_AUTHORIZATION", current=None)
    log("FULL_TRAINING_INTEGRITY_PASS", {"complete": 440, "actual_final_evaluations": 0})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    acquire_lock()
    prevent_sleep(True)
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        plan = load_plan()
        if FINAL_AUTH.exists() or config.get("final_evaluation_authorized"):
            raise FullRunBlocked("actual final evaluation must remain locked")
        connection = connect()
        try:
            initialize_db(connection, plan, sha(PLAN_PATH))
            log("ORCHESTRATOR_START", {"pid": os.getpid(), **counts(connection), "sleep_prevention": True})
            if args.preflight_only:
                pre_cell_gate(config, ledger)
                next_row = connection.execute("SELECT * FROM cells WHERE state!='COMPLETE' ORDER BY ordinal LIMIT 1").fetchone()
                if next_row is None:
                    raise FullRunBlocked("preflight found no pending cell")
                make_spec(json.loads(next_row["plan_json"]), ledger)
                update_status(connection, phase="PREFLIGHT_PASS", current={"next_cell_id": next_row["cell_id"]})
                log("PREFLIGHT_PASS", {**counts(connection), "next_cell_id": next_row["cell_id"]})
                return
            update_status(connection, phase="RUNNING", current=None)
            while counts(connection)["complete"] < 440:
                pre_cell_gate(config, ledger)
                row = connection.execute("SELECT * FROM cells WHERE state!='COMPLETE' ORDER BY ordinal LIMIT 1").fetchone()
                if row is None:
                    raise FullRunBlocked("no runnable cell but completion is below 440")
                if row["state"] == "BLOCKED":
                    raise FullRunBlocked(f"blocked cell encountered: {row['cell_id']}")
                run_cell(connection, row, config, ledger)
            final_training_integrity(connection, plan)
        finally:
            connection.close()
    except BaseException as error:
        try:
            if DB_PATH.exists():
                connection = connect()
                try:
                    update_status(connection, phase="BLOCKED", current={"error_type": type(error).__name__, "message": str(error)})
                finally:
                    connection.close()
            log("ORCHESTRATOR_BLOCKED", {"error_type": type(error).__name__, "message": str(error)})
        finally:
            raise
    finally:
        prevent_sleep(False)
        try:
            LOCK_PATH.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()

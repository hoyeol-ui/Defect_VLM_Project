"""Authorized lifecycle primitive for future smoke and worker entry points."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from . import PortabilityError
from .host import ProcessLock, prevent_sleep
from .protocol import verify_authorization
from .storage import Registry, canonical_hash, checked


def policy_parent(study, layout, registry, node, assets):
    if node["kind"] != "acquisition" or node["policy_slot"] != "A":
        raise PortabilityError("only learner A can supply an acquisition checkpoint")
    parent_id = node["depends_on"][0]
    expected = (f"{node['trajectory']}_SHARED_A_R0" if node["round"] == 1
                else f"{node['trajectory']}_Entropy_A_R{node['round'] - 1}")
    if node["depends_on"] != [expected]:
        raise PortabilityError("policy parent must be the same trajectory's previous A")
    if node["round"] == 1:
        reference = study.r0[parent_id]
        mapped = {}
        for role, key in (("checkpoints", "best_sha256"), ("manifests", "manifest_sha256")):
            choices = [a for a in assets if a.get("reference") == parent_id and a["role"] == role]
            if len(choices) != 1 or choices[0]["sha256"] != reference[key]:
                raise PortabilityError("hash-bound historical R0 mapping missing")
            asset = choices[0]
            mapped[role] = checked(layout.input(role, asset["relative"]), asset["sha256"])
        return {"node_id": parent_id, "slot": "A", "round": 0,
                "path": str(mapped["checkpoints"]), "sha256": reference["best_sha256"]}
    result = registry.completed(parent_id)
    if result.get("slot") != "A" or result.get("round") != node["round"] - 1:
        raise PortabilityError("policy checkpoint role/round mismatch")
    from .paths import contained
    best = contained(layout.output_root / registry.namespace, result["best_path"])
    checked(best, result["best_sha256"])
    return {"node_id": parent_id, "slot": "A", "round": result["round"],
            "path": str(best), "sha256": result["best_sha256"]}


def execute(candidate_path, authorization_path, authorization_sha256, study, layout,
            environment, node_id, source_root, source_commit, worker):
    """Run a supplied worker after seals, lock and dependencies pass; never auto-retry.

    Model jobs require WorkerAdapter; arbitrary callbacks are input-check only.
    The adapter validates worker inputs and artifacts before registry completion.
    """
    candidate = verify_authorization(candidate_path, authorization_path, authorization_sha256,
                                     study, layout, environment, node_id, source_root, source_commit)
    namespace = candidate["namespace"]
    node = next(n for n in candidate["plan"] if n["node_id"] == node_id)
    from .adapter import JobContext, WorkerAdapter
    model_job = node["kind"] in ("training", "acquisition")
    if model_job and type(worker) is not WorkerAdapter:
        raise PortabilityError("model-worker integration requires the guarded WorkerAdapter")
    if not model_job and (namespace != "smoke" or node["kind"] != "input_check"):
        raise PortabilityError("unsupported execution scope")
    with ProcessLock(layout.output(namespace, "scheduler.lock")), prevent_sleep():
        registry = Registry(layout, namespace)
        registry.initialize(candidate["plan"], canonical_hash(candidate))
        state = registry.state(node_id)
        if state["state"] == "COMPLETE":
            result = registry.completed(node_id)
            if model_job:
                context = JobContext(namespace, node, candidate, Path(authorization_path), authorization_sha256)
                bridge = worker.prepare(study, layout, registry, context)
                bridge.audit_result(node, result)
                bridge.maybe_gate()
            return result
        if state["state"] != "PENDING" or state["attempt"]:
            raise PortabilityError("interrupted/failed attempt requires explicit reconciliation")
        bridge = None
        if model_job:
            context = JobContext(namespace, node, candidate, Path(authorization_path), authorization_sha256)
            bridge = worker.prepare(study, layout, registry, context)
        else:
            for parent in node["depends_on"]:
                registry.completed(parent)
        registry.transition(node_id, "PENDING", "RUNNING", detail=json.dumps({
            "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
            "candidate_hash": canonical_hash(candidate),
            "authorization_sha256": authorization_sha256}))
        try:
            artifact = worker.run(bridge) if bridge else worker(node, layout.output(namespace, f"jobs/{node_id}/attempt_01"))
            verify_authorization(candidate_path, authorization_path, authorization_sha256,
                                 study, layout, environment, node_id, source_root, source_commit)
            registry.transition(node_id, "RUNNING", "COMPLETE", artifact=artifact)
        except BaseException as error:
            registry.transition(node_id, "RUNNING", "FAILED", detail=f"{type(error).__name__}: {error}")
            raise
        if bridge:
            bridge.maybe_gate()
        return registry.completed(node_id)

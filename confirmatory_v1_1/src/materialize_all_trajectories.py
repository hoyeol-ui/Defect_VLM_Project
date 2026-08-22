"""Seal all GT-free Random/DINO acquisition trajectories from cache A and audit cache B."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[2]
sys.path.insert(0, str(WORKTREE / "confirmatory_v1" / "src"))

from confirmatory_v1.protocol_core import (  # noqa: E402
    build_random_trajectory,
    initial_manifest,
    manifest_sha256,
    validate_embedding_cache,
)


DATASETS = {
    "GC10-DET": {"n": 1605, "initial": 161, "query": 80},
    "NEU-DET": {"n": 1259, "initial": 126, "query": 63},
}
ROOT = WORKTREE / "confirmatory_v1_1"
SEEDS = WORKTREE / "confirmatory_v1" / "data_freeze" / "seed_manifest.csv"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(dataset: str, label: str):
    directory = ROOT / "cache_seals" / dataset / f"cache_{label}"
    ids = np.load(directory / "sample_ids.npy", allow_pickle=False).tolist()
    matrix = np.load(directory / "embeddings.npy", allow_pickle=False)
    return validate_embedding_cache(ids, ids, matrix, expected_dim=384), directory


def faster_farthest_trajectory(ids: list[str], cache, initial: tuple[str, ...], query_size: int) -> tuple[tuple[str, ...], ...]:
    """Same greedy definition with cached max-cosine updates."""

    ordered_ids = list(cache.sample_ids)
    index = {sample_id: position for position, sample_id in enumerate(ordered_ids)}
    matrix = cache.normalized
    cumulative = set(initial)
    states: list[tuple[str, ...]] = [tuple(sorted(cumulative))]
    for _round in range(1, 6):
        remaining = sorted(set(ids) - cumulative)
        cand_idx = np.asarray([index[sample_id] for sample_id in remaining], dtype=np.int64)
        ref_idx = np.asarray([index[sample_id] for sample_id in sorted(cumulative)], dtype=np.int64)
        similarities = matrix[cand_idx] @ matrix[ref_idx].T
        best_cosine = np.max(similarities, axis=1)
        active = np.ones(len(remaining), dtype=bool)
        selected: list[str] = []
        for _ in range(query_size):
            scores = 1.0 - best_cosine
            best_score = float(np.max(scores[active]))
            tied = [remaining[i] for i in np.flatnonzero(active) if np.isclose(float(scores[i]), best_score, rtol=0.0, atol=1e-12)]
            picked = min(tied)
            picked_position = remaining.index(picked)
            active[picked_position] = False
            selected.append(picked)
            new_cosine = matrix[cand_idx] @ matrix[index[picked]]
            best_cosine = np.maximum(best_cosine, new_cosine)
        cumulative.update(selected)
        states.append(tuple(sorted(cumulative)))
    return tuple(states)


def write_ids(path: Path, ids: tuple[str, ...] | list[str]) -> str:
    ordered = sorted(ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ordered) + "\n", encoding="ascii", newline="\n")
    return manifest_sha256(ordered)


def main() -> None:
    started = time.perf_counter()
    with SEEDS.open("r", newline="", encoding="utf-8") as handle:
        seed_rows = list(csv.DictReader(handle))
    ledger: list[dict[str, object]] = []
    cache_audits: dict[str, object] = {}
    dino_trajectory_comparisons = 0
    random_query_events = 0
    dino_query_events = 0
    unique_cumulative = 0

    for dataset, spec in DATASETS.items():
        cache_a, dir_a = load_cache(dataset, "A")
        cache_b, dir_b = load_cache(dataset, "B")
        ids = list(cache_a.sample_ids)
        if len(ids) != spec["n"] or tuple(cache_a.sample_ids) != tuple(cache_b.sample_ids):
            raise RuntimeError("BLOCKED_CACHE_NONDETERMINISM: manifest identity mismatch")
        exact = file_sha(dir_a / "sample_ids.npy") == file_sha(dir_b / "sample_ids.npy") and file_sha(dir_a / "embeddings.npy") == file_sha(dir_b / "embeddings.npy") and file_sha(dir_a / "ordered_manifest.txt") == file_sha(dir_b / "ordered_manifest.txt")
        numeric_max_abs = float(np.max(np.abs(cache_a.normalized - cache_b.normalized)))
        dataset_seeds = [row for row in seed_rows if row["dataset"] == dataset]
        if len(dataset_seeds) != 10:
            raise RuntimeError("seed manifest count mismatch")
        for row in dataset_seeds:
            acquisition_index = int(row["acquisition_index"])
            acquisition_seed = int(row["acquisition_seed"])
            shared = initial_manifest(ids, size=spec["initial"], dataset=dataset, acquisition_seed=acquisition_seed)
            random_states = build_random_trajectory(ids, initial_size=spec["initial"], query_size=spec["query"], rounds=5, dataset=dataset, acquisition_seed=acquisition_seed)
            dino_a = faster_farthest_trajectory(ids, cache_a, shared, spec["query"])
            dino_b = faster_farthest_trajectory(ids, cache_b, shared, spec["query"])
            dino_trajectory_comparisons += 1
            if dino_a != dino_b:
                raise RuntimeError("BLOCKED_CACHE_NONDETERMINISM: DINO trajectory A/B mismatch")
            base = ROOT / "manifest_seals" / dataset / f"a{acquisition_index:02d}"
            shared_sha = write_ids(base / "shared_R0.txt", shared)
            ledger.append({"dataset": dataset, "acquisition_index": acquisition_index, "acquisition_seed": acquisition_seed, "strategy": "SHARED", "round": 0, "kind": "cumulative", "count": len(shared), "sha256": shared_sha, "path": str((base / "shared_R0.txt").relative_to(ROOT))})
            unique_cumulative += 1
            for strategy, states in (("Random-PCG64-v1", random_states), ("FrozenDINO-FF-v1", dino_a)):
                for round_idx in range(1, 6):
                    token = "Random" if strategy.startswith("Random") else "DINO"
                    cumulative_path = base / f"{token}_R{round_idx}.txt"
                    query_path = base / f"{token}_query_R{round_idx}.txt"
                    cumulative_sha = write_ids(cumulative_path, states[round_idx])
                    query = sorted(set(states[round_idx]) - set(states[round_idx - 1]))
                    query_sha = write_ids(query_path, query)
                    if len(query) != spec["query"] or len(states[round_idx]) != spec["initial"] + round_idx * spec["query"]:
                        raise RuntimeError("trajectory budget/query count mismatch")
                    ledger.append({"dataset": dataset, "acquisition_index": acquisition_index, "acquisition_seed": acquisition_seed, "strategy": strategy, "round": round_idx, "kind": "cumulative", "count": len(states[round_idx]), "sha256": cumulative_sha, "path": str(cumulative_path.relative_to(ROOT))})
                    ledger.append({"dataset": dataset, "acquisition_index": acquisition_index, "acquisition_seed": acquisition_seed, "strategy": strategy, "round": round_idx, "kind": "query", "count": len(query), "sha256": query_sha, "path": str(query_path.relative_to(ROOT))})
                    unique_cumulative += 1
                    if token == "Random":
                        random_query_events += 1
                    else:
                        dino_query_events += 1
        cache_audits[dataset] = {
            "cache_A_B_exact_files": exact,
            "numeric_max_abs_difference": numeric_max_abs,
            "classification": "BITWISE_REPRODUCIBLE" if exact else "NUMERICALLY_STABLE_SELECTION_IDENTICAL",
            "cache_A_canonical": True,
        }

    ledger_path = ROOT / "manifest_seals" / "trajectory_ledger.csv"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "acquisition_index", "acquisition_seed", "strategy", "round", "kind", "count", "sha256", "path"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(ledger)
    summary = {
        "status": "PASS",
        "shared_initial_manifests": 20,
        "random_query_events": random_query_events,
        "dino_query_events": dino_query_events,
        "unique_cumulative_states": unique_cumulative,
        "dino_A_B_trajectory_comparisons": dino_trajectory_comparisons,
        "trajectory_ledger_sha256": file_sha(ledger_path),
        "cache_audits": cache_audits,
        "runtime_seconds": time.perf_counter() - started,
        "gt_fields_used": False,
    }
    (ROOT / "manifest_seals" / "trajectory_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PHASE2B_OPERATIONAL_RESULT " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

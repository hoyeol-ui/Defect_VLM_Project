"""Build the immutable 440-cell scientific training plan from frozen manifests."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[3]
V1_SRC = WORKTREE / "confirmatory_v1" / "src"
sys.path.insert(0, str(V1_SRC))

from confirmatory_v1.protocol_core import manifest_sha256, validate_sample_ids  # noqa: E402


ROOT = WORKTREE / "confirmatory_v1_1"
FULL = ROOT / "full_run"
PROTOCOL_SHA = "6836f452d22c3de26dfa2680c45d0b5cb8f925b1a9274789c4ef81a1770f0b10"
MODEL_SHA = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
DATASETS = ("GC10-DET", "NEU-DET")
STRATEGIES = (("Random", "Random-PCG64-v1"), ("DINO", "FrozenDINO-FF-v1"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    validate_sample_ids(ids)
    return ids


def reusable_source(dataset: str, acquisition_index: int, slot: str, token: str, round_idx: int) -> tuple[Path, Path] | None:
    if acquisition_index != 0 or slot != "A" or token not in {"SHARED", "Random"} or round_idx not in {0, 3, 5}:
        return None
    old_id = f"{dataset}_a00_Random_A_R{round_idx}"
    registry = ROOT / "benchmark_registry" / "jobs" / f"{old_id}.json"
    artifact = ROOT / "outcomes_private" / "scientific_feasibility" / old_id
    return registry, artifact


def main() -> None:
    seed_path = WORKTREE / "confirmatory_v1" / "data_freeze" / "seed_manifest.csv"
    with seed_path.open("r", newline="", encoding="utf-8") as handle:
        seed_rows = list(csv.DictReader(handle))
    plan: list[dict[str, object]] = []
    ordinal = 0
    for dataset in DATASETS:
        rows = sorted((row for row in seed_rows if row["dataset"] == dataset), key=lambda row: int(row["acquisition_index"]))
        if len(rows) != 10:
            raise RuntimeError(f"seed count mismatch for {dataset}")
        for row in rows:
            acquisition_index = int(row["acquisition_index"])
            acquisition_seed = int(row["acquisition_seed"])
            manifest_dir = ROOT / "manifest_seals" / dataset / f"a{acquisition_index:02d}"
            for slot, seed_key in (("A", "training_seed_a"), ("B", "training_seed_b")):
                training_seed = int(row[seed_key])
                states: list[tuple[str, str, int, Path]] = [
                    ("SHARED", "SHARED_R0", 0, manifest_dir / "shared_R0.txt")
                ]
                for token, scientific_name in STRATEGIES:
                    states.extend((token, scientific_name, round_idx, manifest_dir / f"{token}_R{round_idx}.txt") for round_idx in range(1, 6))
                for token, scientific_name, round_idx, manifest_path in states:
                    ids = read_ids(manifest_path)
                    expected = (161 + 80 * round_idx) if dataset == "GC10-DET" else (126 + 63 * round_idx)
                    if len(ids) != expected:
                        raise RuntimeError(f"budget mismatch: {manifest_path}")
                    cell_id = f"{dataset}_a{acquisition_index:02d}_{token}_{slot}_R{round_idx}"
                    source = reusable_source(dataset, acquisition_index, slot, token, round_idx)
                    reusable = source is not None
                    reusable_registry = None
                    reusable_artifact = None
                    reusable_best_sha = None
                    reusable_last_sha = None
                    reusable_data_yaml_sha = None
                    if source:
                        reusable_registry, reusable_artifact = source
                        operation = json.loads(reusable_registry.read_text(encoding="utf-8"))
                        best = reusable_artifact / "weights" / "best.pt"
                        last = reusable_artifact / "weights" / "last.pt"
                        if operation.get("status") != "PASS" or operation.get("protocol_sha256") != PROTOCOL_SHA:
                            raise RuntimeError(f"invalid reusable operation: {reusable_registry}")
                        if operation.get("training_seed") != training_seed or operation.get("cold_start_checkpoint_sha256") != MODEL_SHA:
                            raise RuntimeError(f"reusable seed/model mismatch: {cell_id}")
                        if not best.is_file() or not last.is_file() or sha(best) != operation.get("best_pt_sha256") or sha(last) != operation.get("last_pt_sha256"):
                            raise RuntimeError(f"reusable checkpoint integrity failure: {cell_id}")
                        old_data_yaml = ROOT / "yolo_data_cache" / dataset / f"dataset_R{round_idx}.yaml"
                        old_train_list = ROOT / "yolo_data_cache" / dataset / "lists" / f"train_R{round_idx}.txt"
                        old_ids = sorted(
                            Path(line.strip()).stem
                            for line in old_train_list.read_text(encoding="utf-8").splitlines()
                            if line.strip()
                        )
                        if sha(old_data_yaml) != operation.get("data_yaml_sha256") or manifest_sha256(old_ids) != manifest_sha256(ids):
                            raise RuntimeError(f"reusable data/manifest mismatch: {cell_id}")
                        reusable_best_sha = sha(best)
                        reusable_last_sha = sha(last)
                        reusable_data_yaml_sha = sha(old_data_yaml)
                    plan.append(
                        {
                            "ordinal": ordinal,
                            "cell_id": cell_id,
                            "dataset": dataset,
                            "acquisition_index": acquisition_index,
                            "acquisition_seed": acquisition_seed,
                            "strategy_token": token,
                            "strategy": scientific_name,
                            "round": round_idx,
                            "training_seed_slot": slot,
                            "training_seed": training_seed,
                            "manifest_path": str(manifest_path.resolve()).replace("\\", "/"),
                            "manifest_sha256": manifest_sha256(ids),
                            "labeled_count": len(ids),
                            "protocol_sha256": PROTOCOL_SHA,
                            "checkpoint_sha256": MODEL_SHA,
                            "initial_state": "COMPLETE_REUSABLE" if reusable else "PENDING",
                            "reusable_registry_path": None if reusable_registry is None else str(reusable_registry.resolve()).replace("\\", "/"),
                            "reusable_artifact_path": None if reusable_artifact is None else str(reusable_artifact.resolve()).replace("\\", "/"),
                            "reusable_best_sha256": reusable_best_sha,
                            "reusable_last_sha256": reusable_last_sha,
                            "reusable_data_yaml_sha256": reusable_data_yaml_sha,
                            "final_evaluation_authorized": False,
                        }
                    )
                    ordinal += 1
    if len(plan) != 440 or len({row["cell_id"] for row in plan}) != 440:
        raise RuntimeError("planned registry must contain exactly 440 unique cells")
    if sum(row["initial_state"] == "COMPLETE_REUSABLE" for row in plan) != 6:
        raise RuntimeError("reusable count must be exactly six")
    plan_path = FULL / "planned_cells.jsonl"
    plan_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in plan), encoding="utf-8", newline="\n")
    summary = {
        "status": "PASS",
        "planned": len(plan),
        "completed_reusable": 6,
        "pending_new": 434,
        "duplicate_cell_ids": 0,
        "datasets": {dataset: sum(row["dataset"] == dataset for row in plan) for dataset in DATASETS},
        "round0_shared": sum(row["round"] == 0 for row in plan),
        "post_round_random": sum(row["strategy_token"] == "Random" for row in plan),
        "post_round_dino": sum(row["strategy_token"] == "DINO" for row in plan),
        "plan_sha256": sha(plan_path),
        "final_evaluation_authorized": False,
    }
    (FULL / "planned_registry_seal.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

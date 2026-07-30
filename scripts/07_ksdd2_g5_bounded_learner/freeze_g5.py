"""Create the outcome-blind G5 freeze, exact manifests, and run plan."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from g5_common import (
    CONFIG_PATH,
    HERE,
    ROOT,
    load_config,
    read_csv,
    relative_to_root,
    require_clean_git,
    require_train_row,
    resolve_data_path,
    resolve_source,
    run_root,
    runtime_metadata,
    seal_file,
    set_deterministic,
    sha256_file,
    state_dict_sha256,
    verify_seal,
    write_csv,
    write_json,
)
from learner import build_model, parameter_count
from validate_prefreeze import validate_and_build_ledger


TRAIN_FIELDS = [
    "sample_id",
    "official_split",
    "source_role",
    "image_path",
    "posthoc_mask_path",
    "image_sha256",
    "mask_sha256",
]


def _hashed_row(
    source: dict[str, str],
    role: str,
    cache: dict[Path, str],
) -> dict[str, Any]:
    require_train_row(source)
    image = resolve_data_path(source["image_path"])
    mask = resolve_data_path(source["posthoc_mask_path"])
    for path in (image, mask):
        if path not in cache:
            cache[path] = sha256_file(path)
    return {
        "sample_id": source["sample_id"],
        "official_split": "train",
        "source_role": role,
        "image_path": source["image_path"],
        "posthoc_mask_path": source["posthoc_mask_path"],
        "image_sha256": cache[image],
        "mask_sha256": cache[mask],
    }


def _protocol_files() -> list[Path]:
    return sorted(
        [
            path
            for path in HERE.iterdir()
            if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".md", ".txt"}
        ],
        key=lambda path: path.name,
    )


def main() -> int:
    config = load_config()
    out = run_root(config)
    freeze_path = out / config["outputs"]["freeze_manifest"]
    if freeze_path.exists():
        raise RuntimeError(f"G5 is already frozen: {freeze_path}")
    commit = require_clean_git()
    checks, _ = validate_and_build_ledger(config)
    if checks["paired_training_runs"] != config["comparison"]["total_training_runs"]:
        raise RuntimeError("Run-count mismatch before freeze")
    ledger_path = out / config["outputs"]["input_ledger"]
    if not verify_seal(ledger_path):
        raise RuntimeError("Generate and seal g5_input_ledger.json before freeze")
    qa_path = out / "qa" / "preflight.json"
    overlay_path = out / "qa" / "image_mask_alignment_overlay.png"
    if not verify_seal(qa_path) or not verify_seal(overlay_path):
        raise RuntimeError("Preflight QA and visual alignment must pass before freeze")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if qa["status"] != "PASS" or qa["official_test_accessed"] is not False:
        raise RuntimeError("Preflight QA did not authorize freeze")

    source_root = resolve_source(config)
    acquisition = read_csv(source_root / "acquisition_manifest.csv")
    development = read_csv(source_root / "development_manifest.csv")
    acquisition_by_id = {row["sample_id"]: row for row in acquisition}
    hash_cache: dict[Path, str] = {}
    frozen_inputs = out / "frozen_inputs"
    frozen_inputs.mkdir(parents=True, exist_ok=True)

    development_rows = [
        _hashed_row(row, "development_evaluation_only", hash_cache)
        for row in development
    ]
    development_target = frozen_inputs / "development_evaluation_manifest.csv"
    write_csv(development_target, development_rows, TRAIN_FIELDS)
    seal_file(development_target)

    manifest_paths: dict[tuple[int, str], Path] = {}
    for acquisition_seed in config["comparison"]["acquisition_seeds"]:
        seed = int(acquisition_seed)
        initial = read_csv(source_root / "initial_sets" / f"initial_seed{seed:02d}.csv")
        for strategy in config["comparison"]["strategies"]:
            selected = read_csv(
                source_root / "selections" / f"selected_{strategy}_seed{seed:02d}.csv"
            )
            ids = [row["sample_id"] for row in initial] + [
                row["sample_id"] for row in selected
            ]
            if len(ids) != 200 or len(set(ids)) != 200:
                raise RuntimeError(f"Invalid paired training IDs for seed={seed}, {strategy}")
            rows = [
                _hashed_row(
                    acquisition_by_id[sample_id],
                    "initial" if index < 100 else f"query_{strategy}",
                    hash_cache,
                )
                for index, sample_id in enumerate(ids)
            ]
            target = (
                frozen_inputs
                / f"train_acq{seed:02d}_{strategy}.csv"
            )
            write_csv(target, rows, TRAIN_FIELDS)
            seal_file(target)
            manifest_paths[(seed, strategy)] = target

    run_plan_rows: list[dict[str, Any]] = []
    initial_hashes: dict[str, str] = {}
    model_parameters = None
    for training_seed in config["comparison"]["training_seeds"]:
        training_seed = int(training_seed)
        set_deterministic(training_seed)
        model = build_model(config)
        initial_hashes[str(training_seed)] = state_dict_sha256(model.state_dict())
        model_parameters = parameter_count(model)
    for acquisition_seed in config["comparison"]["acquisition_seeds"]:
        for training_seed in config["comparison"]["training_seeds"]:
            for strategy in config["comparison"]["strategies"]:
                manifest = manifest_paths[(int(acquisition_seed), strategy)]
                run_plan_rows.append(
                    {
                        "run_id": (
                            f"acq{int(acquisition_seed):02d}_{strategy}_"
                            f"train{int(training_seed)}"
                        ),
                        "acquisition_seed": int(acquisition_seed),
                        "strategy": strategy,
                        "training_seed": int(training_seed),
                        "training_manifest": relative_to_root(manifest),
                        "training_manifest_sha256": sha256_file(manifest),
                        "initial_state_sha256": initial_hashes[str(int(training_seed))],
                    }
                )
    if len(run_plan_rows) != int(config["comparison"]["total_training_runs"]):
        raise RuntimeError("Frozen run plan does not contain exactly 60 runs")
    run_plan = frozen_inputs / "run_plan.csv"
    write_csv(
        run_plan,
        run_plan_rows,
        [
            "run_id",
            "acquisition_seed",
            "strategy",
            "training_seed",
            "training_manifest",
            "training_manifest_sha256",
            "initial_state_sha256",
        ],
    )
    seal_file(run_plan)

    files = _protocol_files() + [
        ledger_path,
        qa_path,
        overlay_path,
        development_target,
        run_plan,
        *manifest_paths.values(),
    ]
    frozen_files = [
        {
            "path": relative_to_root(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(files), key=lambda path: str(path))
    ]
    freeze = {
        "protocol_id": config["protocol"]["id"],
        "status": "FROZEN_PENDING_EXECUTION",
        "official_test_accessed": False,
        "official_test_authorized": False,
        "outcome_observed_before_freeze": False,
        "git": {"commit": commit, "worktree_clean": True},
        "runtime": runtime_metadata(),
        "model": {
            "name": config["learner"]["name"],
            "parameter_count": model_parameters,
            "initial_state_sha256_by_training_seed": initial_hashes,
        },
        "design": {
            "strategies": config["comparison"]["strategies"],
            "acquisition_seeds": config["comparison"]["acquisition_seeds"],
            "training_seeds": config["comparison"]["training_seeds"],
            "total_runs": len(run_plan_rows),
            "inferential_unit": config["statistics"]["inferential_unit"],
            "training_seed_aggregation": config["comparison"][
                "training_seed_aggregation"
            ],
        },
        "inputs": {
            "input_ledger_sha256": sha256_file(ledger_path),
            "development_manifest_sha256": sha256_file(development_target),
            "run_plan_sha256": sha256_file(run_plan),
            "unique_hashed_dataset_files": len(hash_cache),
        },
        "frozen_files": frozen_files,
    }
    write_json(freeze_path, freeze)
    seal_file(freeze_path)
    print(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

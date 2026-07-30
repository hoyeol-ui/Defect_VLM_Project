"""Read-only upstream validation and G5 input-ledger preparation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "config.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_seal(path: Path) -> bool:
    seal = path.with_suffix(path.suffix + ".sha256")
    if not seal.exists():
        return False
    return seal.read_text(encoding="ascii").split()[0].lower() == sha256_file(path)


def resolve_source(config: dict[str, Any], override: str | None = None) -> Path:
    value = Path(override or config["upstream"]["default_source_run_root"])
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _require_hash(path: Path, expected: str, role: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{role} missing: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{role} SHA-256 mismatch: {actual} != {expected}")


def _frozen_source_hashes(freeze: dict[str, Any], source: Path) -> dict[Path, str]:
    run_prefix = "runs/ksdd2_patchcore_prospective_v1/"
    mapped: dict[Path, str] = {}
    for item in freeze["files"]:
        relative = str(item["path"])
        if relative.startswith(run_prefix):
            mapped[source / relative.removeprefix(run_prefix)] = str(item["sha256"])
    return mapped


def validate_and_build_ledger(
    config: dict[str, Any],
    *,
    source_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = resolve_source(config, source_override)
    if not source.exists():
        raise FileNotFoundError(f"Upstream G1-G4 run root missing: {source}")

    upstream = config["upstream"]
    freeze_path = source / "freeze_manifest.json"
    decision_path = source / "authorization_decision.json"
    per_seed_path = source / "g2_g4" / "per_seed_metrics.csv"
    _require_hash(freeze_path, upstream["freeze_manifest_sha256"], "upstream freeze")
    _require_hash(
        decision_path,
        upstream["authorization_decision_sha256"],
        "upstream authorization",
    )
    _require_hash(
        per_seed_path,
        upstream["per_seed_metrics_sha256"],
        "upstream per-seed metrics",
    )
    if not verify_seal(freeze_path) or not verify_seal(decision_path) or not verify_seal(per_seed_path):
        raise RuntimeError("One or more upstream summary seals are invalid")

    freeze = read_json(freeze_path)
    decision = read_json(decision_path)
    if freeze["official_freeze"] is not True or freeze["development_only"] is not False:
        raise RuntimeError("Upstream run is not an official freeze")
    if freeze["git"]["commit"] != upstream["frozen_git_commit"]:
        raise RuntimeError("Upstream frozen Git commit changed")
    if decision["protocol_id"] != upstream["protocol_id"]:
        raise RuntimeError("Unexpected upstream protocol ID")
    if decision["authorization"] != upstream["required_authorization"]:
        raise RuntimeError("G1-G4 did not authorize the bounded screen")
    for gate, expected in upstream["required_gates"].items():
        if decision[gate] != expected:
            raise RuntimeError(f"Upstream {gate} is {decision[gate]}, expected {expected}")
    if decision["official_test_use_authorized"] is not False:
        raise RuntimeError("Unexpected official-test authorization in upstream decision")

    archive = ROOT / config["dataset"]["archive_path"]
    _require_hash(archive, config["dataset"]["archive_sha256"], "KSDD2 archive")

    frozen_hashes = _frozen_source_hashes(freeze, source)
    acquisition_path = source / "acquisition_manifest.csv"
    development_path = source / "development_manifest.csv"
    for path in (acquisition_path, development_path):
        if path not in frozen_hashes or sha256_file(path) != frozen_hashes[path]:
            raise RuntimeError(f"Frozen upstream manifest changed: {path}")
    acquisition = read_csv(acquisition_path)
    development = read_csv(development_path)
    acquisition_ids = {row["sample_id"] for row in acquisition}
    if len(acquisition) != 1865 or len(development) != config["dataset"]["evaluation_images"]:
        raise RuntimeError("Unexpected acquisition/development count")
    if any(row["official_split"] != "train" for row in acquisition + development):
        raise RuntimeError("Official-test row entered a G5 input manifest")

    input_files = [freeze_path, decision_path, per_seed_path, acquisition_path, development_path]
    seed_checks: dict[str, Any] = {}
    for seed in [int(value) for value in config["comparison"]["acquisition_seeds"]]:
        initial_path = source / "initial_sets" / f"initial_seed{seed:02d}.csv"
        if initial_path not in frozen_hashes or sha256_file(initial_path) != frozen_hashes[initial_path]:
            raise RuntimeError(f"Frozen initial set changed for seed {seed}")
        initial = read_csv(initial_path)
        initial_ids = {row["sample_id"] for row in initial}
        if len(initial) != config["dataset"]["initial_images_per_seed"]:
            raise RuntimeError(f"Initial-set size mismatch for seed {seed}")
        if not initial_ids <= acquisition_ids:
            raise RuntimeError(f"Initial set {seed} contains non-acquisition IDs")
        input_files.append(initial_path)

        arm_checks: dict[str, Any] = {}
        for strategy in config["comparison"]["strategies"]:
            selected_path = source / "selections" / f"selected_{strategy}_seed{seed:02d}.csv"
            if not verify_seal(selected_path):
                raise RuntimeError(f"Selection seal invalid: {selected_path}")
            selected = read_csv(selected_path)
            selected_ids = {row["sample_id"] for row in selected}
            if len(selected) != config["dataset"]["query_images_per_strategy"]:
                raise RuntimeError(f"{strategy} query size mismatch for seed {seed}")
            if len(selected_ids) != len(selected):
                raise RuntimeError(f"{strategy} query contains duplicate IDs for seed {seed}")
            if selected_ids & initial_ids:
                raise RuntimeError(f"{strategy} query overlaps initial set for seed {seed}")
            if not selected_ids <= acquisition_ids:
                raise RuntimeError(f"{strategy} query contains non-acquisition IDs for seed {seed}")
            arm_checks[strategy] = {
                "initial_count": len(initial_ids),
                "query_count": len(selected_ids),
                "training_count": len(initial_ids | selected_ids),
                "initial_query_overlap": 0,
                "selection_seal_valid": True,
            }
            input_files.append(selected_path)
        seed_checks[f"{seed:02d}"] = arm_checks

    input_files = sorted(set(input_files), key=lambda path: path.as_posix())
    ledger = {
        "protocol_id": config["protocol"]["id"],
        "upstream_protocol_id": upstream["protocol_id"],
        "upstream_frozen_git_commit": upstream["frozen_git_commit"],
        "source_run_root": str(source),
        "official_test_content_accessed": False,
        "development_masks_authorized_for_evaluation_only": True,
        "files": [
            {
                "role": (
                    "selection_manifest"
                    if "selected_" in path.name
                    else "initial_manifest"
                    if path.name.startswith("initial_seed")
                    else path.name
                ),
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_files
        ],
        "dataset_archive": {
            "path": str(archive.resolve()),
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        },
    }
    checks = {
        "upstream_official_freeze": True,
        "upstream_authorization": decision["authorization"],
        "upstream_gates": {gate: decision[gate] for gate in ("g1", "g2", "g3", "g4")},
        "official_test_access": False,
        "acquisition_count": len(acquisition),
        "development_count": len(development),
        "paired_training_runs": config["comparison"]["total_training_runs"],
        "seed_checks": seed_checks,
        "input_file_count": len(input_files),
    }
    return checks, ledger


def write_ledger(config: dict[str, Any], ledger: dict[str, Any]) -> Path:
    output = ROOT / config["outputs"]["run_root"] / config["outputs"]["input_ledger"]
    if (output.parent / config["outputs"]["freeze_manifest"]).exists():
        raise RuntimeError("G5 is already frozen; input ledger will not be overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    seal = output.with_suffix(output.suffix + ".sha256")
    seal.write_text(f"{sha256_file(output)}  {output.name}\n", encoding="ascii")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-root")
    parser.add_argument("--write-ledger", action="store_true")
    args = parser.parse_args()
    config = load_config()
    checks, ledger = validate_and_build_ledger(
        config,
        source_override=args.source_run_root,
    )
    result: dict[str, Any] = {"checks": checks, "ready_to_freeze": True}
    if args.write_ledger:
        path = write_ledger(config, ledger)
        result["ledger"] = str(path)
        result["ledger_sha256"] = sha256_file(path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

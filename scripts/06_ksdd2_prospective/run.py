"""Command-line entry point for the KSDD2 G1-G4 prospective protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import analyze_g1, analyze_g2_g4, create_selections, require_g1_pass
from common import (
    build_freeze_manifest,
    canonical_json,
    ensure_pre_freeze_clean,
    load_config,
    project_path,
    read_json,
    require_official_freeze,
    run_root,
    seal_file,
    sha256_file,
    verify_frozen_hashes,
    verify_seal,
    write_json,
)
from dataset_protocol import (
    prepare_manifests,
    reveal_initial_labels,
    verify_prepared_manifests,
)
from patchcore_scoring import score_g1, score_selection_seed


def download_official(config: dict[str, Any], *, extract: bool) -> dict[str, Any]:
    dataset = config["dataset"]
    archive = project_path(dataset["archive_path"])
    archive.parent.mkdir(parents=True, exist_ok=True)
    expected = str(dataset["archive_sha256"]).lower()
    downloaded = False
    if archive.exists():
        actual = sha256_file(archive).lower()
        if actual != expected:
            raise RuntimeError(
                f"Existing archive hash mismatch: {archive}. Move it aside before retrying."
            )
    else:
        partial = archive.with_suffix(archive.suffix + ".partial")
        if partial.exists():
            raise RuntimeError(f"Partial download already exists: {partial}")
        with urllib.request.urlopen(dataset["official_url"]) as response, partial.open("wb") as handle:
            while block := response.read(4 * 1024 * 1024):
                handle.write(block)
        actual = sha256_file(partial).lower()
        if actual != expected:
            raise RuntimeError(f"Downloaded archive SHA-256 mismatch: {actual}")
        os.replace(partial, archive)
        downloaded = True

    extracted = False
    if extract:
        data_root = project_path(dataset["data_root"])
        train_dir = data_root / dataset["train_dir"]
        test_dir = data_root / dataset["final_test_dir"]
        if train_dir.exists() or test_dir.exists():
            if not train_dir.exists() or not test_dir.exists():
                raise RuntimeError(
                    "Only part of the extraction target exists. Resolve it manually before retrying."
                )
        else:
            data_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(data_root)
            extracted = True
    return {
        "archive": archive.relative_to(project_path(".")).as_posix(),
        "archive_sha256": expected,
        "downloaded_now": downloaded,
        "extracted_now": extracted,
    }


def freeze(config: dict[str, Any], *, allow_dirty: bool) -> dict[str, Any]:
    verify_prepared_manifests(config)
    ensure_pre_freeze_clean(config)
    manifest = build_freeze_manifest(config, allow_dirty=allow_dirty)
    path = run_root(config) / config["outputs"]["freeze_manifest"]
    write_json(path, manifest)
    seal_file(path)
    return {
        "freeze_manifest": path.relative_to(project_path(".")).as_posix(),
        "official_freeze": manifest["official_freeze"],
        "development_only": manifest["development_only"],
        "file_count": len(manifest["files"]),
    }


def _seeds(config: dict[str, Any], args: argparse.Namespace) -> list[int]:
    configured = [int(value) for value in config["acquisition"]["seeds"]]
    if getattr(args, "all_seeds", False):
        return configured
    if args.seed not in configured:
        raise ValueError(f"Seed {args.seed} is not in the frozen seed list {configured}")
    return [int(args.seed)]


def score_selection(config: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    require_official_freeze(config)
    require_g1_pass(config)
    results: list[dict[str, Any]] = []
    for seed in seeds:
        reveal_initial_labels(config, seed)
        results.append(score_selection_seed(config, seed))
    return {"scored_seeds": seeds, "runs": results}


def select(config: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    require_official_freeze(config)
    require_g1_pass(config)
    return {
        "selected_seeds": seeds,
        "selections": [create_selections(config, seed) for seed in seeds],
    }


def verify_run(config: dict[str, Any]) -> dict[str, Any]:
    freeze_manifest = require_official_freeze(config)
    verify_frozen_hashes(config, freeze_manifest)
    prepared = verify_prepared_manifests(config)
    out = run_root(config)
    checks: dict[str, Any] = {
        "official_freeze": freeze_manifest["official_freeze"] is True,
        "frozen_files_unchanged": True,
        "prepared_manifests": all(prepared.values()),
        "official_test_image_content_read": freeze_manifest["locks"][
            "official_test_image_content_read"
        ],
        "official_test_mask_content_read": freeze_manifest["locks"][
            "official_test_mask_content_read"
        ],
        "detector_training_performed": freeze_manifest["locks"][
            "detector_training_performed"
        ],
    }
    decision_path = out / config["outputs"]["authorization_decision"]
    g1_path = out / "g1" / "g1_decision.json"
    if decision_path.exists():
        if not verify_seal(decision_path):
            raise RuntimeError("Final authorization decision seal is invalid")
        decision = read_json(decision_path)
        status = decision["authorization"]
    elif g1_path.exists():
        if not verify_seal(g1_path):
            raise RuntimeError("G1 decision seal is invalid")
        decision = read_json(g1_path)
        status = "G1_PASS_PENDING_G2_G4" if decision["decision"] == "PASS" else "STOP"
    else:
        status = "FROZEN_PENDING_G1"
    checks["lock_conformance"] = (
        checks["official_test_image_content_read"] is False
        and checks["official_test_mask_content_read"] is False
        and checks["detector_training_performed"] is False
    )
    return {"status": status, "checks": checks}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen KSDD2 PatchCore G1-G4 authorization workflow"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Download and verify the official archive")
    download.add_argument("--extract", action="store_true")
    sub.add_parser("prepare", help="Create target-blind manifests and initial sets")
    sub.add_parser("verify-pre-freeze", help="Validate Stage-0 artifacts without reading outcomes")
    freeze_parser = sub.add_parser("freeze", help="Hash-lock the protocol from a clean Git state")
    freeze_parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Create a development-only freeze that cannot authorize scoring",
    )
    reveal = sub.add_parser("reveal-initial", help="Read initial-set masks after official freeze")
    reveal.add_argument("--seed", type=int, required=True)
    sub.add_parser("score-g1", help="Score the locked development set")
    sub.add_parser("analyze-g1", help="Reveal development masks and evaluate G1")
    for command in ("score-selection", "select"):
        item = sub.add_parser(command)
        group = item.add_mutually_exclusive_group(required=True)
        group.add_argument("--seed", type=int)
        group.add_argument("--all-seeds", action="store_true")
    sub.add_parser("analyze-g2-g4", help="Reveal selection outcomes and evaluate G2-G4")
    sub.add_parser("verify-run", help="Verify seals, locks, and current authorization state")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config()
    command = args.command
    if command == "download":
        result = download_official(config, extract=args.extract)
    elif command == "prepare":
        result = prepare_manifests(config)
    elif command == "verify-pre-freeze":
        if (run_root(config) / config["outputs"]["freeze_manifest"]).exists():
            raise RuntimeError("Run is already frozen; use verify-run")
        ensure_pre_freeze_clean(config)
        result = {
            "archive_sha256_matches": sha256_file(
                project_path(config["dataset"]["archive_path"])
            ).lower()
            == str(config["dataset"]["archive_sha256"]).lower(),
            "manifest_checks": verify_prepared_manifests(config),
            "outcome_artifacts_absent": True,
            "ready_to_freeze": True,
        }
    elif command == "freeze":
        result = freeze(config, allow_dirty=args.allow_dirty)
    elif command == "reveal-initial":
        require_official_freeze(config)
        if int(args.seed) != int(config["acquisition"]["g1_reference_seed"]):
            require_g1_pass(config)
        path = reveal_initial_labels(config, int(args.seed))
        result = {"initial_labels": path.relative_to(project_path(".")).as_posix()}
    elif command == "score-g1":
        require_official_freeze(config)
        result = score_g1(config)
    elif command == "analyze-g1":
        require_official_freeze(config)
        result = analyze_g1(config)
    elif command == "score-selection":
        result = score_selection(config, _seeds(config, args))
    elif command == "select":
        result = select(config, _seeds(config, args))
    elif command == "analyze-g2-g4":
        require_official_freeze(config)
        result = analyze_g2_g4(config)
    elif command == "verify-run":
        result = verify_run(config)
    else:
        raise AssertionError(command)
    sys.stdout.write(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

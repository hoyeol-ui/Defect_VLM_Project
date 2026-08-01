"""Generic read-only preflight reporting for an explicit frozen-run config."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .config import REQUIRED_SPLITS, FrozenRunConfig, load_frozen_run_config
from .data.identity import compute_file_sha256
from .data.manifests import (
    ManifestIndex,
    load_manifest_index,
    summarize_split_overlaps,
    validate_split_leakage,
)

HARNESS_VERSION = "1.1"


def _report_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return str(resolved)


def _git_value(project_root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def validate_frozen_run(
    config_path: str | Path,
) -> tuple[FrozenRunConfig, Mapping[str, ManifestIndex], tuple[dict[str, Any], ...]]:
    """Validate seals, identities, duplicates, and all pairwise split boundaries."""

    config = load_frozen_run_config(config_path, verify_manifest_hashes=True)
    indexes = {
        split: load_manifest_index(config.manifests[split].path, split)
        for split in REQUIRED_SPLITS
    }
    overlaps = summarize_split_overlaps(indexes)
    validate_split_leakage(indexes)
    return config, MappingProxyType(indexes), overlaps


def build_preflight_report(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Return a JSON-serializable PASS report or raise before downstream work."""

    root = Path(project_root).resolve(strict=True)
    config, indexes, overlaps = validate_frozen_run(config_path)
    manifests = {}
    for split in REQUIRED_SPLITS:
        reference = config.manifests[split]
        index = indexes[split]
        manifests[split] = {
            "path": _report_path(reference.path, root),
            "manifest_sha256": reference.manifest_sha256,
            "verified_actual_sha256": compute_file_sha256(reference.path),
            "row_count": index.row_count,
            "unique_canonical_ids": len(index.canonical_ids),
            "unique_image_sha256s": len(index.image_sha256s),
            "duplicate_canonical_ids": index.row_count - len(index.canonical_ids),
        }
    return {
        "status": "PASS",
        "run_id": config.run_id,
        "harness_version": HARNESS_VERSION,
        "config_path": _report_path(config.config_path, root),
        "config_sha256": compute_file_sha256(config.config_path),
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "git_branch": _git_value(root, "branch", "--show-current"),
        "git_dirty": bool(_git_value(root, "status", "--porcelain") not in {"", "unknown"}),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "manifests": manifests,
        "split_overlaps": list(overlaps),
        "data_preflight_passed": True,
        "selection_authorized": False,
        "training_authorized": False,
        "note": "PASS verifies only the frozen data boundary. A separate protocol must explicitly authorize selection or training.",
    }

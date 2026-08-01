"""Strict, explicit configuration for frozen active-learning runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .data.identity import compute_file_sha256, validate_sha256

REQUIRED_SPLITS = ("train", "validation", "test")
_TOP_LEVEL_KEYS = frozenset({"run_id", "manifests"})
_MANIFEST_KEYS = frozenset({"path", "manifest_sha256"})
_FORBIDDEN_DISCOVERY_TERMS = (
    "latest",
    "newest",
    "most_recent",
    "mtime",
    "modified_time",
    "auto_discover",
    "autodiscover",
)
_GLOB_CHARACTERS = frozenset("*?[]{}")


@dataclass(frozen=True, slots=True)
class ManifestReference:
    """An explicit, content-sealed manifest reference."""

    split: str
    path: Path
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenRunConfig:
    """Validated frozen-run configuration with an immutable split mapping."""

    run_id: str
    manifests: Mapping[str, ManifestReference]
    config_path: Path


def _load_document(config_path: Path) -> Any:
    suffix = config_path.suffix.lower()
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read frozen-run config {config_path}: {exc}") from exc

    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON frozen-run config {config_path}: {exc}") from exc
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
            raise RuntimeError(
                "YAML frozen-run configs require PyYAML. Install it with `python -m pip install PyYAML`."
            ) from exc
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Invalid YAML frozen-run config {config_path}: {exc}") from exc
    raise RuntimeError(
        f"Unsupported frozen-run config format {suffix or '<none>'!r} for {config_path}; use .json, .yaml, or .yml."
    )


def _normalized_semantic_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _find_forbidden_term(value: str) -> str | None:
    normalized = _normalized_semantic_text(value)
    for term in _FORBIDDEN_DISCOVERY_TERMS:
        if re.search(rf"(?:^|_){re.escape(term)}(?:_|$)", normalized):
            return term
    return None


def _reject_implicit_discovery(value: Any, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise RuntimeError(f"{location} contains a non-string key: {key!r}")
            forbidden = _find_forbidden_term(key)
            if forbidden:
                raise RuntimeError(
                    f"{location} contains forbidden implicit-discovery key {key!r} (term={forbidden!r})."
                )
            _reject_implicit_discovery(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_implicit_discovery(child, f"{location}[{index}]")
    elif isinstance(value, str):
        forbidden = _find_forbidden_term(value)
        if forbidden:
            raise RuntimeError(
                f"{location} contains forbidden implicit-discovery value {value!r} (term={forbidden!r})."
            )


def _require_exact_keys(value: Mapping[str, Any], allowed: frozenset[str], location: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise RuntimeError(f"{location} contains unknown key(s): {sorted(unknown)}")


def _resolve_manifest_path(raw_path: Any, config_path: Path, split: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path or raw_path != raw_path.strip():
        raise RuntimeError(f"manifests.{split}.path must be a non-empty string without surrounding whitespace.")
    if any(character in _GLOB_CHARACTERS for character in raw_path):
        raise RuntimeError(f"manifests.{split}.path must be explicit; glob patterns are forbidden: {raw_path!r}")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Manifest for split {split!r} does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"Manifest for split {split!r} is not a file: {resolved}")
    return resolved


def load_frozen_run_config(
    config_path: str | Path,
    verify_manifest_hashes: bool = True,
) -> FrozenRunConfig:
    """Load and fully validate one explicit YAML or JSON frozen-run config."""

    if not isinstance(config_path, (str, Path)):
        raise RuntimeError(f"config_path must be a string or pathlib.Path, got {type(config_path).__name__}.")
    candidate = Path(config_path).expanduser()
    try:
        resolved_config = candidate.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Frozen-run config does not exist: {candidate}") from exc
    if not resolved_config.is_file():
        raise RuntimeError(f"Frozen-run config is not a file: {resolved_config}")

    document = _load_document(resolved_config)
    if not isinstance(document, Mapping):
        raise RuntimeError(f"Frozen-run config root must be an object/dict: {resolved_config}")
    _reject_implicit_discovery(document)
    _require_exact_keys(document, _TOP_LEVEL_KEYS, "config")

    if "run_id" not in document:
        raise RuntimeError("Frozen-run config must explicitly define run_id.")
    run_id = document["run_id"]
    if not isinstance(run_id, str) or not run_id or run_id != run_id.strip():
        raise RuntimeError("run_id must be a non-empty explicit string without surrounding whitespace.")

    if "manifests" not in document or not isinstance(document["manifests"], Mapping):
        raise RuntimeError("Frozen-run config must define manifests as an object/dict.")
    raw_manifests = document["manifests"]
    unknown_splits = set(raw_manifests) - set(REQUIRED_SPLITS)
    missing_splits = set(REQUIRED_SPLITS) - set(raw_manifests)
    if unknown_splits:
        raise RuntimeError(f"Frozen-run config contains unknown manifest split(s): {sorted(unknown_splits)}")
    if missing_splits:
        raise RuntimeError(f"Frozen-run config is missing required manifest split(s): {sorted(missing_splits)}")

    references: dict[str, ManifestReference] = {}
    for split in REQUIRED_SPLITS:
        raw_reference = raw_manifests[split]
        if not isinstance(raw_reference, Mapping):
            raise RuntimeError(f"manifests.{split} must be an object/dict.")
        _require_exact_keys(raw_reference, _MANIFEST_KEYS, f"manifests.{split}")
        missing_keys = _MANIFEST_KEYS - set(raw_reference)
        if missing_keys:
            raise RuntimeError(f"manifests.{split} is missing required key(s): {sorted(missing_keys)}")
        path = _resolve_manifest_path(raw_reference["path"], resolved_config, split)
        expected_sha = validate_sha256(
            raw_reference["manifest_sha256"],
            f"manifests.{split}.manifest_sha256",
        )
        if verify_manifest_hashes:
            actual_sha = compute_file_sha256(path)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"Manifest SHA-256 mismatch for split {split!r}: path={path}, "
                    f"expected={expected_sha}, actual={actual_sha}"
                )
        references[split] = ManifestReference(
            split=split,
            path=path,
            manifest_sha256=expected_sha,
        )

    return FrozenRunConfig(
        run_id=run_id,
        manifests=MappingProxyType(references),
        config_path=resolved_config,
    )


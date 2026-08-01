"""Strict CSV/JSON manifest parsing and train/validation/test leakage checks."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from .identity import make_canonical_id, validate_canonical_id, validate_sha256

if TYPE_CHECKING:
    from defect_al.config import FrozenRunConfig

REQUIRED_SPLITS = ("train", "validation", "test")
_JSON_COLLECTION_KEYS = ("records", "samples", "items")
_DATASET_ALIASES = ("dataset_name", "dataset", "dataset_type")
_RELATIVE_PATH_ALIASES = ("relative_path", "image_relative_path", "relative_image_path")
_SHA_ALIASES = ("image_sha256", "sha256")
_CANONICAL_ID_ALIASES = ("canonical_id", "canonical_sample_id")


@dataclass(frozen=True, slots=True)
class ManifestIndex:
    split: str
    path: Path
    row_count: int
    canonical_ids: frozenset[str]
    image_sha256s: frozenset[str]


def _context(path: Path, split: str, row_number: int | None = None) -> str:
    base = f"split={split!r}, manifest={path}"
    return f"{base}, row={row_number}" if row_number is not None else base


def _read_csv_rows(path: Path, split: str) -> list[tuple[int, Mapping[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise RuntimeError(f"CSV manifest has no header ({_context(path, split)}).")
            rows = []
            for row in reader:
                rows.append((reader.line_num, row))
            return rows
    except RuntimeError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RuntimeError(f"Cannot parse CSV manifest ({_context(path, split)}): {exc}") from exc


def _read_json_rows(path: Path, split: str) -> list[tuple[int, Mapping[str, Any]]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot parse JSON manifest ({_context(path, split)}): {exc}") from exc

    if isinstance(document, list):
        records = document
    elif isinstance(document, Mapping):
        present = [key for key in _JSON_COLLECTION_KEYS if key in document]
        if len(present) != 1:
            if len(present) > 1:
                raise RuntimeError(
                    f"Ambiguous JSON manifest collection keys {present} ({_context(path, split)})."
                )
            raise RuntimeError(
                f"JSON manifest object must contain exactly one of {_JSON_COLLECTION_KEYS} ({_context(path, split)})."
            )
        records = document[present[0]]
        if not isinstance(records, list):
            raise RuntimeError(
                f"JSON manifest key {present[0]!r} must contain a list ({_context(path, split)})."
            )
    else:
        raise RuntimeError(f"JSON manifest root must be a list or object ({_context(path, split)}).")

    rows: list[tuple[int, Mapping[str, Any]]] = []
    for row_number, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise RuntimeError(f"Manifest row must be an object ({_context(path, split, row_number)}).")
        rows.append((row_number, record))
    return rows


def _read_rows(path: Path, split: str) -> list[tuple[int, Mapping[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_rows(path, split)
    if suffix == ".json":
        return _read_json_rows(path, split)
    raise RuntimeError(
        f"Unsupported manifest format {suffix or '<none>'!r}; use .csv or .json ({_context(path, split)})."
    )


def _alias_value(
    row: Mapping[str, Any],
    aliases: Iterable[str],
    field_name: str,
    path: Path,
    split: str,
    row_number: int,
) -> Any | None:
    populated = []
    for alias in aliases:
        if alias not in row:
            continue
        value = row[alias]
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        populated.append((alias, value))
    if len(populated) > 1:
        raise RuntimeError(
            f"Ambiguous aliases for {field_name}: {[alias for alias, _ in populated]} "
            f"({_context(path, split, row_number)})."
        )
    return populated[0][1] if populated else None


def load_manifest_index(path: str | Path, split: str) -> ManifestIndex:
    """Parse one manifest and return immutable identity/hash sets."""

    if split not in REQUIRED_SPLITS:
        raise RuntimeError(f"Unsupported split {split!r}; expected one of {REQUIRED_SPLITS}.")
    if not isinstance(path, (str, Path)):
        raise RuntimeError(f"manifest path must be a string or pathlib.Path, got {type(path).__name__}.")
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Manifest does not exist ({_context(candidate, split)}).") from exc
    if not resolved.is_file():
        raise RuntimeError(f"Manifest is not a file ({_context(resolved, split)}).")

    rows = _read_rows(resolved, split)
    if not rows:
        raise RuntimeError(f"Manifest must not be empty ({_context(resolved, split)}).")

    canonical_ids: set[str] = set()
    image_sha256s: set[str] = set()
    first_row_by_id: dict[str, int] = {}
    for row_number, row in rows:
        context = _context(resolved, split, row_number)
        try:
            dataset_name = _alias_value(
                row, _DATASET_ALIASES, "dataset name", resolved, split, row_number
            )
            relative_path = _alias_value(
                row, _RELATIVE_PATH_ALIASES, "relative path", resolved, split, row_number
            )
            image_sha = _alias_value(
                row, _SHA_ALIASES, "image SHA-256", resolved, split, row_number
            )
            stored_id = _alias_value(
                row, _CANONICAL_ID_ALIASES, "canonical ID", resolved, split, row_number
            )

            if image_sha is None:
                raise RuntimeError("image SHA-256 is required")
            normalized_sha = validate_sha256(image_sha, "image SHA-256")

            if (dataset_name is None) != (relative_path is None):
                raise RuntimeError(
                    "dataset name and relative path must either both be present or both be absent"
                )
            recomputed_id = None
            if dataset_name is not None and relative_path is not None:
                recomputed_id = make_canonical_id(dataset_name, relative_path, normalized_sha)

            if stored_id is not None:
                validated_stored_id = validate_canonical_id(stored_id)
                if recomputed_id is not None and validated_stored_id != recomputed_id:
                    raise RuntimeError(
                        "stored canonical ID does not match recomputed cid:v1: "
                        f"stored={validated_stored_id}, recomputed={recomputed_id}"
                    )
                canonical_id = validated_stored_id
            else:
                if recomputed_id is None:
                    raise RuntimeError(
                        "canonical ID is absent; dataset name and relative path are required to generate it"
                    )
                canonical_id = recomputed_id
        except RuntimeError as exc:
            message = str(exc)
            if context in message:
                raise
            raise RuntimeError(f"Invalid manifest identity ({context}): {message}") from exc

        if canonical_id in canonical_ids:
            raise RuntimeError(
                f"Duplicate canonical ID {canonical_id} ({context}); first_row={first_row_by_id[canonical_id]}."
            )
        canonical_ids.add(canonical_id)
        image_sha256s.add(normalized_sha)
        first_row_by_id[canonical_id] = row_number

    return ManifestIndex(
        split=split,
        path=resolved,
        row_count=len(rows),
        canonical_ids=frozenset(canonical_ids),
        image_sha256s=frozenset(image_sha256s),
    )


def _coerce_indexes(
    indexes: Mapping[str, ManifestIndex] | Iterable[ManifestIndex],
) -> Mapping[str, ManifestIndex]:
    if isinstance(indexes, Mapping):
        mapping = dict(indexes)
    else:
        materialized = list(indexes)
        mapping = {index.split: index for index in materialized}
        if len(mapping) != len(materialized):
            raise RuntimeError("Multiple manifest indexes declared the same split.")
    unknown = set(mapping) - set(REQUIRED_SPLITS)
    missing = set(REQUIRED_SPLITS) - set(mapping)
    if unknown:
        raise RuntimeError(f"Unknown manifest index split(s): {sorted(unknown)}")
    if missing:
        raise RuntimeError(f"Missing manifest index split(s): {sorted(missing)}")
    for split, index in mapping.items():
        if not isinstance(index, ManifestIndex):
            raise RuntimeError(f"Index for split {split!r} is not a ManifestIndex.")
        if index.split != split:
            raise RuntimeError(f"ManifestIndex split mismatch: mapping key={split!r}, index.split={index.split!r}.")
    return mapping


def validate_split_leakage(
    indexes: Mapping[str, ManifestIndex] | Iterable[ManifestIndex],
) -> None:
    """Fail with actionable details for exact or renamed cross-split duplicates."""

    mapping = _coerce_indexes(indexes)
    problems = []
    for left_name, right_name in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left = mapping[left_name]
        right = mapping[right_name]
        canonical_overlap = left.canonical_ids & right.canonical_ids
        if canonical_overlap:
            example = sorted(canonical_overlap)[0]
            problems.append(
                f"{left_name}/{right_name} canonical ID overlap count={len(canonical_overlap)}, example={example}"
            )
        sha_overlap = left.image_sha256s & right.image_sha256s
        if sha_overlap:
            example = sorted(sha_overlap)[0]
            problems.append(
                f"{left_name}/{right_name} image SHA-256 overlap count={len(sha_overlap)}, example={example}"
            )
    if problems:
        raise RuntimeError("Split leakage detected:\n- " + "\n- ".join(problems))


def load_frozen_manifest_indexes(config: "FrozenRunConfig") -> Mapping[str, ManifestIndex]:
    """Parse and validate every manifest referenced by a frozen-run config."""

    try:
        references = config.manifests
    except AttributeError as exc:
        raise RuntimeError("config must be a FrozenRunConfig-like object with manifests.") from exc
    indexes = {
        split: load_manifest_index(references[split].path, split)
        for split in REQUIRED_SPLITS
    }
    validate_split_leakage(indexes)
    return MappingProxyType(indexes)


# Small compatibility aliases keep call sites readable without creating a
# second implementation or a framework layer.
parse_manifest = load_manifest_index
validate_manifest_leakage = validate_split_leakage


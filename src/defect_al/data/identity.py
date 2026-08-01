"""Portable, deterministic identity for future active-learning samples.

Historical V7 manifests used a resolved absolute path as part of their sample
identity.  That legacy identity remains valid for frozen evidence, but it is not
portable between machines.  This module defines the forward-only ``cid:v1``
scheme, whose payload contains only a dataset name, a dataset-relative POSIX
path, and the image content SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

CANONICAL_ID_VERSION = "v1"
CANONICAL_ID_PREFIX = f"cid:{CANONICAL_ID_VERSION}:"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_GLOB_CHARACTERS = frozenset("*?[]{}")


def _as_text(value: str | Path, field_name: str) -> str:
    if isinstance(value, Path):
        value = os.fspath(value)
    if not isinstance(value, str):
        raise RuntimeError(f"{field_name} must be a string or pathlib.Path, got {type(value).__name__}.")
    if "\x00" in value:
        raise RuntimeError(f"{field_name} must not contain a NUL character.")
    return value


def compute_file_sha256(path: str | Path) -> str:
    """Return a lowercase SHA-256 digest, failing clearly for unreadable files."""

    raw_path = _as_text(path, "path")
    candidate = Path(raw_path)
    try:
        if not candidate.exists():
            raise RuntimeError(f"Cannot hash missing file: {candidate}")
        if not candidate.is_file():
            raise RuntimeError(f"Cannot hash a non-file path: {candidate}")
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except RuntimeError:
        raise
    except OSError as exc:
        raise RuntimeError(f"Cannot read file for SHA-256: {candidate}: {exc}") from exc


def normalize_dataset_name(dataset_name: str) -> str:
    """Normalize a dataset identifier without merging distinct named datasets."""

    if not isinstance(dataset_name, str):
        raise RuntimeError(f"dataset_name must be a string, got {type(dataset_name).__name__}.")
    normalized = unicodedata.normalize("NFC", dataset_name).strip()
    if not normalized:
        raise RuntimeError("dataset_name must not be empty or whitespace-only.")
    if any(ord(character) < 32 for character in normalized):
        raise RuntimeError("dataset_name must not contain control characters.")
    if "/" in normalized or "\\" in normalized:
        raise RuntimeError("dataset_name is an identifier, not a filesystem path.")
    if any(character in _GLOB_CHARACTERS for character in normalized):
        raise RuntimeError("dataset_name must not contain glob-pattern characters.")
    if normalized in {".", ".."}:
        raise RuntimeError(f"Invalid dataset_name: {normalized!r}.")
    return normalized


def normalize_relative_path(relative_path: str | Path) -> str:
    """Return one strict dataset-relative POSIX path.

    Empty components are rejected instead of being silently collapsed.  This
    keeps the canonical payload unambiguous and identical on Windows and POSIX.
    """

    raw = unicodedata.normalize("NFC", _as_text(relative_path, "relative_path"))
    if not raw:
        raise RuntimeError("relative_path must not be empty.")
    if raw != raw.strip():
        raise RuntimeError("relative_path must not have leading or trailing whitespace.")
    if raw.startswith("\\\\") or raw.startswith("//"):
        raise RuntimeError(f"UNC paths are not allowed for relative_path: {raw!r}")
    if _WINDOWS_DRIVE_RE.match(raw):
        raise RuntimeError(f"Windows drive paths are not allowed for relative_path: {raw!r}")

    posix = raw.replace("\\", "/")
    if posix.startswith("/"):
        raise RuntimeError(f"Absolute paths are not allowed for relative_path: {raw!r}")
    if any(character in _GLOB_CHARACTERS for character in posix):
        raise RuntimeError(f"Glob patterns are not allowed for relative_path: {raw!r}")

    components = posix.split("/")
    if any(component == "" for component in components):
        raise RuntimeError(f"Empty path components are not allowed for relative_path: {raw!r}")
    if any(component in {".", ".."} for component in components):
        raise RuntimeError(f"'.' and '..' components are not allowed for relative_path: {raw!r}")
    if any(any(ord(character) < 32 for character in component) for component in components):
        raise RuntimeError("relative_path must not contain control characters.")
    return "/".join(components)


def validate_sha256(value: str, field_name: str = "sha256") -> str:
    """Validate a SHA-256 string and return its canonical lowercase form."""

    if not isinstance(value, str):
        raise RuntimeError(f"{field_name} must be a string, got {type(value).__name__}.")
    if value != value.strip():
        raise RuntimeError(f"{field_name} must not have surrounding whitespace.")
    normalized = value.lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise RuntimeError(f"{field_name} must be exactly 64 hexadecimal characters.")
    return normalized


def make_canonical_id(
    dataset_name: str,
    relative_path: str | Path,
    image_sha256: str,
) -> str:
    """Build ``cid:v1:<sha256>`` from the canonical three-field payload."""

    payload = {
        "dataset_name": normalize_dataset_name(dataset_name),
        "relative_path": normalize_relative_path(relative_path),
        "image_sha256": validate_sha256(image_sha256, "image_sha256"),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return CANONICAL_ID_PREFIX + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def make_canonical_id_from_file(
    dataset_name: str,
    dataset_root: str | Path,
    image_path: str | Path,
    relative_path: str | Path | None = None,
) -> str:
    """Build a canonical ID after proving the image stays inside ``dataset_root``.

    ``image_path`` may be absolute or relative to ``dataset_root``.  An optional
    explicit ``relative_path`` can be supplied when a manifest path must also be
    checked against the physical file location.
    """

    root_raw = _as_text(dataset_root, "dataset_root")
    image_raw = _as_text(image_path, "image_path")
    root_candidate = Path(root_raw)
    try:
        if not root_candidate.exists() or not root_candidate.is_dir():
            raise RuntimeError(f"dataset_root must be an existing directory: {root_candidate}")
        resolved_root = root_candidate.resolve(strict=True)

        image_candidate = Path(image_raw)
        if image_candidate.is_absolute():
            unresolved_image = image_candidate
        else:
            safe_image_relative = normalize_relative_path(image_raw)
            unresolved_image = resolved_root.joinpath(*safe_image_relative.split("/"))
        if not unresolved_image.exists() or not unresolved_image.is_file():
            raise RuntimeError(f"image_path must be an existing file: {unresolved_image}")
        resolved_image = unresolved_image.resolve(strict=True)

        try:
            derived_relative = resolved_image.relative_to(resolved_root)
        except ValueError as exc:
            raise RuntimeError(
                f"image_path escapes dataset_root after path/symlink resolution: {resolved_image}"
            ) from exc

        normalized_derived = normalize_relative_path(derived_relative.as_posix())
        if relative_path is not None:
            normalized_explicit = normalize_relative_path(relative_path)
            expected = resolved_root.joinpath(*normalized_explicit.split("/")).resolve(strict=True)
            if expected != resolved_image:
                raise RuntimeError(
                    "relative_path does not resolve to image_path inside dataset_root: "
                    f"relative_path={normalized_explicit!r}, image_path={resolved_image}"
                )
            canonical_relative = normalized_explicit
        else:
            canonical_relative = normalized_derived
    except RuntimeError:
        raise
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not validate image_path={image_path!r} against dataset_root={dataset_root!r}: {exc}"
        ) from exc

    return make_canonical_id(
        dataset_name=dataset_name,
        relative_path=canonical_relative,
        image_sha256=compute_file_sha256(resolved_image),
    )


def validate_canonical_id(value: str) -> str:
    """Validate and return one exact lowercase ``cid:v1`` identifier."""

    if not isinstance(value, str):
        raise RuntimeError(f"canonical_id must be a string, got {type(value).__name__}.")
    if not value.startswith(CANONICAL_ID_PREFIX):
        raise RuntimeError(f"canonical_id must start with {CANONICAL_ID_PREFIX!r}.")
    digest = value[len(CANONICAL_ID_PREFIX) :]
    if not _SHA256_RE.fullmatch(digest):
        raise RuntimeError("canonical_id digest must be exactly 64 lowercase hexadecimal characters.")
    return value


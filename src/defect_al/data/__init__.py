"""Dataset identity, manifest parsing, and split-integrity validation."""

from .identity import (
    CANONICAL_ID_PREFIX,
    CANONICAL_ID_VERSION,
    compute_file_sha256,
    make_canonical_id,
    make_canonical_id_from_file,
    normalize_dataset_name,
    normalize_relative_path,
    validate_canonical_id,
    validate_sha256,
)
from .manifests import (
    ManifestIndex,
    load_frozen_manifest_indexes,
    load_manifest_index,
    summarize_split_overlaps,
    validate_split_leakage,
)

__all__ = [
    "CANONICAL_ID_PREFIX",
    "CANONICAL_ID_VERSION",
    "ManifestIndex",
    "compute_file_sha256",
    "load_frozen_manifest_indexes",
    "load_manifest_index",
    "make_canonical_id",
    "make_canonical_id_from_file",
    "normalize_dataset_name",
    "normalize_relative_path",
    "summarize_split_overlaps",
    "validate_canonical_id",
    "validate_sha256",
    "validate_split_leakage",
]

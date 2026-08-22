"""Pure protocol primitives for the Phase 2A CPU preflight.

This module deliberately contains no detector training, final-evaluation, XML,
or Hugging Face model-loading code. Acquisition receives only opaque sample
identities, membership state, seeds/rounds, and (for DINO) sealed embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import csv
import hashlib
import inspect
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "confirmatory-random-dino-v1"
RNG_ENGINE = "numpy.random.Generator(PCG64)"
RNG_NAMESPACE_UINT32 = 0x43524431  # ASCII-like namespace: CRD1
DATASET_CODES = {"GC10-DET": 0x47433130, "NEU-DET": 0x4E455531}

ALLOWED_SELECTOR_FIELDS = frozenset(
    {
        "sample_id",
        "membership",
        "acquisition_seed",
        "acquisition_round",
        "already_selected",
    }
)
DENIED_FIELD_TOKENS = frozenset(
    {
        "class",
        "label",
        "bbox",
        "box",
        "rare",
        "frequent",
        "anomaly",
        "filename",
        "file_name",
        "path",
        "metric",
        "map",
        "recall",
        "precision",
        "future",
        "dev",
        "test",
        "ground_truth",
        "gt",
    }
)


class ProtocolIntegrityError(RuntimeError):
    """Base class for protocol hard failures."""


class GTBoundaryError(ProtocolIntegrityError):
    """Raised when label-bearing or otherwise prohibited acquisition data enters."""


class EmbeddingIntegrityError(ProtocolIntegrityError):
    """Raised when the embedding cache is incomplete or numerically invalid."""


class ManifestIntegrityError(ProtocolIntegrityError):
    """Raised when manifest identity, overlap, or cardinality is invalid."""


class FinalEvaluationLockedError(ProtocolIntegrityError):
    """Raised whenever locked final evaluation is requested without authorization."""


def validate_selector_records(records: Sequence[Mapping[str, object]]) -> None:
    """Hard-fail on any acquisition record field outside the frozen allow-list."""

    for index, record in enumerate(records):
        keys = {str(key) for key in record}
        lowered = {key.lower() for key in keys}
        denied = {
            key
            for key in lowered
            if any(token == key or token in key.split("_") for token in DENIED_FIELD_TOKENS)
        }
        unexpected = keys - ALLOWED_SELECTOR_FIELDS
        if denied or unexpected:
            raise GTBoundaryError(
                f"Selector record {index} violates the GT-free boundary: "
                f"denied={sorted(denied)}, unexpected={sorted(unexpected)}"
            )
        if "sample_id" not in record:
            raise GTBoundaryError(f"Selector record {index} is missing sample_id")
    validate_sample_ids([str(record["sample_id"]) for record in records])


def validate_sample_ids(sample_ids: Sequence[str]) -> None:
    """Require opaque, lower-case SHA-256 identities with no semantic text."""

    for sample_id in sample_ids:
        if re.fullmatch(r"[0-9a-f]{64}", sample_id) is None:
            raise ManifestIntegrityError(
                "sample_id must be an opaque lower-case 64-character SHA-256 digest"
            )
    if len(set(sample_ids)) != len(sample_ids):
        raise ManifestIntegrityError("duplicate sample_id detected")


def canonical_manifest_bytes(sample_ids: Iterable[str]) -> bytes:
    ids = sorted(str(value) for value in sample_ids)
    validate_sample_ids(ids)
    return ("\n".join(ids) + "\n").encode("ascii")


def manifest_sha256(sample_ids: Iterable[str]) -> str:
    return hashlib.sha256(canonical_manifest_bytes(sample_ids)).hexdigest()


def validate_split_partitions(partitions: Mapping[str, Sequence[str]]) -> None:
    """Require unique sample IDs within and across split partitions."""

    seen: dict[str, str] = {}
    for split, values in partitions.items():
        ids = [str(value) for value in values]
        validate_sample_ids(ids)
        for sample_id in ids:
            if sample_id in seen:
                raise ManifestIntegrityError(
                    f"sample_id appears in both {seen[sample_id]} and {split}: {sample_id}"
                )
            seen[sample_id] = split


def _seed_sequence(dataset: str, acquisition_seed: int, round_idx: int, purpose: int) -> np.random.SeedSequence:
    if dataset not in DATASET_CODES:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if acquisition_seed < 0 or round_idx < 0:
        raise ValueError("seeds and rounds must be non-negative")
    return np.random.SeedSequence(
        [RNG_NAMESPACE_UINT32, DATASET_CODES[dataset], int(acquisition_seed), int(round_idx), int(purpose)]
    )


def initial_manifest(
    sample_ids: Sequence[str], *, size: int, dataset: str, acquisition_seed: int
) -> tuple[str, ...]:
    """Select the one shared initial set for Random and DINO."""

    canonical = sorted(str(value) for value in sample_ids)
    validate_sample_ids(canonical)
    if not 0 < size <= len(canonical):
        raise ValueError("initial size must be between 1 and universe size")
    rng = np.random.Generator(np.random.PCG64(_seed_sequence(dataset, acquisition_seed, 0, 0x494E4954)))
    chosen = rng.choice(len(canonical), size=size, replace=False, shuffle=False)
    return tuple(sorted(canonical[int(index)] for index in chosen))


def random_query(
    remaining_ids: Sequence[str],
    *,
    size: int,
    dataset: str,
    acquisition_seed: int,
    round_idx: int,
) -> tuple[str, ...]:
    """Uniform Random query using only acquisition seed, dataset, and round."""

    canonical = sorted(str(value) for value in remaining_ids)
    validate_sample_ids(canonical)
    if round_idx < 1:
        raise ValueError("Random query round must be >= 1")
    if not 0 < size <= len(canonical):
        raise ValueError("query size must be between 1 and remaining pool size")
    rng = np.random.Generator(
        np.random.PCG64(_seed_sequence(dataset, acquisition_seed, round_idx, 0x51554552))
    )
    chosen = rng.choice(len(canonical), size=size, replace=False, shuffle=False)
    return tuple(sorted(canonical[int(index)] for index in chosen))


def build_random_trajectory(
    sample_ids: Sequence[str],
    *,
    initial_size: int,
    query_size: int,
    rounds: int,
    dataset: str,
    acquisition_seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Generate and return six cumulative states without accepting a training seed."""

    initial = initial_manifest(
        sample_ids, size=initial_size, dataset=dataset, acquisition_seed=acquisition_seed
    )
    cumulative = set(initial)
    states: list[tuple[str, ...]] = [tuple(sorted(cumulative))]
    for round_idx in range(1, rounds + 1):
        remaining = sorted(set(sample_ids) - cumulative)
        query = random_query(
            remaining,
            size=query_size,
            dataset=dataset,
            acquisition_seed=acquisition_seed,
            round_idx=round_idx,
        )
        cumulative.update(query)
        states.append(tuple(sorted(cumulative)))
    return tuple(states)


@dataclass(frozen=True)
class ValidatedEmbeddingCache:
    sample_ids: tuple[str, ...]
    normalized: np.ndarray

    def as_lookup(self) -> dict[str, np.ndarray]:
        return {sample_id: self.normalized[index] for index, sample_id in enumerate(self.sample_ids)}


def validate_embedding_cache(
    expected_ids: Sequence[str],
    embedding_ids: Sequence[str],
    matrix: np.ndarray,
    *,
    expected_dim: int | None = 384,
) -> ValidatedEmbeddingCache:
    """Validate exact coverage, uniqueness, dimensions, finiteness, and nonzero norms."""

    expected = tuple(str(value) for value in expected_ids)
    actual = tuple(str(value) for value in embedding_ids)
    validate_sample_ids(expected)
    validate_sample_ids(actual)
    if len(expected) != len(actual) or set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise EmbeddingIntegrityError(
            f"embedding identity mismatch: expected={len(expected)}, actual={len(actual)}, "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(actual):
        raise EmbeddingIntegrityError(
            f"embedding shape mismatch: rows={array.shape if array.ndim else 'scalar'}, ids={len(actual)}"
        )
    if expected_dim is not None and array.shape[1] != expected_dim:
        raise EmbeddingIntegrityError(
            f"embedding dimension mismatch: expected={expected_dim}, actual={array.shape[1]}"
        )
    if not np.isfinite(array).all():
        raise EmbeddingIntegrityError("embedding cache contains NaN or infinity")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1e-12):
        raise EmbeddingIntegrityError("embedding cache contains a zero-norm row")
    normalized = array / norms[:, None]
    return ValidatedEmbeddingCache(actual, normalized.astype(np.float32, copy=False))


def farthest_first_query(
    remaining_ids: Sequence[str],
    reference_ids: Sequence[str],
    cache: ValidatedEmbeddingCache,
    *,
    size: int,
) -> tuple[str, ...]:
    """Greedy cosine farthest-first with immediate reference updates.

    Exact ties are resolved by ascending opaque sample_id.
    """

    remaining = sorted(str(value) for value in remaining_ids)
    references = [str(value) for value in reference_ids]
    validate_sample_ids(remaining)
    validate_sample_ids(references)
    if not references:
        raise ValueError("farthest-first requires a non-empty reference set")
    if not 0 < size <= len(remaining):
        raise ValueError("query size must be between 1 and remaining pool size")
    lookup = cache.as_lookup()
    missing = sorted((set(remaining) | set(references)) - set(lookup))
    if missing:
        raise EmbeddingIntegrityError(f"farthest-first references missing embeddings: {missing[:3]}")

    chosen: list[str] = []
    candidate_set = set(remaining)
    reference_set = list(references)
    for _ in range(size):
        scored: list[tuple[float, str]] = []
        ref_matrix = np.stack([lookup[sample_id] for sample_id in reference_set], axis=0)
        for sample_id in sorted(candidate_set):
            max_cosine = float(np.max(ref_matrix @ lookup[sample_id]))
            scored.append((1.0 - max_cosine, sample_id))
        best_score = max(score for score, _ in scored)
        tied = [sample_id for score, sample_id in scored if np.isclose(score, best_score, rtol=0.0, atol=1e-12)]
        picked = min(tied)
        chosen.append(picked)
        candidate_set.remove(picked)
        reference_set.append(picked)
    return tuple(chosen)


def round_half_up_count(total: int, fraction: str | Decimal) -> int:
    value = Decimal(total) * Decimal(str(fraction))
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def budget_grid(
    acquisition_universe_size: int,
    *,
    initial_fraction: str = "0.10",
    query_fraction: str = "0.05",
    rounds: int = 5,
) -> tuple[int, ...]:
    initial = round_half_up_count(acquisition_universe_size, initial_fraction)
    query = round_half_up_count(acquisition_universe_size, query_fraction)
    if initial <= 0 or query <= 0 or initial + rounds * query > acquisition_universe_size:
        raise ValueError("budget grid is incompatible with acquisition universe size")
    return tuple(initial + round_idx * query for round_idx in range(rounds + 1))


def budget_averaged_aulc(budget_fractions: Sequence[float], metrics: Sequence[float]) -> float:
    """Trapezoidal mAP50-95 area divided by the evaluated budget span."""

    x = np.asarray(budget_fractions, dtype=np.float64)
    y = np.asarray(metrics, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("AULC requires equal one-dimensional sequences with at least two points")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
        raise ValueError("AULC inputs must be finite and budget fractions strictly increasing")
    widths = np.diff(x)
    area = float(np.sum(widths * (y[:-1] + y[1:]) / 2.0))
    return area / float(x[-1] - x[0])


def require_final_evaluation_authorization(
    split: str,
    *,
    authorization_flag: bool = False,
    authorization_record: Path | None = None,
    expected_protocol_sha256: str | None = None,
) -> None:
    """Require both an explicit flag and a matching external authorization record."""

    if split not in {"acquisition", "development", "final"}:
        raise ValueError(f"Unknown split: {split}")
    if split != "final":
        return
    if not authorization_flag or authorization_record is None or not authorization_record.is_file():
        raise FinalEvaluationLockedError("final evaluation is locked; explicit external authorization is absent")
    payload = json.loads(authorization_record.read_text(encoding="utf-8"))
    if payload.get("status") != "AUTHORIZED" or payload.get("protocol_sha256") != expected_protocol_sha256:
        raise FinalEvaluationLockedError("final-evaluation authorization record is invalid or mismatched")


def load_private_source_registry_partition(
    registry_path: Path,
    *,
    dataset: str,
    split: str,
    authorization_flag: bool = False,
    authorization_record: Path | None = None,
    expected_protocol_sha256: str | None = None,
) -> list[dict[str, str]]:
    """Load a source registry partition only after enforcing the final lock."""

    require_final_evaluation_authorization(
        split,
        authorization_flag=authorization_flag,
        authorization_record=authorization_record,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    with registry_path.open("r", newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("dataset") == dataset and row.get("split") == split
        ]
    if not rows:
        raise ManifestIntegrityError(f"no registry rows for dataset={dataset}, split={split}")
    validate_sample_ids([row["sample_id"] for row in rows])
    return rows


def acquisition_api_has_no_training_seed() -> bool:
    """Static audit helper used by preflight and readiness documentation."""

    functions = [initial_manifest, random_query, build_random_trajectory, farthest_first_query]
    return all("training_seed" not in inspect.signature(function).parameters for function in functions)

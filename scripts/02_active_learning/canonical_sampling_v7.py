"""Canonical sample identity and deterministic sampling helpers for V7.

All active-learning runners should canonicalize the candidate pool before any
row-order-dependent sampling.  The canonical identity intentionally uses only
filesystem identity, not labels or XML annotations:

    canonical_sample_id = resolved_image_path + "::" + image_sha256
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from audit_detection_pipeline_v7 import compute_file_sha256, resolve_image_path_fast


def make_canonical_sample_id(resolved_image_path: str, image_sha256: str) -> str:
    return f"{resolved_image_path}::{image_sha256}"


def add_canonical_sample_identity(
    df: pd.DataFrame,
    image_index: dict[tuple[str, str], list[Path]],
    *,
    set_sample_id: bool = True,
) -> pd.DataFrame:
    """Attach resolved path, image SHA256, and canonical sample identity."""

    out = df.copy()
    resolved_paths: list[str | None] = []
    sha_values: list[str | None] = []
    canonical_ids: list[str | None] = []
    missing: list[tuple[int, object, object]] = []

    for i, row in out.iterrows():
        path = resolve_image_path_fast(row, image_index)
        if path is None:
            missing.append((i, row.get("dataset_type"), row.get("image_name")))
            resolved_paths.append(None)
            sha_values.append(None)
            canonical_ids.append(None)
            continue
        resolved = str(path.resolve())
        sha = compute_file_sha256(path)
        resolved_paths.append(resolved)
        sha_values.append(sha)
        canonical_ids.append(make_canonical_sample_id(resolved, sha))

    if missing:
        raise FileNotFoundError(
            f"Could not resolve {len(missing)} image paths for canonical sampling. "
            f"First rows: {missing[:5]}"
        )

    out["resolved_image_path"] = resolved_paths
    out["image_sha256"] = sha_values
    out["canonical_sample_id"] = canonical_ids
    if set_sample_id:
        out["sample_id"] = out["canonical_sample_id"]
    return out


def canonicalize_pool_for_sampling(
    df: pd.DataFrame,
    image_index: dict[tuple[str, str], list[Path]],
    *,
    set_sample_id: bool = True,
) -> pd.DataFrame:
    """Deduplicate, canonical-sort, and reset index before sampling."""

    out = add_canonical_sample_identity(df, image_index, set_sample_id=set_sample_id)
    dup = out[out["canonical_sample_id"].duplicated(keep=False)]
    if not dup.empty:
        # Keep deterministic first occurrence, but surface the collision clearly.
        examples = dup[["dataset_type", "image_name", "canonical_sample_id"]].head(20)
        raise ValueError(f"Duplicate canonical_sample_id rows found:\n{examples}")
    return (
        out.drop_duplicates("canonical_sample_id", keep="first")
        .sort_values("canonical_sample_id", kind="mergesort")
        .reset_index(drop=True)
    )


def sample_initial_labeled_set(
    pool: pd.DataFrame,
    *,
    initial_seed_size: int,
    acquisition_seed: int,
) -> pd.DataFrame:
    """Sample the initial labeled set from an already-canonicalized pool."""

    if "canonical_sample_id" not in pool.columns:
        raise ValueError("sample_initial_labeled_set requires canonical_sample_id.")
    canonical = (
        pool.copy()
        .drop_duplicates("canonical_sample_id", keep="first")
        .sort_values("canonical_sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    return canonical.sample(
        n=min(initial_seed_size, len(canonical)),
        random_state=acquisition_seed + 999,
        replace=False,
    )


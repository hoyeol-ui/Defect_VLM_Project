from __future__ import annotations

import os
from pathlib import Path

import pytest

from defect_al.config import load_frozen_run_config
from defect_al.data.identity import make_canonical_id
from defect_al.data.manifests import (
    load_frozen_manifest_indexes,
    load_manifest_index,
    validate_split_leakage,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _row(path: str, sha: str, dataset: str = "GC10") -> dict[str, str]:
    return {"dataset_name": dataset, "relative_path": path, "image_sha256": sha}


def _load_triplet(tmp_path: Path, manifest_writer, rows_by_split, suffixes=None):
    suffixes = suffixes or {"train": ".csv", "validation": ".csv", "test": ".csv"}
    indexes = {}
    for split in ("train", "validation", "test"):
        path = manifest_writer(tmp_path / f"{split}{suffixes[split]}", rows_by_split[split])
        indexes[split] = load_manifest_index(path, split)
    return indexes


def test_clean_three_way_split_passes(tmp_path: Path, manifest_writer) -> None:
    indexes = _load_triplet(
        tmp_path,
        manifest_writer,
        {
            "train": [_row("images/train.png", SHA_A)],
            "validation": [_row("images/validation.png", SHA_B)],
            "test": [_row("images/test.png", SHA_C)],
        },
    )
    validate_split_leakage(indexes)
    assert indexes["train"].row_count == 1


def test_same_canonical_id_across_train_validation_fails(tmp_path: Path, manifest_writer) -> None:
    duplicate = _row("images/same.png", SHA_A)
    indexes = _load_triplet(
        tmp_path,
        manifest_writer,
        {"train": [duplicate], "validation": [duplicate], "test": [_row("images/test.png", SHA_C)]},
    )
    expected_id = make_canonical_id("GC10", "images/same.png", SHA_A)
    with pytest.raises(RuntimeError) as exc_info:
        validate_split_leakage(indexes)
    message = str(exc_info.value)
    assert "train/validation" in message
    assert "canonical ID overlap count=1" in message
    assert expected_id in message


def test_renamed_image_sha_across_train_test_fails(tmp_path: Path, manifest_writer) -> None:
    indexes = _load_triplet(
        tmp_path,
        manifest_writer,
        {
            "train": [_row("images/original.png", SHA_A)],
            "validation": [_row("images/validation.png", SHA_B)],
            "test": [_row("renamed/copied.png", SHA_A)],
        },
    )
    with pytest.raises(RuntimeError) as exc_info:
        validate_split_leakage(indexes)
    message = str(exc_info.value)
    assert "train/test" in message
    assert "image SHA-256 overlap count=1" in message
    assert SHA_A in message


def test_duplicate_canonical_id_inside_manifest_fails(tmp_path: Path, manifest_writer) -> None:
    duplicate = _row("images/same.png", SHA_A)
    path = manifest_writer(tmp_path / "train.csv", [duplicate, duplicate])
    with pytest.raises(RuntimeError) as exc_info:
        load_manifest_index(path, "train")
    message = str(exc_info.value)
    assert "Duplicate canonical ID" in message
    assert "split='train'" in message
    assert "row=" in message
    assert str(path.resolve()) in message


def test_stored_and_recomputed_canonical_id_mismatch_fails(tmp_path: Path, manifest_writer) -> None:
    row = _row("images/sample.png", SHA_A)
    row["canonical_id"] = make_canonical_id("OTHER", "images/sample.png", SHA_A)
    path = manifest_writer(tmp_path / "train.json", [row])
    with pytest.raises(RuntimeError) as exc_info:
        load_manifest_index(path, "train")
    message = str(exc_info.value)
    assert "stored canonical ID does not match" in message
    assert "split='train'" in message
    assert "row=1" in message


@pytest.mark.parametrize("suffix", [".csv", ".json"])
def test_empty_manifest_fails(tmp_path: Path, manifest_writer, suffix: str) -> None:
    path = manifest_writer(tmp_path / f"train{suffix}", [])
    with pytest.raises(RuntimeError, match="must not be empty.*split='train'"):
        load_manifest_index(path, "train")


def test_csv_and_json_manifests_can_be_mixed(tmp_path: Path, manifest_writer) -> None:
    indexes = _load_triplet(
        tmp_path,
        manifest_writer,
        {
            "train": [_row("images/train.png", SHA_A)],
            "validation": [_row("images/validation.png", SHA_B)],
            "test": [_row("images/test.png", SHA_C)],
        },
        {"train": ".csv", "validation": ".json", "test": ".csv"},
    )
    validate_split_leakage(indexes)


def test_manifest_column_aliases_are_supported(tmp_path: Path, manifest_writer) -> None:
    canonical_id = make_canonical_id("GC10", "images/sample.png", SHA_D)
    path = manifest_writer(
        tmp_path / "test.json",
        [
            {
                "dataset_type": "GC10",
                "image_relative_path": "images/sample.png",
                "sha256": SHA_D,
                "canonical_sample_id": canonical_id,
            }
        ],
    )
    index = load_manifest_index(path, "test")
    assert index.canonical_ids == frozenset({canonical_id})
    assert index.image_sha256s == frozenset({SHA_D})


def test_json_collection_key_ambiguity_fails(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    path.write_text('{"records": [], "samples": []}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Ambiguous JSON manifest collection keys"):
        load_manifest_index(path, "train")


def test_error_reports_all_pairwise_overlap_context(tmp_path: Path, manifest_writer) -> None:
    indexes = _load_triplet(
        tmp_path,
        manifest_writer,
        {
            "train": [_row("images/train.png", SHA_A)],
            "validation": [_row("renamed/validation.png", SHA_A)],
            "test": [_row("renamed/test.png", SHA_A)],
        },
    )
    with pytest.raises(RuntimeError) as exc_info:
        validate_split_leakage(indexes)
    message = str(exc_info.value)
    for pair in ("train/validation", "train/test", "validation/test"):
        assert pair in message
    assert message.count("overlap count=1") == 3
    assert f"example={SHA_A}" in message


@pytest.mark.integration
def test_real_frozen_config_if_requested() -> None:
    raw_config = os.environ.get("DEFECT_AL_FROZEN_CONFIG")
    if not raw_config:
        pytest.skip("Set DEFECT_AL_FROZEN_CONFIG to opt in to the real frozen-manifest check.")
    config_path = Path(raw_config).expanduser()
    config = load_frozen_run_config(config_path, verify_manifest_hashes=True)
    indexes = load_frozen_manifest_indexes(config)
    assert tuple(indexes) == ("train", "validation", "test")


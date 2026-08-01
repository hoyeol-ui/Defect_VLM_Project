from __future__ import annotations

import shutil
import unicodedata
from pathlib import Path

import pytest

from defect_al.data.identity import (
    CANONICAL_ID_PREFIX,
    compute_file_sha256,
    make_canonical_id,
    make_canonical_id_from_file,
    normalize_relative_path,
    validate_canonical_id,
    validate_sha256,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def test_windows_and_posix_relative_paths_create_same_id() -> None:
    windows_id = make_canonical_id("GC10", r"images\crease\sample.png", SHA_A)
    posix_id = make_canonical_id("GC10", "images/crease/sample.png", SHA_A)
    assert windows_id == posix_id


def test_identical_payload_is_deterministic() -> None:
    first = make_canonical_id("GC10", "images/sample.png", SHA_A)
    second = make_canonical_id("GC10", "images/sample.png", SHA_A)
    assert first == second


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (("GC10", "images/a.png", SHA_A), ("NEU", "images/a.png", SHA_A)),
        (("GC10", "images/a.png", SHA_A), ("GC10", "images/b.png", SHA_A)),
        (("GC10", "images/a.png", SHA_A), ("GC10", "images/a.png", SHA_B)),
    ],
)
def test_payload_field_changes_change_id(left: tuple[str, str, str], right: tuple[str, str, str]) -> None:
    assert make_canonical_id(*left) != make_canonical_id(*right)


@pytest.mark.parametrize(
    "invalid_path",
    [
        "/absolute/image.png",
        r"C:\dataset\image.png",
        r"\\server\share\image.png",
        "images/../image.png",
        "images/./image.png",
        "images//image.png",
        "images/*.png",
        "images/image?.png",
    ],
)
def test_invalid_relative_paths_are_rejected(invalid_path: str) -> None:
    with pytest.raises(RuntimeError):
        normalize_relative_path(invalid_path)


@pytest.mark.parametrize("invalid_sha", ["", "a" * 63, "g" * 64, " a" * 32])
def test_invalid_sha_is_rejected(invalid_sha: str) -> None:
    with pytest.raises(RuntimeError):
        validate_sha256(invalid_sha)


def test_sha_is_normalized_to_lowercase() -> None:
    assert validate_sha256("A" * 64) == SHA_A


def test_file_outside_dataset_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    with pytest.raises(RuntimeError, match="escapes dataset_root"):
        make_canonical_id_from_file("GC10", root, outside)


def test_symlink_escape_is_rejected_when_supported(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    link = root / "escape.png"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable on this platform/account.")
    with pytest.raises(RuntimeError, match="escapes dataset_root"):
        make_canonical_id_from_file("GC10", root, link)


def test_same_relative_file_in_different_roots_has_same_id(tmp_path: Path) -> None:
    first_root = tmp_path / "machine_a" / "GC10"
    second_root = tmp_path / "machine_b" / "GC10"
    first_image = first_root / "images" / "sample.png"
    second_image = second_root / "images" / "sample.png"
    first_image.parent.mkdir(parents=True)
    second_image.parent.mkdir(parents=True)
    first_image.write_bytes(b"identical image bytes")
    shutil.copyfile(first_image, second_image)

    first_id = make_canonical_id_from_file("GC10", first_root, first_image)
    second_id = make_canonical_id_from_file("GC10", second_root, second_image)
    assert first_id == second_id


def test_explicit_relative_path_must_match_file(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = root / "images" / "sample.png"
    other = root / "images" / "other.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"sample")
    other.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="does not resolve to image_path"):
        make_canonical_id_from_file("GC10", root, image, "images/other.png")


def test_unicode_nfc_and_nfd_create_same_id() -> None:
    dataset_nfc = "결함데이터"
    path_nfc = "이미지/균열.png"
    dataset_nfd = unicodedata.normalize("NFD", dataset_nfc)
    path_nfd = unicodedata.normalize("NFD", path_nfc)
    assert make_canonical_id(dataset_nfc, path_nfc, SHA_A) == make_canonical_id(
        dataset_nfd, path_nfd, SHA_A
    )


def test_canonical_id_validation() -> None:
    canonical_id = make_canonical_id("GC10", "images/sample.png", SHA_A)
    assert canonical_id.startswith(CANONICAL_ID_PREFIX)
    assert validate_canonical_id(canonical_id) == canonical_id
    with pytest.raises(RuntimeError):
        validate_canonical_id("legacy:" + "a" * 64)
    with pytest.raises(RuntimeError):
        validate_canonical_id(CANONICAL_ID_PREFIX + "A" * 64)


def test_compute_file_sha256_and_missing_file_error(tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"abc")
    assert compute_file_sha256(image) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )
    with pytest.raises(RuntimeError, match="missing file"):
        compute_file_sha256(tmp_path / "missing.bin")


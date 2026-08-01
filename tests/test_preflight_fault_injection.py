from __future__ import annotations

import json
from pathlib import Path

import pytest

from defect_al.data.identity import compute_file_sha256, make_canonical_id
from defect_al.preflight import validate_frozen_run

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _row(relative_path: str, image_sha256: str) -> dict[str, str]:
    return {
        "dataset_name": "GC10-DET",
        "relative_path": relative_path,
        "image_sha256": image_sha256,
    }


def _valid_case(tmp_path: Path, manifest_writer):
    rows = {
        "train": [_row("1/train.jpg", SHA_A)],
        "validation": [_row("2/validation.jpg", SHA_B)],
        "test": [_row("3/audit_holdout.jpg", SHA_C)],
    }
    paths = {
        split: manifest_writer(tmp_path / f"{split}.csv", split_rows)
        for split, split_rows in rows.items()
    }
    document = {
        "run_id": "fault_injection_baseline_v1",
        "manifests": {
            split: {
                "path": path.name,
                "manifest_sha256": compute_file_sha256(path),
            }
            for split, path in paths.items()
        },
    }
    return rows, paths, document


def _seal(document: dict, paths: dict[str, Path]) -> None:
    for split, path in paths.items():
        if split in document["manifests"]:
            document["manifests"][split]["manifest_sha256"] = compute_file_sha256(path)


@pytest.mark.parametrize(
    ("fault", "expected"),
    [
        ("canonical_cross_split", "canonical ID overlap"),
        ("renamed_sha_cross_split", "image SHA-256 overlap"),
        ("manifest_tamper", "Manifest SHA-256 mismatch"),
        ("latest_run_id", "implicit-discovery value"),
        ("latest_key", "implicit-discovery key"),
        ("glob_path", "glob patterns are forbidden"),
        ("stored_id_mismatch", "stored canonical ID does not match"),
        ("duplicate_id", "Duplicate canonical ID"),
        ("absolute_relative_path", "Windows drive paths are not allowed"),
        ("missing_test_split", "missing required manifest split"),
    ],
    ids=[
        "train-image-copied-to-validation",
        "renamed-image-copied-to-test",
        "manifest-one-character-tamper",
        "run-id-latest",
        "latest-run-dir-key",
        "glob-manifest-path",
        "stored-canonical-id-tamper",
        "duplicate-canonical-id",
        "absolute-relative-path",
        "missing-test-manifest",
    ],
)
def test_preflight_fault_injection_detects_every_fault(
    tmp_path: Path,
    manifest_writer,
    fault: str,
    expected: str,
) -> None:
    rows, paths, document = _valid_case(tmp_path, manifest_writer)

    if fault == "canonical_cross_split":
        rows["validation"].append(rows["train"][0].copy())
        manifest_writer(paths["validation"], rows["validation"])
        _seal(document, paths)
    elif fault == "renamed_sha_cross_split":
        rows["test"] = [_row("renamed/copied.jpg", SHA_A)]
        manifest_writer(paths["test"], rows["test"])
        _seal(document, paths)
    elif fault == "manifest_tamper":
        pass
    elif fault == "latest_run_id":
        document["run_id"] = "latest"
    elif fault == "latest_key":
        document["latest_run_dir"] = "runs/anything"
    elif fault == "glob_path":
        document["manifests"]["train"]["path"] = "train*.csv"
    elif fault == "stored_id_mismatch":
        rows["train"][0]["canonical_id"] = make_canonical_id(
            "OTHER-DATASET", "1/train.jpg", SHA_A
        )
        manifest_writer(paths["train"], rows["train"])
        _seal(document, paths)
    elif fault == "duplicate_id":
        rows["train"].append(rows["train"][0].copy())
        manifest_writer(paths["train"], rows["train"])
        _seal(document, paths)
    elif fault == "absolute_relative_path":
        rows["train"][0]["relative_path"] = r"C:\dataset\train.jpg"
        manifest_writer(paths["train"], rows["train"])
        _seal(document, paths)
    elif fault == "missing_test_split":
        document["manifests"].pop("test")
    else:  # pragma: no cover
        raise AssertionError(f"Unknown fault case: {fault}")

    config_path = tmp_path / "fault.json"
    config_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    if fault == "manifest_tamper":
        paths["train"].write_text(
            paths["train"].read_text(encoding="utf-8") + "#",
            encoding="utf-8",
        )

    with pytest.raises(RuntimeError, match=expected):
        validate_frozen_run(config_path)

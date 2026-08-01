from __future__ import annotations

import hashlib

from defect_al.data.identity import make_canonical_id, normalize_relative_path


def test_cross_platform_canonical_contract_is_frozen() -> None:
    assert normalize_relative_path(r"images\sample.jpg") == "images/sample.jpg"
    assert make_canonical_id("GC10-DET", r"images\sample.jpg", "a" * 64) == (
        "cid:v1:e32edd4357bf42dfd3d6a05336d8737d8470f5b3853f6aaf2487adcfece06bdd"
    )


def test_cross_platform_manifest_bytes_have_frozen_sha256() -> None:
    payload = (
        b"dataset_name,relative_path,image_sha256\n"
        b"GC10-DET,images/sample.jpg,"
        + b"a" * 64
        + b"\n"
    )
    assert hashlib.sha256(payload).hexdigest() == (
        "0607f4b717b1efd5bed2e815d9c4b7af5177aaf46d7430a7e7603c0adbecb52e"
    )

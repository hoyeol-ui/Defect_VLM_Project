from __future__ import annotations

import json
from pathlib import Path

from defect_al.data.identity import compute_file_sha256
from defect_al.preflight import build_preflight_report


def test_preflight_pass_is_portable_and_does_not_authorize_training(
    tmp_path: Path,
    manifest_writer,
) -> None:
    paths = {}
    for split, token in (("train", "a"), ("validation", "b"), ("test", "c")):
        paths[split] = manifest_writer(
            tmp_path / "manifests" / f"{split}.csv",
            [
                {
                    "dataset_name": "GC10-DET",
                    "relative_path": f"images/{split}.jpg",
                    "image_sha256": token * 64,
                }
            ],
        )
    document = {
        "run_id": "portable_preflight_v1",
        "manifests": {
            split: {
                "path": path.relative_to(tmp_path).as_posix(),
                "manifest_sha256": compute_file_sha256(path),
            }
            for split, path in paths.items()
        },
    }
    config_path = tmp_path / "frozen.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")

    report = build_preflight_report(config_path, project_root=tmp_path)

    assert report["status"] == "PASS"
    assert len(report["preflight_module_sha256"]) == 64
    assert report["data_preflight_passed"] is True
    assert report["selection_authorized"] is False
    assert report["training_authorized"] is False
    assert report["config_path"] == "frozen.json"
    assert report["manifests"]["train"]["path"] == "manifests/train.csv"
    assert all(item["overlap_count"] == 0 for item in report["split_overlaps"])

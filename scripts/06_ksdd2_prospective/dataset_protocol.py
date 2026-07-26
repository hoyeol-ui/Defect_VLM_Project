"""Target-blind KSDD2 manifests and post-hoc mask auditing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from common import (
    project_path,
    read_csv,
    run_root,
    seal_file,
    sha256_file,
    stable_hash_key,
    write_csv,
)


BLIND_FIELDS = [
    "sample_id",
    "official_split",
    "role",
    "image_path",
    "posthoc_mask_path",
    "image_bytes",
    "mask_bytes",
    "image_sha256",
]


def _image_rows(directory: Path, split: str, exclusions: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image in sorted(directory.glob("*.png"), key=lambda path: path.name):
        relative = f"{split}/{image.name}"
        if image.stem.endswith("_GT") or relative in exclusions or "(copy)" in image.name:
            continue
        mask = image.with_name(f"{image.stem}_GT.png")
        if not mask.exists():
            raise FileNotFoundError(f"Missing paired mask for {image}")
        rows.append(
            {
                "sample_id": image.stem,
                "official_split": split,
                "role": "",
                "image_path": image.relative_to(project_path(".")).as_posix(),
                "posthoc_mask_path": (
                    mask.relative_to(project_path(".")).as_posix() if split == "train" else ""
                ),
                "image_bytes": image.stat().st_size,
                "mask_bytes": mask.stat().st_size if split == "train" else "",
                "image_sha256": sha256_file(image) if split == "train" else "",
            }
        )
    return rows


def prepare_manifests(config: dict[str, Any]) -> dict[str, Any]:
    out = run_root(config)
    out.mkdir(parents=True, exist_ok=True)
    if (out / config["outputs"]["freeze_manifest"]).exists():
        raise RuntimeError("Run is already frozen; prepare will not overwrite frozen manifests.")

    data = config["dataset"]
    archive = project_path(data["archive_path"])
    if not archive.exists():
        raise FileNotFoundError(f"Dataset archive missing: {archive}")
    if sha256_file(archive).lower() != data["archive_sha256"].lower():
        raise RuntimeError("Official KSDD2 archive SHA-256 mismatch.")

    root = project_path(data["data_root"])
    exclusions = set(data["known_archive_exclusions"])
    train_rows = _image_rows(root / data["train_dir"], "train", exclusions)
    final_rows = _image_rows(root / data["final_test_dir"], "test", exclusions)
    if len(train_rows) != int(data["expected_train_images"]):
        raise RuntimeError(f"Expected {data['expected_train_images']} train images, got {len(train_rows)}")
    if len(final_rows) != int(data["expected_final_images"]):
        raise RuntimeError(f"Expected {data['expected_final_images']} final images, got {len(final_rows)}")

    development_count = int(config["split"]["development_count"])
    salt = str(config["split"]["development_hash_salt"])
    ordered = sorted(train_rows, key=lambda row: stable_hash_key(salt, str(row["sample_id"])))
    development_ids = {str(row["sample_id"]) for row in ordered[:development_count]}
    for row in train_rows:
        row["role"] = (
            config["split"]["development_role"]
            if str(row["sample_id"]) in development_ids
            else config["split"]["acquisition_role"]
        )
    for row in final_rows:
        row["role"] = config["split"]["final_role"]

    blind = sorted(train_rows + final_rows, key=lambda row: (row["official_split"], row["sample_id"]))
    acquisition = [row for row in train_rows if row["role"] == config["split"]["acquisition_role"]]
    development = [row for row in train_rows if row["role"] == config["split"]["development_role"]]
    final_locked = [
        {
            "sample_id": row["sample_id"],
            "official_split": row["official_split"],
            "role": row["role"],
            "image_path": row["image_path"],
            "image_content_access_authorized": False,
            "mask_content_access_authorized": False,
        }
        for row in final_rows
    ]

    write_csv(out / config["outputs"]["blind_manifest"], blind, BLIND_FIELDS)
    write_csv(out / config["outputs"]["acquisition_manifest"], acquisition, BLIND_FIELDS)
    write_csv(out / config["outputs"]["development_manifest"], development, BLIND_FIELDS)
    write_csv(
        out / config["outputs"]["final_locked_manifest"],
        final_locked,
        [
            "sample_id",
            "official_split",
            "role",
            "image_path",
            "image_content_access_authorized",
            "mask_content_access_authorized",
        ],
    )

    initial_dir = out / "initial_sets"
    initial_dir.mkdir(parents=True, exist_ok=True)
    ids = np.array(sorted(str(row["sample_id"]) for row in acquisition), dtype=object)
    by_id = {str(row["sample_id"]): row for row in acquisition}
    initial_size = int(config["acquisition"]["initial_labeled_size"])
    for seed in config["acquisition"]["seeds"]:
        rng = np.random.default_rng(int(seed))
        selected = rng.choice(ids, size=initial_size, replace=False).tolist()
        rows = [
            {
                "sample_id": sample_id,
                "image_path": by_id[sample_id]["image_path"],
                "initial_rank": rank,
                "acquisition_seed": int(seed),
            }
            for rank, sample_id in enumerate(selected, start=1)
        ]
        write_csv(
            initial_dir / f"initial_seed{int(seed):02d}.csv",
            rows,
            ["sample_id", "image_path", "initial_rank", "acquisition_seed"],
        )

    return {
        "train_images": len(train_rows),
        "development_images": len(development),
        "acquisition_images": len(acquisition),
        "final_locked_images": len(final_locked),
        "initial_sets": len(config["acquisition"]["seeds"]),
        "final_mask_paths_recorded": 0,
        "final_content_read": False,
    }


def audit_mask(mask_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    binary = (mask > int(config["posthoc_defect_definition"]["mask_positive_threshold"])).astype(np.uint8)
    connectivity = int(config["posthoc_defect_definition"]["connectivity"])
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
    image_area = int(binary.shape[0] * binary.shape[1])
    small_max = float(config["posthoc_defect_definition"]["small_component_relative_area_max"])
    medium_max = float(config["posthoc_defect_definition"]["medium_component_relative_area_max"])
    rows = int(config["posthoc_defect_definition"]["centroid_grid_rows"])
    cols = int(config["posthoc_defect_definition"]["centroid_grid_cols"])

    areas: list[int] = []
    size_bins: list[str] = []
    centroid_cells: list[str] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        areas.append(area)
        relative = area / image_area
        size_bins.append("small" if relative <= small_max else "medium" if relative <= medium_max else "large")
        x, y = centroids[index]
        row = min(rows - 1, int(math.floor((y / binary.shape[0]) * rows)))
        col = min(cols - 1, int(math.floor((x / binary.shape[1]) * cols)))
        centroid_cells.append(f"r{row}c{col}")

    return {
        "has_defect": int(bool(areas)),
        "bbox_count": len(areas),
        "defect_area_pixels": int(binary.sum()),
        "defect_area_fraction": float(binary.mean()),
        "small_component_count": size_bins.count("small"),
        "medium_component_count": size_bins.count("medium"),
        "large_component_count": size_bins.count("large"),
        "size_bins_present": "|".join(sorted(set(size_bins))),
        "centroid_cells_present": "|".join(sorted(set(centroid_cells))),
        "mask_height": int(binary.shape[0]),
        "mask_width": int(binary.shape[1]),
    }


POSTHOC_FIELDS = [
    "sample_id",
    "role",
    "has_defect",
    "bbox_count",
    "defect_area_pixels",
    "defect_area_fraction",
    "small_component_count",
    "medium_component_count",
    "large_component_count",
    "size_bins_present",
    "centroid_cells_present",
    "mask_height",
    "mask_width",
]


def reveal_initial_labels(config: dict[str, Any], seed: int) -> Path:
    out = run_root(config)
    initial_path = out / "initial_sets" / f"initial_seed{seed:02d}.csv"
    initial = read_csv(initial_path)
    acquisition = {row["sample_id"]: row for row in read_csv(out / config["outputs"]["acquisition_manifest"])}
    rows: list[dict[str, Any]] = []
    for item in initial:
        source = acquisition[item["sample_id"]]
        metrics = audit_mask(project_path(source["posthoc_mask_path"]), config)
        rows.append(
            {
                "sample_id": item["sample_id"],
                "role": "initial_labeled",
                **metrics,
            }
        )
    target = out / "initial_labels" / f"initial_labels_seed{seed:02d}.csv"
    write_csv(target, rows, POSTHOC_FIELDS)
    seal_file(target)
    return target


def audit_manifest_rows(
    config: dict[str, Any],
    manifest_rows: list[dict[str, str]],
    *,
    role: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in manifest_rows:
        mask_path = item.get("posthoc_mask_path", "")
        if not mask_path:
            raise RuntimeError(f"Post-hoc mask path unavailable for {item['sample_id']}")
        metrics = audit_mask(project_path(mask_path), config)
        rows.append({"sample_id": item["sample_id"], "role": role, **metrics})
    return rows


def verify_prepared_manifests(config: dict[str, Any]) -> dict[str, Any]:
    out = run_root(config)
    acquisition = read_csv(out / config["outputs"]["acquisition_manifest"])
    development = read_csv(out / config["outputs"]["development_manifest"])
    final_locked = read_csv(out / config["outputs"]["final_locked_manifest"])
    initial_paths = sorted((out / "initial_sets").glob("initial_seed*.csv"))
    expected_acquisition = (
        int(config["dataset"]["expected_train_images"])
        - int(config["split"]["development_count"])
    )
    checks = {
        "acquisition_count": len(acquisition) == expected_acquisition,
        "development_count": len(development) == int(config["split"]["development_count"]),
        "final_count": len(final_locked) == int(config["dataset"]["expected_final_images"]),
        "final_has_no_mask_column": not any("mask" in key.lower() and "authorized" not in key.lower() for key in final_locked[0]),
        "final_access_false": all(
            row["image_content_access_authorized"].lower() == "false"
            and row["mask_content_access_authorized"].lower() == "false"
            for row in final_locked
        ),
        "initial_set_count": len(initial_paths) == len(config["acquisition"]["seeds"]),
        "initial_sets_exact_size": all(
            len(read_csv(path)) == int(config["acquisition"]["initial_labeled_size"])
            for path in initial_paths
        ),
        "initial_sets_target_blind": all(
            set(read_csv(path)[0]) == {"sample_id", "image_path", "initial_rank", "acquisition_seed"}
            for path in initial_paths
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Prepared manifest verification failed: {checks}")
    return checks

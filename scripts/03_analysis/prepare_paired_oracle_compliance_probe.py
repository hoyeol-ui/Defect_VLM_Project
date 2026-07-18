"""Prepare paired defect/background crops for a post-hoc VLM compliance probe.

GT is used only to construct and audit diagnostic views. Outputs are forbidden
for acquisition, detector training, and final-test evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = PROJECT_ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715"
DEFAULT_BOXES = PROTOCOL_DIR / "gc10_development_bbox_gt.csv"
DEFAULT_LOCKED = [
    PROJECT_ROOT / "runs" / "evaluation_protocol_v7" / "eval_protocol_20260711_173723" / "final_test_v7.csv",
    PROTOCOL_DIR / "gc10_final_test_locked.csv",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_locked_names(paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                for key in ("image_name", "filename", "image_path", "resolved_image_path"):
                    value = str(row.get(key, "")).strip()
                    if value:
                        names.add(Path(value).name.casefold())
    return names


def centered_crop(width: int, height: int, cx: float, cy: float, side: int) -> tuple[int, int, int, int]:
    if width < side or height < side:
        raise ValueError(f"Image {(width, height)} is smaller than crop side {side}")
    left = min(max(0, int(round(cx - side / 2))), width - side)
    top = min(max(0, int(round(cy - side / 2))), height - side)
    return left, top, left + side, top + side


def intersects(a: tuple[int, int, int, int], b: tuple[float, float, float, float]) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def expanded_box(row: pd.Series, margin: int, width: int, height: int) -> tuple[float, float, float, float]:
    return (
        max(0.0, float(row["x_min"]) - margin),
        max(0.0, float(row["y_min"]) - margin),
        min(float(width), float(row["x_max"]) + margin),
        min(float(height), float(row["y_max"]) + margin),
    )


def appearance_vector(image: Image.Image) -> np.ndarray:
    stat = ImageStat.Stat(image.convert("L"))
    return np.asarray([stat.mean[0], stat.stddev[0]], dtype=np.float64)


def grid_positions(limit: int, side: int, stride: int) -> list[int]:
    values = list(range(0, max(1, limit - side + 1), stride))
    endpoint = limit - side
    if endpoint >= 0 and endpoint not in values:
        values.append(endpoint)
    return sorted(set(values))


def choose_background(
    image: Image.Image,
    side: int,
    stride: int,
    forbidden: list[tuple[float, float, float, float]],
    target_vector: np.ndarray,
) -> tuple[tuple[int, int, int, int], float, int]:
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for top in grid_positions(image.height, side, stride):
        for left in grid_positions(image.width, side, stride):
            crop_box = (left, top, left + side, top + side)
            if any(intersects(crop_box, gt_box) for gt_box in forbidden):
                continue
            vector = appearance_vector(image.crop(crop_box))
            distance = float(np.linalg.norm(vector - target_vector))
            candidates.append((distance, crop_box))
    if not candidates:
        raise RuntimeError("No same-image background crop avoids all expanded GT boxes")
    candidates.sort(key=lambda item: (item[0], item[1][1], item[1][0]))
    return candidates[0][1], candidates[0][0], len(candidates)


def clipped_gt_norm(
    gt_box: tuple[float, float, float, float], crop_box: tuple[int, int, int, int]
) -> tuple[float, float, float, float]:
    left, top, right, bottom = crop_box
    side_x, side_y = right - left, bottom - top
    x1, y1 = max(gt_box[0], left), max(gt_box[1], top)
    x2, y2 = min(gt_box[2], right), min(gt_box[3], bottom)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("Positive crop does not intersect the primary GT box")
    return ((x1 - left) / side_x, (y1 - top) / side_y, (x2 - left) / side_x, (y2 - top) / side_y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--locked-manifest", type=Path, action="append", default=None)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--bbox-margin", type=int, default=16)
    parser.add_argument("--grid-stride", type=int, default=32)
    args = parser.parse_args()
    if min(args.crop_size, args.grid_stride) < 1 or args.bbox_margin < 0:
        raise ValueError("Invalid crop/grid/margin setting")

    oracle = pd.read_csv(args.oracle_audit)
    oracle = oracle[oracle["status"] == "created"].copy()
    all_boxes = pd.read_csv(args.boxes)
    box_groups = {key: group.copy() for key, group in all_boxes.groupby("sample_id", sort=False)}
    locked_paths = args.locked_manifest or DEFAULT_LOCKED
    locked = read_locked_names(locked_paths)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    view_dir = args.output_dir / "paired_views_GT_POSTHOC_ONLY"
    view_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for _, row in oracle.iterrows():
        sample_id = str(row["original_image_id"])
        source_path = Path(str(row["source_image_path"])).resolve()
        if source_path.name.casefold() in locked:
            raise RuntimeError(f"Locked-final source encountered: {source_path.name}")
        if sample_id not in box_groups:
            raise RuntimeError(f"Missing GT boxes for {sample_id}")
        primary = (float(row["gt_x1"]), float(row["gt_y1"]), float(row["gt_x2"]), float(row["gt_y2"]))
        with Image.open(source_path) as opened:
            image = opened.convert("RGB")
            positive_box = centered_crop(
                image.width, image.height,
                (primary[0] + primary[2]) / 2,
                (primary[1] + primary[3]) / 2,
                args.crop_size,
            )
            positive = image.crop(positive_box)
            target = appearance_vector(positive)
            forbidden = [
                expanded_box(gt_row, args.bbox_margin, image.width, image.height)
                for _, gt_row in box_groups[sample_id].iterrows()
            ]
            negative_box, appearance_distance, candidate_count = choose_background(
                image, args.crop_size, args.grid_stride, forbidden, target
            )
            negative = image.crop(negative_box)

        pair_id = f"{sample_id}__pair"
        positive_id = f"{sample_id}__forced_positive"
        negative_id = f"{sample_id}__matched_background"
        positive_path = view_dir / f"{positive_id}.jpg"
        negative_path = view_dir / f"{negative_id}.jpg"
        positive.save(positive_path, quality=95, subsampling=0)
        negative.save(negative_path, quality=95, subsampling=0)
        gt_norm = clipped_gt_norm(primary, positive_box)

        for view_id, view_path, view_type in (
            (positive_id, positive_path, "gt_primary_positive"),
            (negative_id, negative_path, "same_image_gt_excluded_background"),
        ):
            manifest_rows.append(
                {
                    "image_id": view_id,
                    "image_path": str(view_path.resolve()),
                    "dataset": "GC10-DET",
                    "split_role": "development_paired_oracle_compliance_audit",
                }
            )
        common = {
            "pair_id": pair_id,
            "original_image_id": sample_id,
            "source_image_path": str(source_path),
            "gt_used_for_view": True,
            "allowed_for_acquisition": False,
        }
        audit_rows.extend(
            [
                {
                    **common,
                    "image_id": positive_id,
                    "expected_defect_present": True,
                    "view_type": "gt_primary_positive",
                    "crop_left": positive_box[0], "crop_top": positive_box[1],
                    "crop_right": positive_box[2], "crop_bottom": positive_box[3],
                    "gt_box_x1_norm": gt_norm[0], "gt_box_y1_norm": gt_norm[1],
                    "gt_box_x2_norm": gt_norm[2], "gt_box_y2_norm": gt_norm[3],
                    "background_candidate_count": np.nan,
                    "appearance_distance_to_positive": 0.0,
                },
                {
                    **common,
                    "image_id": negative_id,
                    "expected_defect_present": False,
                    "view_type": "same_image_gt_excluded_background",
                    "crop_left": negative_box[0], "crop_top": negative_box[1],
                    "crop_right": negative_box[2], "crop_bottom": negative_box[3],
                    "gt_box_x1_norm": np.nan, "gt_box_y1_norm": np.nan,
                    "gt_box_x2_norm": np.nan, "gt_box_y2_norm": np.nan,
                    "background_candidate_count": candidate_count,
                    "appearance_distance_to_positive": appearance_distance,
                },
            ]
        )

    manifest = pd.DataFrame(manifest_rows)
    audit = pd.DataFrame(audit_rows)
    if len(manifest) != 2 * len(oracle):
        raise RuntimeError("Every oracle image must produce exactly one positive/negative pair")
    manifest_path = args.output_dir / "paired_probe_manifest.csv"
    audit_path = args.output_dir / "paired_probe_gt_audit.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    config = {
        "purpose": "paired structured-output compliance and visual discrimination probe",
        "gt_used_for_view": True,
        "allowed_for_acquisition": False,
        "allowed_for_detector_training": False,
        "final_test_evaluated": False,
        "source_pairs": len(oracle),
        "views": len(manifest),
        "crop_size": args.crop_size,
        "bbox_margin": args.bbox_margin,
        "grid_stride": args.grid_stride,
        "oracle_audit_sha256": sha256_file(args.oracle_audit),
        "boxes_sha256": sha256_file(args.boxes),
        "manifest_sha256": sha256_file(manifest_path),
        "gt_audit_sha256": sha256_file(audit_path),
        "locked_manifests": [str(path.resolve()) for path in locked_paths],
        "frozen_gate": {
            "json_parse_rate_min": 0.90,
            "schema_compliance_rate_min": 0.90,
            "positive_sensitivity_min": 0.70,
            "background_specificity_min": 0.70,
            "balanced_accuracy_min": 0.70,
            "positive_bbox_coverage_min": 0.70,
            "positive_median_bbox_iou_min": 0.10,
        },
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] pairs={len(oracle)} views={len(manifest)}")
    print(f"[MANIFEST] {manifest_path}")
    print(f"[GT AUDIT] {audit_path}")


if __name__ == "__main__":
    main()

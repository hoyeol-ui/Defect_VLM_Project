"""Create GT-oracle crops to diagnose VLM visual-resolution limitations.

The crops are strictly post-hoc diagnostic views. They must never enter an
acquisition score, detector training set, or final-test evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = PROJECT_ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715"
DEFAULT_METADATA = PROTOCOL_DIR / "gc10_development_eval.csv"
DEFAULT_BOXES = PROTOCOL_DIR / "gc10_development_bbox_gt.csv"
DEFAULT_LOCKED = [
    PROJECT_ROOT / "runs" / "evaluation_protocol_v7" / "eval_protocol_20260711_173723" / "final_test_v7.csv",
    PROTOCOL_DIR / "gc10_final_test_locked.csv",
]


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


def choose_square_crop(
    width: int,
    height: int,
    box: tuple[float, float, float, float],
    context_scale: float,
    min_crop_size: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * context_scale
    side = min(max(side, float(min_crop_size)), float(max(width, height)))
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = int(round(cx + side / 2.0))
    bottom = int(round(cy + side / 2.0))

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    left, top = max(0, left), max(0, top)
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop: {(left, top, right, bottom)}")
    return left, top, right, bottom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--locked-manifest", type=Path, action="append", default=None)
    parser.add_argument("--context-scale", type=float, default=2.5)
    parser.add_argument("--min-crop-size", type=int, default=256)
    args = parser.parse_args()

    if args.context_scale < 1.0:
        raise ValueError("--context-scale must be at least 1.0")
    plan = pd.read_csv(args.pilot_plan).drop_duplicates("image_id")
    metadata = pd.read_csv(args.metadata).set_index("sample_id")
    boxes = pd.read_csv(args.boxes)
    locked = read_locked_names(args.locked_manifest or DEFAULT_LOCKED)
    box_groups = {key: group.copy() for key, group in boxes.groupby("sample_id", sort=False)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output_dir / "oracle_crops_GT_POSTHOC_ONLY"
    crop_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for _, plan_row in plan.iterrows():
        image_id = str(plan_row["image_id"])
        source_path = Path(str(plan_row["image_path"])).resolve()
        if source_path.name.casefold() in locked:
            audit_rows.append(
                {
                    "original_image_id": image_id,
                    "source_image_path": str(source_path),
                    "status": "excluded_locked_final",
                }
            )
            continue
        if image_id not in metadata.index or image_id not in box_groups:
            audit_rows.append(
                {
                    "original_image_id": image_id,
                    "source_image_path": str(source_path),
                    "status": "missing_metadata_or_gt_box",
                }
            )
            continue
        meta = metadata.loc[image_id]
        gt_group = box_groups[image_id].copy()
        gt_group["pixel_area"] = gt_group["bbox_width"] * gt_group["bbox_height"]
        primary = gt_group.sort_values("pixel_area", ascending=False).iloc[0]
        gt_box = (
            float(primary["x_min"]),
            float(primary["y_min"]),
            float(primary["x_max"]),
            float(primary["y_max"]),
        )
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            crop_box = choose_square_crop(
                image.width,
                image.height,
                gt_box,
                args.context_scale,
                args.min_crop_size,
            )
            crop = image.crop(crop_box)
            view_id = f"{image_id}__oracle_primary"
            crop_path = crop_dir / f"{view_id}.jpg"
            crop.save(crop_path, quality=95, subsampling=0)

        left, top, right, bottom = crop_box
        crop_width, crop_height = right - left, bottom - top
        manifest_rows.append(
            {
                "image_id": view_id,
                "image_path": str(crop_path.resolve()),
                "dataset": "GC10-DET",
                "split_role": "development_oracle_crop_audit",
                "original_image_id": image_id,
                "original_filename": str(meta["filename"]),
                "view_type": "gt_oracle_primary_crop",
                "gt_used_for_view": True,
            }
        )
        audit_rows.append(
            {
                "original_image_id": image_id,
                "view_id": view_id,
                "source_image_path": str(source_path),
                "crop_image_path": str(crop_path.resolve()),
                "status": "created",
                "gt_class_id": primary["class_id"],
                "gt_class_name": primary["class_name"],
                "gt_x1": gt_box[0],
                "gt_y1": gt_box[1],
                "gt_x2": gt_box[2],
                "gt_y2": gt_box[3],
                "crop_left": left,
                "crop_top": top,
                "crop_right": right,
                "crop_bottom": bottom,
                "gt_box_x1_in_crop_norm": (gt_box[0] - left) / crop_width,
                "gt_box_y1_in_crop_norm": (gt_box[1] - top) / crop_height,
                "gt_box_x2_in_crop_norm": (gt_box[2] - left) / crop_width,
                "gt_box_y2_in_crop_norm": (gt_box[3] - top) / crop_height,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    audit = pd.DataFrame(audit_rows)
    if manifest.empty:
        raise RuntimeError("No safe oracle crops were created")
    manifest_path = args.output_dir / "oracle_crop_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    audit.to_csv(args.output_dir / "oracle_crop_gt_audit.csv", index=False, encoding="utf-8-sig")
    config = {
        "purpose": "posthoc VLM visual upper-bound diagnostic only",
        "gt_used_for_view": True,
        "allowed_for_acquisition": False,
        "allowed_for_detector_training": False,
        "final_test_evaluated": False,
        "context_scale": args.context_scale,
        "min_crop_size": args.min_crop_size,
        "source_pilot_images": len(plan),
        "created_crops": len(manifest),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] oracle_crops={len(manifest)}")
    print(f"[MANIFEST] {manifest_path}")


if __name__ == "__main__":
    main()


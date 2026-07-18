"""Audit MPDD and create a leakage-resistant annotation-triage protocol."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "MPDD"
DEFAULT_OUT = ROOT / "runs" / "mpdd_annotation_triage_protocol" / "mpdd_protocol_20260715"
EXPECTED_CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
SPLIT_SEED = 20260715


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash64(path: Path) -> int:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    dct = cv2.dct(gray)[:8, :8]
    threshold = float(np.median(dct.reshape(-1)[1:]))
    value = 0
    for bit in (dct >= threshold).reshape(-1):
        value = (value << 1) | int(bit)
    return value


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) ^ SPLIT_SEED


def mask_components(mask: np.ndarray, width: int, height: int) -> tuple[list[dict[str, Any]], float]:
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    binary = (mask != 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rows: list[dict[str, Any]] = []
    for component_id in range(1, count):
        x, y, box_width, box_height, area = [int(v) for v in stats[component_id]]
        rows.append({
            "component_id": component_id,
            "x_min": x,
            "y_min": y,
            "x_max_exclusive": x + box_width,
            "y_max_exclusive": y + box_height,
            "bbox_width": box_width,
            "bbox_height": box_height,
            "component_pixel_area": area,
            "bbox_area_ratio": box_width * box_height / float(width * height),
            "component_area_ratio": area / float(width * height),
        })
    return rows, float(binary.mean())


def audit_source(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    categories = sorted(path.name for path in source.iterdir() if path.is_dir())
    if categories != EXPECTED_CATEGORIES:
        raise RuntimeError(f"Unexpected MPDD categories: {categories}")
    rows: list[dict[str, Any]] = []
    boxes: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for category in categories:
        category_root = source / category
        for official_split in ("train", "test"):
            split_root = category_root / official_split
            if not split_root.exists():
                errors.append({"sample_id": category, "error": "missing_split", "path": str(split_root)})
                continue
            for type_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
                anomaly_type = type_dir.name
                is_anomaly = anomaly_type != "good"
                for image_path in sorted(type_dir.glob("*.png")):
                    relative = image_path.relative_to(source).as_posix()
                    sample_id = f"mpdd::{category}::{official_split}::{anomaly_type}::{image_path.stem}"
                    mask_path = category_root / "ground_truth" / anomaly_type / f"{image_path.stem}_mask.png" if is_anomaly else None
                    if is_anomaly and not mask_path.exists():
                        errors.append({"sample_id": sample_id, "error": "missing_mask", "path": str(mask_path)})
                        continue
                    with Image.open(image_path) as image:
                        width, height = image.size
                    component_rows: list[dict[str, Any]] = []
                    defect_area = 0.0
                    mask_hash = ""
                    if mask_path is not None:
                        with Image.open(mask_path) as mask_image:
                            mask_array = np.asarray(mask_image)
                            mask_width, mask_height = mask_image.size
                        if (mask_width, mask_height) != (width, height):
                            errors.append({"sample_id": sample_id, "error": "image_mask_size_mismatch", "path": str(mask_path)})
                            continue
                        component_rows, defect_area = mask_components(mask_array, width, height)
                        if not component_rows or defect_area <= 0:
                            errors.append({"sample_id": sample_id, "error": "empty_mask", "path": str(mask_path)})
                            continue
                        mask_hash = file_sha256(mask_path)
                        boxes.extend({"sample_id": sample_id, "product_category": category, "anomaly_type": anomaly_type, **box} for box in component_rows)
                    rows.append({
                        "sample_id": sample_id,
                        "product_category": category,
                        "official_split": official_split,
                        "anomaly_type": anomaly_type,
                        "is_anomaly": is_anomaly,
                        "image_relative_path": relative,
                        "image_path": str(image_path.resolve()),
                        "mask_path": str(mask_path.resolve()) if mask_path else "",
                        "width": width,
                        "height": height,
                        "image_sha256": file_sha256(image_path),
                        "phash64": f"{phash64(image_path):016x}",
                        "mask_sha256": mask_hash,
                        "num_connected_components": len(component_rows),
                        "defect_pixel_area_ratio": defect_area,
                    })
    manifest = pd.DataFrame(rows)
    box_frame = pd.DataFrame(boxes)
    if len(manifest) != 1346 or int(manifest["is_anomaly"].sum()) != 282:
        raise RuntimeError(f"Expected 1346 images/282 anomalies, got {len(manifest)}/{int(manifest['is_anomaly'].sum())}")
    mask_files = list(source.glob("*/ground_truth/*/*.png"))
    if len(mask_files) != 282:
        raise RuntimeError(f"Expected 282 mask files, got {len(mask_files)}")
    if errors:
        raise RuntimeError(f"Source audit found {len(errors)} errors; first={errors[0]}")
    return manifest, box_frame, errors


def assign_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    output = manifest.copy()
    output["duplicate_group_id"] = output["image_sha256"].map(lambda value: "mpdd_sha_" + str(value)[:16])
    sizes = output.groupby("duplicate_group_id").size()
    output["duplicate_group_size"] = output["duplicate_group_id"].map(sizes).astype(int)
    group_meta = output.groupby("duplicate_group_id").agg(
        group_size=("sample_id", "size"),
        product_categories=("product_category", lambda x: sorted(set(x))),
        official_splits=("official_split", lambda x: sorted(set(x))),
        anomaly_types=("anomaly_type", lambda x: sorted(set(x))),
    ).reset_index()
    mixed = group_meta[
        group_meta["product_categories"].map(len).ne(1)
        | group_meta["official_splits"].map(len).ne(1)
        | group_meta["anomaly_types"].map(len).ne(1)
    ]
    if len(mixed):
        raise RuntimeError(f"Exact duplicates cross protocol strata: {len(mixed)}")
    group_meta["product_category"] = group_meta["product_categories"].str[0]
    group_meta["official_split"] = group_meta["official_splits"].str[0]
    group_meta["anomaly_type"] = group_meta["anomaly_types"].str[0]
    assignments: dict[str, str] = {}
    ratios = {"acquisition": 0.8, "development": 0.1, "final_locked": 0.1}
    for keys, sub in group_meta.groupby(["product_category", "official_split", "anomaly_type"], sort=True):
        rng = np.random.default_rng(stable_seed("|".join(keys)))
        shuffled = sub.iloc[rng.permutation(len(sub))].to_dict("records")
        total = int(sub["group_size"].sum())
        targets = {name: ratio * total for name, ratio in ratios.items()}
        counts = {name: 0 for name in ratios}
        for group in shuffled:
            scores = {name: (targets[name] - counts[name]) / max(targets[name], 1.0) for name in ratios}
            chosen = max(scores, key=lambda name: (scores[name], name))
            assignments[str(group["duplicate_group_id"])] = chosen
            counts[chosen] += int(group["group_size"])
    output["protocol_split"] = output["duplicate_group_id"].map(assignments)
    if output["protocol_split"].isna().any():
        raise RuntimeError("Unassigned protocol rows")
    return output


def leakage_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left, right in itertools.combinations(["acquisition", "development", "final_locked"], 2):
        a = manifest[manifest["protocol_split"].eq(left)]
        b = manifest[manifest["protocol_split"].eq(right)]
        rows.append({
            "split_left": left,
            "split_right": right,
            "sha256_overlap": len(set(a["image_sha256"]) & set(b["image_sha256"])),
            "duplicate_group_overlap": len(set(a["duplicate_group_id"]) & set(b["duplicate_group_id"])),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest, boxes, errors = audit_source(source)
    manifest = assign_splits(manifest)
    leakage = leakage_audit(manifest)
    if leakage[["sha256_overlap", "duplicate_group_overlap"]].to_numpy().sum() != 0:
        raise RuntimeError("Split leakage audit failed")
    acquisition = manifest[manifest["protocol_split"].eq("acquisition")].copy()
    development = manifest[manifest["protocol_split"].eq("development")].copy()
    final_locked = manifest[manifest["protocol_split"].eq("final_locked")].copy()

    manifest.to_csv(out / "mpdd_source_audit.csv", index=False, encoding="utf-8-sig")
    boxes.to_csv(out / "mpdd_mask_bbox_audit.csv", index=False, encoding="utf-8-sig")
    acquisition.to_csv(out / "mpdd_acquisition_pool_gt_audit.csv", index=False, encoding="utf-8-sig")
    acquisition[["sample_id", "image_path"]].to_csv(out / "mpdd_acquisition_loader_private.csv", index=False, encoding="utf-8-sig")
    blind_columns = ["sample_id", "product_category", "image_sha256", "phash64", "duplicate_group_id"]
    acquisition[blind_columns].to_csv(out / "mpdd_acquisition_pool_blind.csv", index=False, encoding="utf-8-sig")
    boxes[boxes["sample_id"].isin(set(acquisition["sample_id"]))].to_csv(out / "mpdd_acquisition_bbox_gt_audit.csv", index=False, encoding="utf-8-sig")
    development.to_csv(out / "mpdd_development_eval.csv", index=False, encoding="utf-8-sig")
    final_locked.to_csv(out / "mpdd_final_test_locked.csv", index=False, encoding="utf-8-sig")
    leakage.to_csv(out / "mpdd_split_leakage_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors, columns=["sample_id", "error", "path"]).to_csv(out / "mpdd_source_errors.csv", index=False, encoding="utf-8-sig")

    final_path = out / "mpdd_final_test_locked.csv"
    config = {
        "protocol": "MPDD Independent Hierarchical-DINO Annotation Triage",
        "created_date": "2026-07-15",
        "source_root": str(source),
        "split_seed": SPLIT_SEED,
        "split_strata": ["product_category", "official_split", "anomaly_type"],
        "split_ratios": {"acquisition": 0.8, "development": 0.1, "final_locked": 0.1},
        "selector_allowed_metadata": ["product_category"],
        "source_paths_visible_to_selector": False,
        "hard_duplicate_grouping": "exact_sha256",
        "phash_used_for_split": False,
        "num_source_images": len(manifest),
        "num_anomaly_images": int(manifest["is_anomaly"].sum()),
        "split_sizes": manifest["protocol_split"].value_counts().to_dict(),
        "final_locked_manifest": str(final_path),
        "final_locked_manifest_sha256": file_sha256(final_path),
        "detector_training_performed": False,
        "selection_performed": False,
        "final_test_evaluated": False,
    }
    (out / "mpdd_protocol_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    public = manifest[~manifest["protocol_split"].eq("final_locked")].groupby(
        ["protocol_split", "product_category", "official_split", "is_anomaly"]
    ).size().reset_index(name="num_images")
    size_counts = manifest.groupby(["width", "height"]).size().reset_index(name="num_images")
    report = [
        "# MPDD Ingestion and Split Audit", "",
        "- Detector training performed: **False**",
        "- Selection performed: **False**",
        "- Final test evaluated: **False**",
        f"- Source inspection images: **{len(manifest)}**",
        f"- Source anomaly images/masks: **{int(manifest['is_anomaly'].sum())}**",
        f"- Connected-component boxes: **{len(boxes)}**",
        f"- Exact duplicate groups with size > 1: **{int((manifest.groupby('duplicate_group_id').size() > 1).sum())}**",
        f"- Unique pHash values (audit only): **{manifest['phash64'].nunique()}**",
        f"- Final locked manifest SHA-256: `{config['final_locked_manifest_sha256']}`", "",
        "## Image sizes", "", size_counts.to_markdown(index=False), "",
        "## Acquisition/development distribution", "", public.to_markdown(index=False), "",
        "Final split label distribution is intentionally not printed or used.", "",
        "## Leakage audit", "", leakage.to_markdown(index=False), "",
    ]
    summary = out / "mpdd_ingestion_audit_summary.md"
    summary.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("=" * 100)
    print("[DONE] MPDD ingestion/split protocol")
    print("Detector training performed: False")
    print("Selection performed: False")
    print("Final test evaluated: False")
    print(f"[SUMMARY] {summary}")
    print("=" * 100)


if __name__ == "__main__":
    main()

"""Prepare the VisA annotation-triage benchmark without training any model.

The script audits the official source, converts masks to post-hoc component
boxes, groups exact duplicates, and creates deterministic 80/10/10
acquisition/development/final manifests. pHash is audit-only. It never runs
inference or training.
"""

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
DEFAULT_SOURCE = ROOT / "data" / "VisA"
DEFAULT_OUT = ROOT / "runs" / "visa_annotation_triage_protocol" / "visa_protocol_20260715"
EXPECTED_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
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
    bits = (dct >= threshold).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) ^ SPLIT_SEED


def component_boxes(mask: np.ndarray, image_width: int, image_height: int) -> tuple[list[dict[str, Any]], float, list[int]]:
    if mask.ndim == 3:
        mask = mask[..., 0]
    ids = [int(value) for value in np.unique(mask) if int(value) != 0]
    rows: list[dict[str, Any]] = []
    union = mask != 0
    for defect_id in ids:
        binary = (mask == defect_id).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for component_id in range(1, count):
            x, y, width, height, area = [int(value) for value in stats[component_id]]
            rows.append(
                {
                    "mask_defect_id": defect_id,
                    "component_id_within_defect_id": component_id,
                    "x_min": x,
                    "y_min": y,
                    "x_max_exclusive": x + width,
                    "y_max_exclusive": y + height,
                    "bbox_width": width,
                    "bbox_height": height,
                    "component_pixel_area": area,
                    "bbox_area_ratio": (width * height) / float(image_width * image_height),
                    "component_area_ratio": area / float(image_width * image_height),
                }
            )
    return rows, float(union.mean()), ids


def audit_source(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    manifest_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    annotation_files = sorted(source.glob("*/image_anno.csv"))
    categories = [path.parent.name for path in annotation_files]
    if categories != sorted(EXPECTED_CATEGORIES):
        raise RuntimeError(f"Unexpected VisA categories: {categories}")

    for annotation_path in annotation_files:
        category = annotation_path.parent.name
        frame = pd.read_csv(annotation_path)
        if list(frame.columns) != ["image", "label", "mask"]:
            raise RuntimeError(f"Unexpected columns in {annotation_path}: {list(frame.columns)}")
        for source_row, row in frame.iterrows():
            relative_image = str(row["image"]).replace("\\", "/")
            image_path = (source / relative_image).resolve()
            label_raw = str(row["label"]).strip()
            is_anomaly = label_raw.lower() != "normal"
            mask_value = "" if pd.isna(row["mask"]) else str(row["mask"]).strip().replace("\\", "/")
            mask_path = (source / mask_value).resolve() if mask_value else None
            sample_id = f"{category}::{relative_image}"
            if not image_path.exists():
                errors.append({"sample_id": sample_id, "error": "missing_image", "path": str(image_path)})
                continue
            if is_anomaly != bool(mask_value):
                errors.append({"sample_id": sample_id, "error": "label_mask_presence_mismatch", "path": str(mask_path or "")})
                continue
            if mask_path is not None and not mask_path.exists():
                errors.append({"sample_id": sample_id, "error": "missing_mask", "path": str(mask_path)})
                continue

            with Image.open(image_path) as image:
                width, height = image.size
            image_hash = file_sha256(image_path)
            perceptual_hash = phash64(image_path)
            mask_hash = ""
            mask_ids: list[int] = []
            defect_area_ratio = 0.0
            boxes: list[dict[str, Any]] = []
            if mask_path is not None:
                mask = np.asarray(Image.open(mask_path))
                mask_height, mask_width = mask.shape[:2]
                if (mask_width, mask_height) != (width, height):
                    errors.append({"sample_id": sample_id, "error": "image_mask_size_mismatch", "path": str(mask_path)})
                    continue
                boxes, defect_area_ratio, mask_ids = component_boxes(mask, width, height)
                if not boxes or defect_area_ratio <= 0:
                    errors.append({"sample_id": sample_id, "error": "empty_anomaly_mask", "path": str(mask_path)})
                    continue
                mask_hash = file_sha256(mask_path)
                for item in boxes:
                    box_rows.append({"sample_id": sample_id, "category": category, **item})

            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "category": category,
                    "source_annotation_csv": str(annotation_path),
                    "source_row": int(source_row),
                    "image_relative_path": relative_image,
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "label_raw": label_raw,
                    "is_anomaly": bool(is_anomaly),
                    "mask_relative_path": mask_value,
                    "mask_path": str(mask_path) if mask_path else "",
                    "width": width,
                    "height": height,
                    "image_sha256": image_hash,
                    "phash64": f"{perceptual_hash:016x}",
                    "mask_sha256": mask_hash,
                    "mask_defect_ids": "|".join(map(str, mask_ids)),
                    "num_mask_defect_ids": len(mask_ids),
                    "num_connected_components": len(boxes),
                    "defect_pixel_area_ratio": defect_area_ratio,
                }
            )
            if len(manifest_rows) % 500 == 0:
                print(f"[AUDIT] images={len(manifest_rows)}", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    boxes = pd.DataFrame(box_rows)
    if len(manifest) != 10821:
        raise RuntimeError(f"Expected 10821 valid images, got {len(manifest)}; errors={len(errors)}")
    if int(manifest["is_anomaly"].sum()) != 1200:
        raise RuntimeError(f"Expected 1200 anomaly images, got {int(manifest['is_anomaly'].sum())}")
    if errors:
        raise RuntimeError(f"Source audit found {len(errors)} errors; first={errors[0]}")
    return manifest, boxes, errors


def duplicate_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    """Use only byte-identical files as hard split groups.

    VisA contains aligned industrial views, so pHash proximity represents the
    task distribution rather than identity leakage. pHash remains in the
    manifest for audit, but it must not determine split membership.
    """
    output = manifest.copy()
    output["duplicate_group_id"] = output["image_sha256"].map(lambda value: "visa_sha_" + str(value)[:16])
    sizes = output.groupby("duplicate_group_id").size()
    output["duplicate_group_size"] = output["duplicate_group_id"].map(sizes).astype(int)
    return output


def assign_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    group_meta = manifest.groupby("duplicate_group_id").agg(
        group_size=("sample_id", "size"),
        categories=("category", lambda values: sorted(set(values))),
        labels=("is_anomaly", lambda values: sorted(set(bool(v) for v in values))),
    ).reset_index()
    mixed_categories = group_meta[group_meta["categories"].map(len) != 1]
    if len(mixed_categories):
        raise RuntimeError(f"Perceptual duplicate groups cross object categories: {len(mixed_categories)}")
    group_meta["category"] = group_meta["categories"].str[0]
    group_meta["label_stratum"] = group_meta["labels"].map(
        lambda values: "mixed" if len(values) > 1 else ("anomaly" if values[0] else "normal")
    )
    assignments: dict[str, str] = {}
    ratios = {"acquisition": 0.8, "development": 0.1, "final_locked": 0.1}
    for (category, label_stratum), sub in group_meta.groupby(["category", "label_stratum"], sort=True):
        sub = sub.copy()
        rng = np.random.default_rng(stable_seed(f"{category}|{label_stratum}"))
        order = rng.permutation(len(sub))
        groups = sub.iloc[order].to_dict("records")
        total = int(sub["group_size"].sum())
        targets = {name: ratio * total for name, ratio in ratios.items()}
        counts = {name: 0 for name in ratios}
        for group in groups:
            scores = {name: (targets[name] - counts[name]) / max(targets[name], 1.0) for name in ratios}
            split = max(scores, key=lambda name: (scores[name], name))
            assignments[str(group["duplicate_group_id"])] = split
            counts[split] += int(group["group_size"])
    output = manifest.copy()
    output["protocol_split"] = output["duplicate_group_id"].map(assignments)
    if output["protocol_split"].isna().any():
        raise RuntimeError("Some samples were not assigned to a split.")
    return output


def leakage_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    splits = ["acquisition", "development", "final_locked"]
    for left, right in itertools.combinations(splits, 2):
        a = manifest[manifest["protocol_split"].eq(left)]
        b = manifest[manifest["protocol_split"].eq(right)]
        rows.append(
            {
                "split_left": left,
                "split_right": right,
                "path_overlap": len(set(a["image_path"]) & set(b["image_path"])),
                "sha256_overlap": len(set(a["image_sha256"]) & set(b["image_sha256"])),
                "exact_duplicate_group_overlap": len(set(a["duplicate_group_id"]) & set(b["duplicate_group_id"])),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-tar", type=Path, default=ROOT / "downloads" / "VisA_20220922.tar")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(source)

    manifest, boxes, errors = audit_source(source)
    manifest = duplicate_groups(manifest)
    manifest = assign_splits(manifest)
    leakage = leakage_audit(manifest)
    if leakage[["path_overlap", "sha256_overlap", "exact_duplicate_group_overlap"]].to_numpy().sum() != 0:
        raise RuntimeError("Split leakage audit failed.")

    source_audit = manifest.copy()
    acquisition = manifest[manifest["protocol_split"].eq("acquisition")].copy()
    development = manifest[manifest["protocol_split"].eq("development")].copy()
    final_locked = manifest[manifest["protocol_split"].eq("final_locked")].copy()
    blind_columns = ["sample_id", "image_path", "image_name", "image_sha256", "phash64", "duplicate_group_id"]

    source_audit.to_csv(out / "visa_source_audit.csv", index=False, encoding="utf-8-sig")
    boxes.to_csv(out / "visa_mask_bbox_audit.csv", index=False, encoding="utf-8-sig")
    boxes[boxes["sample_id"].isin(set(acquisition["sample_id"]))].to_csv(
        out / "visa_acquisition_bbox_gt_audit.csv", index=False, encoding="utf-8-sig"
    )
    acquisition.to_csv(out / "visa_acquisition_pool_gt_audit.csv", index=False, encoding="utf-8-sig")
    acquisition[blind_columns].to_csv(out / "visa_acquisition_pool_blind.csv", index=False, encoding="utf-8-sig")
    development.to_csv(out / "visa_development_eval.csv", index=False, encoding="utf-8-sig")
    final_locked.to_csv(out / "visa_final_test_locked.csv", index=False, encoding="utf-8-sig")
    leakage.to_csv(out / "visa_split_leakage_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors).to_csv(out / "visa_source_errors.csv", index=False, encoding="utf-8-sig")

    final_path = out / "visa_final_test_locked.csv"
    config = {
        "protocol": "VisA GT-Free Annotation Triage Pilot",
        "created_date": "2026-07-15",
        "source_root": str(source),
        "source_tar": str(args.source_tar.resolve()),
        "source_tar_sha256": file_sha256(args.source_tar.resolve()) if args.source_tar.exists() else None,
        "official_dataset_license": "CC BY 4.0",
        "split_seed": SPLIT_SEED,
        "split_ratios": {"acquisition": 0.8, "development": 0.1, "final_locked": 0.1},
        "hard_duplicate_grouping": "exact_sha256",
        "phash_used_for_split": False,
        "phash_role": "audit_only_for_aligned_industrial_images",
        "num_source_images": len(manifest),
        "num_anomaly_images": int(manifest["is_anomaly"].sum()),
        "split_sizes": manifest["protocol_split"].value_counts().to_dict(),
        "acquisition_blind_manifest": str(out / "visa_acquisition_pool_blind.csv"),
        "acquisition_gt_audit_manifest": str(out / "visa_acquisition_pool_gt_audit.csv"),
        "development_manifest": str(out / "visa_development_eval.csv"),
        "final_locked_manifest": str(final_path),
        "final_locked_manifest_sha256": file_sha256(final_path),
        "detector_training_performed": False,
        "selection_performed": False,
        "final_test_evaluated": False,
    }
    (out / "visa_protocol_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    split_summary = manifest.groupby(["protocol_split", "category", "is_anomaly"]).size().reset_index(name="num_images")
    public_summary = split_summary[~split_summary["protocol_split"].eq("final_locked")]
    report = [
        "# VisA Ingestion and Split Audit",
        "",
        "- Detector training performed: **False**",
        "- Selection performed: **False**",
        "- Final test evaluated: **False**",
        f"- Official source images: **{len(manifest)}**",
        f"- Official anomaly images: **{int(manifest['is_anomaly'].sum())}**",
        f"- Connected-component boxes recovered: **{len(boxes)}**",
        f"- Exact SHA-256 duplicate groups with size > 1: **{int((manifest.groupby('duplicate_group_id').size() > 1).sum())}**",
        f"- Unique pHash values (audit only): **{manifest['phash64'].nunique()}**",
        f"- Final locked manifest SHA-256: `{config['final_locked_manifest_sha256']}`",
        "",
        "## Acquisition/development distribution",
        "",
        public_summary.to_markdown(index=False),
        "",
        "Final split label distribution is intentionally not printed. Its manifest is sealed by hash and must not be loaded by selection or development evaluation code.",
        "",
        "## Leakage audit",
        "",
        leakage.to_markdown(index=False),
        "",
        "Phase 0 passes only when the source error table is empty and every leakage count is zero.",
    ]
    summary_path = out / "visa_ingestion_audit_summary.md"
    summary_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("=" * 100)
    print("[DONE] VisA ingestion/split protocol")
    print("Detector training performed: False")
    print("Selection performed: False")
    print("Final test evaluated: False")
    print(f"[SUMMARY] {summary_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()

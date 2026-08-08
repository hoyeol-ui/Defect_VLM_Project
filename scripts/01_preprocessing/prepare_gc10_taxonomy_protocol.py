"""Audit GC10-DET and create a leakage-resistant taxonomy-triage protocol.

Folder names and paths encode defect labels, so selector-facing manifests use
opaque IDs only. Exact duplicates and filename-derived production bursts are
hard-grouped across the 80/10/10 split. No model inference or training occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "GC10-DET"
DEFAULT_OUT = ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715"
SPLIT_SEED = 20260715
EXPECTED_JPGS = 2312
EXPECTED_XMLS = 2294
EXPECTED_FOLDERS = [str(value) for value in range(1, 11)]
CANONICAL_CLASS_NAMES = {
    1: "1_chongkong", 2: "2_hanfeng", 3: "3_yueyawan", 4: "4_shuiban", 5: "5_youban",
    6: "6_siban", 7: "7_yiwu", 8: "8_yahen", 9: "9_zhehen", 10: "10_yaozhe",
}
RATIOS = np.asarray([0.8, 0.1, 0.1], dtype=np.float64)
SPLITS = np.asarray(["acquisition", "development", "final_locked"])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opaque_sample_id(folder: str, filename: str) -> str:
    digest = hashlib.sha256(f"gc10|{folder}|{filename}".encode("utf-8")).hexdigest()
    return "gc10_" + digest[:20]


def phash64(path: Path) -> int:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    dct = cv2.dct(gray)[:8, :8]
    threshold = float(np.median(dct.reshape(-1)[1:]))
    value = 0
    for bit in (dct >= threshold).reshape(-1):
        value = (value << 1) | int(bit)
    return value


def production_group(filename: str) -> str:
    stem = Path(filename).stem
    if "_" not in stem:
        raise ValueError(f"Cannot derive production group: {filename}")
    return stem.rsplit("_", 1)[0]


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            if a > b:
                a, b = b, a
            self.parent[b] = a


def parse_source(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    folders = sorted((path.name for path in source.iterdir() if path.is_dir() and path.name != "lable"), key=int)
    if folders != EXPECTED_FOLDERS:
        raise RuntimeError(f"Unexpected folders: {folders}")
    all_images = sorted(path for folder in folders for path in (source / folder).glob("*.jpg"))
    xml_paths = sorted((source / "lable").glob("*.xml"))
    if len(all_images) != EXPECTED_JPGS or len(xml_paths) != EXPECTED_XMLS:
        raise RuntimeError(f"Expected {EXPECTED_JPGS}/{EXPECTED_XMLS} JPG/XML, got {len(all_images)}/{len(xml_paths)}")

    image_by_basename: dict[str, list[Path]] = {}
    for path in all_images:
        image_by_basename.setdefault(path.name, []).append(path)
    canonical_paths_seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    boxes: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    ignored_objects: list[dict[str, str]] = []
    invalid_empty_paths: set[str] = set()
    for xml_path in xml_paths:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            errors.append({"item": str(xml_path), "error": "xml_parse_error", "detail": str(exc)})
            continue
        xml_folder = (root.findtext("folder") or "").strip()
        filename = (root.findtext("filename") or "").strip()
        candidates = image_by_basename.get(filename, [])
        if len(candidates) == 1:
            image_path = candidates[0]
        else:
            folder_matches = [path for path in candidates if path.parent.name == xml_folder]
            image_path = folder_matches[0] if len(folder_matches) == 1 else None
        if image_path is None:
            errors.append({"item": str(xml_path), "error": "canonical_image_ambiguous_or_missing", "detail": f"xml_folder={xml_folder},filename={filename},candidates={len(candidates)}"})
            continue
        folder = image_path.parent.name
        resolved_image = str(image_path.resolve())
        if resolved_image in canonical_paths_seen:
            errors.append({"item": str(xml_path), "error": "duplicate_xml_target", "detail": resolved_image})
            continue
        canonical_paths_seen.add(resolved_image)
        size = root.find("size")
        xml_width = int(size.findtext("width", "0")) if size is not None else 0
        xml_height = int(size.findtext("height", "0")) if size is not None else 0
        with Image.open(image_path) as image:
            width, height = image.size
        if (width, height) != (xml_width, xml_height):
            errors.append({"item": str(xml_path), "error": "xml_image_size_mismatch", "detail": f"xml={xml_width}x{xml_height},image={width}x{height}"})
            continue
        sample_id = opaque_sample_id(folder, filename)
        image_classes: set[int] = set()
        image_boxes: list[dict[str, Any]] = []
        objects = root.findall("object")
        if not objects:
            invalid_empty_paths.add(resolved_image)
            continue
        for object_index, obj in enumerate(objects):
            class_name_raw = (obj.findtext("name") or "").strip()
            match = re.match(r"^(10|[1-9])_(.+)$", class_name_raw)
            if match is None:
                ignored_objects.append({"xml_path": str(xml_path.resolve()), "object_index": str(object_index), "raw_name": class_name_raw, "reason": "undefined_class_name"})
                continue
            class_id = int(match.group(1))
            class_name = CANONICAL_CLASS_NAMES[class_id]
            bbox = obj.find("bndbox")
            if bbox is None:
                errors.append({"item": str(xml_path), "error": "missing_bbox", "detail": class_name})
                continue
            xmin = int(float(bbox.findtext("xmin", "nan")))
            ymin = int(float(bbox.findtext("ymin", "nan")))
            xmax = int(float(bbox.findtext("xmax", "nan")))
            ymax = int(float(bbox.findtext("ymax", "nan")))
            if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                errors.append({"item": str(xml_path), "error": "invalid_bbox", "detail": f"{xmin},{ymin},{xmax},{ymax} in {width}x{height}"})
                continue
            image_classes.add(class_id)
            image_boxes.append({
                "sample_id": sample_id,
                "object_index": object_index,
                "class_id": class_id,
                "class_name": class_name,
                "x_min": xmin,
                "y_min": ymin,
                "x_max": xmax,
                "y_max": ymax,
                "bbox_width": xmax - xmin,
                "bbox_height": ymax - ymin,
                "bbox_area_ratio": (xmax - xmin) * (ymax - ymin) / float(width * height),
            })
        if not image_boxes:
            invalid_empty_paths.add(resolved_image)
            continue
        image_hash = file_sha256(image_path)
        rows.append({
            "sample_id": sample_id,
            "source_folder_label": folder,
            "xml_folder_raw": xml_folder,
            "filename": filename,
            "image_path": str(image_path.resolve()),
            "xml_path": str(xml_path.resolve()),
            "production_group_raw": production_group(filename),
            "width": width,
            "height": height,
            "image_sha256": image_hash,
            "phash64": f"{phash64(image_path):016x}",
            "class_ids": "|".join(str(value) for value in sorted(image_classes)),
            "num_unique_classes": len(image_classes),
            "num_instances": len(image_boxes),
            "bbox_area_ratio_sum": float(sum(item["bbox_area_ratio"] for item in image_boxes)),
        })
        boxes.extend(image_boxes)

    manifest = pd.DataFrame(rows)
    box_frame = pd.DataFrame(boxes)
    canonical_paths = set(manifest["image_path"])
    excluded_rows = []
    xml_basenames = {path.stem for path in xml_paths}
    for path in all_images:
        resolved = str(path.resolve())
        if resolved in canonical_paths:
            continue
        if resolved in invalid_empty_paths:
            reason = "invalid_empty_annotation"
        elif path.stem in xml_basenames:
            reason = "noncanonical_same_basename"
        else:
            reason = "missing_xml"
        excluded_rows.append({
            "image_path": resolved,
            "folder": path.parent.name,
            "filename": path.name,
            "reason": reason,
            "image_sha256": file_sha256(path),
        })
    excluded = pd.DataFrame(excluded_rows)
    if errors:
        raise RuntimeError(f"GC10 audit found {len(errors)} annotation errors; first={errors[0]}")
    if len(manifest) != EXPECTED_XMLS - 2 or len(excluded) != EXPECTED_JPGS - (EXPECTED_XMLS - 2):
        raise RuntimeError(f"Expected 2292 canonical/20 excluded, got {len(manifest)}/{len(excluded)}")
    expected_exclusions = {"noncanonical_same_basename": 12, "missing_xml": 6, "invalid_empty_annotation": 2}
    if excluded["reason"].value_counts().to_dict() != expected_exclusions:
        raise RuntimeError(f"Unexpected exclusion reasons: {excluded['reason'].value_counts().to_dict()}")
    ignored = pd.DataFrame(ignored_objects)
    if len(ignored) != 1 or ignored.iloc[0]["raw_name"] != "d":
        raise RuntimeError(f"Unexpected ignored object annotations: {ignored.to_dict('records')}")
    return manifest, box_frame, excluded, ignored, errors


def add_hard_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    output = manifest.copy()
    ids = output["sample_id"].tolist()
    union = UnionFind(ids)
    for column in ["production_group_raw", "image_sha256"]:
        for _, sub in output.groupby(column, sort=False):
            members = sub["sample_id"].tolist()
            for member in members[1:]:
                union.union(members[0], member)
    roots = {sample_id: union.find(sample_id) for sample_id in ids}
    root_members: dict[str, list[str]] = {}
    for sample_id, root in roots.items():
        root_members.setdefault(root, []).append(sample_id)
    group_ids = {}
    for members in root_members.values():
        digest = hashlib.sha256("|".join(sorted(members)).encode("utf-8")).hexdigest()[:16]
        for sample_id in members:
            group_ids[sample_id] = "gc10_group_" + digest
    output["hard_group_id"] = output["sample_id"].map(group_ids)
    sizes = output.groupby("hard_group_id").size()
    output["hard_group_size"] = output["hard_group_id"].map(sizes).astype(int)
    return output


def choose_group_split(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    class_presence = manifest[["sample_id", "hard_group_id", "class_ids"]].copy()
    for class_id in range(1, 11):
        class_presence[f"class_{class_id}"] = class_presence["class_ids"].map(lambda value: class_id in {int(x) for x in str(value).split("|")})
    aggregations: dict[str, tuple[str, str]] = {"group_size": ("sample_id", "size")}
    aggregations.update({f"class_{class_id}": (f"class_{class_id}", "sum") for class_id in range(1, 11)})
    groups = class_presence.groupby("hard_group_id").agg(**aggregations).reset_index().sort_values("hard_group_id").reset_index(drop=True)
    feature_columns = ["group_size"] + [f"class_{class_id}" for class_id in range(1, 11)]
    matrix = groups[feature_columns].to_numpy(np.float64)
    totals = matrix.sum(axis=0)
    targets = RATIOS[:, None] * totals[None, :]
    weights = np.ones(len(feature_columns), dtype=np.float64)
    weights[0] = 3.0
    rng = np.random.default_rng(SPLIT_SEED)
    best_objective = np.inf
    best_assignment: np.ndarray | None = None
    candidates_tested = 5000
    for _ in range(candidates_tested):
        uniforms = rng.random(len(groups))
        assignment = np.where(uniforms < RATIOS[0], 0, np.where(uniforms < RATIOS[:2].sum(), 1, 2))
        counts = np.stack([matrix[assignment == split].sum(axis=0) for split in range(3)])
        if np.any(counts[:, 1:] == 0):
            continue
        relative_error = (counts - targets) / np.maximum(targets, 1.0)
        objective = float(np.sum(relative_error * relative_error * weights[None, :]))
        if objective < best_objective:
            best_objective = objective
            best_assignment = assignment.copy()
    if best_assignment is None:
        raise RuntimeError("Could not find a group split with every class represented")
    group_to_split = dict(zip(groups["hard_group_id"], SPLITS[best_assignment]))
    output = manifest.copy()
    output["protocol_split"] = output["hard_group_id"].map(group_to_split)
    diagnostics = {
        "candidate_assignments_tested": candidates_tested,
        "selected_objective": best_objective,
        "num_hard_groups": len(groups),
        "largest_hard_group": int(groups["group_size"].max()),
    }
    return output, diagnostics


def leakage_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left, right in itertools.combinations(SPLITS.tolist(), 2):
        a = manifest[manifest["protocol_split"].eq(left)]
        b = manifest[manifest["protocol_split"].eq(right)]
        rows.append({
            "split_left": left,
            "split_right": right,
            "path_overlap": len(set(a["image_path"]) & set(b["image_path"])),
            "sha256_overlap": len(set(a["image_sha256"]) & set(b["image_sha256"])),
            "production_group_overlap": len(set(a["production_group_raw"]) & set(b["production_group_raw"])),
            "hard_group_overlap": len(set(a["hard_group_id"]) & set(b["hard_group_id"])),
        })
    return pd.DataFrame(rows)


def class_distribution(manifest: pd.DataFrame, boxes: pd.DataFrame, split: str) -> pd.DataFrame:
    ids = set(manifest.loc[manifest["protocol_split"].eq(split), "sample_id"])
    sub = boxes[boxes["sample_id"].isin(ids)]
    image_counts = sub.drop_duplicates(["sample_id", "class_id"]).groupby(["class_id", "class_name"]).size().rename("images_with_class")
    instance_counts = sub.groupby(["class_id", "class_name"]).size().rename("instances")
    return pd.concat([image_counts, instance_counts], axis=1).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest, boxes, excluded, ignored_objects, errors = parse_source(source)
    manifest = add_hard_groups(manifest)
    manifest, split_diagnostics = choose_group_split(manifest)
    leakage = leakage_audit(manifest)
    leakage_columns = ["path_overlap", "sha256_overlap", "production_group_overlap", "hard_group_overlap"]
    if leakage[leakage_columns].to_numpy().sum() != 0:
        raise RuntimeError("Split leakage audit failed")
    acquisition = manifest[manifest["protocol_split"].eq("acquisition")].copy()
    development = manifest[manifest["protocol_split"].eq("development")].copy()
    final_locked = manifest[manifest["protocol_split"].eq("final_locked")].copy()

    manifest.to_csv(out / "gc10_source_audit.csv", index=False, encoding="utf-8-sig")
    boxes.to_csv(out / "gc10_bbox_audit.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(out / "gc10_excluded_images.csv", index=False, encoding="utf-8-sig")
    ignored_objects.to_csv(out / "gc10_ignored_object_annotations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors, columns=["item", "error", "detail"]).to_csv(out / "gc10_source_errors.csv", index=False, encoding="utf-8-sig")
    acquisition.to_csv(out / "gc10_acquisition_pool_gt_audit.csv", index=False, encoding="utf-8-sig")
    acquisition[["sample_id", "image_path"]].to_csv(out / "gc10_acquisition_loader_private.csv", index=False, encoding="utf-8-sig")
    acquisition[["sample_id", "image_sha256", "phash64"]].to_csv(out / "gc10_acquisition_pool_blind.csv", index=False, encoding="utf-8-sig")
    boxes[boxes["sample_id"].isin(set(acquisition["sample_id"]))].to_csv(out / "gc10_acquisition_bbox_gt_audit.csv", index=False, encoding="utf-8-sig")
    development.to_csv(out / "gc10_development_eval.csv", index=False, encoding="utf-8-sig")
    boxes[boxes["sample_id"].isin(set(development["sample_id"]))].to_csv(out / "gc10_development_bbox_gt.csv", index=False, encoding="utf-8-sig")
    final_locked.to_csv(out / "gc10_final_test_locked.csv", index=False, encoding="utf-8-sig")
    leakage.to_csv(out / "gc10_split_leakage_audit.csv", index=False, encoding="utf-8-sig")

    acquisition_classes = class_distribution(manifest, boxes, "acquisition")
    development_classes = class_distribution(manifest, boxes, "development")
    acquisition_classes.to_csv(out / "gc10_acquisition_class_distribution.csv", index=False, encoding="utf-8-sig")
    development_classes.to_csv(out / "gc10_development_class_distribution.csv", index=False, encoding="utf-8-sig")
    final_path = out / "gc10_final_test_locked.csv"
    config = {
        "protocol": "GC10-DET All-Defect Cold-Start Taxonomy Triage",
        "created_date": "2026-07-15",
        "source_root": str(source),
        "source_jpg_files": EXPECTED_JPGS,
        "source_xml_files": EXPECTED_XMLS,
        "canonical_labeled_images": len(manifest),
        "excluded_images": len(excluded),
        "split_seed": SPLIT_SEED,
        "split_ratios": {"acquisition": 0.8, "development": 0.1, "final_locked": 0.1},
        "hard_split_grouping": ["filename_production_group", "exact_sha256"],
        "phash_used_for_split": False,
        "phash_role": "audit_only",
        "selector_visible_columns": ["sample_id", "image_sha256", "phash64"],
        "source_paths_visible_to_selector": False,
        "folder_labels_visible_to_selector": False,
        "split_sizes": manifest["protocol_split"].value_counts().to_dict(),
        "split_diagnostics": split_diagnostics,
        "final_locked_manifest": str(final_path),
        "final_locked_manifest_sha256": file_sha256(final_path),
        "detector_training_performed": False,
        "selection_performed": False,
        "final_test_evaluated": False,
    }
    (out / "gc10_protocol_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    source_class_counts = boxes.drop_duplicates(["sample_id", "class_id"]).groupby(["class_id", "class_name"]).size().reset_index(name="images_with_class")
    source_instance_counts = boxes.groupby(["class_id", "class_name"]).size().reset_index(name="instances")
    source_classes = source_class_counts.merge(source_instance_counts, on=["class_id", "class_name"], validate="one_to_one")
    report = [
        "# GC10-DET Ingestion and Split Audit", "",
        "- Detector training performed: **False**",
        "- Selection performed: **False**",
        "- Final test evaluated: **False**",
        f"- Source JPG/XML files: **{EXPECTED_JPGS}/{EXPECTED_XMLS}**",
        f"- Canonical labeled images: **{len(manifest)}**",
        f"- Excluded images: **{len(excluded)}** (12 noncanonical same-basename copies; 6 missing XML; 2 empty XML annotations)",
        f"- Ignored undefined object annotations: **{len(ignored_objects)}** (`d`; valid annotation in the same image retained)",
        f"- Valid bounding boxes: **{len(boxes)}**",
        f"- Exact SHA duplicate groups among canonical images: **{int((manifest.groupby('image_sha256').size() > 1).sum())}**",
        f"- Production/hard groups: **{manifest['production_group_raw'].nunique()}/{manifest['hard_group_id'].nunique()}**",
        f"- Largest hard group: **{split_diagnostics['largest_hard_group']}**",
        f"- Split sizes: **acquisition {len(acquisition)}, development {len(development)}, final locked {len(final_locked)}**",
        f"- Final locked manifest SHA-256: `{config['final_locked_manifest_sha256']}`", "",
        "## Full-source annotation distribution (pre-split audit)", "", source_classes.to_markdown(index=False), "",
        "## Acquisition class distribution", "", acquisition_classes.to_markdown(index=False), "",
        "## Development class distribution", "", development_classes.to_markdown(index=False), "",
        "Final split annotation distribution is intentionally not printed or loaded by downstream selection code.", "",
        "## Exclusion reasons", "", excluded.groupby("reason").size().reset_index(name="num_images").to_markdown(index=False), "",
        "## Leakage audit", "", leakage.to_markdown(index=False), "",
    ]
    summary = out / "gc10_ingestion_audit_summary.md"
    summary.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("=" * 100)
    print("[DONE] GC10-DET ingestion/split protocol")
    print("Detector training performed: False")
    print("Selection performed: False")
    print("Final test evaluated: False")
    print(f"[SUMMARY] {summary}")
    print("=" * 100)


if __name__ == "__main__":
    main()

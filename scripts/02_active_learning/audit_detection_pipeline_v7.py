"""
Phase-0 methodology audit for the YOLO active-learning pipeline.

This script is intentionally diagnostic-only.  It does not train YOLO and it
does not modify any existing V6 results.  Its job is to separate low-mAP causes
before we spend more GPU time on new acquisition strategies:

1. XML -> YOLO conversion sanity
2. bbox/class mapping validity
3. pool/evaluation leakage and near-duplicate checks
4. pool/initial/evaluation class composition
5. completed YOLO run convergence audit

Output:
    runs/methodology_audit_v7/audit_YYYYMMDD_HHMMSS/
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402


PROJECT_ROOT = v6.PROJECT_ROOT
DATA_ROOT = v6.DATA_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "methodology_audit_v7"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_int_list_env(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if not value:
        return default
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def resolve_priority_csv() -> Path:
    override = os.environ.get("AL_PRIORITY_CSV")
    candidates: list[Path] = []
    if override:
        p = Path(override).expanduser()
        candidates.extend([p, PROJECT_ROOT / p])

    candidates.append(
        PROJECT_ROOT
        / "outputs"
        / "priority_sensitivity_20260706_152020"
        / "penalty_0"
        / "priority_scores_pseudo.csv"
    )

    candidates.extend((PROJECT_ROOT / "outputs").glob("pseudo_boxes_*/priority_scores_pseudo.csv"))

    existing = [p.resolve() for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(
            "Could not locate priority_scores_pseudo.csv. "
            "Set AL_PRIORITY_CSV explicitly."
        )
    return max(existing, key=lambda p: p.stat().st_mtime)


def load_priority_scores() -> tuple[Path, pd.DataFrame]:
    priority_csv = resolve_priority_csv()
    df = pd.read_csv(priority_csv)
    df = v6.prepare_priority_dataframe(df)
    return priority_csv, df


def stable_sample_order(df: pd.DataFrame) -> pd.DataFrame:
    return v6.stable_sample_order(df).reset_index(drop=True)


def build_image_index() -> dict[tuple[str, str], list[Path]]:
    index: dict[tuple[str, str], list[Path]] = {}
    for dataset_type, root in [
        ("NEU-DET", DATA_ROOT / "NEU-DET"),
        ("GC10-DET", DATA_ROOT / "GC10-DET"),
    ]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault((dataset_type, path.name), []).append(path.resolve())
    return index


def build_xml_index() -> dict[tuple[str, str], list[Path]]:
    index: dict[tuple[str, str], list[Path]] = {}
    for dataset_type, root in [
        ("NEU-DET", DATA_ROOT / "NEU-DET"),
        ("GC10-DET", DATA_ROOT / "GC10-DET"),
    ]:
        if not root.exists():
            continue
        for path in root.rglob("*.xml"):
            index.setdefault((dataset_type, path.stem), []).append(path.resolve())
    return index


def resolve_image_path_fast(
    row: pd.Series,
    image_index: dict[tuple[str, str], list[Path]],
) -> Path | None:
    raw = Path(str(row.get("image_path", "")))
    if raw.exists():
        return raw.resolve()
    dataset_type = str(row.get("dataset_type", ""))
    image_name = str(row.get("image_name", ""))
    matches = image_index.get((dataset_type, image_name), [])
    return matches[0] if matches else None


def find_xml_path_fast(
    row: pd.Series,
    image_path: Path | None,
    xml_index: dict[tuple[str, str], list[Path]],
) -> Path | None:
    dataset_type = str(row.get("dataset_type", ""))
    image_name = str(row.get("image_name", ""))
    stem = Path(image_name).stem
    matches = xml_index.get((dataset_type, stem), [])
    if matches:
        if image_path is not None:
            same_parent = [p for p in matches if p.parent == image_path.parent]
            if same_parent:
                return same_parent[0]
        return matches[0]
    if image_path is not None:
        sidecar = image_path.with_suffix(".xml")
        if sidecar.exists():
            return sidecar.resolve()
    return None


def build_fixed_external_eval_df_fast(
    pool_df: pd.DataFrame,
    image_index: dict[tuple[str, str], list[Path]],
    xml_index: dict[tuple[str, str], list[Path]],
) -> pd.DataFrame:
    """Fast equivalent of the V6 fixed external eval builder.

    The V6 helper is correct but can be slow on GC10 because it repeatedly
    performs recursive glob lookups.  For audit we build path indexes once.
    """

    fixed_eval_size = int(os.environ.get("AL_FIXED_EVAL_SIZE", "180"))
    fixed_eval_seed = int(os.environ.get("AL_FIXED_EVAL_SEED", "20260709"))
    eval_match_pool_classes = parse_bool_env("AL_EVAL_MATCH_POOL_CLASSES", True)
    min_pool_class_count = int(os.environ.get("AL_MIN_POOL_CLASS_COUNT_FOR_EVAL", "3"))

    pool_keys = set(zip(pool_df["dataset_type"].astype(str), pool_df["image_name"].astype(str)))
    rows: list[dict[str, str]] = []

    for (dataset_type, image_name), paths in sorted(image_index.items()):
        if (dataset_type, image_name) in pool_keys:
            continue
        image_path = paths[0]
        row = {
            "image_name": image_name,
            "dataset_type": dataset_type,
            "image_path": str(image_path),
        }
        if find_xml_path_fast(pd.Series(row), image_path, xml_index) is None:
            continue
        row["class_hint"] = v6.infer_class_hint(pd.Series(row))
        rows.append(row)

    candidates = stable_sample_order(pd.DataFrame(rows))
    if candidates.empty:
        raise ValueError("No external evaluation candidates found outside the AL pool.")

    if eval_match_pool_classes:
        supported_dist = v6.class_distribution(pool_df)
        supported_dist = supported_dist[supported_dist["count"] >= min_pool_class_count]
        supported = set(
            zip(
                supported_dist["dataset_type"].astype(str),
                supported_dist["class_hint"].astype(str),
            )
        )
        mask = [
            key in supported
            for key in zip(
                candidates["dataset_type"].astype(str),
                candidates["class_hint"].astype(str),
            )
        ]
        candidates = candidates.loc[mask].copy()
        if candidates.empty:
            raise ValueError("No eval candidates remain after matching eval classes to the AL pool.")

    target = min(fixed_eval_size, len(candidates))
    grouped = list(candidates.groupby(["dataset_type", "class_hint"], sort=True, dropna=False))
    base = target // max(1, len(grouped))
    remainder = target % max(1, len(grouped))
    selected_parts = []

    for group_idx, (_, sub) in enumerate(grouped):
        quota = min(base + (1 if group_idx < remainder else 0), len(sub))
        if quota:
            selected_parts.append(
                stable_sample_order(sub).sample(
                    n=quota,
                    random_state=fixed_eval_seed + group_idx * 101,
                )
            )

    selected = pd.concat(selected_parts) if selected_parts else candidates.iloc[0:0].copy()
    if len(selected) < target:
        remaining = candidates.drop(index=selected.index, errors="ignore")
        selected = pd.concat(
            [
                selected,
                stable_sample_order(remaining).sample(
                    n=min(target - len(selected), len(remaining)),
                    random_state=fixed_eval_seed + 999,
                ),
            ]
        )

    selected = stable_sample_order(selected.head(target))
    overlap = set(zip(selected["dataset_type"], selected["image_name"])) & pool_keys
    if overlap:
        raise AssertionError(f"Evaluation/pool leakage detected: {sorted(overlap)[:5]}")
    return selected


@dataclass
class BoxAudit:
    conversion_rows: list[dict]
    suspicious_rows: list[dict]


def _read_image_size(image_path: Path | None) -> tuple[int | None, int | None]:
    if image_path is None or not image_path.exists():
        return None, None
    with Image.open(image_path) as img:
        return img.size


def _safe_float(node) -> float | None:
    try:
        return float(node.text)
    except Exception:
        return None


def audit_xml_to_yolo_for_manifest(
    manifest_df: pd.DataFrame,
    split_name: str,
    image_index: dict[tuple[str, str], list[Path]],
    xml_index: dict[tuple[str, str], list[Path]],
) -> BoxAudit:
    conversion_rows: list[dict] = []
    suspicious_rows: list[dict] = []

    for _, row in manifest_df.iterrows():
        image_path = resolve_image_path_fast(row, image_index)
        xml_path = find_xml_path_fast(row, image_path, xml_index)
        image_width, image_height = _read_image_size(image_path)
        base = {
            "split": split_name,
            "dataset_type": row.get("dataset_type"),
            "image_name": row.get("image_name"),
            "class_hint": row.get("class_hint", "unknown"),
            "image_path": str(image_path) if image_path else None,
            "xml_path": str(xml_path) if xml_path else None,
            "image_width": image_width,
            "image_height": image_height,
            "xml_found": xml_path is not None,
        }

        if image_path is None:
            bad = {**base, "conversion_status": "missing_image", "suspicious_reason": "missing_image"}
            conversion_rows.append(bad)
            suspicious_rows.append(bad)
            continue

        if xml_path is None:
            bad = {**base, "conversion_status": "missing_xml", "suspicious_reason": "missing_xml"}
            conversion_rows.append(bad)
            suspicious_rows.append(bad)
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as exc:
            bad = {
                **base,
                "conversion_status": "xml_parse_error",
                "suspicious_reason": f"xml_parse_error:{exc}",
            }
            conversion_rows.append(bad)
            suspicious_rows.append(bad)
            continue

        try:
            xml_width, xml_height = v6.get_image_size_from_xml_or_image(root, image_path)
        except Exception:
            xml_width, xml_height = image_width, image_height

        objects = list(root.findall("object"))
        if not objects:
            bad = {**base, "conversion_status": "no_objects", "suspicious_reason": "no_objects"}
            conversion_rows.append(bad)
            suspicious_rows.append(bad)
            continue

        for object_idx, obj in enumerate(objects):
            name_node = obj.find("name")
            raw_label = name_node.text if name_node is not None else None
            mapped_class = v6.map_class_name(raw_label, str(row.get("dataset_type")), image_path)
            class_id = v6.CLASS_MAP.get(mapped_class) if mapped_class in v6.CLASS_MAP else None
            bndbox = obj.find("bndbox")

            object_base = {
                **base,
                "object_index": object_idx,
                "raw_label": raw_label,
                "mapped_class": mapped_class,
                "class_id": class_id,
                "xml_width": xml_width,
                "xml_height": xml_height,
            }

            if mapped_class not in v6.CLASS_MAP:
                bad = {
                    **object_base,
                    "conversion_status": "unknown_class",
                    "suspicious_reason": "unknown_class",
                }
                conversion_rows.append(bad)
                suspicious_rows.append(bad)
                continue

            if bndbox is None:
                bad = {
                    **object_base,
                    "conversion_status": "missing_bndbox",
                    "suspicious_reason": "missing_bndbox",
                }
                conversion_rows.append(bad)
                suspicious_rows.append(bad)
                continue

            xmin = _safe_float(bndbox.find("xmin"))
            ymin = _safe_float(bndbox.find("ymin"))
            xmax = _safe_float(bndbox.find("xmax"))
            ymax = _safe_float(bndbox.find("ymax"))
            coords = [xmin, ymin, xmax, ymax]
            coord_payload = {
                "raw_xmin": xmin,
                "raw_ymin": ymin,
                "raw_xmax": xmax,
                "raw_ymax": ymax,
            }

            if any(v is None for v in coords) or xml_width is None or xml_height is None or xml_width <= 0 or xml_height <= 0:
                bad = {
                    **object_base,
                    **coord_payload,
                    "conversion_status": "invalid_numeric_box",
                    "suspicious_reason": "invalid_numeric_box",
                }
                conversion_rows.append(bad)
                suspicious_rows.append(bad)
                continue

            assert xmin is not None and ymin is not None and xmax is not None and ymax is not None
            invalid_before_clip = xmax <= xmin or ymax <= ymin
            clipped_xmin = max(0.0, min(xmin, xml_width - 1))
            clipped_ymin = max(0.0, min(ymin, xml_height - 1))
            clipped_xmax = max(0.0, min(xmax, xml_width - 1))
            clipped_ymax = max(0.0, min(ymax, xml_height - 1))
            clipped = any(
                abs(a - b) > 1e-9
                for a, b in [
                    (xmin, clipped_xmin),
                    (ymin, clipped_ymin),
                    (xmax, clipped_xmax),
                    (ymax, clipped_ymax),
                ]
            )
            invalid_after_clip = clipped_xmax <= clipped_xmin or clipped_ymax <= clipped_ymin

            box_w_px = clipped_xmax - clipped_xmin
            box_h_px = clipped_ymax - clipped_ymin
            x_center = ((clipped_xmin + clipped_xmax) / 2.0) / xml_width
            y_center = ((clipped_ymin + clipped_ymax) / 2.0) / xml_height
            yolo_w = box_w_px / xml_width
            yolo_h = box_h_px / xml_height
            yolo_area = yolo_w * yolo_h
            yolo_in_unit_range = all(0.0 <= v <= 1.0 for v in [x_center, y_center, yolo_w, yolo_h])

            reasons = []
            if invalid_before_clip:
                reasons.append("invalid_before_clip")
            if invalid_after_clip:
                reasons.append("invalid_after_clip")
            if clipped:
                reasons.append("box_clipped")
            if not yolo_in_unit_range:
                reasons.append("yolo_out_of_range")
            if 0.0 < yolo_area < 1e-5:
                reasons.append("very_tiny_box")
            if yolo_area > 0.95:
                reasons.append("very_large_box")
            if box_w_px <= 1.0 or box_h_px <= 1.0:
                reasons.append("width_or_height_leq_1px")

            conversion_status = "converted"
            if invalid_after_clip:
                conversion_status = "dropped_invalid_after_clip"
            elif invalid_before_clip:
                conversion_status = "dropped_invalid_before_clip"

            out = {
                **object_base,
                **coord_payload,
                "clipped_xmin": clipped_xmin,
                "clipped_ymin": clipped_ymin,
                "clipped_xmax": clipped_xmax,
                "clipped_ymax": clipped_ymax,
                "yolo_x_center": x_center,
                "yolo_y_center": y_center,
                "yolo_width": yolo_w,
                "yolo_height": yolo_h,
                "yolo_area": yolo_area,
                "invalid_before_clip": invalid_before_clip,
                "invalid_after_clip": invalid_after_clip,
                "clipped": clipped,
                "yolo_in_unit_range": yolo_in_unit_range,
                "class_hint_matches_mapped_class": str(row.get("class_hint", "unknown")) == str(mapped_class),
                "conversion_status": conversion_status,
                "suspicious_reason": "|".join(reasons),
            }
            conversion_rows.append(out)
            if reasons:
                suspicious_rows.append(out)

    return BoxAudit(conversion_rows=conversion_rows, suspicious_rows=suspicious_rows)


def compute_file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def average_hash(path: Path | None, hash_size: int = 8) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((hash_size, hash_size))
            arr = np.asarray(img, dtype=np.float32)
        bits = arr >= arr.mean()
        value = 0
        for bit in bits.flatten():
            value = (value << 1) | int(bit)
        return f"{value:0{hash_size * hash_size // 4}x}"
    except Exception:
        return None


def hamming_hex(a: str | None, b: str | None) -> int | None:
    if a is None or b is None:
        return None
    return int(int(a, 16) ^ int(b, 16)).bit_count()


def make_overlap_reports(
    pool_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    image_index: dict[tuple[str, str], list[Path]],
    max_hamming: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = pool_df.copy()
    eval_ = eval_df.copy()
    pool["resolved_image_path"] = [
        str(resolve_image_path_fast(row, image_index)) for _, row in pool.iterrows()
    ]
    eval_["resolved_image_path"] = [
        str(resolve_image_path_fast(row, image_index)) for _, row in eval_.iterrows()
    ]

    exact_rows = []
    eval_by_key = {
        (str(row["dataset_type"]), str(row["image_name"])): row
        for _, row in eval_.iterrows()
    }
    eval_by_path = {str(row["resolved_image_path"]): row for _, row in eval_.iterrows()}

    for _, p_row in pool.iterrows():
        key = (str(p_row["dataset_type"]), str(p_row["image_name"]))
        match = eval_by_key.get(key)
        if match is not None:
            exact_rows.append(
                {
                    "overlap_type": "dataset_type_image_name",
                    "pool_dataset_type": p_row["dataset_type"],
                    "pool_image_name": p_row["image_name"],
                    "pool_image_path": p_row["resolved_image_path"],
                    "eval_dataset_type": match["dataset_type"],
                    "eval_image_name": match["image_name"],
                    "eval_image_path": match["resolved_image_path"],
                }
            )
        match = eval_by_path.get(str(p_row["resolved_image_path"]))
        if match is not None:
            exact_rows.append(
                {
                    "overlap_type": "resolved_file_path",
                    "pool_dataset_type": p_row["dataset_type"],
                    "pool_image_name": p_row["image_name"],
                    "pool_image_path": p_row["resolved_image_path"],
                    "eval_dataset_type": match["dataset_type"],
                    "eval_image_name": match["image_name"],
                    "eval_image_path": match["resolved_image_path"],
                }
            )

    exact_columns = [
        "overlap_type",
        "pool_dataset_type",
        "pool_image_name",
        "pool_image_path",
        "eval_dataset_type",
        "eval_image_name",
        "eval_image_path",
    ]
    exact_df = pd.DataFrame(exact_rows, columns=exact_columns)

    phash_pool = []
    for _, row in pool.iterrows():
        path = Path(str(row["resolved_image_path"]))
        phash_pool.append(
            {
                "dataset_type": row["dataset_type"],
                "image_name": row["image_name"],
                "image_path": str(path),
                "sha256": compute_file_sha256(path),
                "ahash": average_hash(path),
            }
        )
    phash_eval = []
    for _, row in eval_.iterrows():
        path = Path(str(row["resolved_image_path"]))
        phash_eval.append(
            {
                "dataset_type": row["dataset_type"],
                "image_name": row["image_name"],
                "image_path": str(path),
                "sha256": compute_file_sha256(path),
                "ahash": average_hash(path),
            }
        )

    near_columns = [
        "pool_dataset_type",
        "pool_image_name",
        "pool_image_path",
        "eval_dataset_type",
        "eval_image_name",
        "eval_image_path",
        "same_sha256",
        "ahash_hamming_distance",
    ]
    near_rows = []
    for p in phash_pool:
        for e in phash_eval:
            exact_hash = p["sha256"] is not None and p["sha256"] == e["sha256"]
            hd = hamming_hex(p["ahash"], e["ahash"])
            if exact_hash or (hd is not None and hd <= max_hamming):
                near_rows.append(
                    {
                        "pool_dataset_type": p["dataset_type"],
                        "pool_image_name": p["image_name"],
                        "pool_image_path": p["image_path"],
                        "eval_dataset_type": e["dataset_type"],
                        "eval_image_name": e["image_name"],
                        "eval_image_path": e["image_path"],
                        "same_sha256": exact_hash,
                        "ahash_hamming_distance": hd,
                    }
                )
    return exact_df, pd.DataFrame(near_rows, columns=near_columns)


def make_class_hint_mismatch_report(conversion_df: pd.DataFrame) -> pd.DataFrame:
    """Find images where the image-level class_hint is absent from XML labels.

    This is not automatically a label-conversion bug because an image can contain
    multiple defects.  It *is* important for this project because class_hint is
    used for diagnostics, pool/eval matching, and oracle balancing controls.
    """

    if conversion_df.empty or "mapped_class" not in conversion_df.columns:
        return pd.DataFrame(
            columns=[
                "split",
                "dataset_type",
                "image_name",
                "class_hint",
                "mapped_classes_in_xml",
                "num_xml_instances",
                "class_hint_present_in_xml",
            ]
        )

    valid = conversion_df[conversion_df["conversion_status"].eq("converted")].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "dataset_type",
                "image_name",
                "class_hint",
                "mapped_classes_in_xml",
                "num_xml_instances",
                "class_hint_present_in_xml",
            ]
        )

    grouped = (
        valid.groupby(["split", "dataset_type", "image_name", "class_hint"], dropna=False)
        .agg(
            mapped_classes_in_xml=("mapped_class", lambda s: "|".join(sorted(set(map(str, s))))),
            num_xml_instances=("mapped_class", "size"),
        )
        .reset_index()
    )
    grouped["class_hint_present_in_xml"] = grouped.apply(
        lambda row: str(row["class_hint"]) in str(row["mapped_classes_in_xml"]).split("|"),
        axis=1,
    )
    return grouped[~grouped["class_hint_present_in_xml"]].reset_index(drop=True)


def make_dataset_distribution(
    manifests: dict[str, pd.DataFrame],
    conversion_df: pd.DataFrame,
) -> pd.DataFrame:
    image_rows = []
    for split, df in manifests.items():
        if df.empty:
            continue
        grouped = (
            df.groupby(["dataset_type", "class_hint"], dropna=False)
            .size()
            .reset_index(name="image_count")
        )
        grouped.insert(0, "split", split)
        image_rows.append(grouped)

    image_dist = pd.concat(image_rows, ignore_index=True) if image_rows else pd.DataFrame()
    valid = conversion_df[conversion_df["conversion_status"].eq("converted")].copy()
    if len(valid):
        instance_dist = (
            valid.groupby(["split", "dataset_type", "mapped_class"], dropna=False)
            .size()
            .reset_index(name="bbox_instance_count")
            .rename(columns={"mapped_class": "class_hint"})
        )
    else:
        instance_dist = pd.DataFrame(
            columns=["split", "dataset_type", "class_hint", "bbox_instance_count"]
        )

    out = image_dist.merge(
        instance_dist,
        on=["split", "dataset_type", "class_hint"],
        how="outer",
    )
    out["image_count"] = out["image_count"].fillna(0).astype(int)
    out["bbox_instance_count"] = out["bbox_instance_count"].fillna(0).astype(int)
    return out.sort_values(["split", "dataset_type", "class_hint"], kind="mergesort")


def latest_run_dir(root: Path, prefix: str) -> Path | None:
    if not root.exists():
        return None
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def _find_metric_column(columns: Iterable[str], contains: str, excludes: tuple[str, ...] = ()) -> str | None:
    for col in columns:
        normalized = col.strip()
        if contains in normalized and not any(ex in normalized for ex in excludes):
            return col
    return None


def audit_training_convergence() -> pd.DataFrame:
    override = os.environ.get("V6_RUN_DIR")
    if override:
        run_dir = Path(override).expanduser()
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
    else:
        run_dir = latest_run_dir(
            PROJECT_ROOT / "runs" / "active_learning_ablation_v6_deficit_diversity",
            "al_ablation_v6_deficit_diversity_",
        )

    if run_dir is None or not run_dir.exists():
        return pd.DataFrame(
            [{"status": "skipped", "reason": "No V6 run directory found", "v6_run_dir": None}]
        )

    results_path = run_dir / "all_round_results.csv"
    if not results_path.exists():
        return pd.DataFrame(
            [{"status": "skipped", "reason": "all_round_results.csv not found", "v6_run_dir": str(run_dir)}]
        )

    results_df = pd.read_csv(results_path)
    rows = []
    for _, result in results_df.iterrows():
        train_run_dir = result.get("train_run_dir")
        train_run_path = Path(str(train_run_dir)) if pd.notna(train_run_dir) else None
        row = {
            "status": "checked",
            "v6_run_dir": str(run_dir),
            "seed": result.get("seed"),
            "strategy": result.get("strategy"),
            "round": result.get("round"),
            "labeled_budget": result.get("labeled_budget"),
            "reported_map50": result.get("map50"),
            "reported_map5095": result.get("map5095"),
            "train_status": result.get("train_status"),
            "train_run_dir": str(train_run_path) if train_run_path else None,
        }
        if train_run_path is None or not train_run_path.exists():
            row.update({"results_csv_found": False, "args_yaml_found": False})
            rows.append(row)
            continue

        args_path = train_run_path / "args.yaml"
        results_csv = train_run_path / "results.csv"
        row["args_yaml_found"] = args_path.exists()
        row["results_csv_found"] = results_csv.exists()
        configured_epochs = np.nan
        patience = np.nan
        if args_path.exists():
            try:
                import yaml

                args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
                configured_epochs = args.get("epochs", np.nan)
                patience = args.get("patience", np.nan)
                row["imgsz"] = args.get("imgsz")
                row["batch"] = args.get("batch")
                row["workers"] = args.get("workers")
                row["cache"] = args.get("cache")
            except Exception as exc:
                row["args_read_error"] = str(exc)

        row["configured_epochs"] = configured_epochs
        row["patience"] = patience

        if results_csv.exists():
            try:
                train_df = pd.read_csv(results_csv)
                train_df.columns = [c.strip() for c in train_df.columns]
                epoch_col = "epoch" if "epoch" in train_df.columns else None
                map50_col = _find_metric_column(train_df.columns, "metrics/mAP50(B)", ("50-95",))
                map5095_col = _find_metric_column(train_df.columns, "metrics/mAP50-95(B)")
                actual_epochs = len(train_df)
                row["actual_epochs"] = actual_epochs
                row["early_stopped_before_configured_epochs"] = (
                    bool(actual_epochs < int(configured_epochs))
                    if pd.notna(configured_epochs)
                    else np.nan
                )
                if map50_col:
                    best_idx = train_df[map50_col].idxmax()
                    row["best_train_results_map50"] = float(train_df.loc[best_idx, map50_col])
                    row["best_epoch_by_results_map50"] = int(train_df.loc[best_idx, epoch_col]) if epoch_col else int(best_idx)
                    row["last_train_results_map50"] = float(train_df[map50_col].iloc[-1])
                if map5095_col:
                    best_idx = train_df[map5095_col].idxmax()
                    row["best_train_results_map5095"] = float(train_df.loc[best_idx, map5095_col])
                    row["best_epoch_by_results_map5095"] = int(train_df.loc[best_idx, epoch_col]) if epoch_col else int(best_idx)
                    row["last_train_results_map5095"] = float(train_df[map5095_col].iloc[-1])
            except Exception as exc:
                row["results_read_error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(
    save_dir: Path,
    config: dict,
    distribution_df: pd.DataFrame,
    conversion_df: pd.DataFrame,
    suspicious_df: pd.DataFrame,
    mismatch_df: pd.DataFrame,
    exact_overlap_df: pd.DataFrame,
    phash_overlap_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
) -> None:
    missing_xml = int(conversion_df["conversion_status"].eq("missing_xml").sum()) if len(conversion_df) else 0
    unknown_class = int(conversion_df["conversion_status"].eq("unknown_class").sum()) if len(conversion_df) else 0
    invalid_after_clip = int(conversion_df["conversion_status"].eq("dropped_invalid_after_clip").sum()) if len(conversion_df) else 0
    severe = missing_xml + unknown_class + invalid_after_clip + len(exact_overlap_df)
    pool_dist = distribution_df[distribution_df["split"].eq("priority_pool")]

    early_stopped_count = 0
    if "early_stopped_before_configured_epochs" in convergence_df.columns:
        early_stopped_count = int(convergence_df["early_stopped_before_configured_epochs"].fillna(False).sum())

    lines = [
        "# Methodology Audit V7",
        "",
        "## Verdict",
        "",
        f"- Severe issue count: {severe}",
        f"- Missing XML object/image rows: {missing_xml}",
        f"- Unknown class rows: {unknown_class}",
        f"- Invalid-after-clip boxes: {invalid_after_clip}",
        f"- Image class_hint absent from XML instance classes: {len(mismatch_df)}",
        f"- Exact pool/eval overlaps: {len(exact_overlap_df)}",
        f"- Perceptual near-overlaps: {len(phash_overlap_df)}",
        f"- Early-stopped YOLO runs found in latest V6 audit target: {early_stopped_count}",
        "",
    ]

    if severe > 0:
        lines.extend(
            [
                "Status: **STOP AND FIX BEFORE NEW AL EXPERIMENTS**",
                "",
                "At least one pipeline-level issue was found that can make mAP interpretation unreliable.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Status: **PIPELINE AUDIT PASSED WITH WARNINGS CHECKED BELOW**",
                "",
                "No severe XML/class/leakage issue was found by this audit. Low mAP should next be separated with full-data upper-bound and training-variance experiments.",
                "",
            ]
        )

    lines.extend(
        [
            "## Pool class composition",
            "",
            pool_dist.to_markdown(index=False) if len(pool_dist) else "_No pool distribution rows._",
            "",
            "## class_hint vs XML instance-class mismatch",
            "",
            mismatch_df.head(30).to_markdown(index=False) if len(mismatch_df) else "_No mismatches found._",
            "",
            "## Config",
            "",
            "```json",
            json.dumps(config, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Recommended next commands",
            "",
            "```powershell",
            r".\.python311\python.exe .\scripts\02_active_learning\run_yolo_upper_bound_v7.py",
            r".\.python311\python.exe .\scripts\02_active_learning\run_training_variance_v7.py",
            "```",
            "",
            "The two commands above default to dry-run mode. Set `AL_DRY_RUN_ONLY=0` only after reading this audit summary.",
            "",
        ]
    )
    (save_dir / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"audit_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    priority_csv, pool_df = load_priority_scores()
    seeds = parse_int_list_env("AL_SEEDS", [42])
    initial_seed_size = int(os.environ.get("AL_INITIAL_SEED_SIZE", "15"))
    max_phash_hamming = int(os.environ.get("AL_PHASH_MAX_HAMMING", "2"))

    print("=" * 100)
    print("[PHASE 0] Methodology audit V7")
    print(f"Priority CSV: {priority_csv}")
    print(f"Output dir  : {save_dir}")
    print("=" * 100)

    image_index = build_image_index()
    xml_index = build_xml_index()
    fixed_eval_df = build_fixed_external_eval_df_fast(pool_df, image_index, xml_index)

    initial_seed_df = stable_sample_order(pool_df).sample(
        n=min(initial_seed_size, len(pool_df)),
        random_state=seeds[0] + 999,
    )

    manifests = {
        "priority_pool": pool_df,
        f"initial_seed_{seeds[0]}": initial_seed_df,
        "fixed_external_eval": fixed_eval_df,
    }

    audit_parts = []
    suspicious_parts = []
    for split, manifest in manifests.items():
        audit = audit_xml_to_yolo_for_manifest(manifest, split, image_index, xml_index)
        audit_parts.append(pd.DataFrame(audit.conversion_rows))
        suspicious_parts.append(pd.DataFrame(audit.suspicious_rows))

    conversion_df = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()
    suspicious_df = pd.concat(suspicious_parts, ignore_index=True) if suspicious_parts else pd.DataFrame()
    distribution_df = make_dataset_distribution(manifests, conversion_df)
    exact_overlap_df, phash_overlap_df = make_overlap_reports(
        pool_df,
        fixed_eval_df,
        image_index,
        max_hamming=max_phash_hamming,
    )
    mismatch_df = make_class_hint_mismatch_report(conversion_df)
    convergence_df = audit_training_convergence()

    config = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "priority_csv": str(priority_csv),
        "num_priority_pool_images": len(pool_df),
        "num_fixed_eval_images": len(fixed_eval_df),
        "initial_seed": seeds[0],
        "initial_seed_size": len(initial_seed_df),
        "AL_FIXED_EVAL_SIZE": int(os.environ.get("AL_FIXED_EVAL_SIZE", "180")),
        "AL_EVAL_MATCH_POOL_CLASSES": parse_bool_env("AL_EVAL_MATCH_POOL_CLASSES", True),
        "AL_MIN_POOL_CLASS_COUNT_FOR_EVAL": int(os.environ.get("AL_MIN_POOL_CLASS_COUNT_FOR_EVAL", "3")),
        "AL_PHASH_MAX_HAMMING": max_phash_hamming,
    }

    (save_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    distribution_df.to_csv(save_dir / "dataset_class_instance_distribution.csv", index=False, encoding="utf-8-sig")
    conversion_df.to_csv(save_dir / "xml_to_yolo_conversion_audit.csv", index=False, encoding="utf-8-sig")
    suspicious_df.to_csv(save_dir / "invalid_or_suspicious_boxes.csv", index=False, encoding="utf-8-sig")
    mismatch_df.to_csv(save_dir / "image_class_hint_vs_xml_class_mismatch.csv", index=False, encoding="utf-8-sig")
    exact_overlap_df.to_csv(save_dir / "pool_eval_exact_overlap.csv", index=False, encoding="utf-8-sig")
    phash_overlap_df.to_csv(save_dir / "pool_eval_perceptual_hash_overlap.csv", index=False, encoding="utf-8-sig")
    convergence_df.to_csv(save_dir / "training_run_convergence_audit.csv", index=False, encoding="utf-8-sig")

    write_summary(
        save_dir=save_dir,
        config=config,
        distribution_df=distribution_df,
        conversion_df=conversion_df,
        suspicious_df=suspicious_df,
        mismatch_df=mismatch_df,
        exact_overlap_df=exact_overlap_df,
        phash_overlap_df=phash_overlap_df,
        convergence_df=convergence_df,
    )

    print("[SAVE] dataset_class_instance_distribution.csv")
    print("[SAVE] xml_to_yolo_conversion_audit.csv")
    print("[SAVE] invalid_or_suspicious_boxes.csv")
    print("[SAVE] image_class_hint_vs_xml_class_mismatch.csv")
    print("[SAVE] pool_eval_exact_overlap.csv")
    print("[SAVE] pool_eval_perceptual_hash_overlap.csv")
    print("[SAVE] training_run_convergence_audit.csv")
    print("[SAVE] audit_summary.md")
    print("=" * 100)
    print(f"[DONE] Audit output dir: {save_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()

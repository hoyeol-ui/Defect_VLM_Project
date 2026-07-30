"""Prepare V7 development/final evaluation manifests without evaluating models.

Important protocol boundary:
    - development_eval_v7 is allowed during method development.
    - final_test_v7 is generated and audited, but it must not be evaluated until
      the method and hyperparameters are locked.

Outputs:
    runs/evaluation_protocol_v7/eval_protocol_YYYYMMDD_HHMMSS/
        development_eval_v7.csv
        final_test_v7.csv
        final_test_v7_hash_audit.csv
        evaluation_protocol_v7.json
        evaluation_protocol_summary.md
"""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402
from audit_detection_pipeline_v7 import (  # noqa: E402
    IMAGE_EXTENSIONS,
    average_hash,
    build_fixed_external_eval_df_fast,
    build_image_index,
    build_xml_index,
    compute_file_sha256,
    find_xml_path_fast,
    hamming_hex,
    load_priority_scores,
    parse_bool_env,
    resolve_image_path_fast,
)
from experiment_registry_v7 import append_registry_row  # noqa: E402


PROJECT_ROOT = v6.PROJECT_ROOT
DATA_ROOT = v6.DATA_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "evaluation_protocol_v7"


def latest_v6_fixed_eval() -> Path | None:
    root = PROJECT_ROOT / "runs" / "active_learning_ablation_v6_deficit_diversity"
    if not root.exists():
        return None
    candidates = sorted(root.glob("al_ablation_v6_deficit_diversity_*/fixed_external_evaluation_split.csv"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def parse_xml_classes_and_boxes(row: pd.Series, image_path: Path, xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_width, image_height = v6.get_image_size_from_xml_or_image(root, image_path)
    mapped_classes: list[str] = []
    areas: list[float] = []

    for obj in root.findall("object"):
        name_node = obj.find("name")
        raw_label = name_node.text if name_node is not None else None
        mapped = v6.map_class_name(raw_label, str(row.get("dataset_type")), image_path)
        if mapped not in v6.CLASS_MAP:
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except Exception:
            continue
        xmin = max(0.0, min(xmin, image_width - 1))
        ymin = max(0.0, min(ymin, image_height - 1))
        xmax = max(0.0, min(xmax, image_width - 1))
        ymax = max(0.0, min(ymax, image_height - 1))
        if xmax <= xmin or ymax <= ymin:
            continue
        mapped_classes.append(mapped)
        areas.append(((xmax - xmin) / image_width) * ((ymax - ymin) / image_height))

    counts = Counter(mapped_classes)
    primary = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if counts else "unknown"
    return {
        "xml_mapped_classes": "|".join(sorted(counts)),
        "primary_xml_class": primary,
        "num_xml_instances": int(sum(counts.values())),
        "mean_bbox_area": float(np.mean(areas)) if areas else np.nan,
    }


def enrich_manifest(
    df: pd.DataFrame,
    image_index: dict[tuple[str, str], list[Path]],
    xml_index: dict[tuple[str, str], list[Path]],
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        image_path = resolve_image_path_fast(row, image_index)
        if image_path is None:
            continue
        xml_path = find_xml_path_fast(row, image_path, xml_index)
        if xml_path is None:
            continue
        payload = dict(row)
        payload["resolved_image_path"] = str(image_path)
        payload["resolved_xml_path"] = str(xml_path)
        payload["sha256"] = compute_file_sha256(image_path)
        payload["ahash"] = average_hash(image_path)
        payload.update(parse_xml_classes_and_boxes(pd.Series(payload), image_path, xml_path))
        rows.append(payload)
    return v6.stable_sample_order(pd.DataFrame(rows)).reset_index(drop=True)


def enumerate_all_labeled_images(
    image_index: dict[tuple[str, str], list[Path]],
    xml_index: dict[tuple[str, str], list[Path]],
) -> pd.DataFrame:
    rows = []
    for (dataset_type, image_name), paths in sorted(image_index.items()):
        image_path = paths[0]
        row = {
            "image_name": image_name,
            "dataset_type": dataset_type,
            "image_path": str(image_path),
        }
        xml_path = find_xml_path_fast(pd.Series(row), image_path, xml_index)
        if xml_path is None:
            continue
        row["class_hint"] = v6.infer_class_hint(pd.Series(row))
        rows.append(row)
    return pd.DataFrame(rows)


def build_hash_audit(
    candidates: pd.DataFrame,
    excluded: pd.DataFrame,
    max_hamming: int,
) -> pd.DataFrame:
    rows = []
    excluded_keys = set(zip(excluded["dataset_type"].astype(str), excluded["image_name"].astype(str)))
    excluded_paths = set(excluded["resolved_image_path"].astype(str))
    excluded_sha = set(excluded["sha256"].dropna().astype(str))
    excluded_hash_rows = excluded[["dataset_type", "image_name", "resolved_image_path", "sha256", "ahash"]].to_dict("records")

    for _, row in candidates.iterrows():
        reasons = []
        if (str(row["dataset_type"]), str(row["image_name"])) in excluded_keys:
            reasons.append("dataset_type_image_name_overlap")
        if str(row["resolved_image_path"]) in excluded_paths:
            reasons.append("resolved_path_overlap")
        if pd.notna(row.get("sha256")) and str(row["sha256"]) in excluded_sha:
            reasons.append("sha256_overlap")

        min_hd = None
        nearest = None
        for ex in excluded_hash_rows:
            hd = hamming_hex(str(row.get("ahash")), str(ex.get("ahash"))) if pd.notna(row.get("ahash")) and pd.notna(ex.get("ahash")) else None
            if hd is None:
                continue
            if min_hd is None or hd < min_hd:
                min_hd = hd
                nearest = ex
        if min_hd is not None and min_hd <= max_hamming:
            reasons.append(f"perceptual_hash_hamming_le_{max_hamming}")

        rows.append(
            {
                "dataset_type": row["dataset_type"],
                "image_name": row["image_name"],
                "resolved_image_path": row["resolved_image_path"],
                "primary_xml_class": row.get("primary_xml_class"),
                "sha256": row.get("sha256"),
                "ahash": row.get("ahash"),
                "exclude": bool(reasons),
                "exclude_reasons": "|".join(reasons),
                "nearest_excluded_hamming": min_hd,
                "nearest_excluded_image": nearest.get("resolved_image_path") if nearest else None,
            }
        )
    return pd.DataFrame(rows)


def stratified_sample_final_test(
    candidates: pd.DataFrame,
    target_size: int,
    seed: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    grouped = list(candidates.groupby(["dataset_type", "primary_xml_class"], sort=True, dropna=False))
    target = min(target_size, len(candidates))
    base = target // len(grouped)
    remainder = target % len(grouped)
    selected_parts = []
    for idx, (_, sub) in enumerate(grouped):
        quota = min(base + (1 if idx < remainder else 0), len(sub))
        if quota:
            selected_parts.append(v6.stable_sample_order(sub).sample(n=quota, random_state=seed + idx * 101))
    selected = pd.concat(selected_parts) if selected_parts else candidates.iloc[0:0].copy()
    if len(selected) < target:
        remaining = candidates.drop(index=selected.index, errors="ignore")
        selected = pd.concat(
            [
                selected,
                v6.stable_sample_order(remaining).sample(
                    n=min(target - len(selected), len(remaining)),
                    random_state=seed + 999,
                ),
            ]
        )
    return v6.stable_sample_order(selected.head(target)).reset_index(drop=True)


def write_summary(
    save_dir: Path,
    config: dict,
    dev_df: pd.DataFrame,
    final_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> None:
    final_limits = audit_df["exclude_reasons"].replace("", np.nan).dropna().value_counts().head(20)
    lines = [
        "# Evaluation Protocol V7",
        "",
        "## Boundary",
        "",
        "- `development_eval_v7` may be used for method development/screening.",
        "- `final_test_v7` must not be evaluated until method and hyperparameters are locked.",
        "- This script only builds manifests and leakage/hash audits; it does not run YOLO evaluation.",
        "",
        "## Split sizes",
        "",
        f"- development_eval_v7: {len(dev_df)} images",
        f"- final_test_v7: {len(final_df)} images",
        "",
        "## Development distribution by XML primary class",
        "",
        dev_df.groupby(["dataset_type", "primary_xml_class"]).size().reset_index(name="count").to_markdown(index=False),
        "",
        "## Final-test distribution by XML primary class",
        "",
        final_df.groupby(["dataset_type", "primary_xml_class"]).size().reset_index(name="count").to_markdown(index=False) if len(final_df) else "_No final-test images available after exclusions._",
        "",
        "## Exclusion reasons among final-test candidates",
        "",
        final_limits.to_frame("count").to_markdown() if len(final_limits) else "_No excluded candidates._",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
    ]
    (save_dir / "evaluation_protocol_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"eval_protocol_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    final_target_size = int(os.environ.get("AL_FINAL_TEST_SIZE", "180"))
    final_seed = int(os.environ.get("AL_FINAL_TEST_SEED", "20260711"))
    max_hamming = int(os.environ.get("AL_FINAL_TEST_PHASH_MAX_HAMMING", "2"))

    priority_csv, pool_df = load_priority_scores()
    image_index = build_image_index()
    xml_index = build_xml_index()

    existing_dev = latest_v6_fixed_eval()
    if existing_dev is not None and parse_bool_env("AL_REUSE_LATEST_V6_FIXED_EVAL", True):
        dev_raw = pd.read_csv(existing_dev)
        dev_source = str(existing_dev)
    else:
        dev_raw = build_fixed_external_eval_df_fast(pool_df, image_index, xml_index)
        dev_source = "rebuilt_from_priority_pool"

    dev_df = enrich_manifest(dev_raw, image_index, xml_index)
    pool_enriched = enrich_manifest(pool_df, image_index, xml_index)
    all_images = enrich_manifest(enumerate_all_labeled_images(image_index, xml_index), image_index, xml_index)
    excluded = pd.concat([pool_enriched, dev_df], ignore_index=True)

    audit_df = build_hash_audit(all_images, excluded, max_hamming=max_hamming)
    allowed_keys = set(
        zip(
            audit_df.loc[~audit_df["exclude"], "dataset_type"].astype(str),
            audit_df.loc[~audit_df["exclude"], "image_name"].astype(str),
        )
    )
    final_candidates = all_images[
        [
            key in allowed_keys
            for key in zip(all_images["dataset_type"].astype(str), all_images["image_name"].astype(str))
        ]
    ].copy()
    final_df = stratified_sample_final_test(final_candidates, target_size=final_target_size, seed=final_seed)

    config = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "priority_csv": str(priority_csv),
        "development_eval_source": dev_source,
        "final_test_seed": final_seed,
        "final_test_target_size": final_target_size,
        "final_test_actual_size": len(final_df),
        "final_test_phash_max_hamming": max_hamming,
        "warning": "final_test_v7 must not be evaluated before method/hyperparameters are locked.",
    }

    dev_df.to_csv(save_dir / "development_eval_v7.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(save_dir / "final_test_v7.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(save_dir / "final_test_v7_hash_audit.csv", index=False, encoding="utf-8-sig")
    (save_dir / "evaluation_protocol_v7.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(save_dir, config, dev_df, final_df, audit_df)

    append_registry_row(
        save_dir / "experiment_registry.csv",
        project_root=PROJECT_ROOT,
        experiment_id=save_dir.name,
        stage="evaluation_protocol",
        eval_split="development_eval_v7/final_test_v7_manifest_only",
        status="success",
        result_path=save_dir,
        hyperparameters=config,
    )

    print("=" * 100)
    print("[DONE] Evaluation protocol V7 manifests created")
    print(f"Output dir: {save_dir}")
    print("IMPORTANT: Do not evaluate final_test_v7 until method/hyperparameters are locked.")
    print("=" * 100)


if __name__ == "__main__":
    main()

"""Post-hoc XML instance-richness diagnostics for fixed selected sets.

This script uses GT XML only after acquisition, for analysis.  It must never be
used by GT-free acquisition logic.

Outputs:
    runs/instance_richness_v7/instance_richness_YYYYMMDD_HHMMSS/
"""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from math import log2
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402
from audit_detection_pipeline_v7 import (  # noqa: E402
    build_image_index,
    build_xml_index,
    find_xml_path_fast,
    resolve_image_path_fast,
)
from experiment_registry_v7 import append_registry_row  # noqa: E402


PROJECT_ROOT = v6.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "instance_richness_v7"


def latest_training_variance_run() -> Path:
    override = os.environ.get("TRAINING_VARIANCE_RUN_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    root = PROJECT_ROOT / "runs" / "training_variance_v7"
    runs = [p for p in root.glob("training_variance_*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError("No training_variance_v7 run found. Set TRAINING_VARIANCE_RUN_DIR.")
    return max(runs, key=lambda p: p.stat().st_mtime)


def entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = [v / total for v in counts.values() if v > 0]
    return float(-sum(p * log2(p) for p in probs))


def parse_image_instances(row: pd.Series, image_index, xml_index) -> list[dict]:
    image_path = resolve_image_path_fast(row, image_index)
    if image_path is None:
        return []
    xml_path = find_xml_path_fast(row, image_path, xml_index)
    if xml_path is None:
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_width, image_height = v6.get_image_size_from_xml_or_image(root, image_path)
    instances = []
    for obj_idx, obj in enumerate(root.findall("object")):
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
        width_norm = (xmax - xmin) / image_width
        height_norm = (ymax - ymin) / image_height
        area = width_norm * height_norm
        if area < 0.01:
            scale = "small"
        elif area < 0.08:
            scale = "medium"
        else:
            scale = "large"
        instances.append(
            {
                "source_strategy": row.get("source_strategy", row.get("strategy")),
                "seed": row.get("seed"),
                "strategy": row.get("strategy"),
                "round": row.get("round"),
                "image_name": row.get("image_name"),
                "dataset_type": row.get("dataset_type"),
                "class_hint": row.get("class_hint"),
                "image_path": str(image_path),
                "xml_path": str(xml_path),
                "object_index": obj_idx,
                "raw_label": raw_label,
                "actual_xml_class": mapped,
                "bbox_area_norm": area,
                "bbox_width_norm": width_norm,
                "bbox_height_norm": height_norm,
                "bbox_scale": scale,
                "image_width": image_width,
                "image_height": image_height,
            }
        )
    return instances


def strategy_stats(frozen_df: pd.DataFrame, instance_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, selected in frozen_df.groupby("source_strategy", dropna=False):
        inst = instance_df[instance_df["source_strategy"].astype(str).eq(str(strategy))]
        per_image = inst.groupby(["dataset_type", "image_name"]).size() if len(inst) else pd.Series(dtype=int)
        class_counts = Counter(inst["actual_xml_class"]) if len(inst) else Counter()
        area = pd.to_numeric(inst["bbox_area_norm"], errors="coerce") if len(inst) else pd.Series(dtype=float)
        rows.append(
            {
                "source_strategy": strategy,
                "num_images": len(selected.drop_duplicates(["dataset_type", "image_name"])),
                "total_bbox_instances": int(len(inst)),
                "mean_bbox_per_image": float(per_image.mean()) if len(per_image) else 0.0,
                "median_bbox_per_image": float(per_image.median()) if len(per_image) else 0.0,
                "multi_object_image_ratio": float((per_image > 1).mean()) if len(per_image) else 0.0,
                "num_actual_classes": int(len(class_counts)),
                "actual_class_entropy": entropy_from_counts(class_counts),
                "actual_class_distribution": json.dumps(dict(sorted(class_counts.items())), ensure_ascii=False),
                "bbox_area_mean": float(area.mean()) if len(area) else np.nan,
                "bbox_area_std": float(area.std(ddof=1)) if len(area) > 1 else np.nan,
                "small_bbox_ratio": float((inst["bbox_scale"] == "small").mean()) if len(inst) else 0.0,
                "medium_bbox_ratio": float((inst["bbox_scale"] == "medium").mean()) if len(inst) else 0.0,
                "large_bbox_ratio": float((inst["bbox_scale"] == "large").mean()) if len(inst) else 0.0,
                "dataset_distribution": json.dumps(
                    selected["dataset_type"].value_counts().sort_index().to_dict(),
                    ensure_ascii=False,
                ),
            }
        )
    return pd.DataFrame(rows)


def make_performance_join(stats_df: pd.DataFrame, variance_run: Path) -> pd.DataFrame:
    summary_path = variance_run / "training_variance_metric_summary.csv"
    if not summary_path.exists():
        return stats_df.copy()
    summary = pd.read_csv(summary_path)
    pivot = summary.pivot(index="strategy", columns="metric", values="mean").reset_index()
    out = stats_df.merge(pivot, left_on="source_strategy", right_on="strategy", how="left")
    numeric = [
        "total_bbox_instances",
        "mean_bbox_per_image",
        "multi_object_image_ratio",
        "num_actual_classes",
        "actual_class_entropy",
        "bbox_area_mean",
        "map50",
        "map5095",
    ]
    correlations = []
    for feature in numeric:
        if feature in out.columns:
            for metric in ["map50", "map5095"]:
                if feature == metric or metric not in out.columns:
                    continue
                corr = out[[feature, metric]].corr().iloc[0, 1]
                correlations.append({"feature": feature, "metric": metric, "pearson_r": corr})
    return out, pd.DataFrame(correlations)


def write_summary(save_dir: Path, stats_df: pd.DataFrame, perf_df: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    lines = [
        "# Instance Richness V7",
        "",
        "GT XML is used here only for post-hoc diagnosis, not acquisition.",
        "",
        "## Frozen-set actual instance statistics",
        "",
        stats_df.to_markdown(index=False),
        "",
        "## Instance richness vs performance",
        "",
        perf_df.to_markdown(index=False),
        "",
        "## Correlations",
        "",
        corr_df.to_markdown(index=False) if len(corr_df) else "_No correlations available._",
        "",
        "Interpret correlations cautiously: with three strategies this is descriptive, not inferential.",
    ]
    (save_dir / "instance_richness_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"instance_richness_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    variance_run = latest_training_variance_run()
    frozen_path = variance_run / "frozen_labeled_sets.csv"
    if not frozen_path.exists():
        raise FileNotFoundError(f"frozen_labeled_sets.csv not found: {frozen_path}")

    frozen_df = pd.read_csv(frozen_path)
    image_index = build_image_index()
    xml_index = build_xml_index()

    instance_rows = []
    for _, row in frozen_df.iterrows():
        instance_rows.extend(parse_image_instances(row, image_index, xml_index))
    instance_df = pd.DataFrame(instance_rows)
    stats_df = strategy_stats(frozen_df, instance_df)

    perf_join = make_performance_join(stats_df, variance_run)
    if isinstance(perf_join, tuple):
        perf_df, corr_df = perf_join
    else:
        perf_df, corr_df = perf_join, pd.DataFrame()

    class_dist = (
        instance_df.groupby(["source_strategy", "actual_xml_class"], dropna=False)
        .size()
        .reset_index(name="bbox_instance_count")
        if len(instance_df)
        else pd.DataFrame(columns=["source_strategy", "actual_xml_class", "bbox_instance_count"])
    )
    geom_stats = (
        instance_df.groupby(["source_strategy", "bbox_scale"], dropna=False)
        .agg(
            count=("bbox_area_norm", "size"),
            area_mean=("bbox_area_norm", "mean"),
            area_std=("bbox_area_norm", "std"),
        )
        .reset_index()
        if len(instance_df)
        else pd.DataFrame(columns=["source_strategy", "bbox_scale", "count", "area_mean", "area_std"])
    )

    config = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "training_variance_run": str(variance_run),
        "warning": "GT XML was used only for post-hoc diagnostics.",
    }
    (save_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_df.to_csv(save_dir / "frozen_set_actual_instance_statistics.csv", index=False, encoding="utf-8-sig")
    class_dist.to_csv(save_dir / "frozen_set_actual_class_distribution.csv", index=False, encoding="utf-8-sig")
    geom_stats.to_csv(save_dir / "frozen_set_bbox_geometry_statistics.csv", index=False, encoding="utf-8-sig")
    perf_df.to_csv(save_dir / "instance_richness_vs_performance.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(save_dir / "instance_richness_correlations.csv", index=False, encoding="utf-8-sig")
    instance_df.to_csv(save_dir / "per_instance_rows.csv", index=False, encoding="utf-8-sig")
    write_summary(save_dir, stats_df, perf_df, corr_df)

    append_registry_row(
        save_dir / "experiment_registry.csv",
        project_root=PROJECT_ROOT,
        experiment_id=save_dir.name,
        stage="posthoc_instance_richness",
        eval_split="development_eval_v7",
        status="success",
        result_path=save_dir,
        hyperparameters=config,
    )

    print("=" * 100)
    print("[DONE] Instance richness diagnostics")
    print(f"Output dir: {save_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()

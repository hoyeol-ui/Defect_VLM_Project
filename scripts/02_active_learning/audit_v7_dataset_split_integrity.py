"""Audit V7 dataset split integrity without training or final-test evaluation.

This script reads only manifests, priority CSV rows, and selection CSVs.  It is
safe to run while the final test is locked because it does not validate a model
or compute any final-test metric.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402
from audit_detection_pipeline_v7 import build_image_index, load_priority_scores  # noqa: E402
from canonical_sampling_v7 import canonicalize_pool_for_sampling  # noqa: E402


PROJECT_ROOT = v6.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "v7_dataset_split_integrity_audit"


def latest_dir(root: Path, prefix: str) -> Path:
    runs = [p for p in root.glob(f"{prefix}*") if p.is_dir()] if root.exists() else []
    if not runs:
        raise FileNotFoundError(f"No {prefix} run found under {root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def latest_eval_protocol_dir() -> Path:
    override = os.environ.get("EVAL_PROTOCOL_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    return latest_dir(PROJECT_ROOT / "runs" / "evaluation_protocol_v7", "eval_protocol_")


def latest_selection_dir() -> Path | None:
    override = os.environ.get("AL_SELECTION_RUN_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    root = PROJECT_ROOT / "runs" / "active_learning_ablation_v7_full_curve"
    runs = [p for p in root.glob("v7_full_curve_*") if (p / "cumulative_labeled_sets_by_round.csv").exists()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def canonical_id_from_eval(df: pd.DataFrame) -> pd.Series:
    return df["resolved_image_path"].astype(str) + "::" + df["sha256"].astype(str)


def distribution(df: pd.DataFrame, class_col: str) -> pd.DataFrame:
    return (
        df.groupby(["dataset_type", class_col], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["dataset_type", class_col], kind="mergesort")
        .reset_index(drop=True)
    )


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"dataset_split_integrity_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    priority_csv, priority_df = load_priority_scores()
    pool = canonicalize_pool_for_sampling(priority_df, build_image_index(), set_sample_id=True)

    protocol_dir = latest_eval_protocol_dir()
    dev_path = protocol_dir / "development_eval_v7.csv"
    final_path = protocol_dir / "final_test_v7.csv"
    dev = pd.read_csv(dev_path)
    final = pd.read_csv(final_path)
    dev["canonical_sample_id"] = canonical_id_from_eval(dev)
    final["canonical_sample_id"] = canonical_id_from_eval(final)

    pool_ids = set(pool["canonical_sample_id"].astype(str))
    dev_ids = set(dev["canonical_sample_id"].astype(str))
    final_ids = set(final["canonical_sample_id"].astype(str))

    overlap_rows = [
        {"left": "acquisition_pool", "right": "development_eval_v7", "overlap_count": len(pool_ids & dev_ids)},
        {"left": "acquisition_pool", "right": "final_test_v7", "overlap_count": len(pool_ids & final_ids)},
        {"left": "development_eval_v7", "right": "final_test_v7", "overlap_count": len(dev_ids & final_ids)},
    ]
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(save_dir / "dataset_split_overlap_summary.csv", index=False, encoding="utf-8-sig")

    pool_dist = distribution(pool, "class_hint")
    dev_dist = distribution(dev, "primary_xml_class")
    final_dist = distribution(final, "primary_xml_class")
    pool_dist.to_csv(save_dir / "pool_class_distribution.csv", index=False, encoding="utf-8-sig")
    dev_dist.to_csv(save_dir / "development_eval_class_distribution.csv", index=False, encoding="utf-8-sig")
    final_dist.to_csv(save_dir / "final_test_class_distribution_manifest_only.csv", index=False, encoding="utf-8-sig")

    selection_dir = latest_selection_dir()
    selected_summary = pd.DataFrame()
    selected_overlap_summary = pd.DataFrame()
    if selection_dir is not None:
        cumulative_path = selection_dir / "cumulative_labeled_sets_by_round.csv"
        selected = pd.read_csv(cumulative_path)
        max_round = int(pd.to_numeric(selected["round"], errors="coerce").max())
        selected_round = selected[selected["round"].eq(max_round)].copy()
        selected_summary = (
            selected_round.groupby(["acquisition_seed", "strategy", "dataset_type"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["acquisition_seed", "strategy", "dataset_type"], kind="mergesort")
            .reset_index(drop=True)
        )
        selected_summary.to_csv(save_dir / "latest_selection_final_round_dataset_distribution.csv", index=False, encoding="utf-8-sig")

        selected_ids = set(selected["canonical_sample_id"].astype(str)) if "canonical_sample_id" in selected.columns else set()
        selected_overlap_summary = pd.DataFrame(
            [
                {"left": "latest_selection_all_rounds", "right": "development_eval_v7", "overlap_count": len(selected_ids & dev_ids)},
                {"left": "latest_selection_all_rounds", "right": "final_test_v7", "overlap_count": len(selected_ids & final_ids)},
            ]
        )
        selected_overlap_summary.to_csv(save_dir / "latest_selection_overlap_summary.csv", index=False, encoding="utf-8-sig")

    neu_classes = set(v6.CLASS_NAMES[:6])
    gc10_classes = set(v6.GC10_DIGIT_MAP.values())
    shared_class_names = sorted(neu_classes & gc10_classes)
    class_space = {
        "class_names": v6.CLASS_NAMES,
        "neu_classes_by_protocol": sorted(neu_classes),
        "gc10_classes_by_mapping": sorted(gc10_classes),
        "shared_semantic_class_names": shared_class_names,
        "note": "A shared name means both datasets are mapped to one YOLO class id. This is intentional only if the defect semantics are treated as the same class.",
    }
    (save_dir / "class_space_integrity.json").write_text(json.dumps(class_space, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "audit_dir": str(save_dir),
        "priority_csv": str(priority_csv),
        "eval_protocol_dir": str(protocol_dir),
        "selection_dir": str(selection_dir) if selection_dir else None,
        "counts": {
            "acquisition_pool": len(pool),
            "development_eval_v7": len(dev),
            "final_test_v7_manifest_only": len(final),
        },
        "overlaps": overlap_rows,
        "pool_dataset_counts": pool["dataset_type"].value_counts().to_dict(),
        "development_eval_dataset_counts": dev["dataset_type"].value_counts().to_dict(),
        "final_test_dataset_counts_manifest_only": final["dataset_type"].value_counts().to_dict(),
        "shared_semantic_class_names": shared_class_names,
    }
    (save_dir / "dataset_split_integrity_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# V7 Dataset Split Integrity Audit",
        "",
        "This audit reads manifests and selection CSVs only. No YOLO training and no final-test evaluation were run.",
        "",
        "## Verdict",
        "",
        "- Path/SHA split leakage: **not detected** if all overlap counts below are 0.",
        "- Important protocol caveat: development evaluation is NEU-heavy while final-test manifest is GC10-heavy.",
        "- Class-space caveat: `inclusion` is shared between NEU and GC10 in the unified YOLO class map.",
        "",
        "## Overlap summary",
        "",
        overlap_df.to_markdown(index=False),
        "",
        "## Dataset counts",
        "",
        f"- Acquisition pool: {len(pool)} rows; {pool['dataset_type'].value_counts().to_dict()}",
        f"- Development eval: {len(dev)} rows; {dev['dataset_type'].value_counts().to_dict()}",
        f"- Final-test manifest only: {len(final)} rows; {final['dataset_type'].value_counts().to_dict()}",
        "",
        "## Acquisition pool class distribution",
        "",
        pool_dist.to_markdown(index=False),
        "",
        "## Development eval class distribution",
        "",
        dev_dist.to_markdown(index=False),
        "",
        "## Final-test manifest class distribution",
        "",
        final_dist.to_markdown(index=False),
        "",
        "## Class-space notes",
        "",
        f"- Unified YOLO class names: {v6.CLASS_NAMES}",
        f"- Shared semantic class names across NEU/GC10 mapping: {shared_class_names}",
        "- Dataset-specific filenames are prefixed by dataset type during YOLO dataset build, so image/label file overwrites are not expected.",
    ]
    if selection_dir is not None:
        lines.extend(
            [
                "",
                "## Latest selection run checked",
                "",
                str(selection_dir),
                "",
                "### Latest selection overlap summary",
                "",
                selected_overlap_summary.to_markdown(index=False) if len(selected_overlap_summary) else "_No selection overlap rows._",
                "",
                "### Final round dataset distribution",
                "",
                selected_summary.to_markdown(index=False) if len(selected_summary) else "_No selected summary rows._",
            ]
        )
    (save_dir / "dataset_split_integrity_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("=" * 100)
    print("[DONE] V7 dataset split integrity audit")
    print(f"Output dir: {save_dir}")
    print("No YOLO training. No final-test evaluation.")
    print("=" * 100)


if __name__ == "__main__":
    main()

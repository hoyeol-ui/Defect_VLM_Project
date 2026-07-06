"""
Audit no_pseudo_box samples against available GT XML annotations.

This script does not train YOLO and does not affect acquisition. It is a
post-hoc audit to separate no_pseudo_box into likely OVD failure vs potentially
hard/ambiguous samples.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path

pd = None
plt = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RUNS_ROOT = PROJECT_ROOT / "runs" / "active_learning_ablation_v3_minimal"
OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs"
NO_PSEUDO_REASON = "no_pseudo_box"

GC10_DIGIT_MAP = {
    "1": "punching_hole",
    "2": "welding_line",
    "3": "crescent_gap",
    "4": "water_spot",
    "5": "oil_spot",
    "6": "silk_spot",
    "7": "inclusion",
    "8": "rolled_pit",
    "9": "crease",
    "10": "waist_folding",
}

GC10_RAW_MAP = {
    "1_chongkong": "punching_hole",
    "2_hanfeng": "welding_line",
    "3_yueyawan": "crescent_gap",
    "4_shuiban": "water_spot",
    "5_youban": "oil_spot",
    "6_siban": "silk_spot",
    "7_yiwu": "inclusion",
    "8_yahen": "rolled_pit",
    "9_zhehen": "crease",
    "10_yaozhe": "waist_folding",
    "chongkong": "punching_hole",
    "hanfeng": "welding_line",
    "yueyawan": "crescent_gap",
    "shuiban": "water_spot",
    "youban": "oil_spot",
    "siban": "silk_spot",
    "yiwu": "inclusion",
    "yahen": "rolled_pit",
    "zhehen": "crease",
    "yaozhe": "waist_folding",
}


def load_dependencies():
    global pd, plt
    import pandas as _pd
    import matplotlib.pyplot as _plt

    pd = _pd
    plt = _plt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="AL run directory. Defaults to latest v3 minimal run.")
    parser.add_argument("--priority-dir", default=None, help="pseudo_boxes_* directory. Defaults to latest with priority CSV.")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Dataset root containing NEU-DET and GC10-DET.")
    return parser.parse_args()


def latest_dir(root: Path, pattern: str, required_file: str | None = None) -> Path:
    candidates = [p for p in root.glob(pattern) if p.is_dir()]
    if required_file:
        candidates = [p for p in candidates if (p / required_file).exists()]
    if not candidates:
        raise FileNotFoundError(f"No directory found: {root}/{pattern}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def infer_class_hint(row) -> str:
    if "class_hint" in row and str(row.get("class_hint")) not in ["", "nan", "None"]:
        return str(row.get("class_hint"))
    dataset_type = str(row.get("dataset_type", ""))
    image_name = str(row.get("image_name", ""))
    image_path = Path(str(row.get("image_path", "")))
    if "NEU" in dataset_type.upper():
        return re.sub(r"_\d+$", "", Path(image_name).stem)
    if "GC10" in dataset_type.upper():
        parent = image_path.parent.name
        return GC10_DIGIT_MAP.get(parent, parent or "unknown")
    return "unknown"


def find_image_path(row, data_root: Path) -> Path | None:
    image_path = Path(str(row.get("image_path", "")))
    if image_path.exists():
        return image_path
    dataset_type = str(row.get("dataset_type", ""))
    image_name = str(row.get("image_name", ""))
    candidates = []
    if "NEU" in dataset_type.upper():
        candidates.append(data_root / "NEU-DET" / "IMAGES" / image_name)
    elif "GC10" in dataset_type.upper():
        candidates.extend((data_root / "GC10-DET").glob(f"**/{image_name}"))
    else:
        candidates.extend(data_root.glob(f"**/{image_name}"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_xml_path(row, image_path: Path | None, data_root: Path) -> Path | None:
    dataset_type = str(row.get("dataset_type", ""))
    image_name = str(row.get("image_name", ""))
    stem = Path(image_name).stem
    candidates = []
    if "NEU" in dataset_type.upper():
        candidates.extend([
            data_root / "NEU-DET" / "ANNOTATIONS" / f"{stem}.xml",
            data_root / "NEU-DET" / "Annotations" / f"{stem}.xml",
            data_root / "NEU-DET" / "annotations" / f"{stem}.xml",
        ])
    elif "GC10" in dataset_type.upper():
        candidates.extend([
            data_root / "GC10-DET" / "lable" / f"{stem}.xml",
            data_root / "GC10-DET" / "label" / f"{stem}.xml",
            data_root / "GC10-DET" / "labels" / f"{stem}.xml",
            data_root / "GC10-DET" / "ANNOTATIONS" / f"{stem}.xml",
            data_root / "GC10-DET" / "Annotations" / f"{stem}.xml",
            data_root / "GC10-DET" / "annotations" / f"{stem}.xml",
        ])
        candidates.extend((data_root / "GC10-DET").glob(f"**/{stem}.xml"))
    else:
        candidates.extend(data_root.glob(f"**/{stem}.xml"))
    if image_path is not None:
        candidates.append(image_path.with_suffix(".xml"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def map_label(raw_label: str, dataset_type: str, image_path: Path | None) -> str:
    raw = str(raw_label).strip()
    lower = raw.lower()
    if lower in GC10_RAW_MAP:
        return GC10_RAW_MAP[lower]
    if raw in GC10_DIGIT_MAP:
        return GC10_DIGIT_MAP[raw]
    if "GC10" in dataset_type.upper() and image_path is not None:
        parent = image_path.parent.name
        if parent in GC10_DIGIT_MAP:
            return GC10_DIGIT_MAP[parent]
    return lower


def parse_xml_objects(xml_path: Path | None, dataset_type: str, image_path: Path | None) -> dict:
    if xml_path is None or not xml_path.exists():
        return {"xml_found": False, "gt_box_count": 0, "gt_classes": [], "gt_area_ratios": []}
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = height = None
    if size is not None and size.find("width") is not None and size.find("height") is not None:
        try:
            width = float(size.find("width").text)
            height = float(size.find("height").text)
        except Exception:
            width = height = None
    classes = []
    area_ratios = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="unknown")
        classes.append(map_label(name, dataset_type, image_path))
        box = obj.find("bndbox")
        if box is not None and width and height:
            try:
                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))
                area_ratios.append(max(0.0, xmax - xmin) * max(0.0, ymax - ymin) / (width * height))
            except Exception:
                pass
    return {
        "xml_found": True,
        "gt_box_count": len(classes),
        "gt_classes": classes,
        "gt_area_ratios": area_ratios,
    }


def choose_merge_keys(left, right) -> list[str]:
    for keys in [["image_name", "dataset_type"], ["image_name", "image_path"], ["image_name"]]:
        if all(k in left.columns and k in right.columns for k in keys):
            return keys
    raise ValueError(f"No stable merge keys. left={list(left.columns)} right={list(right.columns)}")


def main():
    args = parse_args()
    load_dependencies()

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else latest_dir(RUNS_ROOT, "al_ablation_v3_minimal_*")
    priority_dir = (
        Path(args.priority_dir).expanduser().resolve()
        if args.priority_dir
        else latest_dir(OUTPUT_BASE_DIR, "pseudo_boxes_*", "priority_scores_pseudo.csv")
    )
    data_root = Path(args.data_root).expanduser().resolve()

    selected_path = run_dir / "all_selected_samples_by_round.csv"
    priority_path = priority_dir / "priority_scores_pseudo.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected CSV: {selected_path}")
    if not priority_path.exists():
        raise FileNotFoundError(f"Missing priority CSV: {priority_path}")

    selected = pd.read_csv(selected_path)
    priority = pd.read_csv(priority_path)
    if "groundedness_reason" not in priority.columns:
        raise ValueError(f"priority CSV missing groundedness_reason. columns={list(priority.columns)}")

    keys = choose_merge_keys(selected, priority)
    cols = list(dict.fromkeys(keys + ["image_path", "groundedness_reason", "pseudo_box_found", "pseudo_box_count"]))
    merged = selected.merge(
        priority[[c for c in cols if c in priority.columns]].drop_duplicates(keys),
        on=keys,
        how="left",
        suffixes=("", "_priority"),
    )
    for col in ["image_path", "groundedness_reason", "pseudo_box_found", "pseudo_box_count"]:
        priority_col = f"{col}_priority"
        if priority_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[priority_col])
            else:
                merged[col] = merged[priority_col]
            merged = merged.drop(columns=[priority_col])
    if "groundedness_reason" not in merged.columns:
        raise ValueError(
            "Merged dataframe has no groundedness_reason column.\n"
            f"Selected columns: {list(selected.columns)}\n"
            f"Priority columns: {list(priority.columns)}\n"
            f"Merged columns: {list(merged.columns)}"
        )
    acquired = merged[pd.to_numeric(merged["round"], errors="coerce").fillna(0) > 0].copy()
    no_pseudo = acquired[acquired["groundedness_reason"].astype(str) == NO_PSEUDO_REASON].copy()

    rows = []
    for _, row in no_pseudo.iterrows():
        image_path = find_image_path(row, data_root)
        xml_path = find_xml_path(row, image_path, data_root)
        audit = parse_xml_objects(xml_path, str(row.get("dataset_type", "")), image_path)
        gt_classes = audit["gt_classes"]
        rows.append({
            **row.to_dict(),
            "resolved_image_path": str(image_path) if image_path else None,
            "resolved_xml_path": str(xml_path) if xml_path else None,
            "xml_found": audit["xml_found"],
            "gt_box_count": audit["gt_box_count"],
            "gt_has_box": audit["gt_box_count"] > 0,
            "gt_classes": "|".join(gt_classes),
            "gt_primary_class": gt_classes[0] if gt_classes else "none",
            "gt_area_ratio_mean": (
                sum(audit["gt_area_ratios"]) / len(audit["gt_area_ratios"])
                if audit["gt_area_ratios"]
                else None
            ),
            "class_hint_inferred": infer_class_hint(row),
            "no_pseudo_interpretation": (
                "ovd_failure_candidate" if audit["gt_box_count"] > 0 else "possible_empty_or_missing_gt"
            ),
        })

    out_dir = run_dir / f"no_pseudo_box_gt_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    csv_dir = out_dir / "csv"
    fig_dir = out_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(csv_dir / "no_pseudo_box_gt_audit.csv", index=False)

    if len(audit_df) > 0:
        summary = (
            audit_df.groupby(["strategy", "round", "no_pseudo_interpretation"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        summary.to_csv(csv_dir / "no_pseudo_box_gt_audit_by_strategy_round.csv", index=False)

        class_summary = (
            audit_df.groupby(["strategy", "gt_primary_class"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        class_summary.to_csv(csv_dir / "no_pseudo_box_gt_class_distribution.csv", index=False)

        pivot = summary.pivot_table(
            index="strategy",
            columns="no_pseudo_interpretation",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        pivot.plot(kind="bar", stacked=True, figsize=(10, 5))
        plt.title("no_pseudo_box GT audit by strategy")
        plt.xlabel("Strategy")
        plt.ylabel("Count")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "no_pseudo_box_gt_audit_by_strategy.png", dpi=220)
        plt.close()

    lines = [
        "# no_pseudo_box GT Audit Summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- priority_dir: `{priority_dir}`",
        f"- merge_keys: `{keys}`",
        f"- acquired no_pseudo_box rows: {len(audit_df)}",
        "",
    ]
    if len(audit_df) > 0:
        counts = Counter(audit_df["no_pseudo_interpretation"])
        lines.extend([
            "## Interpretation Counts",
            "",
            *[f"- {k}: {v}" for k, v in counts.items()],
            "",
            "## Lab Meeting Use",
            "",
            "- no_pseudo_box can mean either informative hard evidence or OVD pseudo-box failure.",
            "- Rows with GT boxes but no pseudo boxes are OVD failure candidates.",
            "- Rows without resolved GT boxes need dataset/annotation-path verification before interpretation.",
        ])
    (out_dir / "summary_no_pseudo_box_gt_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()

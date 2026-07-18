"""Post-hoc audit for V10c guard failure.

This script does not train models and does not evaluate final test.  It audits
already-selected acquisition samples with GT XML only after the experiment, so
that the GT-free acquisition protocol remains intact.

Default comparison:
  - Random seed47
  - V10c 24/6 core and guard seed47
  - PDF V10c 21/9 core, one-box guard, and fill seed47

Outputs are written under:
  runs/v10c_guard_failure_audit/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V10C24_RUN = PROJECT_ROOT / "runs" / "active_learning_v10c_recall_guard_onecycle" / "v10c_recall_guard_onecycle_20260713_201155"
DEFAULT_PDF_RUN = PROJECT_ROOT / "runs" / "active_learning_v10c_pdf_recall_guard_onecycle" / "v10c_recall_guard_onecycle_20260713_213503"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "runs" / "v10c_guard_failure_audit"
NEU_ANNOTATIONS = PROJECT_ROOT / "data" / "NEU-DET" / "ANNOTATIONS"

NEU6 = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def normalize_class_name(name: Any) -> str:
    return str(name).strip().replace("rolled-in-scale", "rolled-in_scale")


def safe_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def safe_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return None
        return int(float(value))
    except Exception:
        return None


def image_stem_from_row(row: pd.Series) -> str:
    image_name = str(row.get("image_name", "")).strip()
    if image_name:
        return Path(image_name).stem
    image_path = str(row.get("resolved_image_path", row.get("image_path", ""))).strip()
    return Path(image_path.split("::")[0]).stem


def xml_path_for_row(row: pd.Series) -> Path:
    stem = image_stem_from_row(row)
    return NEU_ANNOTATIONS / f"{stem}.xml"


def parse_gt_xml(xml_path: Path) -> dict[str, Any]:
    if not xml_path.exists():
        return {
            "xml_path": str(xml_path),
            "xml_exists": False,
            "gt_box_count": np.nan,
            "gt_classes": "",
            "gt_majority_class": "",
            "gt_area_mean": np.nan,
            "gt_area_median": np.nan,
            "gt_area_min": np.nan,
            "gt_area_max": np.nan,
            "gt_area_frac_mean": np.nan,
        }
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = safe_float(size.findtext("width")) if size is not None else np.nan
    height = safe_float(size.findtext("height")) if size is not None else np.nan
    image_area = width * height if np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0 else np.nan
    classes: list[str] = []
    areas: list[float] = []
    boxes: list[tuple[float, float, float, float, str]] = []
    for obj in root.findall("object"):
        cls = normalize_class_name(obj.findtext("name", ""))
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = safe_float(bnd.findtext("xmin"))
        ymin = safe_float(bnd.findtext("ymin"))
        xmax = safe_float(bnd.findtext("xmax"))
        ymax = safe_float(bnd.findtext("ymax"))
        if not all(np.isfinite(v) for v in [xmin, ymin, xmax, ymax]):
            continue
        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)
        area = w * h
        classes.append(cls)
        areas.append(area)
        boxes.append((xmin, ymin, xmax, ymax, cls))
    counts = Counter(classes)
    majority = counts.most_common(1)[0][0] if counts else ""
    area_arr = np.asarray(areas, dtype=float)
    return {
        "xml_path": str(xml_path),
        "xml_exists": True,
        "image_width": width,
        "image_height": height,
        "gt_box_count": int(len(boxes)),
        "gt_classes": "|".join(sorted(counts.keys())),
        "gt_majority_class": majority,
        "gt_class_counts_json": json.dumps(dict(sorted(counts.items())), ensure_ascii=False),
        "gt_area_mean": float(np.mean(area_arr)) if len(area_arr) else np.nan,
        "gt_area_median": float(np.median(area_arr)) if len(area_arr) else np.nan,
        "gt_area_min": float(np.min(area_arr)) if len(area_arr) else np.nan,
        "gt_area_max": float(np.max(area_arr)) if len(area_arr) else np.nan,
        "gt_area_frac_mean": float(np.mean(area_arr / image_area)) if len(area_arr) and np.isfinite(image_area) else np.nan,
        "_gt_boxes": boxes,
    }


def load_selected(run_dir: Path, *, seed: int, strategy: str, group_prefix: str) -> pd.DataFrame:
    path = run_dir / "v10c_selected_samples.csv"
    df = pd.read_csv(path)
    df = df[pd.to_numeric(df["acquisition_seed"], errors="coerce").eq(seed)].copy()
    df = df[df["strategy"].astype(str).eq(strategy)].copy()
    if df.empty:
        return df
    if strategy == "GTFreeRandom":
        df["audit_group"] = f"{group_prefix}:random"
    else:
        phase = df.get("v10c_phase", pd.Series(["unknown"] * len(df), index=df.index)).fillna("unknown").astype(str)
        df["audit_group"] = f"{group_prefix}:" + phase
    df["source_run_dir"] = str(run_dir)
    df["source_strategy"] = strategy
    return df


def enrich_with_gt(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        gt = parse_gt_xml(xml_path_for_row(row))
        flat = {k: v for k, v in gt.items() if k != "_gt_boxes"}
        out = row.to_dict()
        out.update(flat)
        pred = normalize_class_name(out.get("detector_pred_class", ""))
        gt_classes = set(str(out.get("gt_classes", "")).split("|")) if out.get("gt_classes") else set()
        out["pred_class_in_gt_classes"] = bool(pred in gt_classes) if pred and pred != "__no_box__" else False
        out["class_hint_matches_gt_majority"] = normalize_class_name(out.get("class_hint", "")) == normalize_class_name(out.get("gt_majority_class", ""))
        out["pred_matches_class_hint"] = pred == normalize_class_name(out.get("class_hint", "")) if pred and pred != "__no_box__" else False
        out["pseudo_minus_gt_box_count"] = safe_float(out.get("detector_pseudo_box_count")) - safe_float(out.get("gt_box_count"))
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_numeric(df: pd.DataFrame, group_col: str, columns: list[str]) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby(group_col, dropna=False):
        row: dict[str, Any] = {"audit_group": group, "n": len(sub)}
        for col in columns:
            if col in sub.columns:
                vals = pd.to_numeric(sub[col], errors="coerce")
            else:
                vals = pd.Series([np.nan] * len(sub), index=sub.index)
            row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{col}_median"] = float(vals.median()) if vals.notna().any() else np.nan
            row[f"{col}_min"] = float(vals.min()) if vals.notna().any() else np.nan
            row[f"{col}_max"] = float(vals.max()) if vals.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("audit_group")


def class_distribution(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby("audit_group", dropna=False):
        counts: Counter[str] = Counter()
        for value in sub.get(label_col, pd.Series(dtype=str)).fillna("").astype(str):
            if label_col == "gt_classes" and "|" in value:
                for part in value.split("|"):
                    if part:
                        counts[normalize_class_name(part)] += 1
            elif value:
                counts[normalize_class_name(value)] += 1
        total = sum(counts.values())
        for cls, count in sorted(counts.items()):
            rows.append(
                {
                    "audit_group": group,
                    "label_source": label_col,
                    "class_name": cls,
                    "count": int(count),
                    "fraction": count / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def image_level_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby("audit_group", dropna=False):
        row = {
            "audit_group": group,
            "n": len(sub),
            "pred_class_in_gt_rate": float(sub["pred_class_in_gt_classes"].mean()) if len(sub) else np.nan,
            "pred_matches_class_hint_rate": float(sub["pred_matches_class_hint"].mean()) if len(sub) else np.nan,
            "class_hint_matches_gt_majority_rate": float(sub["class_hint_matches_gt_majority"].mean()) if len(sub) else np.nan,
            "xml_exists_rate": float(sub["xml_exists"].mean()) if len(sub) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("audit_group")


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def compute_pseudo_quality(df: pd.DataFrame, *, weights: Path, device: str, imgsz: int, conf: float, iou_thresh: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is required for --compute-pseudo-quality") from exc

    model = YOLO(str(weights))
    detail_rows = []
    for _, row in df.iterrows():
        image_path = Path(str(row.get("resolved_image_path", row.get("image_path", ""))).split("::")[0])
        gt = parse_gt_xml(xml_path_for_row(row))
        gt_boxes = gt.get("_gt_boxes", [])
        gt_xyxy = [np.asarray(box[:4], dtype=float) for box in gt_boxes]
        result = model.predict(str(image_path), imgsz=imgsz, conf=conf, iou=0.70, device=device, verbose=False)[0]
        pred_boxes = []
        if getattr(result, "boxes", None) is not None and len(result.boxes):
            for xyxy, cls_id, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                pred_boxes.append((np.asarray(xyxy, dtype=float), int(cls_id), float(score)))
        matched_gt = set()
        matched_pred = set()
        for pi, (pbox, _, _) in enumerate(pred_boxes):
            best_iou = 0.0
            best_gi = None
            for gi, gbox in enumerate(gt_xyxy):
                if gi in matched_gt:
                    continue
                value = iou_xyxy(pbox, gbox)
                if value > best_iou:
                    best_iou = value
                    best_gi = gi
            if best_gi is not None and best_iou >= iou_thresh:
                matched_pred.add(pi)
                matched_gt.add(best_gi)
        pred_count = len(pred_boxes)
        gt_count = len(gt_xyxy)
        tp = len(matched_pred)
        fp = pred_count - tp
        fn = gt_count - len(matched_gt)
        detail_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "audit_group": row.get("audit_group"),
                "image_name": row.get("image_name"),
                "gt_box_count_iou": gt_count,
                "pred_box_count_iou": pred_count,
                "pseudo_tp_iou": tp,
                "pseudo_fp_iou": fp,
                "pseudo_fn_iou": fn,
                "pseudo_precision_iou": tp / pred_count if pred_count else np.nan,
                "pseudo_recall_iou": tp / gt_count if gt_count else np.nan,
            }
        )
    detail = pd.DataFrame(detail_rows)
    summary = summarize_numeric(
        detail,
        "audit_group",
        ["gt_box_count_iou", "pred_box_count_iou", "pseudo_tp_iou", "pseudo_fp_iou", "pseudo_fn_iou", "pseudo_precision_iou", "pseudo_recall_iou"],
    )
    return detail, summary


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def write_summary(out_dir: Path, config: dict[str, Any], numeric: pd.DataFrame, quality: pd.DataFrame, gt_dist: pd.DataFrame, pred_dist: pd.DataFrame) -> None:
    lines = [
        "# V10c guard failure audit",
        "",
        "This is a post-hoc GT audit. It does not alter the GT-free acquisition protocol.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## Group numeric summary",
        "",
        md_table(numeric),
        "",
        "## Image-level detector/GT class agreement",
        "",
        md_table(quality),
        "",
        "## GT class distribution",
        "",
        md_table(gt_dist),
        "",
        "## Detector predicted class distribution",
        "",
        md_table(pred_dist),
        "",
        "## Initial interpretation prompts",
        "",
        "- If one-box guard groups have lower GT box count, low pred/GT agreement, or class concentration, pseudo box count is not a reliable false-negative proxy.",
        "- If guard groups overrepresent classes that later lose AP, the guard is introducing biased/noisy supervision rather than recall recovery.",
        "- If pseudo IoU quality is enabled, low precision/recall in one-box groups is strong evidence against low-box-count acquisition.",
    ]
    (out_dir / "guard_failure_audit_summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_bar_from_summary(df: pd.DataFrame, *, value_col: str, title: str, ylabel: str, out_path: Path) -> None:
    if df.empty or value_col not in df.columns:
        return
    plot_df = df[["audit_group", value_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    colors = ["#7f7f7f" if "random" in g else "#1f77b4" if "v10c24" in g else "#d62728" for g in plot_df["audit_group"].astype(str)]
    ax.bar(plot_df["audit_group"], plot_df[value_col], color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    for idx, value in enumerate(plot_df[value_col]):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_class_distribution(dist: pd.DataFrame, *, title: str, out_path: Path) -> None:
    if dist.empty:
        return
    pivot = dist.pivot_table(index="audit_group", columns="class_name", values="count", fill_value=0, aggfunc="sum")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = int(pivot.iloc[i, j])
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, label="count")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_plots(
    out_dir: Path,
    *,
    numeric: pd.DataFrame,
    gt_dist: pd.DataFrame,
    pred_dist: pd.DataFrame,
    pseudo_summary: pd.DataFrame | None = None,
) -> None:
    plot_bar_from_summary(
        numeric,
        value_col="gt_box_count_mean",
        title="Mean GT box count by selection group",
        ylabel="GT boxes / image",
        out_path=out_dir / "fig01_mean_gt_box_count.png",
    )
    plot_bar_from_summary(
        numeric,
        value_col="detector_pseudo_box_count_mean",
        title="Mean Round0 pseudo box count by selection group",
        ylabel="pseudo boxes / image",
        out_path=out_dir / "fig02_mean_pseudo_box_count.png",
    )
    plot_bar_from_summary(
        numeric,
        value_col="pseudo_minus_gt_box_count_mean",
        title="Mean pseudo minus GT box count by selection group",
        ylabel="pseudo boxes - GT boxes",
        out_path=out_dir / "fig03_pseudo_minus_gt_box_count.png",
    )
    plot_class_distribution(
        gt_dist,
        title="GT class distribution by selection group",
        out_path=out_dir / "fig04_gt_class_distribution.png",
    )
    plot_class_distribution(
        pred_dist,
        title="Round0 predicted class distribution by selection group",
        out_path=out_dir / "fig05_pred_class_distribution.png",
    )
    if pseudo_summary is not None and not pseudo_summary.empty:
        plot_bar_from_summary(
            pseudo_summary,
            value_col="pseudo_precision_iou_mean",
            title="Mean pseudo-box IoU precision by selection group",
            ylabel="precision @ IoU threshold",
            out_path=out_dir / "fig06_pseudo_iou_precision.png",
        )
        plot_bar_from_summary(
            pseudo_summary,
            value_col="pseudo_recall_iou_mean",
            title="Mean pseudo-box IoU recall by selection group",
            ylabel="recall @ IoU threshold",
            out_path=out_dir / "fig07_pseudo_iou_recall.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v10c24-run", default=str(DEFAULT_V10C24_RUN))
    parser.add_argument("--pdf-run", default=str(DEFAULT_PDF_RUN))
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--compute-pseudo-quality", action="store_true", help="Run Round0 YOLO predictions and compute IoU-level pseudo quality.")
    parser.add_argument("--device", default=os.environ.get("AL_YOLO_DEVICE", "0"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    args = parser.parse_args()

    v10c24_run = Path(args.v10c24_run)
    pdf_run = Path(args.pdf_run)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"guard_failure_audit_seed{args.seed}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        load_selected(v10c24_run, seed=args.seed, strategy="GTFreeRandom", group_prefix="random_from_v10c24"),
        load_selected(v10c24_run, seed=args.seed, strategy="DetectorRecallGuardDINOInstanceV10c", group_prefix="v10c24"),
        load_selected(pdf_run, seed=args.seed, strategy="DetectorPdfRecallGuardV10c", group_prefix="pdf21_9"),
    ]
    selected = pd.concat([df for df in frames if not df.empty], ignore_index=True, sort=False)
    audited = enrich_with_gt(selected)

    numeric_cols = [
        "detector_pseudo_box_count",
        "detector_uncertainty",
        "detector_max_conf",
        "static_dino_distance_to_initial",
        "_visual_distance_raw",
        "_pdf_dino_distance_raw",
        "gt_box_count",
        "gt_area_mean",
        "gt_area_frac_mean",
        "pseudo_minus_gt_box_count",
    ]
    numeric = summarize_numeric(audited, "audit_group", numeric_cols)
    gt_dist = class_distribution(audited, "gt_classes")
    gt_majority_dist = class_distribution(audited, "gt_majority_class")
    pred_dist = class_distribution(audited, "detector_pred_class")
    hint_dist = class_distribution(audited, "class_hint")
    quality = image_level_quality_summary(audited)

    audited.to_csv(out_dir / "guard_failure_audit_samples.csv", index=False, encoding="utf-8-sig")
    numeric.to_csv(out_dir / "guard_failure_numeric_summary.csv", index=False, encoding="utf-8-sig")
    gt_dist.to_csv(out_dir / "selection_group_gt_class_distribution.csv", index=False, encoding="utf-8-sig")
    gt_majority_dist.to_csv(out_dir / "selection_group_gt_majority_distribution.csv", index=False, encoding="utf-8-sig")
    pred_dist.to_csv(out_dir / "selection_group_pred_class_distribution.csv", index=False, encoding="utf-8-sig")
    hint_dist.to_csv(out_dir / "selection_group_class_hint_distribution.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(out_dir / "image_level_pred_gt_agreement_summary.csv", index=False, encoding="utf-8-sig")

    config = {
        "v10c24_run": str(v10c24_run),
        "pdf_run": str(pdf_run),
        "seed": args.seed,
        "compute_pseudo_quality": bool(args.compute_pseudo_quality),
        "final_test_used": False,
        "gt_used_for_posthoc_audit_only": True,
    }

    pseudo_summary = None
    if args.compute_pseudo_quality:
        weights = pdf_run / "yolo_train_runs" / f"seed{args.seed}___SHARED_ROUND0___R0_trainseed{1000 + args.seed}" / "weights" / "best.pt"
        detail, pseudo_summary = compute_pseudo_quality(
            audited,
            weights=weights,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou_thresh=args.iou_thresh,
        )
        detail.to_csv(out_dir / "pseudo_box_iou_quality_detail.csv", index=False, encoding="utf-8-sig")
        pseudo_summary.to_csv(out_dir / "pseudo_box_iou_quality_summary.csv", index=False, encoding="utf-8-sig")
        config["pseudo_quality_weights"] = str(weights)
        config["pseudo_quality_iou_thresh"] = args.iou_thresh

    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_plots(out_dir, numeric=numeric, gt_dist=gt_dist, pred_dist=pred_dist, pseudo_summary=pseudo_summary)
    write_summary(out_dir, config, numeric, quality, gt_dist, pred_dist)

    print("=" * 100)
    print("[DONE] V10c guard failure audit")
    print(f"Output dir: {out_dir}")
    print(f"Rows audited: {len(audited)}")
    print("Final test used=False")
    print("GT used only for post-hoc audit=True")
    print("=" * 100)


if __name__ == "__main__":
    main()

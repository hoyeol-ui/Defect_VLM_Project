"""Post-hoc GC10 groundedness audit for structured VLM prompt responses.

No model inference or training occurs here. The script joins frozen structured
signals to development XML-derived boxes only after VLM generation and excludes
every filename present in either locked-final manifest.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = PROJECT_ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715"
DEFAULT_METADATA = PROTOCOL_DIR / "gc10_development_eval.csv"
DEFAULT_BOXES = PROTOCOL_DIR / "gc10_development_bbox_gt.csv"
DEFAULT_LOCKED = [
    PROJECT_ROOT / "runs" / "evaluation_protocol_v7" / "eval_protocol_20260711_173723" / "final_test_v7.csv",
    PROTOCOL_DIR / "gc10_final_test_locked.csv",
]


def locked_names(paths: list[Path]) -> set[str]:
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


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def zone_and_scale(box: list[float]) -> tuple[str, str]:
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    vertical = "center" if 0.33 <= cy <= 0.66 else ("top" if cy < 0.33 else "bottom")
    horizontal = "center" if 0.33 <= cx <= 0.66 else ("left" if cx < 0.33 else "right")
    zone = f"{vertical}-{horizontal}" if vertical != "center" or horizontal != "center" else "center"
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    scale = "micro" if area < 0.01 else ("small" if area < 0.05 else "large")
    return zone, scale


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def safe_auc(target: pd.Series, score: pd.Series) -> float:
    mask = target.notna() & score.notna()
    if mask.sum() < 3 or target[mask].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(target[mask].astype(int), score[mask]))


def build_rows(signals: pd.DataFrame, metadata: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    meta = metadata.set_index("sample_id")
    gt_groups = {key: group.copy() for key, group in boxes.groupby("sample_id", sort=False)}
    rows: list[dict[str, Any]] = []
    for _, signal in signals.iterrows():
        image_id = str(signal["image_id"])
        if image_id not in meta.index or image_id not in gt_groups:
            continue
        m = meta.loc[image_id]
        width, height = float(m["width"]), float(m["height"])
        gt_boxes = []
        for _, gt in gt_groups[image_id].iterrows():
            gt_boxes.append(
                [
                    float(gt["x_min"]) / width,
                    float(gt["y_min"]) / height,
                    float(gt["x_max"]) / width,
                    float(gt["y_max"]) / height,
                ]
            )
        primary = max(gt_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        gt_zone, gt_scale = zone_and_scale(primary)
        pred_values = [
            signal.get("consensus_bbox_x1"), signal.get("consensus_bbox_y1"),
            signal.get("consensus_bbox_x2"), signal.get("consensus_bbox_y2"),
        ]
        pred_box = None
        try:
            candidate = [float(x) for x in pred_values]
            if all(np.isfinite(candidate)) and candidate[2] > candidate[0] and candidate[3] > candidate[1]:
                pred_box = candidate
        except (TypeError, ValueError):
            pass
        max_iou = max((box_iou(pred_box, gt) for gt in gt_boxes), default=0.0) if pred_box else 0.0
        location = str(signal.get("consensus_location_zone", ""))
        scale = str(signal.get("consensus_scale", ""))
        location_correct = bool(location and location != "nan" and location == gt_zone)
        scale_correct = bool(scale and scale != "nan" and scale == gt_scale)
        rows.append(
            {
                **signal.to_dict(),
                "filename": str(m["filename"]),
                "gt_box_count": len(gt_boxes),
                "gt_primary_zone": gt_zone,
                "gt_primary_scale": gt_scale,
                "consensus_bbox_available": pred_box is not None,
                "max_iou_to_gt": max_iou,
                "iou_failure_below_0p10": int(max_iou < 0.10),
                "iou_failure_below_0p50": int(max_iou < 0.50),
                "location_correct": location_correct,
                "scale_correct": scale_correct,
                "oracle_structured_groundedness": (float(location_correct) + float(scale_correct)) / 2.0,
            }
        )
    return pd.DataFrame(rows)


def metric_table(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("mean_valid_response_rate", float(rows["valid_response_rate"].mean())),
        (
            "mean_schema_complete_response_rate",
            float(rows["schema_complete_response_rate"].mean()),
        ),
        (
            "mean_informative_response_rate",
            float(rows["informative_response_rate"].mean()),
        ),
        ("bbox_response_coverage", float(rows["consensus_bbox_available"].mean())),
        ("mean_max_iou_to_gt", float(rows["max_iou_to_gt"].mean())),
        ("median_max_iou_to_gt", float(rows["max_iou_to_gt"].median())),
        (
            "spearman_consistency_vs_max_iou",
            safe_spearman(rows["structured_consistency"], rows["max_iou_to_gt"]),
        ),
        (
            "inconsistency_auc_iou_failure_below_0p10",
            safe_auc(rows["iou_failure_below_0p10"], rows["structured_inconsistency"]),
        ),
        (
            "spearman_consistency_vs_oracle_groundedness",
            safe_spearman(rows["structured_consistency"], rows["oracle_structured_groundedness"]),
        ),
        ("location_accuracy", float(rows["location_correct"].mean())),
        ("scale_accuracy", float(rows["scale_correct"].mean())),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def diagnostic_gate(metrics: pd.DataFrame) -> pd.DataFrame:
    values = metrics.set_index("metric")["value"]
    checks = [
        ("json_parse_rate_at_least_0p90", values["mean_valid_response_rate"] >= 0.90),
        (
            "schema_complete_rate_at_least_0p90",
            values["mean_schema_complete_response_rate"] >= 0.90,
        ),
        (
            "informative_response_rate_at_least_0p90",
            values["mean_informative_response_rate"] >= 0.90,
        ),
        ("bbox_coverage_at_least_0p80", values["bbox_response_coverage"] >= 0.80),
        ("consistency_iou_spearman_at_least_0p20", values["spearman_consistency_vs_max_iou"] >= 0.20),
        ("inconsistency_iou_failure_auc_at_least_0p60", values["inconsistency_auc_iou_failure_below_0p10"] >= 0.60),
        (
            "consistency_oracle_groundedness_spearman_at_least_0p20",
            values["spearman_consistency_vs_oracle_groundedness"] >= 0.20,
        ),
    ]
    return pd.DataFrame(
        [{"check": name, "result": "PASS" if bool(ok) else "FAIL"} for name, ok in checks]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--locked-manifest", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    signals = pd.read_csv(args.signals)
    metadata = pd.read_csv(args.metadata)
    boxes = pd.read_csv(args.boxes)
    locked = locked_names(args.locked_manifest or DEFAULT_LOCKED)
    signals["locked_final_excluded"] = signals["image_path"].map(
        lambda value: Path(str(value)).name.casefold() in locked
    )
    excluded = signals[signals["locked_final_excluded"]].copy()
    safe_signals = signals[~signals["locked_final_excluded"]].copy()
    rows = build_rows(safe_signals, metadata, boxes)
    if rows.empty:
        raise RuntimeError("No safe structured-signal rows matched GC10 development GT")
    metrics = metric_table(rows)
    gate = diagnostic_gate(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "structured_groundedness_rows.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(args.output_dir / "locked_final_exclusions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output_dir / "structured_groundedness_metrics.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(args.output_dir / "structured_groundedness_gate.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Structured VLM GC10 Groundedness Audit",
        "",
        "- Detector training performed: **False**",
        "- Final test evaluated: **False**",
        f"- Safe development images analyzed: **{len(rows)}**",
        f"- Locked-final rows excluded: **{len(excluded)}**",
        f"- Diagnostic gate: **{'PASS' if (gate['result'] == 'PASS').all() else 'FAIL'}**",
        "",
        "## Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Diagnostic gate",
        "",
        gate.to_markdown(index=False),
        "",
        "This small pilot validates prompt/grounding behavior only. It cannot authorize detector training.",
        "",
    ]
    (args.output_dir / "structured_groundedness_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"[DONE] analyzed={len(rows)} excluded_locked_final={len(excluded)}")
    print(f"[SUMMARY] {args.output_dir / 'structured_groundedness_summary.md'}")


if __name__ == "__main__":
    main()

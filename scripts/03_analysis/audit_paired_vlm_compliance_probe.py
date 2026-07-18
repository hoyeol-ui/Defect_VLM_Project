"""Score and gate the paired forced-binary VLM compliance probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def strip_fence(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", value).strip()


def valid_box(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(box)) or not all(0 <= item <= 1 for item in box):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def parse_response(raw: str) -> dict[str, object]:
    result: dict[str, object] = {
        "json_parse_ok": False,
        "schema_compliant": False,
        "predicted_defect_present": np.nan,
        "pred_x1": np.nan, "pred_y1": np.nan, "pred_x2": np.nan, "pred_y2": np.nan,
        "appearance": "",
        "confidence": np.nan,
    }
    try:
        value = json.loads(strip_fence(raw))
    except (TypeError, json.JSONDecodeError):
        return result
    if not isinstance(value, dict):
        return result
    result["json_parse_ok"] = True
    present = value.get("defect_present")
    box = valid_box(value.get("bbox_norm"))
    appearance = value.get("appearance")
    confidence = value.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = np.nan
    if isinstance(present, bool):
        result["predicted_defect_present"] = present
    if box is not None:
        result.update(dict(zip(("pred_x1", "pred_y1", "pred_x2", "pred_y2"), box)))
    result["appearance"] = str(appearance or "").strip()
    result["confidence"] = confidence_value
    bbox_rule_ok = (present is True and box is not None) or (present is False and value.get("bbox_norm") is None)
    result["schema_compliant"] = bool(
        isinstance(present, bool)
        and bbox_rule_ok
        and isinstance(appearance, str)
        and bool(appearance.strip())
        and np.isfinite(confidence_value)
        and 0 <= confidence_value <= 1
    )
    return result


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def load_responses(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id", "")).strip()
            if not image_id:
                raise RuntimeError(f"Missing image_id at response line {line_number}")
            records.append(
                {
                    "image_id": image_id,
                    "model_id": str(row.get("model_id", "unknown")),
                    "raw_response": row.get("raw_response", ""),
                    **parse_response(str(row.get("raw_response", ""))),
                }
            )
    frame = pd.DataFrame(records)
    if frame["image_id"].duplicated().any():
        raise RuntimeError("Duplicate response image_id")
    return frame


def bool_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: value if isinstance(value, (bool, np.bool_)) else np.nan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--gt-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    responses = load_responses(args.responses)
    audit = pd.read_csv(args.gt_audit)
    rows = audit.merge(responses, on="image_id", how="left", validate="one_to_one")
    if len(rows) != len(audit) or rows["raw_response"].isna().any():
        raise RuntimeError("Responses do not completely cover the frozen audit manifest")
    expected = audit["expected_defect_present"].map(
        lambda value: str(value).strip().casefold() == "true"
    )
    rows["expected_defect_present"] = expected
    rows["predicted_defect_present"] = bool_series(rows["predicted_defect_present"])
    model_ids = sorted(set(rows["model_id"].dropna().astype(str)))
    model_label = model_ids[0] if len(model_ids) == 1 else "mixed_or_unknown_model"

    positive = rows[rows["expected_defect_present"]].copy()
    negative = rows[~rows["expected_defect_present"]].copy()
    positive["bbox_iou_to_gt"] = [
        box_iou(
            [row.pred_x1, row.pred_y1, row.pred_x2, row.pred_y2],
            [row.gt_box_x1_norm, row.gt_box_y1_norm, row.gt_box_x2_norm, row.gt_box_y2_norm],
        )
        if all(np.isfinite([row.pred_x1, row.pred_y1, row.pred_x2, row.pred_y2]))
        else 0.0
        for row in positive.itertuples()
    ]
    rows = rows.merge(positive[["image_id", "bbox_iou_to_gt"]], on="image_id", how="left")

    sensitivity = float((positive["predicted_defect_present"] == True).mean())  # noqa: E712
    specificity = float((negative["predicted_defect_present"] == False).mean())  # noqa: E712
    bbox_coverage = float(positive[["pred_x1", "pred_y1", "pred_x2", "pred_y2"]].notna().all(axis=1).mean())
    metric_values = {
        "views": len(rows),
        "positive_views": len(positive),
        "background_views": len(negative),
        "json_parse_rate": float(rows["json_parse_ok"].mean()),
        "schema_compliance_rate": float(rows["schema_compliant"].mean()),
        "positive_sensitivity": sensitivity,
        "background_specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "positive_bbox_coverage": bbox_coverage,
        "positive_mean_bbox_iou": float(positive["bbox_iou_to_gt"].mean()),
        "positive_median_bbox_iou": float(positive["bbox_iou_to_gt"].median()),
        "background_false_positive_rate": float((negative["predicted_defect_present"] == True).mean()),  # noqa: E712
    }
    checks = [
        ("json_parse_rate_at_least_0p90", metric_values["json_parse_rate"] >= 0.90),
        ("schema_compliance_rate_at_least_0p90", metric_values["schema_compliance_rate"] >= 0.90),
        ("positive_sensitivity_at_least_0p70", sensitivity >= 0.70),
        ("background_specificity_at_least_0p70", specificity >= 0.70),
        ("balanced_accuracy_at_least_0p70", metric_values["balanced_accuracy"] >= 0.70),
        ("positive_bbox_coverage_at_least_0p70", bbox_coverage >= 0.70),
        ("positive_median_bbox_iou_at_least_0p10", metric_values["positive_median_bbox_iou"] >= 0.10),
    ]
    metrics = pd.DataFrame([{"metric": key, "value": value} for key, value in metric_values.items()])
    gate = pd.DataFrame([{"check": key, "result": "PASS" if ok else "FAIL"} for key, ok in checks])
    passed = bool((gate["result"] == "PASS").all())

    scored = rows[
        rows["predicted_defect_present"].notna()
        & pd.to_numeric(rows["confidence"], errors="coerce").notna()
    ].copy()
    scored["defect_score_posthoc"] = np.where(
        scored["predicted_defect_present"].astype(bool),
        scored["confidence"].astype(float),
        1.0 - scored["confidence"].astype(float),
    )
    if scored["expected_defect_present"].nunique() == 2:
        ranking_auc = float(
            roc_auc_score(scored["expected_defect_present"].astype(int), scored["defect_score_posthoc"])
        )
        ranking_ap = float(
            average_precision_score(scored["expected_defect_present"].astype(int), scored["defect_score_posthoc"])
        )
    else:
        ranking_auc = ranking_ap = np.nan
    pair_orders = []
    for _, pair in scored.groupby("pair_id"):
        pos_scores = pair.loc[pair["expected_defect_present"], "defect_score_posthoc"]
        neg_scores = pair.loc[~pair["expected_defect_present"], "defect_score_posthoc"]
        if len(pos_scores) != 1 or len(neg_scores) != 1:
            continue
        pos_score, neg_score = float(pos_scores.iloc[0]), float(neg_scores.iloc[0])
        pair_orders.append(1.0 if pos_score > neg_score else 0.5 if pos_score == neg_score else 0.0)
    exploratory = pd.DataFrame(
        [
            {"metric": "posthoc_defect_score_auroc", "value": ranking_auc},
            {"metric": "posthoc_defect_score_average_precision", "value": ranking_ap},
            {
                "metric": "posthoc_matched_pair_order_accuracy_with_half_credit_for_ties",
                "value": float(np.mean(pair_orders)) if pair_orders else np.nan,
            },
            {
                "metric": "posthoc_unique_defect_score_count",
                "value": int(scored["defect_score_posthoc"].nunique()),
            },
        ]
    )
    if passed:
        decision = f"PASS: {model_label} separates matched defect/background views and grounds positives. A GT-free blind-tiling pilot may be planned, but this diagnostic output itself remains forbidden for acquisition."
    elif metric_values["schema_compliance_rate"] < 0.90:
        decision = f"FAIL: {model_label} is structurally noncompliant. Retire it from structured acquisition; do not run blind tiling."
    elif sensitivity < 0.70 or specificity < 0.70:
        decision = f"FAIL: structured output works, but {model_label} visual discrimination is inadequate. Retire it from acquisition; do not run blind tiling."
    else:
        decision = "FAIL: image-level discrimination may exist, but localization is inadequate. Do not use bbox/groundedness consistency or run blind tiling as an acquisition signal."

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "paired_compliance_rows.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output_dir / "paired_compliance_metrics.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(args.output_dir / "paired_compliance_gate.csv", index=False, encoding="utf-8-sig")
    exploratory.to_csv(args.output_dir / "paired_compliance_posthoc_ranking.csv", index=False, encoding="utf-8-sig")
    summary = [
        "# Paired Oracle/Background VLM Compliance Probe",
        "",
        "- GT used to construct/evaluate views: **True (post-hoc diagnostic only)**",
        "- Allowed for acquisition: **False**",
        "- Detector training performed: **False**",
        "- Final test evaluated: **False**",
        f"- Model: **{model_label}**",
        f"- Frozen gate: **{'PASS' if passed else 'FAIL'}**",
        "",
        "## Metrics", "", metrics.to_markdown(index=False), "",
        "## Frozen gate", "", gate.to_markdown(index=False), "",
        "## Exploratory ranking audit (not part of frozen gate)", "",
        "The score is confidence for predicted positives and 1-confidence for predicted negatives. Ties are common and receive half credit in matched-pair ordering.", "",
        exploratory.to_markdown(index=False), "",
        "## Decision", "", decision, "",
    ]
    summary_path = args.output_dir / "paired_compliance_summary.md"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print(f"[DONE] views={len(rows)}")
    print(f"[GATE] {'PASS' if passed else 'FAIL'}")
    print(f"[SUMMARY] {summary_path}")


if __name__ == "__main__":
    main()

"""Evaluate the GT-oracle crop VLM upper-bound diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def strip_fenced_json(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def semantic_response_status(raw_response: str) -> tuple[str, str]:
    """Exploratory lexical audit; deliberately excluded from the frozen gate."""
    try:
        parsed = json.loads(strip_fenced_json(raw_response))
    except (TypeError, json.JSONDecodeError):
        return "unparseable", ""
    evidence = " ".join(
        str(parsed.get(key) or "") for key in ("appearance", "visual_evidence")
    ).strip()
    normalized = evidence.casefold()
    negative_patterns = (
        r"\bno visible defects?\b",
        r"\bno visual defects?\b",
        r"\bno defects? present\b",
        r"\bdoes not show any (?:visible )?defects?\b",
        r"\bdoes not show any defects?\b",
    )
    positive_patterns = (
        r"\bdefects?\b",
        r"\bcracks?\b",
        r"\bcrevices?\b",
        r"\bholes?\b",
        r"\btears?\b",
        r"\bcreases?\b",
        r"\bdark spots?\b",
        r"\birregularly shaped\b",
    )
    if any(re.search(pattern, normalized) for pattern in negative_patterns):
        return "explicit_no_defect", evidence
    if any(re.search(pattern, normalized) for pattern in positive_patterns):
        return "defect_mention", evidence
    return "ambiguous", evidence


def load_semantic_responses(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            status, evidence = semantic_response_status(str(row.get("raw_response", "")))
            records.append(
                {
                    "image_id": str(row.get("image_id", "")),
                    "semantic_status_posthoc": status,
                    "semantic_evidence_posthoc": evidence,
                }
            )
    return pd.DataFrame(records)


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--crop-audit", type=Path, required=True)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    signals = pd.read_csv(args.signals)
    audit = pd.read_csv(args.crop_audit)
    audit = audit[audit["status"] == "created"].copy()
    rows = signals.merge(audit, left_on="image_id", right_on="view_id", how="inner")
    if rows.empty:
        raise RuntimeError("No oracle-crop signal rows matched the crop audit")
    if args.responses is not None:
        semantics = load_semantic_responses(args.responses)
        rows = rows.merge(semantics, on="image_id", how="left")

    ious = []
    for _, row in rows.iterrows():
        pred = [
            row.get("consensus_bbox_x1"), row.get("consensus_bbox_y1"),
            row.get("consensus_bbox_x2"), row.get("consensus_bbox_y2"),
        ]
        gt = [
            row["gt_box_x1_in_crop_norm"], row["gt_box_y1_in_crop_norm"],
            row["gt_box_x2_in_crop_norm"], row["gt_box_y2_in_crop_norm"],
        ]
        try:
            pred = [float(x) for x in pred]
            iou = box_iou(pred, [float(x) for x in gt]) if all(np.isfinite(pred)) else 0.0
        except (TypeError, ValueError):
            iou = 0.0
        ious.append(iou)
    rows["bbox_iou_to_oracle_gt"] = ious

    metric_values = {
        "images": len(rows),
        "mean_json_parse_rate": rows["valid_response_rate"].mean(),
        "mean_schema_complete_rate": rows["schema_complete_response_rate"].mean(),
        "mean_informative_response_rate": rows["informative_response_rate"].mean(),
        "mean_positive_presence_rate": rows["positive_presence_response_rate"].mean(),
        "mean_defect_evidence_rate": rows["defect_evidence_response_rate"].mean(),
        "mean_class_available_rate": rows["class_available_response_rate"].mean(),
        "bbox_response_coverage": (rows["box_response_count"] > 0).mean(),
        "mean_bbox_iou_to_oracle_gt": rows["bbox_iou_to_oracle_gt"].mean(),
        "median_bbox_iou_to_oracle_gt": rows["bbox_iou_to_oracle_gt"].median(),
    }
    metrics = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metric_values.items()]
    )
    checks = [
        ("json_parse_rate_at_least_0p90", metric_values["mean_json_parse_rate"] >= 0.90),
        ("informative_rate_at_least_0p90", metric_values["mean_informative_response_rate"] >= 0.90),
        ("positive_presence_rate_at_least_0p50", metric_values["mean_positive_presence_rate"] >= 0.50),
        ("defect_evidence_rate_at_least_0p70", metric_values["mean_defect_evidence_rate"] >= 0.70),
    ]
    gate = pd.DataFrame(
        [{"check": name, "result": "PASS" if bool(ok) else "FAIL"} for name, ok in checks]
    )
    visual_gate_pass = bool((gate["result"] == "PASS").all())
    exploratory = None
    if "semantic_status_posthoc" in rows:
        counts = rows["semantic_status_posthoc"].fillna("missing").value_counts()
        exploratory = pd.DataFrame(
            [
                {
                    "metric": "lexical_defect_mention_rate",
                    "value": counts.get("defect_mention", 0) / len(rows),
                },
                {
                    "metric": "lexical_explicit_no_defect_rate",
                    "value": counts.get("explicit_no_defect", 0) / len(rows),
                },
                {
                    "metric": "lexical_ambiguous_or_unparseable_rate",
                    "value": (
                        counts.get("ambiguous", 0)
                        + counts.get("unparseable", 0)
                        + counts.get("missing", 0)
                    )
                    / len(rows),
                },
            ]
        )
    if visual_gate_pass:
        decision = (
            "Oracle crops make the defects visible to the VLM. The full-image failure is "
            "consistent with resolution/attention loss, so a GT-free blind-tiling pilot is authorized."
        )
    else:
        decision = (
            "The frozen Qwen2-VL-2B structured pipeline fails even on focused oracle crops, "
            "so it is not valid as an acquisition signal and blind tiling is not justified. "
            "This gate alone does not distinguish visual blindness from structured-output "
            "noncompliance; any lexical audit below is exploratory and cannot reverse the gate."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "oracle_crop_diagnostic_rows.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output_dir / "oracle_crop_diagnostic_metrics.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(args.output_dir / "oracle_crop_diagnostic_gate.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Oracle-Crop VLM Visual Upper-Bound Diagnostic",
        "",
        "- GT used to construct crops: **True (post-hoc diagnostic only)**",
        "- Allowed for acquisition: **False**",
        "- Detector training performed: **False**",
        "- Final test evaluated: **False**",
        f"- Visual upper-bound gate: **{'PASS' if visual_gate_pass else 'FAIL'}**",
        "",
        "## Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Gate",
        "",
        gate.to_markdown(index=False),
        "",
    ]
    if exploratory is not None:
        lines.extend(
            [
                "## Exploratory response-semantics audit (not part of gate)",
                "",
                "This lexical audit uses only appearance and visual_evidence text. It is post-hoc, "
                "does not establish correct localization or class recognition, and cannot change the frozen gate.",
                "",
                exploratory.to_markdown(index=False),
                "",
            ]
        )
    lines.extend([
        "## Decision",
        "",
        decision,
        "",
    ])
    (args.output_dir / "oracle_crop_diagnostic_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"[DONE] images={len(rows)}")
    print(f"[GATE] {'PASS' if visual_gate_pass else 'FAIL'}")
    print(f"[SUMMARY] {args.output_dir / 'oracle_crop_diagnostic_summary.md'}")


if __name__ == "__main__":
    main()

"""Convert structured multi-prompt VLM responses into field-level signals."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ALLOWED_ZONES = {
    "top-left", "top-center", "top-right", "center-left", "center",
    "center-right", "bottom-left", "bottom-center", "bottom-right", "unknown",
}
ALLOWED_SCALES = {"micro", "small", "large", "unknown"}


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start : end + 1])
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def normalize_response(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "valid": False,
            "schema_complete": False,
            "informative": False,
            "bbox_norm": None,
            "abstain": None,
        }
    bbox = value.get("bbox_norm")
    valid_bbox = None
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            coords = [float(x) for x in bbox]
            if all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in coords):
                x1, y1, x2, y2 = coords
                if x2 > x1 and y2 > y1:
                    valid_bbox = coords
        except (TypeError, ValueError):
            pass
    raw_zone = value.get("location_zone")
    raw_scale = value.get("scale")
    raw_defect_type = value.get("defect_type")
    zone = str(raw_zone).strip().lower() if raw_zone is not None else "unknown"
    scale = str(raw_scale).strip().lower() if raw_scale is not None else "unknown"
    defect_type = (
        str(raw_defect_type).strip().lower() if raw_defect_type is not None else "unknown"
    ) or "unknown"
    if defect_type in {"none", "null", "n/a"}:
        defect_type = "unknown"
    present = value.get("defect_present")
    if present not in (True, False, None):
        present = None
    raw_abstain = value.get("abstain")
    abstain = raw_abstain if isinstance(raw_abstain, bool) else None
    confidence = value.get("confidence")
    try:
        confidence = float(confidence)
        confidence = confidence if 0.0 <= confidence <= 1.0 else np.nan
    except (TypeError, ValueError):
        confidence = np.nan
    appearance = str(value.get("appearance") or "").strip()
    visual_evidence = str(value.get("visual_evidence") or "").strip()
    schema_complete = isinstance(present, bool) and isinstance(abstain, bool) and np.isfinite(confidence)
    informative = bool(
        isinstance(present, bool)
        or defect_type != "unknown"
        or valid_bbox is not None
        or (zone if zone in ALLOWED_ZONES else "unknown") != "unknown"
        or (scale if scale in ALLOWED_SCALES else "unknown") != "unknown"
        or appearance
        or visual_evidence
    )
    return {
        "valid": True,
        "schema_complete": schema_complete,
        "informative": informative,
        "defect_present": present,
        "defect_type": defect_type,
        "bbox_norm": valid_bbox,
        "location_zone": zone if zone in ALLOWED_ZONES else "unknown",
        "scale": scale if scale in ALLOWED_SCALES else "unknown",
        "appearance": appearance,
        "visual_evidence": visual_evidence,
        "abstain": abstain,
        "confidence": confidence,
    }


def agreement(values: list[Any], ignore: set[Any] | None = None) -> float:
    ignore = ignore or set()
    kept = [value for value in values if value not in ignore and value is not None]
    if not kept:
        return float("nan")
    return float(Counter(kept).most_common(1)[0][1] / len(kept))


def mode_value(values: list[Any], ignore: set[Any] | None = None) -> Any:
    ignore = ignore or set()
    kept = [value for value in values if value not in ignore and value is not None]
    return Counter(kept).most_common(1)[0][0] if kept else None


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def mean_pairwise_box_iou(boxes: list[list[float]]) -> float:
    values = [box_iou(boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))]
    return float(np.mean(values)) if values else float("nan")


def score_image(records: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [normalize_response(parse_json_object(r.get("raw_response", ""))) for r in records]
    valid = [r for r in parsed if r["valid"]]
    boxes = [r["bbox_norm"] for r in valid if r.get("bbox_norm") is not None]
    consensus_box = np.median(np.asarray(boxes, dtype=float), axis=0).tolist() if boxes else None
    component_values = {
        "presence_agreement": agreement([r.get("defect_present") for r in valid]),
        "class_agreement": agreement([r.get("defect_type") for r in valid], {"unknown"}),
        "location_agreement": agreement([r.get("location_zone") for r in valid], {"unknown"}),
        "scale_agreement": agreement([r.get("scale") for r in valid], {"unknown"}),
        "bbox_iou_agreement": mean_pairwise_box_iou(boxes),
    }
    finite_components = [v for v in component_values.values() if np.isfinite(v)]
    structured_consistency = float(np.mean(finite_components)) if finite_components else np.nan
    confidences = [float(r["confidence"]) for r in valid if np.isfinite(r.get("confidence", np.nan))]
    positive_presence_count = sum(r.get("defect_present") is True for r in valid)
    negative_presence_count = sum(r.get("defect_present") is False for r in valid)
    class_available_count = sum(r.get("defect_type") not in {None, "unknown"} for r in valid)
    defect_evidence_count = sum(
        bool(
            r.get("defect_present") is True
            or r.get("defect_type") not in {None, "unknown"}
            or r.get("bbox_norm") is not None
        )
        for r in valid
    )
    result = {
        "image_id": records[0].get("image_id"),
        "image_path": records[0].get("image_path"),
        "dataset": records[0].get("dataset"),
        "split_role": records[0].get("split_role"),
        "num_prompt_records": len(records),
        "valid_response_count": len(valid),
        "valid_response_rate": len(valid) / len(records) if records else 0.0,
        "schema_complete_response_count": sum(bool(r.get("schema_complete")) for r in parsed),
        "schema_complete_response_rate": float(np.mean([bool(r.get("schema_complete")) for r in parsed])) if parsed else 0.0,
        "informative_response_count": sum(bool(r.get("informative")) for r in parsed),
        "informative_response_rate": float(np.mean([bool(r.get("informative")) for r in parsed])) if parsed else 0.0,
        "box_response_count": len(boxes),
        "positive_presence_response_rate": positive_presence_count / len(records) if records else 0.0,
        "negative_presence_response_rate": negative_presence_count / len(records) if records else 0.0,
        "unknown_presence_response_rate": 1.0 - ((positive_presence_count + negative_presence_count) / len(records)) if records else 1.0,
        "class_available_response_rate": class_available_count / len(records) if records else 0.0,
        "defect_evidence_response_rate": defect_evidence_count / len(records) if records else 0.0,
        "consensus_defect_present": mode_value([r.get("defect_present") for r in valid]),
        "consensus_defect_type": mode_value([r.get("defect_type") for r in valid], {"unknown"}),
        "consensus_location_zone": mode_value([r.get("location_zone") for r in valid], {"unknown"}),
        "consensus_scale": mode_value([r.get("scale") for r in valid], {"unknown"}),
        "consensus_bbox_x1": consensus_box[0] if consensus_box else np.nan,
        "consensus_bbox_y1": consensus_box[1] if consensus_box else np.nan,
        "consensus_bbox_x2": consensus_box[2] if consensus_box else np.nan,
        "consensus_bbox_y2": consensus_box[3] if consensus_box else np.nan,
        "abstain_rate": float(np.mean([r["abstain"] for r in valid if isinstance(r.get("abstain"), bool)])) if any(isinstance(r.get("abstain"), bool) for r in valid) else np.nan,
        "mean_confidence": float(np.mean(confidences)) if confidences else np.nan,
        "structured_consistency": structured_consistency,
        "structured_inconsistency": 1.0 - structured_consistency if np.isfinite(structured_consistency) else np.nan,
        **component_values,
    }
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {line_number} is not an object")
                rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_jsonl(args.responses)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        image_id = str(record.get("image_id", "")).strip()
        if not image_id:
            raise ValueError("Every response record requires image_id")
        grouped.setdefault(image_id, []).append(record)
    scores = pd.DataFrame(score_image(group) for group in grouped.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[DONE] images={len(scores)} responses={len(records)}")
    print(f"[OUTPUT] {args.output}")


if __name__ == "__main__":
    main()

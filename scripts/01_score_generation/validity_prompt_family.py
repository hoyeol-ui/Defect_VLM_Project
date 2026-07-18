"""Frozen structured prompt family for VLM signal-validity experiments."""

from __future__ import annotations

import hashlib
import json
from typing import Any


SCHEMA_INSTRUCTION = """Return JSON only with exactly these fields:
{
  "defect_present": true | false | null,
  "defect_type": "short visual defect name or unknown",
  "bbox_norm": [x1, y1, x2, y2] | null,
  "location_zone": "top-left|top-center|top-right|center-left|center|center-right|bottom-left|bottom-center|bottom-right|unknown",
  "scale": "micro|small|large|unknown",
  "appearance": "short shape/texture/boundary description",
  "visual_evidence": "what visible evidence supports the answer",
  "abstain": true | false,
  "confidence": 0.0
}
Coordinates must be between 0 and 1 relative to the whole image. Do not infer a
class from a filename, folder, metadata, or prior label. If the image evidence
is insufficient, set abstain=true and use null/unknown instead of guessing."""


_BASE_REQUESTS = [
    "Inspect this industrial surface image for the single most important visible defect.",
    "Act as a cautious manufacturing inspector and assess the dominant visible defect in this image.",
    "Examine only the pixels in this industrial image and report the primary defect, if one is visible.",
    "Determine whether a dominant surface defect is visually supported and localize it conservatively.",
    "Review this image for annotation triage: identify and localize the clearest defect without guessing.",
]


STRUCTURED_VALIDITY_PROMPT_FAMILY: dict[str, Any] = {
    "id": "structured_defect_validity_v1",
    "name": "structured_defect_validity",
    "version": "v1",
    "is_gt_free": True,
    "purpose": "prompt-sensitivity and groundedness validity; not an acquisition score",
    "prompts": [f"{request}\n\n{SCHEMA_INSTRUCTION}" for request in _BASE_REQUESTS],
}


def prompt_family_hash() -> str:
    payload = json.dumps(
        STRUCTURED_VALIDITY_PROMPT_FAMILY,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_records() -> list[dict[str, str]]:
    family = STRUCTURED_VALIDITY_PROMPT_FAMILY
    return [
        {
            "prompt_id": f"{family['id']}_p{index + 1}",
            "prompt_family_id": str(family["id"]),
            "prompt_family_hash": prompt_family_hash(),
            "prompt_text": text,
        }
        for index, text in enumerate(family["prompts"])
    ]


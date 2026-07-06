"""
Expert-inspired prompt families for VLM-based industrial defect description.

This module promotes prompt design from hidden string literals to a versioned
research component. The current `expert_defect_v1` family encodes industrial
surface-defect inspection axes such as location, scale, and appearance. Future
expert-reviewed v2/v3 families can be compared under the same active-learning
protocol without changing historical strategy IDs or result schemas.

The prompt family itself is GT-free: prompts can be used to elicit VLM
descriptions before any ground-truth labels or boxes are observed.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping


EXPERT_DEFECT_PROMPT_FAMILY: dict[str, Any] = {
    "name": "expert_defect",
    "version": "v1",
    "id": "expert_defect_v1",
    "description": (
        "Expert-inspired prompt family for industrial surface defect "
        "description. It separates location, scale, and appearance axes so "
        "that VLM explanation consistency can be used as a GT-free active "
        "learning acquisition signal."
    ),
    "prompt_keys": ["location", "scale", "appearance"],
    "intended_use": (
        "VLM multi-prompt explanation consistency for industrial defect "
        "active learning"
    ),
    "is_gt_free": True,
    "prompts": {
        "location": (
            "Identify the main defect in this image. Describe its location using standard spatial terms "
            "(e.g., top-left, top-center, top-right, center-left, center, center-right, bottom-left, bottom-center, bottom-right). "
            "Your response must include a phrase like: 'The defect is located in the [spatial term] region.'"
        ),
        "scale": (
            "Analyze the size of the defect relative to the entire image area. "
            "Categorize its scale as either 'micro' (tiny point), 'small' (localized region), or 'large' (covering significant portion). "
            "Your response must include a phrase like: 'The scale of the defect is [scale category].'"
        ),
        "appearance": (
            "As an industrial quality inspector, examine this image. "
            "Describe the shape, texture, and visual boundary of the defect in detail. Keep it brief."
        ),
    },
}


_PROMPT_FAMILIES = {
    EXPERT_DEFECT_PROMPT_FAMILY["name"]: EXPERT_DEFECT_PROMPT_FAMILY,
    EXPERT_DEFECT_PROMPT_FAMILY["id"]: EXPERT_DEFECT_PROMPT_FAMILY,
}


def compute_prompt_family_hash(prompt_family: Mapping[str, Any]) -> str:
    """Return a stable short hash over prompt keys and prompt text."""
    prompts = prompt_family.get("prompts", {})
    payload = {
        "id": prompt_family.get("id"),
        "prompt_keys": list(prompt_family.get("prompt_keys", [])),
        "prompts": {k: prompts[k] for k in sorted(prompts)},
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def get_prompt_family(name_or_id: str = "expert_defect_v1") -> dict[str, Any]:
    """Fetch a prompt family by name or id without exposing mutable globals."""
    if name_or_id not in _PROMPT_FAMILIES:
        available = ", ".join(sorted(_PROMPT_FAMILIES))
        raise KeyError(f"Unknown prompt family: {name_or_id}. Available: {available}")
    return deepcopy(_PROMPT_FAMILIES[name_or_id])


def get_prompt_family_metadata(prompt_family: Mapping[str, Any]) -> dict[str, Any]:
    """Return flat metadata suitable for CSV/JSON logging."""
    return {
        "prompt_family_id": prompt_family.get("id"),
        "prompt_family_name": prompt_family.get("name"),
        "prompt_family_version": prompt_family.get("version"),
        "prompt_family_hash": compute_prompt_family_hash(prompt_family),
        "prompt_keys": "|".join(prompt_family.get("prompt_keys", [])),
        "prompt_family_intended_use": prompt_family.get("intended_use"),
        "prompt_family_is_gt_free": bool(prompt_family.get("is_gt_free", False)),
    }

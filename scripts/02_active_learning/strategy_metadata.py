"""
Research-facing metadata for active-learning strategy IDs.

The strategy IDs are intentionally kept stable for compatibility with existing
CSV files, plots, and lab-meeting materials. Display names and roles can evolve
without breaking historical experiment traces.
"""

from __future__ import annotations

from typing import Any


STRATEGY_METADATA: dict[str, dict[str, str]] = {
    "Random": {
        "display_name": "Random",
        "family": "Baseline",
        "role": "Random sampling baseline",
    },
    "ConsistencyOnly": {
        "display_name": "ExpertPromptConsistency",
        "family": "Core hypothesis",
        "role": "Tests VLM prompt-family inconsistency as a GT-free acquisition signal",
    },
    "GroundednessOnlySoft": {
        "display_name": "PseudoGroundingOnly",
        "family": "Auxiliary signal",
        "role": "Tests weak pseudo visual evidence alone",
    },
    "CombinedSoftPenalty": {
        "display_name": "Consistency + AuxGrounding",
        "family": "Auxiliary extension",
        "role": "Naive combination of explanation inconsistency and pseudo groundedness",
    },
    "LowPrioritySoft": {
        "display_name": "ReverseDirectionControl",
        "family": "Diagnostic control",
        "role": "Tests whether the high-priority score direction is valid",
    },
    "CombinedSuppressNoPseudo": {
        "display_name": "Calibrated Consistency + AuxGrounding",
        "family": "Calibration / failure analysis",
        "role": "Suppresses no_pseudo_box over-selection from the auxiliary visual signal",
    },
    "RandomClassBalanced": {
        "display_name": "Class-balanced Random",
        "family": "Baseline",
        "role": "Class-balanced random sampling baseline",
    },
    "CombinedNoPenalty": {
        "display_name": "Consistency + AuxGrounding NoPenalty",
        "family": "Auxiliary extension",
        "role": "Combined score without missing-box bonus",
    },
    "CombinedNoGroundedness": {
        "display_name": "Consistency Only via Combined Path",
        "family": "Core hypothesis",
        "role": "Checks consistency-only behavior through the combined-score pipeline",
    },
    "CombinedWeighted": {
        "display_name": "Weighted Consistency + AuxGrounding",
        "family": "Auxiliary extension",
        "role": "Weighted combination of consistency and pseudo-grounding uncertainty",
    },
    "CombinedRankCalibrated": {
        "display_name": "Rank-calibrated Consistency + AuxGrounding",
        "family": "Calibration / failure analysis",
        "role": "Rank-calibrated score combination for direction diagnostics",
    },
    "CombinedSoftPenaltyClassBalanced": {
        "display_name": "Class-balanced Consistency + AuxGrounding",
        "family": "Auxiliary extension",
        "role": "Class-balanced selection with original combined score",
    },
    "CombinedSuppressNoPseudoClassBalanced": {
        "display_name": "Class-balanced Calibrated AuxGrounding",
        "family": "Calibration / failure analysis",
        "role": "Class-balanced no_pseudo_box suppression diagnostic",
    },
    "LowPrioritySoftClassBalanced": {
        "display_name": "Class-balanced ReverseDirectionControl",
        "family": "Diagnostic control",
        "role": "Class-balanced reverse-direction diagnostic control",
    },
}


def get_strategy_metadata(strategy: str) -> dict[str, str]:
    """Return stable metadata for a strategy ID, with safe fallback."""
    return STRATEGY_METADATA.get(
        strategy,
        {
            "display_name": strategy,
            "family": "Uncategorized",
            "role": "No strategy metadata registered",
        },
    )


def add_strategy_metadata_columns(df: Any, strategy_col: str = "strategy") -> Any:
    """Append display_name/family/role columns to a pandas DataFrame."""
    if df is None or len(df) == 0 or strategy_col not in df.columns:
        return df
    out = df.copy()
    out["strategy_display_name"] = out[strategy_col].map(
        lambda s: get_strategy_metadata(str(s))["display_name"]
    )
    out["strategy_family"] = out[strategy_col].map(
        lambda s: get_strategy_metadata(str(s))["family"]
    )
    out["strategy_role"] = out[strategy_col].map(
        lambda s: get_strategy_metadata(str(s))["role"]
    )
    return out

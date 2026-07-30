"""V10c-PDF recall-guard one-cycle runner.

This is a local/internal experiment runner derived from the proposal PDF:

    "V10b의 굴레를 깨기 위한 V10c와 V10d 연구 제안.pdf"

It intentionally does not replace the committed V10c runner.  Instead, it
imports the V10c infrastructure and swaps only the query selector with a more
aggressive recall-guard mixture:

    - query size: 30
    - V10b-like core quota: 21
    - recall-guard quota: 9
    - no-box / zero-box quota: up to 3
    - one-box quota: up to 6
    - guard pool: pseudo boxes <= 1, uncertainty >= q70, DINO distance >= q50

No final-test evaluation. No GT oracle acquisition.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_v10c_recall_guard_onecycle as v10c  # noqa: E402


PDF_STRATEGY = "DetectorPdfRecallGuardV10c"
PDF_RUNS_ROOT = v10c.PROJECT_ROOT / "runs" / "active_learning_v10c_pdf_recall_guard_onecycle"
PDF_DATASETS_ROOT = v10c.PROJECT_ROOT / "datasets" / "active_learning_v10c_pdf_recall_guard_onecycle"

PDF_CORE_WEIGHTS = {
    # Keep the V10b core semantics rather than the lighter instance weight used
    # by the first V10c local runner.
    "detector_uncertainty": 0.25,
    "dino_visual_distance": 0.35,
    "predicted_class_deficit": 0.15,
    "pseudo_instance_count": 0.25,
}

PDF_GUARD_WEIGHTS = {
    "detector_uncertainty": 0.45,
    "dino_visual_distance": 0.30,
    "predicted_class_deficit": 0.15,
    "low_confidence": 0.10,
}


def int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        raise ValueError(f"Invalid weights: {weights}")
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def read_pdf_core_weights() -> dict[str, float]:
    return normalize_weights(
        {
            "detector_uncertainty": float_env("AL_V10C_PDF_CORE_W_UNCERTAINTY", PDF_CORE_WEIGHTS["detector_uncertainty"]),
            "dino_visual_distance": float_env("AL_V10C_PDF_CORE_W_DINO", PDF_CORE_WEIGHTS["dino_visual_distance"]),
            "predicted_class_deficit": float_env("AL_V10C_PDF_CORE_W_BALANCE", PDF_CORE_WEIGHTS["predicted_class_deficit"]),
            "pseudo_instance_count": float_env("AL_V10C_PDF_CORE_W_INSTANCE", PDF_CORE_WEIGHTS["pseudo_instance_count"]),
        }
    )


def read_pdf_guard_weights() -> dict[str, float]:
    return normalize_weights(
        {
            "detector_uncertainty": float_env("AL_V10C_PDF_GUARD_W_UNCERTAINTY", PDF_GUARD_WEIGHTS["detector_uncertainty"]),
            "dino_visual_distance": float_env("AL_V10C_PDF_GUARD_W_DINO", PDF_GUARD_WEIGHTS["dino_visual_distance"]),
            "predicted_class_deficit": float_env("AL_V10C_PDF_GUARD_W_BALANCE", PDF_GUARD_WEIGHTS["predicted_class_deficit"]),
            "low_confidence": float_env("AL_V10C_PDF_GUARD_W_LOW_CONF", PDF_GUARD_WEIGHTS["low_confidence"]),
        }
    )


def _sample_id_set(df: pd.DataFrame) -> set[str]:
    if df.empty or "sample_id" not in df.columns:
        return set()
    return set(df["sample_id"].astype(str))


def _prepare_scored(scored_pool: pd.DataFrame) -> pd.DataFrame:
    scored = scored_pool.copy()
    if "detector_pred_class" not in scored.columns:
        scored["detector_pred_class"] = "__unknown__"
    if "detector_no_box" not in scored.columns:
        scored["detector_no_box"] = scored["detector_pred_class"].astype(str).eq("__no_box__").astype(float)
    scored["detector_no_box"] = pd.to_numeric(scored["detector_no_box"], errors="coerce").fillna(0.0)
    scored["detector_pseudo_box_count"] = pd.to_numeric(
        scored.get("detector_pseudo_box_count", 0), errors="coerce"
    ).fillna(0)
    scored["detector_max_conf"] = pd.to_numeric(scored.get("detector_max_conf", 0), errors="coerce").fillna(0.0)
    scored["detector_uncertainty"] = pd.to_numeric(scored.get("detector_uncertainty", 0), errors="coerce").fillna(0.0)
    return scored


def _class_universe(scored_pool: pd.DataFrame) -> list[str]:
    class_universe = sorted(str(v) for v in getattr(v10c.base.v6, "NEU_CLASSES", []) if str(v))
    if class_universe:
        return class_universe
    return sorted(
        c
        for c in scored_pool["detector_pred_class"].dropna().astype(str).unique().tolist()
        if c != "__no_box__"
    )


def _guard_scored_pool(
    pool: pd.DataFrame,
    *,
    initial_df: pd.DataFrame,
    selected_ids: list[str],
    selected_pred_counts: Counter,
    embedding_lookup: dict[str, np.ndarray],
    guard_weights: dict[str, float],
) -> pd.DataFrame:
    out = _prepare_scored(pool)
    refs = initial_df["sample_id"].astype(str).tolist() + selected_ids
    class_universe = _class_universe(out)
    labeled_counts = Counter(initial_df["class_hint"].dropna().astype(str))

    out["_pdf_dino_distance_raw"] = [
        v10c.base.detector_probe.min_cosine_distance(str(sid), refs, embedding_lookup)
        for sid in out["sample_id"].astype(str)
    ]
    out["_pdf_class_deficit_raw"] = [
        v10c.base.detector_probe.class_deficit_score(str(cls), labeled_counts, selected_pred_counts, class_universe)
        if str(cls) != "__no_box__"
        else 0.5
        for cls in out["detector_pred_class"].astype(str)
    ]

    out["_pdf_uncertainty_norm"] = v10c.base.detector_probe.normalize(out["detector_uncertainty"])
    out["_pdf_dino_distance_norm"] = v10c.base.detector_probe.normalize(out["_pdf_dino_distance_raw"])
    out["_pdf_class_deficit_norm"] = v10c.base.detector_probe.normalize(out["_pdf_class_deficit_raw"])
    out["_pdf_max_conf_norm"] = v10c.base.detector_probe.normalize(out["detector_max_conf"])
    out["_pdf_low_conf_norm"] = 1.0 - out["_pdf_max_conf_norm"]
    out["_pdf_guard_score"] = (
        guard_weights["detector_uncertainty"] * out["_pdf_uncertainty_norm"]
        + guard_weights["dino_visual_distance"] * out["_pdf_dino_distance_norm"]
        + guard_weights["predicted_class_deficit"] * out["_pdf_class_deficit_norm"]
        + guard_weights["low_confidence"] * out["_pdf_low_conf_norm"]
    )
    return out


def _append_picks(
    *,
    selected_parts: list[pd.DataFrame],
    picked: pd.DataFrame,
    phase: str,
    selected_ids: list[str],
    selected_pred_counts: Counter,
) -> None:
    if picked.empty:
        return
    picked = picked.copy()
    picked["v10c_phase"] = phase
    selected_parts.append(picked)
    for _, row in picked.iterrows():
        sid = str(row["sample_id"])
        if sid not in selected_ids:
            selected_ids.append(sid)
        pred_cls = str(row.get("detector_pred_class", ""))
        if pred_cls and pred_cls != "__no_box__":
            selected_pred_counts[pred_cls] += 1


def select_v10c_pdf_recall_guard(
    scored_pool: pd.DataFrame,
    initial_df: pd.DataFrame,
    embedding_lookup: dict[str, np.ndarray],
    *,
    query_size: int,
    candidate_fraction: float,
    core_size: int,
    recall_guard_size: int,
    v10c_weights: dict[str, float],
    guard_weights: dict[str, float],
    core_max_no_box: int,
    core_min_pseudo_boxes: int,
    core_max_per_pred_class: int,
    guard_max_no_box: int,
    guard_max_per_pred_class: int,
) -> pd.DataFrame:
    """PDF-proposal selector: V10b-like core + explicit no-box/one-box guard."""
    del guard_max_per_pred_class  # The PDF branch uses explicit no-box/one-box quotas instead.

    scored = _prepare_scored(scored_pool)
    query_size = int(query_size)
    core_size = max(0, min(query_size, int(core_size)))
    recall_guard_size = max(0, min(query_size - core_size, int(recall_guard_size)))

    no_box_quota = min(int_env("AL_V10C_PDF_NO_BOX_QUOTA", 3), recall_guard_size)
    one_box_quota = min(int_env("AL_V10C_PDF_ONE_BOX_QUOTA", 6), recall_guard_size - no_box_quota)
    uncertainty_q = float_env("AL_V10C_PDF_GUARD_UNCERTAINTY_Q", 0.70)
    dino_q = float_env("AL_V10C_PDF_GUARD_DINO_Q", 0.50)

    selected_parts: list[pd.DataFrame] = []
    selected_ids: list[str] = []
    selected_pred_counts: Counter = Counter()

    if core_size > 0:
        core = v10c.base.stage1.select_instance_rich_dino_balanced(
            scored,
            initial_df,
            embedding_lookup,
            query_size=core_size,
            candidate_fraction=candidate_fraction,
            weights=v10c_weights,
            max_no_box=core_max_no_box,
            min_pseudo_boxes=core_min_pseudo_boxes,
            max_per_pred_class=core_max_per_pred_class,
        )
        _append_picks(
            selected_parts=selected_parts,
            picked=core,
            phase="pdf_core_v10b_like",
            selected_ids=selected_ids,
            selected_pred_counts=selected_pred_counts,
        )

    remaining = scored[~scored["sample_id"].astype(str).isin(set(selected_ids))].copy()
    guard_scored = _guard_scored_pool(
        remaining,
        initial_df=initial_df,
        selected_ids=selected_ids,
        selected_pred_counts=selected_pred_counts,
        embedding_lookup=embedding_lookup,
        guard_weights=guard_weights,
    )

    if not guard_scored.empty:
        uncertainty_threshold = guard_scored["_pdf_uncertainty_norm"].quantile(uncertainty_q)
        dino_threshold = guard_scored["_pdf_dino_distance_norm"].quantile(dino_q)
        guard_pool = guard_scored[
            guard_scored["detector_pseudo_box_count"].le(1)
            & guard_scored["_pdf_uncertainty_norm"].ge(uncertainty_threshold)
            & guard_scored["_pdf_dino_distance_norm"].ge(dino_threshold)
        ].copy()
        if len(guard_pool) < recall_guard_size:
            guard_pool = guard_scored[guard_scored["detector_pseudo_box_count"].le(1)].copy()
        if len(guard_pool) < recall_guard_size:
            guard_pool = guard_scored.copy()
    else:
        guard_pool = guard_scored

    no_box_pool = guard_pool[
        guard_pool["detector_pred_class"].astype(str).eq("__no_box__")
        | guard_pool["detector_pseudo_box_count"].eq(0)
    ].copy()
    no_box_k = min(no_box_quota, guard_max_no_box, len(no_box_pool))
    no_box = no_box_pool.sort_values(
        ["_pdf_guard_score", "_pdf_dino_distance_raw", "sample_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).head(no_box_k)
    _append_picks(
        selected_parts=selected_parts,
        picked=no_box,
        phase="pdf_guard_no_box",
        selected_ids=selected_ids,
        selected_pred_counts=selected_pred_counts,
    )

    guard_pool = guard_pool[~guard_pool["sample_id"].astype(str).isin(set(selected_ids))].copy()
    one_box_pool = guard_pool[
        guard_pool["detector_pseudo_box_count"].eq(1)
        & guard_pool["detector_pred_class"].astype(str).ne("__no_box__")
    ].copy()
    one_box = one_box_pool.sort_values(
        ["_pdf_guard_score", "_pdf_dino_distance_raw", "sample_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).head(one_box_quota)
    _append_picks(
        selected_parts=selected_parts,
        picked=one_box,
        phase="pdf_guard_one_box",
        selected_ids=selected_ids,
        selected_pred_counts=selected_pred_counts,
    )

    selected = pd.concat(selected_parts, ignore_index=True, sort=False) if selected_parts else scored.iloc[0:0].copy()
    if len(selected) < query_size:
        fill_n = query_size - len(selected)
        fill_pool = guard_scored[~guard_scored["sample_id"].astype(str).isin(_sample_id_set(selected))].copy()
        fill = fill_pool.sort_values(
            ["_pdf_guard_score", "_pdf_dino_distance_raw", "sample_id"],
            ascending=[False, False, True],
            kind="mergesort",
        ).head(fill_n)
        _append_picks(
            selected_parts=selected_parts,
            picked=fill,
            phase="pdf_guard_fill",
            selected_ids=selected_ids,
            selected_pred_counts=selected_pred_counts,
        )
        selected = pd.concat(selected_parts, ignore_index=True, sort=False)

    if len(selected) < query_size:
        fill_n = query_size - len(selected)
        fallback_pool = scored[~scored["sample_id"].astype(str).isin(_sample_id_set(selected))].copy()
        fallback = v10c.add_v10c_component_scores(
            fallback_pool,
            initial_df=initial_df,
            selected_ids=selected["sample_id"].astype(str).tolist(),
            selected_pred_counts=selected_pred_counts,
            embedding_lookup=embedding_lookup,
            weights=v10c_weights,
        ).sort_values(
            ["_v10c_score", "_visual_distance_raw", "sample_id"],
            ascending=[False, False, True],
            kind="mergesort",
        ).head(fill_n)
        _append_picks(
            selected_parts=selected_parts,
            picked=fallback,
            phase="pdf_core_fallback",
            selected_ids=selected_ids,
            selected_pred_counts=selected_pred_counts,
        )
        selected = pd.concat(selected_parts, ignore_index=True, sort=False)

    selected = selected.drop_duplicates("sample_id", keep="first").head(query_size).reset_index(drop=True)
    selected["pdf_guard_uncertainty_q"] = uncertainty_q
    selected["pdf_guard_dino_q"] = dino_q
    selected["pdf_no_box_quota"] = no_box_quota
    selected["pdf_one_box_quota"] = one_box_quota
    return selected


_original_write_outputs = v10c.write_outputs


def write_outputs_with_pdf_config(save_dir: Path, config: dict[str, Any], accum: dict[str, list[Any]]) -> None:
    config["pdf_proposal_parameters"] = {
        "core_quota": int_env("AL_V10C_CORE_SIZE", 21),
        "recall_guard_quota": int_env("AL_V10C_RECALL_GUARD_SIZE", 9),
        "no_box_quota": int_env("AL_V10C_PDF_NO_BOX_QUOTA", 3),
        "one_box_quota": int_env("AL_V10C_PDF_ONE_BOX_QUOTA", 6),
        "guard_uncertainty_quantile": float_env("AL_V10C_PDF_GUARD_UNCERTAINTY_Q", 0.70),
        "guard_dino_quantile": float_env("AL_V10C_PDF_GUARD_DINO_Q", 0.50),
        "guard_score": "0.45*uncertainty + 0.30*dino_distance + 0.15*class_deficit + 0.10*(1-max_conf)",
    }
    _original_write_outputs(save_dir, config, accum)
    (save_dir / "v10c_pdf_proposal_parameters.json").write_text(
        json.dumps(config["pdf_proposal_parameters"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    os.environ.setdefault("AL_V10C_CORE_SIZE", "21")
    os.environ.setdefault("AL_V10C_RECALL_GUARD_SIZE", "9")
    os.environ.setdefault("AL_V10C_CORE_MAX_NO_BOX", "0")
    os.environ.setdefault("AL_V10C_CORE_MIN_PSEUDO_BOXES", "2")
    os.environ.setdefault("AL_V10C_CORE_MAX_PER_PRED_CLASS", "2")
    os.environ.setdefault("AL_V10C_GUARD_MAX_NO_BOX", "3")
    os.environ.setdefault("AL_V10C_PDF_NO_BOX_QUOTA", "3")
    os.environ.setdefault("AL_V10C_PDF_ONE_BOX_QUOTA", "6")
    os.environ.setdefault("AL_V10C_PDF_GUARD_UNCERTAINTY_Q", "0.70")
    os.environ.setdefault("AL_V10C_PDF_GUARD_DINO_Q", "0.50")

    v10c.V10C_STRATEGY = PDF_STRATEGY
    v10c.RUNS_ROOT = PDF_RUNS_ROOT
    v10c.DATASETS_ROOT = PDF_DATASETS_ROOT
    v10c.DEFAULT_V10C_WEIGHTS = dict(PDF_CORE_WEIGHTS)
    v10c.DEFAULT_RECALL_GUARD_WEIGHTS = {
        "detector_uncertainty": 0.45,
        "dino_visual_distance": 0.30,
        "predicted_class_deficit": 0.15,
        "low_coverage": 0.10,
    }
    v10c.read_v10c_weights = read_pdf_core_weights
    v10c.read_guard_weights = read_pdf_guard_weights
    v10c.select_v10c_recall_guard = select_v10c_pdf_recall_guard
    v10c.write_outputs = write_outputs_with_pdf_config
    v10c.main()


if __name__ == "__main__":
    main()


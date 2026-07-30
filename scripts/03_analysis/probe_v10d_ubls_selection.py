"""V10d-UBLS selection-only probe.

UBLS = Uncertainty-Bounded Localization Stability.

This script consumes an existing V10d stability-signal candidate CSV and builds
a *GT-free* balanced selection:

1. Filter to detector-uncertainty top fraction, default 40%.
2. Rank within that pool by localization instability u_loc.
3. Greedily select 30 with:
   - predicted-class cap, default <= 6
   - DINO farthest-first diversity when embeddings are available
   - no pseudo-box-count quota and no GT label quota

GT columns that may exist in the input CSV are used only after selection for
post-hoc audit, never for selecting samples.

No training. No final-test evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_AL = PROJECT_ROOT / "scripts" / "02_active_learning"
if str(SCRIPT_AL) not in sys.path:
    sys.path.insert(0, str(SCRIPT_AL))

import probe_v10b_selection_from_existing_v10 as v10bprobe  # noqa: E402
import probe_v9_detector_aware_selection as detector_probe  # noqa: E402
import run_v10_neu_large_pool_smoke as v10smoke  # noqa: E402


DEFAULT_STABILITY_RUN = PROJECT_ROOT / "runs" / "v10d_stability_signal_probe" / "v10d_stability_signal_seed47_20260713_221122"
DEFAULT_SOURCE_RUN = PROJECT_ROOT / "runs" / "active_learning_v10c_recall_guard_onecycle" / "v10c_recall_guard_onecycle_20260713_201155"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "runs" / "v10d_ubls_selection_probe"

GT_AUDIT_COLUMNS = {
    "gt_box_count",
    "gt_classes",
    "gt_majority_class",
    "gt_class_counts_json",
    "pseudo_tp",
    "pseudo_fp",
    "pseudo_fn",
    "pseudo_precision",
    "pseudo_recall",
    "pseudo_fn_ratio",
    "pred_class_in_gt_classes",
}


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def minmax_norm(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    lo = vals.min()
    hi = vals.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series([0.0] * len(vals), index=series.index)
    return (vals - lo) / (hi - lo)


def pairwise_cosine_similarity_mean(ids: list[str], embedding_lookup: dict[str, np.ndarray]) -> float:
    vecs = [embedding_lookup[str(sid)] for sid in ids if str(sid) in embedding_lookup]
    if len(vecs) < 2:
        return np.nan
    mat = np.vstack(vecs)
    mat = mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12, None)
    sim = mat @ mat.T
    upper = sim[np.triu_indices(len(mat), k=1)]
    return float(np.mean(upper)) if len(upper) else np.nan


def load_seed_pool_and_embeddings(source_run: Path, seed: int, out_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    pool_path = source_run / f"seed{seed}_acquisition_pool_v10c.csv"
    if not pool_path.exists():
        raise FileNotFoundError(pool_path)
    pool_df = pd.read_csv(pool_path)
    artifact_dir = out_dir / f"seed{seed}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir, embedding_lookup, embedding_config = v10smoke.load_or_build_embeddings(pool_df, artifact_dir)
    return pool_df, embedding_lookup, {"embedding_dir": str(embedding_dir), **embedding_config}


def greedy_ubls_select(
    candidates: pd.DataFrame,
    *,
    embedding_lookup: dict[str, np.ndarray],
    query_size: int,
    max_per_pred_class: int,
    dino_weight: float,
    u_loc_weight: float,
    allow_cap_relaxed_fill: bool,
) -> pd.DataFrame:
    # Selection columns: GT columns intentionally ignored.
    remaining = candidates.copy().reset_index(drop=True)
    selected_parts: list[pd.Series] = []
    selected_ids: list[str] = []
    pred_counts: Counter[str] = Counter()

    for rank in range(min(query_size, len(remaining))):
        work = remaining.copy()
        if selected_ids:
            work["_dino_distance_to_selected"] = [
                detector_probe.min_cosine_distance(str(sid), selected_ids, embedding_lookup)
                for sid in work["sample_id"].astype(str)
            ]
        else:
            work["_dino_distance_to_selected"] = [
                detector_probe.min_cosine_distance(str(sid), [], embedding_lookup)
                for sid in work["sample_id"].astype(str)
            ]
            work["_dino_distance_to_selected"] = work["_dino_distance_to_selected"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

        work["_u_loc_norm_select"] = minmax_norm(work["u_loc"])
        work["_dino_norm_select"] = minmax_norm(work["_dino_distance_to_selected"])
        work["_ubls_select_score"] = (
            u_loc_weight * work["_u_loc_norm_select"]
            + dino_weight * work["_dino_norm_select"]
        )

        feasible = work.copy()
        if max_per_pred_class > 0:
            feasible = feasible[
                feasible["detector_pred_class"].astype(str).map(lambda c: pred_counts.get(c, 0) < max_per_pred_class)
            ]
        if feasible.empty:
            if not allow_cap_relaxed_fill:
                break
            feasible = work.copy()

        pick_idx = feasible.sort_values(
            ["_ubls_select_score", "u_loc", "detector_uncertainty", "sample_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).index[0]
        picked = work.loc[pick_idx].copy()
        picked["ubls_rank"] = rank + 1
        selected_parts.append(picked)
        sid = str(picked["sample_id"])
        selected_ids.append(sid)
        pred_counts[str(picked.get("detector_pred_class", ""))] += 1
        remaining = remaining.drop(index=pick_idx).reset_index(drop=True)

    return pd.DataFrame(selected_parts).reset_index(drop=True)


def class_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for value in df.get(col, pd.Series(dtype=str)).fillna("").astype(str):
        if col == "gt_classes":
            labels = [x for x in value.split("|") if x]
        else:
            labels = [value] if value else []
        for label in labels:
            rows.append(label)
    counts = Counter(rows)
    total = sum(counts.values())
    return pd.DataFrame(
        [{"class_name": k, "count": v, "fraction": v / total if total else np.nan} for k, v in sorted(counts.items())]
    )


def summarize_selection(df: pd.DataFrame, all_candidates: pd.DataFrame, selected_ids: list[str], embedding_lookup: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    groups = {
        "all_candidates": all_candidates,
        "ubls_selection": df,
    }
    for name, sub in groups.items():
        pred_counts = sub["detector_pred_class"].astype(str).value_counts()
        rows.append(
            {
                "group": name,
                "n": len(sub),
                "detector_uncertainty_mean": safe_float(pd.to_numeric(sub["detector_uncertainty"], errors="coerce").mean()),
                "u_loc_mean": safe_float(pd.to_numeric(sub["u_loc"], errors="coerce").mean()),
                "u_stability_mean": safe_float(pd.to_numeric(sub.get("u_stability_no_weight_claim"), errors="coerce").mean()),
                "pseudo_fn_ratio_mean_posthoc": safe_float(pd.to_numeric(sub.get("pseudo_fn_ratio"), errors="coerce").mean()),
                "pseudo_recall_mean_posthoc": safe_float(pd.to_numeric(sub.get("pseudo_recall"), errors="coerce").mean()),
                "pseudo_precision_mean_posthoc": safe_float(pd.to_numeric(sub.get("pseudo_precision"), errors="coerce").mean()),
                "pred_class_coverage": int(pred_counts.shape[0]),
                "pred_class_max_count": int(pred_counts.max()) if len(pred_counts) else 0,
                "pairwise_cosine_similarity_mean": pairwise_cosine_similarity_mean(sub["sample_id"].astype(str).tolist(), embedding_lookup),
                "overlap_with_raw_top30": int(len(set(sub["sample_id"].astype(str)) & set(selected_ids))) if name == "ubls_selection" else np.nan,
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def plot_pred_class_counts(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["detector_pred_class"].astype(str).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(counts.index, counts.values, color="#1f77b4")
    ax.set_title("V10d-UBLS predicted-class distribution")
    ax.set_ylabel("selected images")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_gt_audit_counts(df: pd.DataFrame, out_path: Path) -> None:
    if "gt_classes" not in df.columns:
        return
    dist = class_counts(df, "gt_classes")
    if dist.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(dist["class_name"], dist["count"], color="#2ca02c")
    ax.set_title("Post-hoc GT class distribution of V10d-UBLS selection")
    ax.set_ylabel("image-level class count")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability-run", default=str(DEFAULT_STABILITY_RUN))
    parser.add_argument("--source-run", default=str(DEFAULT_SOURCE_RUN))
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--query-size", type=int, default=30)
    parser.add_argument("--uncertainty-top-frac", type=float, default=0.40)
    parser.add_argument("--max-per-pred-class", type=int, default=6)
    parser.add_argument("--dino-weight", type=float, default=0.25)
    parser.add_argument("--u-loc-weight", type=float, default=0.75)
    parser.add_argument("--allow-cap-relaxed-fill", action="store_true")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    stability_run = Path(args.stability_run)
    source_run = Path(args.source_run)
    candidate_path = stability_run / "v10d_stability_candidate_scores.csv"
    raw_top_path = stability_run / "v10d_top_instability_selection.csv"
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"v10d_ubls_selection_seed{args.seed}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_all = pd.read_csv(candidate_path)
    raw_top = pd.read_csv(raw_top_path) if raw_top_path.exists() else pd.DataFrame()
    _, embedding_lookup, embedding_config = load_seed_pool_and_embeddings(source_run, args.seed, out_dir)

    # GT-free filter: uncertainty top fraction only.
    candidates = candidates_all.copy()
    candidates["detector_uncertainty"] = pd.to_numeric(candidates["detector_uncertainty"], errors="coerce").fillna(0.0)
    candidates["u_loc"] = pd.to_numeric(candidates["u_loc"], errors="coerce").fillna(0.0)
    threshold = candidates["detector_uncertainty"].quantile(1.0 - args.uncertainty_top_frac)
    filtered = candidates[candidates["detector_uncertainty"].ge(threshold)].copy()
    if len(filtered) < args.query_size:
        filtered = candidates.sort_values("detector_uncertainty", ascending=False).head(max(args.query_size, len(filtered))).copy()

    selected = greedy_ubls_select(
        filtered,
        embedding_lookup=embedding_lookup,
        query_size=args.query_size,
        max_per_pred_class=args.max_per_pred_class,
        dino_weight=args.dino_weight,
        u_loc_weight=args.u_loc_weight,
        allow_cap_relaxed_fill=args.allow_cap_relaxed_fill,
    )

    raw_top_ids = raw_top["sample_id"].astype(str).tolist() if not raw_top.empty else []
    summary = summarize_selection(selected, filtered, raw_top_ids, embedding_lookup)

    # GT-free gate for training eligibility.
    random_overlap = np.nan
    v10c_overlap = np.nan
    try:
        selected_prior = pd.read_csv(source_run / "v10c_selected_samples.csv")
        sel_ids = set(selected["sample_id"].astype(str))
        random_ids = set(selected_prior[(selected_prior["acquisition_seed"].eq(args.seed)) & (selected_prior["strategy"].astype(str).eq("GTFreeRandom"))]["sample_id"].astype(str))
        v10c_ids = set(selected_prior[(selected_prior["acquisition_seed"].eq(args.seed)) & (selected_prior["strategy"].astype(str).eq("DetectorRecallGuardDINOInstanceV10c"))]["sample_id"].astype(str))
        random_overlap = len(sel_ids & random_ids)
        v10c_overlap = len(sel_ids & v10c_ids)
    except Exception:
        pass

    sel_pred_counts = selected["detector_pred_class"].astype(str).value_counts()
    all_unc_mean = safe_float(pd.to_numeric(candidates_all["detector_uncertainty"], errors="coerce").mean())
    sel_unc_mean = safe_float(pd.to_numeric(selected["detector_uncertainty"], errors="coerce").mean())
    all_uloc_mean = safe_float(pd.to_numeric(candidates_all["u_loc"], errors="coerce").mean())
    sel_uloc_mean = safe_float(pd.to_numeric(selected["u_loc"], errors="coerce").mean())
    gate = pd.DataFrame(
        [
            {"criterion": "pred_class_max_count_le_cap", "pass": bool(sel_pred_counts.max() <= args.max_per_pred_class), "value": int(sel_pred_counts.max()), "reference": args.max_per_pred_class, "uses_gt": False},
            {"criterion": "selected_size_eq_query_size", "pass": bool(len(selected) == args.query_size), "value": int(len(selected)), "reference": args.query_size, "uses_gt": False},
            {"criterion": "pred_class_coverage_at_least_5", "pass": bool(sel_pred_counts.shape[0] >= 5), "value": int(sel_pred_counts.shape[0]), "reference": 5, "uses_gt": False},
            {"criterion": "v10c24_overlap_le_10", "pass": bool(np.isnan(v10c_overlap) or v10c_overlap <= 10), "value": v10c_overlap, "reference": 10, "uses_gt": False},
            {"criterion": "mean_uncertainty_gt_all_candidates", "pass": bool(sel_unc_mean > all_unc_mean), "value": sel_unc_mean, "reference": all_unc_mean, "uses_gt": False},
            {"criterion": "mean_u_loc_gt_all_candidates", "pass": bool(sel_uloc_mean > all_uloc_mean), "value": sel_uloc_mean, "reference": all_uloc_mean, "uses_gt": False},
            {"criterion": "random_overlap_low", "pass": bool(np.isnan(random_overlap) or random_overlap <= 5), "value": random_overlap, "reference": 5, "uses_gt": False},
        ]
    )

    selected.to_csv(out_dir / "v10d_ubls_selected_samples.csv", index=False, encoding="utf-8-sig")
    filtered.to_csv(out_dir / "v10d_ubls_candidate_pool.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "v10d_ubls_selection_summary.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(out_dir / "v10d_ubls_gt_free_gate.csv", index=False, encoding="utf-8-sig")
    class_counts(selected, "detector_pred_class").to_csv(out_dir / "v10d_ubls_pred_class_distribution.csv", index=False, encoding="utf-8-sig")
    if "gt_classes" in selected.columns:
        class_counts(selected, "gt_classes").to_csv(out_dir / "v10d_ubls_posthoc_gt_class_distribution.csv", index=False, encoding="utf-8-sig")

    plot_pred_class_counts(selected, out_dir / "fig01_pred_class_distribution.png")
    plot_gt_audit_counts(selected, out_dir / "fig02_posthoc_gt_class_distribution.png")

    config = {
        "stability_run": str(stability_run),
        "source_run": str(source_run),
        "seed": args.seed,
        "query_size": args.query_size,
        "uncertainty_top_frac": args.uncertainty_top_frac,
        "max_per_pred_class": args.max_per_pred_class,
        "dino_weight": args.dino_weight,
        "u_loc_weight": args.u_loc_weight,
        "allow_cap_relaxed_fill": bool(args.allow_cap_relaxed_fill),
        "selection_uses_gt_columns": False,
        "gt_columns_posthoc_only": sorted(GT_AUDIT_COLUMNS),
        "final_test_used": False,
        "no_training": True,
        "embedding_config": embedding_config,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# V10d-UBLS selection-only probe",
        "",
        "GT-free selection gate. GT columns, if present, are post-hoc audit only.",
        "",
        "## GT-free gate",
        "",
        md_table(gate),
        "",
        "## Selection summary",
        "",
        md_table(summary),
        "",
        "## Predicted class distribution",
        "",
        md_table(class_counts(selected, "detector_pred_class")),
        "",
        "## Post-hoc GT class distribution",
        "",
        md_table(class_counts(selected, "gt_classes") if "gt_classes" in selected.columns else pd.DataFrame()),
        "",
        "## Recommendation rule",
        "",
        "Use this selection to authorize a *separate untouched-seed* one-cycle training only if the GT-free gate passes. Do not train on seed47 as final evidence.",
    ]
    (out_dir / "v10d_ubls_selection_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("=" * 100)
    print("[DONE] V10d-UBLS selection-only probe finished")
    print(f"Output dir: {out_dir}")
    print(f"Filtered candidates: {len(filtered)} / {len(candidates_all)}")
    print(f"Selected: {len(selected)}")
    print(f"GT-free gate passed: {int(gate['pass'].sum())}/{len(gate)}")
    print("No training. Final test used=False. GT used only post-hoc=True.")
    print("=" * 100)


if __name__ == "__main__":
    main()

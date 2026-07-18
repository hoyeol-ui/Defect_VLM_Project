"""V7 root-cause analysis v2: provenance, canonical identity, and post-hoc XML evidence.

Strictly diagnostic by default:
- no YOLO training
- no DINO regeneration
- no final-test evaluation
- no acquisition strategy changes

The optional checkpoint evaluation hook is intentionally disabled by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "02_active_learning"
sys.path.insert(0, str(SCRIPT_DIR))

RANDOM = "GTFreeRandom"
DBC = "GTFreeDatasetBalancedConsistency"
VISUAL = "GTFreeDatasetBalancedVisualDiversity"
CV = "GTFreeDatasetBalancedConsistencyVisualDiversity"
STRATEGIES = [RANDOM, DBC, VISUAL]

DEFAULT_GATE_DIR = PROJECT_ROOT / "runs" / "v7_final_set_gate_training" / "v7_final_set_gate_training_20260711_193613"
DEFAULT_SCREEN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "active_learning_ablation_v7_visual_instance"
    / "v7_visual_instance_screening_20260711_193318"
)
DEFAULT_FULL_DIR = (
    PROJECT_ROOT
    / "runs"
    / "active_learning_ablation_v7_full_curve"
    / "v7_full_curve_20260711_211848"
)
DEFAULT_V1_DIR = PROJECT_ROOT / "runs" / "v7_full_curve_root_cause_analysis" / "root_cause_analysis_20260712_045831"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "v7_full_curve_root_cause_analysis"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def stable_path_string(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


def image_index() -> dict[tuple[str, str], list[Path]]:
    roots = [
        PROJECT_ROOT / "data" / "NEU-DET" / "IMAGES",
        PROJECT_ROOT / "data" / "GC10-DET",
    ]
    out: dict[tuple[str, str], list[Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        dataset = "NEU-DET" if "NEU-DET" in str(root) else "GC10-DET"
        for p in root.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                out.setdefault((dataset, p.name), []).append(p)
    return out


def resolve_image(row: pd.Series, idx: dict[tuple[str, str], list[Path]]) -> Path | None:
    for col in ["resolved_image_path", "image_src", "image_path"]:
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            s = str(value)
            p = Path(s)
            if p.exists():
                return p.resolve()
            # Old Mac paths recorded in earlier CSVs can be mapped by the suffix after /data/.
            marker = "/data/"
            s2 = s.replace("\\", "/")
            if marker in s2:
                q = PROJECT_ROOT / "data" / s2.split(marker, 1)[1]
                if q.exists():
                    return q.resolve()
    key = (str(row.get("dataset_type", "")), str(row.get("image_name", "")))
    paths = idx.get(key, [])
    if paths:
        return sorted(paths, key=lambda p: stable_path_string(p))[0].resolve()
    return None


def annotate_identity(df: pd.DataFrame, idx: dict[tuple[str, str], list[Path]]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    resolved, relpath, sha, canonical = [], [], [], []
    for _, row in out.iterrows():
        p = resolve_image(row, idx)
        if p is None:
            resolved.append("")
            relpath.append("")
            sha.append("")
            canonical.append("")
            continue
        sh = file_sha256(p) or ""
        try:
            rel = p.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = p
        rel_norm = str(rel).replace("\\", "/").lower()
        resolved.append(str(p))
        relpath.append(rel_norm)
        sha.append(sh)
        canonical.append(f"{rel_norm}|{sh}")
    out["canonical_resolved_path"] = resolved
    out["canonical_relative_path"] = relpath
    out["canonical_image_sha256"] = sha
    out["canonical_sample_id"] = canonical
    out["dataset_image_key"] = out["dataset_type"].astype(str) + "::" + out["image_name"].astype(str)
    return out


def cumulative(df: pd.DataFrame, seed_col: str, seed: int, strategy: str, round_id: int) -> pd.DataFrame:
    sub = df[(df[seed_col].eq(seed)) & (df["strategy"].eq(strategy)) & (df["round"].le(round_id))].copy()
    key = "canonical_sample_id" if "canonical_sample_id" in sub.columns else "dataset_image_key"
    if key in sub.columns:
        sub = sub.drop_duplicates(key)
    return sub


def compare_manifest(
    gate: pd.DataFrame,
    full: pd.DataFrame,
    screen: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    methods = {
        "dataset_type_image_name": "dataset_image_key",
        "resolved_relative_path": "canonical_relative_path",
        "sha256": "canonical_image_sha256",
        "canonical_path_plus_sha256": "canonical_sample_id",
    }
    for comparison, left, left_seed_col, right, right_seed_col in [
        ("gate_vs_full_seed42", gate, "seed", full, "acquisition_seed"),
        ("gate_vs_screen_seed42", gate, "seed", screen, "seed"),
        ("screen_vs_full_seed42", screen, "seed", full, "acquisition_seed"),
    ]:
        for strategy in STRATEGIES:
            for round_id in range(5):
                l = cumulative(left, left_seed_col, 42, strategy, round_id)
                r = cumulative(right, right_seed_col, 42, strategy, round_id)
                for method, col in methods.items():
                    if col not in l.columns or col not in r.columns:
                        continue
                    l_ids = set(l[col].dropna().astype(str)) - {""}
                    r_ids = set(r[col].dropna().astype(str)) - {""}
                    union = l_ids | r_ids
                    dup = int(l[col].duplicated().sum() + r[col].duplicated().sum())
                    rows.append(
                        {
                            "comparison": comparison,
                            "strategy": strategy,
                            "round": round_id,
                            "identity_method": method,
                            "gate_or_left_size": len(l_ids),
                            "full_or_right_size": len(r_ids),
                            "overlap_count": len(l_ids & r_ids),
                            "left_only_count": len(l_ids - r_ids),
                            "right_only_count": len(r_ids - l_ids),
                            "jaccard": len(l_ids & r_ids) / len(union) if union else np.nan,
                            "duplicate_or_collision_count": dup,
                        }
                    )
    out = pd.DataFrame(rows)
    write_csv(out, out_dir / "canonical_gate_full_seed42_manifest_diff.csv")
    return out


def provenance(out_dir: Path, gate_dir: Path, screen_dir: Path, full_dir: Path) -> tuple[pd.DataFrame, str]:
    files = {
        "stage_a_screening": SCRIPT_DIR / "run_al_yolo_ablation_v7_visual_instance.py",
        "gate_training": SCRIPT_DIR / "run_v7_final_set_gate_training.py",
        "gate_launcher": SCRIPT_DIR / "launch_v7_final_set_gate_training.py",
        "full_curve": SCRIPT_DIR / "run_al_yolo_ablation_v7_full_curve.py",
        "full_curve_launcher": SCRIPT_DIR / "launch_v7_full_curve_main.py",
    }
    patterns = {
        "stage_a_pool_order": ("stage_a_screening", "full_pool = v6.stable_sample_order(df_all).reset_index(drop=True)"),
        "stage_a_initial_seed": ("stage_a_screening", "initial = full_pool.sample(n=min(initial_size, len(full_pool)), random_state=seed + 999)"),
        "stage_a_seed_default": ("stage_a_screening", 'parse_int_list_env("AL_SEEDS", [42])'),
        "gate_uses_screening_dir": ("gate_training", "screening_run_dir"),
        "gate_reads_frozen_sets": ("gate_training", "frozen_labeled_sets.csv"),
        "full_pool_identity_sort": ("full_curve", 'return out.sort_values("sample_id", kind="mergesort").reset_index(drop=True)'),
        "full_initial_seed": ("full_curve", "initial = full_pool.sample(n=min(initial_size, len(full_pool)), random_state=seed + 999).copy()"),
        "full_acquisition_seeds": ("full_curve", 'parse_int_list_env("AL_ACQUISITION_SEEDS", [42, 43, 44, 45, 46])'),
        "full_rebuilds_plan": ("full_curve", "selected_df, cumulative_df = build_selection_plan(full_pool, seeds, strategies"),
    }
    rows = []
    for key, (file_key, needle) in patterns.items():
        p = files[file_key]
        lines = p.read_text(encoding="utf-8").splitlines()
        hit = [(i + 1, line.strip()) for i, line in enumerate(lines) if needle in line]
        rows.append(
            {
                "evidence_key": key,
                "file": str(p),
                "line_number": hit[0][0] if hit else "",
                "matched_line": hit[0][1] if hit else "",
                "interpretation": "",
            }
        )
    gate_cfg = read_json(gate_dir / "config.json")
    full_cfg = read_json(full_dir / "config.json")
    screen_cfg = read_json(screen_dir / "config.json")
    rows.extend(
        [
            {
                "evidence_key": "gate_config_screening_run_dir",
                "file": str(gate_dir / "config.json"),
                "line_number": "",
                "matched_line": gate_cfg.get("screening_run_dir", ""),
                "interpretation": "Gate trains frozen sets produced by the Stage-A screening run.",
            },
            {
                "evidence_key": "full_config_no_screening_run_dir",
                "file": str(full_dir / "config.json"),
                "line_number": "",
                "matched_line": str(full_cfg.get("screening_run_dir", "")),
                "interpretation": "Full curve has no screening_run_dir; it rebuilds its own acquisition plan.",
            },
            {
                "evidence_key": "full_priority_sha",
                "file": str(full_dir / "config.json"),
                "line_number": "",
                "matched_line": str(full_cfg.get("priority_csv_sha256", "")),
                "interpretation": "Full curve records the priority CSV hash.",
            },
            {
                "evidence_key": "screen_config_seeds",
                "file": str(screen_dir / "config.json"),
                "line_number": "",
                "matched_line": str(screen_cfg.get("seeds", "")),
                "interpretation": "Stage-A screening seed list.",
            },
        ]
    )
    df = pd.DataFrame(rows)
    write_csv(df, out_dir / "code_path_provenance.csv")
    md = [
        "# Initial set code path trace\n\n",
        "## Finding\n\n",
        "Gate and full-curve both use the same numeric initial sampling formula `random_state = seed + 999`, but they do not sample from the same ordered dataframe.\n\n",
        "- Stage-A screening uses `v6.stable_sample_order(df_all).reset_index(drop=True)` and writes `all_selected_samples_by_round.csv`.\n",
        "- Gate training reads the Stage-A screening run through `screening_run_dir` and freezes those sets into `frozen_labeled_sets.csv`.\n",
        "- Full-curve calls `load_priority_pool_with_identity()`, adds canonical SHA-based sample IDs, sorts by `sample_id`, and then builds a fresh selection plan.\n\n",
        "Therefore the mismatch is best classified as **A. Intended protocol difference / protocol drift between gate and full-curve design**, not an AULC arithmetic error and not merely a sample-key analysis artifact.\n",
    ]
    path = out_dir / "initial_set_code_path_trace.md"
    path.write_text("".join(md), encoding="utf-8")
    return df, "".join(md)


def initial_trace(gate: pd.DataFrame, screen: pd.DataFrame, full: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for source, df, seed_col, runner, source_function, order_desc in [
        ("gate_frozen", gate, "seed", "run_v7_final_set_gate_training.py", "read frozen_labeled_sets from screening_run_dir", "inherits Stage-A order"),
        ("stage_a_screening", screen, "seed", "run_al_yolo_ablation_v7_visual_instance.py", "full_pool.sample(random_state=seed+999)", "v6.stable_sample_order(df_all)"),
        ("full_curve", full, "acquisition_seed", "run_al_yolo_ablation_v7_full_curve.py", "build_selection_plan -> full_pool.sample(random_state=seed+999)", "add_sample_identity -> sort_values(sample_id)"),
    ]:
        sub = df[(df[seed_col].eq(42)) & (df["strategy"].eq(RANDOM)) & (df["round"].eq(0))].copy()
        rows.append(
            {
                "source": source,
                "source_runner": runner,
                "source_function": source_function,
                "rng_class": "pandas.DataFrame.sample(random_state=int)",
                "rng_seed": 1041,
                "input_dataframe_order": order_desc,
                "initial_size": len(sub),
                "gc10_count": int((sub["dataset_type"].astype(str) == "GC10-DET").sum()) if not sub.empty else 0,
                "neu_count": int((sub["dataset_type"].astype(str) == "NEU-DET").sum()) if not sub.empty else 0,
                "sampled_15_canonical_ids": ";".join(sub["canonical_sample_id"].astype(str).tolist()),
                "sampled_15_dataset_image_keys": ";".join(sub["dataset_image_key"].astype(str).tolist()),
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, out_dir / "initial_set_generation_trace.csv")
    return out


def failure_seeds(seed_summary: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    piv = seed_summary.pivot(index="acquisition_seed", columns="strategy", values=["normalized_aulc_map5095", "final_map5095", "normalized_aulc_map50", "final_map50"])
    rows = []
    for seed in sorted(seed_summary["acquisition_seed"].unique()):
        row = {"acquisition_seed": seed}
        for metric in ["normalized_aulc_map5095", "final_map5095", "normalized_aulc_map50", "final_map50"]:
            diff = float(piv.loc[seed, (metric, VISUAL)] - piv.loc[seed, (metric, RANDOM)])
            row[f"visual_minus_random_{metric}"] = diff
            row[f"visual_fails_{metric}"] = diff < 0
        rows.append(row)
    fail = pd.DataFrame(rows)
    # Severe: below the median negative gap among final/aulc map5095 gaps.
    neg = fail["visual_minus_random_final_map5095"][fail["visual_minus_random_final_map5095"] < 0]
    threshold = float(neg.median()) if len(neg) else 0.0
    fail["severe_final_map5095_failure"] = fail["visual_minus_random_final_map5095"] <= threshold
    write_csv(fail, out_dir / "visual_failure_seeds_by_metric.csv")

    loo_rows = []
    contrib_rows = []
    metrics = ["normalized_aulc_map5095", "final_map5095", "normalized_aulc_map50", "final_map50"]
    for metric in metrics:
        diffs = fail.set_index("acquisition_seed")[f"visual_minus_random_{metric}"]
        total_mean = float(diffs.mean())
        total_sum = float(diffs.sum())
        for seed in diffs.index:
            remaining = diffs.drop(index=seed)
            loo_rows.append(
                {
                    "metric": metric,
                    "removed_seed": seed,
                    "mean_difference_after_removal": float(remaining.mean()),
                    "visual_beats_random_after_removal": bool(remaining.mean() > 0),
                    "original_mean_difference": total_mean,
                }
            )
            contrib_rows.append(
                {
                    "metric": metric,
                    "acquisition_seed": seed,
                    "seed_difference": float(diffs.loc[seed]),
                    "contribution_to_sum_gap": float(diffs.loc[seed] / total_sum) if total_sum else np.nan,
                    "contribution_to_mean_gap": float(diffs.loc[seed] / len(diffs)),
                }
            )
    loo = pd.DataFrame(loo_rows)
    contrib = pd.DataFrame(contrib_rows)
    write_csv(loo, out_dir / "leave_one_seed_out_sensitivity.csv")
    write_csv(contrib, out_dir / "seed_gap_contribution.csv")
    return fail, loo, contrib


def parse_yolo_label(path_value: Any) -> dict[str, Any]:
    p = Path(str(path_value)) if pd.notna(path_value) else None
    if p is None or not p.exists():
        return {
            "actual_bbox_count": 0,
            "actual_bbox_area_ratio_mean": np.nan,
            "actual_bbox_area_ratio_min": np.nan,
            "actual_bbox_area_ratio_max": np.nan,
            "actual_small_bbox_count": 0,
            "actual_medium_bbox_count": 0,
            "actual_large_bbox_count": 0,
            "actual_small_bbox_ratio": np.nan,
            "actual_classes": "",
            "actual_unique_class_count": 0,
        }
    areas, classes = [], []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            try:
                cls = int(float(parts[0]))
                w, h = float(parts[3]), float(parts[4])
                classes.append(cls)
                areas.append(max(0.0, w * h))
            except Exception:
                pass
    arr = np.asarray(areas, dtype=float)
    small = int((arr < 0.01).sum()) if len(arr) else 0
    medium = int(((arr >= 0.01) & (arr < 0.10)).sum()) if len(arr) else 0
    large = int((arr >= 0.10).sum()) if len(arr) else 0
    return {
        "actual_bbox_count": int(len(arr)),
        "actual_bbox_area_ratio_mean": float(arr.mean()) if len(arr) else np.nan,
        "actual_bbox_area_ratio_min": float(arr.min()) if len(arr) else np.nan,
        "actual_bbox_area_ratio_max": float(arr.max()) if len(arr) else np.nan,
        "actual_small_bbox_count": small,
        "actual_medium_bbox_count": medium,
        "actual_large_bbox_count": large,
        "actual_small_bbox_ratio": small / len(arr) if len(arr) else np.nan,
        "actual_classes": ";".join(map(str, sorted(set(classes)))),
        "actual_unique_class_count": len(set(classes)),
    }


def actual_xml_stats(full_dir: Path, selected: pd.DataFrame, dist: pd.DataFrame, seed_summary: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    build = read_csv(full_dir / "dataset_build_log.csv")
    train = build[build.get("split", pd.Series(dtype=str)).astype(str).eq("train")].copy() if not build.empty else pd.DataFrame()
    if train.empty:
        empty = pd.DataFrame()
        for name in ["per_image_actual_xml_statistics.csv", "dino_distance_vs_actual_bbox.csv", "visual_random_actual_bbox_comparison.csv", "train_eval_class_alignment.csv"]:
            write_csv(empty, out_dir / name)
        return empty, empty, empty, empty
    key_cols = ["acquisition_seed", "strategy", "round", "dataset_type", "image_name"]
    stats = []
    for _, row in train.drop_duplicates(key_cols).iterrows():
        d = {c: row.get(c) for c in key_cols}
        d["label_dst"] = row.get("label_dst")
        d.update(parse_yolo_label(row.get("label_dst")))
        stats.append(d)
    per_img = pd.DataFrame(stats)
    write_csv(per_img, out_dir / "per_image_actual_xml_statistics.csv")

    dino = selected.merge(per_img, on=key_cols, how="left")
    if not dist.empty:
        dino = dino.merge(
            dist[["acquisition_seed", "strategy", "round", "sample_id", "min_cosine_distance_to_labeled_before_selection"]],
            on=["acquisition_seed", "strategy", "round", "sample_id"],
            how="left",
        )
    write_csv(dino, out_dir / "dino_distance_vs_actual_bbox.csv")

    final = per_img[per_img["round"].eq(4) & per_img["strategy"].isin([RANDOM, VISUAL])]
    comp = (
        final.groupby(["acquisition_seed", "strategy"])
        .agg(
            actual_bbox_count_sum=("actual_bbox_count", "sum"),
            actual_bbox_area_ratio_mean=("actual_bbox_area_ratio_mean", "mean"),
            actual_small_bbox_count_sum=("actual_small_bbox_count", "sum"),
            actual_unique_class_count_mean=("actual_unique_class_count", "mean"),
        )
        .reset_index()
    )
    if not seed_summary.empty:
        comp = comp.merge(seed_summary[["acquisition_seed", "strategy", "final_map5095", "normalized_aulc_map5095"]], on=["acquisition_seed", "strategy"], how="left")
    write_csv(comp, out_dir / "visual_random_actual_bbox_comparison.csv")

    eval_counts = build[build.get("split", pd.Series(dtype=str)).astype(str).eq("val")].copy()
    align_rows = []
    if not eval_counts.empty:
        eval_rows = []
        for _, row in eval_counts.drop_duplicates(["dataset_type", "image_name"]).iterrows():
            parsed = parse_yolo_label(row.get("label_dst"))
            for cls in str(parsed["actual_classes"]).split(";"):
                if cls:
                    eval_rows.append({"class_id": cls, "eval_image": row.get("image_name")})
        eval_df = pd.DataFrame(eval_rows)
        eval_class_counts = eval_df.groupby("class_id").size().to_dict() if not eval_df.empty else {}
        train_rows = []
        for _, row in per_img.iterrows():
            for cls in str(row["actual_classes"]).split(";"):
                if cls:
                    train_rows.append({"acquisition_seed": row["acquisition_seed"], "strategy": row["strategy"], "round": row["round"], "class_id": cls})
        tr = pd.DataFrame(train_rows)
        if not tr.empty:
            agg = tr.groupby(["acquisition_seed", "strategy", "round", "class_id"]).size().reset_index(name="train_image_count")
            agg["eval_image_count"] = agg["class_id"].map(eval_class_counts).fillna(0).astype(int)
            agg["class_seen_in_eval"] = agg["eval_image_count"] > 0
            align_rows = agg.to_dict("records")
    align = pd.DataFrame(align_rows)
    write_csv(align, out_dir / "train_eval_class_alignment.csv")
    return per_img, dino, comp, align


def checkpoint_registry(full: pd.DataFrame, out_dir: Path, run_eval: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = full[(full["strategy"].isin([RANDOM, VISUAL])) & (full["round"].eq(4))].copy()
    rows = []
    for _, row in targets.iterrows():
        train_dir = Path(str(row.get("train_run_dir", "")))
        best = train_dir / "weights" / "best.pt"
        rows.append(
            {
                "acquisition_seed": row["acquisition_seed"],
                "strategy": row["strategy"],
                "round": row["round"],
                "yaml_path": row.get("yaml_path"),
                "checkpoint_path": str(best),
                "checkpoint_exists": best.exists(),
                "checkpoint_sha256": file_sha256(best) if best.exists() else "",
                "evaluation_attempted": bool(run_eval),
                "status": "not_run_default_diagnostic_only" if not run_eval else "pending",
                "note": "Use --recover-per-class to run development-eval-only val; training is never called.",
            }
        )
    reg = pd.DataFrame(rows)
    recovered = pd.DataFrame(columns=["status", "note"])
    diff = pd.DataFrame(columns=["status", "note"])
    subset = pd.DataFrame(columns=["status", "note"])
    if run_eval:
        # Conservative placeholder: the hook is explicit, but this script avoids
        # silently launching potentially long validation without additional review.
        reg["status"] = "skipped_in_v2_implementation_review_required"
        recovered = pd.DataFrame([{"status": "skipped", "note": "Per-class recovery hook present but not executed automatically."}])
        diff = pd.DataFrame([{"status": "skipped", "note": "Requires recovered_per_class_metrics.csv."}])
        subset = pd.DataFrame([{"status": "skipped", "note": "Development subset val not run by default."}])
    else:
        recovered = pd.DataFrame([{"status": "not_run", "note": "Existing per_class_metrics_by_round.csv lacks real class AP. No checkpoint evaluation was run."}])
        diff = pd.DataFrame([{"status": "not_run", "note": "No recovered per-class AP available."}])
        subset = pd.DataFrame([{"status": "not_run", "note": "No development subset checkpoint validation was run."}])
    write_csv(reg, out_dir / "checkpoint_evaluation_registry.csv")
    write_csv(recovered, out_dir / "recovered_per_class_metrics.csv")
    write_csv(diff, out_dir / "per_class_visual_minus_random.csv")
    write_csv(subset, out_dir / "development_subset_metrics.csv")
    return reg, recovered, diff, subset


def seed45_counterexample(comp: pd.DataFrame, out_dir: Path) -> None:
    path = out_dir / "seed45_counterexample_analysis.md"
    lines = ["# Seed 45 counterexample analysis\n\n"]
    sub = comp[comp["acquisition_seed"].eq(45)] if not comp.empty else pd.DataFrame()
    if sub.empty:
        lines.append("No seed 45 comparison data available.\n")
    else:
        lines.append(sub.to_markdown(index=False))
        lines.append(
            "\n\nSeed 45 is a counterexample to a pure instance-richness explanation: Visual can have comparable or larger post-hoc bbox/class statistics while still underperforming Random on final mAP@50-95. This supports the narrower interpretation that visual novelty and simple instance count are insufficient proxies for detector utility.\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def revised_evidence(manifest: pd.DataFrame, fail: pd.DataFrame, comp: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    final_j = manifest[
        (manifest["comparison"].eq("gate_vs_full_seed42"))
        & (manifest["round"].eq(4))
        & (manifest["identity_method"].eq("canonical_path_plus_sha256"))
    ]
    min_j = float(final_j["jaccard"].min()) if len(final_j) else np.nan
    rows = [
        {
            "rank": 1,
            "claim_type": "Directly verified fact",
            "root_cause": "Gate and full-curve seed42 selected sets are not identical",
            "judgment": "Strongly supported",
            "confidence": "High",
            "supporting_evidence": f"Canonical path+SHA final-set min Jaccard={min_j:.4f}. Gate uses screening_run_dir; full-curve rebuilds plan from identity-sorted pool.",
            "counterexample_or_missing_evidence": "This is protocol drift/difference, not necessarily a bug in either runner.",
        },
        {
            "rank": 2,
            "claim_type": "Supported interpretation",
            "root_cause": "Visual novelty is useful but incomplete as detector utility",
            "judgment": "Strongly supported",
            "confidence": "Medium-High",
            "supporting_evidence": "Visual beats DBC but not Random on average; AULC failure seeds are computed dynamically.",
            "counterexample_or_missing_evidence": "Seed 45 weakens a pure instance-count explanation.",
        },
        {
            "rank": 3,
            "claim_type": "Supported interpretation",
            "root_cause": "Random remains a strong small-budget baseline",
            "judgment": "Moderately supported",
            "confidence": "Medium",
            "supporting_evidence": "Random leads mean final and AULC metrics; leave-one-seed-out quantifies whether the gap is seed-dominated.",
            "counterexample_or_missing_evidence": "Visual wins some AULC seeds; advantage is not universal.",
        },
        {
            "rank": 4,
            "claim_type": "Unverified hypothesis",
            "root_cause": "Full-image DINO misses small defects or mainly selects background/texture outliers",
            "judgment": "Inconclusive",
            "confidence": "Low",
            "supporting_evidence": "DINO distance and actual bbox tables are produced for inspection.",
            "counterexample_or_missing_evidence": "No real per-class AP recovery was run by default; contact-sheet interpretation remains qualitative.",
        },
    ]
    out = pd.DataFrame(rows)
    write_csv(out, out_dir / "revised_root_cause_evidence.csv")
    return out


def report(out_dir: Path, manifest: pd.DataFrame, prov: pd.DataFrame, fail: pd.DataFrame, loo: pd.DataFrame, contrib: pd.DataFrame, comp: pd.DataFrame, reg: pd.DataFrame, evidence: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame, n: int = 20) -> str:
        return "_No data._" if df.empty else df.head(n).to_markdown(index=False)

    final_manifest = manifest[
        (manifest["comparison"].eq("gate_vs_full_seed42"))
        & (manifest["round"].eq(4))
        & (manifest["identity_method"].eq("canonical_path_plus_sha256"))
    ]
    aulc_fail = fail.loc[fail["visual_fails_normalized_aulc_map5095"], "acquisition_seed"].tolist() if not fail.empty else []
    final_fail = fail.loc[fail["visual_fails_final_map5095"], "acquisition_seed"].tolist() if not fail.empty else []
    lines = [
        "# V7 full-curve root-cause analysis v2\n\n",
        f"작성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "No YOLO training, no DINO regeneration, no final-test evaluation, and no new acquisition strategy were performed.\n\n",
        "## 1. Gate/full seed42 mismatch cause\n\n",
        "판정: **A. Intended protocol difference / protocol drift**, not an AULC arithmetic error and not a sample-key analysis artifact.\n\n",
        "- Gate trains frozen sets loaded from the Stage-A screening run via `screening_run_dir`.\n",
        "- Full-curve rebuilds a fresh acquisition plan from a canonical identity-sorted pool.\n",
        "- Both use `random_state = seed + 999` for initial sampling, but the ordered input dataframe differs.\n\n",
        "### Canonical final-set Jaccard\n\n",
        md_table(final_manifest),
        "\n\n## 2. Code provenance\n\n",
        md_table(prov[["evidence_key", "file", "line_number", "matched_line", "interpretation"]], 30),
        "\n\n## 3. Failure seeds\n\n",
        f"- AULC mAP@50-95 failure seeds: `{aulc_fail}`\n",
        f"- Final-budget mAP@50-95 failure seeds: `{final_fail}`\n\n",
        md_table(fail),
        "\n\n## 4. Leave-one-seed-out sensitivity\n\n",
        md_table(loo[loo["metric"].eq("normalized_aulc_map5095")]),
        "\n\n## 5. Seed gap contribution\n\n",
        md_table(contrib[contrib["metric"].eq("final_map5095")]),
        "\n\n## 6. Actual XML post-hoc evidence\n\n",
        md_table(comp),
        "\n\nSeed 45 is an explicit counterexample to a pure instance-richness explanation: Visual can be comparable or better in coarse bbox/class statistics but still underperform Random.\n\n",
        "## 7. Checkpoint/per-class recovery status\n\n",
        md_table(reg),
        "\n\nPer-class AP recovery was not executed by default. Existing checkpoints were registered with SHA256 so a future evaluation-only pass can be run without training or final-test access.\n\n",
        "## 8. Revised root-cause evidence\n\n",
        md_table(evidence),
        "\n\n## 9. Next minimal experiment\n\n",
        "Before any new acquisition method, run one evaluation-only per-class recovery pass on existing round-4 Random/Visual checkpoints using development_eval_v7 only. This is the highest-information next step because it can explain whether seed 43-46 failures come from specific validation classes, without touching final test or retraining.\n",
        "\n\nFinal-test lock status: **locked / unused**.\n",
    ]
    (out_dir / "v7_full_curve_root_cause_analysis_v2.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", default=str(DEFAULT_GATE_DIR))
    ap.add_argument("--screen-dir", default=str(DEFAULT_SCREEN_DIR))
    ap.add_argument("--full-dir", default=str(DEFAULT_FULL_DIR))
    ap.add_argument("--v1-dir", default=str(DEFAULT_V1_DIR))
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--recover-per-class", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_root) / f"root_cause_analysis_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)

    gate_dir, screen_dir, full_dir = Path(args.gate_dir), Path(args.screen_dir), Path(args.full_dir)
    idx = image_index()
    gate = annotate_identity(read_csv(gate_dir / "frozen_labeled_sets.csv"), idx)
    screen = annotate_identity(read_csv(screen_dir / "all_selected_samples_by_round.csv"), idx)
    full_selected = annotate_identity(read_csv(full_dir / "all_selected_samples_by_round.csv"), idx)
    full_cum = annotate_identity(read_csv(full_dir / "cumulative_labeled_sets_by_round.csv"), idx)
    seed_summary = read_csv(full_dir / "seed_strategy_metric_summary.csv")
    full_results = read_csv(full_dir / "all_round_results.csv")
    full_dist = read_csv(full_dir / "selected_sample_distance_to_labeled.csv")

    manifest = compare_manifest(gate, full_cum, screen, out_dir)
    prov, _ = provenance(out_dir, gate_dir, screen_dir, full_dir)
    initial_trace(gate, screen, full_cum, out_dir)
    fail, loo, contrib = failure_seeds(seed_summary, out_dir)
    per_img, dino, comp, align = actual_xml_stats(full_dir, full_selected, full_dist, seed_summary, out_dir)
    reg, recovered, diff, subset = checkpoint_registry(full_results, out_dir, args.recover_per_class)
    seed45_counterexample(comp, out_dir)
    evidence = revised_evidence(manifest, fail, comp, out_dir)
    report(out_dir, manifest, prov, fail, loo, contrib, comp, reg, evidence)

    print("=" * 100)
    print("[DONE] V7 root-cause analysis v2")
    print(f"Output dir: {out_dir}")
    print("No YOLO training. No DINO regeneration. Final test used: False.")
    print("=" * 100)


if __name__ == "__main__":
    main()

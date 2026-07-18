"""Audit why GTFreeRandom is a hard baseline against V10c24.

This is a post-hoc analysis script only:
  - no model training
  - no final-test evaluation
  - GT XML is used only after acquisition/training results already exist

Default input is the completed V10c 24/6 one-cycle run.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = PROJECT_ROOT / "runs" / "active_learning_v10c_recall_guard_onecycle" / "v10c_recall_guard_onecycle_20260713_201155"
OUT_ROOT = PROJECT_ROOT / "runs" / "random_baseline_audit_v10c24"
NEU_ANNOTATIONS = PROJECT_ROOT / "data" / "NEU-DET" / "ANNOTATIONS"

RANDOM_STRATEGY = "GTFreeRandom"
V10C_STRATEGY = "DetectorRecallGuardDINOInstanceV10c"
NEU6 = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def normalize_class_name(value: Any) -> str:
    return str(value).strip().replace("rolled-in-scale", "rolled-in_scale")


def safe_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def table_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.head(max_rows)
    try:
        return shown.to_markdown(index=False)
    except Exception:
        return "```text\n" + shown.to_string(index=False) + "\n```"


def image_stem(row: pd.Series) -> str:
    image_name = str(row.get("image_name", "")).strip()
    if image_name:
        return Path(image_name).stem
    image_path = str(row.get("resolved_image_path", row.get("image_path", ""))).split("::")[0]
    return Path(image_path).stem


def parse_xml_for_image(row: pd.Series) -> dict[str, Any]:
    xml_path = NEU_ANNOTATIONS / f"{image_stem(row)}.xml"
    if not xml_path.exists():
        return {
            "xml_path": str(xml_path),
            "xml_exists": False,
            "gt_box_count": np.nan,
            "gt_classes": "",
            "gt_majority_class": "",
            "gt_area_mean": np.nan,
            "gt_area_median": np.nan,
            "gt_area_frac_mean": np.nan,
            "multi_instance": np.nan,
        }

    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = safe_float(size.findtext("width")) if size is not None else np.nan
    height = safe_float(size.findtext("height")) if size is not None else np.nan
    image_area = width * height if np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0 else np.nan
    classes: list[str] = []
    areas: list[float] = []
    for obj in root.findall("object"):
        cls = normalize_class_name(obj.findtext("name", ""))
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = safe_float(bnd.findtext("xmin"))
        ymin = safe_float(bnd.findtext("ymin"))
        xmax = safe_float(bnd.findtext("xmax"))
        ymax = safe_float(bnd.findtext("ymax"))
        if not all(np.isfinite(v) for v in [xmin, ymin, xmax, ymax]):
            continue
        classes.append(cls)
        areas.append(max(0.0, xmax - xmin) * max(0.0, ymax - ymin))

    counts = Counter(classes)
    area_arr = np.asarray(areas, dtype=float)
    return {
        "xml_path": str(xml_path),
        "xml_exists": True,
        "image_width": width,
        "image_height": height,
        "gt_box_count": int(len(classes)),
        "gt_classes": "|".join(sorted(counts)),
        "gt_majority_class": counts.most_common(1)[0][0] if counts else "",
        "gt_class_counts_json": json.dumps(dict(sorted(counts.items())), ensure_ascii=False),
        "gt_area_mean": float(area_arr.mean()) if len(area_arr) else np.nan,
        "gt_area_median": float(np.median(area_arr)) if len(area_arr) else np.nan,
        "gt_area_frac_mean": float((area_arr / image_area).mean()) if len(area_arr) and np.isfinite(image_area) else np.nan,
        "multi_instance": bool(len(classes) >= 2),
    }


def entropy_from_counts(counts: pd.Series) -> float:
    vals = counts.astype(float).to_numpy()
    total = vals.sum()
    if total <= 0:
        return np.nan
    p = vals / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def load_required_csv(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def attach_detector_scores(run_dir: Path, selected: pd.DataFrame) -> pd.DataFrame:
    """Fill detector-score columns for Random rows from saved round1 score CSVs.

    The V10c selected CSV already contains selector scores for V10c rows, but
    Random rows are sampled before selection diagnostics are attached.  The
    saved seed-level detector score CSV is still valid for post-hoc audit.
    """
    out_parts = []
    score_cols = [
        "detector_pseudo_box_count",
        "detector_no_box",
        "detector_max_conf",
        "detector_mean_conf",
        "detector_min_conf",
        "detector_pred_class",
        "detector_uncertainty",
    ]
    for seed, sub in selected.groupby("acquisition_seed", dropna=False):
        sub = sub.copy()
        score_path = run_dir / f"seed{int(seed)}_round1_detector_scores_v10c.csv"
        if not score_path.exists():
            out_parts.append(sub)
            continue
        scores = pd.read_csv(score_path)
        available = [c for c in score_cols if c in scores.columns]
        if not available or "sample_id" not in scores.columns:
            out_parts.append(sub)
            continue
        merged = sub.merge(scores[["sample_id", *available]].drop_duplicates("sample_id"), on="sample_id", how="left", suffixes=("", "_score"))
        for col in available:
            score_col = f"{col}_score"
            if score_col not in merged.columns:
                continue
            if col in merged.columns:
                merged[col] = merged[col].combine_first(merged[score_col])
            else:
                merged[col] = merged[score_col]
            merged = merged.drop(columns=[score_col])
        out_parts.append(merged)
    return pd.concat(out_parts, ignore_index=True, sort=False) if out_parts else selected.copy()


def enrich_selected(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        out = row.to_dict()
        out.update(parse_xml_for_image(row))
        pred = normalize_class_name(out.get("detector_pred_class", ""))
        gt_classes = set(str(out.get("gt_classes", "")).split("|")) if out.get("gt_classes") else set()
        out["pred_class_in_gt_classes"] = bool(pred in gt_classes) if pred and pred != "__no_box__" else False
        out["class_hint_matches_gt_majority"] = (
            normalize_class_name(out.get("class_hint", "")) == normalize_class_name(out.get("gt_majority_class", ""))
        )
        out["detector_pseudo_box_count"] = safe_float(out.get("detector_pseudo_box_count"))
        out["detector_uncertainty"] = safe_float(out.get("detector_uncertainty"))
        out["detector_max_conf"] = safe_float(out.get("detector_max_conf"))
        out["pseudo_minus_gt_box_count"] = out["detector_pseudo_box_count"] - safe_float(out.get("gt_box_count"))
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_selection(audited: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, strategy), sub in audited.groupby(["acquisition_seed", "strategy"], dropna=False):
        gt_counts = sub["gt_majority_class"].fillna("").astype(str).replace("", np.nan).dropna().value_counts()
        pred_counts = sub["detector_pred_class"].fillna("").astype(str).replace("", np.nan).dropna().value_counts()
        rows.append(
            {
                "acquisition_seed": int(seed),
                "strategy": strategy,
                "n": int(len(sub)),
                "gt_class_entropy": entropy_from_counts(gt_counts),
                "pred_class_entropy": entropy_from_counts(pred_counts),
                "gt_coverage_neu6": int(len(set(gt_counts.index) & set(NEU6))),
                "pred_coverage_neu6": int(len(set(normalize_class_name(c) for c in pred_counts.index) & set(NEU6))),
                "gt_box_count_mean": float(pd.to_numeric(sub["gt_box_count"], errors="coerce").mean()),
                "gt_box_count_median": float(pd.to_numeric(sub["gt_box_count"], errors="coerce").median()),
                "multi_instance_rate": float(pd.to_numeric(sub["multi_instance"], errors="coerce").mean()),
                "gt_area_frac_mean": float(pd.to_numeric(sub["gt_area_frac_mean"], errors="coerce").mean()),
                "detector_pseudo_box_count_mean": float(pd.to_numeric(sub["detector_pseudo_box_count"], errors="coerce").mean()),
                "detector_uncertainty_mean": float(pd.to_numeric(sub["detector_uncertainty"], errors="coerce").mean()),
                "detector_max_conf_mean": float(pd.to_numeric(sub["detector_max_conf"], errors="coerce").mean()),
                "pred_class_in_gt_rate": float(sub["pred_class_in_gt_classes"].mean()) if len(sub) else np.nan,
                "class_hint_matches_gt_majority_rate": float(sub["class_hint_matches_gt_majority"].mean()) if len(sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def class_distribution(audited: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, strategy), sub in audited.groupby(["acquisition_seed", "strategy"], dropna=False):
        total = len(sub)
        counts = sub["gt_majority_class"].fillna("").astype(str).map(normalize_class_name).value_counts()
        for cls in NEU6:
            count = int(counts.get(cls, 0))
            rows.append(
                {
                    "acquisition_seed": int(seed),
                    "strategy": strategy,
                    "class_name": cls,
                    "count": count,
                    "fraction": count / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def compute_metric_deltas(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = ["map50", "map5095", "precision", "recall", "f1"]
    round1 = results[pd.to_numeric(results.get("round"), errors="coerce").eq(1)].copy()
    for seed, sub in round1.groupby("acquisition_seed", dropna=False):
        random = sub[sub["strategy"].astype(str).eq(RANDOM_STRATEGY)]
        v10c = sub[sub["strategy"].astype(str).eq(V10C_STRATEGY)]
        if random.empty or v10c.empty:
            continue
        row = {"acquisition_seed": int(seed)}
        for metric in metric_cols:
            r = safe_float(random.iloc[0].get(metric))
            v = safe_float(v10c.iloc[0].get(metric))
            row[f"random_{metric}"] = r
            row[f"v10c24_{metric}"] = v
            row[f"v10c24_minus_random_{metric}"] = v - r
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_delta(per_class: pd.DataFrame, dist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    good = per_class[pd.to_numeric(per_class.get("round"), errors="coerce").eq(1)].copy()
    for (seed, cls), sub in good.groupby(["acquisition_seed", "class_name"], dropna=False):
        random = sub[sub["strategy"].astype(str).eq(RANDOM_STRATEGY)]
        v10c = sub[sub["strategy"].astype(str).eq(V10C_STRATEGY)]
        if random.empty or v10c.empty:
            continue
        r_dist = dist[(dist["acquisition_seed"].eq(seed)) & dist["strategy"].eq(RANDOM_STRATEGY) & dist["class_name"].eq(cls)]
        v_dist = dist[(dist["acquisition_seed"].eq(seed)) & dist["strategy"].eq(V10C_STRATEGY) & dist["class_name"].eq(cls)]
        rows.append(
            {
                "acquisition_seed": int(seed),
                "class_name": normalize_class_name(cls),
                "random_ap5095": safe_float(random.iloc[0].get("ap5095")),
                "v10c24_ap5095": safe_float(v10c.iloc[0].get("ap5095")),
                "v10c24_minus_random_ap5095": safe_float(v10c.iloc[0].get("ap5095")) - safe_float(random.iloc[0].get("ap5095")),
                "random_selected_count": int(r_dist.iloc[0]["count"]) if len(r_dist) else 0,
                "v10c24_selected_count": int(v_dist.iloc[0]["count"]) if len(v_dist) else 0,
                "v10c24_minus_random_selected_count": (int(v_dist.iloc[0]["count"]) if len(v_dist) else 0)
                - (int(r_dist.iloc[0]["count"]) if len(r_dist) else 0),
            }
        )
    return pd.DataFrame(rows)


def add_difference_table(selection_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric_cols = [
        c
        for c in selection_summary.columns
        if c not in {"acquisition_seed", "strategy"} and pd.api.types.is_numeric_dtype(selection_summary[c])
    ]
    for seed, sub in selection_summary.groupby("acquisition_seed", dropna=False):
        random = sub[sub["strategy"].eq(RANDOM_STRATEGY)]
        v10c = sub[sub["strategy"].eq(V10C_STRATEGY)]
        if random.empty or v10c.empty:
            continue
        row = {"acquisition_seed": int(seed), "comparison": "V10c24-Random"}
        for col in numeric_cols:
            row[f"diff_{col}"] = safe_float(v10c.iloc[0].get(col)) - safe_float(random.iloc[0].get(col))
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(
    out_dir: Path,
    run_dir: Path,
    selection_summary: pd.DataFrame,
    selection_diffs: pd.DataFrame,
    metric_deltas: pd.DataFrame,
    class_delta: pd.DataFrame,
) -> None:
    map_diff = pd.to_numeric(metric_deltas.get("v10c24_minus_random_map5095"), errors="coerce").dropna()
    recall_diff = pd.to_numeric(metric_deltas.get("v10c24_minus_random_recall"), errors="coerce").dropna()
    class_mean = (
        class_delta.groupby("class_name", dropna=False)["v10c24_minus_random_ap5095"].mean().reset_index()
        if len(class_delta)
        else pd.DataFrame()
    )
    lines = [
        "# Random baseline audit vs V10c24",
        "",
        f"- Source run: `{run_dir}`",
        "- Training performed by this script: `False`",
        "- Final test used: `False`",
        "- GT XML usage: `post-hoc audit only`",
        "",
        "## Core finding",
        "",
        "This audit checks whether Random is strong because it already covers the NEU acquisition space reasonably well. "
        "If Random has high class entropy, broad class coverage, comparable multi-instance rate, and no obvious detector-quality disadvantage, "
        "then small detector-aware heuristics will be hard to separate from it at budget 90.",
        "",
        "## V10c24 - Random metric deltas",
        "",
        table_md(metric_deltas),
        "",
        f"- Mean mAP50-95 delta: `{map_diff.mean() if len(map_diff) else math.nan:.6f}`",
        f"- Mean recall delta: `{recall_diff.mean() if len(recall_diff) else math.nan:.6f}`",
        "",
        "## Selection summary",
        "",
        table_md(selection_summary),
        "",
        "## Selection property deltas",
        "",
        table_md(selection_diffs),
        "",
        "## Mean per-class AP50-95 delta",
        "",
        table_md(class_mean),
        "",
        "## Interpretation guardrail",
        "",
        "- This does not prove Random is universally optimal.",
        "- It only tests whether, in the current NEU large-pool protocol, Random is already a strong and diverse sampler.",
        "- Do not tune a new selector from this audit; use it to decide whether V10c24 deserves only the pre-planned budget-120 smoke.",
    ]
    (out_dir / "random_baseline_audit_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v10c24-run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()

    run_dir = args.v10c24_run
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"random_baseline_audit_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = load_required_csv(run_dir, "v10c_selected_samples.csv")
    results = load_required_csv(run_dir, "v10c_onecycle_results.csv")
    per_class = load_required_csv(run_dir, "v10c_per_class_metrics.csv")

    selected = selected[
        pd.to_numeric(selected["round"], errors="coerce").eq(1)
        & selected["strategy"].astype(str).isin([RANDOM_STRATEGY, V10C_STRATEGY])
    ].copy()
    selected = attach_detector_scores(run_dir, selected)
    audited = enrich_selected(selected)
    selection_summary = summarize_selection(audited)
    selection_diffs = add_difference_table(selection_summary)
    dist = class_distribution(audited)
    metric_deltas = compute_metric_deltas(results)
    class_delta = per_class_delta(per_class, dist)

    audited.to_csv(out_dir / "audited_selected_samples.csv", index=False, encoding="utf-8-sig")
    selection_summary.to_csv(out_dir / "selection_group_summary.csv", index=False, encoding="utf-8-sig")
    selection_diffs.to_csv(out_dir / "selection_property_deltas_v10c24_minus_random.csv", index=False, encoding="utf-8-sig")
    dist.to_csv(out_dir / "class_distribution_by_strategy.csv", index=False, encoding="utf-8-sig")
    metric_deltas.to_csv(out_dir / "metric_deltas_by_seed.csv", index=False, encoding="utf-8-sig")
    class_delta.to_csv(out_dir / "per_class_delta_with_selection_distribution.csv", index=False, encoding="utf-8-sig")
    write_summary(out_dir, run_dir, selection_summary, selection_diffs, metric_deltas, class_delta)

    mean_map = pd.to_numeric(metric_deltas.get("v10c24_minus_random_map5095"), errors="coerce").mean()
    mean_recall = pd.to_numeric(metric_deltas.get("v10c24_minus_random_recall"), errors="coerce").mean()
    print("=" * 100)
    print("[DONE] Random baseline audit vs V10c24")
    print(f"Output dir: {out_dir}")
    print(f"Rows audited: {len(audited)}")
    print(f"Mean V10c24-Random mAP50-95 diff: {mean_map:.6f}")
    print(f"Mean V10c24-Random recall diff   : {mean_recall:.6f}")
    print("Training performed=False")
    print("Final test used=False")
    print("GT used only post-hoc=True")
    print("=" * 100)


if __name__ == "__main__":
    main()

"""Integrate frozen research outputs into a traceable three-dataset condition map.

Analysis-only contract:
- never reads a path containing ``final`` or ``locked``;
- never trains a model, runs inference, or implements a selector;
- preserves protocol-specific endpoints instead of averaging across tasks.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
OUT = ROOT / "runs" / "integrated_condition_map_20260718"

MASTER_PATH = DOCS / "condition_map_master_20260718.csv"
BRANCH_PATH = DOCS / "branch_decision_table_20260718.csv"
DECISION_PATH = DOCS / "three_dataset_condition_map_decision_20260718.md"
FN_PATH = DOCS / "fn_failure_prediction_extension_feasibility_20260718.md"

SCANNED: list[Path] = []
REGISTRY: list[dict[str, Any]] = []


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def assert_safe_source(path: Path) -> None:
    low = str(path).lower()
    if "final" in low or "locked" in low:
        raise RuntimeError(f"Locked/final path is prohibited: {path}")
    if not path.exists():
        raise FileNotFoundError(path)


def scan(path: Path) -> None:
    assert_safe_source(path)
    resolved = path.resolve()
    if resolved not in SCANNED:
        SCANNED.append(resolved)


def read_csv(path: Path) -> pd.DataFrame:
    scan(path)
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    scan(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_text(path: Path) -> str:
    scan(path)
    return path.read_text(encoding="utf-8-sig")


def discover_relevant_artifacts() -> None:
    """Inventory relevant summaries/tables/configs without touching locked final files."""
    roots = [
        ROOT / "runs/random_baseline_audit_v10c24",
        ROOT / "runs/seed45_fixed_set_stability/seed45_fixed_set_stability_main",
        ROOT / "runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main",
        ROOT / "runs/active_learning_v10c24_round2_scale_smoke/v10c24_round2_scale_smoke_20260713_225326",
        ROOT / "runs/gc10_taxonomy_protocol/gc10_protocol_20260715",
        ROOT / "runs/gc10_taxonomy_selection_audit/gc10_random_vs_dino_200seed_20260715",
        ROOT / "runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715",
        ROOT / "runs/gc10_discovery_representation_audit/gc10_d2r_200seed_20260715",
        ROOT / "runs/mpdd_annotation_triage_protocol/mpdd_protocol_20260715",
        ROOT / "runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715",
        ROOT / "runs/visa_annotation_triage_protocol/visa_protocol_v2_20260715",
        ROOT / "runs/visa_selection_only_audit/visa_random_vs_visual_200seed_20260715",
        ROOT / "runs/dcal_xai/gc10_r1",
        ROOT / "runs/dcal_xai/v2_budget_extended",
        ROOT / "runs/dcal_xai/v2_backbone_main",
        ROOT / "runs/dcal_xai/v2_detector_signal_validity",
        ROOT / "runs/vlm_consistency_groundedness_validity/oracle_crop_diagnostic20_gc10_20260715",
        ROOT / "runs/vlm_consistency_groundedness_validity/paired_model_comparison_gc10_20260715",
    ]
    allowed = {".csv", ".json", ".md", ".txt", ".yaml"}
    excluded_parts = {"yolo_cache", "train_runs", "predictions", "responses", "oracle_crops_gt_posthoc_only"}
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in allowed:
                continue
            lower_parts = {part.lower() for part in path.parts}
            low = str(path).lower()
            if lower_parts & excluded_parts or "final" in low or "locked" in low:
                continue
            scan(path)


def register(
    metric_id: str,
    dataset: str,
    protocol: str,
    source: Path,
    locator: str,
    value: Any,
    verification_status: str = "verified_original_artifact",
    note: str = "",
) -> Any:
    assert_safe_source(source)
    REGISTRY.append(
        {
            "metric_id": metric_id,
            "dataset": dataset,
            "protocol": protocol,
            "value": value,
            "source_file": rel(source),
            "source_locator": locator,
            "verification_status": verification_status,
            "note": note,
        }
    )
    return value


def csv_value(
    path: Path,
    filters: dict[str, Any],
    column: str,
    metric_id: str,
    dataset: str,
    protocol: str,
    expected: float | str | bool | None = None,
) -> Any:
    df = read_csv(path)
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        if key not in df.columns:
            raise KeyError(f"{path}: missing filter column {key}")
        mask &= df[key].astype(str) == str(value)
    selected = df.loc[mask]
    if len(selected) != 1:
        raise RuntimeError(f"{path}: filters {filters} produced {len(selected)} rows")
    if column not in selected.columns:
        raise KeyError(f"{path}: missing value column {column}")
    value = selected.iloc[0][column]
    if isinstance(value, np.generic):
        value = value.item()
    if expected is not None:
        if isinstance(expected, float):
            if not np.isclose(float(value), expected, atol=1e-9, rtol=1e-8):
                raise AssertionError(f"{metric_id}: observed {value}, expected {expected}")
        elif str(value).lower() != str(expected).lower():
            raise AssertionError(f"{metric_id}: observed {value}, expected {expected}")
    locator = "; ".join(f"{k}={v}" for k, v in filters.items()) + f"; column={column}"
    return register(metric_id, dataset, protocol, path, locator, value)


def json_value(
    path: Path,
    keys: list[str],
    metric_id: str,
    dataset: str,
    protocol: str,
    expected: float | str | bool | None = None,
) -> Any:
    data: Any = read_json(path)
    for key in keys:
        data = data[key]
    value = data
    if expected is not None:
        if isinstance(expected, float):
            if not np.isclose(float(value), expected, atol=1e-9, rtol=1e-8):
                raise AssertionError(f"{metric_id}: observed {value}, expected {expected}")
        elif str(value).lower() != str(expected).lower():
            raise AssertionError(f"{metric_id}: observed {value}, expected {expected}")
    return register(metric_id, dataset, protocol, path, "JSON key=" + ".".join(keys), value)


def mean_from_rows(
    path: Path,
    filters: dict[str, Any],
    column: str,
    metric_id: str,
    dataset: str,
    protocol: str,
) -> float:
    df = read_csv(path)
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        mask &= df[key].astype(str) == str(value)
    values = pd.to_numeric(df.loc[mask, column], errors="raise")
    if values.empty:
        raise RuntimeError(f"No values for {path} {filters} {column}")
    value = float(values.mean())
    locator = "; ".join(f"{k}={v}" for k, v in filters.items()) + f"; mean(column={column}, n={len(values)})"
    return register(metric_id, dataset, protocol, path, locator, value)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def build_metrics() -> dict[str, Any]:
    discover_relevant_artifacts()
    m: dict[str, Any] = {}

    gc10_sel = ROOT / "runs/gc10_taxonomy_selection_audit/gc10_random_vs_dino_200seed_20260715/gc10_metric_summary.csv"
    gc10_gate = ROOT / "runs/gc10_taxonomy_selection_audit/gc10_random_vs_dino_200seed_20260715/gc10_selection_only_gate.csv"
    m["gc10_rare_random"] = csv_value(gc10_sel, {"metric": "query_images_with_rare_class"}, "random_mean", "gc10.rare_images.random", "GC10-DET", "gc10_dino_selection_200seed", 2.17)
    m["gc10_rare_dino"] = csv_value(gc10_sel, {"metric": "query_images_with_rare_class"}, "dino_mean", "gc10.rare_images.dino", "GC10-DET", "gc10_dino_selection_200seed", 4.89)
    m["gc10_rare_gain"] = csv_value(gc10_sel, {"metric": "query_images_with_rare_class"}, "mean_difference", "gc10.rare_images.gain", "GC10-DET", "gc10_dino_selection_200seed", 2.72)
    m["gc10_rare_class_gain"] = csv_value(gc10_sel, {"metric": "query_unique_rare_classes"}, "mean_difference", "gc10.rare_classes.gain", "GC10-DET", "gc10_dino_selection_200seed", 1.21)
    m["gc10_classes_random"] = csv_value(gc10_sel, {"metric": "combined_unique_classes"}, "random_mean", "gc10.combined_classes.random", "GC10-DET", "gc10_dino_selection_200seed", 9.07)
    m["gc10_classes_dino"] = csv_value(gc10_sel, {"metric": "combined_unique_classes"}, "dino_mean", "gc10.combined_classes.dino", "GC10-DET", "gc10_dino_selection_200seed", 9.895)
    m["gc10_class_gain"] = csv_value(gc10_sel, {"metric": "combined_unique_classes"}, "mean_difference", "gc10.combined_classes.gain", "GC10-DET", "gc10_dino_selection_200seed", 0.825)
    m["gc10_redundancy_delta"] = csv_value(gc10_sel, {"metric": "query_pairwise_cosine_similarity_mean"}, "mean_difference", "gc10.pairwise_similarity.delta", "GC10-DET", "gc10_dino_selection_200seed", -0.25252962455153466)
    gc10_gate_df = read_csv(gc10_gate)
    m["gc10_gate_pass"] = bool(gc10_gate_df["passed"].astype(str).str.lower().eq("true").all())
    register("gc10.selection_gate.all_checks", "GC10-DET", "gc10_dino_selection_200seed", gc10_gate, "all rows; column=passed", m["gc10_gate_pass"])

    visa = ROOT / "runs/visa_selection_only_audit/visa_random_vs_visual_200seed_20260715/visa_metric_summary.csv"
    visa_gate = ROOT / "runs/visa_selection_only_audit/visa_random_vs_visual_200seed_20260715/visa_selection_only_gate.csv"
    for key, metric, col, expected in [
        ("visa_anom_random", "query_anomaly_images", "random_mean", 2.155),
        ("visa_anom_visual", "query_anomaly_images", "visual_mean", 16.635),
        ("visa_anom_gain", "query_anomaly_images", "mean_difference", 14.48),
        ("visa_cat_random", "query_unique_object_categories", "random_mean", 9.82),
        ("visa_cat_visual", "query_unique_object_categories", "visual_mean", 5.71),
        ("visa_cat_delta", "query_unique_object_categories", "mean_difference", -4.11),
        ("visa_redundancy_delta", "query_pairwise_cosine_similarity_mean", "mean_difference", 0.15425971314311027),
    ]:
        m[key] = csv_value(visa, {"metric": metric}, col, "visa." + key, "VisA", "visa_visual_200seed", expected)
    visa_gate_df = read_csv(visa_gate)
    m["visa_gate_pass"] = bool(visa_gate_df["passed"].astype(str).str.lower().eq("true").all())
    register("visa.selection_gate.all_checks", "VisA", "visa_visual_200seed", visa_gate, "all rows; column=passed", m["visa_gate_pass"])

    mpdd = ROOT / "runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_metric_summary.csv"
    mpdd_gate = ROOT / "runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_selection_only_gate.csv"
    mpdd_source = ROOT / "runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_source_origin_paired_summary.csv"
    comp = "FrozenCategoryBalancedDINO-minus-GTFreeRandom"
    for key, metric, col, expected in [
        ("mpdd_anom_random", "query_anomaly_images", "baseline_mean", 4.06),
        ("mpdd_anom_dino", "query_anomaly_images", "method_mean", 10.305),
        ("mpdd_anom_gain", "query_anomaly_images", "mean_difference", 6.245),
        ("mpdd_cat_random", "query_unique_product_categories", "baseline_mean", 5.8),
        ("mpdd_cat_dino", "query_unique_product_categories", "method_mean", 5.475),
        ("mpdd_cat_delta", "query_unique_product_categories", "mean_difference", -0.325),
        ("mpdd_redundancy_delta", "query_pairwise_cosine_similarity_mean", "mean_difference", -0.04673488311469555),
    ]:
        m[key] = csv_value(mpdd, {"comparison": comp, "metric": metric}, col, "mpdd." + key, "MPDD", "mpdd_hierarchical_dino_200seed", expected)
    m["mpdd_official_test_delta"] = csv_value(mpdd_source, {"comparison": comp, "metric": "official_test_images"}, "mean_difference", "mpdd.official_test_origin_count.delta", "MPDD", "mpdd_source_origin_audit", 6.81)
    mpdd_gate_df = read_csv(mpdd_gate)
    m["mpdd_gate_pass"] = bool(mpdd_gate_df["passed"].astype(str).str.lower().eq("true").all())
    register("mpdd.selection_gate.all_checks", "MPDD", "mpdd_hierarchical_dino_200seed", mpdd_gate, "all rows; column=passed", m["mpdd_gate_pass"])

    det_summary = ROOT / "runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_acquisition_seed_summary.csv"
    det_rare = ROOT / "runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_class_group_macro_summary.csv"
    det_gate = ROOT / "runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_development_confirmation_gate.csv"
    for key, metric, col, expected in [
        ("det_map_random", "map5095", "random_mean", 0.12423537570071104),
        ("det_map_dino", "map5095", "dino_mean", 0.14161378263258018),
        ("det_map_delta", "map5095", "mean_difference", 0.017378406931869143),
        ("det_recall_delta", "recall", "mean_difference", 0.03538142140432067),
    ]:
        m[key] = csv_value(det_summary, {"metric": metric}, col, "gc10.detector." + key, "GC10-DET", "gc10_dino_detector_5acq_3train", expected)
    m["det_rare_random"] = csv_value(det_rare, {"group": "rare"}, "random_mean", "gc10.detector.rare_macro.random", "GC10-DET", "gc10_dino_detector_5acq_3train", 0.03650748874772366)
    m["det_rare_dino"] = csv_value(det_rare, {"group": "rare"}, "dino_mean", "gc10.detector.rare_macro.dino", "GC10-DET", "gc10_dino_detector_5acq_3train", 0.016630743202297307)
    m["det_rare_delta"] = csv_value(det_rare, {"group": "rare"}, "mean_difference", "gc10.detector.rare_macro.delta", "GC10-DET", "gc10_dino_detector_5acq_3train", -0.019876745545426348)
    det_gate_df = read_csv(det_gate)
    m["det_gate_pass"] = bool(det_gate_df["passed"].astype(str).str.lower().eq("true").all())
    register("gc10.detector.gate.all_checks", "GC10-DET", "gc10_dino_detector_5acq_3train", det_gate, "all rows; column=passed", m["det_gate_pass"])

    k40_budget = ROOT / "runs/dcal_xai/v2_budget_extended/budget_summary.csv"
    k40_decision = ROOT / "runs/dcal_xai/v2_budget_extended/decision.json"
    for split in ["design", "holdout"]:
        prefix = f"k40_{split}"
        for policy, suffix in [("Random", "random"), ("DINOClusterCoverageK40", "k40")]:
            filters = {"split": split, "budget": 140, "policy": policy}
            for field in ["all_classes_rate", "rare_images_mean", "unique_production_groups_mean", "pairwise_cosine_similarity_mean"]:
                m[f"{prefix}_{field}_{suffix}"] = csv_value(k40_budget, filters, field, f"gc10.k40.{split}.{field}.{suffix}", "GC10-DET", f"k40_budget_{split}")
    m["k40_holdout_rare_gain"] = m["k40_holdout_rare_images_mean_k40"] - m["k40_holdout_rare_images_mean_random"]
    register("gc10.k40.holdout.rare_images.gain", "GC10-DET", "k40_budget_holdout", k40_budget, "computed: holdout budget=140 K40 rare_images_mean - Random rare_images_mean", m["k40_holdout_rare_gain"])
    k40_dec = read_json(k40_decision)
    m["k40_selection_gate"] = bool(k40_dec["holdout_gate_pass"])
    register("gc10.k40.holdout_gate", "GC10-DET", "k40_budget_holdout", k40_decision, "JSON key=holdout_gate_pass", m["k40_selection_gate"])
    for key, val in k40_dec["thresholds"].items():
        register("gc10.k40.threshold." + key, "GC10-DET", "k40_budget_holdout", k40_decision, "JSON key=thresholds." + key, val)

    k40_metrics = ROOT / "runs/dcal_xai/v2_backbone_main/v8n_screen_metrics.csv"
    k40_contrasts = ROOT / "runs/dcal_xai/v2_backbone_main/v8n_screen_contrasts.csv"
    k40_class = ROOT / "runs/dcal_xai/v2_backbone_main/v8n_screen_class_summary.csv"
    k40_screen = ROOT / "runs/dcal_xai/v2_backbone_main/v8n_screen_decision.json"
    m["k40_map_random"] = mean_from_rows(k40_metrics, {"policy": "Random140", "backbone": "YOLOv8n"}, "map5095", "gc10.k40.detector.map.random", "GC10-DET", "k40_yolov8n_5acq_3train")
    m["k40_map_strategy"] = mean_from_rows(k40_metrics, {"policy": "ClusterK40_140", "backbone": "YOLOv8n"}, "map5095", "gc10.k40.detector.map.strategy", "GC10-DET", "k40_yolov8n_5acq_3train")
    for key, metric, expected in [
        ("k40_map_delta", "map5095", -0.0016777445999850433),
        ("k40_recall_delta", "recall", -0.021870824914440996),
    ]:
        m[key] = csv_value(k40_contrasts, {"contrast": "ClusterK40_140-Random140|backbone=YOLOv8n", "metric": metric}, "mean_difference", "gc10.k40.detector." + key, "GC10-DET", "k40_yolov8n_5acq_3train", expected)
    rare_random = read_csv(k40_class).query("policy == 'Random140' and class_id in [8, 9, 10]")["ap5095"].mean()
    rare_k40 = read_csv(k40_class).query("policy == 'ClusterK40_140' and class_id in [8, 9, 10]")["ap5095"].mean()
    m["k40_rare_random"] = register("gc10.k40.detector.rare_macro.random", "GC10-DET", "k40_yolov8n_5acq_3train", k40_class, "rows policy=Random140,class_id in {8,9,10}; mean(column=ap5095)", float(rare_random))
    m["k40_rare_strategy"] = register("gc10.k40.detector.rare_macro.strategy", "GC10-DET", "k40_yolov8n_5acq_3train", k40_class, "rows policy=ClusterK40_140,class_id in {8,9,10}; mean(column=ap5095)", float(rare_k40))
    m["k40_rare_delta"] = json_value(k40_screen, ["rare_macro_difference"], "gc10.k40.detector.rare_macro.delta", "GC10-DET", "k40_yolov8n_5acq_3train", -0.018289535113315297)
    m["k40_screen_status"] = json_value(k40_screen, ["status"], "gc10.k40.detector.gate", "GC10-DET", "k40_yolov8n_5acq_3train", "FAIL")
    m["k40_screen_models"] = json_value(k40_screen, ["models"], "gc10.k40.detector.models", "GC10-DET", "k40_yolov8n_5acq_3train", 30)

    fixed = ROOT / "runs/seed45_fixed_set_stability/seed45_fixed_set_stability_main/aggregate_metrics.csv"
    fixed_gate = ROOT / "runs/seed45_fixed_set_stability/seed45_fixed_set_stability_main/screen_gate.csv"
    for set_key, suffix in [("Random20", "random"), ("Visual20", "visual")]:
        m[f"seed45_map_{suffix}"] = csv_value(fixed, {"set_key": set_key, "metric": "map5095"}, "mean", f"neu.seed45.map.{suffix}", "NEU-DET", "seed45_fixed_set_5train")
    m["seed45_map_delta"] = csv_value(fixed_gate, {"treatment": "Visual20"}, "mean_map5095_difference", "neu.seed45.map.delta", "NEU-DET", "seed45_fixed_set_5train", 0.01623566489340474)
    m["seed45_recall_delta"] = csv_value(fixed_gate, {"treatment": "Visual20"}, "mean_recall_difference", "neu.seed45.recall.delta", "NEU-DET", "seed45_fixed_set_5train", 0.05757014067724466)
    m["seed45_gate"] = csv_value(fixed_gate, {"treatment": "Visual20"}, "gate_pass", "neu.seed45.gate", "NEU-DET", "seed45_fixed_set_5train", True)

    independent = ROOT / "runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main/confirmatory_metric_summary.csv"
    independent_gate = ROOT / "runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main/confirmatory_gate.csv"
    for key, metric, col, expected in [
        ("ind_map_random", "map5095", "random_mean", 0.1643532),
        ("ind_map_visual", "map5095", "visual_mean", 0.1713719),
        ("ind_map_delta", "map5095", "mean_difference", 0.0070187),
        ("ind_recall_delta", "recall", "mean_difference", 0.0101867),
    ]:
        m[key] = csv_value(independent, {"metric": metric}, col, "neu.independent." + key, "NEU-DET", "v8_independent_acquisition_10seed", expected)
    ind_gate_df = read_csv(independent_gate)
    m["ind_gate"] = bool(ind_gate_df["passed"].astype(str).str.lower().eq("true").all())
    register("neu.independent.gate.all_checks", "NEU-DET", "v8_independent_acquisition_10seed", independent_gate, "all rows; column=passed", m["ind_gate"])

    v10c = ROOT / "runs/active_learning_v10c24_round2_scale_smoke/v10c24_round2_scale_smoke_20260713_225326/roundwise_random_v10c24_comparison.csv"
    v10c_gate = ROOT / "runs/active_learning_v10c24_round2_scale_smoke/v10c24_round2_scale_smoke_20260713_225326/round2_scale_gate.csv"
    for key, col, expected in [
        ("v10c_map_random", "random_map5095", 0.31433295658303634),
        ("v10c_map_strategy", "v10c24_map5095", 0.3094698949199987),
        ("v10c_map_delta", "v10c24_minus_random_map5095", -0.004863061663037627),
        ("v10c_recall_delta", "v10c24_minus_random_recall", 0.0009474423815040156),
    ]:
        m[key] = csv_value(v10c, {"acquisition_seed": 51, "round": 2, "budget": 120}, col, "neu.v10c24." + key, "NEU-DET", "v10c24_seed51_round2", expected)
    m["v10c_gate"] = csv_value(v10c_gate, {"acquisition_seed": 51}, "gate_pass", "neu.v10c24.gate", "NEU-DET", "v10c24_seed51_round2", False)

    r1_metrics = ROOT / "runs/dcal_xai/gc10_r1/selection_metrics_posthoc.csv"
    r1_comp = ROOT / "runs/dcal_xai/gc10_r1/selection_comparisons.csv"
    r1_gate = ROOT / "runs/dcal_xai/gc10_r1/selection_gate.csv"
    for strategy, suffix in [("Random", "random"), ("DetectorDifficultyDiversity", "hybrid")]:
        m[f"r1_rare_{suffix}"] = mean_from_rows(r1_metrics, {"strategy": strategy}, "query_images_with_rare_class", f"gc10.r1.rare_images.{suffix}", "GC10-DET", "dcal_xai_r1_5seed")
        m[f"r1_classes_{suffix}"] = mean_from_rows(r1_metrics, {"strategy": strategy}, "combined_unique_classes", f"gc10.r1.combined_classes.{suffix}", "GC10-DET", "dcal_xai_r1_5seed")
    for key, metric, expected in [
        ("r1_rare_gain", "query_images_with_rare_class", 2.0),
        ("r1_class_gain", "combined_unique_classes", 0.6),
        ("r1_rare_class_gain", "query_unique_rare_classes", 1.0),
        ("r1_instance_delta", "query_instances", -2.2),
    ]:
        m[key] = csv_value(r1_comp, {"primary": "DetectorDifficultyDiversity", "comparator": "Random", "metric": metric}, "mean_difference", "gc10.r1." + key, "GC10-DET", "dcal_xai_r1_5seed", expected)
    r1_gate_df = read_csv(r1_gate)
    m["r1_gate"] = bool(r1_gate_df["passed"].astype(str).str.lower().eq("true").all())
    register("gc10.r1.gate.all_checks", "GC10-DET", "dcal_xai_r1_5seed", r1_gate, "all rows; column=passed", m["r1_gate"])

    d2r = ROOT / "runs/gc10_discovery_representation_audit/gc10_d2r_200seed_20260715/label_aware_retrieval_audit/label_aware_class8_summary.csv"
    d2r_gate = ROOT / "runs/gc10_discovery_representation_audit/gc10_d2r_200seed_20260715/label_aware_retrieval_audit/label_aware_class8_recovery_gate.csv"
    m["d2r_base_hit"] = csv_value(d2r, {"method": "centroid_similarity", "target_class": 8}, "top1_hit_rate", "gc10.d2r.class8.centroid.top1", "GC10-DET", "d2r_label_aware_frozen_replay", 0.1168091168091168)
    m["d2r_best_hit"] = csv_value(d2r, {"method": "nearest_target_exemplar", "target_class": 8}, "top1_hit_rate", "gc10.d2r.class8.nearest.top1", "GC10-DET", "d2r_label_aware_frozen_replay", 0.15384615384615385)
    m["d2r_gain"] = csv_value(d2r, {"method": "nearest_target_exemplar", "target_class": 8}, "top1_hit_improvement_over_centroid", "gc10.d2r.class8.nearest.gain", "GC10-DET", "d2r_label_aware_frozen_replay", 0.03703703703703705)
    d2r_gate_df = read_csv(d2r_gate)
    m["d2r_gate"] = bool(d2r_gate_df["passed"].astype(str).str.lower().eq("true").all())
    register("gc10.d2r.gate.all_checks", "GC10-DET", "d2r_label_aware_frozen_replay", d2r_gate, "all rows; column=passed", m["d2r_gate"])

    v23_decision = ROOT / "runs/dcal_xai/v2_detector_signal_validity/decision.json"
    v23_summary = ROOT / "runs/dcal_xai/v2_detector_signal_validity/signal_validity_summary.csv"
    for key, keys, expected in [
        ("v23_total_enrich", ["primary_metrics", "error_enrichment_mean"], 1.1159008047688797),
        ("v23_fn_enrich", ["primary_metrics", "fn_enrichment_mean"], 1.3796926704945667),
        ("v23_rare_fn_enrich", ["primary_metrics", "rare_fn_enrichment_mean"], 1.7859103354085288),
        ("v23_auroc", ["primary_metrics", "auroc_mean"], 0.7664320782804007),
        ("v23_spearman", ["primary_metrics", "spearman_mean"], 0.2937771068450421),
        ("v23_stability", ["mean_training_seed_rank_stability"], 0.6523785175936274),
        ("v23_status", ["status"], "FAIL"),
    ]:
        m[key] = json_value(v23_decision, keys, "gc10.v23." + key, "GC10-DET", "v23_detector_signal_validity", expected)
    m["v23_conf_rare"] = csv_value(v23_summary, {"signal": "ensemble_confidence_uncertainty"}, "rare_fn_enrichment_mean", "gc10.v23.confidence.rare_fn_enrichment", "GC10-DET", "v23_detector_signal_validity", 2.683021841592697)
    register("gc10.v23.threshold.error_enrichment_mean", "GC10-DET", "v23_detector_signal_validity", ROOT / "scripts/04_dcal_xai/v2_detector_signal_validity_protocol.md", "Frozen primary gate bullet 1", 1.50)
    scan(ROOT / "scripts/04_dcal_xai/v2_detector_signal_validity_protocol.md")

    oracle_metrics = ROOT / "runs/vlm_consistency_groundedness_validity/oracle_crop_diagnostic20_gc10_20260715/oracle_crop_audit/oracle_crop_diagnostic_metrics.csv"
    oracle_gate = ROOT / "runs/vlm_consistency_groundedness_validity/oracle_crop_diagnostic20_gc10_20260715/oracle_crop_audit/oracle_crop_diagnostic_gate.csv"
    for key, metric, expected in [
        ("oracle_parse", "mean_json_parse_rate", 1.0),
        ("oracle_positive", "mean_positive_presence_rate", 0.0),
        ("oracle_evidence", "mean_defect_evidence_rate", 0.0),
        ("oracle_iou", "median_bbox_iou_to_oracle_gt", 0.0),
    ]:
        m[key] = csv_value(oracle_metrics, {"metric": metric}, "value", "vlm.oracle." + key, "GC10-DET", "vlm_oracle_crop_20", expected)
    oracle_gate_df = read_csv(oracle_gate)
    m["oracle_gate"] = bool(oracle_gate_df["result"].astype(str).str.upper().eq("PASS").all())
    register("vlm.oracle.gate.all_checks", "GC10-DET", "vlm_oracle_crop_20", oracle_gate, "all rows; column=result", m["oracle_gate"])

    paired = ROOT / "runs/vlm_consistency_groundedness_validity/paired_model_comparison_gc10_20260715/comparison/paired_model_gate_comparison.csv"
    paired_df = read_csv(paired)
    m["vlm_models_passed"] = int(paired_df["gate_pass"].astype(str).str.lower().eq("true").sum())
    register("vlm.paired.models_passed", "GC10-DET", "paired_vlm_model_comparison", paired, "all rows; sum(column=gate_pass)", m["vlm_models_passed"])
    m["vlm_best_bacc"] = float(paired_df["balanced_accuracy"].max())
    register("vlm.paired.best_balanced_accuracy", "GC10-DET", "paired_vlm_model_comparison", paired, "all rows; max(column=balanced_accuracy)", m["vlm_best_bacc"])

    # Required chronology documents are read for audit context, not as verified numeric sources.
    for p in [
        ROOT / "docs/research_status_external_review_prompt_20260718.txt",
        ROOT / "docs/validity_gated_workflow_and_backbone_research_basis_20260715.md",
        ROOT / "docs/reframed_gt_free_al_workflow_20260713.md",
        ROOT / "docs/research_context_handoff_20260712.md",
        ROOT / "runs/random_baseline_audit_v10c24/random_baseline_audit_20260713_225216/random_baseline_audit_summary.md",
    ]:
        read_text(p)

    return m


MASTER_COLUMNS = [
    "dataset", "task_type", "annotation_type", "pool_condition", "target_definition", "target_prevalence",
    "strategy", "baseline", "acquisition_budget", "acquisition_seed_count", "training_seed_count",
    "target_yield_baseline", "target_yield_strategy", "target_yield_gain", "target_enrichment",
    "category_coverage_baseline", "category_coverage_strategy", "category_coverage_delta",
    "rare_image_delta", "rare_class_delta", "source_coverage_delta", "production_group_delta",
    "redundancy_metric", "redundancy_delta", "downstream_metric_name", "downstream_baseline",
    "downstream_strategy", "downstream_delta", "rare_metric_name", "rare_baseline", "rare_strategy",
    "rare_delta", "recall_delta", "gate_name", "gate_threshold", "gate_result", "evidence_level",
    "confound", "interpretation", "source_file",
]


def master_row(**kwargs: Any) -> dict[str, Any]:
    row = {c: None for c in MASTER_COLUMNS}
    row.update(kwargs)
    missing = set(kwargs) - set(MASTER_COLUMNS)
    if missing:
        raise KeyError(missing)
    return row


def build_master(m: dict[str, Any]) -> pd.DataFrame:
    gc10_sel_src = "runs/gc10_taxonomy_selection_audit/gc10_random_vs_dino_200seed_20260715/gc10_metric_summary.csv"
    rows = [
        master_row(dataset="GC10-DET", task_type="object_detection", annotation_type="native_bbox_10_class", pool_condition="all-defect cold-start taxonomy; initial20/query20", target_definition="query images containing rare classes 8/9/10", target_prevalence="all images defective; rare target heterogeneous", strategy="FrozenDINOVisualDiversity", baseline="GTFreeRandom", acquisition_budget=20, acquisition_seed_count=200, training_seed_count=None, target_yield_baseline=m["gc10_rare_random"], target_yield_strategy=m["gc10_rare_dino"], target_yield_gain=m["gc10_rare_gain"], target_enrichment=m["gc10_rare_dino"] / m["gc10_rare_random"], category_coverage_baseline=m["gc10_classes_random"], category_coverage_strategy=m["gc10_classes_dino"], category_coverage_delta=m["gc10_class_gain"], rare_image_delta=m["gc10_rare_gain"], rare_class_delta=m["gc10_rare_class_gain"], redundancy_metric="query_pairwise_cosine_similarity_mean", redundancy_delta=m["gc10_redundancy_delta"], gate_name="GC10 taxonomy selection-only", gate_threshold="all 11 frozen checks; rare-image gain>=0.75; combined-class gain>=0.25", gate_result="PASS", evidence_level="confirmatory_pass", confound="full-image DINO may encode production texture; downstream evaluated separately", interpretation="Equal-budget discovery and taxonomy gain passed, but does not itself establish detector utility.", source_file=gc10_sel_src + "; runs/gc10_taxonomy_selection_audit/gc10_random_vs_dino_200seed_20260715/gc10_selection_only_gate.csv"),
        master_row(dataset="GC10-DET", task_type="object_detection", annotation_type="native_bbox_10_class", pool_condition="selected-set detector translation; initial20/query20", target_definition="detector development utility from frozen selected sets", target_prevalence="not a discovery endpoint", strategy="FrozenDINOVisualDiversity", baseline="GTFreeRandom", acquisition_budget=20, acquisition_seed_count=5, training_seed_count=3, downstream_metric_name="YOLOv8n development mAP50-95", downstream_baseline=m["det_map_random"], downstream_strategy=m["det_map_dino"], downstream_delta=m["det_map_delta"], rare_metric_name="rare classes 8/9/10 macro AP50-95", rare_baseline=m["det_rare_random"], rare_strategy=m["det_rare_dino"], rare_delta=m["det_rare_delta"], recall_delta=m["det_recall_delta"], gate_name="GC10 detector development confirmation", gate_threshold="mAP gain>=0.010 and rare macro AP gain>=0.020 plus class/recall safety", gate_result="FAIL", evidence_level="confirmatory_fail", confound="five acquisition realizations; development split reused; 15 paired training realizations are not 15 independent acquisitions", interpretation="Overall mAP increased +0.017378, but rare macro AP decreased -0.019877; translation gate failed.", source_file="runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_acquisition_seed_summary.csv; runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_class_group_macro_summary.csv; runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_development_confirmation_gate.csv"),
        master_row(dataset="MPDD", task_type="industrial_anomaly_localization", annotation_type="pixel_mask_and_derived_bbox", pool_condition="normal-centered heterogeneous product/source pool", target_definition="anomaly images in 20-image review query", target_prevalence="sparse anomaly within mixed official train/test-origin pool", strategy="FrozenCategoryBalancedDINO", baseline="GTFreeRandom", acquisition_budget=20, acquisition_seed_count=200, training_seed_count=None, target_yield_baseline=m["mpdd_anom_random"], target_yield_strategy=m["mpdd_anom_dino"], target_yield_gain=m["mpdd_anom_gain"], target_enrichment=m["mpdd_anom_dino"] / m["mpdd_anom_random"], category_coverage_baseline=m["mpdd_cat_random"], category_coverage_strategy=m["mpdd_cat_dino"], category_coverage_delta=m["mpdd_cat_delta"], redundancy_metric="query_pairwise_cosine_similarity_mean", redundancy_delta=m["mpdd_redundancy_delta"], gate_name="MPDD selection-only anomaly+category safety", gate_threshold="anomaly gain>=2; CI low>0; win rate>=0.65; category delta>=-0.25", gate_result="FAIL", evidence_level="confirmatory_fail", confound=f"official-test-origin image count increased by {m['mpdd_official_test_delta']:.2f}/20; source composition explains part of gain", interpretation="Equal-budget anomaly discovery increased +6.245, while product-category coverage delta -0.325 violated safety.", source_file="runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_metric_summary.csv; runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_selection_only_gate.csv; runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_source_origin_paired_summary.csv"),
        master_row(dataset="VisA", task_type="industrial_anomaly_discovery", annotation_type="image_label_and_pixel_mask", pool_condition="anomaly-sparse 12-product pool", target_definition="anomaly images in 20-image review query", target_prevalence="960 anomalies in 8,650-image acquisition pool (protocol audit)", strategy="FrozenDINOVisualDiversity", baseline="GTFreeRandom", acquisition_budget=20, acquisition_seed_count=200, training_seed_count=None, target_yield_baseline=m["visa_anom_random"], target_yield_strategy=m["visa_anom_visual"], target_yield_gain=m["visa_anom_gain"], target_enrichment=m["visa_anom_visual"] / m["visa_anom_random"], category_coverage_baseline=m["visa_cat_random"], category_coverage_strategy=m["visa_cat_visual"], category_coverage_delta=m["visa_cat_delta"], redundancy_metric="query_pairwise_cosine_similarity_mean", redundancy_delta=m["visa_redundancy_delta"], gate_name="VisA selection-only anomaly+category safety", gate_threshold="anomaly gain>=2; CI low>0; win rate>=0.65; category coverage not worse", gate_result="FAIL", evidence_level="confirmatory_fail", confound="strategy concentrated on anomaly-rich visual/product modes; category collapse", interpretation="Largest discovery gain (+14.480/20) co-occurred with -4.110 object categories; success cannot be called coverage-safe.", source_file="runs/visa_selection_only_audit/visa_random_vs_visual_200seed_20260715/visa_metric_summary.csv; runs/visa_selection_only_audit/visa_random_vs_visual_200seed_20260715/visa_selection_only_gate.csv"),
        master_row(dataset="NEU-DET", task_type="object_detection", annotation_type="native_bbox_6_class", pool_condition="seed45 frozen cold-start set; initial15+query5", target_definition="development detector mAP50-95", target_prevalence="balanced six-class dataset", strategy="Visual20", baseline="Random20", acquisition_budget=20, acquisition_seed_count=1, training_seed_count=5, downstream_metric_name="YOLO development mAP50-95", downstream_baseline=m["seed45_map_random"], downstream_strategy=m["seed45_map_visual"], downstream_delta=m["seed45_map_delta"], recall_delta=m["seed45_recall_delta"], gate_name="fixed-set training-seed stability", gate_threshold="mAP gain>=0.01; >=4/5 wins; recall>=-0.01; class safety", gate_result="PASS", evidence_level="diagnostic", confound="single favorable acquisition realization selected post-observation", interpretation="Training-seed stability (5/5 wins) shows a set effect, not acquisition-set generalization.", source_file="runs/seed45_fixed_set_stability/seed45_fixed_set_stability_main/aggregate_metrics.csv; runs/seed45_fixed_set_stability/seed45_fixed_set_stability_main/screen_gate.csv"),
        master_row(dataset="NEU-DET", task_type="object_detection", annotation_type="native_bbox_6_class", pool_condition="independent cold-start acquisitions; initial15+query5", target_definition="development detector mAP50-95", target_prevalence="balanced six-class dataset", strategy="Frozen Visual diversity", baseline="GTFreeRandom", acquisition_budget=20, acquisition_seed_count=10, training_seed_count=1, downstream_metric_name="YOLO development mAP50-95", downstream_baseline=m["ind_map_random"], downstream_strategy=m["ind_map_visual"], downstream_delta=m["ind_map_delta"], recall_delta=m["ind_recall_delta"], gate_name="independent acquisition confirmation", gate_threshold="mean mAP gain>=0.01; >=7/10 wins; exact sign-flip p<=0.05; recall nonnegative", gate_result="FAIL", evidence_level="independent_confirmation_fail", confound="one training seed per acquisition; descriptive bootstrap CI includes zero", interpretation="The fixed-set effect did not generalize across acquisition realizations.", source_file="runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main/confirmatory_metric_summary.csv; runs/active_learning_v8_cold_start_confirmation/v8_cold_start_visual_confirm_main/confirmatory_gate.csv"),
        master_row(dataset="NEU-DET", task_type="object_detection", annotation_type="native_bbox_6_class", pool_condition="balanced large-pool V10c24 scale extension seed51 round2", target_definition="development detector mAP50-95", target_prevalence="balanced large-pool", strategy="DetectorRecallGuardDINOInstanceV10c", baseline="GTFreeRandom", acquisition_budget=120, acquisition_seed_count=1, training_seed_count=1, downstream_metric_name="YOLO development mAP50-95", downstream_baseline=m["v10c_map_random"], downstream_strategy=m["v10c_map_strategy"], downstream_delta=m["v10c_map_delta"], recall_delta=m["v10c_recall_delta"], gate_name="V10c24 round2 scale gate", gate_threshold="six frozen scale checks; positive mAP/recall/F1 and class safety", gate_result="FAIL", evidence_level="confirmatory_fail", confound="single scale-smoke acquisition seed", interpretation="At budget120, mAP50-95 delta was -0.004863; the candidate did not scale.", source_file="runs/active_learning_v10c24_round2_scale_smoke/v10c24_round2_scale_smoke_20260713_225326/roundwise_random_v10c24_comparison.csv; runs/active_learning_v10c24_round2_scale_smoke/v10c24_round2_scale_smoke_20260713_225326/round2_scale_gate.csv"),
        master_row(dataset="GC10-DET", task_type="object_detection", annotation_type="native_bbox_10_class", pool_condition="DCAL-XAI R1 detector difficulty+flip disagreement; initial20/query20", target_definition="rare-image and taxonomy discovery with instance noninferiority", target_prevalence="all-defect pool; rare classes 8/9/10", strategy="DetectorDifficultyDiversity", baseline="Random", acquisition_budget=20, acquisition_seed_count=5, training_seed_count=None, target_yield_baseline=m["r1_rare_random"], target_yield_strategy=m["r1_rare_hybrid"], target_yield_gain=m["r1_rare_gain"], target_enrichment=m["r1_rare_hybrid"] / m["r1_rare_random"], category_coverage_baseline=m["r1_classes_random"], category_coverage_strategy=m["r1_classes_hybrid"], category_coverage_delta=m["r1_class_gain"], rare_image_delta=m["r1_rare_gain"], rare_class_delta=m["r1_rare_class_gain"], gate_name="DCAL-XAI R1 selection gate", gate_threshold="all seven checks; query instances noninferiority vs Random", gate_result="FAIL", evidence_level="confirmatory_fail", confound="94/100 choices dominated by one-view flip detections; geometry/augmentation confound", interpretation="Rare coverage improved, but query instances -2.2 failed the frozen gate and the uncertainty mechanism was invalid.", source_file="runs/dcal_xai/gc10_r1/selection_metrics_posthoc.csv; runs/dcal_xai/gc10_r1/selection_comparisons.csv; runs/dcal_xai/gc10_r1/selection_gate.csv"),
        master_row(dataset="GC10-DET", task_type="object_detection", annotation_type="native_bbox_10_class", pool_condition="budget140 design selection", target_definition="rare images and all-class coverage at initial budget", target_prevalence="all-defect pool; rare classes 8/9/10", strategy="DINOClusterCoverageK40", baseline="Random140", acquisition_budget=140, acquisition_seed_count=200, training_seed_count=None, target_yield_baseline=m["k40_design_rare_images_mean_random"], target_yield_strategy=m["k40_design_rare_images_mean_k40"], target_yield_gain=m["k40_design_rare_images_mean_k40"] - m["k40_design_rare_images_mean_random"], target_enrichment=m["k40_design_rare_images_mean_k40"] / m["k40_design_rare_images_mean_random"], category_coverage_baseline=m["k40_design_all_classes_rate_random"], category_coverage_strategy=m["k40_design_all_classes_rate_k40"], category_coverage_delta=m["k40_design_all_classes_rate_k40"] - m["k40_design_all_classes_rate_random"], rare_image_delta=m["k40_design_rare_images_mean_k40"] - m["k40_design_rare_images_mean_random"], production_group_delta=m["k40_design_unique_production_groups_mean_k40"] - m["k40_design_unique_production_groups_mean_random"], redundancy_metric="pairwise cosine similarity mean", redundancy_delta=m["k40_design_pairwise_cosine_similarity_mean_k40"] - m["k40_design_pairwise_cosine_similarity_mean_random"], gate_name="V2.1 design candidate gate", gate_threshold="all-class>=0.90; all-rare>=0.90; min-two>=0.75; production/instance/redundancy safety", gate_result="PASS", evidence_level="exploratory_positive", confound="candidate selected on design seeds", interpretation="Design-stage coverage candidate; cannot be upgraded to confirmation.", source_file="runs/dcal_xai/v2_budget_extended/budget_summary.csv; runs/dcal_xai/v2_budget_extended/decision.json"),
        master_row(dataset="GC10-DET", task_type="object_detection", annotation_type="native_bbox_10_class", pool_condition="budget140 independent holdout selection", target_definition="rare images and all-class coverage at initial budget", target_prevalence="all-defect pool; rare classes 8/9/10", strategy="DINOClusterCoverageK40", baseline="Random140", acquisition_budget=140, acquisition_seed_count=200, training_seed_count=None, target_yield_baseline=m["k40_holdout_rare_images_mean_random"], target_yield_strategy=m["k40_holdout_rare_images_mean_k40"], target_yield_gain=m["k40_holdout_rare_gain"], target_enrichment=m["k40_holdout_rare_images_mean_k40"] / m["k40_holdout_rare_images_mean_random"], category_coverage_baseline=m["k40_holdout_all_classes_rate_random"], category_coverage_strategy=m["k40_holdout_all_classes_rate_k40"], category_coverage_delta=m["k40_holdout_all_classes_rate_k40"] - m["k40_holdout_all_classes_rate_random"], rare_image_delta=m["k40_holdout_rare_gain"], production_group_delta=m["k40_holdout_unique_production_groups_mean_k40"] - m["k40_holdout_unique_production_groups_mean_random"], redundancy_metric="pairwise cosine similarity mean", redundancy_delta=m["k40_holdout_pairwise_cosine_similarity_mean_k40"] - m["k40_holdout_pairwise_cosine_similarity_mean_random"], gate_name="V2.1 holdout selection gate", gate_threshold="unchanged design thresholds: all-class>=0.90; all-rare>=0.90; min-two>=0.75; safety", gate_result="PASS", evidence_level="confirmatory_pass", confound="selection endpoint is coverage, not learner utility", interpretation="Holdout coverage safety passed and authorized only the frozen YOLOv8n screen.", source_file="runs/dcal_xai/v2_budget_extended/budget_summary.csv; runs/dcal_xai/v2_budget_extended/decision.json"),
        master_row(dataset="GC10-DET", task_type="object_detection", annotation_type="native_bbox_10_class", pool_condition="budget140 YOLOv8n downstream translation", target_definition="development detector learning utility", target_prevalence="all-defect pool", strategy="DINOClusterCoverageK40", baseline="Random140", acquisition_budget=140, acquisition_seed_count=5, training_seed_count=3, downstream_metric_name="YOLOv8n development mAP50-95", downstream_baseline=m["k40_map_random"], downstream_strategy=m["k40_map_strategy"], downstream_delta=m["k40_map_delta"], rare_metric_name="rare classes 8/9/10 macro AP50-95", rare_baseline=m["k40_rare_random"], rare_strategy=m["k40_rare_strategy"], rare_delta=m["k40_rare_delta"], recall_delta=m["k40_recall_delta"], gate_name="V2.2 YOLOv8n screen", gate_threshold="mAP noninferiority; rare macro noninferiority; worst-class safety; stability; recall noninferiority", gate_result="FAIL", evidence_level="confirmatory_fail", confound="development only; acquisition seed is inferential unit", interpretation="Coverage gain did not translate: mAP -0.001678, rare macro AP -0.018290, recall -0.021871.", source_file="runs/dcal_xai/v2_backbone_main/v8n_screen_metrics.csv; runs/dcal_xai/v2_backbone_main/v8n_screen_contrasts.csv; runs/dcal_xai/v2_backbone_main/v8n_screen_decision.json"),
        master_row(dataset="GC10-DET", task_type="object_detection", annotation_type="native_bbox_10_class", pool_condition="frozen 1,000-state label-aware retrieval replay", target_definition="class-8 top-1 retrieval hit", target_prevalence="class 8 target states only (351 rounds; 130 acquisition seeds)", strategy="nearest_target_exemplar", baseline="centroid_similarity", acquisition_budget=32, acquisition_seed_count=130, training_seed_count=None, target_yield_baseline=m["d2r_base_hit"], target_yield_strategy=m["d2r_best_hit"], target_yield_gain=m["d2r_gain"], target_enrichment=m["d2r_best_hit"] / m["d2r_base_hit"], gate_name="class-8 representation recovery", gate_threshold="top1>=0.50; CI low>=0.40; P@5>=0.40; top32 presence>=0.80; gain>=0.25", gate_result="FAIL", evidence_level="diagnostic", confound="frozen replay only; alternative choices not fed into later states", interpretation="Best top-1 hit 0.153846 remained far below 0.50; representation branch is not recoverable.", source_file="runs/gc10_discovery_representation_audit/gc10_d2r_200seed_20260715/label_aware_retrieval_audit/label_aware_class8_summary.csv; runs/gc10_discovery_representation_audit/gc10_d2r_200seed_20260715/label_aware_retrieval_audit/label_aware_class8_recovery_gate.csv"),
        master_row(dataset="GC10-DET", task_type="object_detection", annotation_type="native_bbox_10_class", pool_condition="existing Random140 checkpoints on 232-image development split", target_definition="top-20% total FP+FN error enrichment", target_prevalence="error prevalence varies by checkpoint; not pooled across training seeds", strategy="ensemble_combined_uncertainty", baseline="uniform random review", acquisition_budget=47, acquisition_seed_count=5, training_seed_count=3, target_enrichment=m["v23_total_enrich"], downstream_metric_name="majority-error AUROC", downstream_baseline=0.5, downstream_strategy=m["v23_auroc"], downstream_delta=m["v23_auroc"] - 0.5, gate_name="V2.3 detector signal validity", gate_threshold="total-error enrichment>=1.50; CI low>1; AUROC>=0.65; Spearman>=0.20; stability>=0.50", gate_result="FAIL", evidence_level="confirmatory_fail", confound="development split reused; total error mixes FP and FN; 5 acquisition units", interpretation="Six of seven checks passed, but the frozen primary effect-size gate failed at 1.115901.", source_file="runs/dcal_xai/v2_detector_signal_validity/decision.json; runs/dcal_xai/v2_detector_signal_validity/signal_validity_summary.csv"),
        master_row(dataset="GC10-DET", task_type="object_detection", annotation_type="native_bbox_10_class", pool_condition="same frozen V2.3 development audit", target_definition="top-20% false-negative enrichment (secondary diagnostic)", target_prevalence="rare GT: 18 images / 22 boxes", strategy="ensemble_combined_uncertainty_FN_diagnostic", baseline="uniform random review", acquisition_budget=47, acquisition_seed_count=5, training_seed_count=3, target_enrichment=m["v23_fn_enrich"], rare_metric_name="rare-FN enrichment", rare_baseline=1.0, rare_strategy=m["v23_rare_fn_enrich"], rare_delta=m["v23_rare_fn_enrich"] - 1.0, gate_name="not a preregistered primary gate", gate_threshold="none; hypothesis generation only", gate_result="NOT_PRIMARY", evidence_level="exploratory_positive", confound="post-hoc endpoint; rare counts only 18 images/22 boxes; no untouched validation", interpretation="FN 1.379693 and rare-FN 1.785910 are hypothesis-generating, not a rescued AL PASS.", source_file="runs/dcal_xai/v2_detector_signal_validity/decision.json; runs/dcal_xai/v2_detector_signal_validity/signal_validity_summary.csv"),
        master_row(dataset="GC10-DET", task_type="vlm_signal_validity", annotation_type="oracle_bbox_crop_posthoc_diagnostic", pool_condition="20 positive oracle crops", target_definition="defect presence/evidence and grounded bbox response", target_prevalence="20/20 positive by construction", strategy="Qwen2-VL structured validity response", baseline="frozen validity thresholds", acquisition_budget=20, acquisition_seed_count=None, training_seed_count=None, target_yield_strategy=m["oracle_positive"], gate_name="oracle-crop VLM validity", gate_threshold="parse>=0.90; informative>=0.90; positive presence>=0.50; defect evidence>=0.70", gate_result="FAIL", evidence_level="diagnostic", confound="oracle crop is easier than deployable full-image selection; still all-negative presence collapse", interpretation="Parse/informative rates were 1.0, but presence, evidence and median IoU were 0.", source_file="runs/vlm_consistency_groundedness_validity/oracle_crop_diagnostic20_gc10_20260715/oracle_crop_audit/oracle_crop_diagnostic_metrics.csv; runs/vlm_consistency_groundedness_validity/oracle_crop_diagnostic20_gc10_20260715/oracle_crop_audit/oracle_crop_diagnostic_gate.csv"),
        master_row(dataset="GC10-DET", task_type="vlm_signal_validity", annotation_type="paired_positive_background_bbox_audit", pool_condition="40 paired views per model", target_definition="balanced presence discrimination and bbox grounding", target_prevalence="balanced paired positive/background views", strategy="Qwen2/Qwen3/Qwen2.5 paired comparison", baseline="frozen model-compliance gate", acquisition_budget=40, acquisition_seed_count=None, training_seed_count=None, downstream_metric_name="best paired balanced accuracy", downstream_strategy=m["vlm_best_bacc"], gate_name="paired VLM model comparison", gate_threshold="all frozen compliance, sensitivity, specificity and grounding checks", gate_result="FAIL", evidence_level="confirmatory_fail", confound="three small Qwen-family checkpoints; collapse mode differs by model", interpretation="0/3 models passed; best balanced accuracy 0.70 had median bbox IoU 0.", source_file="runs/vlm_consistency_groundedness_validity/paired_model_comparison_gc10_20260715/comparison/paired_model_gate_comparison.csv"),
    ]
    df = pd.DataFrame(rows, columns=MASTER_COLUMNS)
    df.to_csv(MASTER_PATH, index=False, encoding="utf-8-sig")
    return df


BRANCH_COLUMNS = ["branch", "original_hypothesis", "dataset", "protocol", "primary_endpoint", "frozen_gate", "observed_result", "decision", "stopping_reason", "surviving_signal", "prevented_next_step", "prevented_model_runs", "final_test_consumed", "reusable_asset", "manuscript_role"]


def build_branches(m: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ["VLM consistency", "language consistency predicts annotation utility", "GC10-DET", "structured full-image pilot", "presence/grounding validity", "frozen structured validity gate", "schema output existed but acquisition-semantic validity collapsed", "CLOSE", "signal validity prerequisite failed", "VLM only as evidence-constrained interface candidate", "VLM-scored acquisition and detector training", None, False, "structured response JSONL and scoring/audit code", "negative mechanism evidence"],
        ["VLM groundedness/oracle crop", "better localization context restores grounded defect evidence", "GC10-DET", "20 oracle crops", "positive presence, evidence, bbox IoU", "presence>=0.50; evidence>=0.70", "presence=0; evidence=0; median IoU=0; FAIL", "CLOSE", "failure persisted under oracle localization", "none as acquisition signal", "prompt/selector expansion", None, False, "oracle crop manifest and GT audit", "strong falsification diagnostic"],
        ["paired VLM model comparison", "collapse is model-specific and another small VLM restores validity", "GC10-DET", "3 models x 40 paired views", "paired balanced accuracy plus grounding", "all frozen paired compliance and grounding checks", f"0/3 pass; best balanced accuracy={m['vlm_best_bacc']:.2f}, median IoU=0", "CLOSE", "different models showed different collapse modes", "none as acquisition signal", "further small-model prompt/model search", None, False, "paired manifests, responses, three audit tables", "cross-model negative evidence"],
        ["NEU DINO cold-start seed45", "visual diversity helps initial cold-start", "NEU-DET", "one frozen acquisition set x 5 training seeds", "mAP50-95 stability", "gain>=0.01 and 4/5 wins plus safety", f"PASS: mAP delta={m['seed45_map_delta']:.6f}; 5/5", "DIAGNOSTIC_ONLY", "one acquisition realization cannot establish selector generalization", "fixed-set learner effect", "claim of general cold-start superiority", None, False, "5-seed checkpoints and fixed manifests", "motivation for acquisition confirmation"],
        ["independent acquisition confirmation", "seed45 effect generalizes across acquisition sets", "NEU-DET", "10 new acquisition seeds", "paired mAP50-95", "gain>=0.01; 7/10; p<=0.05; recall>=0", f"FAIL: delta={m['ind_map_delta']:.6f}; CI crosses 0; p=0.322266", "CLOSE", "acquisition-set generalization failed", "training stability does not imply acquisition stability", "cold-start selector claim", None, False, "10-seed frozen results and gate", "generalization failure evidence"],
        ["V10c24 scale extension", "detector-aware selector scales at budget120", "NEU-DET", "seed51 round2 budget120", "mAP50-95 and recall/class safety", "six frozen scale checks", f"FAIL: mAP delta={m['v10c_map_delta']:.6f}", "CLOSE", "round2 scale gate failed", "Random is strong in balanced large pools", "additional V10c/V10d scaling", None, False, "recovered roundwise and per-class CSV", "large-pool boundary condition"],
        ["GC10 global DINO selection-only", "global DINO improves rare/taxonomy discovery safely", "GC10-DET", "initial20/query20, 200 seeds", "rare images/classes plus taxonomy/instance safety", "11 frozen checks", f"PASS: rare images +{m['gc10_rare_gain']:.3f}; rare classes +{m['gc10_rare_class_gain']:.3f}", "TRANSLATION_REQUIRED", "selection PASS cannot substitute for learner utility", "rare/taxonomy discovery", "none at selection stage; authorized one frozen detector confirmation", 0, False, "selection records, DINO cache, gate", "positive discovery case"],
        ["MPDD global DINO selection-only", "global DINO improves anomaly discovery without coverage loss", "MPDD", "query20, 200 seeds", "anomaly yield plus product-category safety", "category delta>=-0.25", f"FAIL: anomaly +{m['mpdd_anom_gain']:.3f}; category {m['mpdd_cat_delta']:.3f}", "CLOSE", "coverage safety failed and source-origin confound is large", "anomaly discovery under source shift", "downstream anomaly-model experiment", None, False, "selection records and source-origin decomposition", "discovery/coverage trade-off"],
        ["VisA global DINO selection-only", "global DINO improves sparse anomaly discovery without category collapse", "VisA", "query20, 200 seeds", "anomaly yield plus 12-category safety", "category coverage not worse", f"FAIL: anomaly +{m['visa_anom_gain']:.3f}; categories {m['visa_cat_delta']:.3f}", "CLOSE", "severe category collapse", "sparse anomaly discovery", "downstream anomaly-model experiment", None, False, "selection records and leave-one-category audit", "strong discovery/coverage trade-off"],
        ["GC10 D2R label-aware retrieval", "label-aware global retrieval repairs class-8 representation", "GC10-DET", "1,000 frozen replay states", "class-8 top1/P@5/top32 recovery", "top1>=0.50 and six additional checks", f"FAIL: best top1={m['d2r_best_hit']:.6f}; 0 eligible methods", "CLOSE", "class-8 representation was not recovered", "none", "15-model D2R detector confirmation", 15, False, "frozen replay states and class-8 audit", "representation failure mechanism"],
        ["DCAL-XAI flip disagreement", "flip disagreement is epistemic detector uncertainty", "GC10-DET", "initial20/query20, 5 acquisitions", "coverage with instance noninferiority", "all seven R1 checks", f"FAIL: query instances {m['r1_instance_delta']:+.1f}; 94/100 one-view detections", "CLOSE", "augmentation geometry dominated signal", "detector difficulty descriptive signal only", "hybrid detector-training expansion", None, False, "sealed scores, selections, grounded explanations", "uncertainty confound evidence"],
        ["K40 budget selection-only", "larger initial budget and cluster coverage prevent taxonomy omission", "GC10-DET", "budget140 design+holdout, 200+200 seeds", "all-class/all-rare/min-two coverage with safety", "frozen design thresholds reused on holdout", "PASS: holdout all-class 0.955 vs 0.940", "TRANSLATION_REQUIRED", "selection gate authorizes only YOLOv8n screen", "taxonomy omission risk reduction", "none; authorized one 30-model YOLOv8n screen", 0, False, "400-seed budget metrics, manifests, DINO cache", "coverage-positive case"],
        ["K40 YOLOv8n downstream", "coverage gain translates to detector and rare AP", "GC10-DET", "5 acquisitions x 3 training seeds x 2 policies", "mAP50-95, recall, rare/worst-class safety", "five frozen downstream checks", f"FAIL: mAP {m['k40_map_delta']:+.6f}; rare {m['k40_rare_delta']:+.6f}; recall {m['k40_recall_delta']:+.6f}", "CLOSE", "rare macro and recall gates failed", "coverage-utility gap", "30 YOLOv8s expansion models", 30, False, "30 YOLOv8n models, per-class metrics", "primary translation failure case"],
        ["detector uncertainty V2.3", "detector-native ensemble uncertainty enriches total FP+FN enough for AL", "GC10-DET", "15 existing checkpoints; 232 development images", "top20% total-error enrichment", "enrichment>=1.50 plus CI/AUROC/correlation/stability", f"FAIL: total={m['v23_total_enrich']:.6f}; AUROC={m['v23_auroc']:.6f}", "CLOSE_AS_AL", "primary effect size 1.115901 below 1.50", "FN-focused diagnostic only", "selection-only query audit and any retraining", 0, False, "15 hashed checkpoints, sealed predictions, signal audit", "failed primary with secondary signal"],
        ["FN/rare-miss exploratory signal", "confidence/no-detection may support missed-defect inspection triage", "GC10-DET", "post-hoc secondary V2.3 endpoints", "FN and rare-FN enrichment", "no preregistered primary gate", f"FN={m['v23_fn_enrich']:.6f}; rare-FN={m['v23_rare_fn_enrich']:.6f}; confidence rare-FN={m['v23_conf_rare']:.6f}", "HYPOTHESIS_ONLY", "post-hoc, 18 rare images/22 boxes, no untouched validation", "possible FN inspection triage, not AL utility", "feature-misalignment implementation or detector training", 0, False, "existing event-level error audit", "exploratory limitation/future protocol only"],
    ]
    df = pd.DataFrame(rows, columns=BRANCH_COLUMNS)
    df.to_csv(BRANCH_PATH, index=False, encoding="utf-8-sig")
    return df


def setup_plotting() -> None:
    plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False, "figure.dpi": 130, "savefig.dpi": 180})
    FIGURES.mkdir(parents=True, exist_ok=True)


def plot_discovery_coverage(m: dict[str, Any]) -> None:
    setup_plotting()
    panels = {
        "GC10-DET": [
            (m["gc10_rare_gain"], m["gc10_classes_dino"] / m["gc10_classes_random"] - 1, "DINO q20"),
            (m["r1_rare_gain"], m["r1_classes_hybrid"] / m["r1_classes_random"] - 1, "DCAL q20"),
            (m["k40_holdout_rare_gain"], m["k40_holdout_all_classes_rate_k40"] / m["k40_holdout_all_classes_rate_random"] - 1, "K40 b140"),
        ],
        "MPDD": [(m["mpdd_anom_gain"], m["mpdd_cat_dino"] / m["mpdd_cat_random"] - 1, "Cat-balanced DINO")],
        "VisA": [(m["visa_anom_gain"], m["visa_cat_visual"] / m["visa_cat_random"] - 1, "Visual DINO")],
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), sharey=True)
    colors = {"GC10-DET": "#1f77b4", "MPDD": "#ff7f0e", "VisA": "#2ca02c"}
    for ax, (dataset, points) in zip(axes, panels.items()):
        ax.axhline(0, color="0.35", lw=1)
        for x, y, label in points:
            ax.scatter(x, y, s=70, color=colors[dataset], edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(label, (x, y), xytext=(5, 6), textcoords="offset points", fontsize=8)
        ax.set_title(dataset)
        ax.set_xlabel("Target discovery gain vs Random (images)")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Coverage safety retention − 1")
    fig.suptitle("Discovery gain can trade off against category/coverage safety", fontsize=13)
    fig.text(0.5, 0.01, "Panels preserve task definitions. y uses strategy coverage / Random coverage − 1; GC10 uses combined-class count (q20) or all-class rate (b140), MPDD product categories, VisA object categories.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(FIGURES / "discovery_coverage_pareto.png", bbox_inches="tight")
    plt.close(fig)


def plot_translation(m: dict[str, Any]) -> None:
    setup_plotting()
    protocols = ["DINO q20", "K40 b140"]
    x = np.array([m["gc10_classes_dino"] / m["gc10_classes_random"] - 1, m["k40_holdout_all_classes_rate_k40"] / m["k40_holdout_all_classes_rate_random"] - 1])
    metrics = {
        "mAP50-95 gain": [m["det_map_delta"], m["k40_map_delta"]],
        "Recall gain": [m["det_recall_delta"], m["k40_recall_delta"]],
        "Rare macro AP gain": [m["det_rare_delta"], m["k40_rare_delta"]],
    }
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True)
    for ax, (name, vals), color in zip(axes, metrics.items(), colors):
        ax.axhline(0, color="0.35", lw=1)
        for xx, yy, label in zip(x, vals, protocols):
            ax.scatter(xx, yy, s=75, color=color, edgecolor="white", linewidth=0.8)
            ax.annotate(f"{label}\n{yy:+.4f}", (xx, yy), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_title(name)
        ax.set_xlabel("Normalized class-coverage retention gain")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Downstream metric gain")
    fig.suptitle("GC10 selection coverage did not guarantee learner or rare-class utility", fontsize=13)
    fig.text(0.5, 0.01, "x = strategy class-coverage / Random class-coverage − 1. q20 uses combined unique classes; b140 uses all-class coverage rate. Descriptive diagnostic only.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(FIGURES / "selection_learning_translation.png", bbox_inches="tight")
    plt.close(fig)


def plot_chain(m: dict[str, Any]) -> None:
    setup_plotting()
    steps = [
        ("Signal validity", "VLM oracle crop\npresence=0, IoU=0\nFAIL"),
        ("Selection validity", f"K40 holdout\nall-class .955 vs .940\nPASS"),
        ("Acquisition-set\ngeneralization", f"NEU fixed +.0162\nindependent +.0070\nFAIL"),
        ("Detector learning\nutility", f"K40 mAP {m['k40_map_delta']:+.4f}\nrare AP {m['k40_rare_delta']:+.4f}\nFAIL"),
        ("Operational value", f"V2.3 AUROC {m['v23_auroc']:.3f}\ntotal enrich {m['v23_total_enrich']:.3f}\nFAIL; FN exploratory"),
    ]
    fig, ax = plt.subplots(figsize=(14, 4.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5)
    ax.axis("off")
    xs = np.linspace(0.3, 12.3, len(steps))
    for i, ((title, body), x) in enumerate(zip(steps, xs)):
        box = FancyBboxPatch((x, 1.0), 2.35, 2.8, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor="#f3f4f6", edgecolor="#4b5563", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + 1.175, 3.25, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 1.175, 2.05, body, ha="center", va="center", fontsize=8.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 2.95, 2.4), xytext=(x + 2.4, 2.4), arrowprops=dict(arrowstyle="->", lw=1.5, color="#374151"))
    ax.set_title("Validity-to-utility translation chain: a pass at one stage does not transfer automatically", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "validity_translation_chain.png", bbox_inches="tight")
    plt.close(fig)


def plot_waterfall() -> None:
    setup_plotting()
    stages = ["VLM validity gate", "D2R selection gate", "K40 YOLOv8n gate", "V2.3 signal gate"]
    cumulative = [0, 15, 45, 45]
    increments = [None, 15, 30, 0]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.plot(range(len(stages)), cumulative, marker="o", ms=8, lw=2.4, color="#1f77b4")
    for i, (cum, inc) in enumerate(zip(cumulative, increments)):
        label = "run count not frozen" if inc is None else (f"+{inc} avoided" if inc else "blocked next audit\n(no frozen model count)")
        ax.annotate(f"{cum} exact cumulative\n{label}", (i, cum), xytext=(0, 12 if i != 2 else -42), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(range(len(stages)), stages, rotation=15, ha="right")
    ax.set_ylabel("Exactly documented prevented model runs")
    ax.set_ylim(-3, 55)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Branch gates prevented at least 45 planned detector models and protected final evaluation")
    fig.text(0.5, 0.01, "Exact count = 15 D2R models + 30 YOLOv8s expansion models. VLM inference/GPU-hours were not estimated. Locked final test consumed: 0.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES / "branch_decision_waterfall.png", bbox_inches="tight")
    plt.close(fig)


def build_decision_doc(m: dict[str, Any], branches: pd.DataFrame) -> None:
    stopped = int(branches["decision"].isin(["CLOSE", "CLOSE_AS_AL"]).sum())
    text = f"""# 냉정한 결론

**B. 일부는 성립하지만 한 가지 추가 분석이 필요하다.**

기존 결과만으로 세 데이터셋을 동일 알고리즘의 성능 검증으로 묶을 수는 없다. 그러나 동일한 20장 review budget에서 VisA anomaly +14.480, MPDD anomaly +6.245, GC10 rare image +2.720이라는 discovery gain과, 동시에 VisA category -4.110, MPDD category -0.325, GC10 downstream rare macro AP -0.019877이라는 안전·번역 실패가 확인된다. 따라서 “효용의 조건부 분리”를 보여주는 다중 case-study는 성립한다. 다만 dataset identity와 pool sparsity/source condition이 완전히 교락되어 있으므로, **기존 200-seed selection records만 이용한 prevalence/source/category-matched stratified reanalysis 한 번**이 필요하다. 새 selector나 학습은 필요하지 않다.

# 확인된 positive evidence

| Dataset / protocol | Random | Strategy | Effect | Evidence | 함께 발생한 위험 |
|---|---:|---:|---:|---|---|
| VisA anomaly query@20 | 2.155 | 16.635 | +14.480; enrichment {m['visa_anom_visual']/m['visa_anom_random']:.3f}x | frozen selection gate의 discovery 항목 통과, 전체 gate FAIL | object category -4.110 |
| MPDD anomaly query@20 | 4.060 | 10.305 | +6.245; enrichment {m['mpdd_anom_dino']/m['mpdd_anom_random']:.3f}x | frozen selection gate의 discovery 항목 통과, 전체 gate FAIL | product category -0.325; official-test origin +6.810/20 |
| GC10 rare-image query@20 | 2.170 | 4.890 | +2.720; enrichment {m['gc10_rare_dino']/m['gc10_rare_random']:.3f}x | confirmatory selection-only PASS | detector rare macro AP -0.019877 |
| GC10 K40 holdout coverage@140 | 0.940 | 0.955 | all-class rate +0.015 | independent holdout selection PASS | downstream mAP -0.001678; recall -0.021871 |
| NEU seed45 fixed set | 0.187550 | 0.203786 | mAP +0.016236; 5/5 | diagnostic fixed-set PASS | independent acquisition confirmation FAIL |

Positive는 모두 endpoint와 gate 범위 안에서만 해석했다. VisA/MPDD는 discovery effect가 크지만 전체 safety gate는 FAIL이므로 `confirmatory_pass`로 승격하지 않았다.

# 확인된 negative evidence

- NEU 독립 acquisition 10 seeds: mAP50-95 +0.007019, descriptive CI [-0.005211, 0.019678], p=0.322266; frozen gate FAIL.
- NEU V10c24 budget120: mAP50-95 -0.004863; scale gate FAIL.
- GC10 최초 DINO detector translation: mAP +0.017378이지만 rare macro AP -0.019877; 전체 gate FAIL.
- GC10 K40/140: selection holdout는 PASS였지만 YOLOv8n mAP -0.001678, rare AP -0.018290, recall -0.021871; downstream FAIL.
- VLM oracle crop: parse/informative 1.0에도 presence/evidence/median IoU가 모두 0; paired 3-model comparison 0/3 PASS.
- DCAL-XAI flip disagreement: query instances -2.2, 94/100 선택이 one-view detection에 지배되어 geometry confound로 종료.
- MPDD effect에는 official train/test-origin composition이 강하게 개입하고, VisA는 anomaly-rich category concentration을 동반한다.

# 3-dataset condition map

| Hypothesis | Supporting dataset | Contradicting dataset | Evidence strength | Remaining uncertainty |
|---|---|---|---|---|
| 1. Sparse-target pool에서 discovery gain이 커진다 | VisA +14.480; MPDD +6.245 | GC10 all-defect rare gain +2.720로 더 작음; NEU balanced pool에서 selector 우위 불안정 | medium | dataset와 prevalence가 교락; within-dataset prevalence strata 필요 |
| 2. Discovery gain은 category/source safety를 자동 보장하지 않는다 | VisA category -4.110; MPDD category -0.325/source-origin +6.810 | GC10 최초 selection은 combined class +0.825 | high | coverage 정의가 dataset별로 다름; 평균 금지 |
| 3. Representation coverage는 learning utility를 자동 보장하지 않는다 | GC10 K40 holdout PASS → mAP -0.001678, rare -0.018290 | GC10 q20 detector overall mAP +0.017378 | high for GC10 | learner/data budget 한 조건; 인과 메커니즘은 미확정 |
| 4. Learner alignment가 downstream translation을 결정한다 | GC10 q20 rare translation 실패, D2R class8 recovery 실패 | 직접 learner 교체의 positive evidence 없음 | low-medium | 현재 결과는 alignment 설명과 양립하지만 직접 조작 실험은 없음 |
| 5. Random 강도는 pool balance와 target prevalence에 의존한다 | NEU balanced large-pool Random 강세; VisA sparse target에서 큰 discovery gap | MPDD source confound와 GC10 rare definition이 단순 prevalence 해석을 방해 | medium | stratified matched analysis 필요 |

# Validity-gated workflow의 실제 가치

- 명시적으로 종료된 branch: **{stopped}개** (`CLOSE` 또는 `CLOSE_AS_AL`; diagnostic/translation-required 행 제외).
- 정확히 문서화된 방지 detector model runs: **최소 45개** = D2R 15 + K40 후속 YOLOv8s 30. 정의되지 않은 계획은 수치에 넣지 않았다.
- 보호된 locked final-test 평가: **모든 branch에서 소비 0회**.
- 구분된 failure mechanism: **6개** — VLM compliance/presence collapse, source/category concentration, global-representation/local-defect mismatch, acquisition-set non-generalization, flip geometry confound, coverage–learner utility gap.
- 재사용 자산: **15개 hash-verified Random140 checkpoints**, sealed V2.3 prediction/inference manifest 1세트, GC10/MPDD/VisA protocol·selection manifest 3계열, 각 branch gate/audit 코드와 CSV.
- GPU-hours는 원본 runtime artifact로 확인되지 않아 추정하지 않았다.

# 논문화 가능성 판정

| 수준 | 판정 | 이유 |
|---|---|---|
| 석사학위 연구로 방어 가능성 | **high** | 실패를 PASS로 바꾸지 않고, 3개 task 조건의 discovery/coverage boundary와 GC10 downstream translation을 traceable gate로 연결함 |
| 국내/국제 학술대회 논문화 가능성 | **medium** | 산업 AL의 negative/conditional evidence는 유효하나, condition effect의 within-dataset matched analysis와 명확한 scope 제한이 필요함 |
| 방법론 중심 저널 논문화 가능성 | **low** | 새 방법 성능, 외부 downstream replication, 독립 operational validation이 없고 dataset-condition 교락이 큼 |

# 반드시 줄여야 할 주장

- “새로운 AL selector가 Random을 능가한다.”
- “VLM이 detector utility를 예측한다.”
- “DINO가 rare defect selection에 일반적으로 유효하다.”
- “detector uncertainty gate가 성공했다.”
- “세 데이터셋에서 동일 알고리즘이 downstream 성능으로 검증됐다.”
- “pool sparsity가 discovery gain의 원인이다.” — 현재는 dataset과 교락된 연관이다.

# 방어 가능한 중심 주장

> 본 연구는 서로 다른 산업 결함 review pool에서 frozen visual 후보 신호의 target-discovery benefit이 category/source safety와 분리되고, GC10-DET에서는 selection coverage가 acquisition-set generalization·rare-class detector utility로 자동 번역되지 않음을 실증하며, 고비용 학습과 locked final 평가 전에 이 분리를 판정하는 validity-gated evaluation workflow를 제시한다.
"""
    DECISION_PATH.write_text(text, encoding="utf-8")


def build_fn_doc(m: dict[str, Any]) -> None:
    text = f"""# FN / missed-defect extension feasibility decision

## 최종 판정

**DESIGN_PROTOCOL_ONLY**

현재 결과는 새 feature-misalignment 구현, selection screen 또는 학습을 정당화하지 않는다. FN 신호는 별도 research question을 작성할 최소한의 hypothesis-generation 근거이지만, 독립 검증 자원과 구현 hook이 없다. 따라서 프로토콜 요건만 문서화하고 실행은 중단한다. 이 판정은 V2.3 FAIL을 사후 PASS로 바꾸지 않는다.

## 1. FN enrichment 1.379693의 의미

V2.3 combined signal의 FN enrichment는 **{m['v23_fn_enrich']:.6f}**로 무작위보다 높다. 동일 primary signal이 majority-error AUROC **{m['v23_auroc']:.6f}**, Spearman **{m['v23_spearman']:.6f}**, rank stability **{m['v23_stability']:.6f}**를 보였으므로 완전한 noise로 보기는 어렵다. 그러나 preregistered primary endpoint는 total FP+FN enrichment였고 **{m['v23_total_enrich']:.6f} < 1.50**로 FAIL했다. FN은 secondary endpoint이므로 독립 가설의 “동기”까지만 제공한다.

## 2. rare-FN 수치와 표본 한계

- combined rare-FN enrichment: **{m['v23_rare_fn_enrich']:.6f}**
- confidence-only rare-FN enrichment: **{m['v23_conf_rare']:.6f}**
- rare GT support: **18 images / 22 boxes**

이 표본에서는 클래스·이미지 몇 개의 순위 변화가 enrichment를 크게 바꾼다. confidence-only 2.683022는 사후 secondary signal이며 independent realization별 rare capture의 안정성·CI가 primary gate로 사전 고정되지 않았다. 안전성 또는 rare-defect recall 개선으로 주장할 수 없다.

## 3. untouched validation resource

현재 확인된 GC10 protocol은 acquisition, 반복 사용된 232-image development, locked final로 구성된다. V2.3은 development를 이미 사용했다. **현재 연구 주장에 쓸 수 있는 untouched non-final bbox validation resource는 확인되지 않았다.** Locked final은 계속 접근 금지다.

## 4. GC10 FN-box endpoint 정의 가능성

기술적으로는 class-aware IoU 0.50 matching과 confidence 0.25를 이미 사용하므로 `FN-box capture@20% review budget`을 정의할 수 있다. 다만 한 GT box가 여러 checkpoint에서 반복되는 구조이므로 이미지/checkpoint 반복을 독립 표본으로 취급하면 안 된다. acquisition realization 또는 독립 pool split이 inferential unit이어야 한다.

## 5. MPDD/VisA 외부 일반화

MPDD/VisA는 pixel-mask anomaly localization/discovery task다. GC10 detector FN-box와 같은 endpoint가 아니므로 기존 결과로 직접 외부 일반화를 주장할 수 없다. 별도의 anomaly score/mask miss definition 없이 YOLO FN-box를 강제하면 task mismatch가 크다.

## 6. 기존 checkpoint event-count audit

가능하다. 15개 Random140 checkpoint, sealed predictions, per-image FP/FN/rare-FN audit가 이미 존재하므로 **추가 학습·추론 없이** event count와 acquisition-realization별 capture 분포를 재집계할 수 있다. 그러나 이는 동일 development 결과의 descriptive feasibility audit일 뿐 독립 confirmation이 아니다. 현재 결과에는 5 acquisition units밖에 없다.

## 7. local feature hook 존재 여부

저장소 검색 결과, 기존 DINO cache 코드는 `last_hidden_state[:, 0]`의 global CLS embedding을 저장한다. detector 중간 feature를 위한 `register_forward_hook`/동등 hook과 DINO patch-token cache는 현재 `scripts/04_dcal_xai`에 존재하지 않는다. 따라서 local misalignment는 단순 분석이 아니라 새 구현 branch다.

## 8. global DINO 실패 원인 해결 여부

local alignment는 small/local defect semantics를 겨냥한다는 점에서 기존 실패 가설과 방향은 맞지만, **실제로 해결한다는 증거는 없다**. D2R label-aware global retrieval도 class8 best top1 0.153846으로 복구에 실패했다. 새 local method는 이름 변경이 아니라 별도 signal-validity 가설과 독립 split을 요구한다.

## 9. 계산비용

기존 event-count 재집계는 CSV 연산만 필요하다. 새 local detector feature를 동일 232장에 계산하면 최소 15 checkpoints × 232 images = **3,480 detector-image forward passes**와 DINO patch feature 232 image passes가 필요하다. selection-only라면 학습은 없지만, feature storage·spatial alignment 구현/검증 비용이 추가된다. 원본에 측정 GPU-hours가 없어 시간은 추정하지 않는다.

## 10. 학위 일정 적합성

event-count/프로토콜 설계는 감당 가능하지만, 현재 untouched resource가 없는 상태에서 local hook 구현까지 들어가면 결과가 다시 development overfitting이 된다. 학위 본문에는 exploratory FN signal과 실행 조건을 future work로 고정하는 편이 안전하다.

## 실행 재개를 위한 필요조건

1. locked final과 분리된 untouched native-bbox pool 확보.
2. primary endpoint와 inferential unit 사전 고정.
3. Random review, confidence deficit, no-detection, frozen V2.3 combined baseline을 동일 budget으로 비교.
4. rare-FN은 primary gate에서 제외.
5. 위 자원이 없으면 구현·feature extraction·학습을 시작하지 않음.

`ONE_GC10_SELECTION_ONLY_SCREEN_JUSTIFIED`가 아니므로 `v3_local_feature_misalignment_protocol.md`와 구현 코드는 생성하지 않았다.
"""
    FN_PATH.write_text(text, encoding="utf-8")


def write_audit_outputs(master: pd.DataFrame, branches: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scanned_lines = []
    for p in sorted(SCANNED):
        scanned_lines.append(f"{rel(p)}\tsha256={hashlib.sha256(p.read_bytes()).hexdigest()}")
    (OUT / "scanned_files.txt").write_text("\n".join(scanned_lines) + "\n", encoding="utf-8")
    pd.DataFrame(REGISTRY).to_csv(OUT / "source_metric_registry.csv", index=False, encoding="utf-8-sig")

    conflicts = [
        {"topic": "GC10 DINO downstream mAP", "value_a": 0.017378406931869143, "source_a": "gc10_detector_confirmation initial20/query20", "value_b": -0.0016777445999850433, "source_b": "dcal_xai K40 budget140", "resolution": "not pooled: different selector, budget and selected-set protocol"},
        {"topic": "NEU seed45 generalization", "value_a": 0.01623566489340474, "source_a": "one fixed acquisition set x5 training seeds", "value_b": 0.0070187, "source_b": "10 independent acquisition seeds", "resolution": "fixed-set diagnostic does not override independent confirmation FAIL"},
        {"topic": "MPDD discovery attribution", "value_a": 6.245, "source_a": "anomaly-image gain", "value_b": 6.81, "source_b": "official-test-origin image-count gain", "resolution": "not equivalent metrics; source-origin composition is a confound, not anomaly gain"},
        {"topic": "VisA discovery vs safety", "value_a": 14.48, "source_a": "anomaly-image gain", "value_b": -4.11, "source_b": "object-category coverage delta", "resolution": "different endpoints retained as an explicit trade-off"},
    ]
    pd.DataFrame(conflicts).to_csv(OUT / "metric_conflicts.csv", index=False, encoding="utf-8-sig")

    missing = [
        {"item": "VLM-gate prevented detector/model count", "status": "unverified_document_value", "reason": "no frozen run cardinality in original gate artifacts", "impact": "excluded from prevented-model total"},
        {"item": "actual or expected GPU-hours for integrated branches", "status": "missing_original_artifact", "reason": "no consistent runtime/GPU-hour registry across branches", "impact": "model-run counts only"},
        {"item": "untouched non-final GC10 bbox validation split for FN extension", "status": "missing_resource", "reason": "development already used; final locked", "impact": "FN extension limited to protocol design"},
        {"item": "MPDD/VisA downstream detector mAP", "status": "not_applicable_task_mismatch", "reason": "anomaly localization/discovery tasks; no new YOLO forced", "impact": "left missing, not zero-filled"},
        {"item": "causal estimate of target prevalence effect", "status": "missing_analysis", "reason": "dataset identity and prevalence/source condition are confounded", "impact": "survival verdict B; one stratified reanalysis required"},
    ]
    pd.DataFrame(missing).to_csv(OUT / "missing_evidence.csv", index=False, encoding="utf-8-sig")

    commands = [
        r".\.venv\Scripts\python.exe scripts\04_dcal_xai\build_condition_map_integration.py",
        r".\.venv\Scripts\python.exe scripts\04_dcal_xai\test_condition_map_integration.py",
    ]
    (OUT / "executed_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    generated = [MASTER_PATH, BRANCH_PATH, DECISION_PATH, FN_PATH] + sorted(FIGURES.glob("*.png")) + [OUT / "scanned_files.txt", OUT / "source_metric_registry.csv", OUT / "metric_conflicts.csv", OUT / "missing_evidence.csv", OUT / "executed_commands.txt"]
    manifest_rows = []
    for p in generated:
        manifest_rows.append(f"{rel(p)}\tbytes={p.stat().st_size}\tsha256={hashlib.sha256(p.read_bytes()).hexdigest()}")
    (OUT / "generated_file_manifest.txt").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")


def validate_internal(master: pd.DataFrame, branches: pd.DataFrame) -> list[str]:
    checks = []
    allowed = {"confirmatory_pass", "confirmatory_fail", "independent_confirmation_fail", "exploratory_positive", "diagnostic", "insufficient_evidence"}
    assert set(master["evidence_level"]).issubset(allowed)
    checks.append("PASS evidence_level vocabulary")
    assert master["source_file"].notna().all() and master["source_file"].str.len().gt(0).all()
    checks.append("PASS every condition row has source_file")
    assert not master["source_file"].str.lower().str.contains("final|locked").any()
    checks.append("PASS no final/locked source included")
    fail_rows = master["gate_result"].eq("FAIL")
    assert not master.loc[fail_rows, "evidence_level"].eq("confirmatory_pass").any()
    checks.append("PASS failed gates not upgraded")
    assert branches["final_test_consumed"].astype(str).str.lower().eq("false").all()
    checks.append("PASS final test consumption remains false")
    return checks


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    m = build_metrics()
    master = build_master(m)
    branches = build_branches(m)
    plot_discovery_coverage(m)
    plot_translation(m)
    plot_chain(m)
    plot_waterfall()
    build_decision_doc(m, branches)
    build_fn_doc(m)
    internal = validate_internal(master, branches)
    (OUT / "condition_map_validation.log").write_text("\n".join(internal) + "\n", encoding="utf-8")
    write_audit_outputs(master, branches)
    print(f"[DONE] condition rows={len(master)} branches={len(branches)} metrics={len(REGISTRY)}")
    print(f"[DECISION] B; FN extension=DESIGN_PROTOCOL_ONLY; final test used=False")


if __name__ == "__main__":
    main()

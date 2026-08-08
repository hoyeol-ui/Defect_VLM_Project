#!/usr/bin/env python3
"""Read-only mechanism audit of the frozen DeepPCB FAIL_STOP selection.

The original selection, score, thresholds, and decision are immutable.  GT is
used only for post-hoc mechanism diagnosis.  No detector or official test is
opened.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from reference_residual_audit_common import (
    CLASS_NAMES,
    DATA_ROOT,
    ELIGIBLE_GROUPS,
    FROZEN_MIN_AREA,
    FROZEN_THRESHOLD,
    ORIGINAL_RUN,
    ROOT,
    SMALL_BOX_AREA,
    frozen_components,
    load_gray,
    load_trainval,
    read_boxes,
    sha256,
    verify_original_freeze,
)


DEFAULT_OUTPUT = ROOT / "runs" / "deeppcb_small_defect_mechanism_audit"
DEFAULT_DOC = ROOT / "docs" / "deeppcb_frozen_small_defect_mechanism_audit.md"
RANDOM_SEED = 20260718
BOOTSTRAP_TRIALS = 50_000
SIZE_BINS = [(-1, 256, "<=256"), (256, 576, "257-576"), (576, 1024, "577-1024"), (1024, 4096, "1025-4096"), (4096, math.inf, ">4096")]


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def matched_outside_mask(box: dict, all_gt_mask: np.ndarray) -> np.ndarray:
    height, width = all_gt_mask.shape
    x1, y1 = max(0, box["x1"]), max(0, box["y1"])
    x2, y2 = min(width, box["x2"]), min(height, box["y2"])
    target = max(1, (x2 - x1) * (y2 - y1))
    candidate_coordinates = None
    for radius in (4, 8, 16, 32, 64, 128, 256, 640):
        xa, ya = max(0, x1 - radius), max(0, y1 - radius)
        xb, yb = min(width, x2 + radius), min(height, y2 + radius)
        available = ~all_gt_mask[ya:yb, xa:xb]
        yy, xx = np.nonzero(available)
        if len(xx) >= target or radius == 640:
            xx = xx + xa
            yy = yy + ya
            dx = np.maximum.reduce([x1 - xx, np.zeros_like(xx), xx - max(x2 - 1, x1)])
            dy = np.maximum.reduce([y1 - yy, np.zeros_like(yy), yy - max(y2 - 1, y1)])
            distance = np.maximum(dx, dy)
            order = np.lexsort((xx, yy, distance))
            candidate_coordinates = (yy[order[:target]], xx[order[:target]])
            break
    output = np.zeros_like(all_gt_mask, dtype=bool)
    if candidate_coordinates is not None:
        output[candidate_coordinates] = True
    return output


def box_mask(box: dict, shape: tuple[int, int], dilation: int = 0) -> np.ndarray:
    height, width = shape
    x1 = max(0, box["x1"] - dilation)
    y1 = max(0, box["y1"] - dilation)
    x2 = min(width, box["x2"] + dilation)
    y2 = min(height, box["y2"] + dilation)
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def bootstrap_small_share(group_effects: pd.DataFrame) -> tuple[float, float, float]:
    values = group_effects[["query_small", "query_total", "expected_small", "expected_total"]].to_numpy(float)
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_TRIALS, len(values)))
    samples = values[indices].sum(axis=1)
    effects = (samples[:, 0] / samples[:, 1]) / (samples[:, 2] / samples[:, 3])
    observed = (values[:, 0].sum() / values[:, 1].sum()) / (values[:, 2].sum() / values[:, 3].sum())
    return float(observed), float(np.quantile(effects, 0.025)), float(np.quantile(effects, 0.975))


def analyze_image(row: pd.Series, score_row: pd.Series, selected_paths: set[str]) -> tuple[dict, list[dict]]:
    tested = load_gray(row["tested_image"])
    template = load_gray(row["template_image"])
    difference, components = frozen_components(tested, template)
    boxes = read_boxes(row["annotation"])
    all_gt = np.zeros(tested.shape, dtype=bool)
    for box in boxes:
        all_gt |= box_mask(box, tested.shape)

    residual = components.mask.astype(bool)
    border = np.zeros_like(residual)
    border_width = 32
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    edges = cv2.Canny(template, 50, 150) > 0
    edge_band = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    shift, response = cv2.phaseCorrelate(template.astype(np.float32), tested.astype(np.float32))
    residual_count = int(residual.sum())
    image_record = {
        "canonical_path": row["canonical_path"],
        "group": row["group"],
        "selected": row["canonical_path"] in selected_paths,
        "reference_residual_richness": float(score_row["reference_residual_richness"]),
        "instance_count": len(boxes),
        "small_instance_count": sum(box["is_small"] for box in boxes),
        "global_intensity_shift": float(tested.astype(np.float32).mean() - template.astype(np.float32).mean()),
        "absolute_global_intensity_shift": float(abs(tested.astype(np.float32).mean() - template.astype(np.float32).mean())),
        "mean_absolute_difference": float(difference.mean()),
        "phase_dx": float(shift[0]),
        "phase_dy": float(shift[1]),
        "phase_translation_magnitude": float(math.hypot(*shift)),
        "phase_response": float(response),
        "residual_pixels": residual_count,
        "border_residual_fraction": safe_ratio(int((residual & border).sum()), residual_count),
        "edge_residual_fraction": safe_ratio(int((residual & edge_band).sum()), residual_count),
        "gt_residual_fraction": safe_ratio(int((residual & all_gt).sum()), residual_count),
    }

    centroids = components.centroids[1:] if len(components.centroids) > 1 else np.empty((0, 2))
    box_records: list[dict] = []
    for index, box in enumerate(boxes):
        inside = box_mask(box, tested.shape)
        outside = matched_outside_mask(box, all_gt)
        inside_pixels = int(inside.sum())
        outside_pixels = int(outside.sum())
        inside_residual = int((residual & inside).sum())
        outside_residual = int((residual & outside).sum())
        inside_density = safe_ratio(inside_residual, inside_pixels)
        outside_density = safe_ratio(outside_residual, outside_pixels)
        record = {
            "canonical_path": row["canonical_path"],
            "group": row["group"],
            "selected": image_record["selected"],
            "box_index": index,
            **box,
            "inside_pixels": inside_pixels,
            "outside_pixels": outside_pixels,
            "inside_residual_pixels": inside_residual,
            "outside_residual_pixels": outside_residual,
            "inside_density": inside_density,
            "outside_density": outside_density,
            "inside_outside_ratio": safe_ratio(inside_density, outside_density),
            "component_hit_d0": bool((residual & inside).any()),
            "component_hit_d3": bool((residual & box_mask(box, tested.shape, 3)).any()),
            "component_hit_d5": bool((residual & box_mask(box, tested.shape, 5)).any()),
            "component_centroid_inside": bool(
                len(centroids)
                and np.any(
                    (centroids[:, 0] >= box["x1"])
                    & (centroids[:, 0] < box["x2"])
                    & (centroids[:, 1] >= box["y1"])
                    & (centroids[:, 1] < box["y2"])
                )
            ),
        }
        box_records.append(record)
    return image_record, box_records


def aggregate_grounding(boxes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, size_label), frame in boxes.assign(size_label=np.where(boxes["is_small"], "small", "non_small")).groupby(["group", "size_label"]):
        inside_density = frame["inside_residual_pixels"].sum() / frame["inside_pixels"].sum()
        outside_density = frame["outside_residual_pixels"].sum() / frame["outside_pixels"].sum()
        rows.append(
            {
                "group": group,
                "size_label": size_label,
                "boxes": len(frame),
                "inside_density": inside_density,
                "matched_outside_density": outside_density,
                "inside_outside_ratio": safe_ratio(inside_density, outside_density),
                "component_hit_rate_d0": frame["component_hit_d0"].mean(),
                "component_hit_rate_d3": frame["component_hit_d3"].mean(),
                "component_hit_rate_d5": frame["component_hit_d5"].mean(),
                "component_centroid_inside_rate": frame["component_centroid_inside"].mean(),
            }
        )
    return pd.DataFrame(rows)


def effect_tables(images: pd.DataFrame, boxes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    group_rows = []
    bin_rows = []
    class_rows = []
    for group in ELIGIBLE_GROUPS:
        image_group = images[images["group"] == group]
        box_group = boxes[boxes["group"] == group]
        query_n = int(image_group["selected"].sum())
        inclusion = query_n / len(image_group)
        query_boxes = box_group[box_group["selected"]]
        query_total = len(query_boxes)
        query_small = int(query_boxes["is_small"].sum())
        expected_total = inclusion * len(box_group)
        expected_small = inclusion * int(box_group["is_small"].sum())
        total_enrichment = safe_ratio(query_total, expected_total)
        small_enrichment = safe_ratio(query_small, expected_small)
        group_rows.append(
            {
                "group": group,
                "pool_images": len(image_group),
                "query_images": query_n,
                "query_total": query_total,
                "expected_total": expected_total,
                "total_enrichment": total_enrichment,
                "query_small": query_small,
                "expected_small": expected_small,
                "small_enrichment": small_enrichment,
                "size_selectivity": safe_ratio(small_enrichment, total_enrichment),
                "small_excess": query_small - expected_small,
            }
        )
        for low, high, label in SIZE_BINS:
            pool_count = int(((box_group["bbox_area"] > low) & (box_group["bbox_area"] <= high)).sum())
            query_count = int(((query_boxes["bbox_area"] > low) & (query_boxes["bbox_area"] <= high)).sum())
            expected = inclusion * pool_count
            bin_rows.append({"group": group, "size_bin": label, "pool_boxes": pool_count, "query_boxes": query_count, "expected_random_boxes": expected, "enrichment": safe_ratio(query_count, expected)})
        for class_id, class_name in CLASS_NAMES.items():
            pool_count = int((box_group["class_id"] == class_id).sum())
            query_count = int((query_boxes["class_id"] == class_id).sum())
            expected = inclusion * pool_count
            class_rows.append({"group": group, "class_id": class_id, "class_name": class_name, "pool_boxes": pool_count, "query_boxes": query_count, "expected_random_boxes": expected, "excess_boxes": query_count - expected})
    group_effects = pd.DataFrame(group_rows)
    size_bins = pd.DataFrame(bin_rows)
    classes = pd.DataFrame(class_rows)
    uplift, low, high = bootstrap_small_share(group_effects)
    summary = {"small_share_uplift": uplift, "small_share_ci95_low": low, "small_share_ci95_high": high}
    return group_effects, size_bins, classes, summary


def dominance_and_loo(group_effects: pd.DataFrame, classes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    positive_group = group_effects["small_excess"].clip(lower=0)
    group_dominance = safe_ratio(float(positive_group.max()), float(positive_group.sum()))
    class_total = classes.groupby(["class_id", "class_name"], as_index=False)[["query_boxes", "expected_random_boxes", "excess_boxes"]].sum()
    positive_class = class_total["excess_boxes"].clip(lower=0)
    class_dominance = safe_ratio(float(positive_class.max()), float(positive_class.sum()))
    class_total["positive_excess_share"] = positive_class / max(float(positive_class.sum()), 1e-12)
    loo_rows = []
    for held_out in ELIGIBLE_GROUPS:
        kept = group_effects[group_effects["group"] != held_out]
        small_enrichment = kept["query_small"].sum() / kept["expected_small"].sum()
        total_enrichment = kept["query_total"].sum() / kept["expected_total"].sum()
        small_share = (kept["query_small"].sum() / kept["query_total"].sum()) / (kept["expected_small"].sum() / kept["expected_total"].sum())
        loo_rows.append({"held_out_group": held_out, "small_enrichment": small_enrichment, "total_enrichment": total_enrichment, "small_share_uplift": small_share})
    return class_total, pd.DataFrame(loo_rows), {"max_group_positive_small_excess_share": group_dominance, "max_class_positive_small_excess_share": class_dominance}


def confound_summary(images: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    variables = ["absolute_global_intensity_shift", "mean_absolute_difference", "phase_translation_magnitude", "border_residual_fraction", "edge_residual_fraction"]
    rows = []
    for variable in variables:
        rho = float(spearmanr(images["reference_residual_richness"], images[variable], nan_policy="omit").statistic)
        rows.append({"variable": variable, "spearman_with_frozen_score": rho, "absolute_spearman": abs(rho)})
    small_rho = float(spearmanr(images["reference_residual_richness"], images["small_instance_count"]).statistic)
    confounds = pd.DataFrame(rows)
    nuisance = confounds[confounds["variable"].isin(["absolute_global_intensity_shift", "phase_translation_magnitude", "border_residual_fraction"])]
    dominated = bool((nuisance["absolute_spearman"] >= max(0.50, abs(small_rho))).any())
    return confounds, {"score_vs_small_count_spearman": small_rho, "nuisance_confound_dominated": dominated}


def make_figures(output: Path, size_bins: pd.DataFrame, grounding: pd.DataFrame, loo: pd.DataFrame, manifest: pd.DataFrame, selected: pd.DataFrame) -> None:
    figure_dir = output / "figures"
    overlay_dir = figure_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    pooled = size_bins.groupby("size_bin", sort=False)[["query_boxes", "expected_random_boxes"]].sum().reset_index()
    pooled["enrichment"] = pooled["query_boxes"] / pooled["expected_random_boxes"]
    plt.figure(figsize=(7, 4))
    plt.plot(pooled["size_bin"], pooled["enrichment"], marker="o")
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Selection enrichment vs expected random")
    plt.xlabel("BBox area bin (px²)")
    plt.tight_layout()
    plt.savefig(figure_dir / "bbox_size_enrichment_curve.png", dpi=180)
    plt.close()

    small = grounding[grounding["size_label"] == "small"]
    plt.figure(figsize=(8, 4))
    plt.bar(small["group"], small["inside_outside_ratio"])
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Residual density: bbox / matched outside")
    plt.tight_layout()
    plt.savefig(figure_dir / "inside_outside_grounding.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(loo["held_out_group"], loo["small_share_uplift"])
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Small-share uplift after leaving group out")
    plt.tight_layout()
    plt.savefig(figure_dir / "group_influence.png", dpi=180)
    plt.close()

    manifest_index = manifest.set_index("canonical_path")
    for group in ELIGIBLE_GROUPS:
        path = selected[selected["group"] == group].sort_values("reference_residual_richness", ascending=False).iloc[0]["canonical_path"]
        row = manifest_index.loc[path]
        tested = cv2.imread(row["tested_image"])
        gray_tested = cv2.cvtColor(tested, cv2.COLOR_BGR2GRAY)
        template = load_gray(row["template_image"])
        _, components = frozen_components(gray_tested, template)
        overlay = tested.copy()
        overlay[components.mask.astype(bool)] = (0, 0, 255)
        for box in read_boxes(row["annotation"]):
            color = (0, 255, 0) if box["is_small"] else (255, 180, 0)
            cv2.rectangle(overlay, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 1)
        cv2.imwrite(str(overlay_dir / f"{group}_{Path(path).stem}_overlay.png"), overlay)


def write_document(path: Path, decision: dict, group_effects: pd.DataFrame, size_bins: pd.DataFrame, grounding: pd.DataFrame, dominance: dict, confound: dict, original_hash: str) -> None:
    lines = [
        "# Frozen DeepPCB small-defect mechanism audit",
        "",
        "## Immutable parent decision",
        "",
        "- Original decision: **FAIL_STOP (unchanged)**",
        f"- Frozen selection SHA-256: `{original_hash}`",
        "- Detector training/inference: **0 / 0**",
        "- Official/final test use: **0 / 0**",
        "- Small definition: bbox area <= 1024 px^2",
        "",
        "## Mechanism decision",
        "",
        f"- Decision: **{decision['phase_a_decision']}**",
        f"- Small-share uplift: {decision['small_share_uplift']:.4f} (group bootstrap 95% CI {decision['small_share_ci95_low']:.4f}, {decision['small_share_ci95_high']:.4f})",
        f"- Groups with size-selectivity > 1: {decision['groups_size_selectivity_gt_1']}/6",
        f"- Groups with small-bbox inside/outside ratio > 1: {decision['groups_small_grounding_gt_1']}/6",
        f"- Leave-one-group-out directions retained: {decision['loo_small_share_gt_1']}/6",
        f"- Max group share of positive small-box excess: {dominance['max_group_positive_small_excess_share']:.4f}",
        f"- Max class share of positive small-box excess: {dominance['max_class_positive_small_excess_share']:.4f}",
        f"- Nuisance-confound dominated: {confound['nuisance_confound_dominated']}",
        "",
        "## Frozen criteria",
        "",
    ]
    for name, passed in decision["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(["", "## Group effects", "", "| Group | Total enrich. | Small enrich. | Size selectivity | Small excess |", "|---|---:|---:|---:|---:|"])
    for row in group_effects.itertuples(index=False):
        lines.append(f"| {row.group} | {row.total_enrichment:.4f} | {row.small_enrichment:.4f} | {row.size_selectivity:.4f} | {row.small_excess:.2f} |")
    pooled_bins = size_bins.groupby("size_bin", sort=False)[["pool_boxes", "query_boxes", "expected_random_boxes"]].sum().reset_index()
    pooled_bins["enrichment"] = pooled_bins["query_boxes"] / pooled_bins["expected_random_boxes"].replace(0, np.nan)
    lines.extend(["", "## Size-bin localization of the effect", "", "| Bbox area bin | Pool boxes | Selected boxes | Expected random | Enrichment |", "|---|---:|---:|---:|---:|"])
    for row in pooled_bins.itertuples(index=False):
        enrichment = "NA" if pd.isna(row.enrichment) else f"{row.enrichment:.4f}"
        lines.append(f"| {row.size_bin} | {int(row.pool_boxes)} | {int(row.query_boxes)} | {row.expected_random_boxes:.2f} | {enrichment} |")
    lines.extend(["", "## Spatial grounding", "", "| Group | Size | Boxes | In/out ratio | Hit@0 | Hit@3 | Hit@5 |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in grounding.itertuples(index=False):
        lines.append(f"| {row.group} | {row.size_label} | {row.boxes} | {row.inside_outside_ratio:.4f} | {row.component_hit_rate_d0:.4f} | {row.component_hit_rate_d3:.4f} | {row.component_hit_rate_d5:.4f} |")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "This is a post-hoc read-only mechanism audit of a prospectively frozen FAIL_STOP branch. "
        "It cannot convert the parent selection gate into a PASS and is not evidence of detector learning utility, external replication, annotation-time reduction, or industrial generalization.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--run-dir", type=Path, default=ORIGINAL_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--document", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    freeze = verify_original_freeze(args.run_dir.resolve())
    manifest = load_trainval(args.data_root.resolve())
    eligible = manifest[manifest["group"].isin(ELIGIBLE_GROUPS)].copy()
    scores = pd.read_csv(args.run_dir / "gt_free_reference_residual_scores.csv", encoding="utf-8-sig")
    selected = pd.read_csv(args.run_dir / "frozen_selected_images.csv", encoding="utf-8-sig")
    if set(scores["canonical_path"]) != set(eligible["canonical_path"]):
        raise RuntimeError("Frozen score rows do not match eligible trainval rows")
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "eligible_images": len(eligible), "selected_images": len(selected), "original_decision": "FAIL_STOP", "training": False, "detector_inference": False, "official_test": False, "final_test": False}, indent=2))
        return

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_paths = set(selected["canonical_path"])
    score_index = scores.set_index("canonical_path")
    image_records = []
    box_records = []
    for index, (_, row) in enumerate(eligible.iterrows(), start=1):
        image_record, rows = analyze_image(row, score_index.loc[row["canonical_path"]], selected_paths)
        image_records.append(image_record)
        box_records.extend(rows)
        if index % 100 == 0 or index == len(eligible):
            print(f"[Phase A] {index}/{len(eligible)}")
    images = pd.DataFrame(image_records)
    boxes = pd.DataFrame(box_records)
    grounding = aggregate_grounding(boxes)
    group_effects, size_bins, classes, effect_summary = effect_tables(images, boxes)
    class_total, loo, dominance = dominance_and_loo(group_effects, classes)
    confounds, confound = confound_summary(images)
    small_grounding = grounding[grounding["size_label"] == "small"]
    checks = {
        "small_share_ci_low_gt_1": effect_summary["small_share_ci95_low"] > 1.0,
        "size_selectivity_gt_1_in_ge_5_groups": int((group_effects["size_selectivity"] > 1.0).sum()) >= 5,
        "small_inside_outside_gt_1_in_ge_5_groups": int((small_grounding["inside_outside_ratio"] > 1.0).sum()) >= 5,
        "leave_one_group_out_small_share_gt_1_all": bool((loo["small_share_uplift"] > 1.0).all()),
        "single_group_positive_excess_share_lt_0_40": dominance["max_group_positive_small_excess_share"] < 0.40,
        "single_class_positive_excess_share_lt_0_50": dominance["max_class_positive_small_excess_share"] < 0.50,
        "not_nuisance_confound_dominated": not confound["nuisance_confound_dominated"],
    }
    if all(checks.values()):
        phase_decision = "A1_MECHANISM_SUPPORTED"
    elif int((small_grounding["inside_outside_ratio"] > 1.0).sum()) <= 2 or (confound["nuisance_confound_dominated"] and int((small_grounding["inside_outside_ratio"] > 1.0).sum()) <= 3):
        phase_decision = "A3_ARTIFACT_DOMINATED_STOP"
    else:
        phase_decision = "A2_MECHANISM_AMBIGUOUS"
    decision = {
        "original_decision": "FAIL_STOP",
        "original_decision_unchanged": True,
        "phase_a_decision": phase_decision,
        **effect_summary,
        "groups_size_selectivity_gt_1": int((group_effects["size_selectivity"] > 1.0).sum()),
        "groups_small_grounding_gt_1": int((small_grounding["inside_outside_ratio"] > 1.0).sum()),
        "loo_small_share_gt_1": int((loo["small_share_uplift"] > 1.0).sum()),
        "checks": checks,
        "dominance": dominance,
        "confound": confound,
        "training_performed": False,
        "detector_inference_performed": False,
        "official_test_used": False,
        "final_test_used": False,
        "selection_sha256": freeze["selection_sha256"],
    }
    images.to_csv(output / "registration_confounds.csv", index=False, encoding="utf-8-sig")
    boxes.to_csv(output / "spatial_grounding_box_level.csv", index=False, encoding="utf-8-sig")
    grounding.to_csv(output / "spatial_grounding.csv", index=False, encoding="utf-8-sig")
    group_effects.to_csv(output / "macro_micro_effects.csv", index=False, encoding="utf-8-sig")
    size_bins.to_csv(output / "size_bin_enrichment.csv", index=False, encoding="utf-8-sig")
    class_total.to_csv(output / "class_group_dominance.csv", index=False, encoding="utf-8-sig")
    loo.to_csv(output / "leave_one_group_out.csv", index=False, encoding="utf-8-sig")
    confounds.to_csv(output / "confound_correlations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"source": str(args.run_dir / "frozen_selected_images.csv"), "sha256": sha256(args.run_dir / "frozen_selected_images.csv"), "role": "frozen selection"},
        {"source": str(args.run_dir / "gate_decision.json"), "sha256": sha256(args.run_dir / "gate_decision.json"), "role": "immutable parent decision"},
        {"source": str(args.data_root / "trainval.txt"), "sha256": sha256(args.data_root / "trainval.txt"), "role": "trainval-only manifest"},
    ]).to_csv(output / "source_registry.csv", index=False, encoding="utf-8-sig")
    (output / "phase_a_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    make_figures(output, size_bins, grounding, loo, eligible, selected)
    args.document.resolve().parent.mkdir(parents=True, exist_ok=True)
    write_document(args.document.resolve(), decision, group_effects, size_bins, grounding, dominance, confound, freeze["selection_sha256"])
    tests = {
        "original_fail_stop_unchanged": decision["original_decision"] == "FAIL_STOP",
        "eligible_images_889": len(images) == 889,
        "all_six_groups_present": set(images["group"]) == set(ELIGIBLE_GROUPS),
        "selected_images_179": int(images["selected"].sum()) == 179,
        "threshold_32": freeze["config"]["threshold"] == FROZEN_THRESHOLD,
        "min_area_9": freeze["config"]["min_component_area"] == FROZEN_MIN_AREA,
        "no_official_test": not decision["official_test_used"],
        "no_final_test": not decision["final_test_used"],
        "no_training": not decision["training_performed"],
        "no_detector_inference": not decision["detector_inference_performed"],
    }
    (output / "test_results.txt").write_text("\n".join(f"{'PASS' if value else 'FAIL'} {name}" for name, value in tests.items()) + "\n", encoding="utf-8")
    if not all(tests.values()):
        raise RuntimeError(f"Phase A integrity tests failed: {tests}")
    print(json.dumps({"status": "DONE", "original_decision": "FAIL_STOP", "phase_a_decision": phase_decision, "document": str(args.document.resolve()), "detector_screen_allowed": False, "training": False, "official_test": False, "final_test": False}, indent=2))


if __name__ == "__main__":
    main()

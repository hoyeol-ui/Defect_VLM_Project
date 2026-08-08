#!/usr/bin/env python3
"""Prospective selection-only gate for DeepPCB paired-reference triage.

The official test split is locked.  Scores and selected paths are written and
hashed before trainval annotations are joined for the post-hoc audit.
No detector training or inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "DeepPCB" / "PCBData"
DEFAULT_PROTOCOL = ROOT / "docs" / "deeppcb_reference_residual_prospective_protocol_20260718.md"
DEFAULT_OUTPUT = ROOT / "runs" / "deeppcb_reference_residual_gate" / "prospective_main_20260718"
CLASS_NAMES = {1: "open", 2: "short", 3: "mousebite", 4: "spur", 5: "copper", 6: "pin_hole"}
ELIGIBLE_GROUPS = ("group00041", "group13000", "group20085", "group44000", "group50600", "group77000")
RESERVED_DEVELOPMENT_GROUP = "group92000"
THRESHOLD = 32
MIN_COMPONENT_AREA = 9
QUERY_FRACTION = 0.20
SMALL_BOX_AREA = 1024
RANDOM_TRIALS = 10_000
RANDOM_SEED = 20260718


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_commit(dataset_root: Path) -> str:
    repo = dataset_root.parent
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNAVAILABLE"


def parse_split(data_root: Path) -> tuple[pd.DataFrame, dict]:
    train_file = data_root / "trainval.txt"
    test_file = data_root / "test.txt"
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError("DeepPCB trainval.txt/test.txt not found")
    rows = []
    for line in train_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_rel, ann_rel = line.split()
        image_base = data_root / image_rel
        test_image = image_base.with_name(image_base.stem + "_test.jpg")
        template_image = image_base.with_name(image_base.stem + "_temp.jpg")
        annotation = data_root / ann_rel
        group = Path(image_rel).parts[0]
        rows.append({
            "image_id": image_base.stem,
            "group": group,
            "tested_image": str(test_image.resolve()),
            "template_image": str(template_image.resolve()),
            "annotation": str(annotation.resolve()),
            "canonical_path": image_rel.replace("\\", "/"),
        })
    frame = pd.DataFrame(rows)
    missing = []
    for col in ("tested_image", "template_image", "annotation"):
        missing.extend(frame.loc[~frame[col].map(lambda x: Path(x).exists()), col].tolist())
    if missing:
        raise FileNotFoundError(f"Missing DeepPCB files, first={missing[:3]}")
    if len(frame) != 1000 or len(test_file.read_text(encoding="utf-8").splitlines()) != 500:
        raise RuntimeError("Official DeepPCB split count mismatch")
    if set(frame["group"]) != set(ELIGIBLE_GROUPS) | {RESERVED_DEVELOPMENT_GROUP}:
        raise RuntimeError(f"Unexpected trainval groups: {sorted(frame['group'].unique())}")
    lock = {
        "trainval_rows": int(len(frame)),
        "official_test_rows": 500,
        "official_test_list_sha256": sha256(test_file),
        "official_test_content_read": False,
        "official_test_annotation_read": False,
        "reserved_development_group": RESERVED_DEVELOPMENT_GROUP,
        "eligible_groups": list(ELIGIBLE_GROUPS),
    }
    return frame, lock


def exact_duplicate_count(paths: list[str]) -> int:
    seen = set()
    duplicates = 0
    for value in paths:
        digest = sha256(Path(value))
        if digest in seen:
            duplicates += 1
        seen.add(digest)
    return duplicates


def residual_features(row: pd.Series) -> dict:
    tested = cv2.imread(row["tested_image"], cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(row["template_image"], cv2.IMREAD_GRAYSCALE)
    if tested is None or template is None:
        raise RuntimeError(f"Cannot read image pair: {row['canonical_path']}")
    if tested.shape != template.shape or tested.shape != (640, 640):
        raise RuntimeError(f"Unexpected aligned image shape: {row['canonical_path']} {tested.shape} {template.shape}")
    diff = cv2.absdiff(tested, template)
    mask = (diff >= THRESHOLD).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        kept_areas = np.array([], dtype=np.int64)
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        kept_areas = areas[areas >= MIN_COMPONENT_AREA]
    return {
        "image_id": row["image_id"],
        "group": row["group"],
        "canonical_path": row["canonical_path"],
        "tested_image": row["tested_image"],
        "template_image": row["template_image"],
        "residual_area": int(kept_areas.sum()) if len(kept_areas) else 0,
        "residual_components": int(len(kept_areas)),
        "residual_fraction": float(kept_areas.sum() / mask.size) if len(kept_areas) else 0.0,
    }


def add_frozen_score(features: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in features.groupby("group", sort=True):
        group = group.copy()
        n = len(group)
        group["area_rank"] = (rankdata(np.log1p(group["residual_area"]), method="average") - 1) / max(n - 1, 1)
        group["component_rank"] = (rankdata(np.log1p(group["residual_components"]), method="average") - 1) / max(n - 1, 1)
        group["reference_residual_richness"] = 0.5 * group["area_rank"] + 0.5 * group["component_rank"]
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def freeze_selection(scores: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for group_name in ELIGIBLE_GROUPS:
        group = scores[scores["group"] == group_name].copy()
        query_n = int(math.ceil(QUERY_FRACTION * len(group)))
        group = group.sort_values(
            ["reference_residual_richness", "canonical_path"], ascending=[False, True]
        ).head(query_n)
        group["query_n"] = query_n
        group["selection_rank"] = np.arange(1, query_n + 1)
        selected.append(group)
    return pd.concat(selected, ignore_index=True)


def read_annotations(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_rows = []
    box_rows = []
    ann_by_path = manifest.set_index("canonical_path")["annotation"].to_dict()
    for canonical_path, ann_path in ann_by_path.items():
        group = Path(canonical_path).parts[0]
        class_counts = Counter()
        instance_count = 0
        small_count = 0
        bbox_area = 0.0
        for line in Path(ann_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            x1, y1, x2, y2, cls = map(int, line.split())
            area = max(0, x2 - x1) * max(0, y2 - y1)
            instance_count += 1
            small_count += int(area <= SMALL_BOX_AREA)
            bbox_area += area
            class_counts[cls] += 1
            box_rows.append({
                "canonical_path": canonical_path,
                "group": group,
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                "bbox_area": area,
                "is_small": area <= SMALL_BOX_AREA,
            })
        row = {
            "canonical_path": canonical_path,
            "group": group,
            "instance_count": instance_count,
            "small_instance_count": small_count,
            "bbox_area_sum": bbox_area,
        }
        for cls in CLASS_NAMES:
            row[f"class_{cls}_count"] = class_counts[cls]
        image_rows.append(row)
    return pd.DataFrame(image_rows), pd.DataFrame(box_rows)


def distribution_metrics(class_counts: np.ndarray) -> tuple[int, float, float, float]:
    total = float(class_counts.sum())
    if total <= 0:
        return 0, 0.0, 0.0, 0.0
    probs = class_counts[class_counts > 0] / total
    coverage = int((class_counts > 0).sum())
    entropy = float(-(probs * np.log(probs)).sum())
    hhi = float((probs**2).sum())
    max_share = float(probs.max())
    return coverage, entropy, hhi, max_share


def random_baseline(group: pd.DataFrame, query_n: int, rng: np.random.Generator) -> dict:
    arrays = {
        "instances": group["instance_count"].to_numpy(float),
        "small": group["small_instance_count"].to_numpy(float),
        "bbox_area": group["bbox_area_sum"].to_numpy(float),
        "classes": group[[f"class_{c}_count" for c in CLASS_NAMES]].to_numpy(float),
    }
    results = defaultdict(list)
    for _ in range(RANDOM_TRIALS):
        idx = rng.choice(len(group), size=query_n, replace=False)
        class_counts = arrays["classes"][idx].sum(axis=0)
        coverage, entropy, hhi, max_share = distribution_metrics(class_counts)
        results["instances"].append(arrays["instances"][idx].sum())
        results["small"].append(arrays["small"][idx].sum())
        results["bbox_area"].append(arrays["bbox_area"][idx].sum())
        results["coverage"].append(coverage)
        results["entropy"].append(entropy)
        results["hhi"].append(hhi)
        results["max_share"].append(max_share)
    summary = {}
    for key, values in results.items():
        arr = np.asarray(values, dtype=float)
        summary[f"random_{key}_mean"] = float(arr.mean())
        summary[f"random_{key}_ci95_low"] = float(np.quantile(arr, 0.025))
        summary[f"random_{key}_ci95_high"] = float(np.quantile(arr, 0.975))
    return summary


def bootstrap_mean_ci(values: list[float], seed: int, trials: int = 50_000) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(trials, len(arr)))].mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def posthoc_audit(scores: pd.DataFrame, selected: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    joined = scores.merge(outcomes, on=["canonical_path", "group"], how="inner", validate="one_to_one")
    selected_paths = set(selected["canonical_path"])
    rng = np.random.default_rng(RANDOM_SEED)
    group_rows = []
    class_rows = []
    for group_name in ELIGIBLE_GROUPS:
        group = joined[joined["group"] == group_name].copy()
        query = group[group["canonical_path"].isin(selected_paths)].copy()
        query_n = len(query)
        random = random_baseline(group, query_n, rng)
        class_counts = query[[f"class_{c}_count" for c in CLASS_NAMES]].sum().to_numpy(float)
        coverage, entropy, hhi, max_share = distribution_metrics(class_counts)
        spearman = float(spearmanr(group["reference_residual_richness"], group["instance_count"]).statistic)
        query_instances = float(query["instance_count"].sum())
        query_small = float(query["small_instance_count"].sum())
        query_area = float(query["bbox_area_sum"].sum())
        row = {
            "group": group_name,
            "pool_images": len(group),
            "query_images": query_n,
            "spearman_score_vs_instances": spearman,
            "query_instances": query_instances,
            "random_instances_mean": random["random_instances_mean"],
            "instance_enrichment_vs_random": query_instances / random["random_instances_mean"],
            "query_small_instances": query_small,
            "random_small_mean": random["random_small_mean"],
            "small_instance_enrichment_vs_random": query_small / random["random_small_mean"],
            "query_bbox_area": query_area,
            "random_bbox_area_mean": random["random_bbox_area_mean"],
            "bbox_area_enrichment_vs_random": query_area / random["random_bbox_area_mean"],
            "query_class_coverage": coverage,
            "random_class_coverage_mean": random["random_coverage_mean"],
            "class_coverage_delta_vs_random": coverage - random["random_coverage_mean"],
            "query_class_entropy": entropy,
            "random_class_entropy_mean": random["random_entropy_mean"],
            "class_entropy_delta_vs_random": entropy - random["random_entropy_mean"],
            "query_class_hhi": hhi,
            "random_class_hhi_mean": random["random_hhi_mean"],
            "class_hhi_delta_vs_random": hhi - random["random_hhi_mean"],
            "query_max_class_share": max_share,
            "random_max_class_share_mean": random["random_max_share_mean"],
            "max_class_share_delta_vs_random": max_share - random["random_max_share_mean"],
        }
        group_rows.append(row)
        for idx, cls in enumerate(CLASS_NAMES):
            class_rows.append({
                "group": group_name,
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                "query_instances": int(class_counts[idx]),
                "query_share": float(class_counts[idx] / class_counts.sum()),
            })
    group_metrics = pd.DataFrame(group_rows)
    class_metrics = pd.DataFrame(class_rows)
    aggregate = {}
    metric_columns = [
        "spearman_score_vs_instances",
        "instance_enrichment_vs_random",
        "small_instance_enrichment_vs_random",
        "bbox_area_enrichment_vs_random",
        "class_coverage_delta_vs_random",
        "max_class_share_delta_vs_random",
    ]
    for offset, metric in enumerate(metric_columns):
        mean, low, high = bootstrap_mean_ci(group_metrics[metric].tolist(), RANDOM_SEED + offset)
        aggregate[metric] = {"mean": mean, "ci95_low": low, "ci95_high": high}
    checks = {
        "g1_mean_spearman_ge_0_30": aggregate["spearman_score_vs_instances"]["mean"] >= 0.30,
        "g1_positive_spearman_groups_ge_5": int((group_metrics["spearman_score_vs_instances"] > 0).sum()) >= 5,
        "g2_instance_enrichment_mean_ge_1_20": aggregate["instance_enrichment_vs_random"]["mean"] >= 1.20,
        "g2_instance_enrichment_ci_low_gt_1": aggregate["instance_enrichment_vs_random"]["ci95_low"] > 1.00,
        "g2_enrichment_groups_gt_1_ge_5": int((group_metrics["instance_enrichment_vs_random"] > 1.0).sum()) >= 5,
        "g3_class_coverage_delta_ge_minus_0_25": aggregate["class_coverage_delta_vs_random"]["mean"] >= -0.25,
        "g3_small_instance_enrichment_ge_0_90": aggregate["small_instance_enrichment_vs_random"]["mean"] >= 0.90,
        "g3_max_class_share_delta_le_0_10": aggregate["max_class_share_delta_vs_random"]["mean"] <= 0.10,
    }
    decision = "PASS_SELECTION_ONLY" if all(checks.values()) else "FAIL_STOP"
    result = {
        "decision": decision,
        "checks": checks,
        "aggregate": aggregate,
        "eligible_groups": len(ELIGIBLE_GROUPS),
        "query_images": int(len(selected)),
        "training_performed": False,
        "detector_inference_performed": False,
        "official_test_used": False,
        "final_test_used": False,
        "authorization": "ONE_BOUNDED_DETECTOR_SCREEN" if decision == "PASS_SELECTION_ONLY" else "STOP",
    }
    return group_metrics, class_metrics, result


def write_summary(path: Path, result: dict, group_metrics: pd.DataFrame, lock: dict, selection_hash: str) -> None:
    agg = result["aggregate"]
    lines = [
        "# DeepPCB Prospective Reference-Residual Selection Gate",
        "",
        f"- Decision: **{result['decision']}**",
        f"- Authorization: **{result['authorization']}**",
        f"- Eligible source/design groups: **{result['eligible_groups']}**",
        f"- Selected images: **{result['query_images']}**",
        f"- Reserved development group: **{lock['reserved_development_group']}**",
        f"- Official test rows locked: **{lock['official_test_rows']}**",
        "- Training performed: **False**",
        "- Detector inference performed: **False**",
        "- Official/final test used: **False**",
        f"- Frozen selection SHA256: `{selection_hash}`",
        "",
        "## Aggregate frozen endpoints",
        "",
        "| endpoint | mean | CI95 low | CI95 high |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "spearman_score_vs_instances": "G1 Spearman: score vs bbox count",
        "instance_enrichment_vs_random": "G2 bbox-instance enrichment@20%",
        "small_instance_enrichment_vs_random": "G3 small-box enrichment",
        "bbox_area_enrichment_vs_random": "Secondary bbox-area enrichment",
        "class_coverage_delta_vs_random": "G3 class-coverage delta",
        "max_class_share_delta_vs_random": "G3 max-class-share delta",
    }
    for key, label in labels.items():
        value = agg[key]
        lines.append(f"| {label} | {value['mean']:.6f} | {value['ci95_low']:.6f} | {value['ci95_high']:.6f} |")
    lines.extend(["", "## Frozen checks", ""])
    for check, passed in result["checks"].items():
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] `{check}`")
    lines.extend([
        "",
        "## Group-level results",
        "",
        "| group | pool | query | Spearman | instance enrichment | small enrichment | bbox-area enrichment | coverage delta | max-share delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in group_metrics.itertuples(index=False):
        lines.append(
            f"| {row.group} | {row.pool_images} | {row.query_images} | {row.spearman_score_vs_instances:.4f} | "
            f"{row.instance_enrichment_vs_random:.4f} | {row.small_instance_enrichment_vs_random:.4f} | "
            f"{row.bbox_area_enrichment_vs_random:.4f} | {row.class_coverage_delta_vs_random:.4f} | "
            f"{row.max_class_share_delta_vs_random:.4f} |"
        )
    lines.extend([
        "",
        "## Claim boundary",
        "",
        "A PASS is evidence only for external aligned-reference annotation-triage potential under a fixed image budget. "
        "It is not evidence of detector improvement, annotation-cost reduction, universal AL superiority, production generalization, or official-test performance.",
        "",
        "A FAIL closes this exact reference-residual branch without threshold or score rescue.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    protocol = args.protocol.resolve()
    output = args.output_dir.resolve()
    if not protocol.exists():
        raise FileNotFoundError(protocol)
    manifest, lock = parse_split(data_root)
    eligible = manifest[manifest["group"].isin(ELIGIBLE_GROUPS)].copy()
    duplicate_count = exact_duplicate_count(manifest["tested_image"].tolist())
    if duplicate_count:
        raise RuntimeError(f"Exact duplicate trainval tested images: {duplicate_count}")
    validation = {
        "status": "DRY_RUN_PASS",
        "trainval_rows": len(manifest),
        "eligible_rows": len(eligible),
        "reserved_development_rows": int((manifest["group"] == RESERVED_DEVELOPMENT_GROUP).sum()),
        "eligible_groups": list(ELIGIBLE_GROUPS),
        "exact_duplicate_trainval_tested_images": duplicate_count,
        "protocol_sha256": sha256(protocol),
        "source_commit": source_commit(data_root),
        **lock,
        "training_performed": False,
        "detector_inference_performed": False,
        "official_test_used": False,
        "final_test_used": False,
    }
    if args.dry_run:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).drop(columns=["annotation"]).to_csv(output / "trainval_manifest_no_gt.csv", index=False, encoding="utf-8-sig")
    feature_rows = [residual_features(row) for _, row in eligible.iterrows()]
    scores = add_frozen_score(pd.DataFrame(feature_rows))
    scores.to_csv(output / "gt_free_reference_residual_scores.csv", index=False, encoding="utf-8-sig")
    selected = freeze_selection(scores)
    selected_path = output / "frozen_selected_images.csv"
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    selection_hash = sha256(selected_path)
    (output / "frozen_selection_sha256.txt").write_text(selection_hash + "\n", encoding="utf-8")

    # Only after the selection has been frozen and hashed do annotations enter.
    outcomes, boxes = read_annotations(manifest[manifest["group"].isin(ELIGIBLE_GROUPS)])
    group_metrics, class_metrics, result = posthoc_audit(scores, selected, outcomes)
    group_metrics.to_csv(output / "posthoc_group_metrics.csv", index=False, encoding="utf-8-sig")
    class_metrics.to_csv(output / "posthoc_class_distribution.csv", index=False, encoding="utf-8-sig")
    boxes.to_csv(output / "posthoc_box_audit.csv", index=False, encoding="utf-8-sig")
    result.update({
        "protocol_sha256": sha256(protocol),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_commit": source_commit(data_root),
        "selection_sha256": selection_hash,
        "split_lock": lock,
        "exact_duplicate_trainval_tested_images": duplicate_count,
    })
    (output / "gate_decision.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {
        "threshold": THRESHOLD,
        "min_component_area": MIN_COMPONENT_AREA,
        "query_fraction": QUERY_FRACTION,
        "small_box_area": SMALL_BOX_AREA,
        "random_trials": RANDOM_TRIALS,
        "random_seed": RANDOM_SEED,
        "eligible_groups": list(ELIGIBLE_GROUPS),
        "reserved_development_group": RESERVED_DEVELOPMENT_GROUP,
        "training_performed": False,
        "detector_inference_performed": False,
        "official_test_used": False,
        "final_test_used": False,
    }
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(output / "deeppcb_reference_residual_gate_summary.md", result, group_metrics, lock, selection_hash)
    print(json.dumps({
        "status": "DONE",
        "decision": result["decision"],
        "authorization": result["authorization"],
        "summary": str(output / "deeppcb_reference_residual_gate_summary.md"),
        "training_performed": False,
        "detector_inference_performed": False,
        "official_test_used": False,
        "final_test_used": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


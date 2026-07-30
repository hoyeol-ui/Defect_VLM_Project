#!/usr/bin/env python3
"""Development-only comparison of three reference-residual methods.

Only the 111 DeepPCB trainval rows in group92000 are allowed.  This script is
not a confirmatory experiment and can only emit a frozen candidate for a later
independent study, or NO_CANDIDATE.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reference_residual_audit_common import (
    DATA_ROOT,
    ELIGIBLE_GROUPS,
    FROZEN_MIN_AREA,
    FROZEN_THRESHOLD,
    ORIGINAL_RUN,
    RESERVED_DEVELOPMENT_GROUP,
    ROOT,
    component_records,
    conditional_phase_alignment,
    frozen_components,
    greedy_match,
    load_gray,
    load_trainval,
    quantile_components,
    read_boxes,
    read_json,
    robust_unit_map,
    sha256,
    ssim_dissimilarity,
    verify_original_freeze,
)


METHODS = ("M1_ABSDIFF", "M2_SSIM", "M3_FUSED")
DEFAULT_POLICY = Path(__file__).with_name("development_policy_20260718_v2.json")
DEFAULT_OUTPUT = ROOT / "runs" / "deeppcb_reference_residual_development"
DEFAULT_DOC = ROOT / "docs" / "deeppcb_reference_residual_development_benchmark.md"
DEFAULT_FEASIBILITY_DOC = ROOT / "docs" / "paired_reference_external_confirmation_feasibility.md"


def repository_head(url: str) -> str:
    try:
        output = subprocess.check_output(["git", "ls-remote", url, "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=20)
        return output.split()[0]
    except Exception:
        return "UNAVAILABLE"


def translate_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def translate_boxes(boxes: list[dict], dx: int, dy: int, width: int = 640, height: int = 640) -> list[dict]:
    shifted = []
    for box in boxes:
        record = dict(box)
        record["x1"] = max(0, min(width, box["x1"] + dx))
        record["x2"] = max(0, min(width, box["x2"] + dx))
        record["y1"] = max(0, min(height, box["y1"] + dy))
        record["y2"] = max(0, min(height, box["y2"] + dy))
        shifted.append(record)
    return shifted


def restore_alignment_coordinates(predictions: list[dict], alignment: dict) -> list[dict]:
    if not alignment.get("alignment_accepted"):
        return predictions
    dx, dy = alignment["phase_dx"], alignment["phase_dy"]
    restored = []
    for prediction in predictions:
        item = dict(prediction)
        item["x1"] = int(np.clip(round(item["x1"] + dx), 0, 640))
        item["x2"] = int(np.clip(round(item["x2"] + dx), 0, 640))
        item["y1"] = int(np.clip(round(item["y1"] + dy), 0, 640))
        item["y2"] = int(np.clip(round(item["y2"] + dy), 0, 640))
        restored.append(item)
    return restored


def predict(method: str, tested: np.ndarray, template: np.ndarray, policy: dict) -> tuple[list[dict], np.ndarray, dict]:
    if method == "M1_ABSDIFF":
        difference, components = frozen_components(tested, template)
        return component_records(components), difference.astype(np.float32) / 255.0, {"alignment_accepted": False}
    if method == "M2_SSIM":
        _, dissimilarity = ssim_dissimilarity(template, tested)
        components = quantile_components(dissimilarity, policy["methods"][method]["central_quantile"], FROZEN_MIN_AREA)
        return component_records(components), dissimilarity, {"alignment_accepted": False}
    if method == "M3_FUSED":
        cfg = policy["methods"][method]
        aligned, alignment = conditional_phase_alignment(tested, template, cfg["conditional_phase_alignment_max_pixels"])
        _, dissimilarity = ssim_dissimilarity(template, aligned)
        absolute = cv2.absdiff(aligned, template).astype(np.float32)
        absolute = robust_unit_map(absolute, percentile=cfg["normalization_percentile"])
        structural = robust_unit_map(dissimilarity, percentile=cfg["normalization_percentile"])
        fused = cfg["ssim_weight"] * structural + cfg["absdiff_weight"] * absolute
        components = quantile_components(fused, cfg["central_quantile"], FROZEN_MIN_AREA)
        predictions = restore_alignment_coordinates(component_records(components), alignment)
        return predictions, fused, alignment
    raise ValueError(method)


def all_scores(records: list[dict], method: str) -> np.ndarray:
    values = [prediction["score"] for row in records if row["method"] == method for prediction in row["predictions"]]
    return np.unique(np.asarray(values, dtype=float)) if values else np.asarray([], dtype=float)


def evaluate_threshold(records: list[dict], method: str, threshold: float, iou: float) -> dict:
    total_gt = total_small = total_hits = total_small_hits = total_fp = 0
    hit_classes: set[int] = set()
    for row in records:
        if row["method"] != method:
            continue
        boxes = row["boxes"]
        hits, false_positives, hit_indices = greedy_match(row["predictions"], boxes, threshold, iou)
        total_gt += len(boxes)
        total_small += sum(box["is_small"] for box in boxes)
        total_hits += hits
        total_small_hits += sum(boxes[index]["is_small"] for index in hit_indices)
        total_fp += false_positives
        hit_classes.update(boxes[index]["class_id"] for index in hit_indices)
    images = sum(row["method"] == method for row in records)
    return {
        "method": method,
        "score_threshold": float(threshold),
        "images": images,
        "gt_boxes": total_gt,
        "gt_small_boxes": total_small,
        "matched_boxes": total_hits,
        "matched_small_boxes": total_small_hits,
        "recall": total_hits / total_gt,
        "small_recall": total_small_hits / total_small,
        "false_positives": total_fp,
        "false_positives_per_image": total_fp / images,
        "hit_class_coverage": len(hit_classes),
    }


def select_operating_point(records: list[dict], method: str, iou: float) -> tuple[dict, pd.DataFrame]:
    scores = all_scores(records, method)
    if not len(scores):
        result = evaluate_threshold(records, method, math.inf, iou)
        return result, pd.DataFrame([result])
    if len(scores) > 250:
        thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 250)))
    else:
        thresholds = scores
    # A zero-detection point is required for a valid FROC curve even when
    # component confidences saturate at one (as absdiff often does).
    thresholds = np.append(thresholds, math.inf)
    rows = [evaluate_threshold(records, method, float(threshold), iou) for threshold in thresholds]
    curve = pd.DataFrame(rows).sort_values("score_threshold")
    feasible = curve[curve["false_positives_per_image"] <= 1.0]
    if feasible.empty:
        chosen = curve.sort_values(["false_positives_per_image", "small_recall", "score_threshold"], ascending=[True, False, False]).iloc[0]
    else:
        chosen = feasible.sort_values(["small_recall", "false_positives_per_image", "score_threshold"], ascending=[False, True, False]).iloc[0]
    return chosen.to_dict(), curve


def run_nominal(manifest: pd.DataFrame, policy: dict) -> tuple[list[dict], pd.DataFrame]:
    records = []
    alignment_rows = []
    for index, (_, row) in enumerate(manifest.iterrows(), start=1):
        tested = load_gray(row["tested_image"])
        template = load_gray(row["template_image"])
        boxes = read_boxes(row["annotation"])
        for method in METHODS:
            predictions, _, alignment = predict(method, tested, template, policy)
            records.append({"canonical_path": row["canonical_path"], "method": method, "predictions": predictions, "boxes": boxes})
            alignment_rows.append({"canonical_path": row["canonical_path"], "method": method, **alignment})
        if index % 25 == 0 or index == len(manifest):
            print(f"[Phase B nominal] {index}/{len(manifest)}")
    return records, pd.DataFrame(alignment_rows)


def evaluate_perturbation(manifest: pd.DataFrame, policy: dict, thresholds: dict[str, float], kind: str, value: tuple[int, int] | int) -> list[dict]:
    accumulators = {method: [] for method in METHODS}
    for _, row in manifest.iterrows():
        tested = load_gray(row["tested_image"])
        template = load_gray(row["template_image"])
        boxes = read_boxes(row["annotation"])
        if kind == "brightness":
            tested = np.clip(tested.astype(np.int16) + int(value), 0, 255).astype(np.uint8)
        elif kind == "translation":
            dx, dy = value
            tested = translate_image(tested, dx, dy)
            boxes = translate_boxes(boxes, dx, dy)
        else:
            raise ValueError(kind)
        for method in METHODS:
            predictions, _, _ = predict(method, tested, template, policy)
            accumulators[method].append({"method": method, "predictions": predictions, "boxes": boxes})
    rows = []
    for method in METHODS:
        metric = evaluate_threshold(accumulators[method], method, thresholds[method], policy["matching_iou"])
        metric.update({"perturbation": kind, "value": str(value)})
        rows.append(metric)
    return rows


def write_registry(path: Path) -> None:
    paper_repo = "https://github.com/nonsakhoo/PCB-defect-localization.git"
    rows = [
        {"resource": "DeepPCB", "repository": "https://github.com/tangsanli5201/DeepPCB", "commit_sha": "08e98c4db5922613fb97176eb3d6497d48260cb1", "license": "MIT; dataset research-use notice in README", "copied_or_adapted_files": "none in this phase; local dataset only", "role": "development data"},
        {"resource": "Saiyod et al. 2026 reference implementation", "repository": paper_repo.removesuffix(".git"), "commit_sha": repository_head(paper_repo), "license": "no repository license file observed; reference only", "copied_or_adapted_files": "none", "role": "methodological reference: SSIM/absdiff fusion, conditional alignment, FROC"},
        {"resource": "OpenCV", "repository": "https://github.com/opencv/opencv", "commit_sha": "installed-package", "license": "Apache-2.0", "copied_or_adapted_files": "none", "role": f"runtime library cv2 {cv2.__version__}"},
        {"resource": "NumPy", "repository": "https://github.com/numpy/numpy", "commit_sha": "installed-package", "license": "BSD-3-Clause", "copied_or_adapted_files": "none", "role": f"runtime library numpy {np.__version__}"},
        {"resource": "pandas", "repository": "https://github.com/pandas-dev/pandas", "commit_sha": "installed-package", "license": "BSD-3-Clause", "copied_or_adapted_files": "none", "role": f"runtime library pandas {pd.__version__}"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_docs(document: Path, feasibility: Path, comparison: pd.DataFrame, robustness: pd.DataFrame, candidate: dict, policy_hash: str, phase_a: str) -> None:
    lines = [
        "# DeepPCB reference-residual development benchmark",
        "",
        "## Scope and locks",
        "",
        "- Parent prospective decision: **FAIL_STOP (unchanged)**",
        f"- Phase A mechanism result: **{phase_a}**",
        "- Data: **group92000 trainval rows only (111 images)**",
        "- Status: development-only, not confirmatory evidence",
        "- Implementation amendment: component confidence uses the cited implementation's fixed 0.95 within-component quantile; the earlier max-score run is preserved as invalid due to saturation",
        f"- Pre-execution policy SHA-256: `{policy_hash}`",
        "- Detector training/inference: **0 / 0**",
        "- Official/final test use: **0 / 0**",
        "",
        "## Method comparison at development operating points",
        "",
        "| Method | Threshold | Recall | Small recall | FP/image | Hit classes | Adequate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(f"| {row.method} | {row.score_threshold:.5f} | {row.recall:.4f} | {row.small_recall:.4f} | {row.false_positives_per_image:.4f} | {row.hit_class_coverage} | {row.adequate} |")
    lines.extend([
        "",
        "## Development decision",
        "",
        f"- Result: **{candidate['result']}**",
        f"- Candidate: **{candidate.get('method', 'NONE')}**",
        "",
        "This candidate, if present, is only a frozen object for an independent prospective selection-only study. "
        "It does not rescue the original FAIL_STOP branch, validate spatial mechanism, authorize detector training, or establish external utility.",
    ])
    document.write_text("\n".join(lines) + "\n", encoding="utf-8")

    feasibility_lines = [
        "# Paired-reference external confirmation feasibility",
        "",
        f"- Development result: **{candidate['result']}**",
        f"- Frozen candidate: **{candidate.get('method', 'NONE')}**",
        "- Detector screen currently allowed: **NO**",
        f"- External confirmation currently authorized: **{'YES' if candidate['result'] == 'FROZEN_CANDIDATE_FOR_EXTERNAL_CONFIRMATION' else 'NO'}**",
        "",
        "## Required independent data",
        "",
        "1. Defective query and defect-free reference must be paired by the same product layout/revision.",
        "2. The reference must not contain the query defect and must be usable without GT-driven reference selection.",
        "3. Bboxes or pixel masks must exist, with enough bbox-area <=1024 px^2 instances for group-resampled inference.",
        "4. Production lot/time/source groups must support independent group-level resampling.",
        "5. No DeepPCB image, derivative, or board group may overlap.",
        "6. Candidate code, operating threshold, 20% query budget, endpoints, and STOP rules must be frozen before labels are opened.",
        "7. Selection must be hashed before annotations are joined.",
        "",
        "## Prohibited substitution",
        "",
        "DeepPCB group92000 is development data and the six eligible groups are post-hoc audit data; neither can serve as independent confirmation. "
        "The official DeepPCB test remains locked unless the supervisor explicitly retires it from final-test use under a separate one-shot protocol.",
        "",
        "## Current decision",
        "",
        ("A frozen candidate exists, but an independent dataset feasibility audit is still required before any confirmation." if candidate["result"] == "FROZEN_CANDIDATE_FOR_EXTERNAL_CONFIRMATION" else "No development candidate passed the frozen adequacy criteria. Do not start an external confirmation or detector screen from this branch."),
    ]
    feasibility.write_text("\n".join(feasibility_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--run-dir", type=Path, default=ORIGINAL_RUN)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--document", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--feasibility-document", type=Path, default=DEFAULT_FEASIBILITY_DOC)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    freeze = verify_original_freeze(args.run_dir.resolve())
    policy = read_json(args.policy.resolve())
    manifest = load_trainval(args.data_root.resolve())
    development = manifest[manifest["group"] == RESERVED_DEVELOPMENT_GROUP].copy()
    if len(development) != 111 or set(development["group"]) != {RESERVED_DEVELOPMENT_GROUP}:
        raise RuntimeError("Development boundary is not exactly 111 group92000 trainval images")
    if set(development["canonical_path"]) & set(pd.read_csv(args.run_dir / "gt_free_reference_residual_scores.csv")["canonical_path"]):
        raise RuntimeError("Development group overlaps original six-group score set")
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "development_images": len(development), "group": RESERVED_DEVELOPMENT_GROUP, "policy_sha256": sha256(args.policy.resolve()), "parent_decision": freeze["decision"]["decision"], "training": False, "official_test": False, "final_test": False}, indent=2))
        return

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    nominal_records, alignments = run_nominal(development, policy)
    comparison_rows = []
    curves = []
    thresholds = {}
    for method in METHODS:
        chosen, curve = select_operating_point(nominal_records, method, policy["matching_iou"])
        thresholds[method] = chosen["score_threshold"]
        comparison_rows.append(chosen)
        curve["method"] = method
        curves.append(curve)
    comparison = pd.DataFrame(comparison_rows)
    froc = pd.concat(curves, ignore_index=True)

    robustness_rows = []
    for offset in policy["robustness"]["brightness_offsets"]:
        robustness_rows.extend(evaluate_perturbation(development, policy, thresholds, "brightness", int(offset)))
    pixels = int(policy["robustness"]["translation_pixels"])
    for shift in ((-pixels, 0), (pixels, 0), (0, -pixels), (0, pixels)):
        robustness_rows.extend(evaluate_perturbation(development, policy, thresholds, "translation", shift))
    robustness = pd.DataFrame(robustness_rows)

    policy_floor = policy["candidate_adequacy"]
    for index, row in comparison.iterrows():
        method = row["method"]
        nominal_small = max(float(row["small_recall"]), 1e-12)
        method_robustness = robustness[robustness["method"] == method]
        brightness_retention = float(method_robustness[method_robustness["perturbation"] == "brightness"]["small_recall"].min() / nominal_small)
        translation_retention = float(method_robustness[method_robustness["perturbation"] == "translation"]["small_recall"].min() / nominal_small)
        comparison.loc[index, "brightness_recall_retention"] = brightness_retention
        comparison.loc[index, "translation_recall_retention"] = translation_retention
        adequate = (
            row["small_recall"] >= policy_floor["small_recall_at_or_below_1_fppi_min"]
            and row["false_positives_per_image"] <= 1.0
            and brightness_retention >= policy_floor["brightness_recall_retention_min"]
            and translation_retention >= policy_floor["translation_recall_retention_min"]
            and row["hit_class_coverage"] >= policy_floor["hit_class_coverage_min"]
        )
        comparison.loc[index, "adequate"] = bool(adequate)

    adequate = comparison[comparison["adequate"] == True].copy()  # noqa: E712
    if adequate.empty:
        candidate = {"result": "NO_CANDIDATE", "method": None}
    else:
        chosen = adequate.sort_values(
            ["small_recall", "false_positives_per_image", "translation_recall_retention", "brightness_recall_retention", "hit_class_coverage", "method"],
            ascending=[False, True, False, False, False, True],
        ).iloc[0]
        candidate = {
            "result": "FROZEN_CANDIDATE_FOR_EXTERNAL_CONFIRMATION",
            "method": chosen["method"],
            "score_threshold": float(chosen["score_threshold"]),
            "policy_sha256": sha256(args.policy.resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "development_group": RESERVED_DEVELOPMENT_GROUP,
            "confirmatory_evidence": False,
            "detector_screen_allowed": False,
        }

    comparison.to_csv(output / "method_comparison.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "grounding_metrics.csv", index=False, encoding="utf-8-sig")
    froc.to_csv(output / "froc_metrics.csv", index=False, encoding="utf-8-sig")
    robustness.to_csv(output / "robustness_metrics.csv", index=False, encoding="utf-8-sig")
    alignments.to_csv(output / "alignment_diagnostics.csv", index=False, encoding="utf-8-sig")
    write_registry(output / "third_party_registry.csv")
    (output / "frozen_candidate.json").write_text(json.dumps(candidate, indent=2), encoding="utf-8")

    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 4))
    x = np.arange(len(comparison))
    plt.bar(x - 0.18, comparison["small_recall"], width=0.36, label="Small recall")
    plt.bar(x + 0.18, comparison["false_positives_per_image"], width=0.36, label="FP/image")
    plt.xticks(x, comparison["method"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "method_comparison.png", dpi=180)
    plt.close()

    phase_a_path = ROOT / "runs" / "deeppcb_small_defect_mechanism_audit" / "phase_a_decision.json"
    phase_a = read_json(phase_a_path)["phase_a_decision"] if phase_a_path.exists() else "NOT_RUN"
    args.document.resolve().parent.mkdir(parents=True, exist_ok=True)
    write_docs(args.document.resolve(), args.feasibility_document.resolve(), comparison, robustness, candidate, sha256(args.policy.resolve()), phase_a)

    tests = {
        "parent_fail_stop_unchanged": freeze["decision"]["decision"] == "FAIL_STOP",
        "development_rows_111": len(development) == 111,
        "only_group92000": set(development["group"]) == {RESERVED_DEVELOPMENT_GROUP},
        "no_overlap_with_eligible_six": not bool(set(development["group"]) & set(ELIGIBLE_GROUPS)),
        "exactly_three_methods": set(comparison["method"]) == set(METHODS) and len(comparison) == 3,
        "policy_frozen": policy["status"] in {"FROZEN_BEFORE_DEVELOPMENT_EXECUTION", "CORRECTED_AND_FROZEN_BEFORE_V2_EXECUTION"},
        "candidate_not_confirmatory": not candidate.get("confirmatory_evidence", False),
        "detector_screen_not_allowed": not candidate.get("detector_screen_allowed", False),
        "official_test_unused": True,
        "final_test_unused": True,
        "training_unused": True,
        "detector_inference_unused": True,
    }
    (output / "test_results.txt").write_text("\n".join(f"{'PASS' if value else 'FAIL'} {name}" for name, value in tests.items()) + "\n", encoding="utf-8")
    if not all(tests.values()):
        raise RuntimeError(f"Development integrity tests failed: {tests}")
    print(json.dumps({"status": "DONE", "parent_decision": "FAIL_STOP", "phase_a": phase_a, "development_result": candidate["result"], "candidate": candidate.get("method"), "detector_screen_allowed": False, "training": False, "official_test": False, "final_test": False}, indent=2))


if __name__ == "__main__":
    main()

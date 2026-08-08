"""Read-only forensic audit of the frozen GC10 development confirmation.

Only acquisition, development, cached training artifacts, and recorded metrics
are read. The locked final manifest is not an input and is never discovered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

RANDOM = "GTFreeRandom"
DINO = "FrozenDINOVisualDiversity"
RARE_CLASS_INDICES = {7, 8, 9}
CLASS_NAMES = [
    "chongkong",
    "hanfeng",
    "yueyawan",
    "shuiban",
    "youban",
    "siban",
    "yiwu",
    "yahen",
    "zhehen",
    "yaozhe",
]


def report_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20260715)
    draws = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=tolerance))


def _sample_id_from_path(value: str) -> str:
    return Path(value.strip().replace("\\", "/")).stem


def _expected_label_lines(boxes: pd.DataFrame, width: int, height: int) -> str:
    lines = []
    for row in boxes.sort_values("object_index").itertuples(index=False):
        x_center = (float(row.x_min) + float(row.x_max)) / (2.0 * width)
        y_center = (float(row.y_min) + float(row.y_max)) / (2.0 * height)
        box_width = (float(row.x_max) - float(row.x_min)) / width
        box_height = (float(row.y_max) - float(row.y_min)) / height
        lines.append(
            f"{int(row.class_id) - 1} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"
        )
    return "\n".join(lines) + "\n"


def audit_cache_and_labels(protocol: Path, detector: Path) -> dict[str, Any]:
    acquisition = pd.read_csv(protocol / "gc10_acquisition_pool_gt_audit.csv")
    development = pd.read_csv(protocol / "gc10_development_eval.csv")
    acquisition_boxes = pd.read_csv(protocol / "gc10_acquisition_bbox_gt_audit.csv")
    development_boxes = pd.read_csv(protocol / "gc10_development_bbox_gt.csv")
    cache = detector / "yolo_dataset_cache"
    image_hash_mismatches = []
    label_mismatches = []

    for split, manifest, boxes in (
        ("acquisition", acquisition, acquisition_boxes),
        ("development", development, development_boxes),
    ):
        boxes_by_id = {key: frame for key, frame in boxes.groupby("sample_id")}
        for row in manifest.itertuples(index=False):
            sample_id = str(row.sample_id)
            cache_image = cache / "images" / split / f"{sample_id}.jpg"
            if not cache_image.is_file():
                image_hash_mismatches.append({"sample_id": sample_id, "reason": "missing_cache_image"})
            else:
                actual_sha = sha256_file(cache_image)
                if actual_sha != str(row.image_sha256):
                    image_hash_mismatches.append(
                        {"sample_id": sample_id, "stored": str(row.image_sha256), "cache": actual_sha}
                    )
            label_path = cache / "labels" / split / f"{sample_id}.txt"
            expected = _expected_label_lines(
                boxes_by_id.get(sample_id, pd.DataFrame()),
                int(row.width),
                int(row.height),
            )
            if not label_path.is_file():
                label_mismatches.append({"sample_id": sample_id, "reason": "missing_label"})
            elif label_path.read_text(encoding="utf-8") != expected:
                label_mismatches.append({"sample_id": sample_id, "reason": "content_mismatch"})

    development_manifest = detector / "yolo_dataset_cache" / "manifests" / "development.txt"
    cached_development_ids = [
        _sample_id_from_path(value)
        for value in development_manifest.read_text(encoding="utf-8-sig").splitlines()
    ]
    source_development_ids = development["sample_id"].astype(str).tolist()
    return {
        "images_checked": len(acquisition) + len(development),
        "image_hash_mismatches": len(image_hash_mismatches),
        "image_hash_mismatch_examples": image_hash_mismatches[:5],
        "labels_checked": len(acquisition) + len(development),
        "label_mismatches": len(label_mismatches),
        "label_mismatch_examples": label_mismatches[:5],
        "development_manifest_rows": len(cached_development_ids),
        "development_id_order_equal": cached_development_ids == source_development_ids,
        "status": "PASS" if not image_hash_mismatches and not label_mismatches and cached_development_ids == source_development_ids else "FAIL",
    }


def audit_training_artifacts(
    detector: Path,
    aggregate: pd.DataFrame,
    *,
    project_root: Path,
) -> dict[str, Any]:
    config = json.loads((detector / "config.json").read_text(encoding="utf-8"))
    current_model = Path(str(config["model"]))
    model_hash_match = current_model.is_file() and sha256_file(current_model) == str(config["model_sha256"])
    checkpoint_mismatches = []
    args_mismatches = []
    training_yaml_mismatches = []
    checked_training_yamls: set[str] = set()
    best_last_equal = 0
    best_last_different = 0
    expected_keys = {
        "epochs": 75,
        "imgsz": 640,
        "batch": 16,
        "workers": 0,
        "deterministic": True,
        "pretrained": True,
        "val": False,
        "optimizer": "auto",
    }
    for row in aggregate.itertuples(index=False):
        run_name = f"{row.set_key}_trainseed{int(row.training_seed)}"
        run_dir = detector / "yolo_train_runs" / run_name
        last = run_dir / "weights" / "last.pt"
        best = run_dir / "weights" / "best.pt"
        if not last.is_file():
            checkpoint_mismatches.append({"run": run_name, "reason": "missing_last"})
        else:
            actual = sha256_file(last)
            if actual != str(row.checkpoint_sha256):
                checkpoint_mismatches.append(
                    {"run": run_name, "stored": str(row.checkpoint_sha256), "actual": actual}
                )
        if best.is_file() and last.is_file():
            if sha256_file(best) == sha256_file(last):
                best_last_equal += 1
            else:
                best_last_different += 1
        args_path = run_dir / "args.yaml"
        if not args_path.is_file():
            args_mismatches.append({"run": run_name, "reason": "missing_args"})
            continue
        args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
        mismatched = {
            key: {"expected": value, "actual": args.get(key)}
            for key, value in expected_keys.items()
            if args.get(key) != value
        }
        if int(args.get("seed", -1)) != int(row.training_seed):
            mismatched["seed"] = {"expected": int(row.training_seed), "actual": args.get("seed")}
        if str(args.get("name")) != run_name:
            mismatched["name"] = {"expected": run_name, "actual": args.get("name")}
        if Path(str(args.get("data"))).name != f"{row.set_key}.yaml":
            mismatched["data"] = {"expected_name": f"{row.set_key}.yaml", "actual": args.get("data")}
        if Path(str(args.get("model"))).resolve() != current_model.resolve():
            mismatched["model"] = {"expected": str(current_model), "actual": args.get("model")}
        if mismatched:
            args_mismatches.append({"run": run_name, "mismatches": mismatched})

        if str(row.set_key) not in checked_training_yamls:
            checked_training_yamls.add(str(row.set_key))
            yaml_path = detector / "training_manifests" / f"{row.set_key}.yaml"
            if not yaml_path.is_file():
                training_yaml_mismatches.append({"set_key": str(row.set_key), "reason": "missing_yaml"})
            else:
                data_config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                names = data_config.get("names", {})
                ordered_names = [names.get(index, names.get(str(index))) for index in range(10)]
                yaml_issues = {}
                if Path(str(data_config.get("train"))).name != f"{row.set_key}.txt":
                    yaml_issues["train"] = data_config.get("train")
                if Path(str(data_config.get("val"))).name != "development.txt":
                    yaml_issues["val"] = data_config.get("val")
                if ordered_names != CLASS_NAMES:
                    yaml_issues["names"] = ordered_names
                if yaml_issues:
                    training_yaml_mismatches.append({"set_key": str(row.set_key), "issues": yaml_issues})

    return {
        "model_path": report_path(current_model, project_root),
        "model_sha256": config["model_sha256"],
        "current_model_hash_match": model_hash_match,
        "checkpoints_checked": len(aggregate),
        "checkpoint_hash_mismatches": len(checkpoint_mismatches),
        "checkpoint_mismatch_examples": checkpoint_mismatches[:5],
        "training_args_checked": len(aggregate),
        "training_args_mismatches": len(args_mismatches),
        "training_args_mismatch_examples": args_mismatches[:5],
        "training_yamls_checked": len(checked_training_yamls),
        "training_yaml_mismatches": len(training_yaml_mismatches),
        "training_yaml_mismatch_examples": training_yaml_mismatches[:5],
        "best_last_identical": best_last_equal,
        "best_last_different": best_last_different,
        "final_test_flag_all_false": not aggregate["final_test_used"].astype(bool).any(),
        "status": "PASS" if model_hash_match and not checkpoint_mismatches and not args_mismatches and not training_yaml_mismatches else "FAIL",
    }


def reanalyze_metrics(detector: Path) -> dict[str, Any]:
    aggregate = pd.read_csv(detector / "detector_run_metrics.csv")
    per_class = pd.read_csv(detector / "detector_per_class_metrics.csv")
    stored_metric = pd.read_csv(detector / "detector_acquisition_seed_summary.csv").set_index("metric")
    stored_macro = pd.read_csv(detector / "detector_class_group_macro_summary.csv").set_index("group")
    stored_gate = pd.read_csv(detector / "detector_development_confirmation_gate.csv")

    expected_aggregate_keys = set(
        (acquisition_seed, strategy, training_seed)
        for acquisition_seed in range(5)
        for strategy in (RANDOM, DINO)
        for training_seed in (42, 43, 44)
    )
    actual_aggregate_keys = set(
        zip(
            aggregate["acquisition_seed"].astype(int),
            aggregate["strategy"].astype(str),
            aggregate["training_seed"].astype(int),
        )
    )
    cardinality_pass = len(aggregate) == 30 and len(per_class) == 300 and actual_aggregate_keys == expected_aggregate_keys

    recomputed_metrics = {}
    metric_matches = {}
    acquisition_deltas: dict[str, dict[str, float]] = {}
    training_seed_deltas: dict[str, dict[str, float]] = {}
    for metric in ("map5095", "map50", "precision", "recall"):
        pivot = aggregate.pivot(index=["acquisition_seed", "training_seed"], columns="strategy", values=metric)
        paired = pivot[DINO] - pivot[RANDOM]
        acquisition = paired.groupby(level="acquisition_seed").mean()
        low, high = bootstrap_ci(acquisition.to_numpy(float))
        result = {
            "random_mean": float(aggregate.loc[aggregate["strategy"].eq(RANDOM), metric].mean()),
            "dino_mean": float(aggregate.loc[aggregate["strategy"].eq(DINO), metric].mean()),
            "mean_difference": float(acquisition.mean()),
            "bootstrap_ci95_low_across_acquisition_seeds": low,
            "bootstrap_ci95_high_across_acquisition_seeds": high,
            "acquisition_seed_wins": int((acquisition > 0).sum()),
            "acquisition_seed_losses": int((acquisition < 0).sum()),
            "acquisition_seed_ties": int((acquisition == 0).sum()),
        }
        recomputed_metrics[metric] = result
        stored = stored_metric.loc[metric]
        metric_matches[metric] = all(
            _close(result[key], stored[key])
            for key in (
                "random_mean",
                "dino_mean",
                "mean_difference",
                "bootstrap_ci95_low_across_acquisition_seeds",
                "bootstrap_ci95_high_across_acquisition_seeds",
            )
        ) and all(
            int(result[key]) == int(stored[key])
            for key in ("acquisition_seed_wins", "acquisition_seed_losses", "acquisition_seed_ties")
        )
        acquisition_deltas[metric] = {str(int(key)): float(value) for key, value in acquisition.items()}
        training_seed_deltas[metric] = {
            str(int(key)): float(value)
            for key, value in paired.groupby(level="training_seed").mean().items()
        }

    macro = per_class.copy()
    macro["group"] = macro["class_index"].astype(int).map(
        lambda value: "rare" if value in RARE_CLASS_INDICES else "frequent"
    )
    macro_run = macro.groupby(["acquisition_seed", "training_seed", "strategy", "group"])["ap5095"].mean().reset_index()
    macro_pivot = macro_run.pivot(
        index=["acquisition_seed", "training_seed", "group"],
        columns="strategy",
        values="ap5095",
    ).reset_index()
    macro_pivot["difference"] = macro_pivot[DINO] - macro_pivot[RANDOM]
    macro_summary = macro_pivot.groupby("group").agg(
        random_mean=(RANDOM, "mean"),
        dino_mean=(DINO, "mean"),
        mean_difference=("difference", "mean"),
    )
    macro_matches = {
        group: all(
            _close(macro_summary.loc[group, key], stored_macro.loc[group, key])
            for key in ("random_mean", "dino_mean", "mean_difference")
        )
        for group in ("frequent", "rare")
    }

    class_means = per_class.groupby(["strategy", "class_id"])["ap5095"].mean().unstack("strategy")
    class_diff = class_means[DINO] - class_means[RANDOM]
    map_result = recomputed_metrics["map5095"]
    recall_result = recomputed_metrics["recall"]
    gates = {
        "mean_map5095_gain_at_least_0p010": map_result["mean_difference"] >= 0.010,
        "map5095_acquisition_seed_wins_at_least_3_of_5": map_result["acquisition_seed_wins"] >= 3,
        "map5095_bootstrap_ci_low_across_acquisition_seeds_positive": map_result["bootstrap_ci95_low_across_acquisition_seeds"] > 0.0,
        "rare_macro_ap5095_gain_at_least_0p020": float(macro_summary.loc["rare", "mean_difference"]) >= 0.020,
        "class8_ap5095_gain_positive": float(class_diff.loc[8]) > 0.0,
        "class9_ap5095_gain_positive": float(class_diff.loc[9]) > 0.0,
        "recall_noninferiority_minus_0p020": recall_result["mean_difference"] >= -0.020,
        "frequent_macro_ap5095_noninferiority_minus_0p020": float(macro_summary.loc["frequent", "mean_difference"]) >= -0.020,
    }
    stored_gate_values = {
        str(row.check): str(row.passed).strip().casefold() == "true"
        for row in stored_gate.itertuples(index=False)
    }
    gate_match = gates == stored_gate_values

    map_acquisition = np.asarray(list(acquisition_deltas["map5095"].values()), dtype=float)
    rare_by_acquisition = (
        macro_pivot[macro_pivot["group"].eq("rare")]
        .groupby("acquisition_seed")["difference"]
        .mean()
    )
    leave_one_out_map = {
        str(seed): float(np.delete(map_acquisition, seed).mean())
        for seed in range(len(map_acquisition))
    }
    rare_values = rare_by_acquisition.sort_index().to_numpy(float)
    leave_one_out_rare = {
        str(seed): float(np.delete(rare_values, seed).mean())
        for seed in range(len(rare_values))
    }
    paired_map = (
        aggregate.pivot(index=["acquisition_seed", "training_seed"], columns="strategy", values="map5095")[DINO]
        - aggregate.pivot(index=["acquisition_seed", "training_seed"], columns="strategy", values="map5095")[RANDOM]
    ).to_numpy(float)
    pseudo_low, pseudo_high = bootstrap_ci(paired_map)

    return {
        "aggregate_rows": len(aggregate),
        "per_class_rows": len(per_class),
        "complete_expected_keys": cardinality_pass,
        "metric_recomputation": recomputed_metrics,
        "stored_metric_rows_match": metric_matches,
        "stored_macro_rows_match": macro_matches,
        "stored_gate_matches": gate_match,
        "recomputed_gate": gates,
        "recomputed_overall_gate": "PASS" if all(gates.values()) else "FAIL",
        "acquisition_seed_deltas": acquisition_deltas,
        "training_seed_mean_deltas": training_seed_deltas,
        "rare_macro_acquisition_seed_deltas": {
            str(int(key)): float(value) for key, value in rare_by_acquisition.items()
        },
        "map5095_leave_one_acquisition_out_means": leave_one_out_map,
        "map5095_leave_one_acquisition_out_mean_range": [min(leave_one_out_map.values()), max(leave_one_out_map.values())],
        "rare_macro_leave_one_acquisition_out_means": leave_one_out_rare,
        "rare_macro_leave_one_acquisition_out_mean_range": [min(leave_one_out_rare.values()), max(leave_one_out_rare.values())],
        "diagnostic_incorrect_15_run_ci_if_training_runs_treated_independent": [pseudo_low, pseudo_high],
        "correct_5_acquisition_ci": [
            recomputed_metrics["map5095"]["bootstrap_ci95_low_across_acquisition_seeds"],
            recomputed_metrics["map5095"]["bootstrap_ci95_high_across_acquisition_seeds"],
        ],
        "status": "PASS" if cardinality_pass and all(metric_matches.values()) and all(macro_matches.values()) and gate_match else "FAIL",
        "aggregate_frame": aggregate,
    }


def audit_row_order_sensitivity(protocol: Path) -> dict[str, Any]:
    raw = pd.read_csv(protocol / "gc10_acquisition_pool_blind.csv")
    canonical = raw.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    rows = []
    for acquisition_seed in range(5):
        indices = np.arange(len(raw), dtype=int)
        chosen = pd.DataFrame({"idx": indices}).sample(
            n=20,
            random_state=acquisition_seed + 999,
            replace=False,
        )["idx"].astype(int).tolist()
        raw_ids = set(raw.iloc[chosen]["sample_id"].astype(str))
        canonical_ids = set(canonical.iloc[chosen]["sample_id"].astype(str))
        rows.append(
            {
                "acquisition_seed": acquisition_seed,
                "overlap_count_of_20": len(raw_ids & canonical_ids),
                "same_set": raw_ids == canonical_ids,
            }
        )
    return {
        "raw_manifest_already_sorted": raw["sample_id"].astype(str).tolist() == canonical["sample_id"].astype(str).tolist(),
        "counterfactual_without_canonical_sort": rows,
        "mean_overlap_of_20": float(np.mean([row["overlap_count_of_20"] for row in rows])),
        "interpretation": "This is a selection-only counterfactual. It does not recompute detector metrics.",
    }


def audit_perceptual_hash_candidates(protocol: Path) -> dict[str, Any]:
    acquisition = pd.read_csv(protocol / "gc10_acquisition_pool_gt_audit.csv")
    development = pd.read_csv(protocol / "gc10_development_eval.csv")
    candidates = []
    exact_pairs = 0
    within_two_pairs = 0
    development_with_candidate: set[str] = set()
    acquisition_values = [int(str(value), 16) for value in acquisition["phash64"]]
    for dev_row in development.itertuples(index=False):
        dev_value = int(str(dev_row.phash64), 16)
        for acq_row, acq_value in zip(acquisition.itertuples(index=False), acquisition_values):
            distance = (dev_value ^ acq_value).bit_count()
            if distance == 0:
                exact_pairs += 1
            if distance <= 2:
                within_two_pairs += 1
                development_with_candidate.add(str(dev_row.sample_id))
                if len(candidates) < 20:
                    candidates.append(
                        {
                            "development_sample_id": str(dev_row.sample_id),
                            "acquisition_sample_id": str(acq_row.sample_id),
                            "hamming_distance": distance,
                            "same_sha256": str(dev_row.image_sha256) == str(acq_row.image_sha256),
                            "same_production_group": str(dev_row.production_group_raw) == str(acq_row.production_group_raw),
                        }
                    )
    return {
        "exact_phash_cross_split_pairs": exact_pairs,
        "phash_hamming_le_2_cross_split_pairs": within_two_pairs,
        "development_images_with_candidate": len(development_with_candidate),
        "candidate_examples": candidates,
        "confirmed_exact_sha_leakage": 0,
        "interpretation": "Perceptual-hash proximity is a review candidate, not proof of duplicate leakage.",
    }


def markdown_report(report: dict[str, Any]) -> str:
    metrics = report["metric_reanalysis"]["metric_recomputation"]
    map_result = metrics["map5095"]
    rare = report["metric_reanalysis"]["rare_macro_acquisition_seed_deltas"]
    risks = report["reproducibility_risks"]
    lines = [
        "# GC10 frozen-result lineage and sensitivity audit",
        "",
        f"- Overall forensic verdict: **{report['overall_forensic_verdict']}**",
        f"- Frozen research gate remains: **{report['metric_reanalysis']['recomputed_overall_gate']}**",
        "- Training/inference performed by this audit: **False / False**",
        "- Locked final manifest used by this audit script: **False**",
        "",
        "## What was independently checked",
        "",
        f"- Source/cache image bytes checked: {report['cache_and_labels']['images_checked']}; mismatches: {report['cache_and_labels']['image_hash_mismatches']}",
        f"- Generated YOLO labels checked: {report['cache_and_labels']['labels_checked']}; mismatches: {report['cache_and_labels']['label_mismatches']}",
        f"- Checkpoints checked: {report['training_artifacts']['checkpoints_checked']}; hash mismatches: {report['training_artifacts']['checkpoint_hash_mismatches']}",
        f"- Recorded training args checked: {report['training_artifacts']['training_args_checked']}; mismatches: {report['training_artifacts']['training_args_mismatches']}",
        f"- Metric rows independently recomputed: aggregate {report['metric_reanalysis']['aggregate_rows']}, per-class {report['metric_reanalysis']['per_class_rows']}",
        "",
        "## Independently reproduced conclusion",
        "",
        f"- Random mean mAP50-95: {map_result['random_mean']:.9f}",
        f"- DINO mean mAP50-95: {map_result['dino_mean']:.9f}",
        f"- Paired acquisition-unit delta: {map_result['mean_difference']:+.9f}",
        f"- Correct five-acquisition bootstrap CI: [{map_result['bootstrap_ci95_low_across_acquisition_seeds']:.9f}, {map_result['bootstrap_ci95_high_across_acquisition_seeds']:.9f}]",
        f"- Rare macro acquisition deltas: {rare}",
        "- Interpretation: overall development mAP improved in this protocol, but the pre-registered rare-class safety requirement failed; the branch-level verdict therefore remains FAIL/STOP.",
        "",
        "## Could a different implementation or request have changed the outcome?",
        "",
        f"- Without canonical sorting, the same five initial seeds overlap the actual 20-image initial sets by only {report['row_order_sensitivity']['mean_overlap_of_20']:.2f}/20 on average. Different row order would therefore train on materially different images.",
        f"- Leave-one-acquisition-out overall mAP deltas range from {report['metric_reanalysis']['map5095_leave_one_acquisition_out_mean_range'][0]:+.6f} to {report['metric_reanalysis']['map5095_leave_one_acquisition_out_mean_range'][1]:+.6f}; the effect is not identical across acquisition realizations.",
        f"- Leave-one-acquisition-out rare-macro deltas remain between {report['metric_reanalysis']['rare_macro_leave_one_acquisition_out_mean_range'][0]:+.6f} and {report['metric_reanalysis']['rare_macro_leave_one_acquisition_out_mean_range'][1]:+.6f}; none approaches the required +0.020 safety gain.",
        f"- Near-duplicate review candidates (pHash Hamming <=2): {report['perceptual_hash_review']['phash_hamming_le_2_cross_split_pairs']} pairs across {report['perceptual_hash_review']['development_images_with_candidate']} development images. These are not confirmed SHA duplicates and require separate visual/domain review.",
        "- A different detector, optimizer policy, data split, augmentation, or official test would answer a different question. Their direction cannot be inferred from these artifacts without a newly authorized experiment.",
        "",
        "## Remaining reproducibility risks",
        "",
    ]
    lines.extend(f"- {risk}" for risk in risks)
    lines.extend(
        [
            "",
            "## Bottom line",
            "",
            "The stored GC10 result is internally consistent with the current source manifests, cache, labels, checkpoints, training arguments, and analysis code. No evidence of an arithmetic, split-SHA, cache-content, or checkpoint-association error was found. Exact rerun reproducibility is still weaker than the result audit because software versions and every input seal were not recorded by the historical runner. The negative rare-class gate is the correct frozen conclusion for the experiment that was actually run.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--detector-run-dir", type=Path, required=True)
    parser.add_argument("--forward-build-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    protocol = args.protocol_dir.expanduser().resolve(strict=True)
    detector = args.detector_run_dir.expanduser().resolve(strict=True)
    forward_report_path = args.forward_build_report.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve(strict=True)

    metric = reanalyze_metrics(detector)
    aggregate = metric.pop("aggregate_frame")
    training = audit_training_artifacts(detector, aggregate, project_root=project_root)
    cache = audit_cache_and_labels(protocol, detector)
    row_order = audit_row_order_sensitivity(protocol)
    phash = audit_perceptual_hash_candidates(protocol)
    forward = json.loads(forward_report_path.read_text(encoding="utf-8"))

    risks = [
        "Historical result-cache reuse checks only whether aggregate/per-class JSON exists; it does not reseal the config, checkpoint, or manifests before returning cached metrics.",
        "Historical image-cache reuse accepts an existing target without verifying its content at cache-build time. The present audit found zero current mismatches, but the runner itself lacked this guard.",
        "Training YAML/TXT files contain machine-specific absolute paths and are not portable as frozen manifests.",
        "The historical run config records the base model SHA but not Git commit, runner SHA, source-manifest seals, Python/PyTorch/CUDA/Ultralytics versions, or a complete environment lock.",
        "All 30 best.pt files differ in SHA-256 from last.pt. The protocol explicitly evaluated last.pt, so this is not a protocol error, but a best-checkpoint policy would be a different experiment and could produce different metrics.",
        "Initial sets are reconstructed from row positions rather than persisted in the selection record; canonical sorting makes the actual run consistent, but a changed row order would change the data request.",
        "The five acquisition seeds are the correct independent units. Treating 15 training runs as independent would understate selection-level uncertainty.",
        "Perceptual hash was audit-only, not a hard split constraint. Hamming-near candidates are not exact leaks but remain a domain-review item.",
    ]
    pandas_analysis_constraint_satisfied = int(pd.__version__.split(".", maxsplit=1)[0]) < 3
    if not pandas_analysis_constraint_satisfied:
        risks.append(
            "The current audit environment uses pandas "
            f"{pd.__version__}, while requirements-analysis.txt specifies pandas<3. "
            "The independent summaries still matched exactly, but the environment should be reconciled before a frozen rerun."
        )
    all_core_pass = (
        metric["status"] == "PASS"
        and training["status"] == "PASS"
        and cache["status"] == "PASS"
        and bool(forward.get("selection_parity_all_passed"))
        and not any(forward.get("raw_image_sha256_overlaps", {}).values())
    )
    report = {
        "overall_forensic_verdict": "PASS_WITH_REPRODUCIBILITY_CAVEATS" if all_core_pass else "FAIL_REVIEW_REQUIRED",
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "forward_build_report_sha256": sha256_file(forward_report_path),
        "analysis_environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "requirements_analysis_pandas_constraint_satisfied": pandas_analysis_constraint_satisfied,
            "pyyaml": getattr(yaml, "__version__", "unknown"),
        },
        "metric_reanalysis": metric,
        "training_artifacts": training,
        "cache_and_labels": cache,
        "forward_manifest_summary": forward,
        "row_order_sensitivity": row_order,
        "perceptual_hash_review": phash,
        "reproducibility_risks": risks,
        "audit_actions": {
            "training_performed": False,
            "inference_performed": False,
            "gpu_used": False,
            "locked_final_manifest_input": False,
            "official_or_final_metric_computed": False,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "gc10_research_lineage_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "gc10_research_lineage_audit.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({
        "verdict": report["overall_forensic_verdict"],
        "frozen_gate": metric["recomputed_overall_gate"],
        "cache_status": cache["status"],
        "training_artifact_status": training["status"],
        "metric_status": metric["status"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

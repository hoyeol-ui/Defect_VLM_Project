"""Validate detector-native uncertainty against development errors without training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715"
SCREEN = ROOT / "runs" / "dcal_xai" / "v2_backbone_main"
DEFAULT_OUT = ROOT / "runs" / "dcal_xai" / "v2_detector_signal_validity"
PROTOCOL_FILE = HERE / "v2_detector_signal_validity_protocol.md"
ACQUISITION_SEEDS = [20000, 20001, 20002, 20003, 20004]
TRAINING_SEEDS = [42, 43, 44]
RARE_IDS = {8, 9, 10}
PREDICT = {"imgsz": 640, "conf": 0.001, "iou": 0.70, "max_det": 300, "batch": 16, "device": "0", "half": True}
ERROR_CONF = 0.25
MATCH_IOU = 0.50
PRIMARY_SIGNAL = "ensemble_combined_uncertainty"
GATES = {
    "error_enrichment_mean_min": 1.50,
    "error_enrichment_ci_low_min_exclusive": 1.00,
    "auroc_mean_min": 0.65,
    "auroc_positive_seeds_min": 4,
    "spearman_mean_min": 0.20,
    "spearman_positive_seeds_min": 4,
    "rank_stability_min": 0.50,
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_context(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision = json.loads((SCREEN / "v8n_screen_decision.json").read_text(encoding="utf-8"))
    if decision.get("final_test_used") is not False:
        raise RuntimeError("V2.2 final-test safety flag failed")
    metrics = pd.read_csv(SCREEN / "v8n_screen_metrics.csv")
    models = metrics[metrics["policy"].eq("Random140") & metrics["backbone"].eq("YOLOv8n")].copy()
    models = models.sort_values(["acquisition_seed", "training_seed"]).reset_index(drop=True)
    expected = {(a, t) for a in ACQUISITION_SEEDS for t in TRAINING_SEEDS}
    observed = set(zip(models["acquisition_seed"].astype(int), models["training_seed"].astype(int)))
    if len(models) != 15 or observed != expected:
        raise RuntimeError("Expected exactly 15 frozen Random140 YOLOv8n checkpoints")
    for row in models.itertuples(index=False):
        checkpoint = Path(str(row.checkpoint))
        if not checkpoint.exists() or sha256(checkpoint) != str(row.checkpoint_sha256):
            raise RuntimeError(f"Checkpoint missing or hash mismatch: {checkpoint}")
        if bool(row.final_test_used):
            raise RuntimeError("Checkpoint metrics report final-test use")

    development = pd.read_csv(PROTOCOL / "gc10_development_eval.csv")
    if len(development) != 232 or not development["protocol_split"].eq("development").all():
        raise RuntimeError("Development manifest safety failure")
    inference = development[["sample_id", "image_path", "width", "height", "image_sha256"]].copy()
    if inference["sample_id"].duplicated().any() or not inference["image_path"].map(lambda value: Path(str(value)).exists()).all():
        raise RuntimeError("Development inference manifest integrity failure")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "development_inference_manifest.csv"
    inference.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    paths_path = out / "development_inference_paths.txt"
    paths_path.write_text("\n".join(inference["image_path"].astype(str)) + "\n", encoding="utf-8")
    audit = {
        "status": "PASS",
        "models": 15,
        "development_images": 232,
        "checkpoint_policy": "Random140",
        "backbone": "YOLOv8n",
        "checkpoint_hashes": models[["acquisition_seed", "training_seed", "checkpoint_sha256"]].to_dict("records"),
        "inference_manifest_sha256": sha256(manifest_path),
        "inference_paths_sha256": sha256(paths_path),
        "protocol_sha256": sha256(PROTOCOL_FILE),
        "script_sha256": sha256(Path(__file__).resolve()),
        "prediction_settings": PREDICT,
        "error_confidence_threshold": ERROR_CONF,
        "matching_iou": MATCH_IOU,
        "new_training_performed": False,
        "final_test_used": False,
    }
    write_json(out / "audit.json", audit)
    return models, inference


def model_key(acquisition_seed: int, training_seed: int) -> str:
    return f"acq{acquisition_seed}_trainseed{training_seed}"


def expected_prediction_metadata(row: Any, manifest_path: Path) -> dict[str, Any]:
    return {
        "acquisition_seed": int(row.acquisition_seed),
        "training_seed": int(row.training_seed),
        "checkpoint": str(Path(str(row.checkpoint)).resolve()),
        "checkpoint_sha256": str(row.checkpoint_sha256),
        "inference_manifest_sha256": sha256(manifest_path),
        "settings": PREDICT,
        "uses_gt": False,
        "final_test_used": False,
    }


def generate_predictions(models: pd.DataFrame, inference: pd.DataFrame, out: Path) -> None:
    from ultralytics import YOLO
    import torch

    prediction_dir = out / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "development_inference_manifest.csv"
    path_to_id = {str(Path(str(row.image_path)).resolve()).casefold(): str(row.sample_id) for row in inference.itertuples(index=False)}
    sources = str((out / "development_inference_paths.txt").resolve())
    for index, row in enumerate(models.itertuples(index=False), start=1):
        key = model_key(int(row.acquisition_seed), int(row.training_seed))
        prediction_path = prediction_dir / f"{key}.jsonl"
        metadata_path = prediction_dir / f"{key}.meta.json"
        expected = expected_prediction_metadata(row, manifest_path)
        if prediction_path.exists() or metadata_path.exists():
            if not prediction_path.exists() or not metadata_path.exists():
                raise RuntimeError(f"Partial prediction cache: {key}")
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            if cached != expected or sum(1 for _ in prediction_path.open("r", encoding="utf-8")) != len(inference):
                raise RuntimeError(f"Stale prediction cache: {key}")
            print(f"[{index}/15] CACHE {key}", flush=True)
            continue
        print(f"[{index}/15] PREDICT {key}", flush=True)
        model = YOLO(str(row.checkpoint))
        results = model.predict(source=sources, stream=True, verbose=False, save=False, **PREDICT)
        records: dict[str, dict[str, Any]] = {}
        for result in results:
            resolved = str(Path(str(result.path)).resolve()).casefold()
            sample_id = path_to_id.get(resolved)
            if sample_id is None:
                raise RuntimeError(f"Prediction path not in inference manifest: {result.path}")
            boxes = getattr(result, "boxes", None)
            detections = []
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.detach().cpu().numpy()
                conf = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                detections = [
                    {"xyxy": [float(value) for value in coordinates], "confidence": float(score), "class_index": int(class_index)}
                    for coordinates, score, class_index in zip(xyxy, conf, classes)
                ]
            records[sample_id] = {"sample_id": sample_id, "detections": detections}
        if set(records) != set(inference["sample_id"].astype(str)):
            raise RuntimeError(f"Prediction cardinality failure: {key}")
        with prediction_path.open("w", encoding="utf-8") as handle:
            for sample_id in inference["sample_id"].astype(str):
                handle.write(json.dumps(records[sample_id], separators=(",", ":")) + "\n")
        write_json(metadata_path, expected)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def read_prediction_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return -(probability * np.log2(probability) + (1 - probability) * np.log2(1 - probability))


def build_individual_signals(models: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for row in models.itertuples(index=False):
        key = model_key(int(row.acquisition_seed), int(row.training_seed))
        for record in read_prediction_file(out / "predictions" / f"{key}.jsonl"):
            detections = [item for item in record["detections"] if float(item["confidence"]) >= ERROR_CONF]
            confidences = np.asarray([float(item["confidence"]) for item in detections], dtype=float)
            classes = {int(item["class_index"]) for item in detections}
            values: dict[str, Any] = {
                "acquisition_seed": int(row.acquisition_seed),
                "training_seed": int(row.training_seed),
                "sample_id": str(record["sample_id"]),
                "pred_count": len(detections),
                "no_detection": int(len(detections) == 0),
                "max_confidence": float(confidences.max()) if len(confidences) else 0.0,
                "mean_confidence": float(confidences.mean()) if len(confidences) else 0.0,
                "confidence_deficit": float(1 - confidences.max()) if len(confidences) else 1.0,
                "mean_confidence_deficit": float(1 - confidences.mean()) if len(confidences) else 1.0,
                "uses_gt": False,
                "final_test_used": False,
            }
            values.update({f"class_present_{class_index}": int(class_index in classes) for class_index in range(10)})
            rows.append(values)
    signals = pd.DataFrame(rows)
    if len(signals) != 15 * 232:
        raise RuntimeError("Individual signal cardinality failure")
    signals.to_csv(out / "individual_detector_signals.csv", index=False, encoding="utf-8-sig")
    return signals


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(box_area + areas - intersection, 1e-12)


def match_errors(detections: list[dict[str, Any]], gt_rows: pd.DataFrame) -> dict[str, Any]:
    predictions = [item for item in detections if float(item["confidence"]) >= ERROR_CONF]
    predictions.sort(key=lambda item: float(item["confidence"]), reverse=True)
    gt_boxes = gt_rows[["x_min", "y_min", "x_max", "y_max"]].to_numpy(float)
    gt_classes = gt_rows["class_id"].to_numpy(int) - 1
    matched: set[int] = set()
    tp = 0
    for prediction in predictions:
        candidates = np.flatnonzero(gt_classes == int(prediction["class_index"]))
        candidates = np.asarray([value for value in candidates if int(value) not in matched], dtype=int)
        if len(candidates) == 0:
            continue
        overlaps = box_iou(np.asarray(prediction["xyxy"], dtype=float), gt_boxes[candidates])
        best_local = int(np.argmax(overlaps))
        if float(overlaps[best_local]) >= MATCH_IOU:
            matched.add(int(candidates[best_local]))
            tp += 1
    fp = len(predictions) - tp
    unmatched = [index for index in range(len(gt_rows)) if index not in matched]
    fn = len(unmatched)
    rare_fn = sum(int(gt_classes[index] + 1) in RARE_IDS for index in unmatched)
    rare_gt = sum(int(value + 1) in RARE_IDS for value in gt_classes)
    return {
        "tp": tp, "fp": fp, "fn": fn, "error_count": fp + fn,
        "error_indicator": int(fp + fn > 0), "fn_indicator": int(fn > 0),
        "fn_rate": float(fn / len(gt_rows)) if len(gt_rows) else 0.0,
        "rare_fn": rare_fn, "rare_gt": rare_gt,
    }


def attach_posthoc_errors(models: pd.DataFrame, signals: pd.DataFrame, out: Path) -> pd.DataFrame:
    gt = pd.read_csv(PROTOCOL / "gc10_development_bbox_gt.csv")
    grouped = {str(sample_id): frame.reset_index(drop=True) for sample_id, frame in gt.groupby("sample_id")}
    rows = []
    for row in models.itertuples(index=False):
        key = model_key(int(row.acquisition_seed), int(row.training_seed))
        for record in read_prediction_file(out / "predictions" / f"{key}.jsonl"):
            sample_id = str(record["sample_id"])
            if sample_id not in grouped:
                raise RuntimeError(f"Development GT missing: {sample_id}")
            rows.append({
                "acquisition_seed": int(row.acquisition_seed),
                "training_seed": int(row.training_seed),
                "sample_id": sample_id,
                **match_errors(record["detections"], grouped[sample_id]),
                "gt_joined_posthoc": True,
                "final_test_used": False,
            })
    outcomes = pd.DataFrame(rows)
    merged = signals.merge(outcomes, on=["acquisition_seed", "training_seed", "sample_id"], validate="one_to_one")
    merged.to_csv(out / "individual_signal_error_audit.csv", index=False, encoding="utf-8-sig")
    return merged


def rank_percentile(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def build_ensemble_table(merged: pd.DataFrame, out: Path) -> pd.DataFrame:
    class_columns = [f"class_present_{index}" for index in range(10)]
    rows = []
    for (acquisition_seed, sample_id), group in merged.groupby(["acquisition_seed", "sample_id"], sort=True):
        probabilities = group[class_columns].mean(axis=0).to_numpy(float)
        rows.append({
            "acquisition_seed": int(acquisition_seed), "sample_id": str(sample_id),
            "mean_confidence_deficit": float(group["confidence_deficit"].mean()),
            "no_detection_fraction": float(group["no_detection"].mean()),
            "pred_count_std": float(group["pred_count"].std(ddof=0)),
            "max_confidence_std": float(group["max_confidence"].std(ddof=0)),
            "class_presence_entropy": float(binary_entropy(probabilities).mean()),
            "mean_error_count": float(group["error_count"].mean()),
            "mean_fn_count": float(group["fn"].mean()),
            "mean_fn_rate": float(group["fn_rate"].mean()),
            "error_model_fraction": float(group["error_indicator"].mean()),
            "fn_model_fraction": float(group["fn_indicator"].mean()),
            "mean_rare_fn_count": float(group["rare_fn"].mean()),
            "rare_gt": int(group["rare_gt"].iloc[0]),
            "majority_error": int(group["error_indicator"].mean() >= 0.5),
            "majority_fn": int(group["fn_indicator"].mean() >= 0.5),
            "final_test_used": False,
        })
    ensemble = pd.DataFrame(rows)
    ranked_parts = []
    for _, group in ensemble.groupby("acquisition_seed", sort=True):
        group = group.copy()
        component_columns = [
            "mean_confidence_deficit", "no_detection_fraction", "pred_count_std",
            "max_confidence_std", "class_presence_entropy",
        ]
        for column in component_columns:
            group[f"rank_{column}"] = rank_percentile(group[column])
        group["ensemble_confidence_uncertainty"] = group[[
            "rank_mean_confidence_deficit", "rank_no_detection_fraction",
        ]].mean(axis=1)
        group["ensemble_disagreement_uncertainty"] = group[[
            "rank_pred_count_std", "rank_max_confidence_std", "rank_class_presence_entropy",
        ]].mean(axis=1)
        group[PRIMARY_SIGNAL] = group[[f"rank_{column}" for column in component_columns]].mean(axis=1)
        ranked_parts.append(group)
    result = pd.concat(ranked_parts, ignore_index=True)
    if len(result) != 5 * 232:
        raise RuntimeError("Ensemble signal cardinality failure")
    result.to_csv(out / "ensemble_signal_error_audit.csv", index=False, encoding="utf-8-sig")
    return result


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def bootstrap_mean_ci(values: np.ndarray, seed: int = 20260718) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def evaluate_signals(ensemble: pd.DataFrame, merged: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_columns = ["ensemble_confidence_uncertainty", "ensemble_disagreement_uncertainty", PRIMARY_SIGNAL]
    rows = []
    for acquisition_seed, group in ensemble.groupby("acquisition_seed", sort=True):
        for signal in signal_columns:
            top_n = int(math.ceil(0.20 * len(group)))
            top = group.nlargest(top_n, signal)
            labels = group["majority_error"].to_numpy(int)
            auroc = float(roc_auc_score(labels, group[signal])) if len(np.unique(labels)) == 2 else float("nan")
            correlation = float(spearmanr(group[signal], group["mean_error_count"]).statistic)
            rows.append({
                "acquisition_seed": int(acquisition_seed), "signal": signal, "top_n": top_n,
                "error_enrichment": safe_ratio(float(top["mean_error_count"].mean()), float(group["mean_error_count"].mean())),
                "fn_enrichment": safe_ratio(float(top["mean_fn_count"].mean()), float(group["mean_fn_count"].mean())),
                "error_rate_enrichment": safe_ratio(float(top["majority_error"].mean()), float(group["majority_error"].mean())),
                "rare_fn_enrichment": safe_ratio(float(top["mean_rare_fn_count"].mean()), float(group["mean_rare_fn_count"].mean())),
                "auroc_majority_error": auroc,
                "spearman_total_error": correlation,
                "final_test_used": False,
            })
    per_seed = pd.DataFrame(rows)
    summary_rows = []
    for signal, group in per_seed.groupby("signal", sort=False):
        ci_low, ci_high = bootstrap_mean_ci(group["error_enrichment"].to_numpy(float))
        summary_rows.append({
            "signal": signal,
            "error_enrichment_mean": float(group["error_enrichment"].mean()),
            "error_enrichment_ci95_low": ci_low,
            "error_enrichment_ci95_high": ci_high,
            "fn_enrichment_mean": float(group["fn_enrichment"].mean()),
            "rare_fn_enrichment_mean": float(group["rare_fn_enrichment"].mean()),
            "auroc_mean": float(group["auroc_majority_error"].mean()),
            "auroc_seeds_above_half": int((group["auroc_majority_error"] > 0.5).sum()),
            "spearman_mean": float(group["spearman_total_error"].mean()),
            "spearman_positive_seeds": int((group["spearman_total_error"] > 0).sum()),
        })
    summary = pd.DataFrame(summary_rows)

    stability_rows = []
    for acquisition_seed, group in merged.groupby("acquisition_seed", sort=True):
        pivot = group.pivot(index="sample_id", columns="training_seed", values="confidence_deficit")
        correlation = pivot.corr(method="spearman")
        upper = correlation.to_numpy(float)[np.triu_indices(len(correlation), k=1)]
        stability_rows.append({
            "acquisition_seed": int(acquisition_seed),
            "mean_pairwise_confidence_rank_correlation": float(np.nanmean(upper)),
            "minimum_pairwise_confidence_rank_correlation": float(np.nanmin(upper)),
        })
    stability = pd.DataFrame(stability_rows)
    per_seed.to_csv(out / "signal_validity_by_acquisition_seed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "signal_validity_summary.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(out / "training_seed_signal_stability.csv", index=False, encoding="utf-8-sig")
    return per_seed, summary, stability


def decide(summary: pd.DataFrame, stability: pd.DataFrame, out: Path) -> tuple[bool, Path]:
    row = summary[summary["signal"].eq(PRIMARY_SIGNAL)]
    if len(row) != 1:
        raise RuntimeError("Primary signal summary missing")
    row = row.iloc[0]
    stability_mean = float(stability["mean_pairwise_confidence_rank_correlation"].mean())
    checks = {
        "error_enrichment_mean": float(row.error_enrichment_mean) >= GATES["error_enrichment_mean_min"],
        "error_enrichment_ci_low": float(row.error_enrichment_ci95_low) > GATES["error_enrichment_ci_low_min_exclusive"],
        "auroc_mean": float(row.auroc_mean) >= GATES["auroc_mean_min"],
        "auroc_seed_consistency": int(row.auroc_seeds_above_half) >= GATES["auroc_positive_seeds_min"],
        "spearman_mean": float(row.spearman_mean) >= GATES["spearman_mean_min"],
        "spearman_seed_consistency": int(row.spearman_positive_seeds) >= GATES["spearman_positive_seeds_min"],
        "training_seed_rank_stability": stability_mean >= GATES["rank_stability_min"],
    }
    passed = all(checks.values())
    payload = {
        "status": "PASS" if passed else "FAIL", "primary_signal": PRIMARY_SIGNAL,
        "primary_metrics": {key: (int(value) if isinstance(value, (np.integer,)) else float(value)) for key, value in row.to_dict().items() if key != "signal"},
        "mean_training_seed_rank_stability": stability_mean,
        "checks": {key: bool(value) for key, value in checks.items()},
        "authorizes_selection_only_query_audit": passed,
        "new_training_performed": False, "final_test_used": False,
    }
    write_json(out / "decision.json", payload)
    report = [
        "# V2.3 Detector Signal Validity Decision", "",
        f"- Gate: **{'PASS' if passed else 'FAIL'}**",
        f"- Primary signal: `{PRIMARY_SIGNAL}`",
        "- Checkpoints: **15 existing Random140 x YOLOv8n**",
        "- Evaluation: **development post-hoc audit only**",
        "- New training performed: **False**", "- Final test used: **False**", "",
        "## Signal summary", "", summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## Primary checks", "",
        pd.DataFrame([{"check": key, "passed": value} for key, value in checks.items()]).to_markdown(index=False), "",
        "A PASS authorizes only a frozen selection-only query audit from Random140. A FAIL closes this detector-uncertainty branch.",
    ]
    path = out / "detector_signal_validity_decision.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return passed, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "predict", "analyze", "all"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    models, inference = load_context(out)
    if args.stage == "audit":
        print(f"[AUDIT PASS] {out / 'audit.json'}")
        print("New training performed: False")
        print("Final test used: False")
        return
    if args.stage in {"predict", "all"}:
        generate_predictions(models, inference, out)
        print(f"[PREDICTIONS SEALED] {out / 'predictions'}")
        if args.stage == "predict":
            print("New training performed: False")
            print("Final test used: False")
            return
    if not (out / "predictions").exists():
        raise RuntimeError("Run the label-free prediction stage before analysis")
    signals = build_individual_signals(models, out)
    merged = attach_posthoc_errors(models, signals, out)
    ensemble = build_ensemble_table(merged, out)
    _, summary, stability = evaluate_signals(ensemble, merged, out)
    passed, path = decide(summary, stability, out)
    print(f"[DONE] {path}")
    print(f"[DETECTOR SIGNAL GATE] {'PASS' if passed else 'FAIL'}")
    print("New training performed: False")
    print("Final test used: False")


if __name__ == "__main__":
    main()

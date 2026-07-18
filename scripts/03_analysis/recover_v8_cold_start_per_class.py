"""Recover round-1 per-class metrics from existing cold-start checkpoints.

Safety contract:
- validation only; ``YOLO.train`` is never called
- evaluates only the frozen 177-image NEU development set
- verifies every validation image by SHA-256 before loading a checkpoint
- refuses any config/YAML containing a final-test reference
- XML-derived selection statistics are used only for post-hoc audit
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs" / "active_learning_v8_cold_start_confirmation" / "v8_cold_start_visual_confirm_main"
RANDOM = "GTFreeRandom"
VISUAL = "GTFreeDatasetBalancedVisualDiversity"
NEU6 = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
EXPECTED_SEEDS = list(range(52, 62))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_class(value: Any) -> str:
    return str(value).replace("rolled-in-scale", "rolled-in_scale")


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def exact_sign_flip(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    observed = abs(float(values.mean()))
    null = [abs(float(np.mean(values * np.asarray(signs)))) for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    valid = pvalues.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype=float)
    running = 0.0
    count = len(valid)
    for rank, (idx, value) in enumerate(valid.items()):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted.loc[idx] = running
    return adjusted


def read_yaml_val_dir(yaml_path: Path) -> Path:
    text = yaml_path.read_text(encoding="utf-8")
    lowered = text.lower().replace("\\", "/")
    if "final_test" in lowered or "final-test" in lowered:
        raise RuntimeError(f"Final-test reference found in YAML: {yaml_path}")
    base: Path | None = None
    val_value: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("path:"):
            base = Path(line.split(":", 1)[1].strip())
        elif line.startswith("val:"):
            val_value = line.split(":", 1)[1].strip()
    if base is None or val_value is None:
        raise RuntimeError(f"Cannot resolve val split from {yaml_path}")
    val_dir = Path(val_value)
    if not val_dir.is_absolute():
        val_dir = base / val_dir
    return val_dir.resolve()


def verify_development_split(config: dict[str, Any], yaml_paths: list[Path]) -> dict[str, Any]:
    if bool(config.get("final_test_used", False)):
        raise RuntimeError("Run config records final_test_used=True.")
    config_text = json.dumps(config, ensure_ascii=False).lower().replace("\\", "/")
    if "final_test_v7" in config_text or "final-test" in config_text:
        raise RuntimeError("Run config contains a final-test reference; refusing validation.")
    manifest = Path(str(config["development_eval_path"]))
    if manifest.stem != "development_eval_v7":
        raise RuntimeError(f"Unexpected development manifest: {manifest}")
    dev = pd.read_csv(manifest)
    filters = list(config.get("development_eval_dataset_filter", []))
    if filters:
        dev = dev[dev["dataset_type"].astype(str).isin(filters)].copy()
    expected = set(dev["sha256"].astype(str))
    expected_size = int(config.get("development_eval_size_after_filter", -1))
    if len(dev) != expected_size or len(expected) != 177:
        raise RuntimeError(f"Expected 177 unique NEU development images, got rows={len(dev)} hashes={len(expected)}")

    verified_dirs: dict[str, dict[str, Any]] = {}
    for yaml_path in sorted(set(yaml_paths)):
        val_dir = read_yaml_val_dir(yaml_path)
        key = str(val_dir)
        if key in verified_dirs:
            continue
        images = sorted(p for p in val_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
        actual = {sha256(path) for path in images}
        if len(images) != 177 or actual != expected:
            raise RuntimeError(
                f"Development split hash mismatch: {val_dir} images={len(images)} "
                f"missing={len(expected - actual)} extra={len(actual - expected)}"
            )
        verified_dirs[key] = {"val_dir": key, "num_images": len(images), "hash_set_matches_manifest": True}
    return {"manifest": str(manifest), "manifest_sha256": sha256(manifest), "num_images": 177, "verified_val_dirs": list(verified_dirs.values())}


def recover_one(row: pd.Series, out_dir: Path, device: str, batch: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = int(row["acquisition_seed"])
    strategy = str(row["strategy"])
    yaml_path = Path(str(row["yaml_path"]))
    checkpoint = Path(str(row["train_run_dir"])) / "weights" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = sha256(checkpoint)
    print(f"[DEV VAL ONLY] seed={seed} strategy={strategy}", flush=True)
    metrics = YOLO(str(checkpoint)).val(
        data=str(yaml_path),
        split="val",
        imgsz=640,
        batch=batch,
        device=device,
        workers=0,
        plots=False,
        save_json=False,
        verbose=False,
        project=str(out_dir / "ultralytics_val_runs"),
        name=f"seed{seed}_{strategy}_round1_dev_only",
        exist_ok=True,
    )
    box = metrics.box
    aggregate = {
        "acquisition_seed": seed,
        "training_seed": int(row["training_seed"]),
        "strategy": strategy,
        "round": 1,
        "yaml_path": str(yaml_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "status": "success",
        "development_eval_only": True,
        "final_test_used": False,
        "map50_recovered": as_float(getattr(box, "map50", np.nan)),
        "map5095_recovered": as_float(getattr(box, "map", np.nan)),
        "precision_recovered": as_float(getattr(box, "mp", np.nan)),
        "recall_recovered": as_float(getattr(box, "mr", np.nan)),
        "map50_original": as_float(row["map50"]),
        "map5095_original": as_float(row["map5095"]),
    }
    aggregate["abs_map5095_recovery_difference"] = abs(aggregate["map5095_recovered"] - aggregate["map5095_original"])

    per_class: list[dict[str, Any]] = []
    for item in metrics.summary():
        class_name = normalize_class(item.get("Class", ""))
        if class_name not in NEU6:
            continue
        precision = as_float(item.get("Box-P"))
        recall = as_float(item.get("Box-R"))
        per_class.append(
            {
                "acquisition_seed": seed,
                "training_seed": int(row["training_seed"]),
                "strategy": strategy,
                "round": 1,
                "class_name": class_name,
                "val_images": as_float(item.get("Images")),
                "val_instances": as_float(item.get("Instances")),
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan,
                "ap50": as_float(item.get("mAP50")),
                "ap5095": as_float(item.get("mAP50-95")),
                "checkpoint_sha256": checkpoint_hash,
            }
        )
    recovered_classes = {item["class_name"] for item in per_class}
    if recovered_classes != set(NEU6):
        raise RuntimeError(
            f"Expected all NEU6 per-class metrics for seed={seed} strategy={strategy}; "
            f"recovered={sorted(recovered_classes)} expected={sorted(NEU6)}"
        )
    return aggregate, per_class


def paired_outputs(per_class: pd.DataFrame, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = ["ap5095", "ap50", "precision", "recall", "f1"]
    visual = per_class[per_class["strategy"].eq(VISUAL)]
    random = per_class[per_class["strategy"].eq(RANDOM)]
    paired = visual.merge(random, on=["acquisition_seed", "class_name", "round"], suffixes=("_visual", "_random"), validate="one_to_one")
    keep = ["acquisition_seed", "class_name", "val_images_visual", "val_instances_visual"]
    output = paired[keep].rename(columns={"val_images_visual": "val_images", "val_instances_visual": "val_instances"}).copy()
    for metric in metrics:
        output[f"{metric}_visual"] = paired[f"{metric}_visual"]
        output[f"{metric}_random"] = paired[f"{metric}_random"]
        output[f"delta_{metric}"] = paired[f"{metric}_visual"] - paired[f"{metric}_random"]

    summaries: list[dict[str, Any]] = []
    for class_name, sub in output.groupby("class_name", sort=False):
        for metric in metrics:
            values = sub[f"delta_{metric}"].to_numpy(float)
            summaries.append(
                {
                    "class_name": class_name,
                    "metric": metric,
                    "n_pairs": len(values),
                    "mean_difference": float(np.nanmean(values)),
                    "median_difference": float(np.nanmedian(values)),
                    "std_difference": float(np.nanstd(values, ddof=1)),
                    "wins": int(np.sum(values > 0)),
                    "losses": int(np.sum(values < 0)),
                    "ties": int(np.sum(values == 0)),
                    "exact_sign_flip_p_raw": exact_sign_flip(values),
                }
            )
    summary = pd.DataFrame(summaries)
    ap_mask = summary["metric"].eq("ap5095")
    summary["holm_p_across_6_classes_ap5095"] = np.nan
    summary.loc[ap_mask, "holm_p_across_6_classes_ap5095"] = holm_adjust(
        summary.loc[ap_mask].set_index("class_name")["exact_sign_flip_p_raw"]
    ).reindex(summary.loc[ap_mask, "class_name"]).to_numpy()

    dist = pd.read_csv(run_dir / "actual_class_distribution_by_round.csv")
    dist["actual_xml_class"] = dist["actual_xml_class"].map(normalize_class)
    dist = dist[dist["actual_xml_class"].isin(NEU6)]
    wide = dist.pivot_table(index=["acquisition_seed", "strategy", "actual_xml_class"], columns="round", values="bbox_instance_count", aggfunc="sum", fill_value=0)
    wide["query_bbox_instances"] = wide.get(1, 0) - wide.get(0, 0)
    query = wide["query_bbox_instances"].unstack("strategy").reset_index().rename(columns={"actual_xml_class": "class_name"})
    query["delta_query_bbox_instances"] = query[VISUAL] - query[RANDOM]
    audit = output.merge(query[["acquisition_seed", "class_name", RANDOM, VISUAL, "delta_query_bbox_instances"]], on=["acquisition_seed", "class_name"], how="left")
    correlations: list[dict[str, Any]] = []
    for class_name, sub in audit.groupby("class_name", sort=False):
        rho, pvalue = spearmanr(sub["delta_query_bbox_instances"], sub["delta_ap5095"])
        correlations.append(
            {
                "class_name": class_name,
                "n_seeds": len(sub),
                "spearman_rho_query_bbox_delta_vs_ap5095_delta": rho,
                "p_value_exploratory": pvalue,
            }
        )
    return output, summary, pd.DataFrame(correlations)


def table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if df.empty:
        return "_No rows._"
    return df[columns].to_markdown(index=False, floatfmt=".6f") if columns else df.to_markdown(index=False, floatfmt=".6f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = run_dir / "round1_per_class_recovery"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    results = pd.read_csv(run_dir / "all_round_results.csv")
    targets = results[
        results["round"].eq(1)
        & results["strategy"].isin([RANDOM, VISUAL])
        & results["train_status"].eq("success")
    ].sort_values(["acquisition_seed", "strategy"], kind="mergesort")
    if len(targets) != 20 or sorted(targets["acquisition_seed"].unique().tolist()) != EXPECTED_SEEDS:
        raise RuntimeError("Expected 20 successful round1 checkpoints across seeds 52-61.")
    safety = verify_development_split(config, [Path(value) for value in targets["yaml_path"]])
    (out_dir / "development_split_verification.json").write_text(json.dumps(safety, indent=2), encoding="utf-8")

    registry_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for _, row in targets.iterrows():
        aggregate, per_class = recover_one(row, out_dir, args.device, args.batch)
        registry_rows.append(aggregate)
        per_class_rows.extend(per_class)
        pd.DataFrame(registry_rows).to_csv(out_dir / "checkpoint_evaluation_registry.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(per_class_rows).to_csv(out_dir / "recovered_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    registry = pd.DataFrame(registry_rows)
    per_class = pd.DataFrame(per_class_rows)
    paired, summary, correlations = paired_outputs(per_class, run_dir)
    paired.to_csv(out_dir / "paired_per_class_visual_minus_random.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "per_class_paired_summary.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(out_dir / "selection_bbox_ap_correlation.csv", index=False, encoding="utf-8-sig")

    aggregate_ok = bool((registry["abs_map5095_recovery_difference"] <= 1e-4).all())
    ap_summary = summary[summary["metric"].eq("ap5095")].copy()
    report = [
        "# V8 Cold-Start Round1 Per-Class Recovery",
        "",
        "- Training performed: **False**",
        "- Final test used: **False**",
        "- Evaluation: frozen 177-image NEU development split only",
        "- Checkpoints evaluated: **20**",
        f"- Recovered aggregate mAP50-95 matches stored values within 1e-4: **{aggregate_ok}**",
        "- Purpose: post-hoc mechanism audit; this cannot reverse the confirmatory gate failure",
        "",
        "## Per-class AP50-95 paired result",
        "",
        table(ap_summary, ["class_name", "n_pairs", "mean_difference", "median_difference", "wins", "losses", "exact_sign_flip_p_raw", "holm_p_across_6_classes_ap5095"]),
        "",
        "## Query XML bbox change vs AP50-95 change",
        "",
        "XML is used only after acquisition for audit. Correlations are exploratory with n=10 per class.",
        "",
        table(correlations),
        "",
        "## Checkpoint registry",
        "",
        table(registry, ["acquisition_seed", "strategy", "status", "map5095_original", "map5095_recovered", "abs_map5095_recovery_difference", "final_test_used"]),
        "",
        "The primary Visual-vs-Random confirmation remains FAIL regardless of these class-level findings.",
    ]
    summary_path = out_dir / "round1_per_class_recovery_summary.md"
    summary_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("=" * 100)
    print("[DONE] V8 cold-start per-class recovery")
    print("Training performed: False")
    print("Final test used: False")
    print(f"[SUMMARY] {summary_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()

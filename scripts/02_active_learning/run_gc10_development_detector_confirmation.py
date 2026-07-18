"""Train the frozen GC10 Random-vs-DINO 30-model development confirmation.

Only acquisition images are used for fitting and only development images are
used for evaluation. The final manifest is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715"
DEFAULT_SELECTION = ROOT / "runs" / "gc10_taxonomy_selection_audit" / "gc10_random_vs_dino_200seed_20260715"
DEFAULT_OUT = ROOT / "runs" / "gc10_detector_confirmation" / "gc10_dev_confirm_5acq_3train_20260715"
DEFAULT_MODEL = ROOT / "yolov8n.pt"
ACQUISITION_SEEDS = [0, 1, 2, 3, 4]
TRAINING_SEEDS = [42, 43, 44]
STRATEGIES = ["GTFreeRandom", "FrozenDINOVisualDiversity"]
RANDOM = STRATEGIES[0]
DINO = STRATEGIES[1]
INITIAL_SIZE = 20
QUERY_SIZE = 20
CLASS_NAMES = [
    "chongkong", "hanfeng", "yueyawan", "shuiban", "youban",
    "siban", "yiwu", "yahen", "zhehen", "yaozhe",
]
RARE_CLASS_INDICES = [7, 8, 9]
FREQUENT_CLASS_INDICES = list(range(7))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def write_label(path: Path, boxes: pd.DataFrame, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in boxes.sort_values("object_index").itertuples(index=False):
        x_center = (float(row.x_min) + float(row.x_max)) / (2.0 * width)
        y_center = (float(row.y_min) + float(row.y_max)) / (2.0 * height)
        box_width = (float(row.x_max) - float(row.x_min)) / width
        box_height = (float(row.y_max) - float(row.y_min)) / height
        values = [x_center, y_center, box_width, box_height]
        if not all(0.0 <= value <= 1.0 for value in values) or box_width <= 0 or box_height <= 0:
            raise RuntimeError(f"Invalid normalized bbox for {row.sample_id}: {values}")
        lines.append(f"{int(row.class_id) - 1} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}")
    if not lines:
        raise RuntimeError(f"No boxes for label file: {path}")
    content = "\n".join(lines) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def build_yolo_cache(protocol: Path, out: Path) -> tuple[dict[str, Path], Path, dict[str, Any]]:
    cache = out / "yolo_dataset_cache"
    acquisition = pd.read_csv(protocol / "gc10_acquisition_pool_gt_audit.csv")
    development = pd.read_csv(protocol / "gc10_development_eval.csv")
    acquisition_boxes = pd.read_csv(protocol / "gc10_acquisition_bbox_gt_audit.csv")
    development_boxes = pd.read_csv(protocol / "gc10_development_bbox_gt.csv")
    image_lookup: dict[str, Path] = {}
    modes: dict[str, int] = {}
    for split, manifest, boxes in [
        ("acquisition", acquisition, acquisition_boxes),
        ("development", development, development_boxes),
    ]:
        grouped = {key: frame for key, frame in boxes.groupby("sample_id")}
        for row in manifest.itertuples(index=False):
            sample_id = str(row.sample_id)
            source = Path(str(row.image_path))
            target = cache / "images" / split / f"{sample_id}.jpg"
            mode = link_or_copy(source, target)
            modes[mode] = modes.get(mode, 0) + 1
            write_label(
                cache / "labels" / split / f"{sample_id}.txt",
                grouped.get(sample_id, pd.DataFrame()),
                int(row.width),
                int(row.height),
            )
            image_lookup[sample_id] = target.resolve()
    val_txt = cache / "manifests" / "development.txt"
    val_txt.parent.mkdir(parents=True, exist_ok=True)
    dev_paths = [image_lookup[sample_id].as_posix() for sample_id in development["sample_id"]]
    val_txt.write_text("\n".join(dev_paths) + "\n", encoding="utf-8")
    return image_lookup, val_txt, {"cache_modes": modes, "acquisition_images": len(acquisition), "development_images": len(development)}


def reconstruct_initial(blind: pd.DataFrame, acquisition_seed: int) -> list[str]:
    indices = np.arange(len(blind), dtype=int)
    initial = pd.DataFrame({"idx": indices}).sample(n=INITIAL_SIZE, random_state=acquisition_seed + 999, replace=False)["idx"].astype(int)
    return blind.iloc[initial.tolist()]["sample_id"].astype(str).tolist()


def prepare_training_sets(protocol: Path, selection: Path, out: Path, image_lookup: dict[str, Path], val_txt: Path) -> pd.DataFrame:
    blind = pd.read_csv(protocol / "gc10_acquisition_pool_blind.csv").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    gt = pd.read_csv(protocol / "gc10_acquisition_pool_gt_audit.csv").set_index("sample_id")
    records = pd.read_csv(selection / "gc10_selection_records_posthoc.csv")
    rows = []
    manifests = out / "training_manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for acquisition_seed in ACQUISITION_SEEDS:
        initial_ids = reconstruct_initial(blind, acquisition_seed)
        for strategy in STRATEGIES:
            query = records[(records["acquisition_seed"] == acquisition_seed) & records["strategy"].eq(strategy)].sort_values("rank_in_query")
            query_ids = query["sample_id"].astype(str).tolist()
            if len(query_ids) != QUERY_SIZE:
                raise RuntimeError(f"Expected {QUERY_SIZE} query IDs for seed={acquisition_seed}, strategy={strategy}")
            train_ids = initial_ids + query_ids
            if len(train_ids) != INITIAL_SIZE + QUERY_SIZE or len(set(train_ids)) != len(train_ids):
                raise RuntimeError("Training-set cardinality/overlap failure")
            key = f"acq{acquisition_seed}_{strategy}"
            train_txt = manifests / f"{key}.txt"
            train_txt.write_text("\n".join(image_lookup[sample_id].as_posix() for sample_id in train_ids) + "\n", encoding="utf-8")
            yaml_path = manifests / f"{key}.yaml"
            yaml_path.write_text(yaml.safe_dump({
                "path": str((out / "yolo_dataset_cache").resolve()),
                "train": str(train_txt.resolve()),
                "val": str(val_txt.resolve()),
                "names": {index: name for index, name in enumerate(CLASS_NAMES)},
            }, sort_keys=False), encoding="utf-8")
            class_ids = set()
            total_instances = 0
            for sample_id in train_ids:
                item = gt.loc[sample_id]
                class_ids.update(int(value) for value in str(item["class_ids"]).split("|"))
                total_instances += int(item["num_instances"])
            rows.append({
                "acquisition_seed": acquisition_seed,
                "strategy": strategy,
                "set_key": key,
                "yaml_path": str(yaml_path.resolve()),
                "train_images": len(train_ids),
                "train_instances": total_instances,
                "train_unique_classes": len(class_ids),
                "train_class_ids": "|".join(str(value) for value in sorted(class_ids)),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "frozen_training_sets.csv", index=False, encoding="utf-8-sig")
    return frame


def recover_metrics(metrics: Any) -> tuple[dict[str, float], list[dict[str, Any]]]:
    results = getattr(metrics, "results_dict", {}) or {}
    precision = float(results.get("metrics/precision(B)", getattr(metrics.box, "mp", np.nan)))
    recall = float(results.get("metrics/recall(B)", getattr(metrics.box, "mr", np.nan)))
    aggregate = {
        "precision": precision,
        "recall": recall,
        "map50": float(results.get("metrics/mAP50(B)", getattr(metrics.box, "map50", np.nan))),
        "map5095": float(results.get("metrics/mAP50-95(B)", getattr(metrics.box, "map", np.nan))),
    }
    maps = np.asarray(getattr(metrics.box, "maps", []), dtype=float)
    if maps.shape != (10,):
        raise RuntimeError(f"Expected 10 per-class AP values, got {maps.shape}")
    per_class = [{"class_index": index, "class_id": index + 1, "class_name": CLASS_NAMES[index], "ap5095": float(maps[index])} for index in range(10)]
    return aggregate, per_class


def train_or_recover(item: pd.Series, training_seed: int, out: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from ultralytics import YOLO

    run_name = f"{item['set_key']}_trainseed{training_seed}"
    result_dir = out / "run_results" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = result_dir / "aggregate.json"
    per_class_path = result_dir / "per_class.json"
    if aggregate_path.exists() and per_class_path.exists():
        print(f"[RESULT CACHE] {run_name}", flush=True)
        return json.loads(aggregate_path.read_text(encoding="utf-8")), json.loads(per_class_path.read_text(encoding="utf-8"))
    train_dir = out / "yolo_train_runs" / run_name
    last = train_dir / "weights" / "last.pt"
    started = time.perf_counter()
    if not last.exists():
        print(f"[TRAIN] {run_name}", flush=True)
        model = YOLO(str(args.model))
        model.train(
            data=str(item["yaml_path"]),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            cache=False,
            device=args.device,
            project=str(out / "yolo_train_runs"),
            name=run_name,
            exist_ok=True,
            pretrained=True,
            optimizer="auto",
            verbose=False,
            seed=training_seed,
            deterministic=True,
            plots=False,
            val=False,
            save_json=False,
        )
    if not last.exists():
        raise FileNotFoundError(f"Missing last checkpoint: {last}")
    print(f"[DEV VAL] {run_name}", flush=True)
    metrics = YOLO(str(last)).val(
        data=str(item["yaml_path"]),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        save_json=False,
        verbose=False,
        project=str(out / "development_val_runs"),
        name=run_name,
        exist_ok=True,
    )
    aggregate, per_class = recover_metrics(metrics)
    common = {
        "acquisition_seed": int(item["acquisition_seed"]),
        "strategy": str(item["strategy"]),
        "set_key": str(item["set_key"]),
        "training_seed": training_seed,
        "train_images": int(item["train_images"]),
        "train_instances": int(item["train_instances"]),
        "train_unique_classes": int(item["train_unique_classes"]),
        "train_class_ids": str(item["train_class_ids"]),
        "checkpoint_path": str(last.resolve()),
        "checkpoint_sha256": sha256(last),
        "runtime_seconds": time.perf_counter() - started,
        "final_test_used": False,
    }
    aggregate_row = {**common, **aggregate}
    per_class_rows = [{**common, **row} for row in per_class]
    aggregate_path.write_text(json.dumps(aggregate_row, indent=2), encoding="utf-8")
    per_class_path.write_text(json.dumps(per_class_rows, indent=2), encoding="utf-8")
    return aggregate_row, per_class_rows


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20260715)
    draws = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def analyze(aggregate: pd.DataFrame, per_class: pd.DataFrame, out: Path) -> tuple[bool, Path]:
    paired_rows = []
    acquisition_rows = []
    for metric in ["map5095", "map50", "precision", "recall"]:
        pivot = aggregate.pivot(index=["acquisition_seed", "training_seed"], columns="strategy", values=metric)
        delta = pivot[DINO] - pivot[RANDOM]
        for (acquisition_seed, training_seed), value in delta.items():
            paired_rows.append({"acquisition_seed": acquisition_seed, "training_seed": training_seed, "metric": metric, "difference": value})
        acq = delta.groupby(level="acquisition_seed").mean()
        low, high = bootstrap_ci(acq.to_numpy(float))
        acquisition_rows.append({
            "metric": metric,
            "random_mean": float(aggregate[aggregate["strategy"].eq(RANDOM)][metric].mean()),
            "dino_mean": float(aggregate[aggregate["strategy"].eq(DINO)][metric].mean()),
            "mean_difference": float(acq.mean()),
            "bootstrap_ci95_low_across_acquisition_seeds": low,
            "bootstrap_ci95_high_across_acquisition_seeds": high,
            "acquisition_seed_wins": int((acq > 0).sum()),
            "acquisition_seed_losses": int((acq < 0).sum()),
            "acquisition_seed_ties": int((acq == 0).sum()),
        })
    paired = pd.DataFrame(paired_rows)
    metric_summary = pd.DataFrame(acquisition_rows)

    class_means = per_class.groupby(["strategy", "class_index", "class_id", "class_name"])["ap5095"].mean().reset_index()
    class_pivot = class_means.pivot(index=["class_index", "class_id", "class_name"], columns="strategy", values="ap5095").reset_index()
    class_pivot["difference"] = class_pivot[DINO] - class_pivot[RANDOM]
    per_run_macro = per_class.copy()
    per_run_macro["group"] = per_run_macro["class_index"].map(lambda value: "rare" if int(value) in RARE_CLASS_INDICES else "frequent")
    macro = per_run_macro.groupby(["acquisition_seed", "training_seed", "strategy", "group"])["ap5095"].mean().reset_index()
    macro_pivot = macro.pivot(index=["acquisition_seed", "training_seed", "group"], columns="strategy", values="ap5095").reset_index()
    macro_pivot["difference"] = macro_pivot[DINO] - macro_pivot[RANDOM]
    macro_summary = macro_pivot.groupby("group").agg(
        random_mean=(RANDOM, "mean"), dino_mean=(DINO, "mean"), mean_difference=("difference", "mean")
    ).reset_index()

    map_row = metric_summary.set_index("metric").loc["map5095"]
    recall_row = metric_summary.set_index("metric").loc["recall"]
    rare_diff = float(macro_summary.set_index("group").loc["rare", "mean_difference"])
    frequent_diff = float(macro_summary.set_index("group").loc["frequent", "mean_difference"])
    class_diff = class_pivot.set_index("class_id")["difference"]
    gates = [
        ("mean_map5095_gain_at_least_0p010", float(map_row["mean_difference"]) >= 0.010),
        ("map5095_acquisition_seed_wins_at_least_3_of_5", int(map_row["acquisition_seed_wins"]) >= 3),
        ("map5095_bootstrap_ci_low_across_acquisition_seeds_positive", float(map_row["bootstrap_ci95_low_across_acquisition_seeds"]) > 0.0),
        ("rare_macro_ap5095_gain_at_least_0p020", rare_diff >= 0.020),
        ("class8_ap5095_gain_positive", float(class_diff.loc[8]) > 0.0),
        ("class9_ap5095_gain_positive", float(class_diff.loc[9]) > 0.0),
        ("recall_noninferiority_minus_0p020", float(recall_row["mean_difference"]) >= -0.020),
        ("frequent_macro_ap5095_noninferiority_minus_0p020", frequent_diff >= -0.020),
    ]
    gate = pd.DataFrame([{"check": name, "passed": passed} for name, passed in gates])
    overall = all(passed for _, passed in gates)
    aggregate.to_csv(out / "detector_run_metrics.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(out / "detector_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(out / "detector_training_seed_paired_differences.csv", index=False, encoding="utf-8-sig")
    metric_summary.to_csv(out / "detector_acquisition_seed_summary.csv", index=False, encoding="utf-8-sig")
    class_pivot.to_csv(out / "detector_per_class_mean_comparison.csv", index=False, encoding="utf-8-sig")
    macro_summary.to_csv(out / "detector_class_group_macro_summary.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(out / "detector_development_confirmation_gate.csv", index=False, encoding="utf-8-sig")
    report = [
        "# GC10-DET Development-Only Detector Confirmation", "",
        "- Final test used: **False**",
        "- Detector models trained: **30**",
        "- Acquisition/training seeds: **5/3**",
        "- Labeled images per model: **40**",
        f"- Development confirmation gate: **{'PASS' if overall else 'FAIL'}**", "",
        "## Aggregate detector metrics", "", metric_summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## Class-group AP50-95", "", macro_summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## Per-class AP50-95", "", class_pivot.to_markdown(index=False, floatfmt=".6f"), "",
        "## Pre-registered gate", "", gate.assign(result=gate["passed"].map({True: "PASS", False: "FAIL"})).drop(columns="passed").to_markdown(index=False), "",
        "The final test remains locked regardless of this development result.",
    ]
    summary_path = out / "gc10_detector_confirmation_summary.md"
    summary_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return overall, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    protocol = args.protocol_dir.expanduser().resolve()
    selection = args.selection_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol_config = json.loads((protocol / "gc10_protocol_config.json").read_text(encoding="utf-8"))
    selection_config = json.loads((selection / "config.json").read_text(encoding="utf-8"))
    if bool(protocol_config.get("final_test_evaluated", True)) or bool(selection_config.get("final_test_used", True)):
        raise RuntimeError("Final-test safety flag failed")
    if not bool(selection_config.get("overall_gate_pass", False)):
        raise RuntimeError("Selection-only gate did not authorize detector training")
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    image_lookup, val_txt, cache_info = build_yolo_cache(protocol, out)
    training_sets = prepare_training_sets(protocol, selection, out, image_lookup, val_txt)
    run_config = {
        "acquisition_seeds": ACQUISITION_SEEDS,
        "training_seeds": TRAINING_SEEDS,
        "strategies": STRATEGIES,
        "initial_size": INITIAL_SIZE,
        "query_size": QUERY_SIZE,
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "training_validation_enabled": False,
        "evaluation_split": "development",
        "final_test_used": False,
        **cache_info,
    }
    (out / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    aggregate_rows = []
    per_class_rows = []
    for item in training_sets.to_dict("records"):
        series = pd.Series(item)
        for training_seed in TRAINING_SEEDS:
            aggregate, per_class = train_or_recover(series, training_seed, out, args)
            aggregate_rows.append(aggregate)
            per_class_rows.extend(per_class)
    aggregate_frame = pd.DataFrame(aggregate_rows)
    per_class_frame = pd.DataFrame(per_class_rows)
    if len(aggregate_frame) != 30 or len(per_class_frame) != 300:
        raise RuntimeError(f"Incomplete result cardinality: {len(aggregate_frame)}/{len(per_class_frame)}")
    overall, summary = analyze(aggregate_frame, per_class_frame, out)
    print("=" * 100)
    print("[DONE] GC10 development-only detector confirmation")
    print(f"[GATE] {'PASS' if overall else 'FAIL'}")
    print(f"[SUMMARY] {summary}")
    print("Final test used: False")
    print("=" * 100)


if __name__ == "__main__":
    main()

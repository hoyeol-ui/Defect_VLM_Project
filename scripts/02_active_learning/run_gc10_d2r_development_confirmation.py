"""Conditionally train the 15-model GC10 D2R development confirmation.

The frozen 15 Random and 15 pure-DINO results are reused only after exact
selection/configuration checks. New training is restricted to 5 acquisition
seeds x 3 training seeds for D2R. The final split is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE_RUNNER_PATH = ROOT / "scripts" / "02_active_learning" / "run_gc10_development_detector_confirmation.py"
DEFAULT_PROTOCOL = ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715"
DEFAULT_SELECTION = ROOT / "runs" / "gc10_discovery_representation_audit" / "gc10_d2r_200seed_20260715"
DEFAULT_FROZEN_SELECTION = ROOT / "runs" / "gc10_taxonomy_selection_audit" / "gc10_random_vs_dino_200seed_20260715"
DEFAULT_BASELINE_DETECTOR = ROOT / "runs" / "gc10_detector_confirmation" / "gc10_dev_confirm_5acq_3train_20260715"
DEFAULT_OUT = ROOT / "runs" / "gc10_detector_confirmation" / "gc10_d2r_dev_confirm_5acq_3train_20260715"
DEFAULT_MODEL = ROOT / "yolov8n.pt"

RANDOM = "GTFreeRandom"
DINO = "FrozenDINOVisualDiversity"
D2R = "D2RDiscoveryRepresentationGuard"
STRATEGIES = [RANDOM, DINO, D2R]
ACQUISITION_SEEDS = [0, 1, 2, 3, 4]
TRAINING_SEEDS = [42, 43, 44]
INITIAL_SIZE = 20
QUERY_SIZE = 20
CLASS_NAMES = [
    "chongkong", "hanfeng", "yueyawan", "shuiban", "youban",
    "siban", "yiwu", "yahen", "zhehen", "yaozhe",
]
RARE_CLASS_INDICES = [7, 8, 9]


def load_base_runner() -> Any:
    spec = importlib.util.spec_from_file_location("gc10_frozen_base_runner", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen base runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_runner()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_set_hash(sample_ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(sample_ids) + "\n").encode("utf-8")).hexdigest()


def require_selection_pass(selection: Path) -> dict[str, Any]:
    config_path = selection / "config.json"
    records_path = selection / "gc10_d2r_selection_records_posthoc.csv"
    gate_path = selection / "gc10_d2r_selection_gate.csv"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gate = pd.read_csv(gate_path)
    if bool(config.get("final_test_used", True)):
        raise RuntimeError("D2R selection final-test safety flag failed")
    if bool(config.get("selector_uses_unselected_gt", True)):
        raise RuntimeError("D2R selector reports unselected-pool GT use")
    if int(config.get("feedback_access_violations", -1)) != 0:
        raise RuntimeError("D2R feedback-access audit failed")
    if sha256(records_path) != config.get("selection_records_sha256"):
        raise RuntimeError("D2R selection-record hash mismatch")
    if not bool(config.get("overall_gate_pass", False)) or not gate["passed"].astype(str).str.lower().eq("true").all():
        raise RuntimeError("D2R selection-only gate did not authorize detector training")
    return config


def reconstruct_initial(blind: pd.DataFrame, acquisition_seed: int) -> list[str]:
    indices = np.arange(len(blind), dtype=int)
    selected = (
        pd.DataFrame({"idx": indices})
        .sample(n=INITIAL_SIZE, random_state=acquisition_seed + 999, replace=False)["idx"]
        .astype(int).tolist()
    )
    return blind.iloc[selected]["sample_id"].astype(str).tolist()


def validate_baseline_reuse(
    args: argparse.Namespace, protocol: Path, selection: Path, frozen_selection: Path, baseline_detector: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol_config = json.loads((protocol / "gc10_protocol_config.json").read_text(encoding="utf-8"))
    baseline_config = json.loads((baseline_detector / "config.json").read_text(encoding="utf-8"))
    if bool(protocol_config.get("final_test_evaluated", True)) or bool(baseline_config.get("final_test_used", True)):
        raise RuntimeError("Protocol/baseline final-test safety flag failed")
    expected = {
        "acquisition_seeds": ACQUISITION_SEEDS,
        "training_seeds": TRAINING_SEEDS,
        "initial_size": INITIAL_SIZE,
        "query_size": QUERY_SIZE,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
    }
    for key, value in expected.items():
        if baseline_config.get(key) != value:
            raise RuntimeError(f"Frozen baseline configuration mismatch for {key}: {baseline_config.get(key)} != {value}")
    if baseline_config.get("strategies") != [RANDOM, DINO]:
        raise RuntimeError("Unexpected frozen detector strategies")
    if baseline_config.get("model_sha256") != sha256(args.model):
        raise RuntimeError("Model hash differs from the frozen 30-model baseline")

    new_records = pd.read_csv(selection / "gc10_d2r_selection_records_posthoc.csv")
    frozen_records = pd.read_csv(frozen_selection / "gc10_selection_records_posthoc.csv")
    for seed in ACQUISITION_SEEDS:
        for strategy in [RANDOM, DINO]:
            new_ids = new_records[
                (new_records["acquisition_seed"] == seed) & new_records["strategy"].eq(strategy)
            ].sort_values("rank_in_query")["sample_id"].astype(str).tolist()
            frozen_ids = frozen_records[
                (frozen_records["acquisition_seed"] == seed) & frozen_records["strategy"].eq(strategy)
            ].sort_values("rank_in_query")["sample_id"].astype(str).tolist()
            if new_ids != frozen_ids or len(new_ids) != QUERY_SIZE:
                raise RuntimeError(f"Baseline selection replay mismatch: seed={seed}, strategy={strategy}")

    aggregate = pd.read_csv(baseline_detector / "detector_run_metrics.csv")
    per_class = pd.read_csv(baseline_detector / "detector_per_class_metrics.csv")
    if len(aggregate) != 30 or len(per_class) != 300:
        raise RuntimeError("Frozen baseline result cardinality mismatch")
    if aggregate["final_test_used"].astype(str).str.lower().ne("false").any():
        raise RuntimeError("Frozen aggregate results report final-test use")
    if per_class["final_test_used"].astype(str).str.lower().ne("false").any():
        raise RuntimeError("Frozen per-class results report final-test use")
    expected_pairs = {(seed, strategy, train_seed) for seed in ACQUISITION_SEEDS for strategy in [RANDOM, DINO] for train_seed in TRAINING_SEEDS}
    actual_pairs = set(zip(aggregate["acquisition_seed"], aggregate["strategy"], aggregate["training_seed"]))
    if actual_pairs != expected_pairs:
        raise RuntimeError("Frozen baseline seed/strategy grid mismatch")
    return aggregate, per_class


def baseline_cache_lookup(protocol: Path, baseline_detector: Path) -> tuple[dict[str, Path], Path]:
    cache = baseline_detector / "yolo_dataset_cache"
    acquisition = pd.read_csv(protocol / "gc10_acquisition_pool_gt_audit.csv")
    development = pd.read_csv(protocol / "gc10_development_eval.csv")
    image_lookup: dict[str, Path] = {}
    for split, frame in [("acquisition", acquisition), ("development", development)]:
        for sample_id in frame["sample_id"].astype(str):
            image = cache / "images" / split / f"{sample_id}.jpg"
            label = cache / "labels" / split / f"{sample_id}.txt"
            if not image.exists() or not label.exists():
                raise FileNotFoundError(f"Frozen YOLO cache incomplete: {image} / {label}")
            image_lookup[sample_id] = image.resolve()
    val_txt = cache / "manifests" / "development.txt"
    if not val_txt.exists() or len(val_txt.read_text(encoding="utf-8").splitlines()) != len(development):
        raise RuntimeError("Frozen development manifest is missing or incomplete")
    return image_lookup, val_txt.resolve()


def prepare_d2r_training_sets(
    protocol: Path, selection: Path, out: Path, baseline_detector: Path,
) -> pd.DataFrame:
    blind = pd.read_csv(protocol / "gc10_acquisition_pool_blind.csv").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    gt = pd.read_csv(protocol / "gc10_acquisition_pool_gt_audit.csv").set_index("sample_id")
    records = pd.read_csv(selection / "gc10_d2r_selection_records_posthoc.csv")
    image_lookup, val_txt = baseline_cache_lookup(protocol, baseline_detector)
    manifests = out / "training_manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    rows = []
    for acquisition_seed in ACQUISITION_SEEDS:
        initial_ids = reconstruct_initial(blind, acquisition_seed)
        query_ids = records[
            (records["acquisition_seed"] == acquisition_seed) & records["strategy"].eq(D2R)
        ].sort_values("rank_in_query")["sample_id"].astype(str).tolist()
        train_ids = initial_ids + query_ids
        if len(query_ids) != QUERY_SIZE or len(train_ids) != INITIAL_SIZE + QUERY_SIZE or len(set(train_ids)) != len(train_ids):
            raise RuntimeError(f"D2R training-set cardinality failure for acquisition seed {acquisition_seed}")
        key = f"acq{acquisition_seed}_{D2R}"
        train_txt = manifests / f"{key}.txt"
        train_txt.write_text("\n".join(image_lookup[sample_id].as_posix() for sample_id in train_ids) + "\n", encoding="utf-8")
        yaml_path = manifests / f"{key}.yaml"
        yaml_path.write_text(yaml.safe_dump({
            "path": str((baseline_detector / "yolo_dataset_cache").resolve()),
            "train": str(train_txt.resolve()),
            "val": str(val_txt),
            "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        }, sort_keys=False), encoding="utf-8")
        classes: set[int] = set()
        instances = 0
        for sample_id in train_ids:
            item = gt.loc[sample_id]
            classes.update(int(value) for value in str(item["class_ids"]).split("|"))
            instances += int(item["num_instances"])
        rows.append({
            "acquisition_seed": acquisition_seed,
            "strategy": D2R,
            "set_key": key,
            "yaml_path": str(yaml_path.resolve()),
            "train_images": len(train_ids),
            "train_instances": instances,
            "train_unique_classes": len(classes),
            "train_class_ids": "|".join(str(value) for value in sorted(classes)),
            "train_sample_id_sha256": sample_set_hash(train_ids),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "frozen_d2r_training_sets.csv", index=False, encoding="utf-8-sig")
    return frame


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20260715)
    draws = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def paired_summaries(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparator in [RANDOM, DINO]:
        for metric in ["map5095", "map50", "precision", "recall"]:
            pivot = aggregate.pivot(index=["acquisition_seed", "training_seed"], columns="strategy", values=metric)
            paired_delta = pivot[D2R] - pivot[comparator]
            acquisition_delta = paired_delta.groupby(level="acquisition_seed").mean().loc[ACQUISITION_SEEDS]
            low, high = bootstrap_ci(acquisition_delta.to_numpy(float))
            rows.append({
                "comparator": comparator,
                "metric": metric,
                "comparator_mean": float(aggregate[aggregate["strategy"].eq(comparator)][metric].mean()),
                "d2r_mean": float(aggregate[aggregate["strategy"].eq(D2R)][metric].mean()),
                "mean_difference_across_acquisition_seeds": float(acquisition_delta.mean()),
                "bootstrap_ci95_low_across_acquisition_seeds": low,
                "bootstrap_ci95_high_across_acquisition_seeds": high,
                "acquisition_seed_wins": int((acquisition_delta > 0).sum()),
                "acquisition_seed_losses": int((acquisition_delta < 0).sum()),
                "acquisition_seed_ties": int((acquisition_delta == 0).sum()),
            })
    return pd.DataFrame(rows)


def analyze(aggregate: pd.DataFrame, per_class: pd.DataFrame, out: Path) -> tuple[bool, Path]:
    paired = paired_summaries(aggregate)
    class_means = per_class.groupby(["strategy", "class_index", "class_id", "class_name"])["ap5095"].mean().reset_index()
    class_pivot = class_means.pivot(index=["class_index", "class_id", "class_name"], columns="strategy", values="ap5095").reset_index()
    class_pivot["d2r_minus_random"] = class_pivot[D2R] - class_pivot[RANDOM]
    class_pivot["d2r_minus_dino"] = class_pivot[D2R] - class_pivot[DINO]

    grouped = per_class.copy()
    grouped["class_group"] = grouped["class_index"].map(lambda value: "rare" if int(value) in RARE_CLASS_INDICES else "frequent")
    run_macro = grouped.groupby(["acquisition_seed", "training_seed", "strategy", "class_group"])["ap5095"].mean().reset_index()
    macro_summary = run_macro.groupby(["strategy", "class_group"])["ap5095"].mean().reset_index()
    macro_pivot = macro_summary.pivot(index="class_group", columns="strategy", values="ap5095").reset_index()
    macro_pivot["d2r_minus_random"] = macro_pivot[D2R] - macro_pivot[RANDOM]
    macro_pivot["d2r_minus_dino"] = macro_pivot[D2R] - macro_pivot[DINO]

    paired_index = paired.set_index(["comparator", "metric"])
    macro_index = macro_pivot.set_index("class_group")
    map_vs_random = paired_index.loc[(RANDOM, "map5095")]
    map_vs_dino = paired_index.loc[(DINO, "map5095")]
    recall_vs_random = paired_index.loc[(RANDOM, "recall")]
    rare_vs_random = float(macro_index.loc["rare", "d2r_minus_random"])
    rare_vs_dino = float(macro_index.loc["rare", "d2r_minus_dino"])
    frequent_vs_random = float(macro_index.loc["frequent", "d2r_minus_random"])
    minimum_class_difference = float(class_pivot["d2r_minus_random"].min())
    gate_specs = [
        ("map5095_gain_vs_random", float(map_vs_random["mean_difference_across_acquisition_seeds"]), ">= 0.010", float(map_vs_random["mean_difference_across_acquisition_seeds"]) >= 0.010),
        ("map5095_ci_low_vs_random", float(map_vs_random["bootstrap_ci95_low_across_acquisition_seeds"]), "> 0", float(map_vs_random["bootstrap_ci95_low_across_acquisition_seeds"]) > 0.0),
        ("map5095_acquisition_seed_wins_vs_random", int(map_vs_random["acquisition_seed_wins"]), ">= 4/5", int(map_vs_random["acquisition_seed_wins"]) >= 4),
        ("recall_gain_vs_random", float(recall_vs_random["mean_difference_across_acquisition_seeds"]), ">= 0.020", float(recall_vs_random["mean_difference_across_acquisition_seeds"]) >= 0.020),
        ("rare_macro_ap_gain_vs_random", rare_vs_random, ">= 0", rare_vs_random >= 0.0),
        ("rare_macro_ap_gain_vs_dino", rare_vs_dino, ">= 0.015", rare_vs_dino >= 0.015),
        ("frequent_macro_ap_noninferiority_vs_random", frequent_vs_random, ">= -0.020", frequent_vs_random >= -0.020),
        ("map5095_noninferiority_vs_dino", float(map_vs_dino["mean_difference_across_acquisition_seeds"]), ">= -0.005", float(map_vs_dino["mean_difference_across_acquisition_seeds"]) >= -0.005),
        ("minimum_per_class_ap_difference_vs_random", minimum_class_difference, ">= -0.030", minimum_class_difference >= -0.030),
    ]
    gate = pd.DataFrame([
        {"check": name, "observed": observed, "threshold": threshold, "passed": passed}
        for name, observed, threshold, passed in gate_specs
    ])
    overall = bool(gate["passed"].all())

    aggregate.to_csv(out / "detector_all_strategy_run_metrics.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(out / "detector_all_strategy_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(out / "detector_d2r_paired_summary.csv", index=False, encoding="utf-8-sig")
    class_pivot.to_csv(out / "detector_d2r_per_class_mean_comparison.csv", index=False, encoding="utf-8-sig")
    macro_pivot.to_csv(out / "detector_d2r_macro_group_summary.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(out / "detector_d2r_development_gate.csv", index=False, encoding="utf-8-sig")

    report = [
        "# GC10 D2R Development-Only Detector Confirmation", "",
        "- Newly trained detector models: **15 D2R models**",
        "- Reused frozen baseline models/results: **30 (15 Random + 15 pure DINO)**",
        "- Acquisition/training seeds: **5/3**",
        "- Labeled images per model: **40**",
        "- Evaluation split: **development only**",
        "- Final test used: **False**",
        f"- D2R development gate: **{'PASS' if overall else 'FAIL'}**", "",
        "## Aggregate paired comparisons", "", paired.to_markdown(index=False, floatfmt=".6f"), "",
        "## Class-group AP50-95", "", macro_pivot.to_markdown(index=False, floatfmt=".6f"), "",
        "## Per-class AP50-95", "", class_pivot.to_markdown(index=False, floatfmt=".6f"), "",
        "## Frozen detector gate", "", gate.assign(result=gate["passed"].map({True: "PASS", False: "FAIL"})).drop(columns="passed").to_markdown(index=False, floatfmt=".6f"), "",
        "This remains GC10 development evidence. The final test remains locked regardless of the gate result.",
    ]
    summary_path = out / "gc10_d2r_detector_confirmation_summary.md"
    summary_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return overall, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--frozen-selection-dir", type=Path, default=DEFAULT_FROZEN_SELECTION)
    parser.add_argument("--baseline-detector-dir", type=Path, default=DEFAULT_BASELINE_DETECTOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-development-only", action="store_true")
    args = parser.parse_args()

    protocol = args.protocol_dir.expanduser().resolve()
    selection = args.selection_dir.expanduser().resolve()
    frozen_selection = args.frozen_selection_dir.expanduser().resolve()
    baseline_detector = args.baseline_detector_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    require_selection_pass(selection)
    baseline_aggregate, baseline_per_class = validate_baseline_reuse(
        args, protocol, selection, frozen_selection, baseline_detector
    )
    out.mkdir(parents=True, exist_ok=True)
    training_sets = prepare_d2r_training_sets(protocol, selection, out, baseline_detector)
    run_config = {
        "acquisition_seeds": ACQUISITION_SEEDS,
        "training_seeds": TRAINING_SEEDS,
        "strategies": STRATEGIES,
        "new_training_strategy": D2R,
        "new_model_count": 15,
        "reused_baseline_result_count": 30,
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
        "development_only": True,
        "final_test_used": False,
    }
    (out / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    if args.dry_run:
        print("[DRY-RUN PASS] selection gate, feedback audit, hashes, baseline replay, YOLO configuration, and 5 D2R manifests")
        print("New detector training performed: False")
        print("Final test used: False")
        return
    if not args.confirm_development_only:
        raise RuntimeError("Full training requires explicit --confirm-development-only")

    d2r_aggregate_rows = []
    d2r_per_class_rows = []
    for item in training_sets.to_dict("records"):
        series = pd.Series(item)
        for training_seed in TRAINING_SEEDS:
            aggregate, per_class = BASE.train_or_recover(series, training_seed, out, args)
            d2r_aggregate_rows.append(aggregate)
            d2r_per_class_rows.extend(per_class)
    d2r_aggregate = pd.DataFrame(d2r_aggregate_rows)
    d2r_per_class = pd.DataFrame(d2r_per_class_rows)
    if len(d2r_aggregate) != 15 or len(d2r_per_class) != 150:
        raise RuntimeError(f"Incomplete D2R result cardinality: {len(d2r_aggregate)}/{len(d2r_per_class)}")
    aggregate = pd.concat([baseline_aggregate, d2r_aggregate], ignore_index=True, sort=False)
    per_class = pd.concat([baseline_per_class, d2r_per_class], ignore_index=True, sort=False)
    overall, summary = analyze(aggregate, per_class, out)
    print("=" * 100)
    print("[DONE] GC10 D2R development-only detector confirmation")
    print(f"[GATE] {'PASS' if overall else 'FAIL'}")
    print(f"[SUMMARY] {summary}")
    print("Final test used: False")
    print("=" * 100)


if __name__ == "__main__":
    main()

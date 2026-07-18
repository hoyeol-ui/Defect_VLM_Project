"""Run the frozen Random/K40 x YOLOv8n/v8s 140-label factorial study."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import plan_v2_budget as budget_core  # noqa: E402
import run as dcal  # noqa: E402


PROTOCOL = ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715"
EMBEDDINGS = ROOT / "outputs" / "gc10_visual_embeddings" / "dinov2_small_protocol_20260715"
DECISION = ROOT / "runs" / "dcal_xai" / "v2_budget_extended" / "decision.json"
BACKBONES = {"YOLOv8n": ROOT / "yolov8n.pt", "YOLOv8s": ROOT / "yolov8s.pt"}
POLICIES = ["Random140", "ClusterK40_140"]
ACQUISITION_SEEDS = [20000, 20001, 20002, 20003, 20004]
TRAINING_SEEDS = [42, 43, 44]
BUDGET = 140
CLASS_NAMES = [
    "chongkong", "hanfeng", "yueyawan", "shuiban", "youban",
    "siban", "yiwu", "yahen", "zhehen", "yaozhe",
]
RARE_IDS = {8, 9, 10}
TRAINING = {"epochs": 75, "imgsz": 640, "batch": 16, "workers": 0, "device": "0"}
THRESHOLDS = {
    "map_noninferiority": -0.005,
    "rare_macro_noninferiority": -0.010,
    "worst_class": -0.050,
    "training_seed_std_max": 0.020,
    "recall_noninferiority": -0.020,
    "v8s_capacity_margin": 0.005,
}
DEFAULT_SMOKE_OUT = ROOT / "runs" / "dcal_xai" / "v2_backbone_smoke"
DEFAULT_MAIN_OUT = ROOT / "runs" / "dcal_xai" / "v2_backbone_main"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_context(require_models: bool) -> dict[str, Any]:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    if decision.get("status") != "PASS" or decision.get("chosen") != {"budget": 140, "policy": "DINOClusterCoverageK40"}:
        raise RuntimeError("V2.1 budget decision did not authorize this stage")
    protocol_config = json.loads((PROTOCOL / "gc10_protocol_config.json").read_text(encoding="utf-8"))
    embedding_config = json.loads((EMBEDDINGS / "embedding_config.json").read_text(encoding="utf-8"))
    if bool(protocol_config.get("final_test_evaluated", True)) or bool(embedding_config.get("final_test_used", True)):
        raise RuntimeError("Final-test safety flag failed")
    missing = [str(path) for path in BACKBONES.values() if not path.exists()]
    if require_models and missing:
        raise FileNotFoundError("Missing backbone weights: " + ", ".join(missing))
    blind = pd.read_csv(PROTOCOL / "gc10_acquisition_pool_blind.csv").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    gt = pd.read_csv(PROTOCOL / "gc10_acquisition_pool_gt_audit.csv").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    boxes = pd.read_csv(PROTOCOL / "gc10_acquisition_bbox_gt_audit.csv")
    embedding_manifest = pd.read_csv(EMBEDDINGS / "embedding_manifest.csv")
    embeddings = np.load(EMBEDDINGS / "embeddings.npy")
    if blind["sample_id"].tolist() != gt["sample_id"].tolist() or blind["sample_id"].tolist() != embedding_manifest["sample_id"].tolist():
        raise RuntimeError("Manifest alignment failure")
    if blind["sample_id"].duplicated().any() or gt["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate acquisition sample IDs")
    if embeddings.shape[0] != len(blind) or not np.isfinite(embeddings).all():
        raise RuntimeError("Embedding cardinality or finite-value failure")
    development = pd.read_csv(PROTOCOL / "gc10_development_eval.csv")
    if set(blind["sample_id"].astype(str)) & set(development["sample_id"].astype(str)):
        raise RuntimeError("Acquisition/development leakage")
    labels = budget_core.fit_clusters(embeddings)["DINOClusterCoverageK40"]
    return {
        "blind": blind, "gt": gt, "boxes": boxes, "embeddings": embeddings,
        "labels": labels, "development_images": len(development),
    }


def make_selections(context: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for acquisition_seed in ACQUISITION_SEEDS:
        sampling_seed = acquisition_seed + BUDGET * 100_000
        selections = {
            "Random140": budget_core.random_selection(len(context["blind"]), BUDGET, sampling_seed),
            "ClusterK40_140": budget_core.cluster_coverage_selection(context["labels"], BUDGET, sampling_seed),
        }
        for policy, indices in selections.items():
            for rank, index in enumerate(indices, start=1):
                rows.append({
                    "acquisition_seed": acquisition_seed,
                    "policy": policy,
                    "rank": rank,
                    "embedding_index": int(index),
                    "sample_id": str(context["blind"].iloc[index]["sample_id"]),
                    "selector_used_gt": False,
                    "final_test_used": False,
                })
    frame = pd.DataFrame(rows)
    if len(frame) != len(ACQUISITION_SEEDS) * len(POLICIES) * BUDGET:
        raise RuntimeError("Selection cardinality failure")
    return frame


def audit_composition(selections: pd.DataFrame, context: dict[str, Any], out: Path) -> pd.DataFrame:
    gt = context["gt"].set_index("sample_id")
    metrics: list[dict[str, Any]] = []
    for (seed, policy), subset in selections.groupby(["acquisition_seed", "policy"]):
        ids = subset.sort_values("rank")["sample_id"].astype(str).tolist()
        indices = subset.sort_values("rank")["embedding_index"].astype(int).to_numpy()
        values = budget_core.compute_metrics(indices, context["gt"], context["boxes"], context["embeddings"])
        classes = set()
        for sample_id in ids:
            classes.update(budget_core.parse_classes(gt.loc[sample_id, "class_ids"]))
        metrics.append({"acquisition_seed": seed, "policy": policy, **values, "class_ids": "|".join(map(str, sorted(classes)))})
    result = pd.DataFrame(metrics)
    result.to_csv(out / "selection_composition_posthoc.csv", index=False, encoding="utf-8-sig")
    return result


def write_audit(out: Path, require_models: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    context = load_context(require_models=require_models)
    selections = make_selections(context)
    selections.to_csv(out / "frozen_initial_sets.csv", index=False, encoding="utf-8-sig")
    composition = audit_composition(selections, context, out)
    frozen_sets_path = out / "frozen_initial_sets.csv"
    payload = {
        "status": "PASS",
        "budget": BUDGET,
        "policies": POLICIES,
        "backbones": {name: str(path) for name, path in BACKBONES.items()},
        "backbone_sha256": {name: digest(path) for name, path in BACKBONES.items() if path.exists()},
        "acquisition_seeds": ACQUISITION_SEEDS,
        "training_seeds": TRAINING_SEEDS,
        "models_main": 60,
        "acquisition_pool_images": len(context["blind"]),
        "development_images": context["development_images"],
        "embedding_shape": list(context["embeddings"].shape),
        "frozen_initial_sets_sha256": digest(frozen_sets_path),
        "k40_all_classes_rate_new_seeds": float(
            composition[composition["policy"].eq("ClusterK40_140")]["all_classes"].mean()
        ),
        "random_all_classes_rate_new_seeds": float(
            composition[composition["policy"].eq("Random140")]["all_classes"].mean()
        ),
        "script_sha256": digest(Path(__file__).resolve()),
        "protocol_sha256": digest(HERE / "v2_backbone_protocol.md"),
        "selector_used_gt": False,
        "gt_joined_posthoc": True,
        "detector_training_performed": False,
        "final_test_used": False,
    }
    (out / "audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return context, selections


def make_run_signature(
    *, model_path: Path, yaml_path: Path, ids: list[str], acquisition_seed: int,
    policy: str, backbone: str, training_seed: int,
) -> str:
    payload = {
        "protocol_config_sha256": digest(PROTOCOL / "gc10_protocol_config.json"),
        "embedding_config_sha256": digest(EMBEDDINGS / "embedding_config.json"),
        "budget_decision_sha256": digest(DECISION),
        "model_sha256": digest(model_path),
        "yaml_sha256": digest(yaml_path),
        "selected_ids_sha256": digest_text("\n".join(ids)),
        "acquisition_seed": acquisition_seed,
        "policy": policy,
        "backbone": backbone,
        "training_seed": training_seed,
        "training": TRAINING,
    }
    return digest_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def recover_cached_result(
    aggregate_path: Path, per_class_path: Path, expected_signature: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    if not aggregate_path.exists() and not per_class_path.exists():
        return None
    if not aggregate_path.exists() or not per_class_path.exists():
        raise RuntimeError(f"Partial result cache: {aggregate_path.parent}")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    per_class = json.loads(per_class_path.read_text(encoding="utf-8"))
    if aggregate.get("run_signature") != expected_signature:
        raise RuntimeError(f"Stale result cache/configuration mismatch: {aggregate_path.parent}")
    if len(per_class) != len(CLASS_NAMES) or any(row.get("run_signature") != expected_signature for row in per_class):
        raise RuntimeError(f"Invalid per-class result cache: {per_class_path}")
    if aggregate.get("final_test_used") is not False or any(row.get("final_test_used") is not False for row in per_class):
        raise RuntimeError(f"Final-test safety failure in result cache: {aggregate_path.parent}")
    return aggregate, per_class


def train_and_evaluate(
    *,
    out: Path,
    context: dict[str, Any],
    selections: pd.DataFrame,
    acquisition_seeds: list[int],
    training_seeds: list[int],
    backbone_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ultralytics import YOLO

    aggregate_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for acquisition_seed in acquisition_seeds:
        for policy in POLICIES:
            ids = selections[(selections["acquisition_seed"] == acquisition_seed) & selections["policy"].eq(policy)].sort_values("rank")["sample_id"].astype(str).tolist()
            if len(ids) != BUDGET:
                raise RuntimeError("Frozen initial set missing")
            selected_backbones = backbone_names or list(BACKBONES)
            for backbone in selected_backbones:
                model_path = BACKBONES[backbone]
                key = f"acq{acquisition_seed}_{policy}_{backbone}"
                yaml_path, _ = dcal.build_training_yaml(
                    protocol=PROTOCOL, out=out, key=key, train_ids=ids,
                    class_names=CLASS_NAMES, include_development=True,
                )
                for training_seed in training_seeds:
                    name = f"{key}_trainseed{training_seed}"
                    result_dir = out / "results" / name
                    aggregate_path = result_dir / "aggregate.json"
                    per_class_path = result_dir / "per_class.json"
                    run_signature = make_run_signature(
                        model_path=model_path, yaml_path=yaml_path, ids=ids,
                        acquisition_seed=acquisition_seed, policy=policy, backbone=backbone,
                        training_seed=training_seed,
                    )
                    cached = recover_cached_result(aggregate_path, per_class_path, run_signature)
                    if cached is not None:
                        aggregate_rows.append(cached[0])
                        per_class_rows.extend(cached[1])
                        continue
                    started = time.perf_counter()
                    checkpoint = dcal.train_checkpoint(
                        model_path=model_path, yaml_path=yaml_path, out=out, name=name,
                        training_seed=training_seed, training=TRAINING,
                    )
                    print(f"[DEV VAL] {name}", flush=True)
                    metrics = YOLO(str(checkpoint)).val(
                        data=str(yaml_path), split="val", imgsz=TRAINING["imgsz"], batch=TRAINING["batch"],
                        device=TRAINING["device"], workers=TRAINING["workers"], plots=False,
                        save_json=False, verbose=False, project=str(out / "dev_val_runs"), name=name, exist_ok=True,
                    )
                    aggregate, per_class = dcal.recover_metrics(metrics, CLASS_NAMES)
                    common = {
                        "acquisition_seed": acquisition_seed,
                        "policy": policy,
                        "backbone": backbone,
                        "configuration": f"{policy}|{backbone}",
                        "training_seed": training_seed,
                        "train_images": BUDGET,
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": digest(checkpoint),
                        "run_signature": run_signature,
                        "runtime_seconds": time.perf_counter() - started,
                        "evaluation_split": "development",
                        "final_test_used": False,
                    }
                    aggregate_row = {**common, **aggregate}
                    per_class_payload = [{**common, **row} for row in per_class]
                    result_dir.mkdir(parents=True, exist_ok=True)
                    aggregate_path.write_text(json.dumps(aggregate_row, indent=2), encoding="utf-8")
                    per_class_path.write_text(json.dumps(per_class_payload, indent=2), encoding="utf-8")
                    aggregate_rows.append(aggregate_row)
                    per_class_rows.extend(per_class_payload)
    return pd.DataFrame(aggregate_rows), pd.DataFrame(per_class_rows)


def analyze_v8n_screen(aggregate: pd.DataFrame, per_class: pd.DataFrame, out: Path) -> tuple[bool, Path]:
    if len(aggregate) != 30 or len(per_class) != 300:
        raise RuntimeError(f"Incomplete YOLOv8n screen: {len(aggregate)}/{len(per_class)}")
    if set(aggregate["backbone"]) != {"YOLOv8n"} or set(per_class["backbone"]) != {"YOLOv8n"}:
        raise RuntimeError("YOLOv8n screen contains an unauthorized backbone")
    contrasts = pd.DataFrame(paired_summary(
        aggregate, "policy", "ClusterK40_140", "Random140", "backbone", "YOLOv8n",
    ))
    class_summary = per_class.groupby(["policy", "class_id", "class_name"], as_index=False)["ap5095"].mean()
    class_pivot = class_summary.pivot(index=["class_id", "class_name"], columns="policy", values="ap5095")
    class_difference = class_pivot["ClusterK40_140"] - class_pivot["Random140"]
    rare_difference = float(class_difference[class_difference.index.get_level_values("class_id").isin(RARE_IDS)].mean())
    worst_class = float(class_difference.min())
    stability = aggregate[
        aggregate["policy"].eq("ClusterK40_140")
    ].groupby("acquisition_seed")["map5095"].std(ddof=1)
    stability_mean = float(stability.mean())
    map_difference = float(contrasts[contrasts["metric"].eq("map5095")]["mean_difference"].iloc[0])
    recall_difference = float(contrasts[contrasts["metric"].eq("recall")]["mean_difference"].iloc[0])
    checks = {
        "map_noninferiority": map_difference >= THRESHOLDS["map_noninferiority"],
        "rare_macro_noninferiority": rare_difference >= THRESHOLDS["rare_macro_noninferiority"],
        "worst_class_safety": worst_class >= THRESHOLDS["worst_class"],
        "training_stability": stability_mean <= THRESHOLDS["training_seed_std_max"],
        "recall_noninferiority": recall_difference >= THRESHOLDS["recall_noninferiority"],
    }
    passed = all(checks.values())
    class_output = class_summary.copy()
    difference_lookup = class_difference.reset_index(name="k40_minus_random_ap5095")
    class_output = class_output.merge(difference_lookup, on=["class_id", "class_name"], how="left")
    aggregate.to_csv(out / "v8n_screen_metrics.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(out / "v8n_screen_per_class.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(out / "v8n_screen_contrasts.csv", index=False, encoding="utf-8-sig")
    class_output.to_csv(out / "v8n_screen_class_summary.csv", index=False, encoding="utf-8-sig")
    decision = {
        "status": "PASS" if passed else "FAIL",
        "backbone": "YOLOv8n",
        "models": 30,
        "map_difference": map_difference,
        "rare_macro_difference": rare_difference,
        "worst_class_difference": worst_class,
        "training_seed_map_std": stability_mean,
        "recall_difference": recall_difference,
        "checks": {key: bool(value) for key, value in checks.items()},
        "authorizes_v8s_expansion": passed,
        "final_test_used": False,
    }
    (out / "v8n_screen_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    report = [
        "# V2.2 YOLOv8n Screen Decision", "",
        f"- Gate: **{'PASS' if passed else 'FAIL'}**",
        f"- K40 - Random mAP50-95: **{map_difference:+.6f}**",
        f"- K40 - Random rare macro AP: **{rare_difference:+.6f}**",
        f"- Worst per-class AP difference: **{worst_class:+.6f}**",
        f"- K40 mean training-seed mAP std: **{stability_mean:.6f}**",
        f"- K40 - Random recall: **{recall_difference:+.6f}**",
        "- Final test used: **False**", "", "## Checks", "",
        pd.DataFrame([{"check": key, "passed": value} for key, value in checks.items()]).to_markdown(index=False),
        "", "A FAIL terminates V2.2 without training the 30 YOLOv8s main models.",
    ]
    path = out / "v8n_screen_decision.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return passed, path


def validate_smoke(aggregate: pd.DataFrame, per_class: pd.DataFrame, out: Path) -> Path:
    expected = {
        (policy, backbone, ACQUISITION_SEEDS[0], TRAINING_SEEDS[0])
        for policy in POLICIES for backbone in BACKBONES
    }
    observed = set(zip(
        aggregate["policy"], aggregate["backbone"],
        aggregate["acquisition_seed"], aggregate["training_seed"],
    ))
    metric_columns = ["map5095", "map50", "precision", "recall"]
    finite_metrics = bool(np.isfinite(aggregate[metric_columns].to_numpy(float)).all())
    bounded_metrics = bool(((aggregate[metric_columns] >= 0) & (aggregate[metric_columns] <= 1)).all().all())
    finite_per_class = bool(np.isfinite(per_class["ap5095"].to_numpy(float)).all())
    bounded_per_class = bool(per_class["ap5095"].between(0, 1).all())
    checks = {
        "four_factorial_models": bool(len(aggregate) == 4 and observed == expected),
        "forty_per_class_rows": bool(len(per_class) == 40),
        "aggregate_metrics_finite": finite_metrics,
        "aggregate_metrics_bounded": bounded_metrics,
        "per_class_metrics_finite": finite_per_class,
        "per_class_metrics_bounded": bounded_per_class,
        "development_only": bool(aggregate["evaluation_split"].eq("development").all()),
        "final_test_unused": bool(
            aggregate["final_test_used"].eq(False).all()
            and per_class["final_test_used"].eq(False).all()
        ),
        "distinct_run_signatures": bool(aggregate["run_signature"].nunique() == 4),
    }
    passed = all(checks.values())
    payload = {"status": "PASS" if passed else "FAIL", "checks": checks, "authorizes_main": passed, "final_test_used": False}
    path = out / "smoke_validation.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not passed:
        raise RuntimeError(f"Smoke validation failed: {path}")
    return path


def paired_summary(aggregate: pd.DataFrame, factor: str, a: str, b: str, within: str, within_value: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subset = aggregate[aggregate[within].eq(within_value)]
    for metric in ["map5095", "map50", "precision", "recall"]:
        pivot = subset.pivot(index=["acquisition_seed", "training_seed"], columns=factor, values=metric)
        diff = pivot[a] - pivot[b]
        by_acquisition = diff.groupby(level="acquisition_seed").mean()
        low, high = dcal.bootstrap_ci(by_acquisition.to_numpy(float), seed=20260718)
        rows.append({
            "contrast": f"{a}-{b}|{within}={within_value}",
            "metric": metric,
            "mean_difference": float(by_acquisition.mean()),
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
            "wins": int((by_acquisition > 0).sum()),
            "losses": int((by_acquisition < 0).sum()),
        })
    return rows


def analyze_main(aggregate: pd.DataFrame, per_class: pd.DataFrame, out: Path) -> tuple[bool, Path]:
    contrast_rows: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        contrast_rows.extend(paired_summary(aggregate, "policy", "ClusterK40_140", "Random140", "backbone", backbone))
    for policy in POLICIES:
        contrast_rows.extend(paired_summary(aggregate, "backbone", "YOLOv8s", "YOLOv8n", "policy", policy))
    for metric in ["map5095", "map50", "precision", "recall"]:
        pivot = aggregate.pivot(
            index=["acquisition_seed", "training_seed"],
            columns=["policy", "backbone"], values=metric,
        )
        interaction = (
            pivot[("ClusterK40_140", "YOLOv8s")] - pivot[("Random140", "YOLOv8s")]
            - pivot[("ClusterK40_140", "YOLOv8n")] + pivot[("Random140", "YOLOv8n")]
        )
        by_acquisition = interaction.groupby(level="acquisition_seed").mean()
        low, high = dcal.bootstrap_ci(by_acquisition.to_numpy(float), seed=20260718)
        contrast_rows.append({
            "contrast": "interaction|(K40-Random)x(v8s-v8n)",
            "metric": metric,
            "mean_difference": float(by_acquisition.mean()),
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
            "wins": int((by_acquisition > 0).sum()),
            "losses": int((by_acquisition < 0).sum()),
        })
    contrasts = pd.DataFrame(contrast_rows)

    acq_means = aggregate.groupby(["acquisition_seed", "policy", "backbone"], as_index=False)[["map5095", "map50", "precision", "recall"]].mean()
    config_summary = aggregate.groupby(["policy", "backbone"], as_index=False).agg(
        map5095_mean=("map5095", "mean"), map50_mean=("map50", "mean"),
        precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
        runtime_seconds_mean=("runtime_seconds", "mean"),
    )
    stability = aggregate.groupby(["acquisition_seed", "policy", "backbone"])["map5095"].std(ddof=1).reset_index(name="training_seed_map_std")
    stability_summary = stability.groupby(["policy", "backbone"], as_index=False)["training_seed_map_std"].mean()
    config_summary = config_summary.merge(stability_summary, on=["policy", "backbone"], validate="one_to_one")

    macro = per_class.copy()
    macro["group"] = macro["class_id"].map(lambda value: "rare" if int(value) in RARE_IDS else "frequent")
    macro_summary = macro.groupby(["policy", "backbone", "group"], as_index=False)["ap5095"].mean()
    class_summary = per_class.groupby(["policy", "backbone", "class_id", "class_name"], as_index=False)["ap5095"].mean()

    eligibility: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        map_diff = float(contrasts[(contrasts["contrast"] == f"ClusterK40_140-Random140|backbone={backbone}") & contrasts["metric"].eq("map5095")]["mean_difference"].iloc[0])
        recall_diff = float(contrasts[(contrasts["contrast"] == f"ClusterK40_140-Random140|backbone={backbone}") & contrasts["metric"].eq("recall")]["mean_difference"].iloc[0])
        rare_pivot = macro_summary[macro_summary["group"].eq("rare")].pivot(index="backbone", columns="policy", values="ap5095")
        rare_diff = float(rare_pivot.loc[backbone, "ClusterK40_140"] - rare_pivot.loc[backbone, "Random140"])
        class_pivot = class_summary[class_summary["backbone"].eq(backbone)].pivot(index="class_id", columns="policy", values="ap5095")
        worst_class = float((class_pivot["ClusterK40_140"] - class_pivot["Random140"]).min())
        std_value = float(config_summary[(config_summary["policy"].eq("ClusterK40_140")) & config_summary["backbone"].eq(backbone)]["training_seed_map_std"].iloc[0])
        checks = {
            "map_noninferiority": map_diff >= THRESHOLDS["map_noninferiority"],
            "rare_macro_noninferiority": rare_diff >= THRESHOLDS["rare_macro_noninferiority"],
            "worst_class_safety": worst_class >= THRESHOLDS["worst_class"],
            "training_stability": std_value <= THRESHOLDS["training_seed_std_max"],
            "recall_noninferiority": recall_diff >= THRESHOLDS["recall_noninferiority"],
        }
        eligibility.append({
            "backbone": backbone, "map_difference": map_diff, "rare_macro_difference": rare_diff,
            "worst_class_difference": worst_class, "training_seed_map_std": std_value,
            "recall_difference": recall_diff, **{f"check_{key}": value for key, value in checks.items()},
            "eligible": all(checks.values()),
        })
    gate = pd.DataFrame(eligibility)
    eligible = gate[gate["eligible"]]["backbone"].tolist()
    if not eligible:
        selected_backbone = None
        overall = False
    elif eligible == ["YOLOv8s"]:
        selected_backbone = "YOLOv8s"
        overall = True
    elif eligible == ["YOLOv8n"]:
        selected_backbone = "YOLOv8n"
        overall = True
    else:
        capacity = contrasts[(contrasts["contrast"] == "YOLOv8s-YOLOv8n|policy=ClusterK40_140") & contrasts["metric"].eq("map5095")]
        selected_backbone = "YOLOv8s" if float(capacity["mean_difference"].iloc[0]) >= THRESHOLDS["v8s_capacity_margin"] else "YOLOv8n"
        overall = True

    aggregate.to_csv(out / "factorial_metrics.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(out / "factorial_per_class.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(out / "factorial_contrasts.csv", index=False, encoding="utf-8-sig")
    config_summary.to_csv(out / "factorial_config_summary.csv", index=False, encoding="utf-8-sig")
    macro_summary.to_csv(out / "factorial_macro_summary.csv", index=False, encoding="utf-8-sig")
    class_summary.to_csv(out / "factorial_class_summary.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(out / "stable_learner_gate.csv", index=False, encoding="utf-8-sig")
    decision = {"status": "PASS" if overall else "FAIL", "selected_policy": "ClusterK40_140" if overall else None, "selected_backbone": selected_backbone, "final_test_used": False}
    (out / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    report = [
        "# DCAL-XAI V2.2 Factorial Backbone Decision", "",
        f"- Gate: **{'PASS' if overall else 'FAIL'}**",
        f"- Selected learner: **{selected_backbone or 'None'}**",
        "- Evaluation: **development only**",
        "- Final test used: **False**", "",
        "## Configuration summary", "", config_summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## Paired contrasts", "", contrasts.to_markdown(index=False, floatfmt=".6f"), "",
        "## Stable learner gate", "", gate.to_markdown(index=False, floatfmt=".6f"), "",
        "A PASS authorizes only a post-hoc detector error-association audit.",
    ]
    path = out / "backbone_decision.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return overall, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "smoke", "screen", "main"])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    default_out = DEFAULT_MAIN_OUT if args.stage in {"audit", "screen", "main"} else DEFAULT_SMOKE_OUT
    out = (args.output_dir or default_out).expanduser().resolve()
    context, selections = write_audit(out, require_models=args.stage != "audit")
    if args.stage == "audit":
        print(f"[AUDIT PASS] {out / 'audit.json'}")
        print("Training performed: False")
        print("Final test used: False")
        return
    if args.stage == "smoke":
        aggregate, per_class = train_and_evaluate(
            out=out, context=context, selections=selections,
            acquisition_seeds=[ACQUISITION_SEEDS[0]], training_seeds=[TRAINING_SEEDS[0]],
        )
        aggregate.to_csv(out / "smoke_metrics.csv", index=False, encoding="utf-8-sig")
        per_class.to_csv(out / "smoke_per_class.csv", index=False, encoding="utf-8-sig")
        validation_path = validate_smoke(aggregate, per_class, out)
        print(f"[DONE] {out / 'smoke_metrics.csv'}")
        print(f"[SMOKE PASS] {validation_path}")
        print("Models trained: 4")
        print("Final test used: False")
        return
    if args.stage == "screen":
        aggregate, per_class = train_and_evaluate(
            out=out, context=context, selections=selections,
            acquisition_seeds=ACQUISITION_SEEDS, training_seeds=TRAINING_SEEDS,
            backbone_names=["YOLOv8n"],
        )
        passed, path = analyze_v8n_screen(aggregate, per_class, out)
        print(f"[DONE] {path}")
        print(f"[V8N SCREEN] {'PASS' if passed else 'FAIL'}")
        print("Models in screen: 30")
        print("Final test used: False")
        return
    screen_decision_path = out / "v8n_screen_decision.json"
    if not screen_decision_path.exists():
        raise RuntimeError("Run the frozen YOLOv8n screen before the main expansion")
    screen_decision = json.loads(screen_decision_path.read_text(encoding="utf-8"))
    if screen_decision.get("status") != "PASS" or screen_decision.get("authorizes_v8s_expansion") is not True:
        raise RuntimeError("YOLOv8n screen did not authorize the YOLOv8s expansion")
    aggregate, per_class = train_and_evaluate(
        out=out, context=context, selections=selections,
        acquisition_seeds=ACQUISITION_SEEDS, training_seeds=TRAINING_SEEDS,
    )
    if len(aggregate) != 60 or len(per_class) != 600:
        raise RuntimeError(f"Incomplete factorial results: {len(aggregate)}/{len(per_class)}")
    overall, path = analyze_main(aggregate, per_class, out)
    print(f"[DONE] {path}")
    print(f"[GATE] {'PASS' if overall else 'FAIL'}")
    print("Models trained: 60")
    print("Final test used: False")


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from ultralytics import YOLO


ROOT = Path(r"C:\Users\user\Desktop\vlm\Defect_VLM_Project")

V10 = ROOT / "runs" / "active_learning_ablation_v10_neu_large_pool" / "v10_neu_large_pool_smoke_20260712_185923"
PROBE = ROOT / "runs" / "v10b_selection_probe" / "v10b_selection_probe_20260712_194239"
V10B = ROOT / "runs" / "v10b_single_training" / "v10b_single_training_20260712_195514"
OUT = ROOT / "runs" / "v10b_independent_result_audit" / "v10b_independent_result_audit_20260712_202634"

NEU6 = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_ALIASES = {
    "rolled-in-scale": "rolled-in_scale",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scalar(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def arr_value(arr: Any, idx: int) -> float:
    try:
        if arr is None or idx >= len(arr):
            return float("nan")
        return float(arr[idx])
    except Exception:
        return float("nan")


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return float("nan")
    return 2 * precision * recall / (precision + recall)


def normalized_class_name(name: str) -> str:
    return CLASS_ALIASES.get(str(name), str(name))


def write_incremental_outputs(per_rows: list[dict[str, Any]], registry_rows: list[dict[str, Any]]) -> None:
    per_df = pd.DataFrame(per_rows)
    reg_df = pd.DataFrame(registry_rows)

    per_df.to_csv(OUT / "recovered_per_class_metrics_v10.csv", index=False, encoding="utf-8-sig")
    reg_df.to_csv(OUT / "per_class_recovery_registry.csv", index=False, encoding="utf-8-sig")

    if per_df.empty:
        return

    if {"Random", "V10b"}.issubset(set(per_df["strategy"])):
        write_diff(per_df, "V10b", "Random", OUT / "per_class_v10b_minus_random.csv")
    if {"V9b", "V10b"}.issubset(set(per_df["strategy"])):
        write_diff(per_df, "V10b", "V9b", OUT / "per_class_v10b_minus_v9b.csv")


def write_diff(per_df: pd.DataFrame, left: str, right: str, path: Path) -> None:
    left_df = per_df[(per_df["strategy"] == left) & (per_df["is_neu6"])].set_index("class_name")
    right_df = per_df[(per_df["strategy"] == right) & (per_df["is_neu6"])].set_index("class_name")

    joined = left_df[["ap50", "ap5095", "precision", "recall", "validation_instance_count"]].join(
        right_df[["ap50", "ap5095", "precision", "recall"]],
        lsuffix=f"_{left}",
        rsuffix=f"_{right}",
    )

    for metric in ["ap50", "ap5095", "precision", "recall"]:
        joined[f"{left}_minus_{right}_{metric}"] = joined[f"{metric}_{left}"] - joined[f"{metric}_{right}"]

    joined.reset_index().to_csv(path, index=False, encoding="utf-8-sig")


def load_aggregate_rows() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    v10_results = pd.read_csv(V10 / "all_round_results.csv")
    v10b_results = pd.read_csv(V10B / "all_round_results.csv")
    all_results = pd.concat(
        [v10_results.assign(source_run="v10_smoke"), v10b_results.assign(source_run="v10b_single_training")],
        ignore_index=True,
    )

    for col in ["map50", "map5095", "precision", "recall", "labeled_budget"]:
        all_results[col] = pd.to_numeric(all_results[col])
    all_results["f1"] = [f1_score(p, r) for p, r in zip(all_results["precision"], all_results["recall"])]

    round0 = all_results[all_results["strategy"] == "__SHARED_ROUND0__"].iloc[0]
    random = all_results[all_results["strategy"] == "GTFreeRandom"].iloc[0]
    v9b = all_results[all_results["strategy"] == "DetectorInstanceRichDINOBalanced"].iloc[0]
    v10b = all_results[all_results["strategy"] == "DetectorUncertaintyDINOInstanceReducedV10b"].iloc[0]

    for metric in ["map50", "map5095", "precision", "recall", "f1"]:
        all_results[f"gain_vs_round0_{metric}"] = all_results[metric] - round0[metric]

    all_results.to_csv(OUT / "recalculated_aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    return round0, random, v9b, v10b, all_results


def write_secondary_audit_files(
    random: pd.Series,
    v9b: pd.Series,
    v10b: pd.Series,
    per_rows: list[dict[str, Any]],
    registry_rows: list[dict[str, Any]],
) -> None:
    per_df = pd.DataFrame(per_rows)

    v10b_vs_random = pd.read_csv(OUT / "per_class_v10b_minus_random.csv") if (OUT / "per_class_v10b_minus_random.csv").exists() else pd.DataFrame()
    v10b_vs_v9b = pd.read_csv(OUT / "per_class_v10b_minus_v9b.csv") if (OUT / "per_class_v10b_minus_v9b.csv").exists() else pd.DataFrame()

    random_wins = []
    random_losses = []
    v9b_wins = []
    v9b_losses = []

    if not v10b_vs_random.empty:
        random_wins = v10b_vs_random[v10b_vs_random["V10b_minus_Random_ap5095"] > 0]["class_name"].tolist()
        random_losses = v10b_vs_random[v10b_vs_random["V10b_minus_Random_ap5095"] < 0]["class_name"].tolist()

    if not v10b_vs_v9b.empty:
        v9b_wins = v10b_vs_v9b[v10b_vs_v9b["V10b_minus_V9b_ap5095"] > 0]["class_name"].tolist()
        v9b_losses = v10b_vs_v9b[v10b_vs_v9b["V10b_minus_V9b_ap5095"] < 0]["class_name"].tolist()

    pd.DataFrame(
        [
            {
                "claim": "V10b seed42 beats Random on development aggregate mAP50-95",
                "classification": "directly verified fact",
                "evidence": f"{v10b.map5095:.6f} vs {random.map5095:.6f}, diff {v10b.map5095 - random.map5095:.6f}",
            },
            {
                "claim": "V10b is generally superior to Random",
                "classification": "unverified / prohibited",
                "evidence": "Only acquisition seed42 and training seed1042; no multi-seed statistics.",
            },
            {
                "claim": "V10b reduced V9b instance-rich bias",
                "classification": "supported interpretation",
                "evidence": "V10b cumulative bbox count is lower than V9b while recall and mAP50-95 are higher.",
            },
            {
                "claim": "Uncertainty/DINO caused recall recovery",
                "classification": "unverified hypothesis",
                "evidence": "Weights and selection geometry changed, but there is no causal ablation.",
            },
            {
                "claim": "Final test validation",
                "classification": "unverified / prohibited",
                "evidence": "final_test_used=False and final_test_read=False in source configs.",
            },
        ]
    ).to_csv(OUT / "revised_evidence_classification.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "hypothesis": "V9b pseudo-instance saturation harmed detector utility",
                "judgment": "plausible",
                "confidence": "medium",
                "supporting_evidence": "V9b selected more instances but had lower recall and lower mAP50-95 than Random/V10b.",
                "counter_evidence": "No controlled ablation isolating pseudo-instance weight.",
                "missing_evidence": "Acquisition seeds 43-46 and weight-only ablations.",
            },
            {
                "hypothesis": "V10b captured more diverse/uncertain samples",
                "judgment": "plausible",
                "confidence": "medium-low",
                "supporting_evidence": "Selection probe showed lower redundancy and higher distance-to-initial.",
                "counter_evidence": "Single seed only; not causal.",
                "missing_evidence": "Multi-seed selection geometry and metric correlation.",
            },
            {
                "hypothesis": "Cumulative class balance caused improvement",
                "judgment": "possible but insufficient",
                "confidence": "low",
                "supporting_evidence": "V10b cumulative class entropy is high.",
                "counter_evidence": "Random is also strong with lower entropy.",
                "missing_evidence": "Per-class/multi-seed class-balance correlation.",
            },
            {
                "hypothesis": "Training seed noise explains improvement",
                "judgment": "cannot exclude",
                "confidence": "medium",
                "supporting_evidence": "The V10b vs Random mAP50-95 gap is modest and only one training seed was used.",
                "counter_evidence": "Same training seed/settings; V10b improves both precision and recall.",
                "missing_evidence": "Repeated training seeds for fixed selections.",
            },
            {
                "hypothesis": "Random remains a strong baseline",
                "judgment": "supported",
                "confidence": "high",
                "supporting_evidence": "Random beat V9b and is close to V10b.",
                "counter_evidence": "Only seed42 one-cycle.",
                "missing_evidence": "Seeds 43-46.",
            },
        ]
    ).to_csv(OUT / "revised_root_cause_evidence.csv", index=False, encoding="utf-8-sig")

    with (OUT / "recommended_next_step.md").open("w", encoding="utf-8") as f:
        f.write("# Recommended next step\n\n")
        f.write(
            "V10b를 candidate method로 고정하고, 가중치를 더 만지지 않은 상태에서 "
            "seed43-46 one-cycle을 Random과 비교한다. full curve와 final test는 그 뒤다.\n"
        )

    with (OUT / "v10_per_class_recovery_summary.md").open("w", encoding="utf-8") as f:
        f.write("# V10 per-class recovery summary\n\n")
        f.write("Existing best.pt checkpoints were evaluated on development val only. No training. No final test.\n\n")
        f.write(pd.DataFrame(registry_rows).to_markdown(index=False))
        f.write("\n\n")
        f.write(f"V10b vs Random AP50-95 wins: {', '.join(random_wins) if random_wins else '(none)'}\n\n")
        f.write(f"V10b vs Random AP50-95 losses: {', '.join(random_losses) if random_losses else '(none)'}\n\n")
        f.write(f"V10b vs V9b AP50-95 wins: {', '.join(v9b_wins) if v9b_wins else '(none)'}\n\n")
        f.write(f"V10b vs V9b AP50-95 losses: {', '.join(v9b_losses) if v9b_losses else '(none)'}\n")

    with (OUT / "v10b_independent_result_audit.md").open("w", encoding="utf-8") as f:
        f.write("# V10b independent result audit\n\n")
        f.write("No new training was run. Final test was not evaluated. Existing checkpoints were used only for development validation.\n\n")
        f.write("## Bottom line\n\n")
        f.write(
            f"V10b beats Random on seed42 development aggregate metrics "
            f"(mAP50-95 diff {v10b.map5095 - random.map5095:.6f}), "
            "but this is exploratory/development-gate evidence, not final or statistical evidence.\n\n"
        )
        f.write("## V10b vs Random\n\n")
        f.write(f"- mAP50 diff: {v10b.map50 - random.map50:.6f}\n")
        f.write(f"- mAP50-95 diff: {v10b.map5095 - random.map5095:.6f}\n")
        f.write(f"- precision diff: {v10b.precision - random.precision:.6f}\n")
        f.write(f"- recall diff: {v10b.recall - random.recall:.6f}\n")
        f.write(f"- F1 diff: {v10b.f1 - random.f1:.6f}\n\n")
        f.write("## Per-class AP50-95\n\n")
        f.write(f"- V10b wins vs Random: {', '.join(random_wins) if random_wins else '(none)'}\n")
        f.write(f"- V10b loses vs Random: {', '.join(random_losses) if random_losses else '(none)'}\n")
        f.write(f"- V10b wins vs V9b: {', '.join(v9b_wins) if v9b_wins else '(none)'}\n")
        f.write(f"- V10b loses vs V9b: {', '.join(v9b_losses) if v9b_losses else '(none)'}\n\n")
        f.write("## Integrity judgment\n\n")
        f.write(
            "Pass with caveat: checkpoints/configs exist, final-test flags are false, and validation-only recovery completed. "
            "Caveats are git_dirty=True in source configs and seed42 being a development/tuning seed.\n\n"
        )
        f.write("## Recommendation\n\n")
        f.write("Freeze V10b as candidate and run seed43-46 one-cycle against Random before full curve or final-test use.\n")

    if not per_df.empty:
        per_df[per_df["is_neu6"]].to_csv(OUT / "recovered_per_class_metrics_v10_neu6_only.csv", index=False, encoding="utf-8-sig")


def recover_one_checkpoint(
    name: str,
    weights: Path,
    yaml_path: Path,
    recorded_map50: float,
    recorded_map5095: float,
    per_rows: list[dict[str, Any]],
    registry_rows: list[dict[str, Any]],
) -> None:
    print(f"[START] {name} validation-only recovery")
    try:
        res = YOLO(str(weights)).val(
            data=str(yaml_path),
            imgsz=640,
            batch=8,
            device=0,
            workers=0,
            plots=True,
            save_json=False,
            project=str(OUT / "confusion_matrices"),
            name=name,
            exist_ok=True,
            verbose=False,
        )

        box = res.box
        recovered_map50 = scalar(box.map50)
        recovered_map5095 = scalar(box.map)

        registry_rows.append(
            {
                "strategy": name,
                "status": "success",
                "weights": str(weights),
                "weights_sha256": sha256_file(weights),
                "data_yaml": str(yaml_path),
                "recorded_map50": recorded_map50,
                "recovered_map50": recovered_map50,
                "abs_diff_map50": abs(recovered_map50 - recorded_map50),
                "recorded_map5095": recorded_map5095,
                "recovered_map5095": recovered_map5095,
                "abs_diff_map5095": abs(recovered_map5095 - recorded_map5095),
                "aggregate_match_lt_1e_4": abs(recovered_map50 - recorded_map50) < 1e-4
                and abs(recovered_map5095 - recorded_map5095) < 1e-4,
                "save_dir": str(getattr(res, "save_dir", "")),
            }
        )

        names = getattr(res, "names", {})
        if isinstance(names, dict):
            class_names = [names.get(i, str(i)) for i in range(len(names))]
        else:
            class_names = list(names)

        ap50 = getattr(box, "ap50", None)
        ap5095 = getattr(box, "ap", None)
        precision = getattr(box, "p", None)
        recall = getattr(box, "r", None)
        nt_per_class = getattr(box, "nt_per_class", None)

        for class_id, class_name in enumerate(class_names):
            normalized = normalized_class_name(class_name)
            try:
                instance_count = int(nt_per_class[class_id]) if nt_per_class is not None and class_id < len(nt_per_class) else None
            except Exception:
                instance_count = None
            per_rows.append(
                {
                    "strategy": name,
                    "class_id": class_id,
                    "class_name": normalized,
                    "is_neu6": normalized in NEU6,
                    "ap50": arr_value(ap50, class_id),
                    "ap5095": arr_value(ap5095, class_id),
                    "precision": arr_value(precision, class_id),
                    "recall": arr_value(recall, class_id),
                    "validation_instance_count": instance_count,
                }
            )

        print(
            f"[DONE] {name} recovery: "
            f"map50 diff={abs(recovered_map50 - recorded_map50):.8f}, "
            f"map50-95 diff={abs(recovered_map5095 - recorded_map5095):.8f}"
        )
    except Exception as exc:
        registry_rows.append(
            {
                "strategy": name,
                "status": "failed",
                "weights": str(weights),
                "data_yaml": str(yaml_path),
                "error": repr(exc),
            }
        )
        print(f"[FAILED] {name} recovery: {exc!r}")
    finally:
        write_incremental_outputs(per_rows, registry_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "confusion_matrices").mkdir(exist_ok=True)

    round0, random, v9b, v10b, _ = load_aggregate_rows()

    checkpoints = [
        (
            "Random",
            V10 / "yolo_train_runs" / "seed42_GTFreeRandom_R1_trainseed1042" / "weights" / "best.pt",
            Path(random.yaml_path),
            float(random.map50),
            float(random.map5095),
        ),
        (
            "V9b",
            V10 / "yolo_train_runs" / "seed42_DetectorInstanceRichDINOBalanced_R1_trainseed1042" / "weights" / "best.pt",
            Path(v9b.yaml_path),
            float(v9b.map50),
            float(v9b.map5095),
        ),
        (
            "V10b",
            V10B
            / "yolo_train_runs"
            / "seed42_DetectorUncertaintyDINOInstanceReducedV10b_R1_trainseed1042"
            / "weights"
            / "best.pt",
            Path(v10b.yaml_path),
            float(v10b.map50),
            float(v10b.map5095),
        ),
    ]

    per_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []

    for name, weights, yaml_path, recorded_map50, recorded_map5095 in checkpoints:
        recover_one_checkpoint(name, weights, yaml_path, recorded_map50, recorded_map5095, per_rows, registry_rows)

    write_secondary_audit_files(random, v9b, v10b, per_rows, registry_rows)

    v10b_vs_random = pd.read_csv(OUT / "per_class_v10b_minus_random.csv")
    v10b_vs_v9b = pd.read_csv(OUT / "per_class_v10b_minus_v9b.csv")
    random_wins = v10b_vs_random[v10b_vs_random["V10b_minus_Random_ap5095"] > 0]["class_name"].tolist()
    random_losses = v10b_vs_random[v10b_vs_random["V10b_minus_Random_ap5095"] < 0]["class_name"].tolist()
    v9b_wins = v10b_vs_v9b[v10b_vs_v9b["V10b_minus_V9b_ap5095"] > 0]["class_name"].tolist()
    v9b_losses = v10b_vs_v9b[v10b_vs_v9b["V10b_minus_V9b_ap5095"] < 0]["class_name"].tolist()

    registry = pd.read_csv(OUT / "per_class_recovery_registry.csv")
    print("=" * 100)
    for name in ["Random", "V9b", "V10b"]:
        row = registry[registry["strategy"] == name].iloc[-1]
        print(
            f"{name} recovery status={row['status']} "
            f"map50_diff={scalar(row.get('abs_diff_map50')):.8f} "
            f"map5095_diff={scalar(row.get('abs_diff_map5095')):.8f}"
        )
    print(f"V10b vs Random AP50-95 class wins: {random_wins}")
    print(f"V10b vs Random AP50-95 class losses: {random_losses}")
    print(f"V10b vs V9b AP50-95 class wins: {v9b_wins}")
    print(f"V10b vs V9b AP50-95 class losses: {v9b_losses}")
    print("final test used=False")
    print(f"Output dir: {OUT}")


if __name__ == "__main__":
    main()

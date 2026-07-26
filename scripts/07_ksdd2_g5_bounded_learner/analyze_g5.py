"""Apply the frozen G5 gate after all learner runs have completed."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from g5_common import (
    load_config,
    read_csv,
    run_root,
    seal_file,
    verify_frozen_execution,
    verify_seal,
    write_csv,
    write_json,
)


METRICS = [
    "positive_image_macro_hard_dice",
    "all_pixel_average_precision_hist4096",
    "image_auroc_using_max_pixel_probability",
    "normal_image_false_positive_rate",
    "positive_image_recall_at_overlapping_component",
]


def _bootstrap_ci(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    bootstrapped = values[indices].mean(axis=1)
    return (
        float(np.quantile(bootstrapped, 0.025)),
        float(np.quantile(bootstrapped, 0.975)),
    )


def _leave_one_out_min(values: np.ndarray) -> float:
    return float(min(np.delete(values, index).mean() for index in range(len(values))))


def _presentation_summary(
    decision: dict[str, Any],
    paired_rows: list[dict[str, Any]],
) -> str:
    summary = decision["summary"]
    return f"""# KSDD2 G5 bounded learner 결과

최종 판정: **{decision['decision']}**

## Frozen design

- Compact U-Net configuration: 1
- Acquisition seeds: 10
- Training seeds: 3
- Arms: PatchCore vs Random
- Completed runs: {decision['run_conformance']['completed_runs']}/60
- Official test use: 0

## Primary translation

- PatchCore positive-image macro Dice: **{summary['patchcore_macro_dice_mean']:.4f}**
- Random positive-image macro Dice: **{summary['random_macro_dice_mean']:.4f}**
- Mean paired delta: **{summary['macro_dice_delta_mean']:+.4f}**
- 95% CI: **[{summary['macro_dice_delta_ci_low']:+.4f}, {summary['macro_dice_delta_ci_high']:+.4f}]**
- Positive acquisition-seed fraction: **{summary['positive_acquisition_seed_fraction']:.1%}**
- Leave-one-acquisition-seed-out minimum delta: **{summary['leave_one_seed_out_min_delta']:+.4f}**

## Corroboration and safety

- Pixel AP delta: **{summary['pixel_ap_delta_mean']:+.4f}**
- PatchCore normal-image FPR: **{summary['patchcore_normal_fpr_mean']:.4f}**
- Normal-image FPR delta: **{summary['normal_fpr_delta_mean']:+.4f}**

## Claim boundary

이 결과는 하나의 사전 동결 compact U-Net과 재사용 development split에서의
screen-stage evidence다. Official test, production-pool generalization,
동일 annotation cost 또는 보편적 PatchCore 우위를 주장하지 않는다.
"""


def main() -> int:
    config = load_config()
    freeze_path, _ = verify_frozen_execution(config)
    out = run_root(config)
    plan = read_csv(out / "frozen_inputs" / "run_plan.csv")
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in plan:
        path = out / "learner_runs" / row["run_id"] / "run_result.json"
        if not verify_seal(path):
            missing.append(row["run_id"])
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "COMPLETED":
            missing.append(row["run_id"])
            continue
        results.append(result)
    required = int(config["g5_gate"]["run_conformance"]["required_completed_runs"])
    if missing or len(results) != required:
        decision = {
            "protocol_id": config["protocol"]["id"],
            "decision": config["decision"]["repeated_infrastructure_failure"],
            "run_conformance": {
                "completed_runs": len(results),
                "required_runs": required,
                "missing_run_ids": missing,
            },
            "official_test_accessed": False,
            "official_test_authorized": False,
        }
        target = out / config["outputs"]["decision"]
        write_json(target, decision)
        seal_file(target)
        print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    per_run_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        metrics = result["metrics"]
        values = [float(metrics[name]) for name in METRICS]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite frozen result: {result['run_id']}")
        row = {
            "run_id": result["run_id"],
            "acquisition_seed": int(result["acquisition_seed"]),
            "strategy": result["strategy"],
            "training_seed": int(result["training_seed"]),
            "training_seconds": float(result["training_seconds"]),
            "final_training_loss": float(result["final_training_loss"]),
            **{name: float(metrics[name]) for name in METRICS},
        }
        per_run_rows.append(row)
        grouped[(row["acquisition_seed"], row["strategy"])].append(row)

    training_seeds = {int(seed) for seed in config["comparison"]["training_seeds"]}
    paired_rows: list[dict[str, Any]] = []
    for acquisition_seed in config["comparison"]["acquisition_seeds"]:
        seed = int(acquisition_seed)
        arms: dict[str, dict[str, float]] = {}
        for strategy in config["comparison"]["strategies"]:
            runs = grouped[(seed, strategy)]
            if {row["training_seed"] for row in runs} != training_seeds:
                raise RuntimeError(f"Training-seed coverage mismatch: {seed}, {strategy}")
            arms[strategy] = {
                metric: float(np.mean([row[metric] for row in runs]))
                for metric in METRICS
            }
        row = {"acquisition_seed": seed}
        for metric in METRICS:
            row[f"patchcore_{metric}"] = arms["patchcore"][metric]
            row[f"random_{metric}"] = arms["random"][metric]
            row[f"delta_{metric}"] = (
                arms["patchcore"][metric] - arms["random"][metric]
            )
        paired_rows.append(row)

    primary_name = "positive_image_macro_hard_dice"
    primary_deltas = np.asarray(
        [row[f"delta_{primary_name}"] for row in paired_rows],
        dtype=np.float64,
    )
    ci_low, ci_high = _bootstrap_ci(
        primary_deltas,
        int(config["statistics"]["bootstrap_replicates"]),
        int(config["statistics"]["bootstrap_seed"]),
    )
    primary_gate = config["g5_gate"]["primary_learning_translation"]
    positive_fraction = float(np.mean(primary_deltas > 0))
    loo_min = _leave_one_out_min(primary_deltas)
    primary_checks = {
        "mean_delta_at_least_0_05": bool(
            primary_deltas.mean()
            >= float(primary_gate["macro_dice_delta_mean_min"])
        ),
        "ci_lower_above_zero": bool(
            ci_low
            > float(primary_gate["macro_dice_delta_ci_low_min_exclusive"])
        ),
        "positive_acquisition_seed_fraction": bool(
            positive_fraction
            >= float(primary_gate["positive_seed_fraction_min"])
        ),
        "leave_one_seed_out_above_zero": bool(
            loo_min
            > float(primary_gate["leave_one_seed_out_delta_min_exclusive"])
        ),
    }
    pixel_ap_name = "all_pixel_average_precision_hist4096"
    pixel_ap_delta = float(
        np.mean([row[f"delta_{pixel_ap_name}"] for row in paired_rows])
    )
    corroboration_pass = bool(
        pixel_ap_delta
        > float(
            config["g5_gate"]["corroboration"][
                "all_pixel_average_precision_delta_mean_min_exclusive"
            ]
        )
    )
    fpr_name = "normal_image_false_positive_rate"
    patchcore_fpr = float(
        np.mean([row[f"patchcore_{fpr_name}"] for row in paired_rows])
    )
    fpr_delta = float(np.mean([row[f"delta_{fpr_name}"] for row in paired_rows]))
    safety_config = config["g5_gate"]["safety"]
    safety_checks = {
        "patchcore_normal_fpr_at_most_0_20": bool(
            patchcore_fpr
            <= float(safety_config["candidate_normal_image_false_positive_rate_mean_max"])
        ),
        "normal_fpr_delta_at_most_0_05": bool(
            fpr_delta
            <= float(safety_config["normal_image_false_positive_rate_delta_mean_max"])
        ),
    }
    learning_pass = all(primary_checks.values()) and corroboration_pass
    safety_pass = all(safety_checks.values())
    if learning_pass and safety_pass:
        final_decision = config["decision"]["all_checks_pass"]
    elif learning_pass:
        final_decision = config["decision"]["learning_pass_safety_fail"]
    else:
        final_decision = config["decision"]["learning_translation_fail"]

    summary = {
        "patchcore_macro_dice_mean": float(
            np.mean([row[f"patchcore_{primary_name}"] for row in paired_rows])
        ),
        "random_macro_dice_mean": float(
            np.mean([row[f"random_{primary_name}"] for row in paired_rows])
        ),
        "macro_dice_delta_mean": float(primary_deltas.mean()),
        "macro_dice_delta_ci_low": ci_low,
        "macro_dice_delta_ci_high": ci_high,
        "positive_acquisition_seed_fraction": positive_fraction,
        "leave_one_seed_out_min_delta": loo_min,
        "pixel_ap_delta_mean": pixel_ap_delta,
        "patchcore_normal_fpr_mean": patchcore_fpr,
        "normal_fpr_delta_mean": fpr_delta,
    }
    decision = {
        "protocol_id": config["protocol"]["id"],
        "decision": final_decision,
        "run_conformance": {
            "completed_runs": len(results),
            "required_runs": required,
            "nonfinite_runs": 0,
        },
        "primary_checks": primary_checks,
        "corroboration": {
            "pixel_ap_delta_above_zero": corroboration_pass,
        },
        "safety_checks": safety_checks,
        "summary": summary,
        "official_test_accessed": False,
        "official_test_authorized": False,
        "hyperparameter_rescue_authorized": False,
        "freeze_manifest_sha256": (
            freeze_path.with_suffix(freeze_path.suffix + ".sha256")
            .read_text(encoding="ascii")
            .split()[0]
        ),
    }

    per_run_target = out / config["outputs"]["per_run_metrics"]
    write_csv(
        per_run_target,
        sorted(
            per_run_rows,
            key=lambda row: (
                row["acquisition_seed"],
                row["training_seed"],
                row["strategy"],
            ),
        ),
        list(per_run_rows[0]),
    )
    seal_file(per_run_target)
    paired_target = out / config["outputs"]["paired_metrics"]
    write_csv(paired_target, paired_rows, list(paired_rows[0]))
    seal_file(paired_target)
    decision_target = out / config["outputs"]["decision"]
    write_json(decision_target, decision)
    seal_file(decision_target)
    presentation_target = out / config["outputs"]["presentation_summary"]
    presentation_target.write_text(
        _presentation_summary(decision, paired_rows),
        encoding="utf-8",
    )
    seal_file(presentation_target)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

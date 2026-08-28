"""Supplementary development-to-locked-final comparison; never substitutes primary values."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_final_results import (
    NAMES,
    OUTPUT,
    STRATEGIES,
    acquisition_contrasts,
    build_curve_table,
    load_conceptual_metric_rows,
    paired_summary,
    write_csv,
)
from final_common import FINAL_LOCK, atomic_json, sha256_file


def load_development_rows() -> list[dict[str, Any]]:
    internal = json.loads((FINAL_LOCK / "INTERNAL_FINAL_EVALUATION_INPUTS.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for record in internal:
        metric_path = Path(record["best_pt_path"]).parent.parent / "OUTCOME_PRIVATE_DEVELOPMENT_METRICS.json"
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        if metric.get("metric_source") != "development":
            raise RuntimeError(f"development source mismatch: {record['cell_id']}")
        per_class = [float(value) for value in metric["per_class_map50_95"]]
        base: dict[str, Any] = {
            "cell_id": record["cell_id"],
            "dataset": record["dataset"],
            "acquisition_index": record["acquisition_index"],
            "acquisition_seed": record["acquisition_seed"],
            "training_seed_slot": record["training_seed_slot"],
            "training_seed": record["training_seed"],
            "round": record["round"],
            "labeled_count": record["labeled_count"],
            "budget_fraction": float(record["labeled_count"]) / (1605 if record["dataset"] == "GC10-DET" else 1259),
            "map50_95": float(metric["map50_95"]),
            "map50": float(metric["map50"]),
            "precision": float(metric["precision"]),
            "recall": float(metric["recall"]),
        }
        for index, name in enumerate(NAMES[str(record["dataset"])]):
            base[f"class_ap5095_{index}_{name}"] = per_class[index]
        if record["dataset"] == "GC10-DET":
            base["gc10_rare_macro_ap5095"] = float(np.mean(per_class[7:10]))
        else:
            base["neu_macro_ap5095"] = float(np.mean(per_class))
            base["neu_worst_class_ap5095"] = float(np.min(per_class))
            base["neu_worst_class_name"] = NAMES["NEU-DET"][int(np.argmin(per_class))]
        if record["strategy"] == "SHARED_R0":
            for strategy in STRATEGIES:
                rows.append({**base, "strategy": strategy, "shared_round0_source": True})
        else:
            rows.append({**base, "strategy": record["strategy"], "shared_round0_source": False})
    if len(rows) != 480:
        raise RuntimeError("development conceptual row count mismatch")
    return rows


def main() -> None:
    development_curves = build_curve_table(load_development_rows())
    final_curves = build_curve_table(load_conceptual_metric_rows())
    development = acquisition_contrasts(development_curves, "aulc_map50_95")
    final = acquisition_contrasts(final_curves, "aulc_map50_95")
    final_lookup = {(row["dataset"], int(row["acquisition_index"])): row for row in final}
    paired: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for row in development:
        locked = final_lookup[(row["dataset"], int(row["acquisition_index"]))]
        paired.append({
            "dataset": row["dataset"],
            "acquisition_index": row["acquisition_index"],
            "acquisition_seed": row["acquisition_seed"],
            "development_dino_minus_random_aulc": row["dino_minus_random"],
            "locked_final_dino_minus_random_aulc": locked["dino_minus_random"],
            "locked_final_minus_development_contrast": float(locked["dino_minus_random"]) - float(row["dino_minus_random"]),
        })
    for dataset in ("GC10-DET", "NEU-DET"):
        dev_values = [float(row["development_dino_minus_random_aulc"]) for row in paired if row["dataset"] == dataset]
        final_values = [float(row["locked_final_dino_minus_random_aulc"]) for row in paired if row["dataset"] == dataset]
        gap_values = [float(row["locked_final_minus_development_contrast"]) for row in paired if row["dataset"] == dataset]
        summary.append({
            **paired_summary(dev_values, dataset=dataset, metric="supplementary_development_aulc", test=False),
            "locked_final_mean": float(np.mean(final_values)),
            "locked_final_minus_development_mean": float(np.mean(gap_values)),
            "pearson_acquisition_delta_development_vs_final": float(np.corrcoef(dev_values, final_values)[0, 1]),
            "role": "SUPPLEMENTARY_DIAGNOSTIC_NOT_PRIMARY",
        })
    write_csv(OUTPUT / "development_vs_locked_final_acquisition_deltas.csv", paired)
    write_csv(OUTPUT / "development_vs_locked_final_summary.csv", summary)
    audit = {
        "status": "PASS",
        "development_substituted_into_primary": False,
        "development_curves": len(development_curves),
        "locked_final_curves": len(final_curves),
        "paired_acquisition_rows": len(paired),
        "tables": {
            "development_vs_locked_final_acquisition_deltas.csv": sha256_file(OUTPUT / "development_vs_locked_final_acquisition_deltas.csv"),
            "development_vs_locked_final_summary.csv": sha256_file(OUTPUT / "development_vs_locked_final_summary.csv"),
        },
    }
    atomic_json(OUTPUT / "DEVELOPMENT_FINAL_DIAGNOSTIC_AUDIT.json", audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()


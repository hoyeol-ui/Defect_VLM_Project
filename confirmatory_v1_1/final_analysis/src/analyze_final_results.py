"""Run the frozen confirmatory estimands after all locked-final evaluations close."""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_common import (
    EXPECTED_CELLS,
    FINAL_ANALYSIS,
    FINAL_EVALUATION,
    WORKTREE,
    atomic_json,
    read_json,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prepare_yolo_benchmark_data import NAMES, class_id  # noqa: E402


OUTPUT = FINAL_ANALYSIS / "results"
RAW_METRICS = FINAL_EVALUATION / "raw_metrics"
PLAN = ROOT / "full_run" / "planned_cells.jsonl"
SOURCE_REGISTRY = WORKTREE / "confirmatory_v1" / "data_freeze" / "private_source_registry.csv"
DATA_ROOT = Path("C:/Users/user/Desktop/vlm/Defect_VLM_Project/data")
ACQUISITION_N = {"GC10-DET": 1605, "NEU-DET": 1259}
BOOTSTRAP_SEEDS = {"GC10-DET": 421397045, "NEU-DET": 1672358496}
STRATEGIES = ("Random-PCG64-v1", "FrozenDINO-FF-v1")
STRATEGY_SHORT = {"Random-PCG64-v1": "Random", "FrozenDINO-FF-v1": "DINO"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_conceptual_metric_rows() -> list[dict[str, Any]]:
    audit = read_json(FINAL_EVALUATION / "FINAL_EVALUATION_INTEGRITY_AUDIT.json")
    if audit.get("status") != "PASS" or audit.get("completed_evaluations") != EXPECTED_CELLS:
        raise RuntimeError("final evaluation integrity is not PASS")
    files = sorted(RAW_METRICS.glob("*.json"))
    if len(files) != EXPECTED_CELLS:
        raise RuntimeError("raw final metric count mismatch")
    conceptual: list[dict[str, Any]] = []
    for path in files:
        row = read_json(path)
        if row.get("status") != "COMPLETE" or row.get("metric_source") != "locked_final":
            raise RuntimeError(f"invalid raw final row: {path.name}")
        base = {
            key: row[key]
            for key in (
                "cell_id", "dataset", "acquisition_index", "acquisition_seed",
                "training_seed_slot", "training_seed", "round", "labeled_count",
                "manifest_sha256", "best_pt_sha256", "map50_95", "map50",
                "precision", "recall", "evaluation_runtime_seconds",
            )
        }
        base["budget_fraction"] = float(row["labeled_count"]) / ACQUISITION_N[str(row["dataset"])]
        per_class = [float(value) for value in row["per_class_map50_95"]]
        names = list(row["class_names"])
        if names != NAMES[str(row["dataset"])] or len(per_class) != len(names):
            raise RuntimeError(f"class metric mismatch: {path.name}")
        for index, name in enumerate(names):
            base[f"class_ap5095_{index}_{name}"] = per_class[index]
        if row["dataset"] == "GC10-DET":
            base["gc10_rare_macro_ap5095"] = float(np.mean(per_class[7:10]))
        else:
            base["neu_macro_ap5095"] = float(np.mean(per_class))
            base["neu_worst_class_ap5095"] = float(np.min(per_class))
            base["neu_worst_class_name"] = names[int(np.argmin(per_class))]
        if row["strategy"] == "SHARED_R0":
            for strategy in STRATEGIES:
                conceptual.append({**base, "strategy": strategy, "shared_round0_source": True})
        else:
            conceptual.append({**base, "strategy": row["strategy"], "shared_round0_source": False})
    if len(conceptual) != 480:
        raise RuntimeError(f"conceptual row count mismatch: {len(conceptual)}")
    keys = {
        (
            row["dataset"], row["acquisition_index"], row["strategy"],
            row["training_seed_slot"], row["round"],
        )
        for row in conceptual
    }
    if len(keys) != 480:
        raise RuntimeError("conceptual metric keys are not unique")
    return sorted(
        conceptual,
        key=lambda row: (
            row["dataset"], row["acquisition_index"], row["strategy"],
            row["training_seed_slot"], row["round"],
        ),
    )


def trapezoid_aulc(rows: list[dict[str, Any]], metric: str) -> float:
    ordered = sorted(rows, key=lambda row: int(row["round"]))
    if [int(row["round"]) for row in ordered] != list(range(6)):
        raise RuntimeError("AULC requires exactly rounds 0..5")
    x = np.asarray([float(row["budget_fraction"]) for row in ordered], dtype=np.float64)
    y = np.asarray([float(row[metric]) for row in ordered], dtype=np.float64)
    if not np.all(np.isfinite(y)) or not np.all(np.diff(x) > 0):
        raise RuntimeError("AULC received invalid values")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def group_rows(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    return grouped


def build_curve_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = ["map50_95", "map50", "recall"]
    curves: list[dict[str, Any]] = []
    keys = ("dataset", "acquisition_index", "acquisition_seed", "strategy", "training_seed_slot", "training_seed")
    for key, values in sorted(group_rows(rows, keys).items()):
        if len(values) != 6:
            raise RuntimeError(f"incomplete curve: {key}")
        curve = dict(zip(keys, key))
        for metric in metric_names:
            curve[f"aulc_{metric}"] = trapezoid_aulc(values, metric)
            curve[f"final_{metric}"] = float(next(row[metric] for row in values if int(row["round"]) == 5))
        dataset = str(curve["dataset"])
        if dataset == "GC10-DET":
            metric = "gc10_rare_macro_ap5095"
            curve["aulc_gc10_rare_macro_ap5095"] = trapezoid_aulc(values, metric)
            curve["final_gc10_rare_macro_ap5095"] = float(next(row[metric] for row in values if int(row["round"]) == 5))
        else:
            for metric in ("neu_macro_ap5095", "neu_worst_class_ap5095"):
                curve[f"aulc_{metric}"] = trapezoid_aulc(values, metric)
                curve[f"final_{metric}"] = float(next(row[metric] for row in values if int(row["round"]) == 5))
        curves.append(curve)
    if len(curves) != 80:
        raise RuntimeError(f"curve count mismatch: {len(curves)}")
    return curves


def percentile_ci(values: list[float], seed: int, replicates: int = 100_000) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    boot = array[indices].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def exact_sign_flip_p(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(array))), dtype=np.float64)
    permuted = np.abs((signs * array).mean(axis=1))
    return float(np.count_nonzero(permuted >= observed - 1e-15) / len(permuted))


def paired_summary(values: list[float], *, dataset: str, metric: str, test: bool) -> dict[str, Any]:
    if len(values) != 10:
        raise RuntimeError(f"paired acquisition count is not 10: {dataset}/{metric}")
    low, high = percentile_ci(values, BOOTSTRAP_SEEDS[dataset])
    result = {
        "dataset": dataset,
        "metric": metric,
        "n_acquisition_seeds": len(values),
        "mean_dino_minus_random": float(np.mean(values)),
        "median_dino_minus_random": float(np.median(values)),
        "sd_dino_minus_random": float(np.std(values, ddof=1)),
        "ci95_low": low,
        "ci95_high": high,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "wins": int(np.count_nonzero(np.asarray(values) > 0)),
        "losses": int(np.count_nonzero(np.asarray(values) < 0)),
        "ties": int(np.count_nonzero(np.asarray(values) == 0)),
    }
    if test:
        result["exact_sign_flip_p_raw"] = exact_sign_flip_p(values)
    return result


def acquisition_contrasts(curves: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    means: dict[tuple[str, int, int, str], list[float]] = {}
    for row in curves:
        key = (str(row["dataset"]), int(row["acquisition_index"]), int(row["acquisition_seed"]), str(row["strategy"]))
        means.setdefault(key, []).append(float(row[metric]))
    output: list[dict[str, Any]] = []
    for dataset in sorted({str(row["dataset"]) for row in curves}):
        for acquisition_index in range(10):
            matches = [key for key in means if key[0] == dataset and key[1] == acquisition_index]
            seeds = {key[2] for key in matches}
            if len(seeds) != 1:
                raise RuntimeError("acquisition seed identity mismatch")
            acquisition_seed = next(iter(seeds))
            random_values = means[(dataset, acquisition_index, acquisition_seed, STRATEGIES[0])]
            dino_values = means[(dataset, acquisition_index, acquisition_seed, STRATEGIES[1])]
            if len(random_values) != 2 or len(dino_values) != 2:
                raise RuntimeError("training seed reduction mismatch")
            random_mean = float(np.mean(random_values))
            dino_mean = float(np.mean(dino_values))
            output.append({
                "dataset": dataset,
                "acquisition_index": acquisition_index,
                "acquisition_seed": acquisition_seed,
                "metric": metric,
                "random_two_training_seed_mean": random_mean,
                "dino_two_training_seed_mean": dino_mean,
                "dino_minus_random": dino_mean - random_mean,
            })
    return output


def holm_adjust_two(summary: list[dict[str, Any]]) -> None:
    indexed = sorted(
        ((index, float(row["exact_sign_flip_p_raw"])) for index, row in enumerate(summary)),
        key=lambda item: item[1],
    )
    running = 0.0
    for rank, (index, raw) in enumerate(indexed):
        adjusted = min(1.0, raw * (len(indexed) - rank))
        running = max(running, adjusted)
        summary[index]["holm_adjusted_p"] = running


def budget_curve_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reduced: dict[tuple[str, int, int, int, str], list[float]] = {}
    for row in rows:
        key = (
            str(row["dataset"]), int(row["acquisition_index"]), int(row["acquisition_seed"]),
            int(row["round"]), str(row["strategy"]),
        )
        reduced.setdefault(key, []).append(float(row["map50_95"]))
    output: list[dict[str, Any]] = []
    for dataset in ("GC10-DET", "NEU-DET"):
        for round_index in range(6):
            random_means: list[float] = []
            dino_means: list[float] = []
            fractions: list[float] = []
            for acquisition_index in range(10):
                matching = [key for key in reduced if key[0] == dataset and key[1] == acquisition_index and key[3] == round_index]
                acquisition_seed = matching[0][2]
                random_means.append(float(np.mean(reduced[(dataset, acquisition_index, acquisition_seed, round_index, STRATEGIES[0])])))
                dino_means.append(float(np.mean(reduced[(dataset, acquisition_index, acquisition_seed, round_index, STRATEGIES[1])])))
                fractions.append(next(float(row["budget_fraction"]) for row in rows if row["dataset"] == dataset and int(row["acquisition_index"]) == acquisition_index and int(row["round"]) == round_index))
            deltas = (np.asarray(dino_means) - np.asarray(random_means)).tolist()
            low, high = percentile_ci(deltas, BOOTSTRAP_SEEDS[dataset])
            output.append({
                "dataset": dataset,
                "round": round_index,
                "budget_fraction": float(np.mean(fractions)),
                "random_mean_map50_95": float(np.mean(random_means)),
                "dino_mean_map50_95": float(np.mean(dino_means)),
                "mean_dino_minus_random": float(np.mean(deltas)),
                "delta_ci95_low": low,
                "delta_ci95_high": high,
                "wins": int(np.count_nonzero(np.asarray(deltas) > 0)),
                "losses": int(np.count_nonzero(np.asarray(deltas) < 0)),
            })
    return output


def classwise_final_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_rows = [row for row in rows if int(row["round"]) == 5]
    output: list[dict[str, Any]] = []
    for dataset in ("GC10-DET", "NEU-DET"):
        for class_index, class_name in enumerate(NAMES[dataset]):
            metric = f"class_ap5095_{class_index}_{class_name}"
            values: list[float] = []
            for acquisition_index in range(10):
                subset = [row for row in final_rows if row["dataset"] == dataset and int(row["acquisition_index"]) == acquisition_index]
                random_mean = float(np.mean([row[metric] for row in subset if row["strategy"] == STRATEGIES[0]]))
                dino_mean = float(np.mean([row[metric] for row in subset if row["strategy"] == STRATEGIES[1]]))
                values.append(dino_mean - random_mean)
            summary = paired_summary(values, dataset=dataset, metric=f"final_class_ap5095:{class_name}", test=False)
            summary["class_index_yolo_zero_based"] = class_index
            summary["class_name"] = class_name
            output.append(summary)
    return output


def build_selection_audit() -> list[dict[str, Any]]:
    source = read_csv(SOURCE_REGISTRY)
    source_lookup = {(row["dataset"], row["sample_id"]): row for row in source}
    plans = [json.loads(line) for line in PLAN.read_text(encoding="utf-8").splitlines() if line]
    unique: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for plan in plans:
        key = (str(plan["dataset"]), int(plan["acquisition_index"]), str(plan["strategy_token"]), int(plan["round"]))
        unique.setdefault(key, plan)
    if len(unique) != 220:
        raise RuntimeError("unique acquisition state count mismatch")
    raw: list[dict[str, Any]] = []
    for key, plan in sorted(unique.items()):
        dataset, acquisition_index, strategy_token, round_index = key
        ids = [line for line in Path(str(plan["manifest_path"])).read_text(encoding="ascii").splitlines() if line]
        image_counts = np.zeros(len(NAMES[dataset]), dtype=np.int64)
        instance_counts = np.zeros(len(NAMES[dataset]), dtype=np.int64)
        target_count = 0
        for sample_id in ids:
            row = source_lookup[(dataset, sample_id)]
            image_path = DATA_ROOT / row["relative_image_path"]
            xml_path = DATA_ROOT / row["relative_annotation_path"]
            present: set[int] = set()
            for obj in ET.parse(xml_path).getroot().findall("object"):
                cid = class_id(dataset, obj.findtext("name") or "", image_path)
                if cid is None:
                    continue
                instance_counts[cid] += 1
                present.add(cid)
            for cid in present:
                image_counts[cid] += 1
            if dataset == "GC10-DET" and present.intersection({7, 8, 9}):
                target_count += 1
        positive = image_counts[image_counts > 0].astype(np.float64)
        if len(positive) <= 1:
            entropy = 0.0
        else:
            probabilities = positive / positive.sum()
            entropy = float(-(probabilities * np.log(probabilities)).sum() / math.log(len(NAMES[dataset])))
        base = {
            "dataset": dataset,
            "acquisition_index": acquisition_index,
            "acquisition_seed": int(plan["acquisition_seed"]),
            "round": round_index,
            "labeled_count": len(ids),
            "manifest_sha256": plan["manifest_sha256"],
            "unique_class_coverage": int(np.count_nonzero(image_counts)),
            "normalized_shannon_image_incidence": entropy,
            "total_instances": int(instance_counts.sum()),
        }
        for index, name in enumerate(NAMES[dataset]):
            base[f"class_image_count_{index}_{name}"] = int(image_counts[index])
            base[f"class_instance_count_{index}_{name}"] = int(instance_counts[index])
        if dataset == "GC10-DET":
            base["gc10_rare_target_image_count"] = target_count
        if strategy_token == "SHARED":
            for strategy in STRATEGIES:
                raw.append({**base, "strategy": strategy, "shared_round0_source": True})
        else:
            strategy = STRATEGIES[0] if strategy_token == "Random" else STRATEGIES[1]
            raw.append({**base, "strategy": strategy, "shared_round0_source": False})
    if len(raw) != 240:
        raise RuntimeError("conceptual selection state count mismatch")
    return raw


def selection_contrast_summary(selection: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics_by_dataset = {
        "GC10-DET": ["gc10_rare_target_image_count", "normalized_shannon_image_incidence", "total_instances"],
        "NEU-DET": ["normalized_shannon_image_incidence", "total_instances"]
        + [f"class_image_count_{i}_{name}" for i, name in enumerate(NAMES["NEU-DET"])],
    }
    output: list[dict[str, Any]] = []
    for dataset, metrics in metrics_by_dataset.items():
        for round_index in range(1, 6):
            subset = [row for row in selection if row["dataset"] == dataset and int(row["round"]) == round_index]
            for metric in metrics:
                deltas: list[float] = []
                for acquisition_index in range(10):
                    acquisition = [row for row in subset if int(row["acquisition_index"]) == acquisition_index]
                    random_value = float(next(row[metric] for row in acquisition if row["strategy"] == STRATEGIES[0]))
                    dino_value = float(next(row[metric] for row in acquisition if row["strategy"] == STRATEGIES[1]))
                    deltas.append(dino_value - random_value)
                summary = paired_summary(deltas, dataset=dataset, metric=metric, test=False)
                summary["round"] = round_index
                output.append(summary)
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metric_rows = load_conceptual_metric_rows()
    curves = build_curve_table(metric_rows)
    primary_deltas = acquisition_contrasts(curves, "aulc_map50_95")
    primary_summary: list[dict[str, Any]] = []
    for dataset in ("GC10-DET", "NEU-DET"):
        values = [float(row["dino_minus_random"]) for row in primary_deltas if row["dataset"] == dataset]
        primary_summary.append(paired_summary(values, dataset=dataset, metric="primary_aulc_map50_95", test=True))
    holm_adjust_two(primary_summary)

    secondary_metrics = ["final_map50_95", "aulc_recall", "final_recall"]
    secondary_metrics += ["aulc_gc10_rare_macro_ap5095", "final_gc10_rare_macro_ap5095"]
    secondary_metrics += ["aulc_neu_macro_ap5095", "final_neu_macro_ap5095", "aulc_neu_worst_class_ap5095", "final_neu_worst_class_ap5095"]
    secondary_deltas: list[dict[str, Any]] = []
    secondary_summary: list[dict[str, Any]] = []
    for metric in secondary_metrics:
        dataset = "GC10-DET" if "gc10" in metric else "NEU-DET" if "neu" in metric else None
        relevant_curves = curves if dataset is None else [row for row in curves if row["dataset"] == dataset]
        contrasts = acquisition_contrasts(relevant_curves, metric)
        secondary_deltas.extend(contrasts)
        for current_dataset in ("GC10-DET", "NEU-DET"):
            values = [float(row["dino_minus_random"]) for row in contrasts if row["dataset"] == current_dataset]
            if values:
                secondary_summary.append(paired_summary(values, dataset=current_dataset, metric=metric, test=False))

    selection = build_selection_audit()
    selection_summary = selection_contrast_summary(selection)
    curves_by_key = {(row["dataset"], int(row["acquisition_index"]), row["strategy"], row["training_seed_slot"]): row for row in curves}
    training_seed_sensitivity: list[dict[str, Any]] = []
    for dataset in ("GC10-DET", "NEU-DET"):
        for strategy in STRATEGIES:
            differences = []
            for acquisition_index in range(10):
                a = curves_by_key[(dataset, acquisition_index, strategy, "A")]["aulc_map50_95"]
                b = curves_by_key[(dataset, acquisition_index, strategy, "B")]["aulc_map50_95"]
                differences.append(abs(float(a) - float(b)))
            training_seed_sensitivity.append({
                "dataset": dataset,
                "strategy": strategy,
                "mean_absolute_A_minus_B_aulc": float(np.mean(differences)),
                "median_absolute_A_minus_B_aulc": float(np.median(differences)),
                "maximum_absolute_A_minus_B_aulc": float(np.max(differences)),
            })

    tables = {
        "raw_locked_final_metrics_conceptual_480.csv": metric_rows,
        "curve_level_aulc_80.csv": curves,
        "primary_acquisition_deltas_20.csv": primary_deltas,
        "primary_summary_2.csv": primary_summary,
        "secondary_acquisition_deltas.csv": secondary_deltas,
        "secondary_summary.csv": secondary_summary,
        "budget_curve_summary.csv": budget_curve_summary(metric_rows),
        "classwise_final_summary.csv": classwise_final_summary(metric_rows),
        "selection_composition_raw_240.csv": selection,
        "selection_composition_summary.csv": selection_summary,
        "training_seed_sensitivity.csv": training_seed_sensitivity,
    }
    for name, table in tables.items():
        write_csv(OUTPUT / name, table)
    audit = {
        "status": "PASS",
        "raw_scientific_evaluations": EXPECTED_CELLS,
        "conceptual_strategy_budget_rows": len(metric_rows),
        "complete_six_point_curves": len(curves),
        "primary_acquisition_units": len(primary_deltas),
        "selection_states_conceptual": len(selection),
        "bootstrap_replicates": 100_000,
        "exact_sign_assignments_per_dataset": 1024,
        "cross_dataset_pooling": False,
        "development_metric_substitutions": 0,
        "missing_or_interpolated_primary_points": 0,
        "table_sha256": {name: sha256_file(OUTPUT / name) for name in tables},
        "primary_summary": primary_summary,
    }
    atomic_json(OUTPUT / "CONFIRMATORY_ANALYSIS_INTEGRITY_AND_SUMMARY.json", audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()

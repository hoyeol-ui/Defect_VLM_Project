"""V10c24 budget/round scale smoke: Random vs frozen V10c24 only.

Protocol:
  - NEU large-pool split, final test locked and unused
  - initial budget 60
  - query size 30
  - rounds 2 => budgets 60/90/120
  - Round2 rescoring uses each strategy's latest detector state
  - no V9b, no V10b, no PDF 21/9, no V10d

AL_DRY_RUN_ONLY=1 is a true structural dry-run: it does not train YOLO.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("AL_EMBEDDING_BACKEND", "dinov2")
os.environ.setdefault("AL_ALLOW_MODEL_DOWNLOAD", "0")

import run_v10b_multiseed_onecycle as base  # noqa: E402
import run_v10c_recall_guard_onecycle as v10c  # noqa: E402


PROJECT_ROOT = base.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "active_learning_v10c24_round2_scale_smoke"
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "active_learning_v10c24_round2_scale_smoke"

ROUND0_STRATEGY = base.ROUND0_STRATEGY
RANDOM_STRATEGY = base.RANDOM_STRATEGY
V10C_STRATEGY = v10c.V10C_STRATEGY


def int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or not str(value).strip() else int(value)


def float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or not str(value).strip() else float(value)


def parse_bool_env(name: str, default: bool = False) -> bool:
    return base.parse_bool_env(name, default)


def parse_seed_list() -> list[int]:
    raw = os.environ.get("AL_ACQUISITION_SEEDS", "51")
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise ValueError("AL_ACQUISITION_SEEDS is empty.")
    if 42 in seeds:
        raise ValueError("Seed42 is forbidden here; it was used for development.")
    return seeds


def table_md(df: pd.DataFrame) -> str:
    return base.table_md(df)


def trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def sample_ids(df: pd.DataFrame) -> set[str]:
    if df.empty or "sample_id" not in df.columns:
        return set()
    return set(df["sample_id"].astype(str))


def add_selection_metadata(
    df: pd.DataFrame,
    *,
    acquisition_seed: int,
    training_seed: int,
    strategy: str,
    round_idx: int,
    selection_type: str,
    dry_run_placeholder: bool = False,
) -> pd.DataFrame:
    return base.add_selection_metadata(
        df,
        acquisition_seed=acquisition_seed,
        training_seed=training_seed,
        strategy=strategy,
        round_idx=round_idx,
        selection_type=selection_type,
        dry_run_placeholder=dry_run_placeholder,
    )


def dry_result_row(
    *,
    acquisition_seed: int,
    training_seed: int,
    strategy: str,
    round_idx: int,
    labeled_budget: int,
    dev_eval_size: int,
) -> dict[str, Any]:
    return {
        "acquisition_seed": acquisition_seed,
        "training_seed": training_seed,
        "strategy": strategy,
        "round": round_idx,
        "labeled_budget": labeled_budget,
        "development_eval_size": dev_eval_size,
        "yaml_path": "",
        "map50": np.nan,
        "map5095": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "train_status": "dry_run_skipped",
        "train_run_dir": "",
        "error": "",
        "retry_count": 0,
    }


def validate_no_overlap(initial_df: pd.DataFrame, query: pd.DataFrame, dev_df: pd.DataFrame, *, expected_query: int, label: str) -> dict[str, Any]:
    initial = sample_ids(initial_df)
    query_ids = sample_ids(query)
    dev = sample_ids(dev_df)
    row = {
        "label": label,
        "query_size": len(query_ids),
        "expected_query_size": expected_query,
        "initial_query_overlap": len(initial & query_ids),
        "train_dev_overlap": len((initial | query_ids) & dev),
        "pass": len(query_ids) == expected_query and len(initial & query_ids) == 0 and len((initial | query_ids) & dev) == 0,
    }
    if not row["pass"]:
        raise ValueError(f"Labeled-set validation failed for {label}: {row}")
    return row


def select_v10c_from_detector(
    *,
    detector_weights: Path,
    remaining_pool: pd.DataFrame,
    labeled_reference: pd.DataFrame,
    embedding_lookup: dict[str, np.ndarray],
    save_dir: Path,
    seed: int,
    round_idx: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not detector_weights.exists():
        raise FileNotFoundError(detector_weights)
    model = base.detector_probe.YOLO(str(detector_weights))
    detector_scores = base.detector_probe.prediction_rows(
        model,
        remaining_pool,
        device=str(config["device"]),
        imgsz=int(config["imgsz"]),
        conf=float(config["predict_conf"]),
        iou=float(config["predict_iou"]),
    )
    scored_pool = remaining_pool.merge(detector_scores, on="sample_id", how="left")
    scored_pool.to_csv(save_dir / f"seed{seed}_round{round_idx}_v10c_detector_scores.csv", index=False, encoding="utf-8-sig")
    selected = v10c.select_v10c_recall_guard(
        scored_pool,
        labeled_reference,
        embedding_lookup,
        query_size=int(config["query_size"]),
        candidate_fraction=float(config["candidate_fraction"]),
        core_size=int(config["v10c_core_size"]),
        recall_guard_size=int(config["v10c_recall_guard_size"]),
        v10c_weights=dict(config["v10c_weights"]),
        guard_weights=dict(config["recall_guard_weights"]),
        core_max_no_box=int(config["core_constraints"]["max_no_box"]),
        core_min_pseudo_boxes=int(config["core_constraints"]["min_pseudo_boxes"]),
        core_max_per_pred_class=int(config["core_constraints"]["max_per_pred_class"]),
        guard_max_no_box=int(config["guard_constraints"]["max_no_box"]),
        guard_max_per_pred_class=int(config["guard_constraints"]["max_per_pred_class"]),
    )
    if len(selected) != int(config["query_size"]):
        raise ValueError(f"V10c selected {len(selected)} samples at seed {seed} round {round_idx}; expected {config['query_size']}")
    return selected.reset_index(drop=True), scored_pool


def train_round(
    *,
    labeled_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    dataset_root: Path,
    save_dir: Path,
    strategy: str,
    acquisition_seed: int,
    training_seed: int,
    round_idx: int,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if dry_run:
        return (
            dry_result_row(
                acquisition_seed=acquisition_seed,
                training_seed=training_seed,
                strategy=strategy,
                round_idx=round_idx,
                labeled_budget=len(labeled_df),
                dev_eval_size=len(dev_df),
            ),
            {"strategy": strategy, "round": round_idx, "dry_run": True},
            pd.DataFrame(),
        )
    yaml_path, build_log = base.v6.build_yolo_dataset(labeled_df, dev_df, dataset_root)
    build_log.insert(0, "acquisition_seed", acquisition_seed)
    build_log.insert(1, "strategy", strategy)
    build_log.insert(2, "round", round_idx)
    row, runtime = base.train_row(
        yaml_path=yaml_path,
        save_dir=save_dir,
        strategy=strategy,
        acquisition_seed=acquisition_seed,
        training_seed=training_seed,
        round_idx=round_idx,
        labeled_budget=len(labeled_df),
        dev_eval_size=len(dev_df),
    )
    return row, runtime, build_log


def recover_if_possible(row: dict[str, Any], *, strategy: str, seed: int, training_seed: int, round_idx: int, save_dir: Path) -> pd.DataFrame:
    if row.get("train_status") != "success":
        return pd.DataFrame(
            [
                {
                    "acquisition_seed": seed,
                    "training_seed": training_seed,
                    "strategy": strategy,
                    "round": round_idx,
                    "status": "skipped",
                    "note": str(row.get("train_status", "not_success")),
                }
            ]
        )
    return base.recover_per_class_metrics(
        result_row=row,
        strategy_label=strategy,
        acquisition_seed=seed,
        training_seed=training_seed,
        round_idx=round_idx,
        save_dir=save_dir,
    )


def summarize_outputs(results_df: pd.DataFrame, per_class_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = base.add_f1(results_df)
    comparison_rows = []
    for (seed, round_idx), sub in results.groupby(["acquisition_seed", "round"], dropna=False):
        random = sub[sub["strategy"].eq(RANDOM_STRATEGY)]
        v10c_rows = sub[sub["strategy"].eq(V10C_STRATEGY)]
        if random.empty or v10c_rows.empty:
            continue
        r = random.iloc[0]
        v = v10c_rows.iloc[0]
        row = {"acquisition_seed": int(seed), "round": int(round_idx), "budget": int(v.get("labeled_budget", np.nan))}
        for metric in ["map50", "map5095", "precision", "recall", "f1"]:
            row[f"random_{metric}"] = base.safe_float(r.get(metric))
            row[f"v10c24_{metric}"] = base.safe_float(v.get(metric))
            row[f"v10c24_minus_random_{metric}"] = base.safe_float(v.get(metric)) - base.safe_float(r.get(metric))
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)

    aulc_rows = []
    for seed, sub in results[results["strategy"].isin([ROUND0_STRATEGY, RANDOM_STRATEGY, V10C_STRATEGY])].groupby("acquisition_seed", dropna=False):
        round0 = sub[sub["strategy"].eq(ROUND0_STRATEGY)]
        for strategy in [RANDOM_STRATEGY, V10C_STRATEGY]:
            curve = pd.concat([round0, sub[sub["strategy"].eq(strategy)]], ignore_index=True, sort=False)
            curve = curve[["round", "labeled_budget", "map5095", "recall", "precision", "f1"]].copy()
            curve = curve.dropna(subset=["labeled_budget"]).sort_values("labeled_budget")
            row = {"acquisition_seed": int(seed), "strategy": strategy, "num_points": int(len(curve))}
            if len(curve) >= 2:
                x = pd.to_numeric(curve["labeled_budget"], errors="coerce").to_numpy(dtype=float)
                denom = max(1e-12, float(x.max() - x.min()))
                for metric in ["map5095", "recall", "precision", "f1"]:
                    y = pd.to_numeric(curve[metric], errors="coerce").to_numpy(dtype=float)
                    mask = np.isfinite(x) & np.isfinite(y)
                    row[f"aulc_{metric}"] = trapz_compat(y[mask], x[mask]) if mask.sum() >= 2 else np.nan
                    row[f"naulc_{metric}"] = row[f"aulc_{metric}"] / denom if np.isfinite(row[f"aulc_{metric}"]) else np.nan
            aulc_rows.append(row)
    aulc = pd.DataFrame(aulc_rows)

    class_delta_rows = []
    good = (
        per_class_df[pd.to_numeric(per_class_df.get("round"), errors="coerce").eq(2)].copy()
        if len(per_class_df) and {"acquisition_seed", "class_name", "strategy", "ap5095"}.issubset(per_class_df.columns)
        else pd.DataFrame()
    )
    if len(good):
        for (seed, cls), sub in good.groupby(["acquisition_seed", "class_name"], dropna=False):
            random = sub[sub["strategy"].eq(RANDOM_STRATEGY)]
            v10c_rows = sub[sub["strategy"].eq(V10C_STRATEGY)]
            if random.empty or v10c_rows.empty:
                continue
            class_delta_rows.append(
                {
                    "acquisition_seed": int(seed),
                    "class_name": cls,
                    "random_ap5095_round2": base.safe_float(random.iloc[0].get("ap5095")),
                    "v10c24_ap5095_round2": base.safe_float(v10c_rows.iloc[0].get("ap5095")),
                    "v10c24_minus_random_ap5095_round2": base.safe_float(v10c_rows.iloc[0].get("ap5095")) - base.safe_float(random.iloc[0].get("ap5095")),
                }
            )
    return comparison, aulc, pd.DataFrame(class_delta_rows)


def evaluate_gate(comparison: pd.DataFrame, class_delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    round2 = comparison[pd.to_numeric(comparison.get("round"), errors="coerce").eq(2)].copy() if len(comparison) else pd.DataFrame()
    round1 = comparison[pd.to_numeric(comparison.get("round"), errors="coerce").eq(1)].copy() if len(comparison) else pd.DataFrame()
    for seed, r2_sub in round2.groupby("acquisition_seed", dropna=False):
        r2 = r2_sub.iloc[0]
        r1_sub = round1[round1["acquisition_seed"].eq(seed)]
        r1 = r1_sub.iloc[0] if len(r1_sub) else None
        budget_gain_advantage = np.nan
        if r1 is not None:
            random_gain = base.safe_float(r2.get("random_map5095")) - base.safe_float(r1.get("random_map5095"))
            v10c_gain = base.safe_float(r2.get("v10c24_map5095")) - base.safe_float(r1.get("v10c24_map5095"))
            budget_gain_advantage = v10c_gain - random_gain
        cls = class_delta[class_delta["acquisition_seed"].eq(seed)] if len(class_delta) else pd.DataFrame()
        class_wins = int((pd.to_numeric(cls.get("v10c24_minus_random_ap5095_round2"), errors="coerce") >= 0).sum()) if len(cls) else 0
        worst_class_drop = float(pd.to_numeric(cls.get("v10c24_minus_random_ap5095_round2"), errors="coerce").min()) if len(cls) else np.nan
        checks = {
            "round2_map5095_positive": base.safe_float(r2.get("v10c24_minus_random_map5095")) > 0,
            "round2_recall_nonnegative": base.safe_float(r2.get("v10c24_minus_random_recall")) >= 0,
            "round2_f1_nonnegative": base.safe_float(r2.get("v10c24_minus_random_f1")) >= 0,
            "budget90_to_120_gain_beats_random": budget_gain_advantage > 0 if np.isfinite(budget_gain_advantage) else False,
            "class_wins_at_least_4_of_6": class_wins >= 4,
            "no_class_drop_worse_than_minus_0p05": worst_class_drop >= -0.05 if np.isfinite(worst_class_drop) else False,
        }
        rows.append(
            {
                "acquisition_seed": int(seed),
                **checks,
                "passed_checks": int(sum(bool(v) for v in checks.values())),
                "total_checks": int(len(checks)),
                "gate_pass": bool(all(checks.values())),
                "budget_gain_advantage_map5095": budget_gain_advantage,
                "class_wins": class_wins,
                "worst_class_drop": worst_class_drop,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(save_dir: Path, config: dict[str, Any], accum: dict[str, list[Any]]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    results_df = base.add_f1(pd.DataFrame(accum["results_rows"]))
    selected_df = pd.concat(accum["selected_rows"], ignore_index=True, sort=False) if accum["selected_rows"] else pd.DataFrame()
    cumulative_df = pd.concat(accum["cumulative_rows"], ignore_index=True, sort=False) if accum["cumulative_rows"] else pd.DataFrame()
    build_df = pd.concat(accum["build_logs"], ignore_index=True, sort=False) if accum["build_logs"] else pd.DataFrame()
    per_class_df = pd.concat(accum["per_class_rows"], ignore_index=True, sort=False) if accum["per_class_rows"] else pd.DataFrame()
    split_df = pd.concat(accum["split_logs"], ignore_index=True, sort=False) if accum["split_logs"] else pd.DataFrame()
    overlap_df = pd.DataFrame(accum["overlap_rows"])
    comparison, aulc, class_delta = summarize_outputs(results_df, per_class_df)
    gate = evaluate_gate(comparison, class_delta)

    (save_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(accum["seed_registry"]).to_csv(save_dir / "seed_registry.csv", index=False, encoding="utf-8-sig")
    results_df.to_csv(save_dir / "all_round_results.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(save_dir / "all_selected_samples_by_round.csv", index=False, encoding="utf-8-sig")
    cumulative_df.to_csv(save_dir / "cumulative_labeled_sets_by_round.csv", index=False, encoding="utf-8-sig")
    build_df.to_csv(save_dir / "dataset_build_log.csv", index=False, encoding="utf-8-sig")
    per_class_df.to_csv(save_dir / "per_class_metrics_by_round.csv", index=False, encoding="utf-8-sig")
    split_df.to_csv(save_dir / "protocol_split_registry.csv", index=False, encoding="utf-8-sig")
    overlap_df.to_csv(save_dir / "selection_overlap_by_seed_round.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(accum["runtime_rows"]).to_csv(save_dir / "runtime_profile.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(save_dir / "roundwise_random_v10c24_comparison.csv", index=False, encoding="utf-8-sig")
    aulc.to_csv(save_dir / "aulc_summary.csv", index=False, encoding="utf-8-sig")
    class_delta.to_csv(save_dir / "round2_per_class_v10c24_minus_random.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(save_dir / "round2_scale_gate.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# V10c24 round2 scale smoke",
        "",
        f"- Experiment: `{config['experiment_id']}`",
        f"- Seeds: {config['acquisition_seeds']}",
        f"- Dry run: {config['dry_run']}",
        "- Final test used: `False`",
        "- Methods: `GTFreeRandom` vs `DetectorRecallGuardDINOInstanceV10c` only",
        "",
        "## Roundwise comparison",
        "",
        table_md(comparison),
        "",
        "## AULC / nAULC",
        "",
        table_md(aulc),
        "",
        "## Round2 gate",
        "",
        table_md(gate),
        "",
        "## Round2 per-class AP50-95",
        "",
        table_md(class_delta),
        "",
        "## Guardrails",
        "",
        "- Round2 V10c24 re-scores the remaining pool with the V10c24 Round1 detector.",
        "- Round2 Random samples only from its own remaining pool.",
        "- Final test is written as locked protocol metadata only and is never evaluated.",
    ]
    (save_dir / "v10c24_round2_scale_smoke_summary.md").write_text("\n".join(lines), encoding="utf-8")


def overlap_row(seed: int, round_idx: int, random_query: pd.DataFrame, v10c_query: pd.DataFrame) -> dict[str, Any]:
    random_ids = sample_ids(random_query)
    v10c_ids = sample_ids(v10c_query)
    union = random_ids | v10c_ids
    return {
        "acquisition_seed": seed,
        "round": round_idx,
        "random_size": len(random_ids),
        "v10c24_size": len(v10c_ids),
        "overlap_count": len(random_ids & v10c_ids),
        "jaccard": len(random_ids & v10c_ids) / len(union) if union else np.nan,
    }


def run_one_seed(*, seed: int, save_dir: Path, dataset_root: Path, manifest: pd.DataFrame, config: dict[str, Any], accum: dict[str, list[Any]]) -> None:
    training_seed = 1000 + seed
    dry_run = bool(config["dry_run"])
    t0 = time.perf_counter()
    status = "started"
    error = None
    try:
        print("=" * 100)
        print(f"[SEED {seed}] V10c24 round2 scale smoke training_seed={training_seed} dry_run={dry_run}")
        pool_df, dev_df, final_df, unused_df = base.v10smoke.build_v10_protocol_split(
            manifest,
            seed=seed,
            pool_per_class=int(config["pool_per_class"]),
            dev_per_class=int(config["dev_per_class"]),
            final_per_class=int(config["final_per_class"]),
        )
        overlap_checks = base.validate_split(pool_df, dev_df, final_df)
        split_df = pd.DataFrame(
            [
                {"acquisition_seed": seed, "split": "pool", "num_images": len(pool_df), **overlap_checks},
                {"acquisition_seed": seed, "split": "development_eval", "num_images": len(dev_df), **overlap_checks},
                {"acquisition_seed": seed, "split": "final_test_LOCKED_UNUSED", "num_images": len(final_df), **overlap_checks},
                {"acquisition_seed": seed, "split": "unused_reserve", "num_images": len(unused_df), **overlap_checks},
            ]
        )
        accum["split_logs"].append(split_df)
        pool_df.to_csv(save_dir / f"seed{seed}_acquisition_pool.csv", index=False, encoding="utf-8-sig")
        dev_df.to_csv(save_dir / f"seed{seed}_development_eval.csv", index=False, encoding="utf-8-sig")
        final_df.to_csv(save_dir / f"seed{seed}_final_test_LOCKED_UNUSED.csv", index=False, encoding="utf-8-sig")
        unused_df.to_csv(save_dir / f"seed{seed}_unused_reserve.csv", index=False, encoding="utf-8-sig")

        seed_artifact_dir = save_dir / f"seed{seed}_artifacts"
        seed_artifact_dir.mkdir(parents=True, exist_ok=True)
        embedding_dir, embedding_lookup, embedding_config = base.v10smoke.load_or_build_embeddings(pool_df, seed_artifact_dir)

        initial_df = base.sample_initial_labeled_set(
            pool_df,
            initial_seed_size=int(config["initial_seed_size"]),
            acquisition_seed=seed,
        ).reset_index(drop=True)
        current_pool = pool_df[~pool_df["sample_id"].isin(initial_df["sample_id"])].copy().reset_index(drop=True)
        seed_dataset_root = dataset_root / f"seed{seed}"

        row0, runtime0, build0 = train_round(
            labeled_df=initial_df,
            dev_df=dev_df,
            dataset_root=seed_dataset_root / ROUND0_STRATEGY,
            save_dir=save_dir,
            strategy=ROUND0_STRATEGY,
            acquisition_seed=seed,
            training_seed=training_seed,
            round_idx=0,
            dry_run=dry_run,
        )
        accum["results_rows"].append(row0)
        accum["runtime_rows"].append(runtime0)
        if len(build0):
            accum["build_logs"].append(build0)
        accum["per_class_rows"].append(recover_if_possible(row0, strategy=ROUND0_STRATEGY, seed=seed, training_seed=training_seed, round_idx=0, save_dir=save_dir))

        round0_weight = Path(str(row0.get("train_run_dir", ""))) / "weights" / "best.pt"

        random_r1 = base.v10smoke.select_random(current_pool, int(config["query_size"]), seed).reset_index(drop=True)
        if dry_run:
            v10c_r1 = current_pool.sort_values("sample_id", kind="mergesort").head(int(config["query_size"])).copy().reset_index(drop=True)
            v10c_r1["v10c_phase"] = "dry_run_placeholder_round1"
        else:
            v10c_r1, _ = select_v10c_from_detector(
                detector_weights=round0_weight,
                remaining_pool=current_pool,
                labeled_reference=initial_df,
                embedding_lookup=embedding_lookup,
                save_dir=save_dir,
                seed=seed,
                round_idx=1,
                config=config,
            )
        validate_no_overlap(initial_df, random_r1, dev_df, expected_query=int(config["query_size"]), label=f"seed{seed}_random_r1")
        validate_no_overlap(initial_df, v10c_r1, dev_df, expected_query=int(config["query_size"]), label=f"seed{seed}_v10c_r1")

        random_labeled_r1 = pd.concat([initial_df, random_r1], ignore_index=True, sort=False).drop_duplicates("sample_id", keep="first")
        v10c_labeled_r1 = pd.concat([initial_df, v10c_r1], ignore_index=True, sort=False).drop_duplicates("sample_id", keep="first")

        selected_frames = [
            add_selection_metadata(initial_df, acquisition_seed=seed, training_seed=training_seed, strategy=ROUND0_STRATEGY, round_idx=0, selection_type="shared_initial_random"),
            add_selection_metadata(random_r1, acquisition_seed=seed, training_seed=training_seed, strategy=RANDOM_STRATEGY, round_idx=1, selection_type=RANDOM_STRATEGY),
            add_selection_metadata(v10c_r1, acquisition_seed=seed, training_seed=training_seed, strategy=V10C_STRATEGY, round_idx=1, selection_type=V10C_STRATEGY, dry_run_placeholder=dry_run),
        ]
        cumulative_frames = [
            add_selection_metadata(initial_df, acquisition_seed=seed, training_seed=training_seed, strategy=ROUND0_STRATEGY, round_idx=0, selection_type="cumulative_labeled"),
            add_selection_metadata(random_labeled_r1, acquisition_seed=seed, training_seed=training_seed, strategy=RANDOM_STRATEGY, round_idx=1, selection_type="cumulative_labeled"),
            add_selection_metadata(v10c_labeled_r1, acquisition_seed=seed, training_seed=training_seed, strategy=V10C_STRATEGY, round_idx=1, selection_type="cumulative_labeled", dry_run_placeholder=dry_run),
        ]
        accum["overlap_rows"].append(overlap_row(seed, 1, random_r1, v10c_r1))

        for strategy, labeled in [(RANDOM_STRATEGY, random_labeled_r1), (V10C_STRATEGY, v10c_labeled_r1)]:
            row, runtime, build = train_round(
                labeled_df=labeled,
                dev_df=dev_df,
                dataset_root=seed_dataset_root / strategy / "round_1",
                save_dir=save_dir,
                strategy=strategy,
                acquisition_seed=seed,
                training_seed=training_seed,
                round_idx=1,
                dry_run=dry_run,
            )
            accum["results_rows"].append(row)
            accum["runtime_rows"].append(runtime)
            if len(build):
                accum["build_logs"].append(build)
            accum["per_class_rows"].append(recover_if_possible(row, strategy=strategy, seed=seed, training_seed=training_seed, round_idx=1, save_dir=save_dir))

        random_remaining_r2 = pool_df[~pool_df["sample_id"].isin(sample_ids(random_labeled_r1))].copy().reset_index(drop=True)
        v10c_remaining_r2 = pool_df[~pool_df["sample_id"].isin(sample_ids(v10c_labeled_r1))].copy().reset_index(drop=True)
        random_r2 = base.v10smoke.select_random(random_remaining_r2, int(config["query_size"]), seed + 2000).reset_index(drop=True)
        if dry_run:
            v10c_r2 = v10c_remaining_r2.sort_values("sample_id", kind="mergesort").head(int(config["query_size"])).copy().reset_index(drop=True)
            v10c_r2["v10c_phase"] = "dry_run_placeholder_round2"
        else:
            v10c_r1_row = [
                r
                for r in accum["results_rows"]
                if r.get("acquisition_seed") == seed and r.get("strategy") == V10C_STRATEGY and r.get("round") == 1
            ][-1]
            v10c_round1_weight = Path(str(v10c_r1_row.get("train_run_dir", ""))) / "weights" / "best.pt"
            v10c_r2, _ = select_v10c_from_detector(
                detector_weights=v10c_round1_weight,
                remaining_pool=v10c_remaining_r2,
                labeled_reference=v10c_labeled_r1,
                embedding_lookup=embedding_lookup,
                save_dir=save_dir,
                seed=seed,
                round_idx=2,
                config=config,
            )
        validate_no_overlap(random_labeled_r1, random_r2, dev_df, expected_query=int(config["query_size"]), label=f"seed{seed}_random_r2")
        validate_no_overlap(v10c_labeled_r1, v10c_r2, dev_df, expected_query=int(config["query_size"]), label=f"seed{seed}_v10c_r2")

        random_labeled_r2 = pd.concat([random_labeled_r1, random_r2], ignore_index=True, sort=False).drop_duplicates("sample_id", keep="first")
        v10c_labeled_r2 = pd.concat([v10c_labeled_r1, v10c_r2], ignore_index=True, sort=False).drop_duplicates("sample_id", keep="first")
        selected_frames.extend(
            [
                add_selection_metadata(random_r2, acquisition_seed=seed, training_seed=training_seed, strategy=RANDOM_STRATEGY, round_idx=2, selection_type=RANDOM_STRATEGY),
                add_selection_metadata(v10c_r2, acquisition_seed=seed, training_seed=training_seed, strategy=V10C_STRATEGY, round_idx=2, selection_type=V10C_STRATEGY, dry_run_placeholder=dry_run),
            ]
        )
        cumulative_frames.extend(
            [
                add_selection_metadata(random_labeled_r2, acquisition_seed=seed, training_seed=training_seed, strategy=RANDOM_STRATEGY, round_idx=2, selection_type="cumulative_labeled"),
                add_selection_metadata(v10c_labeled_r2, acquisition_seed=seed, training_seed=training_seed, strategy=V10C_STRATEGY, round_idx=2, selection_type="cumulative_labeled", dry_run_placeholder=dry_run),
            ]
        )
        accum["overlap_rows"].append(overlap_row(seed, 2, random_r2, v10c_r2))

        for strategy, labeled in [(RANDOM_STRATEGY, random_labeled_r2), (V10C_STRATEGY, v10c_labeled_r2)]:
            row, runtime, build = train_round(
                labeled_df=labeled,
                dev_df=dev_df,
                dataset_root=seed_dataset_root / strategy / "round_2",
                save_dir=save_dir,
                strategy=strategy,
                acquisition_seed=seed,
                training_seed=training_seed,
                round_idx=2,
                dry_run=dry_run,
            )
            accum["results_rows"].append(row)
            accum["runtime_rows"].append(runtime)
            if len(build):
                accum["build_logs"].append(build)
            accum["per_class_rows"].append(recover_if_possible(row, strategy=strategy, seed=seed, training_seed=training_seed, round_idx=2, save_dir=save_dir))

        accum["selected_rows"].extend(selected_frames)
        accum["cumulative_rows"].extend(cumulative_frames)
        status = "success"
        accum["seed_registry"].append(
            {
                "acquisition_seed": seed,
                "training_seed": training_seed,
                "status": status,
                "error": error,
                "elapsed_sec": time.perf_counter() - t0,
                "embedding_dir": str(embedding_dir),
                "embedding_backend": embedding_config.get("backend"),
                "dry_run": dry_run,
                "final_test_used": False,
                "expected_yolo_trainings": 5 if not dry_run else 0,
            }
        )
    except Exception as exc:
        traceback.print_exc()
        status = "failed"
        error = repr(exc)
        accum["seed_registry"].append(
            {
                "acquisition_seed": seed,
                "training_seed": training_seed,
                "status": status,
                "error": error,
                "elapsed_sec": time.perf_counter() - t0,
                "dry_run": dry_run,
                "final_test_used": False,
            }
        )
    finally:
        write_outputs(save_dir, config, accum)
        print(f"[SEED {seed}] status={status} elapsed={time.perf_counter() - t0:.1f}s")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"v10c24_round2_scale_smoke_{timestamp}"
    dataset_root = DATASETS_ROOT / save_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    seeds = parse_seed_list()
    query_size = int_env("AL_QUERY_SIZE", 30)
    recall_guard_size = int_env("AL_V10C_RECALL_GUARD_SIZE", 6)
    core_size = int_env("AL_V10C_CORE_SIZE", max(0, query_size - recall_guard_size))
    rounds = int_env("AL_ROUNDS", 2)
    if rounds != 2:
        raise ValueError("This smoke runner is intentionally fixed to AL_ROUNDS=2.")
    dry_run = parse_bool_env("AL_DRY_RUN_ONLY", True)
    manifest = base.v10smoke.ensure_neu_manifest()

    config: dict[str, Any] = {
        "experiment_id": save_dir.name,
        "experiment_label": "V10c24 Random-vs-V10c budget120 round2 scale smoke",
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "save_dir": str(save_dir),
        "dataset_root": str(dataset_root),
        "acquisition_seeds": seeds,
        "training_seed_rule": "training_seed = 1000 + acquisition_seed",
        "final_test_used": False,
        "method_weights_frozen": True,
        "new_selector_development": False,
        "strategies": [RANDOM_STRATEGY, V10C_STRATEGY],
        "pool_per_class": int_env("AL_V10_POOL_PER_CLASS", 150),
        "dev_per_class": int_env("AL_V10_DEV_PER_CLASS", 50),
        "final_per_class": int_env("AL_V10_FINAL_PER_CLASS", 50),
        "initial_seed_size": int_env("AL_INITIAL_SEED_SIZE", 60),
        "query_size": query_size,
        "rounds": rounds,
        "budgets": [int_env("AL_INITIAL_SEED_SIZE", 60) + query_size * r for r in range(rounds + 1)],
        "v10c_weights": v10c.read_v10c_weights(),
        "recall_guard_weights": v10c.read_guard_weights(),
        "v10c_core_size": core_size,
        "v10c_recall_guard_size": recall_guard_size,
        "core_constraints": {
            "max_no_box": int_env("AL_V10C_CORE_MAX_NO_BOX", 0),
            "min_pseudo_boxes": int_env("AL_V10C_CORE_MIN_PSEUDO_BOXES", 2),
            "max_per_pred_class": int_env("AL_V10C_CORE_MAX_PER_PRED_CLASS", 2),
        },
        "guard_constraints": {
            "max_no_box": int_env("AL_V10C_GUARD_MAX_NO_BOX", 2),
            "max_per_pred_class": int_env("AL_V10C_GUARD_MAX_PER_PRED_CLASS", 3),
        },
        "candidate_fraction": float_env("AL_V9_CANDIDATE_FRACTION", 1.0),
        "predict_conf": float_env("AL_V9_PREDICT_CONF", 0.05),
        "predict_iou": float_env("AL_V9_PREDICT_IOU", 0.70),
        "model": os.environ.get("AL_YOLO_MODEL_NAME", "yolov8n.pt"),
        "epochs": int_env("AL_EPOCHS_PER_ROUND", 100),
        "patience": int_env("AL_YOLO_PATIENCE", int_env("AL_EPOCHS_PER_ROUND", 100)),
        "imgsz": int_env("AL_IMGSZ", 640),
        "batch": int_env("AL_BATCH_SIZE", 8),
        "workers": int_env("AL_WORKERS", 4),
        "cache": os.environ.get("AL_YOLO_CACHE", "false"),
        "plots": os.environ.get("AL_YOLO_PLOTS", "false"),
        "device": os.environ.get("AL_YOLO_DEVICE", "0"),
        "embedding_backend": os.environ.get("AL_EMBEDDING_BACKEND", "dinov2"),
        "dry_run": dry_run,
        "runner_sha256": base.file_sha256(Path(__file__).resolve()),
        "git_commit": base.git_commit(),
        "git_dirty": base.git_dirty(),
    }
    accum: dict[str, list[Any]] = {
        "seed_registry": [],
        "results_rows": [],
        "selected_rows": [],
        "cumulative_rows": [],
        "build_logs": [],
        "runtime_rows": [],
        "per_class_rows": [],
        "split_logs": [],
        "overlap_rows": [],
    }

    print("=" * 100)
    print("[V10c24 round2 scale smoke]")
    print(f"Output dir: {save_dir}")
    print(f"Seeds     : {seeds}")
    print(f"Dry run   : {dry_run}")
    print("Strategies: Random vs frozen V10c24 only")
    print("Final test: locked / not evaluated")
    print("=" * 100)

    for seed in seeds:
        run_one_seed(seed=seed, save_dir=save_dir, dataset_root=dataset_root, manifest=manifest, config=config, accum=accum)

    registry = pd.DataFrame(accum["seed_registry"])
    completed = registry[registry["status"].astype(str).eq("success")]["acquisition_seed"].tolist() if len(registry) else []
    failed = registry[~registry["status"].astype(str).eq("success")]["acquisition_seed"].tolist() if len(registry) else []
    gate_path = save_dir / "round2_scale_gate.csv"
    gate = pd.read_csv(gate_path) if gate_path.exists() and gate_path.stat().st_size else pd.DataFrame()
    print("=" * 100)
    print("[DONE] V10c24 round2 scale smoke runner finished")
    print(f"Output dir: {save_dir}")
    print(f"Completed seeds: {completed}")
    print(f"Failed seeds   : {failed}")
    if len(gate):
        print(f"Round2 gate passed: {int(gate['gate_pass'].sum())}/{len(gate)}")
    print(f"Dry run={dry_run}")
    print("Final test used=False")
    print("New selector development=False")
    print("=" * 100)


if __name__ == "__main__":
    main()

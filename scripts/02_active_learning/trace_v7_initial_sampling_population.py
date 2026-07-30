"""Trace Stage-A vs full-curve initial sampling populations.

Diagnostic-only script. It does not modify existing experiment outputs, train
YOLO, evaluate checkpoints, regenerate DINO embeddings, or touch final_test_v7.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "02_active_learning"
sys.path.insert(0, str(SCRIPT_DIR))

import audit_detection_pipeline_v7 as audit_v7  # noqa: E402
import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402
import run_al_yolo_ablation_v7_full_curve as full_v7  # noqa: E402


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "v7_initial_sampling_population_trace"
DEFAULT_SCREEN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "active_learning_ablation_v7_visual_instance"
    / "v7_visual_instance_screening_20260711_193318"
)
DEFAULT_FULL_DIR = (
    PROJECT_ROOT
    / "runs"
    / "active_learning_ablation_v7_full_curve"
    / "v7_full_curve_20260711_211848"
)


def sha256_text(values: list[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_id_from_full_identity(df: pd.DataFrame) -> pd.Series:
    return df["resolved_image_path"].astype(str) + "::" + df["image_sha256"].astype(str)


def add_full_canonical_identity(df: pd.DataFrame) -> pd.DataFrame:
    image_index = full_v7.build_image_index()
    out = full_v7.add_sample_identity(df, image_index).copy()
    out["canonical_sample_id"] = canonical_id_from_full_identity(out)
    return out


def add_canonical_identity_without_reordering(df: pd.DataFrame) -> pd.DataFrame:
    image_index = full_v7.build_image_index()
    rows = []
    for _, row in df.iterrows():
        path = full_v7.resolve_image_path_fast(row, image_index)
        if path is None:
            rows.append((None, None, None))
            continue
        resolved = str(path.resolve())
        sha = full_v7.compute_file_sha256(path)
        rows.append((resolved, sha, f"{resolved}::{sha}"))
    out = df.copy()
    out["resolved_image_path"] = [x[0] for x in rows]
    out["image_sha256"] = [x[1] for x in rows]
    out["canonical_sample_id"] = [x[2] for x in rows]
    return out


def trace_dataframe(stage: str, df: pd.DataFrame, priority_csv: Path, priority_sha: str, step: str, note: str) -> dict[str, Any]:
    canonical = df["canonical_sample_id"].astype(str).tolist() if "canonical_sample_id" in df.columns else []
    return {
        "stage": stage,
        "step": step,
        "note": note,
        "priority_csv_abs_path": str(priority_csv.resolve()),
        "priority_csv_sha256": priority_sha,
        "row_count": len(df),
        "column_names": "|".join(map(str, df.columns.tolist())),
        "index_head": "|".join(map(str, df.index[:10].tolist())),
        "index_tail": "|".join(map(str, df.index[-10:].tolist())),
        "first20_dataset_image": "|".join(
            f"{r.dataset_type}::{r.image_name}" for r in df[["dataset_type", "image_name"]].head(20).itertuples(index=False)
        )
        if {"dataset_type", "image_name"}.issubset(df.columns)
        else "",
        "first20_canonical_ids": "|".join(canonical[:20]),
        "population_set_hash": sha256_text(sorted(canonical)) if canonical else "",
        "population_order_hash": sha256_text(canonical) if canonical else "",
        "canonical_id_sequence_sha256": sha256_text(canonical) if canonical else "",
        "row_order_sha256": sha256_text([f"{idx}::{cid}" for idx, cid in zip(df.index.astype(str), canonical)]) if canonical else "",
        "duplicate_canonical_count": int(df["canonical_sample_id"].duplicated().sum()) if "canonical_sample_id" in df.columns else 0,
        "duplicate_dataset_image_count": int((df["dataset_type"].astype(str) + "::" + df["image_name"].astype(str)).duplicated().sum())
        if {"dataset_type", "image_name"}.issubset(df.columns)
        else 0,
    }


def duplicate_keys(df: pd.DataFrame, key_col: str, stage: str) -> pd.DataFrame:
    if key_col not in df.columns:
        return pd.DataFrame()
    dup = df[df[key_col].duplicated(keep=False)].copy()
    if dup.empty:
        return pd.DataFrame(columns=["stage", "key_col", "duplicate_key", "count", "dataset_image_examples"])
    rows = []
    for key, sub in dup.groupby(key_col):
        rows.append(
            {
                "stage": stage,
                "key_col": key_col,
                "duplicate_key": key,
                "count": len(sub),
                "dataset_image_examples": "|".join((sub["dataset_type"].astype(str) + "::" + sub["image_name"].astype(str)).head(10)),
            }
        )
    return pd.DataFrame(rows)


def sampled_initial_from_csv(run_dir: Path, seed_col: str) -> list[str]:
    selected = pd.read_csv(run_dir / "all_selected_samples_by_round.csv")
    sub = selected[
        selected[seed_col].eq(42)
        & selected["strategy"].eq("GTFreeRandom")
        & selected["round"].eq(0)
    ].copy()
    if "canonical_sample_id" in sub.columns:
        return sub["canonical_sample_id"].astype(str).tolist()
    sub = add_canonical_identity_without_reordering(sub)
    return sub["canonical_sample_id"].astype(str).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--screen-dir", default=str(DEFAULT_SCREEN_DIR))
    ap.add_argument("--full-dir", default=str(DEFAULT_FULL_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--initial-size", type=int, default=15)
    args = ap.parse_args()

    out_dir = Path(args.output_root) / f"initial_sampling_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)

    priority_csv = audit_v7.resolve_priority_csv().resolve()
    priority_sha = file_sha256(priority_csv)

    raw = pd.read_csv(priority_csv)
    prepared = v6.prepare_priority_dataframe(raw.copy())

    # Stage-A actual path.
    stagea_filtered = prepared.copy()
    stagea_ordered = v6.stable_sample_order(stagea_filtered).reset_index(drop=True)
    stagea_population = add_canonical_identity_without_reordering(stagea_ordered)

    # Full-curve actual path.
    full_filtered = prepared.copy()
    full_population = add_full_canonical_identity(full_filtered)

    stagea_initial = stagea_population.sample(
        n=min(args.initial_size, len(stagea_population)),
        random_state=args.seed + 999,
        replace=False,
    )
    full_initial = full_population.sample(
        n=min(args.initial_size, len(full_population)),
        random_state=args.seed + 999,
        replace=False,
    )

    # Canonical-sort diagnostic reproduction.
    canonical_population = (
        stagea_population.copy()
        .drop_duplicates("canonical_sample_id")
        .sort_values("canonical_sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    stagea_canonical_resample = canonical_population.sample(
        n=min(args.initial_size, len(canonical_population)),
        random_state=args.seed + 999,
        replace=False,
    )
    full_canonical_population = (
        full_population.copy()
        .drop_duplicates("canonical_sample_id")
        .sort_values("canonical_sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    full_canonical_resample = full_canonical_population.sample(
        n=min(args.initial_size, len(full_canonical_population)),
        random_state=args.seed + 999,
        replace=False,
    )

    common_meta = {
        "exact_function": "pandas.DataFrame.sample",
        "exact_random_state": args.seed + 999,
        "rng_kind": "pandas random_state=int (MT19937/RandomState-compatible behavior)",
        "replace": False,
        "initial_seed_size": args.initial_size,
    }

    stagea_rows = [
        trace_dataframe("stage_a", raw, priority_csv, priority_sha, "read_csv_immediate", "pd.read_csv(priority_csv)"),
        trace_dataframe("stage_a", stagea_filtered, priority_csv, priority_sha, "after_filtering_prepare_priority_dataframe", "type coercion/required column validation; no row filtering expected"),
        trace_dataframe("stage_a", stagea_ordered, priority_csv, priority_sha, "after_stagea_stable_sample_order", "v6.stable_sample_order -> sort by dataset_type,image_name,image_path"),
        trace_dataframe("stage_a", stagea_population, priority_csv, priority_sha, "sampling_population", "sample_id/path canonicalization added without changing Stage-A order"),
        trace_dataframe("stage_a", stagea_initial, priority_csv, priority_sha, "sampled_initial_15", "diagnostic reproduction of Stage-A initial sampling"),
    ]
    full_rows = [
        trace_dataframe("full_curve", raw, priority_csv, priority_sha, "read_csv_immediate", "pd.read_csv(priority_csv)"),
        trace_dataframe("full_curve", full_filtered, priority_csv, priority_sha, "after_filtering_prepare_priority_dataframe", "type coercion/required column validation; no row filtering expected"),
        trace_dataframe("full_curve", full_population, priority_csv, priority_sha, "after_sample_id_path_canonicalization", "full_v7.add_sample_identity -> sort by sample_id"),
        trace_dataframe("full_curve", full_population, priority_csv, priority_sha, "sampling_population", "actual full-curve sampling population"),
        trace_dataframe("full_curve", full_initial, priority_csv, priority_sha, "sampled_initial_15", "diagnostic reproduction of full-curve initial sampling"),
    ]
    for row in stagea_rows + full_rows:
        row.update(common_meta)

    stagea_df = pd.DataFrame(stagea_rows)
    full_df = pd.DataFrame(full_rows)
    write_csv(stagea_df, out_dir / "stagea_sampling_population_trace.csv")
    write_csv(full_df, out_dir / "fullcurve_sampling_population_trace.csv")

    dup_df = pd.concat(
        [
            duplicate_keys(stagea_population, "canonical_sample_id", "stage_a"),
            duplicate_keys(full_population, "canonical_sample_id", "full_curve"),
            duplicate_keys(stagea_population.assign(dataset_image_key=stagea_population["dataset_type"].astype(str) + "::" + stagea_population["image_name"].astype(str)), "dataset_image_key", "stage_a"),
            duplicate_keys(full_population.assign(dataset_image_key=full_population["dataset_type"].astype(str) + "::" + full_population["image_name"].astype(str)), "dataset_image_key", "full_curve"),
        ],
        ignore_index=True,
    )
    write_csv(dup_df, out_dir / "sampling_duplicate_key_diagnostics.csv")

    stagea_ids = stagea_population["canonical_sample_id"].astype(str).tolist()
    full_ids = full_population["canonical_sample_id"].astype(str).tolist()
    set_same = set(stagea_ids) == set(full_ids)
    order_same = stagea_ids == full_ids
    sampled_same = set(stagea_initial["canonical_sample_id"]) == set(full_initial["canonical_sample_id"])
    stagea_canon_ids = stagea_canonical_resample["canonical_sample_id"].astype(str).tolist()
    full_canon_ids = full_canonical_resample["canonical_sample_id"].astype(str).tolist()
    canonical_resample_same = stagea_canon_ids == full_canon_ids

    comparison = pd.DataFrame(
        [
            {
                "stage_a_priority_csv_abs_path": str(priority_csv),
                "full_curve_priority_csv_abs_path": str(priority_csv),
                "same_priority_csv": True,
                "priority_csv_sha256": priority_sha,
                "stage_a_population_count": len(stagea_ids),
                "full_curve_population_count": len(full_ids),
                "stage_a_population_set_hash": sha256_text(sorted(stagea_ids)),
                "full_curve_population_set_hash": sha256_text(sorted(full_ids)),
                "population_set_hash_equal": set_same,
                "stage_a_population_order_hash": sha256_text(stagea_ids),
                "full_curve_population_order_hash": sha256_text(full_ids),
                "population_order_hash_equal": order_same,
                "stage_a_row_order_sha256": sha256_text([f"{idx}::{cid}" for idx, cid in zip(stagea_population.index.astype(str), stagea_ids)]),
                "full_curve_row_order_sha256": sha256_text([f"{idx}::{cid}" for idx, cid in zip(full_population.index.astype(str), full_ids)]),
                "sampled_initial_set_equal_actual_order": sampled_same,
                "canonical_sort_resampled_equal": canonical_resample_same,
                "stage_a_sampled_initial_ids": "|".join(stagea_initial["canonical_sample_id"].astype(str).tolist()),
                "full_curve_sampled_initial_ids": "|".join(full_initial["canonical_sample_id"].astype(str).tolist()),
                "canonical_sort_sampled_initial_ids": "|".join(stagea_canon_ids),
            }
        ]
    )
    write_csv(comparison, out_dir / "sampling_population_hash_comparison.csv")

    deterministic_rows = []
    for source, df in [
        ("stage_a_actual_order", stagea_initial),
        ("full_curve_actual_order", full_initial),
        ("stage_a_canonical_sort_resample", stagea_canonical_resample),
        ("full_curve_canonical_sort_resample", full_canonical_resample),
    ]:
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            deterministic_rows.append(
                {
                    "source": source,
                    "rank": rank,
                    "canonical_sample_id": row["canonical_sample_id"],
                    "dataset_type": row["dataset_type"],
                    "image_name": row["image_name"],
                    "resolved_image_path": row.get("resolved_image_path", ""),
                    "image_sha256": row.get("image_sha256", ""),
                }
            )
    det_df = pd.DataFrame(deterministic_rows)
    write_csv(det_df, out_dir / "deterministic_resampling_comparison.csv")

    # Existing result CSV cross-check: make sure diagnostic reproduction matches saved outputs.
    screen_dir = Path(args.screen_dir)
    full_dir = Path(args.full_dir)
    saved_rows = []
    for label, run_dir, seed_col, expected in [
        ("stage_a_saved", screen_dir, "seed", set(stagea_initial["canonical_sample_id"].astype(str))),
        ("full_curve_saved", full_dir, "acquisition_seed", set(full_initial["canonical_sample_id"].astype(str))),
    ]:
        selected = pd.read_csv(run_dir / "all_selected_samples_by_round.csv")
        sub = selected[
            selected[seed_col].eq(args.seed)
            & selected["strategy"].eq("GTFreeRandom")
            & selected["round"].eq(0)
        ].copy()
        sub = add_canonical_identity_without_reordering(sub)
        saved = set(sub["canonical_sample_id"].astype(str))
        saved_rows.append(
            {
                "source": label,
                "saved_initial_size": len(saved),
                "diagnostic_reproduction_size": len(expected),
                "overlap": len(saved & expected),
                "jaccard": len(saved & expected) / len(saved | expected) if (saved | expected) else 1.0,
                "matches_diagnostic_reproduction": saved == expected,
            }
        )
    write_csv(pd.DataFrame(saved_rows), out_dir / "saved_initial_reproduction_check.csv")

    if not set_same:
        verdict = "A. Different input population"
        explanation = "population_set_hash differs, indicating input pool/filtering/canonicalization changed the sample set."
    elif not order_same:
        verdict = "B. Same population, different row order"
        explanation = "population_set_hash is equal but population_order_hash differs. Pandas sample(random_state=int) is row-order dependent."
    elif not sampled_same:
        verdict = "C. Same population/order, different index or RNG behavior"
        explanation = "population and order are equal, but sampled sets differ."
    else:
        verdict = "D. Analysis identity artifact"
        explanation = "population/order/sample are equal under canonical identity; earlier mismatch was likely identity comparison error."

    report = [
        "# Initial set mismatch root cause\n\n",
        f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "No existing experiment outputs were modified. No training, checkpoint validation, DINO regeneration, or final-test evaluation was performed.\n\n",
        f"## Final verdict\n\n**{verdict}**\n\n{explanation}\n\n",
        "## Hash comparison\n\n",
        comparison.to_markdown(index=False),
        "\n\n## Answers\n\n",
        f"1. 같은 priority CSV를 사용했는가? **Yes.** `{priority_csv}` / SHA256 `{priority_sha}`.\n",
        f"2. pool의 sample 집합 자체가 같은가? **{'Yes' if set_same else 'No'}**.\n",
        f"3. row order만 달랐는가? **{'Yes' if set_same and not order_same else 'No'}**.\n",
        f"4. canonical sort 후 동일 seed로 같은 initial 15장이 재현되는가? **{'Yes' if canonical_resample_same else 'No'}**.\n",
        "5. 향후 모든 acquisition runner에서 sampling 전에 canonical sort가 필요한가? **Yes.** 특히 gate와 full-curve를 같은 selection plan에서 파생하려면 canonical_sample_id 기반 dedup/sort/reset_index를 공통 utility로 강제해야 한다.\n\n",
        "## Code implication\n\n",
        "If the verdict is B, the fix is to canonicalize and sort the pool before any `.sample()` call. The intended common rule is:\n\n",
        "```python\n",
        "pool = pool.copy()\n",
        "pool['canonical_sample_id'] = pool['resolved_image_path'].astype(str) + '::' + pool['image_sha256'].astype(str)\n",
        "pool = pool.drop_duplicates('canonical_sample_id').sort_values('canonical_sample_id').reset_index(drop=True)\n",
        "initial = pool.sample(n=initial_seed_size, random_state=acquisition_seed + 999, replace=False)\n",
        "```\n",
    ]
    (out_dir / "initial_set_mismatch_root_cause.md").write_text("".join(report), encoding="utf-8")

    print("=" * 100)
    print("[DONE] V7 initial sampling population trace")
    print(f"Output dir: {out_dir}")
    print(f"Verdict: {verdict}")
    print("No training. No checkpoint validation. No final-test evaluation.")
    print("=" * 100)


if __name__ == "__main__":
    main()


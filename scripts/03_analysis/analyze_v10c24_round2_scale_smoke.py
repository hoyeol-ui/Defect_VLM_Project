"""Analyze a completed V10c24 round2 scale-smoke run.

This script reads runner outputs only. It does not train and does not evaluate
the final test.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "runs" / "active_learning_v10c24_round2_scale_smoke"
RANDOM_STRATEGY = "GTFreeRandom"
V10C_STRATEGY = "DetectorRecallGuardDINOInstanceV10c"
ROUND0_STRATEGY = "__SHARED_ROUND0__"


def safe_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def f1(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall) or precision + recall == 0:
        return np.nan
    return float(2 * precision * recall / (precision + recall))


def table_md(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.head(max_rows)
    try:
        return shown.to_markdown(index=False)
    except Exception:
        return "```text\n" + shown.to_string(index=False) + "\n```"


def trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def latest_run() -> Path:
    candidates = [p for p in RUNS_ROOT.glob("v10c24_round2_scale_smoke_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No runs found under {RUNS_ROOT}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_csv(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def add_f1(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "f1" not in out.columns or out["f1"].isna().all():
        out["f1"] = [f1(safe_float(p), safe_float(r)) for p, r in zip(out.get("precision", []), out.get("recall", []))]
    return out


def recompute_roundwise(results: pd.DataFrame) -> pd.DataFrame:
    results = add_f1(results)
    rows = []
    for (seed, round_idx), sub in results.groupby(["acquisition_seed", "round"], dropna=False):
        random = sub[sub["strategy"].astype(str).eq(RANDOM_STRATEGY)]
        v10c = sub[sub["strategy"].astype(str).eq(V10C_STRATEGY)]
        if random.empty or v10c.empty:
            continue
        r = random.iloc[0]
        v = v10c.iloc[0]
        row = {"acquisition_seed": int(seed), "round": int(round_idx), "budget": int(safe_float(v.get("labeled_budget")))}
        for metric in ["map50", "map5095", "precision", "recall", "f1"]:
            row[f"random_{metric}"] = safe_float(r.get(metric))
            row[f"v10c24_{metric}"] = safe_float(v.get(metric))
            row[f"v10c24_minus_random_{metric}"] = safe_float(v.get(metric)) - safe_float(r.get(metric))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_aulc(results: pd.DataFrame) -> pd.DataFrame:
    results = add_f1(results)
    rows = []
    for seed, sub in results.groupby("acquisition_seed", dropna=False):
        r0 = sub[sub["strategy"].astype(str).eq(ROUND0_STRATEGY)]
        for strategy in [RANDOM_STRATEGY, V10C_STRATEGY]:
            curve = pd.concat([r0, sub[sub["strategy"].astype(str).eq(strategy)]], ignore_index=True, sort=False)
            curve = curve.dropna(subset=["labeled_budget"]).sort_values("labeled_budget")
            row = {"acquisition_seed": int(seed), "strategy": strategy, "num_points": int(len(curve))}
            if len(curve) >= 2:
                x = pd.to_numeric(curve["labeled_budget"], errors="coerce").to_numpy(dtype=float)
                denom = max(1e-12, float(np.nanmax(x) - np.nanmin(x)))
                for metric in ["map5095", "recall", "precision", "f1"]:
                    y = pd.to_numeric(curve[metric], errors="coerce").to_numpy(dtype=float)
                    mask = np.isfinite(x) & np.isfinite(y)
                    row[f"aulc_{metric}"] = trapz_compat(y[mask], x[mask]) if mask.sum() >= 2 else np.nan
                    row[f"naulc_{metric}"] = row[f"aulc_{metric}"] / denom if np.isfinite(row[f"aulc_{metric}"]) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def compute_gate(roundwise: pd.DataFrame, per_class_delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    round2 = roundwise[pd.to_numeric(roundwise.get("round"), errors="coerce").eq(2)].copy() if len(roundwise) else pd.DataFrame()
    round1 = roundwise[pd.to_numeric(roundwise.get("round"), errors="coerce").eq(1)].copy() if len(roundwise) else pd.DataFrame()
    for seed, sub in round2.groupby("acquisition_seed", dropna=False):
        r2 = sub.iloc[0]
        r1_sub = round1[round1["acquisition_seed"].eq(seed)]
        r1 = r1_sub.iloc[0] if len(r1_sub) else None
        gain_adv = np.nan
        if r1 is not None:
            gain_adv = (
                safe_float(r2.get("v10c24_map5095"))
                - safe_float(r1.get("v10c24_map5095"))
                - (safe_float(r2.get("random_map5095")) - safe_float(r1.get("random_map5095")))
            )
        cls = per_class_delta[per_class_delta["acquisition_seed"].eq(seed)] if len(per_class_delta) else pd.DataFrame()
        cls_delta = pd.to_numeric(cls.get("v10c24_minus_random_ap5095_round2"), errors="coerce") if len(cls) else pd.Series(dtype=float)
        checks = {
            "round2_map5095_positive": safe_float(r2.get("v10c24_minus_random_map5095")) > 0,
            "round2_recall_nonnegative": safe_float(r2.get("v10c24_minus_random_recall")) >= 0,
            "round2_f1_nonnegative": safe_float(r2.get("v10c24_minus_random_f1")) >= 0,
            "budget90_to_120_gain_beats_random": gain_adv > 0 if np.isfinite(gain_adv) else False,
            "class_wins_at_least_4_of_6": int((cls_delta >= 0).sum()) >= 4 if len(cls_delta) else False,
            "no_class_drop_worse_than_minus_0p05": float(cls_delta.min()) >= -0.05 if len(cls_delta.dropna()) else False,
        }
        rows.append(
            {
                "acquisition_seed": int(seed),
                **checks,
                "passed_checks": int(sum(bool(v) for v in checks.values())),
                "total_checks": len(checks),
                "gate_pass": bool(all(checks.values())),
                "budget_gain_advantage_map5095": gain_adv,
                "class_wins": int((cls_delta >= 0).sum()) if len(cls_delta) else 0,
                "worst_class_drop": float(cls_delta.min()) if len(cls_delta.dropna()) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def per_class_round2_delta(per_class: pd.DataFrame) -> pd.DataFrame:
    if per_class.empty or not {"acquisition_seed", "class_name", "strategy", "ap5095"}.issubset(per_class.columns):
        return pd.DataFrame()
    good = per_class[pd.to_numeric(per_class.get("round"), errors="coerce").eq(2)].copy()
    rows = []
    for (seed, cls), sub in good.groupby(["acquisition_seed", "class_name"], dropna=False):
        random = sub[sub["strategy"].astype(str).eq(RANDOM_STRATEGY)]
        v10c = sub[sub["strategy"].astype(str).eq(V10C_STRATEGY)]
        if random.empty or v10c.empty:
            continue
        rows.append(
            {
                "acquisition_seed": int(seed),
                "class_name": cls,
                "random_ap5095_round2": safe_float(random.iloc[0].get("ap5095")),
                "v10c24_ap5095_round2": safe_float(v10c.iloc[0].get("ap5095")),
                "v10c24_minus_random_ap5095_round2": safe_float(v10c.iloc[0].get("ap5095")) - safe_float(random.iloc[0].get("ap5095")),
            }
        )
    return pd.DataFrame(rows)


def write_report(run_dir: Path, out_dir: Path, roundwise: pd.DataFrame, aulc: pd.DataFrame, gate: pd.DataFrame, per_class_delta: pd.DataFrame) -> None:
    r2 = roundwise[pd.to_numeric(roundwise.get("round"), errors="coerce").eq(2)].copy() if len(roundwise) else pd.DataFrame()
    mean_diff = pd.to_numeric(r2.get("v10c24_minus_random_map5095"), errors="coerce").mean() if len(r2) else np.nan
    gate_passes = int(gate["gate_pass"].sum()) if len(gate) and "gate_pass" in gate.columns else 0
    lines = [
        "# V10c24 round2 scale-smoke analysis",
        "",
        f"- Source run: `{run_dir}`",
        "- Training performed by this script: `False`",
        "- Final test used: `False`",
        "",
        "## Verdict",
        "",
        f"- Round2 mean V10c24-Random mAP50-95 delta: `{mean_diff:.6f}`",
        f"- Gate pass count: `{gate_passes}/{len(gate)}`",
        "",
        "## Roundwise comparison",
        "",
        table_md(roundwise),
        "",
        "## AULC / nAULC",
        "",
        table_md(aulc),
        "",
        "## Gate",
        "",
        table_md(gate),
        "",
        "## Round2 per-class delta",
        "",
        table_md(per_class_delta),
    ]
    (out_dir / "v10c24_round2_scale_smoke_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run()
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    out_dir = run_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_csv(run_dir, "all_round_results.csv")
    per_class = load_csv(run_dir, "per_class_metrics_by_round.csv")
    roundwise = recompute_roundwise(results)
    aulc = compute_aulc(results)
    per_class_delta = per_class_round2_delta(per_class)
    gate = compute_gate(roundwise, per_class_delta)

    roundwise.to_csv(out_dir / "roundwise_random_v10c24_comparison_recomputed.csv", index=False, encoding="utf-8-sig")
    aulc.to_csv(out_dir / "aulc_summary_recomputed.csv", index=False, encoding="utf-8-sig")
    per_class_delta.to_csv(out_dir / "round2_per_class_v10c24_minus_random_recomputed.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(out_dir / "round2_scale_gate_recomputed.csv", index=False, encoding="utf-8-sig")
    write_report(run_dir, out_dir, roundwise, aulc, gate, per_class_delta)

    print("=" * 100)
    print("[DONE] V10c24 round2 scale-smoke analysis")
    print(f"Source run: {run_dir}")
    print(f"Output dir: {out_dir}")
    if len(gate):
        print(f"Round2 gate passed: {int(gate['gate_pass'].sum())}/{len(gate)}")
    r2 = roundwise[pd.to_numeric(roundwise.get("round"), errors="coerce").eq(2)].copy() if len(roundwise) else pd.DataFrame()
    if len(r2):
        print(f"Round2 mean mAP50-95 diff: {pd.to_numeric(r2['v10c24_minus_random_map5095'], errors='coerce').mean():.6f}")
    print("Training performed=False")
    print("Final test used=False")
    print("=" * 100)


if __name__ == "__main__":
    main()

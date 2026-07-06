"""
Reconstruct round-wise remaining-pool state and compare it with selections.

This helps diagnose curriculum/order effects: what each strategy could choose
from before every round, and what it actually selected.
It does not train YOLO.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

pd = None
np = None
plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "runs" / "active_learning_ablation_v3_minimal"
OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs"
SCORE_COL = "score_combined_soft_penalty"
NO_PSEUDO_REASON = "no_pseudo_box"


def load_dependencies():
    global pd, np, plt
    import pandas as _pd
    import numpy as _np
    import matplotlib.pyplot as _plt

    pd = _pd
    np = _np
    plt = _plt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="AL run directory. Defaults to latest v3 minimal run.")
    parser.add_argument("--priority-dir", default=None, help="pseudo_boxes_* directory. Defaults to latest priority dir.")
    parser.add_argument("--quantile-bins", type=int, default=5)
    return parser.parse_args()


def latest_dir(root: Path, pattern: str, required_file: str | None = None) -> Path:
    candidates = [p for p in root.glob(pattern) if p.is_dir()]
    if required_file:
        candidates = [p for p in candidates if (p / required_file).exists()]
    if not candidates:
        raise FileNotFoundError(f"No directory found: {root}/{pattern}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def infer_class_hint(row) -> str:
    if "class_hint" in row and str(row.get("class_hint")) not in ["", "nan", "None"]:
        return str(row.get("class_hint"))
    dataset_type = str(row.get("dataset_type", ""))
    image_name = str(row.get("image_name", ""))
    image_path = Path(str(row.get("image_path", "")))
    if "NEU" in dataset_type.upper():
        return re.sub(r"_\d+$", "", Path(image_name).stem)
    if "GC10" in dataset_type.upper():
        return image_path.parent.name or "unknown"
    return "unknown"


def choose_merge_keys(left, right) -> list[str]:
    for keys in [["image_name", "dataset_type"], ["image_name", "image_path"], ["image_name"]]:
        if all(k in left.columns and k in right.columns for k in keys):
            return keys
    raise ValueError(f"No stable merge keys. left={list(left.columns)} right={list(right.columns)}")


def normalize_priority(priority, quantile_bins: int):
    df = priority.copy()
    if SCORE_COL not in df.columns:
        raise ValueError(f"priority CSV missing {SCORE_COL}. columns={list(df.columns)}")
    if "class_hint" not in df.columns:
        df["class_hint"] = df.apply(infer_class_hint, axis=1)
    else:
        df["class_hint"] = df.apply(infer_class_hint, axis=1)
    if "groundedness_reason" not in df.columns:
        df["groundedness_reason"] = "unknown"
    df["groundedness_reason"] = df["groundedness_reason"].fillna("unknown").astype(str)
    score = pd.to_numeric(df[SCORE_COL], errors="coerce")
    pct = score.rank(method="average", pct=True)
    bin_id = np.ceil(pct * quantile_bins).fillna(0).astype(int).clip(1, quantile_bins)
    df["score_quantile"] = [
        "unknown" if b <= 0 else ("Q1_lowest" if b == 1 else (f"Q{quantile_bins}_highest" if b == quantile_bins else f"Q{b}"))
        for b in bin_id
    ]
    return df


def make_key_tuples(df, keys):
    return set(tuple(row[k] for k in keys) for _, row in df.iterrows())


def add_missing_score_columns(df):
    out = df.copy()
    for col in ["score_consistency_only", "score_groundedness_only_soft", SCORE_COL]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def main():
    args = parse_args()
    load_dependencies()

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else latest_dir(RUNS_ROOT, "al_ablation_v3_minimal_*")
    priority_dir = (
        Path(args.priority_dir).expanduser().resolve()
        if args.priority_dir
        else latest_dir(OUTPUT_BASE_DIR, "pseudo_boxes_*", "priority_scores_pseudo.csv")
    )
    selected_path = run_dir / "all_selected_samples_by_round.csv"
    priority_path = priority_dir / "priority_scores_pseudo.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected CSV: {selected_path}")
    if not priority_path.exists():
        raise FileNotFoundError(f"Missing priority CSV: {priority_path}")

    selected = pd.read_csv(selected_path)
    selected["round"] = pd.to_numeric(selected["round"], errors="coerce").fillna(0).astype(int)
    priority = normalize_priority(pd.read_csv(priority_path), args.quantile_bins)
    priority = add_missing_score_columns(priority)
    keys = choose_merge_keys(selected, priority)

    priority_cols = list(dict.fromkeys(keys + [
        "image_path",
        "class_hint",
        "groundedness_reason",
        "score_quantile",
        "score_consistency_only",
        "score_groundedness_only_soft",
        SCORE_COL,
    ]))
    selected = selected.merge(
        priority[[c for c in priority_cols if c in priority.columns]].drop_duplicates(keys),
        on=keys,
        how="left",
        suffixes=("", "_priority"),
    )
    for col in priority_cols:
        suffix = f"{col}_priority"
        if suffix in selected.columns:
            selected[col] = selected[col].where(selected[col].notna(), selected[suffix])
            selected = selected.drop(columns=[suffix])
    if "class_hint" not in selected.columns:
        selected["class_hint"] = selected.apply(infer_class_hint, axis=1)
    else:
        selected["class_hint"] = selected.apply(infer_class_hint, axis=1)
    selected["groundedness_reason"] = selected.get("groundedness_reason", "unknown")
    selected["groundedness_reason"] = selected["groundedness_reason"].fillna("unknown").astype(str)

    out_dir = run_dir / f"round_pool_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    csv_dir = out_dir / "csv"
    fig_dir = out_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pool_rows = []
    selected_vs_pool_rows = []
    selected_quantile_rows = []

    initial_rows = selected[selected["round"] == 0]
    acquired_rows = selected[selected["round"] > 0]

    for (seed, strategy), strat_selected in acquired_rows.groupby(["seed", "strategy"], dropna=False):
        initial_for_seed = initial_rows[initial_rows["seed"] == seed]
        removed = make_key_tuples(initial_for_seed, keys)
        max_round = int(strat_selected["round"].max())

        for round_idx in range(1, max_round + 1):
            current_pool = priority[
                ~priority.apply(lambda row: tuple(row[k] for k in keys) in removed, axis=1)
            ].copy()
            selected_round = strat_selected[strat_selected["round"] == round_idx].copy()
            selected_keys = make_key_tuples(selected_round, keys)

            pool_rows.append({
                "seed": seed,
                "strategy": strategy,
                "round": round_idx,
                "remaining_pool_size": len(current_pool),
                "pool_score_mean": pd.to_numeric(current_pool[SCORE_COL], errors="coerce").mean(),
                "pool_score_median": pd.to_numeric(current_pool[SCORE_COL], errors="coerce").median(),
                "pool_no_pseudo_ratio": (current_pool["groundedness_reason"] == NO_PSEUDO_REASON).mean(),
                "selected_size": len(selected_round),
                "selected_score_mean": pd.to_numeric(selected_round[SCORE_COL], errors="coerce").mean(),
                "selected_score_median": pd.to_numeric(selected_round[SCORE_COL], errors="coerce").median(),
                "selected_no_pseudo_ratio": (selected_round["groundedness_reason"] == NO_PSEUDO_REASON).mean(),
            })

            for group_col in ["class_hint", "groundedness_reason", "score_quantile"]:
                pool_dist = current_pool.groupby(group_col, dropna=False).size().reset_index(name="pool_count")
                pool_dist["pool_ratio"] = pool_dist["pool_count"] / max(1, len(current_pool))
                sel_dist = selected_round.groupby(group_col, dropna=False).size().reset_index(name="selected_count")
                sel_dist["selected_ratio"] = sel_dist["selected_count"] / max(1, len(selected_round))
                comp = sel_dist.merge(pool_dist, on=group_col, how="outer").fillna(0)
                comp.insert(0, "round", round_idx)
                comp.insert(0, "strategy", strategy)
                comp.insert(0, "seed", seed)
                comp.insert(3, "group_col", group_col)
                comp = comp.rename(columns={group_col: "group_value"})
                comp["enrichment_vs_current_pool"] = np.where(
                    comp["pool_ratio"] > 0,
                    comp["selected_ratio"] / comp["pool_ratio"],
                    np.nan,
                )
                selected_vs_pool_rows.append(comp)

            selected_quantile_rows.extend([
                {
                    "seed": seed,
                    "strategy": strategy,
                    "round": round_idx,
                    "image_name": row.get("image_name"),
                    "dataset_type": row.get("dataset_type"),
                    "score_quantile": row.get("score_quantile"),
                    "score": row.get(SCORE_COL),
                    "groundedness_reason": row.get("groundedness_reason"),
                    "class_hint": row.get("class_hint"),
                }
                for _, row in selected_round.iterrows()
            ])

            removed |= selected_keys

    pool_state = pd.DataFrame(pool_rows)
    selected_vs_pool = pd.concat(selected_vs_pool_rows, ignore_index=True) if selected_vs_pool_rows else pd.DataFrame()
    selected_quantiles = pd.DataFrame(selected_quantile_rows)

    pool_state.to_csv(csv_dir / "round_pool_state_summary.csv", index=False)
    selected_vs_pool.to_csv(csv_dir / "round_selected_vs_pool_distribution.csv", index=False)
    selected_quantiles.to_csv(csv_dir / "round_selected_score_quantiles.csv", index=False)

    if len(pool_state) > 0:
        plt.figure(figsize=(10, 5))
        for strategy, sub in pool_state.groupby("strategy"):
            grouped = sub.groupby("round")["selected_no_pseudo_ratio"].mean().reset_index()
            plt.plot(grouped["round"], grouped["selected_no_pseudo_ratio"], marker="o", label=strategy)
        plt.title("Selected no_pseudo_box ratio by round")
        plt.xlabel("Round")
        plt.ylabel("Mean selected no_pseudo_box ratio")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(fig_dir / "selected_no_pseudo_ratio_by_round.png", dpi=220)
        plt.close()

        plt.figure(figsize=(10, 5))
        for strategy, sub in pool_state.groupby("strategy"):
            grouped = sub.groupby("round")["selected_score_mean"].mean().reset_index()
            plt.plot(grouped["round"], grouped["selected_score_mean"], marker="o", label=strategy)
        plt.title("Selected combined score mean by round")
        plt.xlabel("Round")
        plt.ylabel("Mean selected score")
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(fig_dir / "selected_score_mean_by_round.png", dpi=220)
        plt.close()

    lines = [
        "# Round Pool State Summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- priority_dir: `{priority_dir}`",
        f"- merge_keys: `{keys}`",
        "",
        "## Lab Meeting Use",
        "",
        "- This reconstructs the remaining pool before each acquisition round.",
        "- Compare selected ratios against current-pool ratios to diagnose curriculum/order effects.",
        "- If R1/R2 selected distributions diverge but R3 converges, final mAP may be order-sensitive.",
    ]
    (out_dir / "summary_round_pool_state.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()

"""
Compare selected samples against the full priority pool.

This script answers whether a strategy over-selects certain classes or
groundedness reasons relative to what was available in the full pool.
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


def choose_merge_keys(selected, priority) -> list[str]:
    for keys in [["image_name", "dataset_type"], ["image_name", "image_path"], ["image_name"]]:
        if all(k in selected.columns and k in priority.columns for k in keys):
            return keys
    raise ValueError(f"No stable merge keys. selected={list(selected.columns)} priority={list(priority.columns)}")


def normalize(df):
    out = df.copy()
    if "class_hint" not in out.columns:
        out["class_hint"] = out.apply(infer_class_hint, axis=1)
    else:
        out["class_hint"] = out.apply(infer_class_hint, axis=1)
    if "groundedness_reason" not in out.columns:
        out["groundedness_reason"] = "unknown"
    out["groundedness_reason"] = out["groundedness_reason"].fillna("unknown").astype(str)
    return out


def make_distribution(df, group_col: str, base_cols: list[str], count_name="count"):
    grouped = df.groupby(base_cols + [group_col], dropna=False).size().reset_index(name=count_name)
    if base_cols:
        denom = grouped.groupby(base_cols)[count_name].transform("sum")
    else:
        denom = grouped[count_name].sum()
    grouped["ratio"] = np.where(denom > 0, grouped[count_name] / denom, 0.0)
    return grouped


def compute_enrichment(selected, pool, group_col: str):
    pool_dist = make_distribution(pool, group_col, [], "pool_count")
    pool_dist = pool_dist.rename(columns={"ratio": "pool_ratio"})

    selected_dist = make_distribution(selected, group_col, ["strategy"], "selected_count")
    selected_dist = selected_dist.rename(columns={"ratio": "selected_ratio"})

    out = selected_dist.merge(pool_dist[[group_col, "pool_count", "pool_ratio"]], on=group_col, how="left")
    out["pool_count"] = out["pool_count"].fillna(0).astype(int)
    out["pool_ratio"] = out["pool_ratio"].fillna(0.0)
    out["enrichment"] = np.where(out["pool_ratio"] > 0, out["selected_ratio"] / out["pool_ratio"], np.nan)
    return out.sort_values(["strategy", "enrichment"], ascending=[True, False])


def save_heatmap(df, row_col, col_col, value_col, path: Path, title: str):
    pivot = df.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="mean", fill_value=0.0)
    plt.figure(figsize=(max(8, len(pivot.columns) * 0.7), max(4, len(pivot.index) * 0.55)))
    plt.imshow(pivot.values, aspect="auto", cmap="Blues")
    plt.colorbar(label=value_col)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(title)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def df_to_simple_markdown(df) -> str:
    if len(df) == 0:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


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
    priority = pd.read_csv(priority_path)
    keys = choose_merge_keys(selected, priority)
    priority_cols = list(dict.fromkeys(keys + ["image_path", "class_hint", "groundedness_reason"]))

    merged = selected.merge(
        priority[[c for c in priority_cols if c in priority.columns]].drop_duplicates(keys),
        on=keys,
        how="left",
        suffixes=("", "_priority"),
    )
    for col in ["image_path", "class_hint", "groundedness_reason"]:
        suffix = f"{col}_priority"
        if suffix in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[suffix])
            merged = merged.drop(columns=[suffix])

    selected_acquired = merged[pd.to_numeric(merged["round"], errors="coerce").fillna(0) > 0].copy()
    selected_acquired = normalize(selected_acquired)
    pool = normalize(priority)

    out_dir = run_dir / f"selection_enrichment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    csv_dir = out_dir / "csv"
    fig_dir = out_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    class_enrichment = compute_enrichment(selected_acquired, pool, "class_hint")
    reason_enrichment = compute_enrichment(selected_acquired, pool, "groundedness_reason")

    class_enrichment.to_csv(csv_dir / "strategy_class_enrichment.csv", index=False)
    reason_enrichment.to_csv(csv_dir / "strategy_reason_enrichment.csv", index=False)

    make_distribution(pool, "class_hint", [], "pool_count").to_csv(csv_dir / "pool_class_distribution.csv", index=False)
    make_distribution(pool, "groundedness_reason", [], "pool_count").to_csv(csv_dir / "pool_reason_distribution.csv", index=False)
    make_distribution(selected_acquired, "class_hint", ["strategy"], "selected_count").to_csv(
        csv_dir / "selected_class_distribution.csv",
        index=False,
    )
    make_distribution(selected_acquired, "groundedness_reason", ["strategy"], "selected_count").to_csv(
        csv_dir / "selected_reason_distribution.csv",
        index=False,
    )

    save_heatmap(
        class_enrichment,
        "strategy",
        "class_hint",
        "enrichment",
        fig_dir / "strategy_class_enrichment_heatmap.png",
        "Selected class enrichment vs full priority pool",
    )
    save_heatmap(
        reason_enrichment,
        "strategy",
        "groundedness_reason",
        "enrichment",
        fig_dir / "strategy_reason_enrichment_heatmap.png",
        "Selected groundedness-reason enrichment vs full priority pool",
    )

    top_class = class_enrichment.sort_values("enrichment", ascending=False).head(12)
    top_reason = reason_enrichment.sort_values("enrichment", ascending=False).head(12)
    lines = [
        "# Selection Enrichment Summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- priority_dir: `{priority_dir}`",
        f"- merge_keys: `{keys}`",
        "",
        "## Top Class Enrichment",
        "",
        df_to_simple_markdown(top_class),
        "",
        "## Top Reason Enrichment",
        "",
        df_to_simple_markdown(top_reason),
        "",
        "## Lab Meeting Use",
        "",
        "- Enrichment > 1 means a strategy selects that group more often than the full priority pool contains it.",
        "- Use this to test whether strategy gains are confounded by class or groundedness-reason distribution.",
    ]
    (out_dir / "summary_selection_enrichment.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()

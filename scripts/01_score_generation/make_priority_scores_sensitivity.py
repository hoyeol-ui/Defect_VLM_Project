"""
Generate priority-score CSV variants for direction-fix experiments.

This script reuses an existing pseudo groundedness result or priority score CSV
and writes multiple score-calibrated variants without recomputing VLM, SBERT,
OWL-ViT, or pseudo groundedness.

It is intended for:
- missing_box_penalty sensitivity
- groundedness ablation
- weighted score calibration
- rank-normalized score calibration
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

pd = None
np = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs"
NO_PSEUDO_REASON = "no_pseudo_box"


def load_dependencies():
    global pd, np
    import pandas as _pd
    import numpy as _np

    pd = _pd
    np = _np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority-csv", default=None, help="Existing priority_scores_pseudo.csv. Defaults to latest.")
    parser.add_argument(
        "--penalties",
        default="0,0.1,0.2,0.5,1.0",
        help="Comma-separated no_pseudo_box penalty values.",
    )
    parser.add_argument(
        "--groundedness-weights",
        default="0,0.25,0.5,1.0",
        help="Comma-separated beta values for weighted score alpha*U_C + beta*U_G + gamma*P.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Consistency uncertainty weight.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to outputs/priority_sensitivity_YYYYMMDD_HHMMSS.",
    )
    return parser.parse_args()


def latest_priority_csv() -> Path:
    candidates = list(OUTPUT_BASE_DIR.glob("pseudo_boxes_*/priority_scores_pseudo.csv"))
    if not candidates:
        raise FileNotFoundError(f"No priority_scores_pseudo.csv found under {OUTPUT_BASE_DIR}/pseudo_boxes_*")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_float_list(text: str) -> list[float]:
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def ensure_base_columns(df):
    out = df.copy()
    required = ["image_name", "dataset_type", "consistency_score", "groundedness_reason"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Priority CSV missing required columns: {missing}. columns={list(out.columns)}")

    out["consistency_score"] = pd.to_numeric(out["consistency_score"], errors="coerce").fillna(0.0).clip(0, 1)
    out["groundedness_reason"] = out["groundedness_reason"].fillna("unknown").astype(str)

    if "groundedness_norm" in out.columns:
        out["groundedness_norm"] = pd.to_numeric(out["groundedness_norm"], errors="coerce")
    else:
        out["groundedness_norm"] = np.nan

    out["groundedness_effective_soft"] = out["groundedness_norm"].fillna(0.5).clip(0, 1)
    out["uncertainty_consistency"] = 1.0 - out["consistency_score"]
    out["uncertainty_groundedness_soft"] = 1.0 - out["groundedness_effective_soft"]
    out["is_no_pseudo_box"] = (out["groundedness_reason"] == NO_PSEUDO_REASON).astype(int)

    return out


def add_scores(df, penalty: float, alpha: float, betas: list[float]):
    out = df.copy()
    penalty_col = f"missing_box_penalty_p{penalty:g}"
    combined_col = f"score_combined_soft_penalty_p{penalty:g}"

    out[penalty_col] = penalty * out["is_no_pseudo_box"]
    out[combined_col] = (
        out["uncertainty_consistency"]
        + out["uncertainty_groundedness_soft"]
        + out[penalty_col]
    ).round(6)

    out["score_consistency_only"] = out["uncertainty_consistency"].round(6)
    out["score_groundedness_only_soft"] = (out["uncertainty_groundedness_soft"] + out[penalty_col]).round(6)
    out["score_combined_no_penalty"] = (
        out["uncertainty_consistency"] + out["uncertainty_groundedness_soft"]
    ).round(6)
    out["score_combined_no_groundedness"] = out["uncertainty_consistency"].round(6)

    u_c_rank = out["uncertainty_consistency"].rank(method="average", pct=True)
    u_g_rank = out["uncertainty_groundedness_soft"].rank(method="average", pct=True)
    out[f"score_rank_calibrated_p{penalty:g}"] = (u_c_rank + u_g_rank + out[penalty_col]).round(6)

    for beta in betas:
        col = f"score_weighted_a{alpha:g}_b{beta:g}_p{penalty:g}"
        out[col] = (
            alpha * out["uncertainty_consistency"]
            + beta * out["uncertainty_groundedness_soft"]
            + out[penalty_col]
        ).round(6)

    # Compatibility column for existing AL runner.
    out["missing_box_penalty"] = out[penalty_col].round(6)
    out["score_combined_soft_penalty"] = out[combined_col]

    return out.sort_values("score_combined_soft_penalty", ascending=False).reset_index(drop=True)


def summarize_variant(df, score_col: str) -> dict:
    top = df.sort_values(score_col, ascending=False).head(20)
    bottom = df.sort_values(score_col, ascending=True).head(20)
    return {
        "score_col": score_col,
        "num_items": int(len(df)),
        "top20_no_pseudo_ratio": float((top["groundedness_reason"] == NO_PSEUDO_REASON).mean()) if len(top) else None,
        "bottom20_no_pseudo_ratio": float((bottom["groundedness_reason"] == NO_PSEUDO_REASON).mean()) if len(bottom) else None,
        "overall_no_pseudo_ratio": float((df["groundedness_reason"] == NO_PSEUDO_REASON).mean()) if len(df) else None,
        "score_mean": float(pd.to_numeric(df[score_col], errors="coerce").mean()),
        "score_std": float(pd.to_numeric(df[score_col], errors="coerce").std()),
    }


def main():
    args = parse_args()
    load_dependencies()

    priority_csv = Path(args.priority_csv).expanduser().resolve() if args.priority_csv else latest_priority_csv()
    if not priority_csv.exists():
        raise FileNotFoundError(f"Priority CSV does not exist: {priority_csv}")

    penalties = parse_float_list(args.penalties)
    betas = parse_float_list(args.groundedness_weights)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else OUTPUT_BASE_DIR / f"priority_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    base = ensure_base_columns(pd.read_csv(priority_csv))
    summaries = []

    for penalty in penalties:
        variant = add_scores(base, penalty=penalty, alpha=args.alpha, betas=betas)
        variant_dir = out_dir / f"penalty_{penalty:g}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        csv_path = variant_dir / "priority_scores_pseudo.csv"
        json_path = variant_dir / "priority_scores_pseudo.json"
        summary_path = variant_dir / "priority_scores_summary.json"

        variant.to_csv(csv_path, index=False)
        variant.to_json(json_path, orient="records", force_ascii=False, indent=2)

        score_cols = [
            c for c in variant.columns
            if c.startswith("score_") and pd.api.types.is_numeric_dtype(variant[c])
        ]
        summary = {
            "source_priority_csv": str(priority_csv),
            "penalty": penalty,
            "alpha": args.alpha,
            "groundedness_weights": betas,
            "variants": [summarize_variant(variant, col) for col in score_cols],
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append({
            "penalty": penalty,
            "variant_dir": str(variant_dir),
            "priority_csv": str(csv_path),
            "combined_top20_no_pseudo_ratio": summary["variants"][0]["top20_no_pseudo_ratio"] if summary["variants"] else None,
        })

    pd.DataFrame(summaries).to_csv(out_dir / "sensitivity_manifest.csv", index=False)

    lines = [
        "# Priority Score Sensitivity Manifest",
        "",
        f"- source_priority_csv: `{priority_csv}`",
        f"- output_dir: `{out_dir}`",
        f"- penalties: {penalties}",
        f"- groundedness weights: {betas}",
        "",
        "## How to Run AL for One Variant",
        "",
        "Use the generated `priority_scores_pseudo.csv` as `AL_PRIORITY_CSV` with the v3 runner.",
        "",
        "```bash",
        "AL_PRIORITY_CSV=/path/to/priority_sensitivity_*/penalty_0.1/priority_scores_pseudo.csv \\",
        "AL_STRATEGIES=Random,ConsistencyOnly,GroundednessOnlySoft,CombinedSoftPenalty,LowPrioritySoft \\",
        "python scripts/02_active_learning/run_al_yolo_ablation_v3_minimal.py",
        "```",
    ]
    (out_dir / "README_sensitivity.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()

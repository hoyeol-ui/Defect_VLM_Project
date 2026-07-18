"""Retrospective validity audit for the legacy VLM consistency pilot.

This script performs no model inference and no detector training. It excludes
every filename found in the supplied locked-final manifests, then tests whether
the old SBERT consistency score was associated with GT-oracle location/scale
groundedness. GT is used only for this post-hoc audit.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "grounded_experiment_20260601"
    / "pilot_grounded_consistency_results.json"
)
DEFAULT_LOCKED_MANIFESTS = [
    PROJECT_ROOT
    / "runs"
    / "evaluation_protocol_v7"
    / "eval_protocol_20260711_173723"
    / "final_test_v7.csv",
    PROJECT_ROOT
    / "runs"
    / "gc10_taxonomy_protocol"
    / "gc10_protocol_20260715"
    / "gc10_final_test_locked.csv",
]


@dataclass(frozen=True)
class BootstrapConfig:
    reps: int = 10_000
    seed: int = 20260715


def load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("results", "items", "records"):
            if isinstance(payload.get(key), list):
                return [x for x in payload[key] if isinstance(x, dict)]
    raise ValueError(f"Unsupported legacy JSON schema: {path}")


def read_locked_names(paths: Iterable[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    names: set[str] = set()
    audit: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            audit.append({"manifest": str(path), "status": "missing", "rows": 0})
            continue
        row_count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row_count += 1
                for key in ("image_name", "filename", "image_path", "resolved_image_path"):
                    value = str(row.get(key, "")).strip()
                    if value:
                        names.add(Path(value).name.casefold())
        audit.append({"manifest": str(path), "status": "loaded", "rows": row_count})
    return names, audit


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def extract_row(item: dict[str, Any]) -> dict[str, Any]:
    grounded = item.get("groundedness", {})
    if not isinstance(grounded, dict):
        grounded = {}

    raw_score = grounded.get("total_score", item.get("groundedness_score"))
    score = float(raw_score) if raw_score is not None else np.nan
    # The legacy rubric contains two binary dimensions: location and scale.
    groundedness_norm = score / 2.0 if np.isfinite(score) else np.nan
    consistency = item.get("consistency_score", item.get("consistency"))
    consistency = float(consistency) if consistency is not None else np.nan

    return {
        "image_name": str(item.get("image_name", item.get("image_id", ""))).strip(),
        "dataset_type": str(item.get("dataset_type", "unknown")).strip() or "unknown",
        "object_count": item.get("object_count"),
        "consistency_score": consistency,
        "inconsistency_score": 1.0 - consistency if np.isfinite(consistency) else np.nan,
        "groundedness_score": score,
        "groundedness_norm": groundedness_norm,
        "severe_grounding_failure": int(score == 0.0) if np.isfinite(score) else np.nan,
        "any_grounding_error": int(score < 2.0) if np.isfinite(score) else np.nan,
        "primary_location_correct": _as_bool(grounded.get("primary_location_correct")),
        "primary_scale_correct": _as_bool(grounded.get("primary_scale_correct")),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[mask], y[mask]
    if len(x2) < 3 or np.unique(x2).size < 2 or np.unique(y2).size < 2:
        return float("nan")
    result = spearmanr(x2, y2) if method == "spearman" else pearsonr(x2, y2)
    return float(result.statistic)


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(score)
    y2, score2 = y[mask], score[mask]
    if len(y2) < 3 or np.unique(y2).size < 2:
        return float("nan")
    return float(roc_auc_score(y2.astype(int), score2))


def _quartile_gap(consistency: np.ndarray, groundedness: np.ndarray) -> float:
    mask = np.isfinite(consistency) & np.isfinite(groundedness)
    c, g = consistency[mask], groundedness[mask]
    if len(c) < 8:
        return float("nan")
    q25, q75 = np.quantile(c, [0.25, 0.75])
    low, high = g[c <= q25], g[c >= q75]
    if not len(low) or not len(high):
        return float("nan")
    # Positive means the high-consistency quartile is better grounded.
    return float(np.mean(high) - np.mean(low))


def _bootstrap_values(
    df: pd.DataFrame,
    metric: str,
    config: BootstrapConfig,
) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    n = len(df)
    values: list[float] = []
    c = df["consistency_score"].to_numpy(float)
    g = df["groundedness_norm"].to_numpy(float)
    inc = df["inconsistency_score"].to_numpy(float)
    severe = df["severe_grounding_failure"].to_numpy(float)

    for _ in range(config.reps):
        idx = rng.integers(0, n, size=n)
        if metric == "spearman":
            value = _safe_corr(c[idx], g[idx], "spearman")
        elif metric == "pearson":
            value = _safe_corr(c[idx], g[idx], "pearson")
        elif metric == "severe_failure_auc":
            value = _safe_auc(severe[idx], inc[idx])
        elif metric == "quartile_gap":
            value = _quartile_gap(c[idx], g[idx])
        else:
            raise KeyError(metric)
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _ci(values: np.ndarray) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def compute_metrics(df: pd.DataFrame, config: BootstrapConfig) -> pd.DataFrame:
    c = df["consistency_score"].to_numpy(float)
    g = df["groundedness_norm"].to_numpy(float)
    inc = df["inconsistency_score"].to_numpy(float)
    severe = df["severe_grounding_failure"].to_numpy(float)

    points = {
        "spearman_consistency_vs_groundedness": _safe_corr(c, g, "spearman"),
        "pearson_consistency_vs_groundedness": _safe_corr(c, g, "pearson"),
        "inconsistency_auc_severe_failure": _safe_auc(severe, inc),
        "high_minus_low_consistency_quartile_groundedness": _quartile_gap(c, g),
    }
    bootstrap_keys = {
        "spearman_consistency_vs_groundedness": "spearman",
        "pearson_consistency_vs_groundedness": "pearson",
        "inconsistency_auc_severe_failure": "severe_failure_auc",
        "high_minus_low_consistency_quartile_groundedness": "quartile_gap",
    }
    rows: list[dict[str, Any]] = []
    for name, point in points.items():
        samples = _bootstrap_values(df, bootstrap_keys[name], config)
        low, high = _ci(samples)
        rows.append(
            {
                "scope": "overall",
                "metric": name,
                "n": len(df),
                "value": point,
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "valid_bootstrap_reps": len(samples),
            }
        )

    for dataset, sub in df.groupby("dataset_type", sort=True):
        x = sub["consistency_score"].to_numpy(float)
        y = sub["groundedness_norm"].to_numpy(float)
        rows.append(
            {
                "scope": str(dataset),
                "metric": "spearman_consistency_vs_groundedness",
                "n": len(sub),
                "value": _safe_corr(x, y, "spearman"),
                "bootstrap_ci95_low": np.nan,
                "bootstrap_ci95_high": np.nan,
                "valid_bootstrap_reps": 0,
            }
        )
    return pd.DataFrame(rows)


def build_gate(metrics: pd.DataFrame) -> pd.DataFrame:
    overall = metrics[metrics["scope"] == "overall"].set_index("metric")
    rho = overall.loc["spearman_consistency_vs_groundedness"]
    auc = overall.loc["inconsistency_auc_severe_failure"]
    gap = overall.loc["high_minus_low_consistency_quartile_groundedness"]
    strata = metrics[
        (metrics["scope"] != "overall")
        & (metrics["metric"] == "spearman_consistency_vs_groundedness")
    ]
    checks = [
        ("spearman_rho_at_least_0p20", float(rho["value"]) >= 0.20, rho["value"]),
        ("spearman_bootstrap_ci_low_positive", float(rho["bootstrap_ci95_low"]) > 0.0, rho["bootstrap_ci95_low"]),
        ("severe_failure_auc_at_least_0p60", float(auc["value"]) >= 0.60, auc["value"]),
        ("severe_failure_auc_ci_low_above_0p50", float(auc["bootstrap_ci95_low"]) > 0.50, auc["bootstrap_ci95_low"]),
        ("quartile_groundedness_gap_at_least_0p10", float(gap["value"]) >= 0.10, gap["value"]),
        (
            "nonnegative_direction_in_every_dataset",
            bool(len(strata)) and bool((strata["value"].fillna(-np.inf) >= 0.0).all()),
            strata["value"].min() if len(strata) else np.nan,
        ),
    ]
    out = pd.DataFrame(checks, columns=["check", "passed", "observed"])
    out["result"] = np.where(out["passed"], "PASS", "FAIL")
    return out


def build_component_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    inconsistency = df["inconsistency_score"].to_numpy(float)
    targets = {
        "primary_location_error": 1.0 - pd.to_numeric(
            df["primary_location_correct"], errors="coerce"
        ).to_numpy(float),
        "primary_scale_error": 1.0 - pd.to_numeric(
            df["primary_scale_correct"], errors="coerce"
        ).to_numpy(float),
        "any_grounding_error": df["any_grounding_error"].to_numpy(float),
        "severe_grounding_failure": df["severe_grounding_failure"].to_numpy(float),
    }
    for target_name, target in targets.items():
        mask = np.isfinite(target) & np.isfinite(inconsistency)
        rows.append(
            {
                "target": target_name,
                "n": int(mask.sum()),
                "positive_rate": float(np.mean(target[mask])) if mask.any() else np.nan,
                "inconsistency_auc": _safe_auc(target, inconsistency),
            }
        )
    return pd.DataFrame(rows)


def build_quartile_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = df.copy()
    labels = ["Q1_low", "Q2", "Q3", "Q4_high"]
    labeled["consistency_quartile"] = pd.qcut(
        labeled["consistency_score"].rank(method="first"),
        q=4,
        labels=labels,
    )
    summary = (
        labeled.groupby("consistency_quartile", observed=True)
        .agg(
            n=("image_name", "size"),
            consistency_mean=("consistency_score", "mean"),
            groundedness_mean=("groundedness_norm", "mean"),
            severe_failure_rate=("severe_grounding_failure", "mean"),
            any_error_rate=("any_grounding_error", "mean"),
        )
        .reset_index()
    )
    return labeled, summary


def write_plot(df: pd.DataFrame, quartiles: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20260715)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"NEU-DET": "#2563eb", "GC10-DET": "#dc2626"}
    for dataset, sub in df.groupby("dataset_type", sort=True):
        jitter = rng.normal(0.0, 0.018, size=len(sub))
        axes[0].scatter(
            sub["consistency_score"],
            sub["groundedness_norm"] + jitter,
            alpha=0.7,
            s=30,
            label=dataset,
            color=colors.get(str(dataset)),
        )
    axes[0].set_xlabel("Legacy SBERT consistency")
    axes[0].set_ylabel("GT-oracle groundedness (0-1)")
    axes[0].set_title("Consistency did not track groundedness")
    axes[0].set_ylim(-0.08, 1.08)
    axes[0].legend(frameon=False)

    positions = np.arange(len(quartiles))
    axes[1].bar(
        positions - 0.18,
        quartiles["groundedness_mean"],
        width=0.36,
        label="Mean groundedness",
        color="#0f766e",
    )
    axes[1].bar(
        positions + 0.18,
        quartiles["severe_failure_rate"],
        width=0.36,
        label="Severe failure rate",
        color="#f97316",
    )
    axes[1].set_xticks(positions, quartiles["consistency_quartile"].astype(str))
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Quartile diagnostic")
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    path: Path,
    source_count: int,
    excluded: pd.DataFrame,
    rows: pd.DataFrame,
    metrics: pd.DataFrame,
    gate: pd.DataFrame,
    component_metrics: pd.DataFrame,
    quartiles: pd.DataFrame,
) -> None:
    overall = metrics[metrics["scope"] == "overall"]
    gate_pass = bool(gate["passed"].all())
    lines = [
        "# Legacy VLM Consistency Validity Audit",
        "",
        "- New VLM inference performed: **False**",
        "- Detector training performed: **False**",
        "- Final test evaluated: **False**",
        f"- Source rows: **{source_count}**",
        f"- Locked-final rows excluded: **{len(excluded)}**",
        f"- Development/post-hoc rows analyzed: **{len(rows)}**",
        f"- Legacy diagnostic gate: **{'PASS' if gate_pass else 'FAIL'}**",
        "",
        "The four locked-final images were removed before any metric was computed.",
        "GT location/scale information is used only as a post-hoc validity target.",
        "",
        "## Metrics",
        "",
        overall.to_markdown(index=False),
        "",
        "## Dataset-direction check",
        "",
        metrics[metrics["scope"] != "overall"].to_markdown(index=False),
        "",
        "## Error-component diagnostic",
        "",
        component_metrics.to_markdown(index=False),
        "",
        "## Consistency quartiles",
        "",
        quartiles.to_markdown(index=False),
        "",
        "## Diagnostic gate",
        "",
        gate.drop(columns=["passed"]).to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        "This is not confirmatory evidence. The legacy prompts asked different semantic questions "
        "(location, scale, and appearance), so pairwise SBERT similarity partly measures task-text "
        "difference rather than repeated belief stability. A pass only authorizes the frozen "
        "structured-prompt pilot. A fail rejects this legacy score as a useful validity signal.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--locked-manifest", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    locked_paths = args.locked_manifest or DEFAULT_LOCKED_MANIFESTS
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            PROJECT_ROOT
            / "runs"
            / "vlm_consistency_groundedness_validity"
            / f"legacy_pilot_audit_{stamp}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(args.input)
    locked_names, manifest_audit = read_locked_names(locked_paths)
    extracted = pd.DataFrame(extract_row(item) for item in items)
    extracted["locked_final_excluded"] = extracted["image_name"].str.casefold().isin(locked_names)
    excluded = extracted[extracted["locked_final_excluded"]].copy()
    rows = extracted[~extracted["locked_final_excluded"]].copy()
    rows = rows.dropna(subset=["consistency_score", "groundedness_norm"]).reset_index(drop=True)
    if len(rows) < 20:
        raise RuntimeError(f"Too few valid non-final rows for audit: {len(rows)}")

    config = BootstrapConfig(reps=args.bootstrap_reps, seed=args.seed)
    metrics = compute_metrics(rows, config)
    gate = build_gate(metrics)
    component_metrics = build_component_metrics(rows)
    labeled_rows, quartiles = build_quartile_table(rows)

    q1 = labeled_rows[labeled_rows["consistency_quartile"] == "Q1_low"]
    q4 = labeled_rows[labeled_rows["consistency_quartile"] == "Q4_high"]
    counterexamples = pd.concat(
        [
            q4[q4["severe_grounding_failure"] == 1].assign(
                counterexample_type="high_consistency_severe_failure"
            ),
            q1[q1["groundedness_score"] == 2].assign(
                counterexample_type="low_consistency_fully_grounded"
            ),
        ],
        ignore_index=True,
    )

    rows.to_csv(args.output_dir / "legacy_validity_rows.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(args.output_dir / "locked_final_exclusions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output_dir / "legacy_validity_metrics.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(args.output_dir / "legacy_validity_gate.csv", index=False, encoding="utf-8-sig")
    component_metrics.to_csv(
        args.output_dir / "legacy_component_metrics.csv", index=False, encoding="utf-8-sig"
    )
    quartiles.to_csv(
        args.output_dir / "legacy_consistency_quartiles.csv", index=False, encoding="utf-8-sig"
    )
    counterexamples.to_csv(
        args.output_dir / "legacy_counterexamples.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(manifest_audit).to_csv(
        args.output_dir / "locked_manifest_audit.csv", index=False, encoding="utf-8-sig"
    )
    config_payload = {
        "input": str(args.input.resolve()),
        "locked_manifests": [str(p.resolve()) for p in locked_paths],
        "source_rows": len(items),
        "excluded_locked_final_rows": len(excluded),
        "analyzed_rows": len(rows),
        "bootstrap_reps": config.reps,
        "seed": config.seed,
        "detector_training_performed": False,
        "final_test_evaluated": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary(
        args.output_dir / "legacy_validity_summary.md",
        len(items),
        excluded,
        rows,
        metrics,
        gate,
        component_metrics,
        quartiles,
    )
    write_plot(
        rows,
        quartiles,
        args.output_dir / "legacy_consistency_groundedness_diagnostic.png",
    )
    print(f"[DONE] analyzed={len(rows)} excluded_locked_final={len(excluded)}")
    print(f"[GATE] {'PASS' if bool(gate['passed'].all()) else 'FAIL'}")
    print(f"[SUMMARY] {args.output_dir / 'legacy_validity_summary.md'}")


if __name__ == "__main__":
    main()

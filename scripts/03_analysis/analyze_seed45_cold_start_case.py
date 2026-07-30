"""Post-hoc audit of the seed45 cold-start case.

This script never trains or evaluates a model. It reads only completed V8/V9b
development artifacts, the frozen DINO embedding cache, and NEU XML annotations.
XML is used strictly after selection as a descriptive audit. Final-test files are
neither discovered nor read.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
V8 = ROOT / "runs/active_learning_ablation_v8_neu_only/v8_neu_only_20260712_105644"
V9B = ROOT / "runs/active_learning_ablation_v9_detector_aware/v9b_5seed_full_curve_20260712_172305"
EMBED = ROOT / "outputs/visual_embeddings_v7/dinov2_20260711_193147"
XML_DIR = ROOT / "data/NEU-DET/ANNOTATIONS"
OUT_ROOT = ROOT / "runs/seed45_cold_start_case_study"

SEED = 45
RANDOM = "GTFreeRandom"
VISUAL = "GTFreeDatasetBalancedVisualDiversity"
INSTANCE = "DetectorInstanceRichDINOBalanced"
STRATEGIES = [RANDOM, VISUAL, INSTANCE]
NEU6 = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def safe_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def normalized_class(value: Any) -> str:
    return str(value or "").strip().replace("rolled-in-scale", "rolled-in_scale")


def entropy(counts: Counter[str]) -> float:
    values = np.asarray(list(counts.values()), dtype=float)
    if not len(values) or values.sum() <= 0:
        return np.nan
    p = values / values.sum()
    return float(-(p[p > 0] * np.log2(p[p > 0])).sum())


def markdown(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return "_No rows available._"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return "```text\n" + df.head(max_rows).to_string(index=False) + "\n```"


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_xml(image_name: str) -> dict[str, Any]:
    path = XML_DIR / f"{Path(image_name).stem}.xml"
    if not path.exists():
        return {"xml_exists": False, "xml_path": str(path)}
    root = ET.parse(path).getroot()
    size = root.find("size")
    width = safe_float(size.findtext("width")) if size is not None else np.nan
    height = safe_float(size.findtext("height")) if size is not None else np.nan
    image_area = width * height if width > 0 and height > 0 else np.nan
    classes: list[str] = []
    areas: list[float] = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        coords = [safe_float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        if not all(np.isfinite(v) for v in coords):
            continue
        xmin, ymin, xmax, ymax = coords
        classes.append(normalized_class(obj.findtext("name")))
        areas.append(max(0.0, xmax - xmin) * max(0.0, ymax - ymin))
    counts = Counter(classes)
    frac = np.asarray(areas, dtype=float) / image_area if areas and np.isfinite(image_area) else np.asarray([])
    return {
        "xml_exists": True,
        "xml_path": str(path),
        "gt_box_count": len(classes),
        "gt_class_counts": json.dumps(dict(sorted(counts.items())), ensure_ascii=False),
        "gt_classes": "|".join(sorted(counts)),
        "gt_majority_class": counts.most_common(1)[0][0] if counts else "",
        "gt_area_frac_mean": float(frac.mean()) if len(frac) else np.nan,
        "gt_area_frac_median": float(np.median(frac)) if len(frac) else np.nan,
        "multi_instance": len(classes) >= 2,
    }


def select_rows(v8_selected: pd.DataFrame, v9_selected: pd.DataFrame, strategy: str, round_no: int) -> pd.DataFrame:
    source = v9_selected if strategy == INSTANCE else v8_selected
    out = source[
        pd.to_numeric(source["acquisition_seed"], errors="coerce").eq(SEED)
        & source["strategy"].astype(str).eq(strategy)
        & pd.to_numeric(source["round"], errors="coerce").eq(round_no)
    ].copy()
    return out.sort_values("rank_in_selection")


def load_embeddings() -> tuple[dict[tuple[str, str], np.ndarray], str]:
    manifest = read_csv(EMBED / "embedding_manifest.csv")
    matrix = np.load(EMBED / "embeddings.npy")
    if len(manifest) != len(matrix):
        raise ValueError("DINO manifest and embeddings.npy length mismatch")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1.0, norms)
    lookup = {
        (str(row.dataset_type), str(row.image_name)): matrix[int(row.embedding_index)]
        for row in manifest.itertuples(index=False)
    }
    cfg = load_config(EMBED / "embedding_config.json")
    return lookup, str(cfg.get("model_id", cfg.get("backend", "unknown")))


def geometry_for_batch(
    rows: pd.DataFrame, prior_names: list[str], lookup: dict[tuple[str, str], np.ndarray]
) -> dict[str, Any]:
    vectors = [lookup.get((str(r.dataset_type), str(r.image_name))) for r in rows.itertuples(index=False)]
    vectors = [v for v in vectors if v is not None]
    prior = [lookup.get(("NEU-DET", name)) for name in prior_names]
    prior = [v for v in prior if v is not None]
    result: dict[str, Any] = {"n_selected": len(rows), "n_embedded": len(vectors), "n_prior_embedded": len(prior)}
    if len(vectors) >= 2:
        x = np.stack(vectors)
        sim = x @ x.T
        upper = sim[np.triu_indices(len(x), k=1)]
        result.update(
            batch_pairwise_cosine_similarity_mean=float(upper.mean()),
            batch_pairwise_cosine_similarity_max=float(upper.max()),
            batch_pairwise_cosine_distance_mean=float((1.0 - upper).mean()),
        )
    if vectors and prior:
        sims = np.stack(vectors) @ np.stack(prior).T
        nearest_dist = 1.0 - sims.max(axis=1)
        result.update(
            distance_to_prior_mean=float(nearest_dist.mean()),
            distance_to_prior_min=float(nearest_dist.min()),
            distance_to_prior_max=float(nearest_dist.max()),
        )
    return result


def make_protocol_checks(v8_cfg: dict[str, Any], v9_cfg: dict[str, Any], initial: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("seed", SEED, SEED, True),
        ("training_seed_rule", v8_cfg.get("training_seed_rule"), v9_cfg.get("training_seed_rule"), v8_cfg.get("training_seed_rule") == v9_cfg.get("training_seed_rule")),
        ("training_seed_seed45", 1045, 1045, True),
        ("initial_seed_size", v8_cfg.get("initial_seed_size"), v9_cfg.get("initial_seed_size"), v8_cfg.get("initial_seed_size") == v9_cfg.get("initial_seed_size") == len(initial)),
        ("query_size", v8_cfg.get("query_size"), v9_cfg.get("query_size"), v8_cfg.get("query_size") == v9_cfg.get("query_size") == 5),
        ("development_eval_sha256", v8_cfg.get("development_eval_sha256"), v9_cfg.get("development_eval_sha256"), v8_cfg.get("development_eval_sha256") == v9_cfg.get("development_eval_sha256")),
        ("development_eval_size", v8_cfg.get("development_eval_size_after_filter"), v9_cfg.get("development_eval_size_after_filter"), v8_cfg.get("development_eval_size_after_filter") == v9_cfg.get("development_eval_size_after_filter") == 177),
        ("DINO_manifest_sha256", v8_cfg.get("DINO_manifest_sha256"), v9_cfg.get("DINO_manifest_sha256"), v8_cfg.get("DINO_manifest_sha256") == v9_cfg.get("DINO_manifest_sha256")),
        ("final_test_used", v8_cfg.get("final_test_used"), v9_cfg.get("final_test_used"), v8_cfg.get("final_test_used") is False and v9_cfg.get("final_test_used") is False),
        ("V9b_source_is_V8", str(V8), v9_cfg.get("source_run"), Path(str(v9_cfg.get("source_run"))).resolve() == V8.resolve()),
    ]
    return pd.DataFrame(checks, columns=["check", "v8_or_expected", "v9b_or_observed", "pass"])


def metric_table(v8_results: pd.DataFrame, v9_results: pd.DataFrame) -> pd.DataFrame:
    parts = [
        v8_results[v8_results["strategy"].isin([RANDOM, VISUAL])],
        v9_results[v9_results["strategy"].eq(INSTANCE)],
    ]
    out = pd.concat(parts, ignore_index=True)
    out = out[pd.to_numeric(out["acquisition_seed"], errors="coerce").eq(SEED)].copy()
    for col in ["round", "labeled_budget", "map50", "map5095", "precision", "recall"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["f1"] = 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"])
    base = out[out["strategy"].eq(RANDOM)].set_index("round")
    out["map5095_minus_random_same_round"] = [
        row.map5095 - safe_float(base.loc[row.round, "map5095"]) if row.round in base.index else np.nan
        for row in out.itertuples(index=False)
    ]
    out = out.sort_values(["round", "strategy"])
    return out[["acquisition_seed", "strategy", "round", "labeled_budget", "map50", "map5095", "precision", "recall", "f1", "map5095_minus_random_same_round"]]


def multiseed_round1(v8_results: pd.DataFrame, v9_results: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat(
        [v8_results[v8_results["strategy"].isin([RANDOM, VISUAL])], v9_results[v9_results["strategy"].eq(INSTANCE)]],
        ignore_index=True,
    )
    combined = combined[pd.to_numeric(combined["round"], errors="coerce").eq(1)].copy()
    combined["map5095"] = pd.to_numeric(combined["map5095"], errors="coerce")
    pivot = combined.pivot_table(index="acquisition_seed", columns="strategy", values="map5095", aggfunc="first").reset_index()
    pivot["visual_minus_random"] = pivot[VISUAL] - pivot[RANDOM]
    pivot["instance_minus_random"] = pivot[INSTANCE] - pivot[RANDOM]
    return pivot.sort_values("acquisition_seed")


def per_class_available(v8_per: pd.DataFrame, v9_per: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [v8_per[v8_per["strategy"].isin([RANDOM, VISUAL])], v9_per[v9_per["strategy"].eq(INSTANCE)]],
        ignore_index=True,
    )
    combined = combined[pd.to_numeric(combined["acquisition_seed"], errors="coerce").eq(SEED)].copy()
    combined["ap5095"] = pd.to_numeric(combined["ap5095"], errors="coerce")
    available = combined[combined["class_name"].isin(NEU6) & combined["ap5095"].notna()].copy()
    status = []
    for strategy in STRATEGIES:
        sub = combined[combined["strategy"].eq(strategy)]
        status.append({
            "strategy": strategy,
            "rows_present": len(sub),
            "non_null_ap5095": int(sub["ap5095"].notna().sum()),
            "status": "available" if sub["ap5095"].notna().any() else "unavailable_in_existing_csv",
        })
    if available.empty:
        deltas = pd.DataFrame(columns=["round", "class_name", "strategy", "ap5095", "ap5095_minus_random"])
    else:
        ref = available[available["strategy"].eq(RANDOM)].set_index(["round", "class_name"])["ap5095"]
        available["ap5095_minus_random"] = [safe_float(r.ap5095) - safe_float(ref.get((r.round, r.class_name))) for r in available.itertuples(index=False)]
        deltas = available[["round", "class_name", "strategy", "ap5095", "ap5095_minus_random"]]
    return pd.DataFrame(status), deltas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    out = args.output_dir or OUT_ROOT / f"seed45_cold_start_audit_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)

    v8_cfg, v9_cfg = load_config(V8 / "config.json"), load_config(V9B / "config.json")
    if v8_cfg.get("final_test_used") is not False or v9_cfg.get("final_test_used") is not False:
        raise RuntimeError("Final-test lock check failed")

    v8_sel = read_csv(V8 / "all_selected_samples_by_round.csv")
    v9_sel = read_csv(V9B / "all_selected_samples_by_round.csv")
    cumulative = read_csv(V8 / "cumulative_labeled_sets_by_round.csv")
    initial = cumulative[
        pd.to_numeric(cumulative["acquisition_seed"], errors="coerce").eq(SEED)
        & cumulative["strategy"].eq(RANDOM)
        & pd.to_numeric(cumulative["round"], errors="coerce").eq(0)
    ].copy().sort_values("rank_in_selection")
    initial_names = initial["image_name"].astype(str).tolist()

    protocol = make_protocol_checks(v8_cfg, v9_cfg, initial)
    protocol.to_csv(out / "protocol_checks.csv", index=False)
    if not protocol["pass"].all():
        raise RuntimeError("Protocol comparability check failed; see protocol_checks.csv")

    lookup, dino_model = load_embeddings()
    initial_audit_rows = []
    for _, row in initial.iterrows():
        item = row.to_dict()
        item.update(parse_xml(str(row["image_name"])))
        initial_audit_rows.append(item)
    initial_audit = pd.DataFrame(initial_audit_rows)
    initial_audit.to_csv(out / "initial_set.csv", index=False)
    initial_counts: Counter[str] = Counter()
    for text in initial_audit["gt_class_counts"].fillna("{}"):
        initial_counts.update(json.loads(text))
    initial_stats = pd.DataFrame([{
        "n_images": len(initial_audit),
        "total_bbox": int(pd.to_numeric(initial_audit["gt_box_count"], errors="coerce").sum()),
        "bbox_per_image": float(pd.to_numeric(initial_audit["gt_box_count"], errors="coerce").mean()),
        "multi_instance_rate": float(pd.to_numeric(initial_audit["multi_instance"], errors="coerce").mean()),
        "class_coverage": len(set(initial_counts) & set(NEU6)),
        "class_entropy_instances": entropy(initial_counts),
        "class_counts": json.dumps(dict(sorted(initial_counts.items())), ensure_ascii=False),
        **geometry_for_batch(initial, [], lookup),
    }])
    initial_stats.to_csv(out / "initial_set_stats.csv", index=False)

    selected_parts, geometry_rows, xml_stats_rows = [], [], []
    prior_by_strategy = {s: list(initial_names) for s in STRATEGIES}
    for round_no in range(1, 5):
        for strategy in STRATEGIES:
            rows = select_rows(v8_sel, v9_sel, strategy, round_no)
            enriched = []
            for _, row in rows.iterrows():
                item = row.to_dict()
                item.update(parse_xml(str(row["image_name"])))
                enriched.append(item)
            batch = pd.DataFrame(enriched)
            selected_parts.append(batch)
            geometry_rows.append({"strategy": strategy, "round": round_no, **geometry_for_batch(rows, prior_by_strategy[strategy], lookup)})
            class_counts: Counter[str] = Counter()
            if not batch.empty:
                for text in batch["gt_class_counts"].fillna("{}"):
                    class_counts.update(json.loads(text))
            xml_stats_rows.append({
                "strategy": strategy,
                "round": round_no,
                "n_images": len(batch),
                "total_bbox": int(pd.to_numeric(batch.get("gt_box_count"), errors="coerce").sum()) if not batch.empty else 0,
                "bbox_per_image": float(pd.to_numeric(batch.get("gt_box_count"), errors="coerce").mean()) if not batch.empty else np.nan,
                "multi_instance_rate": float(pd.to_numeric(batch.get("multi_instance"), errors="coerce").mean()) if not batch.empty else np.nan,
                "class_coverage": len(set(class_counts) & set(NEU6)),
                "class_entropy_instances": entropy(class_counts),
                "class_counts": json.dumps(dict(sorted(class_counts.items())), ensure_ascii=False),
                "bbox_area_fraction_mean": float(pd.to_numeric(batch.get("gt_area_frac_mean"), errors="coerce").mean()) if not batch.empty else np.nan,
            })
            prior_by_strategy[strategy].extend(rows["image_name"].astype(str).tolist())

    selected = pd.concat(selected_parts, ignore_index=True, sort=False)
    selected.to_csv(out / "selected_images.csv", index=False)
    geometry = pd.DataFrame(geometry_rows)
    geometry.to_csv(out / "dino_geometry.csv", index=False)
    xml_stats = pd.DataFrame(xml_stats_rows)
    xml_stats.to_csv(out / "xml_batch_stats.csv", index=False)

    round1_sets = {
        s: set(selected[(selected["strategy"].eq(s)) & pd.to_numeric(selected["round"], errors="coerce").eq(1)]["image_name"].astype(str))
        for s in STRATEGIES
    }
    overlap_rows = []
    for left in STRATEGIES:
        for right in STRATEGIES:
            inter, union = round1_sets[left] & round1_sets[right], round1_sets[left] | round1_sets[right]
            overlap_rows.append({"left": left, "right": right, "overlap_count": len(inter), "jaccard": len(inter) / len(union) if union else np.nan, "overlap_images": "|".join(sorted(inter))})
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(out / "selection_overlap.csv", index=False)

    v8_results, v9_results = read_csv(V8 / "all_round_results.csv"), read_csv(V9B / "all_round_results.csv")
    metrics = metric_table(v8_results, v9_results)
    metrics.to_csv(out / "round_metrics.csv", index=False)
    multi = multiseed_round1(v8_results, v9_results)
    multi.to_csv(out / "round1_multiseed_context.csv", index=False)

    status, per_class = per_class_available(read_csv(V8 / "per_class_metrics_by_round.csv"), read_csv(V9B / "per_class_metrics_by_round.csv"))
    status.to_csv(out / "per_class_artifact_status.csv", index=False)
    per_class.to_csv(out / "per_class_deltas.csv", index=False)

    r1 = metrics[metrics["round"].eq(1)].set_index("strategy")
    later = metrics.pivot(index="round", columns="strategy", values="map5095")
    geom1 = geometry[geometry["round"].eq(1)].set_index("strategy")
    xml1 = xml_stats[xml_stats["round"].eq(1)].set_index("strategy")
    multi_means = multi[["visual_minus_random", "instance_minus_random"]].mean()
    overlap_vi = overlap[(overlap["left"].eq(VISUAL)) & (overlap["right"].eq(INSTANCE))].iloc[0]
    per_class_note = (
        "Existing per-class CSVs contain no AP values, so per-class AP change is not recoverable without a new validation pass. "
        "This audit deliberately does not run that pass."
        if per_class.empty else "Per-class deltas were recovered from existing CSV values."
    )
    summary = f"""# Seed45 cold-start post-hoc case study

- Training performed: `False`
- New detector inference/evaluation performed: `False`
- Final test used/read: `False`
- GT/XML usage: `post-hoc descriptive audit only`
- DINO cache: `{dino_model}`
- V8 source: `{V8}`
- V9b source: `{V9B}` (uses V8 as its source run)

## Protocol checks

{markdown(protocol)}

## Confirmed round1 result

| strategy | mAP50-95 | delta vs Random |
|---|---:|---:|
| Random | {r1.loc[RANDOM, 'map5095']:.6f} | 0.000000 |
| Visual Diversity | {r1.loc[VISUAL, 'map5095']:.6f} | {r1.loc[VISUAL, 'map5095_minus_random_same_round']:+.6f} |
| Instance-rich DINO | {r1.loc[INSTANCE, 'map5095']:.6f} | {r1.loc[INSTANCE, 'map5095_minus_random_same_round']:+.6f} |

The common round0 value was `{metrics[metrics['round'].eq(0)]['map5095'].iloc[0]:.6f}` at budget 15. All three round1 models used training seed 1045 and the same 177-image development split.

## Shared initial 15

{markdown(initial_audit[['rank_in_selection','image_name','gt_class_counts','gt_box_count','gt_area_frac_mean']])}

{markdown(initial_stats)}

## Round1 selections and overlap

{markdown(selected[pd.to_numeric(selected['round'], errors='coerce').eq(1)][['strategy','rank_in_selection','image_name','gt_class_counts','gt_box_count','gt_area_frac_mean']])}

Visual and Instance-rich overlap on `{int(overlap_vi['overlap_count'])}/5` images (`{overlap_vi['overlap_images']}`); their Jaccard is `{overlap_vi['jaccard']:.3f}`. The full pair matrix is in `selection_overlap.csv`.

## XML composition (post-hoc only)

{markdown(xml_stats[xml_stats['round'].eq(1)])}

## DINO geometry

{markdown(geometry[geometry['round'].eq(1)])}

At round1, a larger `distance_to_prior_mean` means the new batch is farther from the initial 15. A lower pairwise cosine similarity means less within-batch redundancy. These are descriptive properties, not causal estimates.

## Why the early advantage did not persist

{markdown(metrics)}

- At round2, Random reaches `{later.loc[2, RANDOM]:.6f}`, versus Visual `{later.loc[2, VISUAL]:.6f}` and Instance-rich `{later.loc[2, INSTANCE]:.6f}`.
- At round3, Instance-rich is ahead of Random by `{later.loc[3, INSTANCE] - later.loc[3, RANDOM]:+.6f}`, showing that the trajectory is not a monotonic selector effect.
- At round4, Random is ahead of Visual by `{later.loc[4, RANDOM] - later.loc[4, VISUAL]:+.6f}` and ahead of Instance-rich by `{later.loc[4, RANDOM] - later.loc[4, INSTANCE]:+.6f}`.
- Across all five seeds, the mean round1 differences are Visual `{multi_means['visual_minus_random']:+.6f}` and Instance-rich `{multi_means['instance_minus_random']:+.6f}`. Seed45 should therefore be treated as a case-generating observation, not confirmatory evidence.

The defensible post-hoc interpretation is that two different DINO-informed selectors found a useful early batch for seed45, but the benefit was transient and path-dependent. Random's later acquisitions supplied complementary coverage and caught up by budget 25. The non-monotonic Instance-rich curve (ahead again at budget 30, then substantially worse at 35) is also consistent with small-sample training/selection variance. None of these associations identifies a causal image property by itself.

## Per-class AP limitation

{per_class_note}

`per_class_artifact_status.csv` records this explicitly; `per_class_deltas.csv` is intentionally empty when the stored AP values are absent. Recovering per-class AP would require a separately authorized development-only validation pass from saved weights, never final test.

## Decision for the research direction

Seed45 is sufficient to justify a narrowly scoped **cold-start annotation triage hypothesis**, because two distinct DINO-informed strategies each improved round1 mAP50-95 by about 0.023--0.025 over the same Random batch. It is not sufficient to claim a winning selector: the advantage is not stable across later budgets, and the five-seed round1 context must remain primary. The next research step should therefore be a pre-registered cold-start analysis/validation protocol focused on early slope and annotation prioritization, not further tuning on this seed.
"""
    (out / "seed45_cold_start_case_summary.md").write_text(summary, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

"""
V10b selection-only probe from an existing V10 NEU large-pool smoke run.

Purpose
-------
Reuse the completed V10 smoke artifacts and compare:

1. Existing V9b query
   - detector uncertainty: 0.20
   - DINO diversity:       0.25
   - predicted balance:    0.15
   - pseudo instances:     0.40

2. Proposed V10b query
   - detector uncertainty: 0.25
   - DINO diversity:       0.35
   - predicted balance:    0.15
   - pseudo instances:     0.25

This script:
- DOES NOT train YOLO.
- DOES NOT regenerate DINO embeddings.
- DOES NOT evaluate or read the final-test split.
- Reuses the existing acquisition pool, initial set, detector scores,
  V9b selection, and DINO cache from the source V10 run.
- Uses XML only after selection for post-hoc diagnostics.

Expected source run
-------------------
runs/active_learning_ablation_v10_neu_large_pool/
v10_neu_large_pool_smoke_20260712_185923

Environment variables
---------------------
AL_V10_SOURCE_RUN
AL_V10B_W_UNCERTAINTY   default 0.25
AL_V10B_W_DINO          default 0.35
AL_V10B_W_BALANCE       default 0.15
AL_V10B_W_INSTANCE      default 0.25
AL_V9_CANDIDATE_FRACTION
AL_V9B_MAX_NO_BOX
AL_V9B_MIN_PSEUDO_BOXES
AL_V9B_MAX_PER_PRED_CLASS
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# These defaults are intentionally safe. No downloads, training, or final-test use.
os.environ.setdefault("AL_ALLOW_MODEL_DOWNLOAD", "0")
os.environ.setdefault("AL_REUSE_EXISTING_EMBEDDING_CACHE", "1")
os.environ.setdefault("AL_FORCE_REBUILD_EMBEDDINGS", "0")

import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402
import run_v9_stage1_seed42_round1 as stage1  # noqa: E402
from analyze_instance_richness_v7 import parse_image_instances, strategy_stats  # noqa: E402
from audit_detection_pipeline_v7 import build_image_index, build_xml_index  # noqa: E402


PROJECT_ROOT = v6.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "v10b_selection_probe"

DEFAULT_SOURCE_RUN = (
    PROJECT_ROOT
    / "runs"
    / "active_learning_ablation_v10_neu_large_pool"
    / "v10_neu_large_pool_smoke_20260712_185923"
)

RANDOM = "GTFreeRandom"


def table_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required CSV is missing or empty: {path}")
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_source_run() -> Path:
    override = os.environ.get("AL_V10_SOURCE_RUN")
    if override:
        p = Path(override).expanduser()
        return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    return DEFAULT_SOURCE_RUN.resolve()


def float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def infer_v9b_strategy(config: dict[str, Any]) -> str:
    strategies = [str(v) for v in config.get("strategies", [])]
    non_random = [s for s in strategies if s != RANDOM]
    if not non_random:
        raise ValueError(f"Cannot infer V9b strategy from config strategies={strategies}")
    return non_random[0]


def load_source_artifacts(
    source_run: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = read_json(source_run / "config.json")
    if bool(config.get("final_test_used", False)):
        raise RuntimeError("Source run indicates final_test_used=True. Refusing to continue.")

    pool = read_csv(source_run / "acquisition_pool_v10.csv")
    selected = read_csv(source_run / "all_selected_samples_by_round.csv")
    detector_scores = read_csv(source_run / "v9b_round1_detector_scores.csv")
    cumulative = read_csv(source_run / "cumulative_labeled_sets_by_round.csv")

    required_pool = {"sample_id", "resolved_image_path", "image_sha256"}
    missing_pool = required_pool - set(pool.columns)
    if missing_pool:
        raise ValueError(f"Pool is missing required columns: {sorted(missing_pool)}")

    return config, pool, selected, detector_scores, cumulative


def load_existing_sets(
    config: dict[str, Any],
    selected: pd.DataFrame,
    cumulative: pd.DataFrame,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = int(config["acquisition_seed"])
    v9b_strategy = infer_v9b_strategy(config)

    seed_selected = selected[
        pd.to_numeric(selected["acquisition_seed"], errors="coerce").eq(seed)
    ].copy()

    initial = seed_selected[
        pd.to_numeric(seed_selected["round"], errors="coerce").eq(0)
    ].drop_duplicates("sample_id", keep="first").reset_index(drop=True)

    v9b_query = seed_selected[
        pd.to_numeric(seed_selected["round"], errors="coerce").eq(1)
        & seed_selected["strategy"].astype(str).eq(v9b_strategy)
        & seed_selected["selection_type"].astype(str).ne("cumulative_labeled")
    ].drop_duplicates("sample_id", keep="first").reset_index(drop=True)

    random_query = seed_selected[
        pd.to_numeric(seed_selected["round"], errors="coerce").eq(1)
        & seed_selected["strategy"].astype(str).eq(RANDOM)
        & seed_selected["selection_type"].astype(str).ne("cumulative_labeled")
    ].drop_duplicates("sample_id", keep="first").reset_index(drop=True)

    # Fallback: some runners may only have cumulative rows in one artifact.
    if initial.empty:
        initial = cumulative[
            pd.to_numeric(cumulative["acquisition_seed"], errors="coerce").eq(seed)
            & pd.to_numeric(cumulative["round"], errors="coerce").eq(0)
        ].drop_duplicates("sample_id", keep="first").reset_index(drop=True)

    if initial.empty:
        raise ValueError("Could not recover the shared round-0 initial set.")
    if v9b_query.empty:
        raise ValueError(f"Could not recover existing V9b query for strategy={v9b_strategy}")

    return v9b_strategy, initial, v9b_query, random_query


def load_embedding_lookup(
    source_run: Path,
    config: dict[str, Any],
    pool: pd.DataFrame,
) -> tuple[Path, dict[str, np.ndarray], dict[str, Any]]:
    embedding_dir = Path(str(config.get("embedding_dir", ""))).expanduser()
    if not embedding_dir.is_absolute():
        embedding_dir = (PROJECT_ROOT / embedding_dir).resolve()

    # Prefer the exact cache recorded in the source run.
    used_txt = source_run / "embedding_cache_used.txt"
    if used_txt.exists():
        txt_path = Path(used_txt.read_text(encoding="utf-8").strip()).expanduser()
        if txt_path.exists():
            embedding_dir = txt_path.resolve()

    manifest_path = embedding_dir / "embedding_manifest.csv"
    embeddings_path = embedding_dir / "embeddings.npy"
    config_path = embedding_dir / "embedding_config.json"

    if not manifest_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            "Existing DINO cache is incomplete. This script never rebuilds embeddings.\n"
            f"manifest={manifest_path}\nembeddings={embeddings_path}"
        )

    manifest = pd.read_csv(manifest_path)
    embeddings = np.load(embeddings_path)
    emb_config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )

    if len(manifest) != embeddings.shape[0]:
        raise ValueError(
            f"Embedding manifest/array mismatch: {len(manifest)} vs {embeddings.shape}"
        )

    lookup: dict[str, np.ndarray] = {}

    # Most robust mapping: image SHA256.
    if "sha256" in manifest.columns:
        sha_to_embedding: dict[str, np.ndarray] = {}
        for _, row in manifest.iterrows():
            idx = int(row.get("embedding_index", row.name))
            sha_to_embedding[str(row["sha256"])] = embeddings[idx]
        for _, row in pool.iterrows():
            emb = sha_to_embedding.get(str(row["image_sha256"]))
            if emb is not None:
                lookup[str(row["sample_id"])] = emb

    # Fallback mapping for the exact V10 runner cache layout.
    if len(lookup) != len(pool):
        lookup = {}
        if len(manifest) != len(pool):
            raise ValueError(
                "Could not map all embeddings by SHA256 and row counts differ: "
                f"mapped={len(lookup)} manifest={len(manifest)} pool={len(pool)}"
            )
        for i, row in pool.reset_index(drop=True).iterrows():
            emb_idx = int(manifest.iloc[i].get("embedding_index", i))
            lookup[str(row["sample_id"])] = embeddings[emb_idx]

    if len(lookup) != len(pool):
        raise ValueError(f"Embedding lookup incomplete: {len(lookup)} / {len(pool)}")

    return embedding_dir, lookup, emb_config


def min_cosine_distance(
    sample_id: str,
    reference_ids: list[str],
    embedding_lookup: dict[str, np.ndarray],
) -> float:
    emb = embedding_lookup.get(str(sample_id))
    if emb is None:
        return np.nan
    refs = [embedding_lookup[str(sid)] for sid in reference_ids if str(sid) in embedding_lookup]
    if not refs:
        return 1.0
    ref_mat = np.vstack(refs)
    return float(1.0 - np.max(ref_mat @ emb))


def pairwise_cosine_similarity(
    sample_ids: list[str],
    embedding_lookup: dict[str, np.ndarray],
) -> float:
    vecs = [embedding_lookup[str(s)] for s in sample_ids if str(s) in embedding_lookup]
    if len(vecs) < 2:
        return np.nan
    mat = np.vstack(vecs)
    sims = mat @ mat.T
    upper = sims[np.triu_indices(len(vecs), k=1)]
    return float(np.mean(upper)) if len(upper) else np.nan


def normalize(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    if len(x) == 0:
        return x
    span = float(x.max() - x.min())
    if not np.isfinite(span) or span <= 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - x.min()) / span


def initial_predicted_class_counts(
    initial: pd.DataFrame,
    scored_pool: pd.DataFrame,
) -> Counter:
    # Initial samples are already labeled in active learning. Actual class_hint is
    # therefore permitted for deficit initialization, matching the V9/V10 protocol.
    if "class_hint" in initial.columns:
        return Counter(initial["class_hint"].dropna().astype(str))
    # Defensive fallback only.
    merged = initial[["sample_id"]].merge(
        scored_pool[["sample_id", "detector_pred_class"]],
        on="sample_id",
        how="left",
    )
    return Counter(merged["detector_pred_class"].dropna().astype(str))


def static_component_diagnostics(
    scored_pool: pd.DataFrame,
    initial: pd.DataFrame,
    embedding_lookup: dict[str, np.ndarray],
) -> pd.DataFrame:
    out = scored_pool.copy()
    initial_ids = initial["sample_id"].astype(str).tolist()

    out["static_dino_distance_to_initial"] = [
        min_cosine_distance(sid, initial_ids, embedding_lookup)
        for sid in out["sample_id"].astype(str)
    ]

    for raw, norm in [
        ("detector_uncertainty", "detector_uncertainty_norm_static"),
        ("static_dino_distance_to_initial", "dino_distance_norm_static"),
        ("pseudo_instance_score", "pseudo_instance_norm_static"),
    ]:
        if raw in out.columns:
            out[norm] = normalize(out[raw])

    return out


def selection_component_summary(
    static_scores: pd.DataFrame,
    selections: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = [
        "detector_uncertainty",
        "static_dino_distance_to_initial",
        "pseudo_instance_score",
        "detector_pseudo_box_count",
        "detector_max_conf",
    ]
    for strategy, selected in selections.items():
        merged = selected[["sample_id"]].merge(static_scores, on="sample_id", how="left")
        for col in columns:
            if col not in merged.columns:
                continue
            values = pd.to_numeric(merged[col], errors="coerce").dropna()
            rows.append(
                {
                    "strategy": strategy,
                    "component": col,
                    "n": int(len(values)),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "min": float(values.min()) if len(values) else np.nan,
                    "median": float(values.median()) if len(values) else np.nan,
                    "max": float(values.max()) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def predicted_class_distribution(selections: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for strategy, selected in selections.items():
        counts = selected.get(
            "detector_pred_class",
            pd.Series(["__missing__"] * len(selected)),
        ).fillna("__missing__").astype(str).value_counts()
        for cls, count in counts.items():
            rows.append(
                {
                    "strategy": strategy,
                    "detector_pred_class": cls,
                    "selected_images": int(count),
                }
            )
    return pd.DataFrame(rows)


def posthoc_stats(
    selections: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_index = build_image_index()
    xml_index = build_xml_index()

    instance_frames = []
    stats_frames = []
    class_rows = []

    for strategy, selected in selections.items():
        selected_with_source = selected.copy()
        selected_with_source["source_strategy"] = strategy

        rows = []
        for _, row in selected_with_source.iterrows():
            rows.extend(parse_image_instances(row, image_index, xml_index))
        instances = pd.DataFrame(rows)

        if not instances.empty:
            instances["source_strategy"] = strategy
            for cls, count in (
                instances["actual_xml_class"].astype(str).value_counts().items()
                if "actual_xml_class" in instances.columns
                else []
            ):
                class_rows.append(
                    {
                        "strategy": strategy,
                        "actual_xml_class": cls,
                        "actual_bbox_instances": int(count),
                    }
                )
        instance_frames.append(instances)

        stats = strategy_stats(selected_with_source, instances)
        if not stats.empty:
            if "source_strategy" not in stats.columns:
                stats.insert(0, "source_strategy", strategy)
            stats_frames.append(stats)

    all_instances = (
        pd.concat(instance_frames, ignore_index=True, sort=False)
        if instance_frames
        else pd.DataFrame()
    )
    all_stats = (
        pd.concat(stats_frames, ignore_index=True, sort=False)
        if stats_frames
        else pd.DataFrame()
    )
    class_dist = pd.DataFrame(class_rows)
    return all_instances, all_stats, class_dist


def overlap_tables(
    v9b: pd.DataFrame,
    v10b: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    v9_ids = set(v9b["sample_id"].astype(str))
    v10_ids = set(v10b["sample_id"].astype(str))
    union = v9_ids | v10_ids
    overlap = v9_ids & v10_ids

    summary = pd.DataFrame(
        [
            {
                "strategy_a": "V9b",
                "strategy_b": "V10b",
                "size_a": len(v9_ids),
                "size_b": len(v10_ids),
                "overlap_count": len(overlap),
                "v9b_only_count": len(v9_ids - v10_ids),
                "v10b_only_count": len(v10_ids - v9_ids),
                "jaccard": len(overlap) / len(union) if union else np.nan,
            }
        ]
    )

    rows = []
    for group, ids, frame in [
        ("overlap", overlap, pd.concat([v9b, v10b], ignore_index=True)),
        ("v9b_only", v9_ids - v10_ids, v9b),
        ("v10b_only", v10_ids - v9_ids, v10b),
    ]:
        sub = frame[frame["sample_id"].astype(str).isin(ids)].drop_duplicates("sample_id")
        for _, row in sub.iterrows():
            rows.append(
                {
                    "group": group,
                    "sample_id": row.get("sample_id", ""),
                    "dataset_type": row.get("dataset_type", ""),
                    "image_name": row.get("image_name", ""),
                    "resolved_image_path": row.get("resolved_image_path", ""),
                    "detector_pred_class": row.get("detector_pred_class", ""),
                    "detector_pseudo_box_count": row.get("detector_pseudo_box_count", np.nan),
                    "detector_uncertainty": row.get("detector_uncertainty", np.nan),
                    "pseudo_instance_score": row.get("pseudo_instance_score", np.nan),
                }
            )
    return summary, pd.DataFrame(rows)


def selection_geometry_summary(
    initial: pd.DataFrame,
    selections: dict[str, pd.DataFrame],
    embedding_lookup: dict[str, np.ndarray],
) -> pd.DataFrame:
    initial_ids = initial["sample_id"].astype(str).tolist()
    rows = []
    for strategy, selected in selections.items():
        ids = selected["sample_id"].astype(str).tolist()
        distances = [
            min_cosine_distance(sid, initial_ids, embedding_lookup) for sid in ids
        ]
        rows.append(
            {
                "strategy": strategy,
                "selected_images": len(ids),
                "batch_pairwise_cosine_similarity_mean": pairwise_cosine_similarity(
                    ids, embedding_lookup
                ),
                "distance_to_initial_mean": float(np.nanmean(distances))
                if len(distances)
                else np.nan,
                "distance_to_initial_std": float(np.nanstd(distances, ddof=1))
                if len(distances) > 1
                else np.nan,
                "distance_to_initial_min": float(np.nanmin(distances))
                if len(distances)
                else np.nan,
                "distance_to_initial_max": float(np.nanmax(distances))
                if len(distances)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def saturation_summary(scored_pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "pseudo_instance_score" in scored_pool.columns:
        vals = pd.to_numeric(
            scored_pool["pseudo_instance_score"], errors="coerce"
        ).dropna()
        max_value = float(vals.max()) if len(vals) else np.nan
        rows.append(
            {
                "metric": "pseudo_instance_score",
                "pool_n": int(len(vals)),
                "min": float(vals.min()) if len(vals) else np.nan,
                "median": float(vals.median()) if len(vals) else np.nan,
                "max": max_value,
                "share_equal_to_1": float((vals >= 1.0 - 1e-12).mean())
                if len(vals)
                else np.nan,
                "share_equal_to_observed_max": float(
                    (vals >= max_value - 1e-12).mean()
                )
                if len(vals)
                else np.nan,
                "unique_values": int(vals.nunique()) if len(vals) else 0,
            }
        )
    if "detector_pseudo_box_count" in scored_pool.columns:
        vals = pd.to_numeric(
            scored_pool["detector_pseudo_box_count"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "metric": "detector_pseudo_box_count",
                "pool_n": int(len(vals)),
                "min": float(vals.min()) if len(vals) else np.nan,
                "median": float(vals.median()) if len(vals) else np.nan,
                "max": float(vals.max()) if len(vals) else np.nan,
                "share_equal_to_1": np.nan,
                "share_equal_to_observed_max": float(
                    (vals >= vals.max() - 1e-12).mean()
                )
                if len(vals)
                else np.nan,
                "unique_values": int(vals.nunique()) if len(vals) else 0,
            }
        )
    return pd.DataFrame(rows)


def recommend_training(overlap: pd.DataFrame, geometry: pd.DataFrame) -> dict[str, Any]:
    jaccard = float(overlap.iloc[0]["jaccard"]) if len(overlap) else np.nan
    geom = geometry.set_index("strategy") if len(geometry) else pd.DataFrame()

    v9_sim = (
        float(geom.loc["V9b", "batch_pairwise_cosine_similarity_mean"])
        if "V9b" in geom.index
        else np.nan
    )
    v10_sim = (
        float(geom.loc["V10b", "batch_pairwise_cosine_similarity_mean"])
        if "V10b" in geom.index
        else np.nan
    )
    v9_dist = (
        float(geom.loc["V9b", "distance_to_initial_mean"])
        if "V9b" in geom.index
        else np.nan
    )
    v10_dist = (
        float(geom.loc["V10b", "distance_to_initial_mean"])
        if "V10b" in geom.index
        else np.nan
    )

    conditions = {
        "jaccard_below_0_70": bool(np.isfinite(jaccard) and jaccard < 0.70),
        "at_least_10_images_changed": bool(
            len(overlap)
            and int(overlap.iloc[0]["v10b_only_count"]) >= 10
        ),
        "batch_redundancy_not_worse": bool(
            np.isfinite(v9_sim) and np.isfinite(v10_sim) and v10_sim <= v9_sim
        ),
        "distance_to_initial_not_worse": bool(
            np.isfinite(v9_dist) and np.isfinite(v10_dist) and v10_dist >= v9_dist
        ),
    }
    passed = sum(bool(v) for v in conditions.values())
    return {
        "recommend_single_v10b_training": passed >= 3,
        "passed_conditions": passed,
        "total_conditions": len(conditions),
        "conditions": conditions,
        "note": (
            "This is a selection-only gate. It does not claim detector improvement."
        ),
    }


def write_summary(
    save_dir: Path,
    *,
    source_run: Path,
    source_config: dict[str, Any],
    v9b_weights: dict[str, float],
    v10b_weights: dict[str, float],
    overlap: pd.DataFrame,
    geometry: pd.DataFrame,
    component_summary: pd.DataFrame,
    saturation: pd.DataFrame,
    posthoc: pd.DataFrame,
    actual_class: pd.DataFrame,
    recommendation: dict[str, Any],
) -> None:
    lines = [
        "# V10b selection-only probe",
        "",
        "No YOLO training. No DINO regeneration. Final test was not read or evaluated.",
        "",
        "## Source run",
        "",
        f"- `{source_run}`",
        f"- acquisition seed: {source_config.get('acquisition_seed')}",
        f"- initial size: {source_config.get('initial_seed_size')}",
        f"- query size: {source_config.get('query_size')}",
        "",
        "## Weight comparison",
        "",
        "### Existing V9b",
        "",
        "```json",
        json.dumps(v9b_weights, ensure_ascii=False, indent=2),
        "```",
        "",
        "### Proposed V10b",
        "",
        "```json",
        json.dumps(v10b_weights, ensure_ascii=False, indent=2),
        "```",
        "",
        "## V9b vs V10b overlap",
        "",
        table_md(overlap),
        "",
        "## Selection geometry",
        "",
        table_md(geometry),
        "",
        "## Selected component distributions",
        "",
        table_md(component_summary),
        "",
        "## Pool saturation diagnostics",
        "",
        table_md(saturation),
        "",
        "## Post-hoc actual XML statistics",
        "",
        table_md(posthoc),
        "",
        "## Post-hoc actual class distribution",
        "",
        table_md(actual_class),
        "",
        "## Selection-only gate recommendation",
        "",
        "```json",
        json.dumps(recommendation, ensure_ascii=False, indent=2),
        "```",
        "",
        "A positive gate only means that V10b changed the selected batch enough to justify one"
        " development-set training run. It does not mean V10b is better than Random.",
    ]
    (save_dir / "v10b_selection_probe_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"v10b_selection_probe_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=False)

    source_run = resolve_source_run()
    config, pool, selected, detector_scores, cumulative = load_source_artifacts(
        source_run
    )
    v9b_strategy, initial, existing_v9b, random_query = load_existing_sets(
        config, selected, cumulative
    )

    source_seed = int(config["acquisition_seed"])
    query_size = int(config["query_size"])
    candidate_fraction = float(
        os.environ.get(
            "AL_V9_CANDIDATE_FRACTION",
            str(config.get("candidate_fraction", 1.0)),
        )
    )

    constraints = config.get("v9b_constraints", {})
    max_no_box = int_env(
        "AL_V9B_MAX_NO_BOX", int(constraints.get("max_no_box", 0))
    )
    min_pseudo_boxes = int_env(
        "AL_V9B_MIN_PSEUDO_BOXES",
        int(constraints.get("min_pseudo_boxes", 2)),
    )
    max_per_pred_class = int_env(
        "AL_V9B_MAX_PER_PRED_CLASS",
        int(constraints.get("max_per_pred_class", 2)),
    )

    v9b_weights = {
        k: float(v)
        for k, v in config.get("stage1b_instance_rich_weights", {}).items()
    }
    v10b_weights = {
        "detector_uncertainty": float_env("AL_V10B_W_UNCERTAINTY", 0.25),
        "dino_visual_distance": float_env("AL_V10B_W_DINO", 0.35),
        "predicted_class_deficit": float_env("AL_V10B_W_BALANCE", 0.15),
        "pseudo_instance_count": float_env("AL_V10B_W_INSTANCE", 0.25),
    }

    total_weight = sum(v10b_weights.values())
    if not np.isclose(total_weight, 1.0, atol=1e-8):
        raise ValueError(
            f"V10b weights must sum to 1.0, got {total_weight}: {v10b_weights}"
        )

    embedding_dir, embedding_lookup, embedding_config = load_embedding_lookup(
        source_run, config, pool
    )

    current_pool = pool[
        ~pool["sample_id"].astype(str).isin(initial["sample_id"].astype(str))
    ].copy()
    scored_pool = current_pool.merge(
        detector_scores.drop_duplicates("sample_id", keep="first"),
        on="sample_id",
        how="left",
        suffixes=("", "_score"),
    )

    missing_scores = scored_pool["detector_uncertainty"].isna().sum()
    if missing_scores:
        raise ValueError(
            f"{missing_scores} unlabeled samples are missing detector scores."
        )

    # Re-score the same pool with the proposed V10b weights only.
    v10b_query = stage1.select_instance_rich_dino_balanced(
        scored_pool,
        initial,
        embedding_lookup,
        query_size=query_size,
        candidate_fraction=candidate_fraction,
        weights=v10b_weights,
        max_no_box=max_no_box,
        min_pseudo_boxes=min_pseudo_boxes,
        max_per_pred_class=max_per_pred_class,
    ).drop_duplicates("sample_id", keep="first").reset_index(drop=True)

    if len(v10b_query) != query_size:
        raise ValueError(
            f"V10b selector returned {len(v10b_query)} samples; expected {query_size}"
        )

    existing_v9b = existing_v9b.merge(
        detector_scores.drop_duplicates("sample_id", keep="first"),
        on="sample_id",
        how="left",
        suffixes=("", "_score"),
    )

    static_scores = static_component_diagnostics(
        scored_pool, initial, embedding_lookup
    )
    selections = {
        "V9b": existing_v9b,
        "V10b": v10b_query,
    }
    if not random_query.empty:
        selections["Random"] = random_query.merge(
            detector_scores.drop_duplicates("sample_id", keep="first"),
            on="sample_id",
            how="left",
            suffixes=("", "_score"),
        )

    overlap, overlap_samples = overlap_tables(existing_v9b, v10b_query)
    geometry = selection_geometry_summary(initial, selections, embedding_lookup)
    component_summary = selection_component_summary(static_scores, selections)
    pred_class = predicted_class_distribution(selections)
    instances, actual_stats, actual_class = posthoc_stats(selections)
    saturation = saturation_summary(scored_pool)
    recommendation = recommend_training(overlap, geometry)

    # Persist reproducible inputs and outputs.
    (save_dir / "config.json").write_text(
        json.dumps(
            {
                "status": "selection_only",
                "final_test_used": False,
                "yolo_training_run": False,
                "dino_regenerated": False,
                "source_run": str(source_run),
                "source_experiment_id": config.get("experiment_id"),
                "acquisition_seed": source_seed,
                "query_size": query_size,
                "candidate_fraction": candidate_fraction,
                "source_v9b_strategy": v9b_strategy,
                "v9b_weights": v9b_weights,
                "v10b_weights": v10b_weights,
                "constraints": {
                    "max_no_box": max_no_box,
                    "min_pseudo_boxes": min_pseudo_boxes,
                    "max_per_pred_class": max_per_pred_class,
                },
                "embedding_dir": str(embedding_dir),
                "embedding_config": embedding_config,
                "recommendation": recommendation,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    initial.to_csv(
        save_dir / "source_initial_60.csv", index=False, encoding="utf-8-sig"
    )
    existing_v9b.to_csv(
        save_dir / "v9b_selected_samples.csv", index=False, encoding="utf-8-sig"
    )
    v10b_query.to_csv(
        save_dir / "v10b_selected_samples.csv", index=False, encoding="utf-8-sig"
    )
    overlap.to_csv(
        save_dir / "v9b_v10b_selection_overlap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overlap_samples.to_csv(
        save_dir / "v9b_v10b_changed_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    component_summary.to_csv(
        save_dir / "selection_component_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    saturation.to_csv(
        save_dir / "pool_score_saturation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pred_class.to_csv(
        save_dir / "predicted_class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    geometry.to_csv(
        save_dir / "selection_geometry_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    instances.to_csv(
        save_dir / "posthoc_actual_instances.csv",
        index=False,
        encoding="utf-8-sig",
    )
    actual_stats.to_csv(
        save_dir / "posthoc_actual_instance_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )
    actual_class.to_csv(
        save_dir / "posthoc_actual_class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_summary(
        save_dir,
        source_run=source_run,
        source_config=config,
        v9b_weights=v9b_weights,
        v10b_weights=v10b_weights,
        overlap=overlap,
        geometry=geometry,
        component_summary=component_summary,
        saturation=saturation,
        posthoc=actual_stats,
        actual_class=actual_class,
        recommendation=recommendation,
    )

    print("=" * 100)
    print("[DONE] V10b selection-only probe finished")
    print(f"Output dir : {save_dir}")
    print(f"Source run : {source_run}")
    print(f"V9b/V10b Jaccard: {float(overlap.iloc[0]['jaccard']):.4f}")
    print(
        "Recommend one V10b training:",
        recommendation["recommend_single_v10b_training"],
    )
    print("No YOLO training. No DINO regeneration. Final test used: False.")
    print("=" * 100)


if __name__ == "__main__":
    main()

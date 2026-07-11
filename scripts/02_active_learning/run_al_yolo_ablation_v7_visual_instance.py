"""V7 visual/instance-aware active-learning screening runner.

Stage A default:
    - selection only
    - no YOLO training
    - development eval only
    - final_test_v7 is never used

GT-free strategies in this file operate on a restricted dataframe view that
excludes class_hint, XML labels, and actual bbox statistics.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402
from analyze_instance_richness_v7 import parse_image_instances, strategy_stats  # noqa: E402
from audit_detection_pipeline_v7 import build_image_index, build_xml_index, load_priority_scores  # noqa: E402
from experiment_registry_v7 import append_registry_row  # noqa: E402


PROJECT_ROOT = v6.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "active_learning_ablation_v7_visual_instance"

PROHIBITED_GTFREE_COLUMNS = {
    "class_hint",
    "actual_xml_class",
    "actual_bbox_count",
    "num_xml_instances",
    "xml_mapped_classes",
    "primary_xml_class",
    "map50",
    "map5095",
}

HYBRID_CANDIDATES = {
    "A": {"alpha": 0.50, "beta": 0.50, "gamma": 0.00},
    "B": {"alpha": 0.40, "beta": 0.40, "gamma": 0.20},
    "C": {"alpha": 0.25, "beta": 0.50, "gamma": 0.25},
}

DEFAULT_STAGE_A_STRATEGIES = [
    "GTFreeRandom",
    "GTFreeDatasetBalancedConsistency",
    "GTFreeDatasetBalancedVisualDiversity",
    "GTFreeDatasetBalancedConsistencyVisualDiversity",
]


def parse_csv_env(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return default
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_int_list_env(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if not value:
        return default
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def normalize(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    span = values.max() - values.min()
    if not np.isfinite(span) or span <= 1e-12:
        return pd.Series([0.0] * len(values), index=values.index)
    return (values - values.min()) / span


def make_gtfree_view(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in PROHIBITED_GTFREE_COLUMNS if c in df.columns], errors="ignore").copy()


def assert_gtfree_view(df: pd.DataFrame) -> None:
    forbidden = sorted(PROHIBITED_GTFREE_COLUMNS.intersection(df.columns))
    if forbidden:
        raise AssertionError(f"GT-free acquisition view contains prohibited columns: {forbidden}")


def sample_key(row: pd.Series) -> tuple[str, str]:
    return str(row["dataset_type"]), str(row["image_name"])


def latest_embedding_dir() -> Path | None:
    override = os.environ.get("VISUAL_EMBEDDING_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    root = PROJECT_ROOT / "outputs" / "visual_embeddings_v7"
    if not root.exists():
        return None
    backend_filter = (
        os.environ.get("AL_EMBEDDING_BACKEND_FOR_SELECTION")
        or os.environ.get("AL_EMBEDDING_BACKEND")
        or ""
    ).strip().lower()
    runs = [p for p in root.iterdir() if p.is_dir()]
    if backend_filter:
        runs = [p for p in runs if p.name.lower().startswith(f"{backend_filter}_")]
    valid_runs = []
    for p in runs:
        manifest_path = p / "embedding_manifest.csv"
        embeddings_path = p / "embeddings.npy"
        config_path = p / "embedding_config.json"
        if not manifest_path.exists() or not embeddings_path.exists():
            continue
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if config.get("status") not in {None, "success"}:
                    continue
            except Exception:
                continue
        valid_runs.append(p)
    runs = valid_runs
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def load_embeddings() -> tuple[pd.DataFrame, np.ndarray, dict] | tuple[None, None, None]:
    embedding_dir = latest_embedding_dir()
    if embedding_dir is None:
        return None, None, None
    manifest_path = embedding_dir / "embedding_manifest.csv"
    embeddings_path = embedding_dir / "embeddings.npy"
    config_path = embedding_dir / "embedding_config.json"
    if not manifest_path.exists() or not embeddings_path.exists():
        return None, None, None
    manifest = pd.read_csv(manifest_path)
    embeddings = np.load(embeddings_path)
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return manifest, embeddings, config


def build_embedding_lookup(manifest: pd.DataFrame, embeddings: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    lookup = {}
    for _, row in manifest.iterrows():
        idx = int(row["embedding_index"])
        lookup[(str(row["dataset_type"]), str(row["image_name"]))] = embeddings[idx]
    return lookup


def cosine_min_distance(
    row: pd.Series,
    reference_df: pd.DataFrame,
    embedding_lookup: dict[tuple[str, str], np.ndarray],
) -> float:
    emb = embedding_lookup.get(sample_key(row))
    if emb is None:
        return 0.0
    refs = []
    for _, ref in reference_df.iterrows():
        ref_emb = embedding_lookup.get(sample_key(ref))
        if ref_emb is not None:
            refs.append(ref_emb)
    if not refs:
        return 1.0
    ref_mat = np.vstack(refs)
    sims = ref_mat @ emb
    return float(1.0 - np.max(sims))


def choose_dataset_by_deficit(current_pool: pd.DataFrame, labeled_df: pd.DataFrame, selected_df: pd.DataFrame) -> str:
    available = sorted(current_pool["dataset_type"].dropna().astype(str).unique())
    if not available:
        return str(current_pool.iloc[0]["dataset_type"])
    labeled_counts = labeled_df["dataset_type"].astype(str).value_counts().to_dict() if len(labeled_df) else {}
    selected_counts = selected_df["dataset_type"].astype(str).value_counts().to_dict() if len(selected_df) else {}
    total_after = sum(labeled_counts.get(ds, 0) + selected_counts.get(ds, 0) for ds in available) + 1
    target = total_after / max(1, len(available))
    ranked = sorted(
        available,
        key=lambda ds: (-(target - (labeled_counts.get(ds, 0) + selected_counts.get(ds, 0))), ds),
    )
    return ranked[0]


def score_candidates(
    candidates: pd.DataFrame,
    labeled_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    embedding_lookup: dict[tuple[str, str], np.ndarray] | None,
    alpha: float,
    beta: float,
    gamma: float,
    use_consistency: bool,
    use_visual: bool,
    use_pseudo_instance: bool,
) -> pd.Series:
    parts = []
    if use_consistency:
        parts.append(alpha * normalize(candidates["score_consistency_only"]))
    if use_visual:
        if embedding_lookup is None:
            raise FileNotFoundError("Visual embedding cache is required for visual diversity strategies.")
        reference = pd.concat([labeled_df, selected_df], ignore_index=True) if len(selected_df) else labeled_df
        distances = pd.Series(
            [cosine_min_distance(row, reference, embedding_lookup) for _, row in candidates.iterrows()],
            index=candidates.index,
            dtype=float,
        )
        parts.append(beta * normalize(distances))
    if use_pseudo_instance:
        # GT-free proxy only.  This must not use XML bbox counts.
        col = "pseudo_box_count" if "pseudo_box_count" in candidates.columns else None
        proxy = candidates[col] if col else pd.Series([0.0] * len(candidates), index=candidates.index)
        parts.append(gamma * normalize(proxy))
    if not parts:
        return pd.Series([0.0] * len(candidates), index=candidates.index)
    return sum(parts)


def dataset_balanced_utility_select(
    current_pool: pd.DataFrame,
    labeled_df: pd.DataFrame,
    sample_size: int,
    seed: int,
    round_idx: int,
    embedding_lookup: dict[tuple[str, str], np.ndarray] | None,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    use_consistency: bool,
    use_visual: bool,
    use_pseudo_instance: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + round_idx * 1009)
    remaining = v6.stable_sample_order(current_pool).copy()
    selected_parts = []
    selected_df = remaining.iloc[0:0].copy()
    for _ in range(min(sample_size, len(remaining))):
        ds = choose_dataset_by_deficit(remaining, labeled_df, selected_df)
        candidates = remaining[remaining["dataset_type"].astype(str).eq(ds)].copy()
        if candidates.empty:
            candidates = remaining.copy()
        scores = score_candidates(
            candidates,
            labeled_df=labeled_df,
            selected_df=selected_df,
            embedding_lookup=embedding_lookup,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            use_consistency=use_consistency,
            use_visual=use_visual,
            use_pseudo_instance=use_pseudo_instance,
        )
        jitter = pd.Series(rng.uniform(0, 1e-9, len(candidates)), index=candidates.index)
        picked_idx = (scores + jitter).idxmax()
        picked = remaining.loc[[picked_idx]]
        selected_parts.append(picked)
        selected_df = pd.concat([selected_df, picked])
        remaining = remaining.drop(index=[picked_idx])
    return pd.concat(selected_parts) if selected_parts else current_pool.iloc[0:0].copy()


def select_samples(
    strategy: str,
    current_pool_full: pd.DataFrame,
    labeled_full: pd.DataFrame,
    sample_size: int,
    seed: int,
    round_idx: int,
    embedding_lookup: dict[tuple[str, str], np.ndarray] | None,
    hybrid_weights: dict,
) -> pd.DataFrame:
    if strategy.startswith("GTFree"):
        current_pool = make_gtfree_view(current_pool_full)
        labeled_df = make_gtfree_view(labeled_full)
        assert_gtfree_view(current_pool)
        assert_gtfree_view(labeled_df)
    else:
        current_pool = current_pool_full.copy()
        labeled_df = labeled_full.copy()

    if strategy == "GTFreeRandom":
        return current_pool.sample(n=min(sample_size, len(current_pool)), random_state=seed + round_idx * 101)
    if strategy == "GTFreeConsistency":
        return v6.sort_select(current_pool, sample_size, "score_consistency_only", ascending=False)
    if strategy == "GTFreeDatasetBalancedConsistency":
        return dataset_balanced_utility_select(
            current_pool,
            labeled_df,
            sample_size,
            seed,
            round_idx,
            embedding_lookup=None,
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
            use_consistency=True,
            use_visual=False,
            use_pseudo_instance=False,
        )
    if strategy == "GTFreeDatasetBalancedVisualDiversity":
        return dataset_balanced_utility_select(
            current_pool,
            labeled_df,
            sample_size,
            seed,
            round_idx,
            embedding_lookup=embedding_lookup,
            alpha=0.0,
            beta=1.0,
            gamma=0.0,
            use_consistency=False,
            use_visual=True,
            use_pseudo_instance=False,
        )
    if strategy == "GTFreeDatasetBalancedConsistencyVisualDiversity":
        return dataset_balanced_utility_select(
            current_pool,
            labeled_df,
            sample_size,
            seed,
            round_idx,
            embedding_lookup=embedding_lookup,
            alpha=hybrid_weights["alpha"],
            beta=hybrid_weights["beta"],
            gamma=0.0,
            use_consistency=True,
            use_visual=True,
            use_pseudo_instance=False,
        )
    if strategy == "GTFreeDatasetBalancedConsistencyVisualDiversityPseudoInstance":
        return dataset_balanced_utility_select(
            current_pool,
            labeled_df,
            sample_size,
            seed,
            round_idx,
            embedding_lookup=embedding_lookup,
            alpha=hybrid_weights["alpha"],
            beta=hybrid_weights["beta"],
            gamma=hybrid_weights["gamma"],
            use_consistency=True,
            use_visual=True,
            use_pseudo_instance=True,
        )
    if strategy == "OracleClassDatasetBalancedRandom":
        # Diagnostic only.  This intentionally uses class_hint after explicitly
        # opting into an Oracle strategy.
        return v6.dataset_then_class_balanced_select(
            current_pool,
            sample_size,
            score_col=None,
            ascending=False,
            seed=seed,
            round_idx=round_idx,
            labeled_df=labeled_df,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def add_selection_metadata(selected: pd.DataFrame, seed: int, strategy: str, round_idx: int, selection_type: str) -> pd.DataFrame:
    out = selected.copy()
    out.insert(0, "seed", seed)
    out.insert(1, "strategy", strategy)
    out.insert(2, "round", round_idx)
    out.insert(3, "selection_type", selection_type)
    out.insert(4, "rank_in_selection", range(1, len(out) + 1))
    return out


def compute_visual_redundancy(selected_df: pd.DataFrame, embedding_lookup: dict[tuple[str, str], np.ndarray] | None) -> pd.DataFrame:
    if embedding_lookup is None or selected_df.empty:
        return pd.DataFrame()
    rows = []
    for (seed, strategy, round_idx), sub in selected_df[selected_df["round"] > 0].groupby(["seed", "strategy", "round"]):
        embeddings = [embedding_lookup.get(sample_key(row)) for _, row in sub.iterrows()]
        embeddings = [e for e in embeddings if e is not None]
        if len(embeddings) < 2:
            pair_mean = np.nan
            pair_max = np.nan
        else:
            mat = np.vstack(embeddings)
            sims = mat @ mat.T
            upper = sims[np.triu_indices(len(mat), k=1)]
            pair_mean = float(np.mean(upper))
            pair_max = float(np.max(upper))
        rows.append(
            {
                "seed": seed,
                "strategy": strategy,
                "round": round_idx,
                "selected_batch_pairwise_cosine_similarity_mean": pair_mean,
                "selected_batch_pairwise_cosine_similarity_max": pair_max,
                "num_embedded_selected": len(embeddings),
            }
        )
    return pd.DataFrame(rows)


def compute_selected_distance_diagnostics(
    selected_df: pd.DataFrame,
    embedding_lookup: dict[tuple[str, str], np.ndarray] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if embedding_lookup is None or selected_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for (seed, strategy), strategy_df in selected_df.groupby(["seed", "strategy"], dropna=False):
        labeled_before = strategy_df.iloc[0:0].copy()
        for round_idx in sorted(pd.to_numeric(strategy_df["round"], errors="coerce").dropna().astype(int).unique()):
            round_df = strategy_df[pd.to_numeric(strategy_df["round"], errors="coerce").eq(round_idx)].copy()
            if round_idx == 0:
                labeled_before = pd.concat([labeled_before, round_df], ignore_index=True)
                continue
            for _, row in round_df.iterrows():
                dist = cosine_min_distance(row, labeled_before, embedding_lookup)
                rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "round": round_idx,
                        "dataset_type": row.get("dataset_type"),
                        "image_name": row.get("image_name"),
                        "rank_in_selection": row.get("rank_in_selection"),
                        "min_cosine_distance_to_labeled_before_selection": dist,
                        "num_reference_labeled_before_selection": len(labeled_before),
                    }
                )
            labeled_before = pd.concat([labeled_before, round_df], ignore_index=True)
    sample_df = pd.DataFrame(rows)
    if sample_df.empty:
        return sample_df, pd.DataFrame()
    summary_df = (
        sample_df.groupby(["seed", "strategy", "round"], dropna=False)[
            "min_cosine_distance_to_labeled_before_selection"
        ]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "num_selected_with_embedding",
                "mean": "min_distance_mean",
                "std": "min_distance_std",
                "min": "min_distance_min",
                "max": "min_distance_max",
            }
        )
    )
    return sample_df, summary_df


def compute_consistency_score_distribution(selected_df: pd.DataFrame) -> pd.DataFrame:
    acquired = selected_df[pd.to_numeric(selected_df["round"], errors="coerce").fillna(0).gt(0)].copy()
    if acquired.empty:
        return pd.DataFrame()
    candidate_cols = [
        "score_consistency_only",
        "pseudo_box_count",
        "caption_disagreement_score",
        "explanation_disagreement_score",
        "score",
    ]
    metric_cols = [c for c in candidate_cols if c in acquired.columns]
    rows = []
    for (seed, strategy, round_idx), sub in acquired.groupby(["seed", "strategy", "round"], dropna=False):
        for col in metric_cols:
            values = pd.to_numeric(sub[col], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    "round": round_idx,
                    "metric": col,
                    "count": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) >= 2 else np.nan,
                    "min": float(values.min()),
                    "q25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q75": float(values.quantile(0.75)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def selection_overlap_matrix(selected_df: pd.DataFrame) -> pd.DataFrame:
    final_sets = {}
    max_round = int(selected_df["round"].max()) if len(selected_df) else 0
    for strategy, sub in selected_df[selected_df["round"] <= max_round].groupby("strategy"):
        final_sets[strategy] = set(zip(sub["dataset_type"].astype(str), sub["image_name"].astype(str)))
    rows = []
    for a, set_a in final_sets.items():
        for b, set_b in final_sets.items():
            rows.append(
                {
                    "strategy_a": a,
                    "strategy_b": b,
                    "overlap_count": len(set_a & set_b),
                    "strategy_a_size": len(set_a),
                    "strategy_b_size": len(set_b),
                    "jaccard": len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else np.nan,
                }
            )
    return pd.DataFrame(rows)


def actual_stats_by_round(selected_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_index = build_image_index()
    xml_index = build_xml_index()
    rows = []
    for _, row in selected_df.iterrows():
        rows.extend(parse_image_instances(row, image_index, xml_index))
    instance_df = pd.DataFrame(rows)
    stats_rows = []
    class_rows = []
    for (seed, strategy, round_idx), sub in selected_df.groupby(["seed", "strategy", "round"]):
        cumulative = selected_df[
            (selected_df["seed"] == seed)
            & (selected_df["strategy"] == strategy)
            & (selected_df["round"] <= round_idx)
        ].copy()
        cumulative["source_strategy"] = strategy
        cumulative_instances = instance_df[
            (instance_df["seed"] == seed)
            & (instance_df["strategy"] == strategy)
            & (pd.to_numeric(instance_df["round"], errors="coerce") <= round_idx)
        ].copy()
        stats = strategy_stats(cumulative, cumulative_instances)
        if len(stats):
            row = stats.iloc[0].to_dict()
            row.update({"seed": seed, "strategy": strategy, "round": round_idx})
            stats_rows.append(row)
        if len(cumulative_instances):
            dist = cumulative_instances.groupby("actual_xml_class").size().reset_index(name="bbox_instance_count")
            dist.insert(0, "round", round_idx)
            dist.insert(0, "strategy", strategy)
            dist.insert(0, "seed", seed)
            class_rows.append(dist)
    return pd.DataFrame(stats_rows), pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame()


def write_summary(
    save_dir: Path,
    config: dict,
    selected_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
    distance_summary_df: pd.DataFrame,
    consistency_distribution_df: pd.DataFrame,
) -> None:
    acquired = selected_df[selected_df["round"] > 0]
    lines = [
        "# V7 Visual/Instance Screening Dry-run",
        "",
        "No YOLO training and no final-test evaluation were run by this script.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Acquisition dataset distribution",
        "",
        acquired.groupby(["strategy", "dataset_type"]).size().reset_index(name="count").to_markdown(index=False) if len(acquired) else "_No acquired samples._",
        "",
        "## Visual redundancy",
        "",
        redundancy_df.to_markdown(index=False) if len(redundancy_df) else "_No visual redundancy rows._",
        "",
        "## Selected-sample distance to already labeled set",
        "",
        distance_summary_df.to_markdown(index=False) if len(distance_summary_df) else "_No distance rows._",
        "",
        "## Selected consistency / proxy score distribution",
        "",
        consistency_distribution_df.to_markdown(index=False) if len(consistency_distribution_df) else "_No score distribution rows._",
    ]
    (save_dir / "development_screening_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"v7_visual_instance_screening_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    _, df_all = load_priority_scores()
    full_pool = v6.stable_sample_order(df_all).reset_index(drop=True)

    seeds = parse_int_list_env("AL_SEEDS", [42])
    initial_size = int(os.environ.get("AL_INITIAL_SEED_SIZE", "15"))
    rounds = int(os.environ.get("AL_ROUNDS", "4"))
    query_size = int(os.environ.get("AL_QUERY_SIZE", "5"))
    hybrid_id = os.environ.get("AL_V7_HYBRID_CANDIDATE", "A").strip().upper()
    hybrid_weights = HYBRID_CANDIDATES.get(hybrid_id, HYBRID_CANDIDATES["A"])
    strategies = parse_csv_env(
        "AL_STRATEGIES",
        DEFAULT_STAGE_A_STRATEGIES,
    )

    manifest, embeddings, embedding_config = load_embeddings()
    visual_requested = any("VisualDiversity" in s for s in strategies)
    if visual_requested and (manifest is None or embeddings is None):
        raise FileNotFoundError(
            "Visual strategies require an embedding cache. Run build_visual_embedding_cache_v7.py first."
        )
    embedding_lookup = build_embedding_lookup(manifest, embeddings) if manifest is not None and embeddings is not None else None

    config = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "VERSION": "v7_visual_instance_screening",
        "stage": "Stage A dry-run selection only",
        "final_test_used": False,
        "strategies": strategies,
        "seeds": seeds,
        "initial_seed_size": initial_size,
        "rounds": rounds,
        "query_size": query_size,
        "hybrid_candidate": hybrid_id,
        "hybrid_weights": hybrid_weights,
        "embedding_config": embedding_config,
        "gtfree_prohibited_columns": sorted(PROHIBITED_GTFREE_COLUMNS),
    }
    (save_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_logs = []
    for seed in seeds:
        initial = full_pool.sample(n=min(initial_size, len(full_pool)), random_state=seed + 999)
        for strategy in strategies:
            current_pool = full_pool.drop(index=initial.index).copy()
            labeled = initial.copy()
            selected_logs.append(add_selection_metadata(initial, seed, strategy, 0, "shared_initial_seed_random"))
            for round_idx in range(1, rounds + 1):
                picked_view = select_samples(
                    strategy,
                    current_pool,
                    labeled,
                    query_size,
                    seed,
                    round_idx,
                    embedding_lookup,
                    hybrid_weights,
                )
                picked_keys = set(zip(picked_view["dataset_type"].astype(str), picked_view["image_name"].astype(str)))
                picked_full = current_pool[
                    [key in picked_keys for key in zip(current_pool["dataset_type"].astype(str), current_pool["image_name"].astype(str))]
                ].copy()
                picked_full = picked_full.head(len(picked_view))
                selected_logs.append(add_selection_metadata(picked_full, seed, strategy, round_idx, strategy))
                labeled = pd.concat([labeled, picked_full])
                current_pool = current_pool.drop(index=picked_full.index)

    selected_df = pd.concat(selected_logs, ignore_index=True) if selected_logs else pd.DataFrame()
    redundancy_df = compute_visual_redundancy(selected_df, embedding_lookup)
    distance_sample_df, distance_summary_df = compute_selected_distance_diagnostics(selected_df, embedding_lookup)
    consistency_distribution_df = compute_consistency_score_distribution(selected_df)
    overlap_df = selection_overlap_matrix(selected_df)
    actual_stats_df, actual_class_df = actual_stats_by_round(selected_df)

    selected_df.to_csv(save_dir / "all_selected_samples_by_round.csv", index=False, encoding="utf-8-sig")
    redundancy_df.to_csv(save_dir / "visual_redundancy_by_round.csv", index=False, encoding="utf-8-sig")
    distance_sample_df.to_csv(save_dir / "selected_sample_distance_to_labeled.csv", index=False, encoding="utf-8-sig")
    distance_summary_df.to_csv(save_dir / "selected_sample_distance_summary.csv", index=False, encoding="utf-8-sig")
    consistency_distribution_df.to_csv(save_dir / "consistency_score_distribution.csv", index=False, encoding="utf-8-sig")
    overlap_df.to_csv(save_dir / "selection_overlap_matrix.csv", index=False, encoding="utf-8-sig")
    actual_stats_df.to_csv(save_dir / "actual_instance_statistics_by_round.csv", index=False, encoding="utf-8-sig")
    actual_class_df.to_csv(save_dir / "actual_class_distribution_by_round.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(save_dir / "all_round_results.csv", index=False, encoding="utf-8-sig")
    write_summary(save_dir, config, selected_df, redundancy_df, distance_summary_df, consistency_distribution_df)

    append_registry_row(
        save_dir / "experiment_registry.csv",
        project_root=PROJECT_ROOT,
        experiment_id=save_dir.name,
        stage="stage_a_dry_run_selection",
        eval_split="development_eval_v7",
        status="success",
        result_path=save_dir,
        hyperparameters=config,
    )

    print("=" * 100)
    print("[DONE] V7 visual/instance Stage A dry-run")
    print(f"Output dir: {save_dir}")
    print("No YOLO training. No final-test evaluation.")
    print("=" * 100)


if __name__ == "__main__":
    main()

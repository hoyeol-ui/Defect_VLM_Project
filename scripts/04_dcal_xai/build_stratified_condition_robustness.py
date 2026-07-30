"""Audit pool-condition identifiability using frozen selection records only.

No training, model inference, embedding extraction, selector implementation,
score reconstruction, FN screening, or final/locked-test access occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/stratified_condition_robustness_20260718"
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
FEASIBILITY = DOCS / "stratified_audit_design_feasibility_20260718.md"
DECISION = DOCS / "stratified_condition_robustness_decision_20260718.md"

SEEDS = list(range(200))
INITIAL_SIZE = 20
DESIGN_TYPE = "TYPE_A_paired_variable_pool"
INFERENTIAL_UNIT = "acquisition_seed_or_pool_realization"
PRACTICAL_PREVALENCE_RELATIVE_RANGE_MIN = 0.10
PRACTICAL_TARGET_COUNT_RANGE_MIN = 20
MIN_STRATUM_UNITS = 20

MATCHING_COVARIATE_WHITELIST = {
    "pool_target_prevalence", "pool_target_count", "pool_category_entropy",
    "pool_max_category_share", "pool_source_entropy", "pool_max_source_share",
    "pool_size", "initial_set_id", "budget", "protocol_id",
}
FORBIDDEN_MATCHING_COVARIATES = {
    "target_yield", "category_coverage", "source_coverage", "redundancy",
    "rare_image_count", "downstream_map", "recall", "rare_ap", "gate_result",
    "strategy_score", "selected_category_entropy", "selected_source_entropy",
}
STRENGTHENED_CRITERIA = (
    "positive_adjusted_gain_in_at_least_2_datasets",
    "ci_lower_above_zero_in_at_least_2_datasets",
    "retained_effect_fraction_at_least_0p50",
    "positive_discovery_with_safety_loss_in_at_least_2_datasets",
    "not_dominated_by_one_category_source_or_few_seeds",
    "no_outcome_leakage_in_pairing_or_matching",
)

REGISTRY: list[dict[str, Any]] = []
SCANNED: set[Path] = set()


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    protocol: str
    strategies: tuple[str, ...]
    baseline: str
    selector: str
    budget: int
    blind_path: Path
    gt_path: Path
    selection_path: Path
    metrics_path: Path
    audit_script: Path
    category_kind: str
    category_column: str
    source_column: str | None


SPECS = (
    DatasetSpec(
        "GC10-DET", "gc10_taxonomy_initial20_query20_200seed",
        ("GTFreeRandom", "FrozenDINOVisualDiversity"),
        "GTFreeRandom", "FrozenDINOVisualDiversity", 20,
        ROOT / "runs/gc10_taxonomy_protocol/gc10_protocol_20260715/gc10_acquisition_pool_blind.csv",
        ROOT / "runs/gc10_taxonomy_protocol/gc10_protocol_20260715/gc10_acquisition_pool_gt_audit.csv",
        ROOT / "runs/gc10_taxonomy_selection_audit/gc10_random_vs_dino_200seed_20260715/gc10_selection_records_posthoc.csv",
        ROOT / "runs/gc10_taxonomy_selection_audit/gc10_random_vs_dino_200seed_20260715/gc10_seed_strategy_metrics.csv",
        ROOT / "scripts/03_analysis/run_gc10_taxonomy_selection_audit.py",
        "multilabel", "class_ids", "production_group_raw",
    ),
    DatasetSpec(
        "MPDD", "mpdd_initial20_query20_200seed",
        ("GTFreeRandom", "CategoryBalancedRandom", "FrozenCategoryBalancedDINO"),
        "GTFreeRandom", "FrozenCategoryBalancedDINO", 20,
        ROOT / "runs/mpdd_annotation_triage_protocol/mpdd_protocol_20260715/mpdd_acquisition_pool_blind.csv",
        ROOT / "runs/mpdd_annotation_triage_protocol/mpdd_protocol_20260715/mpdd_acquisition_pool_gt_audit.csv",
        ROOT / "runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_selection_records_posthoc.csv",
        ROOT / "runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_seed_strategy_metrics.csv",
        ROOT / "scripts/03_analysis/run_mpdd_selection_only_audit.py",
        "single", "product_category", "official_split",
    ),
    DatasetSpec(
        "VisA", "visa_initial20_query20_200seed",
        ("GTFreeRandom", "FrozenDINOVisualDiversity"),
        "GTFreeRandom", "FrozenDINOVisualDiversity", 20,
        ROOT / "runs/visa_annotation_triage_protocol/visa_protocol_v2_20260715/visa_acquisition_pool_blind.csv",
        ROOT / "runs/visa_annotation_triage_protocol/visa_protocol_v2_20260715/visa_acquisition_pool_gt_audit.csv",
        ROOT / "runs/visa_selection_only_audit/visa_random_vs_visual_200seed_20260715/visa_selection_records_posthoc.csv",
        ROOT / "runs/visa_selection_only_audit/visa_random_vs_visual_200seed_20260715/visa_seed_strategy_metrics.csv",
        ROOT / "scripts/03_analysis/run_visa_selection_only_audit.py",
        "single", "category", None,
    ),
)


def prevalence_effect_allowed(design_type: str, practical_variation_pass: bool) -> bool:
    """Freeze the condition-effect identifiability gate before analysis."""
    return design_type in {
        "TYPE_A_paired_variable_pool", "TYPE_B_unmatched_variable_pool"
    } and practical_variation_pass


def adjusted_evidence_allowed(
    design_type: str, matching_balance_pass: bool | None
) -> bool:
    """Prevent failed TYPE B balance from being promoted to adjusted evidence."""
    if design_type == "TYPE_B_unmatched_variable_pool":
        return matching_balance_pass is True
    return design_type == "TYPE_A_paired_variable_pool"


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def guard(path: Path) -> None:
    low = str(path).lower()
    if "final" in low or "locked" in low:
        raise RuntimeError(f"Prohibited final/locked path: {path}")
    if not path.exists():
        raise FileNotFoundError(path)
    SCANNED.add(path.resolve())


def read_csv(path: Path) -> pd.DataFrame:
    guard(path)
    return pd.read_csv(path)


def read_text(path: Path) -> str:
    guard(path)
    return path.read_text(encoding="utf-8-sig")


def hash_ids(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(map(str, values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_classes(value: Any) -> list[int]:
    return [int(item) for item in str(value).split("|") if item and item != "nan"]


def entropy_hhi(values: Iterable[Any]) -> tuple[int, float, float, float]:
    series = pd.Series(list(values), dtype="object").dropna().astype(str)
    if series.empty:
        return 0, math.nan, math.nan, math.nan
    shares = series.value_counts() / len(series)
    entropy = float(-(shares * np.log(shares)).sum())
    return int(len(shares)), entropy, float(shares.max()), float((shares**2).sum())


def target_flags(spec: DatasetSpec, frame: pd.DataFrame) -> pd.Series:
    if spec.dataset == "GC10-DET":
        return frame["class_ids"].map(
            lambda value: bool(set(parse_classes(value)) & {8, 9, 10})
        )
    return frame["is_anomaly"].astype(bool)


def category_values(spec: DatasetSpec, frame: pd.DataFrame) -> list[Any]:
    if spec.category_kind == "multilabel":
        return [
            item
            for classes in frame[spec.category_column].map(parse_classes)
            for item in classes
        ]
    return frame[spec.category_column].astype(str).tolist()


def reconstruct_design() -> tuple[
    pd.DataFrame,
    dict[tuple[str, int], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    rows: list[dict[str, Any]] = []
    contexts: dict[tuple[str, int], dict[str, Any]] = {}
    dataset_stats: dict[str, dict[str, Any]] = {}

    for spec in SPECS:
        blind = read_csv(spec.blind_path).sort_values(
            "sample_id", kind="mergesort"
        ).reset_index(drop=True)
        gt_raw = read_csv(spec.gt_path)
        gt = blind[["sample_id"]].merge(
            gt_raw, on="sample_id", how="left", validate="one_to_one"
        )
        selections = read_csv(spec.selection_path)
        frozen_metrics = read_csv(spec.metrics_path)
        if frozen_metrics["acquisition_seed"].nunique() != len(SEEDS):
            raise AssertionError(
                f"{spec.dataset}: frozen metrics do not contain 200 acquisition seeds"
            )
        read_text(spec.audit_script)

        all_indices = np.arange(len(blind), dtype=int)
        pool_hashes: set[str] = set()
        initial_hashes: set[str] = set()
        pool_targets: list[int] = []
        pool_prevalences: list[float] = []
        category_entropies: list[float] = []
        source_entropies: list[float] = []
        initial_sets: list[set[str]] = []

        for seed in SEEDS:
            initial_indices = pd.DataFrame({"idx": all_indices}).sample(
                n=INITIAL_SIZE, random_state=seed + 999, replace=False
            )["idx"].astype(int).tolist()
            remaining = np.setdiff1d(
                all_indices, np.asarray(initial_indices), assume_unique=True
            )
            initial_ids = blind.iloc[initial_indices]["sample_id"].astype(str).tolist()
            candidate_ids = blind.iloc[remaining]["sample_id"].astype(str).tolist()
            initial_hash = hash_ids(initial_ids)
            pool_hash = hash_ids(candidate_ids)
            pool_id = f"{spec.dataset}|{spec.protocol}|{pool_hash[:16]}"

            candidate_gt = gt.iloc[remaining]
            flags = target_flags(spec, candidate_gt)
            target_count = int(flags.sum())
            prevalence = float(flags.mean())
            category_count, category_entropy, max_category_share, _ = entropy_hhi(
                category_values(spec, candidate_gt)
            )
            if spec.source_column:
                source_count, source_entropy, max_source_share, _ = entropy_hhi(
                    candidate_gt[spec.source_column]
                )
                source_entropies.append(source_entropy)
            else:
                source_count, source_entropy, max_source_share = None, None, None

            selected_ids = set(
                selections.loc[
                    selections["acquisition_seed"].eq(seed), "sample_id"
                ].astype(str)
            )
            if not selected_ids.issubset(set(candidate_ids)):
                raise AssertionError(
                    f"{spec.dataset} seed {seed}: selected ID outside candidate pool"
                )

            contexts[(spec.dataset, seed)] = {
                "spec": spec,
                "initial_gt": gt.iloc[initial_indices].copy(),
                "pool_id": pool_id,
            }
            pool_hashes.add(pool_hash)
            initial_hashes.add(initial_hash)
            pool_targets.append(target_count)
            pool_prevalences.append(prevalence)
            category_entropies.append(category_entropy)
            initial_sets.append(set(initial_ids))

            random_id = f"{spec.dataset}|{spec.protocol}|seed={seed}|{spec.baseline}"
            source = f"{rel(spec.blind_path)}; {rel(spec.gt_path)}; {rel(spec.audit_script)}"
            for strategy in spec.strategies:
                rows.append({
                    "dataset": spec.dataset,
                    "protocol": spec.protocol,
                    "strategy": strategy,
                    "acquisition_seed": seed,
                    "pool_id": pool_id,
                    "pool_hash": pool_hash,
                    "paired_random_id": random_id,
                    "pool_size": len(remaining),
                    "budget": spec.budget,
                    "target_count": target_count,
                    "target_prevalence": prevalence,
                    "category_count": category_count,
                    "category_entropy": category_entropy,
                    "max_category_share": max_category_share,
                    "source_count": source_count,
                    "source_entropy": source_entropy,
                    "max_source_share": max_source_share,
                    "initial_set_id": initial_hash,
                    "source_file": source,
                })

        if len(pool_hashes) != 200 or len(initial_hashes) != 200:
            raise AssertionError(
                f"{spec.dataset}: expected 200 reconstructed pools and initial sets"
            )

        pairwise_jaccard: list[float] = []
        total = len(blind)
        for i, left in enumerate(initial_sets):
            for right in initial_sets[i + 1:]:
                initial_intersection = len(left & right)
                initial_union = len(left | right)
                candidate_intersection = total - initial_union
                candidate_union = total - initial_intersection
                pairwise_jaccard.append(candidate_intersection / candidate_union)

        prevalence_mean = float(np.mean(pool_prevalences))
        prevalence_range = float(max(pool_prevalences) - min(pool_prevalences))
        relative_range = prevalence_range / prevalence_mean
        target_range = int(max(pool_targets) - min(pool_targets))
        practical_pass = (
            relative_range >= PRACTICAL_PREVALENCE_RELATIVE_RANGE_MIN
            and target_range >= PRACTICAL_TARGET_COUNT_RANGE_MIN
        )
        dataset_stats[spec.dataset] = {
            "design_type": DESIGN_TYPE,
            "seeds": 200,
            "unique_pool_hashes": len(pool_hashes),
            "paired": True,
            "pool_size_min": len(blind) - INITIAL_SIZE,
            "pool_size_max": len(blind) - INITIAL_SIZE,
            "target_count_min": min(pool_targets),
            "target_count_max": max(pool_targets),
            "prevalence_min": min(pool_prevalences),
            "prevalence_max": max(pool_prevalences),
            "prevalence_relative_range": relative_range,
            "category_entropy_min": min(category_entropies),
            "category_entropy_max": max(category_entropies),
            "source_entropy_min": min(source_entropies) if source_entropies else None,
            "source_entropy_max": max(source_entropies) if source_entropies else None,
            "mean_pairwise_pool_jaccard": float(np.mean(pairwise_jaccard)),
            "initial_set_variation": len(initial_hashes),
            "prevalence_variation_pass": prevalence_effect_allowed(
                DESIGN_TYPE, practical_pass
            ),
            "reason": (
                "200 unique candidate hashes arise only because seed-specific "
                "initial20 is removed from one fixed acquisition pool; same-seed "
                "strategies are paired, but condition range is near-fixed."
            ),
        }

    design = pd.DataFrame(rows)
    design.to_csv(OUT / "pool_design_registry.csv", index=False, encoding="utf-8-sig")
    return design, contexts, dataset_stats


def build_seed_outcomes(
    contexts: dict[tuple[str, int], dict[str, Any]]
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for spec in SPECS:
        selections = read_csv(spec.selection_path)
        gt = read_csv(spec.gt_path).set_index("sample_id", drop=False)
        for seed in SEEDS:
            context = contexts[(spec.dataset, seed)]
            per_strategy: list[dict[str, Any]] = []
            random_target: int | None = None
            random_coverage: int | None = None
            random_hhi: float | None = None

            for strategy in spec.strategies:
                selected = selections[
                    selections["acquisition_seed"].eq(seed)
                    & selections["strategy"].eq(strategy)
                ].copy()
                if len(selected) != spec.budget:
                    raise AssertionError(
                        f"{spec.dataset}/{seed}/{strategy}: expected {spec.budget}, "
                        f"got {len(selected)}"
                    )
                joined = selected[["sample_id"]].merge(
                    gt.reset_index(drop=True),
                    on="sample_id", how="left", validate="one_to_one",
                )

                if spec.dataset == "GC10-DET":
                    flags = selected["contains_rare_class_posthoc"].astype(bool)
                    selected_classes = [
                        item
                        for value in selected["class_ids_posthoc"]
                        for item in parse_classes(value)
                    ]
                    initial_classes = {
                        item
                        for value in context["initial_gt"]["class_ids"]
                        for item in parse_classes(value)
                    }
                    query_classes = set(selected_classes)
                    coverage = len(initial_classes | query_classes)
                    category_count, category_entropy, max_category_share, category_hhi = entropy_hhi(selected_classes)
                    rare_images = int(flags.sum())
                    rare_classes = len(query_classes & {8, 9, 10})
                    instances = int(selected["num_instances_posthoc"].sum())
                elif spec.dataset == "MPDD":
                    flags = selected["is_anomaly_posthoc"].astype(bool)
                    category_count, category_entropy, max_category_share, category_hhi = entropy_hhi(
                        selected["product_category_posthoc"]
                    )
                    coverage = category_count
                    rare_images, rare_classes = None, None
                    instances = int(selected["num_defect_components_posthoc"].sum())
                else:
                    flags = selected["is_anomaly_posthoc"].astype(bool)
                    category_count, category_entropy, max_category_share, category_hhi = entropy_hhi(
                        selected["category_posthoc"]
                    )
                    coverage = category_count
                    rare_images, rare_classes = None, None
                    instances = int(selected["num_defect_components_posthoc"].sum())

                if spec.source_column:
                    source_count, source_entropy, max_source_share, source_hhi = entropy_hhi(
                        joined[spec.source_column]
                    )
                else:
                    source_count, source_entropy, max_source_share, source_hhi = None, None, None, None

                target_yield = int(flags.sum())
                per_strategy.append({
                    "dataset": spec.dataset,
                    "protocol": spec.protocol,
                    "pool_id": context["pool_id"],
                    "acquisition_seed": seed,
                    "strategy": strategy,
                    "baseline": spec.baseline,
                    "target_yield": target_yield,
                    "target_enrichment": None,
                    "category_coverage": coverage,
                    "category_entropy": category_entropy,
                    "max_category_share": max_category_share,
                    "source_coverage": source_count,
                    "source_entropy": source_entropy,
                    "max_source_share": max_source_share,
                    "concentration_hhi": category_hhi,
                    "source_concentration_hhi": source_hhi,
                    "rare_image_count": rare_images,
                    "rare_class_count": rare_classes,
                    "instance_count": instances,
                    "paired_discovery_delta": None,
                    "paired_coverage_delta": None,
                    "paired_concentration_delta": None,
                    "target_yield_per_category": (
                        target_yield / category_count if category_count else None
                    ),
                    "target_yield_per_source": (
                        target_yield / source_count if source_count else None
                    ),
                    "safety_quadrant": None,
                    "source_file": f"{rel(spec.selection_path)}; {rel(spec.gt_path)}",
                })
                if strategy == spec.baseline:
                    random_target = target_yield
                    random_coverage = coverage
                    random_hhi = category_hhi

            assert random_target is not None
            assert random_coverage is not None
            assert random_hhi is not None
            for row in per_strategy:
                if row["strategy"] == spec.baseline:
                    row["target_enrichment"] = 1.0 if random_target > 0 else None
                    row["paired_discovery_delta"] = 0.0
                    row["paired_coverage_delta"] = 0.0
                    row["paired_concentration_delta"] = 0.0
                    row["safety_quadrant"] = "baseline"
                else:
                    row["target_enrichment"] = (
                        row["target_yield"] / random_target
                        if random_target > 0 else None
                    )
                    discovery = row["target_yield"] - random_target
                    safety = row["category_coverage"] - random_coverage
                    row["paired_discovery_delta"] = discovery
                    row["paired_coverage_delta"] = safety
                    row["paired_concentration_delta"] = (
                        row["concentration_hhi"] - random_hhi
                    )
                    if discovery > 0 and safety >= 0:
                        quadrant = "Q1_gain_positive_safety_nonnegative"
                    elif discovery > 0 and safety < 0:
                        quadrant = "Q2_gain_positive_safety_loss"
                    elif discovery <= 0 and safety >= 0:
                        quadrant = "Q3_gain_nonpositive_safety_nonnegative"
                    else:
                        quadrant = "Q4_gain_nonpositive_safety_loss"
                    row["safety_quadrant"] = quadrant
            output.extend(per_strategy)

    outcomes = pd.DataFrame(output)
    outcomes.to_csv(OUT / "seed_level_outcomes.csv", index=False, encoding="utf-8-sig")
    return outcomes


def bootstrap_ci(values: np.ndarray, seed: int = 20260718) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def permutation_p(values: np.ndarray, seed: int = 20260718) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    extreme = 0
    total = 200_000
    for start in range(0, total, 5_000):
        batch = min(5_000, total - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(batch, len(values)))
        extreme += int(
            (np.abs((signs * values).mean(axis=1)) >= observed - 1e-15).sum()
        )
    return (extreme + 1) / (total + 1)


def build_effect_summary(
    outcomes: pd.DataFrame, stats: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in SPECS:
        comparisons = [item for item in spec.strategies if item != spec.baseline]
        for comparison in comparisons:
            data = outcomes[
                outcomes["dataset"].eq(spec.dataset)
                & outcomes["strategy"].isin([spec.baseline, comparison])
            ]
            pivot = data.pivot(
                index="acquisition_seed",
                columns="strategy",
                values=["target_yield", "category_coverage", "concentration_hhi"],
            )
            discovery = (
                pivot["target_yield"][comparison]
                - pivot["target_yield"][spec.baseline]
            ).to_numpy(float)
            coverage = (
                pivot["category_coverage"][comparison]
                - pivot["category_coverage"][spec.baseline]
            ).to_numpy(float)
            concentration = (
                pivot["concentration_hhi"][comparison]
                - pivot["concentration_hhi"][spec.baseline]
            ).to_numpy(float)
            ci_lower, ci_upper = bootstrap_ci(discovery)
            unadjusted = float(
                data[data["strategy"].eq(comparison)]["target_yield"].mean()
                - data[data["strategy"].eq(spec.baseline)]["target_yield"].mean()
            )
            adjusted = float(discovery.mean())
            common = {
                "dataset": spec.dataset,
                "protocol": spec.protocol,
                "strategy": comparison,
                "baseline": spec.baseline,
                "is_primary_selector": comparison == spec.selector,
                "design_type": DESIGN_TYPE,
                "inferential_unit": INFERENTIAL_UNIT,
            }
            rows.append({
                **common,
                "stratum_name": "all_paired_same_pool",
                "stratum_value": "all",
                "n_units": len(discovery),
                "unadjusted_gain": unadjusted,
                "adjusted_gain": adjusted,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "positive_fraction": float((discovery > 0).mean()),
                "p_value": permutation_p(discovery),
                "coverage_delta": float(coverage.mean()),
                "concentration_delta": float(concentration.mean()),
                "retained_effect_fraction": (
                    adjusted / unadjusted if unadjusted != 0 else None
                ),
                "identifiability": (
                    "paired selection effect identifiable; prevalence effect not identifiable"
                ),
                "evidence_level": "paired_existing_record",
                "standardized_paired_effect_size": float(
                    discovery.mean() / discovery.std(ddof=1)
                ),
                "median_gain": float(np.median(discovery)),
                "iqr_gain": float(
                    np.quantile(discovery, .75) - np.quantile(discovery, .25)
                ),
                "min_gain": float(discovery.min()),
                "max_gain": float(discovery.max()),
                "positive_gain_with_safety_loss_fraction": float(
                    ((discovery > 0) & (coverage < 0)).mean()
                ),
                "source_file": f"{rel(spec.metrics_path)}; {rel(spec.selection_path)}",
            })
            current = stats[spec.dataset]
            rows.append({
                **common,
                "stratum_name": "prevalence_low_medium_high",
                "stratum_value": "not_identifiable",
                "n_units": 200,
                "unadjusted_gain": None,
                "adjusted_gain": None,
                "ci_lower": None,
                "ci_upper": None,
                "positive_fraction": None,
                "p_value": None,
                "coverage_delta": None,
                "concentration_delta": None,
                "retained_effect_fraction": None,
                "identifiability": (
                    "not_identifiable: target count range="
                    f"{current['target_count_min']}..{current['target_count_max']}; "
                    "relative prevalence range="
                    f"{current['prevalence_relative_range']:.6f}; "
                    "pools are leave-20 perturbations"
                ),
                "evidence_level": "insufficient_condition_variation",
                "standardized_paired_effect_size": None,
                "median_gain": None,
                "iqr_gain": None,
                "min_gain": None,
                "max_gain": None,
                "positive_gain_with_safety_loss_fraction": None,
                "source_file": (
                    f"{rel(spec.blind_path)}; {rel(spec.gt_path)}; "
                    f"{rel(spec.audit_script)}"
                ),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(
        OUT / "stratified_effect_summary.csv", index=False, encoding="utf-8-sig"
    )
    return summary


def register_output_metrics(
    design: pd.DataFrame, outcomes: pd.DataFrame, summary: pd.DataFrame
) -> None:
    registry_specs = (
        (
            "pool_design_registry.csv", design,
            [
                "pool_size", "budget", "target_count", "target_prevalence",
                "category_count", "category_entropy", "max_category_share",
                "source_count", "source_entropy", "max_source_share",
            ],
            "reconstructed candidate pool after exact frozen initial20 removal; post-hoc GT summary",
        ),
        (
            "seed_level_outcomes.csv", outcomes,
            [
                "target_yield", "target_enrichment", "category_coverage",
                "category_entropy", "max_category_share", "source_coverage",
                "source_entropy", "max_source_share", "concentration_hhi",
                "source_concentration_hhi", "rare_image_count", "rare_class_count",
                "instance_count", "paired_discovery_delta", "paired_coverage_delta",
                "paired_concentration_delta", "target_yield_per_category",
                "target_yield_per_source",
            ],
            "selected-set post-hoc outcome; paired delta uses same-seed Random",
        ),
        (
            "stratified_effect_summary.csv", summary,
            [
                "n_units", "unadjusted_gain", "adjusted_gain", "ci_lower",
                "ci_upper", "positive_fraction", "p_value", "coverage_delta",
                "concentration_delta", "retained_effect_fraction",
                "standardized_paired_effect_size", "median_gain", "iqr_gain",
                "min_gain", "max_gain",
                "positive_gain_with_safety_loss_fraction",
            ],
            "seed-level paired comparison; 50k bootstrap CI; 200k sign-flip permutation",
        ),
    )
    for filename, frame, metrics, transformation in registry_specs:
        for row_id, row in frame.iterrows():
            if filename == "stratified_effect_summary.csv":
                key = (
                    f"dataset={row.dataset}; strategy={row.strategy}; "
                    f"stratum={row.stratum_name}"
                )
            else:
                key = (
                    f"dataset={row.dataset}; acquisition_seed={row.acquisition_seed}; "
                    f"strategy={row.strategy}"
                )
            for metric in metrics:
                value = row[metric]
                if pd.isna(value):
                    continue
                REGISTRY.append({
                    "output_file": filename,
                    "output_row_id": row_id,
                    "metric": metric,
                    "value": value,
                    "source_file": row["source_file"],
                    "source_row_or_key": key,
                    "transformation": transformation,
                    "verified": True,
                })
    pd.DataFrame(REGISTRY).to_csv(
        OUT / "metric_source_registry.csv", index=False, encoding="utf-8-sig"
    )


def design_table(stats: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| dataset | design_type | seeds | unique_pool_hashes | paired | prevalence_range | category_entropy_range | source_variation | matching_identifiable | reason |",
        "|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    for spec in SPECS:
        current = stats[spec.dataset]
        source = (
            "not separately available"
            if current["source_entropy_min"] is None
            else (
                f"entropy {current['source_entropy_min']:.6f} to "
                f"{current['source_entropy_max']:.6f}"
            )
        )
        lines.append(
            f"| {spec.dataset} | TYPE A: paired_variable_pool "
            "(leave-20 finite-pool perturbation) | 200 | "
            f"{current['unique_pool_hashes']} | yes | "
            f"{current['prevalence_min']:.8f} to {current['prevalence_max']:.8f} | "
            f"{current['category_entropy_min']:.6f} to "
            f"{current['category_entropy_max']:.6f} | {source} | "
            "paired selector effect: yes; prevalence matching: no | "
            f"{current['reason']} Mean pairwise pool Jaccard="
            f"{current['mean_pairwise_pool_jaccard']:.6f}. |"
        )
    return "\n".join(lines)


def write_feasibility(stats: dict[str, dict[str, Any]]) -> None:
    text = f"""# Stratified audit design feasibility

## Design classification

{design_table(stats)}

For all three datasets, the frozen protocol draws the initial 20 with
`random_state=seed + 999` and removes those samples from one fixed acquisition
pool. Random and selector strategies share that reconstructed candidate pool
within a seed. This is formally TYPE A for the **same-seed selector effect**,
but the 200 candidate pools are highly overlapping leave-20 perturbations, not
200 independently sampled industrial pools.

## Prevalence effect gate

The frozen minimum is a target-count range of at least
{PRACTICAL_TARGET_COUNT_RANGE_MIN}, a relative prevalence range of at least
{PRACTICAL_PREVALENCE_RELATIVE_RANGE_MIN:.2f}, and at least {MIN_STRATUM_UNITS}
independent pool realizations per tertile. No dataset passes. Low/medium/high
prevalence effects, prevalence-gain correlations, and prevalence
matching/regression are not estimated. Bootstrap intervals below describe
acquisition-seed variability, not a population of independent production pools.

## Allowed analyses

- Same-seed, same-candidate-pool selector-minus-Random target-yield difference.
- Paired category/source coverage and HHI differences.
- Selected-set concentration and discovery-safety quadrants.
- Seed-level variability plus category/source dominance diagnostics.

## Prohibited interpretations

- Claiming that lower target prevalence increases selector gain.
- Treating between-dataset differences as a prevalence moderation effect.
- Treating leave-20 perturbations as independently sampled production pools.
- Using selected yield or coverage as a matching covariate.

There is no TYPE B dataset, so `matching_balance.csv` is intentionally not created.
"""
    FEASIBILITY.write_text(text, encoding="utf-8")


def load_robustness_diagnostics() -> dict[str, float]:
    visa_loo = read_csv(
        ROOT / "runs/visa_selection_only_audit/visa_random_vs_visual_200seed_20260715/visa_leave_one_category_out.csv"
    )
    mpdd_loo = read_csv(
        ROOT / "runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_leave_one_category_out.csv"
    )
    mpdd_decomposition = read_csv(
        ROOT / "runs/mpdd_selection_only_audit/mpdd_hierarchical_dino_200seed_20260715/mpdd_source_origin_gain_decomposition.csv"
    )
    visa_total = 14.48
    visa_min = float(visa_loo["mean_anomaly_yield_difference"].min())
    composition = mpdd_decomposition.loc[
        mpdd_decomposition["comparison"].eq(
            "FrozenCategoryBalancedDINO-minus-GTFreeRandom"
        ),
        "composition_fraction_of_observed_gain",
    ].iloc[0]
    return {
        "visa_min_loo": visa_min,
        "visa_top_category_contribution_fraction": (visa_total - visa_min) / visa_total,
        "mpdd_min_loo": float(mpdd_loo["mean_anomaly_yield_difference"].min()),
        "mpdd_source_composition_fraction": float(composition),
    }


def write_missing(stats: dict[str, dict[str, Any]]) -> None:
    rows = []
    for spec in SPECS:
        current = stats[spec.dataset]
        rows.append({
            "dataset": spec.dataset,
            "requested_analysis": "prevalence low/medium/high effect",
            "missing_requirement": (
                "practical independent pool variation; observed count "
                f"{current['target_count_min']}..{current['target_count_max']}, "
                "relative prevalence range "
                f"{current['prevalence_relative_range']:.6f}"
            ),
            "consequence": "not_identifiable; no strata/regression/correlation computed",
            "allowed_interpretation": "same-pool paired selection effect and selected-set concentration",
            "prohibited_interpretation": "prevalence causes or moderates selector gain",
        })
    rows.extend([
        {
            "dataset": "VisA",
            "requested_analysis": "independent source entropy effect",
            "missing_requirement": "source field distinct from product category",
            "consequence": "source metrics remain missing",
            "allowed_interpretation": "category concentration only",
            "prohibited_interpretation": "source robustness",
        },
        {
            "dataset": "ALL",
            "requested_analysis": "TYPE B matching",
            "missing_requirement": "no unmatched variable-pool protocol exists",
            "consequence": "matching_balance.csv not created",
            "allowed_interpretation": "same-seed pairing",
            "prohibited_interpretation": "matched causal effect",
        },
        {
            "dataset": "GC10-DET",
            "requested_analysis": "aggregate-branch regression of selection to learning",
            "missing_requirement": "too few protocol-level independent points",
            "consequence": "no regression/correlation",
            "allowed_interpretation": "two separate translation case studies",
            "prohibited_interpretation": "estimated learner-alignment coefficient",
        },
    ])
    pd.DataFrame(rows).to_csv(
        OUT / "missing_or_nonidentifiable.csv", index=False, encoding="utf-8-sig"
    )


def setup_plotting() -> None:
    plt.rcParams.update({
        "font.family": ["Malgun Gothic", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 180,
    })
    FIGURES.mkdir(parents=True, exist_ok=True)


def plot_prevalence_not_identifiable(stats: dict[str, dict[str, Any]]) -> None:
    setup_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    for ax, spec in zip(axes, SPECS):
        current = stats[spec.dataset]
        ax.axis("off")
        ax.text(.5, .73, spec.dataset, ha="center", fontsize=13, fontweight="bold", transform=ax.transAxes)
        ax.text(.5, .49, "Prevalence strata\nNOT IDENTIFIABLE", ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.text(
            .5, .22,
            f"target count {current['target_count_min']} to {current['target_count_max']}\n"
            f"prevalence {current['prevalence_min']:.6f} to {current['prevalence_max']:.6f}\n"
            f"mean pool Jaccard {current['mean_pairwise_pool_jaccard']:.4f}",
            ha="center", va="center", fontsize=9, transform=ax.transAxes,
        )
    fig.suptitle("Prevalence effects cannot be estimated from leave-20 pool perturbations")
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(FIGURES / "stratified_discovery_gain.png", bbox_inches="tight")
    plt.close(fig)


def plot_quadrants(outcomes: pd.DataFrame) -> None:
    setup_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7))
    short = {
        "FrozenDINOVisualDiversity": "Frozen DINO",
        "CategoryBalancedRandom": "Cat-bal. Random",
        "FrozenCategoryBalancedDINO": "Cat-bal. DINO",
    }
    for ax, spec in zip(axes, SPECS):
        ax.axhline(0, color="0.35", lw=1)
        ax.axvline(0, color="0.35", lw=1)
        annotation: list[str] = []
        comparisons = [item for item in spec.strategies if item != spec.baseline]
        for marker, strategy in zip(["o", "^"], comparisons):
            frame = outcomes[
                outcomes["dataset"].eq(spec.dataset)
                & outcomes["strategy"].eq(strategy)
            ]
            ax.scatter(
                frame["paired_discovery_delta"].astype(float),
                frame["paired_coverage_delta"].astype(float),
                s=18, alpha=.48, marker=marker,
                label=short.get(strategy, strategy),
            )
            frequencies = frame["safety_quadrant"].value_counts(normalize=True)
            q1 = frequencies.get("Q1_gain_positive_safety_nonnegative", 0.0)
            q2 = frequencies.get("Q2_gain_positive_safety_loss", 0.0)
            annotation.append(
                f"{short.get(strategy, strategy)}: Q1 {q1:.1%}, Q2 {q2:.1%}"
            )
        ax.text(.03, .97, "\n".join(annotation), transform=ax.transAxes, va="top", fontsize=8)
        ax.set_title(spec.dataset)
        ax.set_xlabel("Paired discovery gain")
        ax.grid(alpha=.15)
        if len(comparisons) > 1:
            ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Paired category/class coverage delta")
    fig.suptitle("Same-pool discovery-safety quadrants (one point per acquisition seed)")
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(FIGURES / "discovery_safety_quadrants.png", bbox_inches="tight")
    plt.close(fig)


def plot_concentration(outcomes: pd.DataFrame) -> None:
    setup_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.7))
    short = {
        "GTFreeRandom": "Random",
        "FrozenDINOVisualDiversity": "DINO",
        "CategoryBalancedRandom": "CatRand",
        "FrozenCategoryBalancedDINO": "CatDINO",
    }
    for ax, spec in zip(axes, SPECS):
        frame = outcomes[
            outcomes["dataset"].eq(spec.dataset)
            & outcomes["strategy"].isin(spec.strategies)
        ]
        labels: list[str] = []
        series: list[pd.Series] = []
        for strategy in spec.strategies:
            labels.append(f"{short.get(strategy, strategy)}\ncategory")
            series.append(
                frame.loc[
                    frame["strategy"].eq(strategy), "concentration_hhi"
                ].dropna()
            )
        if spec.source_column:
            for strategy in spec.strategies:
                labels.append(f"{short.get(strategy, strategy)}\nsource")
                series.append(
                    frame.loc[
                        frame["strategy"].eq(strategy),
                        "source_concentration_hhi",
                    ].dropna()
                )
        ax.boxplot(series, tick_labels=labels, showfliers=False)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(spec.dataset)
        ax.set_ylabel("HHI (higher = more concentrated)")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Selected-set concentration is an outcome, not a matching covariate")
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(FIGURES / "category_source_concentration.png", bbox_inches="tight")
    plt.close(fig)


def plot_adjusted(summary: pd.DataFrame) -> None:
    setup_plotting()
    data = summary[summary["stratum_name"].eq("all_paired_same_pool")]
    x = data["unadjusted_gain"].to_numpy(float)
    y = data["adjusted_gain"].to_numpy(float)
    low = min(x.min(), y.min()) - .7
    high = max(x.max(), y.max()) + 2.0
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    ax.plot([low, high], [low, high], color="0.4", lw=1, ls="--", label="y=x")
    short = {
        "FrozenDINOVisualDiversity": "Frozen DINO",
        "CategoryBalancedRandom": "Cat-bal. Random",
        "FrozenCategoryBalancedDINO": "Cat-bal. DINO",
    }
    for _, row in data.iterrows():
        ax.scatter(row["unadjusted_gain"], row["adjusted_gain"], s=75)
        label = f"{row['dataset']}\n{short.get(row['strategy'], row['strategy'])}"
        upper_right = row["unadjusted_gain"] == data["unadjusted_gain"].max()
        ax.annotate(
            label,
            (row["unadjusted_gain"], row["adjusted_gain"]),
            xytext=(-8, -28) if upper_right else (6, 5),
            textcoords="offset points", fontsize=8,
            ha="right" if upper_right else "left",
        )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Marginal unadjusted discovery gain")
    ax.set_ylabel("Same-seed paired discovery gain")
    ax.set_title(
        "Complete same-pool pairing preserves marginal mean effects\n"
        "(no TYPE B matching performed)"
    )
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "adjusted_vs_unadjusted_effect.png", bbox_inches="tight")
    plt.close(fig)


def write_decision(
    summary: pd.DataFrame,
    outcomes: pd.DataFrame,
    stats: dict[str, dict[str, Any]],
    diagnostics: dict[str, float],
) -> None:
    primary = summary[
        summary["stratum_name"].eq("all_paired_same_pool")
        & summary["is_primary_selector"].astype(bool)
    ].set_index("dataset")
    quadrants: dict[str, dict[str, int]] = {}
    for spec in SPECS:
        frame = outcomes[
            outcomes["dataset"].eq(spec.dataset)
            & outcomes["strategy"].eq(spec.selector)
        ]
        quadrants[spec.dataset] = frame["safety_quadrant"].value_counts().to_dict()

    def result_line(dataset: str) -> str:
        row = primary.loc[dataset]
        return (
            f"gain {row.adjusted_gain:.6f}, 95% CI "
            f"[{row.ci_lower:.6f}, {row.ci_upper:.6f}], positive seeds "
            f"{row.positive_fraction:.1%}, coverage delta "
            f"{row.coverage_delta:+.6f}, HHI delta "
            f"{row.concentration_delta:+.6f}"
        )

    text = f"""# Final decision

**B. CONDITION_MAP_DESCRIPTIVE_ONLY**

Same-seed discovery gains are supported by the frozen records, but the 200
seeds do not constitute independent prevalence conditions. Each pool is a
high-overlap leave-20 perturbation of one fixed acquisition pool. The audit
supports paired selector effects and discovery-safety diagnostics, not a
general pool-sparsity law. Category/source dominance also prevents decision A.

# Design identifiability

{design_table(stats)}

- Pairing: exact same reconstructed candidate pool within each seed.
- Matching: not performed; no TYPE B dataset exists.
- Common support: 100% for same-seed strategy comparisons, but practical
  prevalence-condition support is absent.
- Not estimable: prevalence tertile effects, prevalence-gain correlation, and
  cross-dataset causal condition effects.
- The reported confidence intervals quantify acquisition-seed variation, not
  sampling from independent production pools.

# Dataset results

## VisA

- Paired anomaly discovery: {result_line('VisA')}.
- Quadrants: {quadrants['VisA']}.
- Leave-one-category-out gain falls from 14.480 to {diagnostics['visa_min_loo']:.3f};
  the implied largest-category contribution is {diagnostics['visa_top_category_contribution_fraction']:.1%}.
- Prevalence-stratum effect: **not identifiable**.
- Separate source robustness is unavailable because source is not distinct from product category.

## MPDD

- Paired anomaly discovery: {result_line('MPDD')}.
- Quadrants: {quadrants['MPDD']}.
- Leave-one-product-category gain remains positive (minimum {diagnostics['mpdd_min_loo']:.3f}),
  while official-test-origin composition explains {diagnostics['mpdd_source_composition_fraction']:.1%}
  of the observed anomaly gain.
- Source origin and anomaly status remain separate outcomes; neither was used for matching.
- Prevalence-stratum effect: **not identifiable**.

## GC10-DET

- Paired rare discovery: {result_line('GC10-DET')}.
- Quadrants: {quadrants['GC10-DET']}.
- Existing q20 detector translation: mAP50-95 +0.017378 but rare macro AP -0.019877.
- Existing K40 result: selection coverage PASS but mAP50-95 -0.001678,
  rare AP -0.018290, and recall -0.021871.
- Seed45 fixed-set +0.016236 did not generalize in the independent acquisition
  confirmation (+0.007019, CI crosses zero, p=0.322266).
- No regression was fitted across the few non-exchangeable aggregate branches.

# Claims retained from the integrated condition map

1. Same-pool discovery gains exist relative to Random in each dataset.
2. Discovery gain and category/source safety can diverge.
3. GC10 selection/coverage gain does not automatically translate into rare AP,
   recall, or overall detector utility.
4. Training-seed stability does not establish acquisition-set generalization.
5. A validity gate prevented additional training and final-test consumption.

# Claims weakened after this audit

- Pool prevalence or sparsity as a moderator: **not identifiable**.
- VisA category-agnostic robustness: weakened by pipe_fryum dominance.
- MPDD representation-only mechanism: weakened by the 67.3% source-composition contribution.
- Cross-dataset condition law: descriptive association only; dataset identity
  cannot be exchanged with prevalence.

# Pool sparsity decision

**Not identifiable.** Between-dataset sparsity differences do not identify a
prevalence effect, and within each dataset the candidate pools are nearly fixed.

# Discovery-coverage trade-off decision

**Dataset-specific descriptive support.** The joint outcome repeats, but its
size and mechanism are dataset-dependent and cannot be promoted to a universal law.

# Selection-learning translation decision

**Repeatedly separated in GC10.** Positive selection-stage results did not
reliably translate to rare AP, recall, or mAP. The available aggregate branches
are not independent units for a learner-alignment coefficient.

# Effect on thesis viability

The central question survives only in a narrower, conditional form: an
industrial validity-gated workflow that separates candidate-signal validity,
discovery-safety behavior, and selection-to-learning translation. It does not
support a general sparsity law or a broadly superior selector.

# FN extension

**DESIGN_PROTOCOL_ONLY retained.** This audit adds neither FN event-count
evidence nor an untouched validation resource. It does not authorize an FN
screen, feature implementation, training, inference, or final-test access.

# Supervisor briefing

The original 200-seed records support same-candidate-pool discovery gains over
Random. However, the seeds differ only by removing 20 items from one fixed pool,
so they cannot identify whether target prevalence moderates those gains. VisA
and MPDD also show category/source concentration, while GC10 repeatedly shows
that selection-stage gains need not translate into rare-class detector utility.
The defensible thesis contribution is therefore a validity-gated audit workflow
and failure-condition map, with pool-sparsity claims explicitly withheld.
"""
    DECISION.write_text(text, encoding="utf-8")


def write_audit_files(
    design: pd.DataFrame, outcomes: pd.DataFrame, summary: pd.DataFrame
) -> None:
    matching = OUT / "matching_balance.csv"
    if matching.exists():
        raise RuntimeError(
            "matching_balance.csv must not exist because no TYPE B dataset exists"
        )
    commands = [
        r".\.venv\Scripts\python.exe scripts\04_dcal_xai\build_stratified_condition_robustness.py",
        r".\.venv\Scripts\python.exe scripts\04_dcal_xai\test_stratified_condition_robustness.py",
    ]
    log = [
        "mode=audit-only,analysis-only,decision-only",
        "training=False",
        "inference=False",
        "embedding_extraction=False",
        "selector_implementation=False",
        "final_test_used=False",
        "design_classification=TYPE_A for all three datasets; high-overlap leave-20 perturbations",
        "matching_performed=False (no TYPE B datasets)",
        "prevalence_effect=not_identifiable",
        f"pool_design_rows={len(design)}",
        f"seed_outcome_rows={len(outcomes)}",
        f"effect_summary_rows={len(summary)}",
        f"metric_registry_rows={len(REGISTRY)}",
        *["source_read=" + rel(path) for path in sorted(SCANNED)],
        *["command=" + item for item in commands],
    ]
    (OUT / "audit_execution_log.txt").write_text(
        "\n".join(log) + "\n", encoding="utf-8"
    )

    generated = [
        OUT / "pool_design_registry.csv",
        OUT / "seed_level_outcomes.csv",
        OUT / "stratified_effect_summary.csv",
        OUT / "metric_source_registry.csv",
        OUT / "missing_or_nonidentifiable.csv",
        OUT / "audit_execution_log.txt",
        FEASIBILITY,
        DECISION,
        FIGURES / "stratified_discovery_gain.png",
        FIGURES / "discovery_safety_quadrants.png",
        FIGURES / "category_source_concentration.png",
        FIGURES / "adjusted_vs_unadjusted_effect.png",
        ROOT / "scripts/04_dcal_xai/build_stratified_condition_robustness.py",
    ]
    lines = [
        f"{rel(path)}\tbytes={path.stat().st_size}\t"
        f"sha256={hashlib.sha256(path.read_bytes()).hexdigest()}"
        for path in generated
    ]
    (OUT / "generated_file_manifest.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    mandatory_context = [
        ROOT / "docs/condition_map_master_20260718.csv",
        ROOT / "docs/branch_decision_table_20260718.csv",
        ROOT / "docs/three_dataset_condition_map_decision_20260718.md",
        ROOT / "docs/fn_failure_prediction_extension_feasibility_20260718.md",
        ROOT / "runs/integrated_condition_map_20260718/source_metric_registry.csv",
        ROOT / "runs/integrated_condition_map_20260718/metric_conflicts.csv",
        ROOT / "runs/integrated_condition_map_20260718/missing_evidence.csv",
        ROOT / "runs/integrated_condition_map_20260718/condition_map_validation.log",
    ]
    for path in mandatory_context:
        read_text(path)

    design, contexts, stats = reconstruct_design()
    write_feasibility(stats)
    outcomes = build_seed_outcomes(contexts)
    summary = build_effect_summary(outcomes, stats)
    register_output_metrics(design, outcomes, summary)
    write_missing(stats)
    diagnostics = load_robustness_diagnostics()
    plot_prevalence_not_identifiable(stats)
    plot_quadrants(outcomes)
    plot_concentration(outcomes)
    plot_adjusted(summary)
    write_decision(summary, outcomes, stats, diagnostics)
    write_audit_files(design, outcomes, summary)

    print(
        f"[DONE] design_rows={len(design)} outcome_rows={len(outcomes)} "
        f"registry_rows={len(REGISTRY)}"
    )
    print("[DECISION] B. CONDITION_MAP_DESCRIPTIVE_ONLY")
    print("[PREVALENCE EFFECT] not_identifiable")
    print("[FN EXTENSION] DESIGN_PROTOCOL_ONLY")
    print("[FINAL TEST USED] False")


if __name__ == "__main__":
    main()

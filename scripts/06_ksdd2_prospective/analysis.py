"""Frozen G1-G4 analyses and authorization decisions for KSDD2."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import (
    assert_no_label_columns,
    read_csv,
    read_json,
    run_root,
    seal_file,
    verify_seal,
    write_csv,
    write_json,
)
from dataset_protocol import POSTHOC_FIELDS, audit_manifest_rows


SELECTION_FIELDS = [
    "sample_id",
    "image_path",
    "strategy",
    "acquisition_seed",
    "selection_rank",
    "patchcore_score",
]
SIZE_BINS = ("small", "medium", "large")


def _write_text_and_seal(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    seal_file(path)


def _ci(values: list[float], confidence: float) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    alpha = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    ]


def _bootstrap_rows(
    rows: list[dict[str, Any]],
    *,
    statistic: Callable[[list[dict[str, Any]]], float],
    replicates: int,
    confidence: float,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        sample = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        try:
            value = float(statistic(sample))
        except (ValueError, ZeroDivisionError):
            continue
        if math.isfinite(value):
            values.append(value)
    if len(values) < max(100, replicates // 4):
        raise RuntimeError(f"Only {len(values)} valid bootstrap replicates")
    return _ci(values, confidence)


def _auroc(rows: list[dict[str, Any]]) -> float:
    return float(roc_auc_score([row["has_defect"] for row in rows], [row["score"] for row in rows]))


def _spearman(rows: list[dict[str, Any]]) -> float:
    value = spearmanr(
        [row["score"] for row in rows],
        [row["bbox_count"] for row in rows],
    ).statistic
    return float(value)


def _top_enrichment(rows: list[dict[str, Any]], query_size: int) -> float:
    ordered = sorted(rows, key=lambda row: (-row["score"], row["sample_id"]))
    top = ordered[: min(query_size, len(ordered))]
    prevalence = sum(row["has_defect"] for row in rows) / len(rows)
    if prevalence == 0:
        raise ZeroDivisionError("Development pool has no defect images")
    return float((sum(row["has_defect"] for row in top) / len(top)) / prevalence)


def decide_g1(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gate = config["gates"]["g1_signal_validity"]
    checks = {
        "image_auroc": metrics["image_auroc"]["estimate"] >= float(gate["image_auroc_min"]),
        "image_auroc_ci_low": metrics["image_auroc"]["ci95"][0]
        >= float(gate["image_auroc_ci_low_min"]),
        "bbox_count_spearman": metrics["bbox_count_spearman"]["estimate"]
        >= float(gate["bbox_count_spearman_min"]),
        "bbox_count_spearman_ci_low": metrics["bbox_count_spearman"]["ci95"][0]
        >= float(gate["bbox_count_spearman_ci_low_min"]),
        "top_query_enrichment": metrics["top_query_enrichment"]["estimate"]
        >= float(gate["top_query_enrichment_min"]),
        "top_query_enrichment_ci_low": metrics["top_query_enrichment"]["ci95"][0]
        >= float(gate["top_query_enrichment_ci_low_min"]),
    }
    return {
        "gate": "G1_SIGNAL_VALIDITY",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "next_action": "PROCEED_TO_G2_G4_SELECTION" if all(checks.values()) else "STOP",
        "detector_training_authorized": False,
    }


def analyze_g1(config: dict[str, Any]) -> dict[str, Any]:
    out = run_root(config)
    seed = int(config["acquisition"]["g1_reference_seed"])
    score_path = out / "g1" / f"g1_scores_seed{seed:02d}.csv"
    if not verify_seal(score_path):
        raise RuntimeError("G1 score file is missing or its seal is invalid")
    scores = read_csv(score_path)
    assert_no_label_columns(scores, artifact="sealed G1 score file")
    development = read_csv(out / config["outputs"]["development_manifest"])
    score_ids = {row["sample_id"] for row in scores}
    if score_ids != {row["sample_id"] for row in development}:
        raise RuntimeError("G1 score IDs do not exactly match the frozen development manifest")

    labels = audit_manifest_rows(config, development, role="development_posthoc")
    label_path = out / "g1" / "g1_posthoc_development_labels.csv"
    write_csv(label_path, labels, POSTHOC_FIELDS)
    seal_file(label_path)
    labels_by_id = {row["sample_id"]: row for row in labels}
    joined = [
        {
            "sample_id": row["sample_id"],
            "score": float(row["patchcore_score"]),
            "has_defect": int(labels_by_id[row["sample_id"]]["has_defect"]),
            "bbox_count": int(labels_by_id[row["sample_id"]]["bbox_count"]),
        }
        for row in scores
    ]
    stats_cfg = config["statistics"]
    replicates = int(stats_cfg["bootstrap_replicates"])
    confidence = float(stats_cfg["confidence_level"])
    bootstrap_seed = int(stats_cfg["bootstrap_seed"])
    query_size = int(config["acquisition"]["query_size"])
    metrics = {
        "development_count": len(joined),
        "development_positive_images": sum(row["has_defect"] for row in joined),
        "query_size": query_size,
        "image_auroc": {
            "estimate": _auroc(joined),
            "ci95": _bootstrap_rows(
                joined,
                statistic=_auroc,
                replicates=replicates,
                confidence=confidence,
                seed=bootstrap_seed,
            ),
        },
        "bbox_count_spearman": {
            "estimate": _spearman(joined),
            "ci95": _bootstrap_rows(
                joined,
                statistic=_spearman,
                replicates=replicates,
                confidence=confidence,
                seed=bootstrap_seed + 1,
            ),
        },
        "top_query_enrichment": {
            "estimate": _top_enrichment(joined, query_size),
            "ci95": _bootstrap_rows(
                joined,
                statistic=lambda rows: _top_enrichment(rows, query_size),
                replicates=replicates,
                confidence=confidence,
                seed=bootstrap_seed + 2,
            ),
        },
    }
    decision = decide_g1(metrics, config)
    write_json(out / "g1" / "g1_metrics.json", metrics)
    write_json(out / "g1" / "g1_decision.json", decision)
    seal_file(out / "g1" / "g1_metrics.json")
    seal_file(out / "g1" / "g1_decision.json")
    _write_text_and_seal(
        out / "g1" / "g1_presentation_summary.md",
        f"""# KSDD2 prospective G1 결과

- 판정: **{decision['decision']}**
- Image AUROC: {metrics['image_auroc']['estimate']:.3f} (95% CI {metrics['image_auroc']['ci95'][0]:.3f}–{metrics['image_auroc']['ci95'][1]:.3f})
- BBox-count Spearman: {metrics['bbox_count_spearman']['estimate']:.3f} (95% CI {metrics['bbox_count_spearman']['ci95'][0]:.3f}–{metrics['bbox_count_spearman']['ci95'][1]:.3f})
- Top-{query_size} enrichment: {metrics['top_query_enrichment']['estimate']:.3f}× (95% CI {metrics['top_query_enrichment']['ci95'][0]:.3f}–{metrics['top_query_enrichment']['ci95'][1]:.3f})
- 다음 조치: **{decision['next_action']}**

해석 범위: frozen KSDD2 development split에서의 selection-signal validity만 다룬다.
Detector utility 또는 selector 우위를 뜻하지 않는다.
""",
    )
    return {"metrics": metrics, "decision": decision}


def require_g1_pass(config: dict[str, Any]) -> dict[str, Any]:
    path = run_root(config) / "g1" / "g1_decision.json"
    if not verify_seal(path):
        raise RuntimeError("G1 decision is missing or its seal is invalid")
    decision = read_json(path)
    if decision["decision"] != "PASS":
        raise RuntimeError("G1 did not pass. Frozen protocol requires STOP before selection expansion.")
    return decision


def create_selections(config: dict[str, Any], seed: int) -> dict[str, Any]:
    require_g1_pass(config)
    out = run_root(config)
    score_path = out / "selection_scores" / f"selection_scores_seed{seed:02d}.csv"
    if not verify_seal(score_path):
        raise RuntimeError(f"Selection scores for seed {seed} are missing or unsealed")
    scores = read_csv(score_path)
    assert_no_label_columns(scores, artifact=f"selection scores seed {seed}")
    query_size = int(config["acquisition"]["query_size"])
    if len(scores) < 2 * query_size:
        raise RuntimeError("Scoring pool is too small for the frozen query budget")

    ordered = sorted(
        scores,
        key=lambda row: (-float(row["patchcore_score"]), str(row["sample_id"])),
    )
    candidate = ordered[:query_size]
    rng = np.random.default_rng(
        int(seed) + int(config["acquisition"]["random_query_seed_offset"])
    )
    random_indices = rng.choice(len(scores), size=query_size, replace=False)
    random_rows = [scores[index] for index in sorted(random_indices.tolist())]

    def format_rows(rows: list[dict[str, str]], strategy: str) -> list[dict[str, Any]]:
        return [
            {
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                "strategy": strategy,
                "acquisition_seed": seed,
                "selection_rank": rank,
                "patchcore_score": row["patchcore_score"],
            }
            for rank, row in enumerate(rows, start=1)
        ]

    candidate_output = format_rows(candidate, "patchcore")
    random_output = format_rows(random_rows, "random")
    assert_no_label_columns(candidate_output, artifact="candidate selection")
    assert_no_label_columns(random_output, artifact="random selection")
    candidate_path = out / "selections" / f"selected_patchcore_seed{seed:02d}.csv"
    random_path = out / "selections" / f"selected_random_seed{seed:02d}.csv"
    write_csv(candidate_path, candidate_output, SELECTION_FIELDS)
    write_csv(random_path, random_output, SELECTION_FIELDS)
    seal_file(candidate_path)
    seal_file(random_path)
    return {
        "seed": seed,
        "candidate_count": len(candidate_output),
        "random_count": len(random_output),
        "selection_labels_consumed": False,
    }


def _tokens(value: str) -> set[str]:
    return {token for token in value.split("|") if token}


def _size_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        size: sum(int(row[f"{size}_component_count"]) for row in rows)
        for size in SIZE_BINS
    }


def _distribution(counts: dict[str, int]) -> np.ndarray:
    values = np.array([counts[size] for size in SIZE_BINS], dtype=float)
    if values.sum() == 0:
        return np.full(len(SIZE_BINS), 1.0 / len(SIZE_BINS))
    return values / values.sum()


def _jsd(selected: dict[str, int], pool: dict[str, int]) -> float:
    return float(jensenshannon(_distribution(selected), _distribution(pool), base=2.0) ** 2)


def _selection_summary(
    selected: list[dict[str, Any]],
    pool: list[dict[str, Any]],
) -> dict[str, Any]:
    size_counts = _size_counts(selected)
    pool_counts = _size_counts(pool)
    return {
        "positive_images": sum(int(row["has_defect"]) for row in selected),
        "bbox_count": sum(int(row["bbox_count"]) for row in selected),
        "small_defect_images": sum(
            int(int(row["small_component_count"]) > 0) for row in selected
        ),
        "size_bin_coverage": sum(value > 0 for value in size_counts.values()),
        "centroid_cell_coverage": len(
            set().union(*(_tokens(str(row["centroid_cells_present"])) for row in selected))
        ),
        "small_components": size_counts["small"],
        "medium_components": size_counts["medium"],
        "large_components": size_counts["large"],
        "js_divergence_vs_pool": _jsd(size_counts, pool_counts),
    }


def summarize_seed(
    *,
    seed: int,
    candidate_ids: set[str],
    random_ids: set[str],
    pool_rows: list[dict[str, Any]],
    ratio_correction: float,
) -> dict[str, Any]:
    labels = {str(row["sample_id"]): row for row in pool_rows}
    candidate = [labels[sample_id] for sample_id in sorted(candidate_ids)]
    random = [labels[sample_id] for sample_id in sorted(random_ids)]
    cand = _selection_summary(candidate, pool_rows)
    rand = _selection_summary(random, pool_rows)
    row: dict[str, Any] = {"acquisition_seed": seed}
    for prefix, values in (("patchcore", cand), ("random", rand)):
        row.update({f"{prefix}_{key}": value for key, value in values.items()})
    row.update(
        {
            "positive_image_delta": cand["positive_images"] - rand["positive_images"],
            "positive_image_enrichment": (
                (cand["positive_images"] + ratio_correction)
                / (rand["positive_images"] + ratio_correction)
            ),
            "small_defect_image_delta": cand["small_defect_images"]
            - rand["small_defect_images"],
            "size_bin_coverage_delta": cand["size_bin_coverage"]
            - rand["size_bin_coverage"],
            "js_divergence_vs_pool_delta": cand["js_divergence_vs_pool"]
            - rand["js_divergence_vs_pool"],
        }
    )
    return row


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    *,
    key: str,
    replicates: int,
    confidence: float,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.array([float(row[key]) for row in rows], dtype=float)
    means = [
        float(rng.choice(values, size=len(values), replace=True).mean())
        for _ in range(replicates)
    ]
    return _ci(means, confidence)


def decide_g2_g4(
    per_seed: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stats = config["statistics"]
    replicates = int(stats["bootstrap_replicates"])
    confidence = float(stats["confidence_level"])
    seed = int(stats["bootstrap_seed"]) + 100
    deltas = np.array([float(row["positive_image_delta"]) for row in per_seed])
    enrichments = np.array([float(row["positive_image_enrichment"]) for row in per_seed])
    small_deltas = np.array([float(row["small_defect_image_delta"]) for row in per_seed])
    coverage_deltas = np.array([float(row["size_bin_coverage_delta"]) for row in per_seed])
    jsd_deltas = np.array([float(row["js_divergence_vs_pool_delta"]) for row in per_seed])

    excess = {
        size: max(
            0,
            sum(int(row[f"patchcore_{size}_components"]) for row in per_seed)
            - sum(int(row[f"random_{size}_components"]) for row in per_seed),
        )
        for size in SIZE_BINS
    }
    total_excess = sum(excess.values())
    max_excess_share = max(excess.values()) / total_excess if total_excess else 0.0
    leave_one_out = [
        float(np.delete(deltas, index).mean())
        for index in range(len(deltas))
    ]
    metrics = {
        "seed_count": len(per_seed),
        "positive_image_delta_mean": float(deltas.mean()),
        "positive_image_delta_ci95": _paired_bootstrap(
            per_seed,
            key="positive_image_delta",
            replicates=replicates,
            confidence=confidence,
            seed=seed,
        ),
        "positive_image_enrichment_mean": float(enrichments.mean()),
        "positive_seed_fraction": float((deltas > 0).mean()),
        "small_defect_image_delta_mean": float(small_deltas.mean()),
        "size_bin_coverage_delta_mean": float(coverage_deltas.mean()),
        "js_divergence_vs_pool_delta_mean": float(jsd_deltas.mean()),
        "positive_excess_by_size_bin": excess,
        "positive_excess_max_size_bin_share": float(max_excess_share),
        "leave_one_seed_out_delta_means": leave_one_out,
        "leave_one_seed_out_min_delta": float(min(leave_one_out)),
    }
    g2 = config["gates"]["g2_target_discovery"]
    g3 = config["gates"]["g3_composition_safety"]
    g4 = config["gates"]["g4_acquisition_reproducibility"]
    checks = {
        "g2": {
            "positive_image_delta_mean": metrics["positive_image_delta_mean"]
            > float(g2["positive_image_delta_mean_min_exclusive"]),
            "positive_image_delta_ci_low": metrics["positive_image_delta_ci95"][0]
            > float(g2["positive_image_delta_ci_low_min_exclusive"]),
            "positive_image_enrichment_mean": metrics["positive_image_enrichment_mean"]
            >= float(g2["positive_image_enrichment_mean_min"]),
            "positive_seed_fraction": metrics["positive_seed_fraction"]
            >= float(g2["positive_seed_fraction_min"]),
        },
        "g3": {
            "small_defect_image_delta_mean": metrics["small_defect_image_delta_mean"]
            >= float(g3["small_defect_image_delta_mean_min"]),
            "size_bin_coverage_delta_mean": metrics["size_bin_coverage_delta_mean"]
            >= float(g3["size_bin_coverage_delta_mean_min"]),
            "js_divergence_vs_pool_delta_mean": metrics["js_divergence_vs_pool_delta_mean"]
            <= float(g3["js_divergence_vs_pool_delta_mean_max"]),
            "positive_excess_max_size_bin_share": metrics[
                "positive_excess_max_size_bin_share"
            ]
            <= float(g3["positive_excess_max_size_bin_share_max"]),
        },
        "g4": {
            "required_seed_count": len(per_seed) == int(g4["required_seed_count"]),
            "leave_one_seed_out_min_delta": metrics["leave_one_seed_out_min_delta"]
            > float(g4["leave_one_seed_out_min_delta_min_exclusive"]),
            "positive_seed_fraction": metrics["positive_seed_fraction"]
            >= float(g4["positive_seed_fraction_min"]),
        },
    }
    passed = {gate: all(values.values()) for gate, values in checks.items()}
    if not passed["g2"]:
        action = config["authorization"]["g2_fail_action"]
    elif not passed["g3"]:
        action = config["authorization"]["g2_pass_g3_fail_action"]
    elif not passed["g4"]:
        action = config["authorization"]["g4_fail_action"]
    else:
        action = config["authorization"]["g1_g2_g3_g4_pass_action"]
    decision = {
        "protocol_id": config["protocol"]["id"],
        "g1": "PASS",
        "g2": "PASS" if passed["g2"] else "FAIL",
        "g3": "PASS" if passed["g3"] else "FAIL",
        "g4": "PASS" if passed["g4"] else "FAIL",
        "checks": checks,
        "authorization": action,
        "detector_training_authorized": action == "BOUNDED_SCREEN",
        "official_test_use_authorized": False,
        "claim_scope": config["protocol"]["contribution_scope"].strip(),
        "bounded_screen_scope": config["authorization"]["bounded_screen_scope"].strip(),
    }
    return metrics, decision


def analyze_g2_g4(config: dict[str, Any]) -> dict[str, Any]:
    require_g1_pass(config)
    out = run_root(config)
    seeds = [int(seed) for seed in config["acquisition"]["seeds"]]

    # Selection manifests for every seed must be sealed before any pool mask is opened.
    for seed in seeds:
        for strategy in ("patchcore", "random"):
            path = out / "selections" / f"selected_{strategy}_seed{seed:02d}.csv"
            if not verify_seal(path):
                raise RuntimeError(f"Missing or invalid selection seal: {path.name}")
            assert_no_label_columns(read_csv(path), artifact=path.name)

    acquisition = read_csv(out / config["outputs"]["acquisition_manifest"])
    posthoc = audit_manifest_rows(config, acquisition, role="acquisition_posthoc")
    posthoc_path = out / "g2_g4" / "posthoc_acquisition_labels.csv"
    write_csv(posthoc_path, posthoc, POSTHOC_FIELDS)
    seal_file(posthoc_path)
    posthoc_by_id = {row["sample_id"]: row for row in posthoc}

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        initial_ids = {
            row["sample_id"]
            for row in read_csv(out / "initial_sets" / f"initial_seed{seed:02d}.csv")
        }
        pool = [row for row in posthoc if row["sample_id"] not in initial_ids]
        candidate = {
            row["sample_id"]
            for row in read_csv(out / "selections" / f"selected_patchcore_seed{seed:02d}.csv")
        }
        random_ids = {
            row["sample_id"]
            for row in read_csv(out / "selections" / f"selected_random_seed{seed:02d}.csv")
        }
        query_size = int(config["acquisition"]["query_size"])
        if len(candidate) != query_size or len(random_ids) != query_size:
            raise RuntimeError(
                f"Seed {seed} selection cardinality changed: "
                f"PatchCore={len(candidate)}, Random={len(random_ids)}, expected={query_size}"
            )
        pool_ids = {row["sample_id"] for row in pool}
        if not candidate <= pool_ids or not random_ids <= pool_ids:
            raise RuntimeError(f"Seed {seed} selection is not a subset of its frozen query pool")
        if not candidate <= posthoc_by_id.keys() or not random_ids <= posthoc_by_id.keys():
            raise RuntimeError(f"Seed {seed} selection includes an ID outside the acquisition manifest")
        per_seed.append(
            summarize_seed(
                seed=seed,
                candidate_ids=candidate,
                random_ids=random_ids,
                pool_rows=pool,
                ratio_correction=float(config["statistics"]["zero_count_ratio_correction"]),
            )
        )

    metrics, decision = decide_g2_g4(per_seed, config)
    per_seed_path = out / "g2_g4" / "per_seed_metrics.csv"
    write_csv(per_seed_path, per_seed, list(per_seed[0]))
    write_json(out / "g2_g4" / "g2_g4_metrics.json", metrics)
    write_json(out / config["outputs"]["authorization_decision"], decision)
    for path in (
        per_seed_path,
        out / "g2_g4" / "g2_g4_metrics.json",
        out / config["outputs"]["authorization_decision"],
    ):
        seal_file(path)
    _write_text_and_seal(
        out / "g2_g4" / "presentation_summary.md",
        f"""# KSDD2 prospective G1–G4 결과

- 최종 authorization: **{decision['authorization']}**
- G1 / G2 / G3 / G4: **PASS / {decision['g2']} / {decision['g3']} / {decision['g4']}**
- PatchCore−Random 결함 이미지 차이: 평균 {metrics['positive_image_delta_mean']:.2f}장
  (paired 95% CI {metrics['positive_image_delta_ci95'][0]:.2f}–{metrics['positive_image_delta_ci95'][1]:.2f})
- 평균 enrichment: {metrics['positive_image_enrichment_mean']:.3f}×
- 양의 효과 seed 비율: {metrics['positive_seed_fraction']:.0%}
- 작은 결함 이미지 차이: 평균 {metrics['small_defect_image_delta_mean']:.2f}장
- Pool 대비 JSD의 Random 대비 차이: {metrics['js_divergence_vs_pool_delta_mean']:.4f}
- Leave-one-seed-out 최소 결함 이미지 차이: {metrics['leave_one_seed_out_min_delta']:.2f}장

발표 문장: 이 결과는 성능 우위 주장이 아니라, 사전 동결된 gate가 실제로
다음 단계의 허용/중단을 결정한 prospective process-conformance 사례다.

금지된 해석: detector 성능 향상, 독립 생산 pool 일반화, 비용 절감,
workflow 예측 정확도, 공식 test 성능.
""",
    )
    return {"metrics": metrics, "decision": decision}

"""Build the GC10 cost-translation protocol-candidate manifests.

This script performs no annotation, model training, or final-test access.  It
reads only the frozen GC10 acquisition-pool audits and the existing 200-seed
selection record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_SOURCE = ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715"
DEFAULT_SELECTION_SOURCE = (
    ROOT
    / "runs"
    / "gc10_taxonomy_selection_audit"
    / "gc10_random_vs_dino_200seed_20260715"
)
DEFAULT_OUTPUT = ROOT / "protocols" / "gc10_cost_translation_v1"

PROTOCOL_VERSION = "gc10-cost-translation-v1-candidate"
SOURCE_COMMIT = "ee331ffd9e7f93951d570ef2020735f1f66d904a"
PRIMARY_SEEDS = list(range(5, 15))
RANDOM = "GTFreeRandom"
DINO = "FrozenDINOVisualDiversity"
STRATEGIES = [RANDOM, DINO]
QUERY_SIZE = 20
RARE_CLASSES = {8, 9, 10}
SMALL_BOX_AREA_RATIO_THRESHOLD = 0.004831
STAGE0_RANDOM_SEED = 20260730

SELECTION_REL = (
    "runs/gc10_taxonomy_selection_audit/"
    "gc10_random_vs_dino_200seed_20260715/gc10_selection_records_posthoc.csv"
)
GT_REL = "runs/gc10_taxonomy_protocol/gc10_protocol_20260715/gc10_acquisition_pool_gt_audit.csv"
BBOX_REL = "runs/gc10_taxonomy_protocol/gc10_protocol_20260715/gc10_acquisition_bbox_gt_audit.csv"
LOADER_REL = "runs/gc10_taxonomy_protocol/gc10_protocol_20260715/gc10_acquisition_loader_private.csv"

IMAGE_MANIFEST_REQUIRED = [
    "sample_id",
    "image_path",
    "split",
    "acquisition_seed",
    "strategy",
    "query_rank",
    "blinded_task_id",
    "bbox_count",
    "class_ids",
    "rare_flag",
    "small_box_flag",
    "multilabel_flag",
    "stage",
    "exclusion_reason",
    "source_file",
    "source_commit",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def task_id(stage: str, sample_id: str) -> str:
    prefix = "S0" if stage == "stage0_instrumentation_smoke" else "PA"
    token = sha256_text(f"{PROTOCOL_VERSION}|{stage}|{sample_id}")[:16].upper()
    return f"GC10CT-{prefix}-{token}"


def class_set(value: object) -> set[int]:
    if pd.isna(value):
        return set()
    return {int(float(item)) for item in str(value).split("|") if item}


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def repository_relative(path_value: object) -> str:
    path = Path(str(path_value)).expanduser().resolve()
    return path.relative_to(ROOT.resolve()).as_posix()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def deterministic_order_key(sample_id: str, stratum: str) -> str:
    return sha256_text(f"{STAGE0_RANDOM_SEED}|{stratum}|{sample_id}")


def load_inputs(
    protocol_source: Path, selection_source: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection = pd.read_csv(selection_source / "gc10_selection_records_posthoc.csv")
    gt = pd.read_csv(protocol_source / "gc10_acquisition_pool_gt_audit.csv")
    boxes = pd.read_csv(protocol_source / "gc10_acquisition_bbox_gt_audit.csv")
    loader = pd.read_csv(protocol_source / "gc10_acquisition_loader_private.csv")

    if set(selection["strategy"].astype(str)) != set(STRATEGIES):
        raise RuntimeError("Unexpected selection strategies")
    if set(gt["protocol_split"].astype(str)) != {"acquisition"}:
        raise RuntimeError("Acquisition GT source contains a non-acquisition split")
    if set(gt["sample_id"]) != set(loader["sample_id"]):
        raise RuntimeError("Acquisition loader and GT sample IDs differ")

    box_summary = (
        boxes.groupby("sample_id", as_index=False)
        .agg(
            bbox_count=("object_index", "count"),
            min_bbox_area_ratio=("bbox_area_ratio", "min"),
            max_bbox_area_ratio=("bbox_area_ratio", "max"),
        )
    )
    metadata = gt.merge(
        loader[["sample_id", "image_path"]].rename(columns={"image_path": "loader_image_path"}),
        on="sample_id",
        validate="one_to_one",
    ).merge(box_summary, on="sample_id", validate="one_to_one")
    if not (metadata["num_instances"].astype(int) == metadata["bbox_count"].astype(int)).all():
        raise RuntimeError("GT instance count and bbox audit count differ")
    metadata["image_path"] = metadata["loader_image_path"].map(repository_relative)
    metadata["class_id_set"] = metadata["class_ids"].map(class_set)
    metadata["rare_flag"] = metadata["class_id_set"].map(lambda values: bool(values & RARE_CLASSES))
    metadata["small_box_flag"] = (
        metadata["min_bbox_area_ratio"].astype(float) <= SMALL_BOX_AREA_RATIO_THRESHOLD
    )
    metadata["multilabel_flag"] = metadata["num_unique_classes"].astype(int) > 1
    metadata["large_box_threshold"] = float(boxes["bbox_area_ratio"].quantile(0.75))
    metadata["large_box_flag"] = (
        metadata["max_bbox_area_ratio"].astype(float) >= metadata["large_box_threshold"]
    )
    metadata["split"] = "acquisition"

    q25 = float(boxes["bbox_area_ratio"].quantile(0.25))
    if abs(q25 - 0.0048310546875) > 1e-12:
        raise RuntimeError(f"Unexpected acquisition bbox q25: {q25}")
    return selection, metadata, boxes


def build_phase_a_batch(selection: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    selected = selection[
        selection["acquisition_seed"].isin(PRIMARY_SEEDS)
        & selection["strategy"].isin(STRATEGIES)
    ].copy()
    expected = len(PRIMARY_SEEDS) * len(STRATEGIES) * QUERY_SIZE
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} selected rows, found {len(selected)}")
    selected = selected.rename(
        columns={
            "rank_in_query": "query_rank",
            "class_ids_posthoc": "selection_class_ids_posthoc",
            "num_instances_posthoc": "selection_num_instances_posthoc",
        }
    )
    keep = [
        "sample_id",
        "image_path",
        "split",
        "bbox_count",
        "class_ids",
        "rare_flag",
        "small_box_flag",
        "multilabel_flag",
        "large_box_flag",
        "min_bbox_area_ratio",
        "max_bbox_area_ratio",
    ]
    batch = selected.merge(metadata[keep], on="sample_id", validate="many_to_one")
    batch["blinded_task_id"] = batch["sample_id"].map(
        lambda value: task_id("phase_a_annotation_burden_quality", str(value))
    )
    batch["stage"] = "phase_a_annotation_burden_quality"
    batch["exclusion_reason"] = ""
    batch["source_file"] = SELECTION_REL
    batch["source_commit"] = SOURCE_COMMIT

    aggregates = (
        batch.groupby(["acquisition_seed", "strategy"], as_index=False)
        .agg(
            batch_bbox_count=("bbox_count", "sum"),
            batch_rare_image_count=("rare_flag", "sum"),
            batch_multilabel_image_count=("multilabel_flag", "sum"),
            batch_small_box_image_count=("small_box_flag", "sum"),
        )
    )
    batch = batch.merge(aggregates, on=["acquisition_seed", "strategy"], validate="many_to_one")
    batch["rare_flag"] = batch["rare_flag"].map(bool_text)
    batch["small_box_flag"] = batch["small_box_flag"].map(bool_text)
    batch["multilabel_flag"] = batch["multilabel_flag"].map(bool_text)
    batch["large_box_flag"] = batch["large_box_flag"].map(bool_text)
    order = IMAGE_MANIFEST_REQUIRED + [
        "large_box_flag",
        "min_bbox_area_ratio",
        "max_bbox_area_ratio",
        "batch_bbox_count",
        "batch_rare_image_count",
        "batch_multilabel_image_count",
        "batch_small_box_image_count",
    ]
    return batch[order].sort_values(
        ["acquisition_seed", "strategy", "query_rank"], kind="mergesort"
    )


def build_phase_a_unique(batch: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id, frame in batch.groupby("sample_id", sort=True):
        frame = frame.sort_values(["strategy", "acquisition_seed", "query_rank"], kind="mergesort")
        first = frame.iloc[0]
        memberships = [
            f"{row.strategy}:{int(row.acquisition_seed)}:{int(row.query_rank)}"
            for row in frame.itertuples(index=False)
        ]
        strategies = sorted(set(frame["strategy"].astype(str)))
        seeds = sorted(set(frame["acquisition_seed"].astype(int)))
        ranks = [int(value) for value in frame["query_rank"]]
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": first["image_path"],
                "split": "acquisition",
                "acquisition_seed": "|".join(map(str, seeds)),
                "strategy": "|".join(strategies),
                "query_rank": "|".join(map(str, ranks)),
                "blinded_task_id": first["blinded_task_id"],
                "bbox_count": int(first["bbox_count"]),
                "class_ids": first["class_ids"],
                "rare_flag": first["rare_flag"],
                "small_box_flag": first["small_box_flag"],
                "multilabel_flag": first["multilabel_flag"],
                "stage": "phase_a_annotation_burden_quality",
                "exclusion_reason": "",
                "source_file": SELECTION_REL,
                "source_commit": SOURCE_COMMIT,
                "membership_count": len(frame),
                "membership_descriptors": "|".join(memberships),
                "selection_frequency": len(frame),
                "cross_strategy_flag": bool_text(len(strategies) > 1),
                "large_box_flag": first["large_box_flag"],
                "min_bbox_area_ratio": first["min_bbox_area_ratio"],
                "max_bbox_area_ratio": first["max_bbox_area_ratio"],
            }
        )
    unique = pd.DataFrame(rows)
    return unique.sort_values("blinded_task_id", kind="mergesort").reset_index(drop=True)


def choose_stage0(metadata: pd.DataFrame, phase_a_ids: set[str]) -> pd.DataFrame:
    candidates = metadata[~metadata["sample_id"].astype(str).isin(phase_a_ids)].copy()
    candidates["bbox_count"] = candidates["bbox_count"].astype(int)

    strata = [
        ("rare_small", candidates["rare_flag"] & candidates["small_box_flag"]),
        (
            "rare_multibox",
            candidates["rare_flag"] & (candidates["bbox_count"] >= 2),
        ),
        (
            "rare_regular",
            candidates["rare_flag"]
            & ~candidates["small_box_flag"]
            & ~candidates["multilabel_flag"],
        ),
        (
            "small_common",
            ~candidates["rare_flag"] & candidates["small_box_flag"],
        ),
        (
            "large_common",
            ~candidates["rare_flag"] & candidates["large_box_flag"],
        ),
        (
            "multilabel_common",
            ~candidates["rare_flag"] & candidates["multilabel_flag"],
        ),
        (
            "single_box_common",
            ~candidates["rare_flag"] & (candidates["bbox_count"] == 1),
        ),
        (
            "high_box_count_common",
            ~candidates["rare_flag"] & (candidates["bbox_count"] >= 3),
        ),
    ]
    selected_ids: set[str] = set()
    chosen: list[pd.Series] = []
    for stratum, mask in strata:
        eligible = candidates[mask & ~candidates["sample_id"].astype(str).isin(selected_ids)].copy()
        eligible["_order"] = eligible["sample_id"].astype(str).map(
            lambda value: deterministic_order_key(value, stratum)
        )
        eligible = eligible.sort_values(["_order", "sample_id"], kind="mergesort")
        if len(eligible) < 3:
            raise RuntimeError(f"Stage 0 stratum {stratum} has only {len(eligible)} candidates")
        for _, row in eligible.head(3).iterrows():
            copy = row.copy()
            copy["stage0_stratum"] = stratum
            chosen.append(copy)
            selected_ids.add(str(row["sample_id"]))
    stage0 = pd.DataFrame(chosen)
    if len(stage0) != 24 or stage0["sample_id"].nunique() != 24:
        raise RuntimeError("Stage 0 selection cardinality failure")
    stage0["acquisition_seed"] = ""
    stage0["strategy"] = ""
    stage0["query_rank"] = ""
    stage0["blinded_task_id"] = stage0["sample_id"].map(
        lambda value: task_id("stage0_instrumentation_smoke", str(value))
    )
    stage0["stage"] = "stage0_instrumentation_smoke"
    stage0["exclusion_reason"] = ""
    stage0["source_file"] = f"{GT_REL}|{BBOX_REL}"
    stage0["source_commit"] = SOURCE_COMMIT
    stage0["selection_random_seed"] = STAGE0_RANDOM_SEED
    stage0["selection_rule"] = (
        "3 per predeclared stress stratum; order by "
        "SHA256(random_seed|stratum|sample_id); Phase A union excluded"
    )
    stage0["no_box_available_in_acquisition"] = "false"
    stage0["low_burden_proxy"] = stage0["stage0_stratum"].map(
        lambda value: "single_box" if value == "single_box_common" else ""
    )
    for column in ["rare_flag", "small_box_flag", "multilabel_flag", "large_box_flag"]:
        stage0[column] = stage0[column].map(bool_text)
    order = IMAGE_MANIFEST_REQUIRED + [
        "stage0_stratum",
        "large_box_flag",
        "min_bbox_area_ratio",
        "max_bbox_area_ratio",
        "selection_random_seed",
        "selection_rule",
        "no_box_available_in_acquisition",
        "low_burden_proxy",
    ]
    return stage0[order].sort_values("blinded_task_id", kind="mergesort")


def build_blinded_export(frame: pd.DataFrame, include_order_template: bool = True) -> pd.DataFrame:
    export = frame[["blinded_task_id", "sample_id", "image_path", "split", "stage"]].copy()
    if include_order_template:
        export["order_index"] = ""
        export["session_id"] = ""
    export["protocol_version"] = PROTOCOL_VERSION
    return export.sort_values("blinded_task_id", kind="mergesort").reset_index(drop=True)


def build_overlap_matrix(batch: pd.DataFrame) -> pd.DataFrame:
    sets = {
        (strategy, seed): set(
            batch[
                batch["strategy"].eq(strategy)
                & batch["acquisition_seed"].eq(seed)
            ]["sample_id"].astype(str)
        )
        for strategy in STRATEGIES
        for seed in PRIMARY_SEEDS
    }
    rows: list[dict[str, object]] = []
    for strategy_a in STRATEGIES:
        for strategy_b in STRATEGIES:
            comparison = (
                "within_strategy" if strategy_a == strategy_b else "cross_strategy"
            )
            for seed_a in PRIMARY_SEEDS:
                for seed_b in PRIMARY_SEEDS:
                    left, right = sets[(strategy_a, seed_a)], sets[(strategy_b, seed_b)]
                    rows.append(
                        {
                            "comparison_type": comparison,
                            "strategy_a": strategy_a,
                            "seed_a": seed_a,
                            "strategy_b": strategy_b,
                            "seed_b": seed_b,
                            "set_size_a": len(left),
                            "set_size_b": len(right),
                            "intersection_size": len(left & right),
                            "union_size": len(left | right),
                            "jaccard": jaccard(left, right),
                            "source_file": SELECTION_REL,
                            "source_commit": SOURCE_COMMIT,
                        }
                    )
    return pd.DataFrame(rows)


def build_image_seed_membership(batch: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (sample_id, strategy), frame in batch.groupby(["sample_id", "strategy"], sort=True):
        frame = frame.sort_values("acquisition_seed", kind="mergesort")
        first = frame.iloc[0]
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": first["image_path"],
                "split": first["split"],
                "acquisition_seed": "|".join(map(str, frame["acquisition_seed"].astype(int))),
                "strategy": strategy,
                "query_rank": "|".join(map(str, frame["query_rank"].astype(int))),
                "blinded_task_id": first["blinded_task_id"],
                "bbox_count": first["bbox_count"],
                "class_ids": first["class_ids"],
                "rare_flag": first["rare_flag"],
                "small_box_flag": first["small_box_flag"],
                "multilabel_flag": first["multilabel_flag"],
                "stage": first["stage"],
                "exclusion_reason": "",
                "source_file": SELECTION_REL,
                "source_commit": SOURCE_COMMIT,
                "selection_frequency": len(frame),
                "seed_list": "|".join(map(str, frame["acquisition_seed"].astype(int))),
                "rank_list": "|".join(map(str, frame["query_rank"].astype(int))),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["strategy", "selection_frequency", "sample_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )


def overlap_minimized_seed_sensitivity(selection: pd.DataFrame) -> pd.DataFrame:
    candidates = list(range(5, 200))
    sets = {
        (strategy, seed): set(
            selection[
                selection["strategy"].eq(strategy)
                & selection["acquisition_seed"].eq(seed)
            ]["sample_id"].astype(str)
        )
        for strategy in STRATEGIES
        for seed in candidates
    }
    for key, values in sets.items():
        if len(values) != QUERY_SIZE:
            raise RuntimeError(f"Incomplete selection set for {key}: {len(values)}")

    chosen = [5]
    rows = [
        {
            "selection_order": 1,
            "acquisition_seed": 5,
            "criterion_max_same_strategy_jaccard": 0.0,
            "max_random_jaccard": 0.0,
            "max_dino_jaccard": 0.0,
            "rule": "first seed fixed to 5",
            "primary_seed_replacement_allowed": "false",
        }
    ]
    while len(chosen) < 10:
        scored: list[tuple[float, int, float, float]] = []
        for candidate in candidates:
            if candidate in chosen:
                continue
            random_max = max(
                jaccard(sets[(RANDOM, candidate)], sets[(RANDOM, existing)])
                for existing in chosen
            )
            dino_max = max(
                jaccard(sets[(DINO, candidate)], sets[(DINO, existing)])
                for existing in chosen
            )
            score = max(random_max, dino_max)
            scored.append((score, candidate, random_max, dino_max))
        score, candidate, random_max, dino_max = min(scored, key=lambda item: (item[0], item[1]))
        chosen.append(candidate)
        rows.append(
            {
                "selection_order": len(chosen),
                "acquisition_seed": candidate,
                "criterion_max_same_strategy_jaccard": score,
                "max_random_jaccard": random_max,
                "max_dino_jaccard": dino_max,
                "rule": (
                    "among learner-unused seeds 5-199, minimize the maximum "
                    "same-strategy sample-ID Jaccard to already selected seeds; "
                    "ties use smaller seed"
                ),
                "primary_seed_replacement_allowed": "false",
            }
        )
    return pd.DataFrame(rows)


def build_phase_b_template(batch: pd.DataFrame) -> pd.DataFrame:
    template = batch.copy()
    template["primary_time_budget_T_ms"] = ""
    template["annotator_active_time_ms"] = ""
    template["annotator_id_summary"] = ""
    template["cumulative_strategy_time_ms"] = ""
    template["included_at_primary_T"] = ""
    template["human_label_path"] = ""
    template["adjudicated_label_path"] = ""
    template["quality_gate_pass"] = ""
    template["phase_a_authorization"] = ""
    template["training_seed"] = ""
    template["learner_run_status"] = "NOT_AUTHORIZED_CANDIDATE"
    template["official_final_test_used"] = "false"
    template["stage"] = "phase_b_equal_time_learner_template"
    return template


def write_source_ledger(output: Path, source_paths: list[Path]) -> None:
    lines = []
    for path in source_paths:
        lines.append(f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}")
    (output / "source_evidence_hashes.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_manifest_ledger(output: Path, paths: list[Path]) -> None:
    lines = []
    for path in sorted(paths, key=lambda item: item.name):
        lines.append(f"{sha256_file(path)}  {path.name}")
    (output / "manifest_hashes.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-source", type=Path, default=DEFAULT_PROTOCOL_SOURCE)
    parser.add_argument("--selection-source", type=Path, default=DEFAULT_SELECTION_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    protocol_source = args.protocol_source.expanduser().resolve()
    selection_source = args.selection_source.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    selection, metadata, boxes = load_inputs(protocol_source, selection_source)
    batch = build_phase_a_batch(selection, metadata)
    unique = build_phase_a_unique(batch)
    if len(unique) != 233:
        raise RuntimeError(f"Expected Phase A union 233, found {len(unique)}")
    stage0 = choose_stage0(metadata, set(unique["sample_id"].astype(str)))
    if set(stage0["sample_id"]) & set(unique["sample_id"]):
        raise RuntimeError("Stage 0 overlaps Phase A")

    overlap = build_overlap_matrix(batch)
    membership = build_image_seed_membership(batch)
    phase_b = build_phase_b_template(batch)
    sensitivity = overlap_minimized_seed_sensitivity(selection)
    stage0_blinded = build_blinded_export(stage0)
    phase_a_blinded = build_blinded_export(unique)

    outputs = {
        "stage0_instrumentation_manifest.csv": stage0,
        "phase_a_unique_image_manifest.csv": unique,
        "phase_a_seed_batch_manifest.csv": batch,
        "phase_b_frozen_candidate_manifest_template.csv": phase_b,
        "acquisition_overlap_matrix.csv": overlap,
        "image_seed_membership.csv": membership,
        "overlap_minimized_seed_sensitivity.csv": sensitivity,
        "stage0_annotator_blinded_manifest.csv": stage0_blinded,
        "phase_a_annotator_blinded_manifest.csv": phase_a_blinded,
    }
    written: list[Path] = []
    for filename, frame in outputs.items():
        path = output / filename
        write_csv(frame, path)
        written.append(path)

    metadata_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "source_commit": SOURCE_COMMIT,
        "primary_acquisition_seeds": PRIMARY_SEEDS,
        "query_size": QUERY_SIZE,
        "strategies": STRATEGIES,
        "phase_a_unique_images": len(unique),
        "phase_a_batch_rows": len(batch),
        "stage0_images": len(stage0),
        "stage0_random_seed": STAGE0_RANDOM_SEED,
        "stage0_no_box_available": False,
        "stage0_low_burden_proxy": "single-box images",
        "small_box_area_ratio_threshold": SMALL_BOX_AREA_RATIO_THRESHOLD,
        "small_box_source_q25_exact": float(boxes["bbox_area_ratio"].quantile(0.25)),
        "large_box_source_q75": float(boxes["bbox_area_ratio"].quantile(0.75)),
        "official_or_final_source_read_by_builder": False,
        "candidate_status": "UNFROZEN_RESEARCHER_REVIEW_REQUIRED",
    }
    (output / "manifest_build_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    source_paths = [
        selection_source / "gc10_selection_records_posthoc.csv",
        protocol_source / "gc10_acquisition_pool_gt_audit.csv",
        protocol_source / "gc10_acquisition_bbox_gt_audit.csv",
        protocol_source / "gc10_acquisition_loader_private.csv",
    ]
    write_source_ledger(output, source_paths)
    write_manifest_ledger(output, written)

    print(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "stage0_images": len(stage0),
                "phase_a_unique_images": len(unique),
                "phase_a_batch_rows": len(batch),
                "primary_seeds": PRIMARY_SEEDS,
                "sensitivity_seeds": sensitivity["acquisition_seed"].astype(int).tolist(),
                "official_final_source_read": False,
                "output_dir": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Validate the GC10 cost-translation protocol-candidate package.

The validator is read-only.  It does not open the final/official split manifest,
run annotation, or train a learner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = ROOT / "protocols" / "gc10_cost_translation_v1"
PROTOCOL_VERSION = "gc10-cost-translation-v1-candidate"
PRIMARY_SEEDS = set(range(5, 15))
STRATEGIES = {"GTFreeRandom", "FrozenDINOVisualDiversity"}
QUERY_RANKS = set(range(1, 21))
REQUIRED_EVENTS = {
    "SESSION_START",
    "IMAGE_START",
    "IMAGE_PAUSE",
    "IMAGE_RESUME",
    "BOX_CREATE",
    "BOX_MOVE",
    "BOX_RESIZE",
    "BOX_DELETE",
    "CLASS_ASSIGN",
    "CLASS_CHANGE",
    "IMAGE_SUBMIT",
    "SESSION_END",
}
REQUIRED_EVENT_FIELDS = {
    "protocol_version",
    "tool_version",
    "annotator_id",
    "session_id",
    "blinded_task_id",
    "sample_id",
    "order_index",
    "event_id",
    "event_type",
    "event_utc",
    "monotonic_timestamp_ms",
    "active_ms_since_previous",
    "idle_flag",
    "pause_reason",
    "box_id",
    "bbox_before",
    "bbox_after",
    "class_before",
    "class_after",
    "checksum",
}
REQUIRED_QUALITY_METRICS = {
    "box_precision",
    "box_recall",
    "false_negative_count",
    "image_level_missed_defect_rate",
    "matched_box_mean_iou",
    "matched_box_median_iou",
    "class_accuracy",
    "duplicate_count",
    "small_defect_recall",
    "rare_class_recall",
    "adjudication_correction_count",
    "adjudication_active_time_ms",
}
IMAGE_MANIFEST_REQUIRED = {
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
}
BLINDED_FORBIDDEN = {
    "strategy",
    "acquisition_seed",
    "query_rank",
    "bbox_count",
    "class_ids",
    "rare_flag",
    "small_box_flag",
    "multilabel_flag",
    "large_box_flag",
    "source_file",
    "source_commit",
    "selection_frequency",
    "membership_descriptors",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_hash_ledger(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, filename = re.split(r"\s{2,}", line.strip(), maxsplit=1)
        rows.append((digest, filename))
    return rows


def add_check(checks: list[dict[str, object]], name: str, passed: bool, detail: object) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable validation report.",
    )
    args = parser.parse_args()
    package = args.package_dir.expanduser().resolve()
    checks: list[dict[str, object]] = []

    required_files = [
        "stage0_instrumentation_manifest.csv",
        "phase_a_unique_image_manifest.csv",
        "phase_a_seed_batch_manifest.csv",
        "phase_b_frozen_candidate_manifest_template.csv",
        "acquisition_overlap_matrix.csv",
        "image_seed_membership.csv",
        "manifest_hashes.sha256",
        "source_evidence_hashes.sha256",
        "stage0_annotator_blinded_manifest.csv",
        "phase_a_annotator_blinded_manifest.csv",
        "annotation_event_schema.json",
        "annotation_event_examples.jsonl",
        "annotation_logger_requirements.md",
        "annotation_export_validation_spec.md",
        "annotation_quality_protocol.md",
        "annotation_quality_metric_schema.json",
        "statistical_analysis_plan.md",
        "statistical_analysis_config.template.yaml",
        "unresolved_researcher_decisions.md",
        "threshold_option_matrix.md",
        "human_participation_checklist.md",
        "annotator_information_sheet_draft.md",
        "annotation_guide_outline.md",
    ]
    missing = [filename for filename in required_files if not (package / filename).exists()]
    add_check(checks, "required_package_files_present", not missing, missing)
    if missing:
        result = {
            "protocol_version": PROTOCOL_VERSION,
            "overall_pass": False,
            "status": "NOT_READY",
            "checks": checks,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit(1)

    stage0 = pd.read_csv(package / "stage0_instrumentation_manifest.csv")
    unique = pd.read_csv(package / "phase_a_unique_image_manifest.csv")
    batch = pd.read_csv(package / "phase_a_seed_batch_manifest.csv")
    stage0_blind = pd.read_csv(package / "stage0_annotator_blinded_manifest.csv")
    phase_a_blind = pd.read_csv(package / "phase_a_annotator_blinded_manifest.csv")

    for filename, frame in [
        ("stage0_instrumentation_manifest.csv", stage0),
        ("phase_a_unique_image_manifest.csv", unique),
        ("phase_a_seed_batch_manifest.csv", batch),
    ]:
        missing_columns = sorted(IMAGE_MANIFEST_REQUIRED - set(frame.columns))
        add_check(
            checks,
            f"{filename}:required_columns",
            not missing_columns,
            missing_columns,
        )

    overlap = set(stage0["sample_id"].astype(str)) & set(unique["sample_id"].astype(str))
    add_check(checks, "stage0_phase_a_overlap_zero", not overlap, sorted(overlap))
    add_check(checks, "stage0_image_count_24", len(stage0) == 24, len(stage0))
    add_check(checks, "phase_a_unique_image_count_233", len(unique) == 233, len(unique))

    seeds = set(batch["acquisition_seed"].astype(int))
    add_check(checks, "phase_a_seeds_exactly_5_to_14", seeds == PRIMARY_SEEDS, sorted(seeds))
    batch_structure_errors = []
    for (seed, strategy), frame in batch.groupby(["acquisition_seed", "strategy"]):
        ranks = set(frame["query_rank"].astype(int))
        if len(frame) != 20 or ranks != QUERY_RANKS:
            batch_structure_errors.append(
                {"seed": int(seed), "strategy": strategy, "rows": len(frame), "ranks": sorted(ranks)}
            )
    observed_pairs = set(
        zip(batch["acquisition_seed"].astype(int), batch["strategy"].astype(str))
    )
    expected_pairs = {(seed, strategy) for seed in PRIMARY_SEEDS for strategy in STRATEGIES}
    add_check(
        checks,
        "strategy_seed_query_rank_complete",
        not batch_structure_errors and observed_pairs == expected_pairs,
        {"errors": batch_structure_errors, "pair_count": len(observed_pairs)},
    )

    duplicate_blind_ids = pd.concat(
        [
            stage0_blind[["blinded_task_id"]],
            phase_a_blind[["blinded_task_id"]],
        ],
        ignore_index=True,
    )["blinded_task_id"].duplicated().sum()
    add_check(
        checks,
        "duplicate_blinded_task_id_zero",
        int(duplicate_blind_ids) == 0,
        int(duplicate_blind_ids),
    )
    for filename, frame in [
        ("stage0_annotator_blinded_manifest.csv", stage0_blind),
        ("phase_a_annotator_blinded_manifest.csv", phase_a_blind),
    ]:
        leaked = sorted(BLINDED_FORBIDDEN & set(frame.columns))
        add_check(checks, f"{filename}:no_strategy_or_gt_fields", not leaked, leaked)

    path_frames = [stage0, unique]
    missing_paths = []
    for frame in path_frames:
        for value in frame["image_path"].astype(str):
            if not (ROOT / Path(value)).exists():
                missing_paths.append(value)
    add_check(checks, "all_sample_paths_exist", not missing_paths, missing_paths[:20])

    split_values = sorted(
        set(pd.concat([stage0["split"], unique["split"], batch["split"]]).astype(str))
    )
    non_acquisition_count = int(
        sum(
            (~frame["split"].astype(str).eq("acquisition")).sum()
            for frame in [stage0, unique, batch]
        )
    )
    add_check(
        checks,
        "official_final_split_sample_count_zero",
        non_acquisition_count == 0 and split_values == ["acquisition"],
        {"non_acquisition_rows": non_acquisition_count, "split_values": split_values},
    )

    manifest_hash_errors = []
    for expected, filename in parse_hash_ledger(package / "manifest_hashes.sha256"):
        path = package / filename
        if not path.exists() or sha256_file(path) != expected:
            manifest_hash_errors.append(filename)
    add_check(checks, "manifest_hashes_match", not manifest_hash_errors, manifest_hash_errors)

    source_hash_errors = []
    for expected, relative in parse_hash_ledger(package / "source_evidence_hashes.sha256"):
        path = ROOT / relative
        if not path.exists() or sha256_file(path) != expected:
            source_hash_errors.append(relative)
    add_check(
        checks,
        "source_evidence_hashes_match",
        not source_hash_errors,
        source_hash_errors,
    )

    event_schema = json.loads((package / "annotation_event_schema.json").read_text(encoding="utf-8"))
    event_required = set(event_schema.get("required", []))
    event_enum = set(
        event_schema.get("properties", {}).get("event_type", {}).get("enum", [])
    )
    add_check(
        checks,
        "event_schema_required_fields_complete",
        REQUIRED_EVENT_FIELDS <= event_required,
        sorted(REQUIRED_EVENT_FIELDS - event_required),
    )
    add_check(
        checks,
        "event_schema_event_enum_complete",
        event_enum == REQUIRED_EVENTS,
        {"missing": sorted(REQUIRED_EVENTS - event_enum), "extra": sorted(event_enum - REQUIRED_EVENTS)},
    )

    example_events = []
    example_errors = []
    for line_no, line in enumerate(
        (package / "annotation_event_examples.jsonl").read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            example_events.append(str(event.get("event_type")))
            absent = REQUIRED_EVENT_FIELDS - set(event)
            if absent:
                example_errors.append({"line": line_no, "missing": sorted(absent)})
            checksum = str(event.get("checksum", ""))
            canonical = dict(event)
            canonical.pop("checksum", None)
            calculated = hashlib.sha256(
                json.dumps(
                    canonical,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if checksum != calculated:
                example_errors.append(
                    {
                        "line": line_no,
                        "checksum_expected": calculated,
                        "checksum_observed": checksum,
                    }
                )
        except json.JSONDecodeError as error:
            example_errors.append({"line": line_no, "json_error": str(error)})
    add_check(
        checks,
        "event_examples_complete",
        set(example_events) == REQUIRED_EVENTS and not example_errors,
        {
            "missing_events": sorted(REQUIRED_EVENTS - set(example_events)),
            "errors": example_errors,
        },
    )

    quality_schema = json.loads(
        (package / "annotation_quality_metric_schema.json").read_text(encoding="utf-8")
    )
    quality_fields = set(quality_schema.get("properties", {}))
    add_check(
        checks,
        "quality_metric_schema_complete",
        REQUIRED_QUALITY_METRICS <= quality_fields,
        sorted(REQUIRED_QUALITY_METRICS - quality_fields),
    )

    protocol_version_occurrences = 0
    inconsistent_version_files = []
    for path in package.iterdir():
        if path.suffix.lower() not in {".md", ".json", ".jsonl", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8")
        if PROTOCOL_VERSION in text:
            protocol_version_occurrences += 1
        elif path.name in {
            "annotation_event_schema.json",
            "annotation_event_examples.jsonl",
            "statistical_analysis_config.template.yaml",
            "annotation_logger_requirements.md",
            "annotation_quality_protocol.md",
            "statistical_analysis_plan.md",
        }:
            inconsistent_version_files.append(path.name)
    add_check(
        checks,
        "protocol_version_consistent",
        not inconsistent_version_files and protocol_version_occurrences >= 6,
        {
            "occurrences_in_files": protocol_version_occurrences,
            "missing_from_core_files": inconsistent_version_files,
        },
    )

    overall = all(bool(item["passed"]) for item in checks)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "overall_pass": overall,
        "status": "READY_FOR_RESEARCHER_DECISIONS" if overall else "NOT_READY",
        "stage0_images": len(stage0),
        "phase_a_unique_images": len(unique),
        "acquisition_seed_count": len(seeds),
        "overlap_aware_primary_analysis_implementable": (
            overall
            and (package / "image_seed_membership.csv").exists()
            and (package / "acquisition_overlap_matrix.csv").exists()
        ),
        "official_final_split_rows": non_acquisition_count,
        "checks": checks,
    }
    if args.json_output:
        output = args.json_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()

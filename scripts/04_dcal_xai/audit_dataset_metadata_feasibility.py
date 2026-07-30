"""Audit whether existing datasets can support independent-pool validation.

This is an acquisition-only metadata audit. It reads frozen acquisition
manifests, image EXIF, and existing selection records. It never trains a model,
runs inference, extracts embeddings, re-runs a selector, screens false
negatives, or accesses a final/locked split.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import ExifTags, Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "runs"
    / "metadata_feasibility_audit_20260718"
    / "metadata_feasibility_main"
)
DEFAULT_DECISION = ROOT / "docs" / "metadata_feasibility_decision_20260718.md"

INITIAL_SIZE = 20
QUERY_SIZE = 20
MIN_POOL_SIZE = INITIAL_SIZE + QUERY_SIZE
MIN_INDEPENDENT_GROUPS = 20
MIN_MIXED_TARGET_GROUPS = 20
MIN_PREVALENCE_RANGE = 0.10
MIN_REPEATED_CATEGORIES = 2

INDEPENDENT_PROVENANCE = "genuine_source_group"
CAMERA_TIME_PROXY = "camera_datetime_proxy"
WEAK_TIME_PROXY = "edited_datetime_proxy"
FILENAME_PROXY = "filename_sequence_proxy"
CATEGORY_ONLY = "category_or_domain_only"
PROTOCOL_FIELD = "protocol_or_label_confounded"

PROHIBITED_PATH_TOKENS = ("final", "locked")


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    manifest: Path
    selections: Path
    target_kind: str
    category_column: str
    source_column: str | None
    image_name_column: str
    primary_strategy: str
    baseline: str = "GTFreeRandom"


SPECS = (
    DatasetSpec(
        dataset="GC10-DET",
        manifest=(
            ROOT
            / "runs/gc10_taxonomy_protocol/gc10_protocol_20260715/"
            "gc10_acquisition_pool_gt_audit.csv"
        ),
        selections=(
            ROOT
            / "runs/gc10_taxonomy_selection_audit/"
            "gc10_random_vs_dino_200seed_20260715/"
            "gc10_selection_records_posthoc.csv"
        ),
        target_kind="rare_classes_8_9_10",
        category_column="source_folder_label",
        source_column=None,
        image_name_column="filename",
        primary_strategy="FrozenDINOVisualDiversity",
    ),
    DatasetSpec(
        dataset="MPDD",
        manifest=(
            ROOT
            / "runs/mpdd_annotation_triage_protocol/mpdd_protocol_20260715/"
            "mpdd_acquisition_pool_gt_audit.csv"
        ),
        selections=(
            ROOT
            / "runs/mpdd_selection_only_audit/"
            "mpdd_hierarchical_dino_200seed_20260715/"
            "mpdd_selection_records_posthoc.csv"
        ),
        target_kind="anomaly_image",
        category_column="product_category",
        source_column="official_split",
        image_name_column="image_relative_path",
        primary_strategy="FrozenCategoryBalancedDINO",
    ),
    DatasetSpec(
        dataset="VisA",
        manifest=(
            ROOT
            / "runs/visa_annotation_triage_protocol/visa_protocol_v2_20260715/"
            "visa_acquisition_pool_gt_audit.csv"
        ),
        selections=(
            ROOT
            / "runs/visa_selection_only_audit/"
            "visa_random_vs_visual_200seed_20260715/"
            "visa_selection_records_posthoc.csv"
        ),
        target_kind="anomaly_image",
        category_column="category",
        source_column=None,
        image_name_column="image_name",
        primary_strategy="FrozenDINOVisualDiversity",
    ),
)


@dataclass(frozen=True)
class GroupDefinition:
    dataset: str
    grouping_name: str
    column: str
    provenance: str
    target_blind: bool
    documented_source_unit: bool
    eligible_for_independent_pool: bool
    notes: str


GROUP_DEFINITIONS = (
    GroupDefinition(
        "GC10-DET", "filename_sequence_group", "production_group_raw",
        FILENAME_PROXY, True, False, False,
        "Derived by removing the last underscore-delimited filename token; not documented as a production lot.",
    ),
    GroupDefinition(
        "GC10-DET", "exif_calendar_day", "exif_calendar_day",
        WEAK_TIME_PROXY, True, False, False,
        "EXIF contains ACDSee software and lacks camera make/model; treat as edit-time proxy.",
    ),
    GroupDefinition(
        "GC10-DET", "exif_hour", "exif_hour",
        WEAK_TIME_PROXY, True, False, False,
        "Hour bin from weak EXIF DateTime; descriptive sensitivity only.",
    ),
    GroupDefinition(
        "MPDD", "exif_calendar_day", "exif_calendar_day",
        CAMERA_TIME_PROXY, True, False, False,
        "Camera-origin DateTime proxy, but not a documented production lot ID.",
    ),
    GroupDefinition(
        "MPDD", "exif_hour", "exif_hour",
        CAMERA_TIME_PROXY, True, False, False,
        "Camera-origin hour bin; smaller bins are not independent lots.",
    ),
    GroupDefinition(
        "MPDD", "official_split", "official_split",
        PROTOCOL_FIELD, False, False, False,
        "Official train/test origin is strongly associated with anomaly status.",
    ),
    GroupDefinition(
        "VisA", "object_category", "category",
        CATEGORY_ONLY, False, False, False,
        "Object category is a domain/task label, not a production source or lot.",
    ),
    GroupDefinition(
        "VisA", "source_annotation_csv", "source_annotation_csv",
        CATEGORY_ONLY, False, False, False,
        "One annotation CSV per object category; not an independent capture unit.",
    ),
)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def path_is_prohibited(path: Path | str) -> bool:
    parts = [part.lower() for part in Path(str(path)).parts]
    return any(token in parts for token in PROHIBITED_PATH_TOKENS)


def require_safe_existing(path: Path) -> None:
    if path_is_prohibited(path):
        raise RuntimeError(f"Prohibited final/locked path: {path}")
    if not path.exists():
        raise FileNotFoundError(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entropy_hhi(values: Iterable[Any]) -> tuple[int, float, float, float]:
    series = pd.Series(list(values), dtype="object").dropna().astype(str)
    if series.empty:
        return 0, math.nan, math.nan, math.nan
    shares = series.value_counts() / len(series)
    return (
        int(len(shares)),
        float(-(shares * np.log(shares)).sum()),
        float(shares.max()),
        float((shares**2).sum()),
    )


def target_flag(spec: DatasetSpec, frame: pd.DataFrame) -> pd.Series:
    if spec.dataset == "GC10-DET":
        return frame["class_ids"].map(
            lambda value: bool(
                {int(item) for item in str(value).split("|") if item}
                & {8, 9, 10}
            )
        )
    return frame["is_anomaly"].astype(bool)


def validate_inputs() -> dict[str, dict[str, Any]]:
    required_manifest = {
        "sample_id", "image_path", "image_sha256", "protocol_split"
    }
    required_selection = {
        "acquisition_seed", "strategy", "sample_id"
    }
    report: dict[str, dict[str, Any]] = {}
    for spec in SPECS:
        require_safe_existing(spec.manifest)
        require_safe_existing(spec.selections)
        manifest = pd.read_csv(spec.manifest)
        selections = pd.read_csv(spec.selections)
        missing_manifest = required_manifest - set(manifest.columns)
        missing_selection = required_selection - set(selections.columns)
        if missing_manifest or missing_selection:
            raise RuntimeError(
                f"{spec.dataset}: missing manifest={sorted(missing_manifest)}, "
                f"selection={sorted(missing_selection)}"
            )
        if spec.category_column not in manifest.columns:
            raise RuntimeError(
                f"{spec.dataset}: missing category column {spec.category_column}"
            )
        if spec.source_column and spec.source_column not in manifest.columns:
            raise RuntimeError(
                f"{spec.dataset}: missing source column {spec.source_column}"
            )
        if not manifest["protocol_split"].eq("acquisition").all():
            values = sorted(manifest["protocol_split"].astype(str).unique())
            raise RuntimeError(
                f"{spec.dataset}: non-acquisition protocol split present: {values}"
            )
        image_paths = manifest["image_path"].map(Path)
        prohibited = image_paths.map(path_is_prohibited)
        if prohibited.any():
            raise RuntimeError(
                f"{spec.dataset}: prohibited image path in acquisition manifest"
            )
        missing_images = [str(path) for path in image_paths if not path.exists()]
        if missing_images:
            raise FileNotFoundError(
                f"{spec.dataset}: {len(missing_images)} images missing; first={missing_images[0]}"
            )
        unknown_ids = set(selections["sample_id"].astype(str)) - set(
            manifest["sample_id"].astype(str)
        )
        if unknown_ids:
            raise RuntimeError(
                f"{spec.dataset}: selection IDs outside acquisition manifest: "
                f"{len(unknown_ids)}"
            )
        report[spec.dataset] = {
            "manifest_rows": len(manifest),
            "selection_rows": len(selections),
            "seeds": int(selections["acquisition_seed"].nunique()),
            "strategies": sorted(selections["strategy"].astype(str).unique()),
            "manifest": relative(spec.manifest),
            "selections": relative(spec.selections),
        }
    return report


def extract_exif(spec: DatasetSpec, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, row in manifest.iterrows():
        image_path = Path(str(row["image_path"]))
        require_safe_existing(image_path)
        try:
            with Image.open(image_path) as image:
                exif = {
                    ExifTags.TAGS.get(tag, str(tag)): value
                    for tag, value in image.getexif().items()
                }
        except Exception as exc:  # pragma: no cover - recorded as audit data
            exif = {}
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = None
        datetime_kind = None
        datetime_raw = None
        for candidate in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
            if exif.get(candidate):
                datetime_kind = candidate
                datetime_raw = str(exif[candidate]).strip()
                break
        timestamp = pd.to_datetime(
            datetime_raw, format="%Y:%m:%d %H:%M:%S", errors="coerce"
        )
        rows.append({
            "dataset": spec.dataset,
            "sample_id": str(row["sample_id"]),
            "exif_tag_count": len(exif),
            "exif_datetime_kind": datetime_kind,
            "exif_datetime_raw": datetime_raw,
            "exif_timestamp": timestamp,
            "exif_calendar_day": (
                timestamp.strftime("%Y-%m-%d") if pd.notna(timestamp) else None
            ),
            "exif_hour": (
                timestamp.strftime("%Y-%m-%d %H:00:00")
                if pd.notna(timestamp) else None
            ),
            "exif_make": str(exif.get("Make", "")).strip() or None,
            "exif_model": str(exif.get("Model", "")).strip() or None,
            "exif_software": str(exif.get("Software", "")).strip() or None,
            "exif_error": error,
            "source_manifest": relative(spec.manifest),
            "source_row": int(index),
        })
    return pd.DataFrame(rows)


def inventory_rows(
    spec: DatasetSpec, manifest: pd.DataFrame, exif: pd.DataFrame
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        field: str,
        series: pd.Series,
        origin: str,
        provenance: str,
        target_blind: bool,
        usable_as_independent_group: bool,
        notes: str,
    ) -> None:
        present = series.notna() & series.astype(str).str.strip().ne("")
        rows.append({
            "dataset": spec.dataset,
            "field": field,
            "origin": origin,
            "provenance": provenance,
            "coverage_count": int(present.sum()),
            "total_images": len(series),
            "coverage_fraction": float(present.mean()),
            "unique_count": int(series[present].astype(str).nunique()),
            "target_blind": target_blind,
            "usable_as_independent_group": usable_as_independent_group,
            "notes": notes,
        })

    for field in (
        "exif_datetime_raw", "exif_make", "exif_model", "exif_software"
    ):
        add(
            field, exif[field], "embedded_image_exif",
            CAMERA_TIME_PROXY if spec.dataset == "MPDD" else WEAK_TIME_PROXY,
            True, False,
            "Embedded field; not documented as production lot metadata.",
        )
    if spec.dataset == "GC10-DET":
        add(
            "production_group_raw", manifest["production_group_raw"],
            "filename_derived", FILENAME_PROXY, True, False,
            "Prefix before final underscore token; grouping rule is local, not source-documented.",
        )
        add(
            "source_folder_label", manifest["source_folder_label"],
            "folder_label", PROTOCOL_FIELD, False, False,
            "Defect-category folder; forbidden as production source.",
        )
    elif spec.dataset == "MPDD":
        add(
            "product_category", manifest["product_category"],
            "folder_category", CATEGORY_ONLY, False, False,
            "Object category, not production lot.",
        )
        add(
            "official_split", manifest["official_split"],
            "folder_split", PROTOCOL_FIELD, False, False,
            "Train/test origin is associated with anomaly status.",
        )
    else:
        add(
            "category", manifest["category"],
            "annotation_category", CATEGORY_ONLY, False, False,
            "Object category, not production lot.",
        )
        add(
            "source_row", manifest["source_row"],
            "annotation_row_order", PROTOCOL_FIELD, True, False,
            "Annotation order cannot be promoted to acquisition time.",
        )
    return rows


def repeated_category_support(
    group_stats: pd.DataFrame, category_membership: pd.DataFrame
) -> tuple[int, int, int]:
    usable_ids = set(
        group_stats.loc[group_stats["pool_size_pass"], "group_id"].astype(str)
    )
    membership = category_membership[
        category_membership["group_id"].astype(str).isin(usable_ids)
    ].drop_duplicates(["group_id", "category"])
    groups_per_category = membership.groupby("category")["group_id"].nunique()
    if groups_per_category.empty:
        return 0, 0, 0
    return (
        int((groups_per_category >= 2).sum()),
        int(groups_per_category.min()),
        int(groups_per_category.max()),
    )


def group_feasibility(
    spec: DatasetSpec,
    enriched: pd.DataFrame,
    definition: GroupDefinition,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if definition.column not in enriched.columns:
        raise RuntimeError(
            f"{spec.dataset}/{definition.grouping_name}: missing {definition.column}"
        )
    valid = enriched[enriched[definition.column].notna()].copy()
    valid["group_id"] = valid[definition.column].astype(str)
    hash_group_counts = valid.groupby("image_sha256")["group_id"].nunique()
    cross_group_hashes = set(hash_group_counts[hash_group_counts > 1].index)
    rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, str]] = []
    for group_id, frame in valid.groupby("group_id", sort=True):
        category_count, category_entropy, max_category_share, category_hhi = (
            entropy_hhi(frame[spec.category_column])
        )
        if spec.source_column:
            source_count, source_entropy, max_source_share, source_hhi = (
                entropy_hhi(frame[spec.source_column])
            )
        else:
            source_count, source_entropy, max_source_share, source_hhi = (
                None, None, None, None
            )
        target_count = int(frame["target_flag"].sum())
        size = len(frame)
        duplicate_cross_group = int(
            frame["image_sha256"].isin(cross_group_hashes).sum()
        )
        rows.append({
            "dataset": spec.dataset,
            "grouping_name": definition.grouping_name,
            "group_id": str(group_id),
            "group_provenance": definition.provenance,
            "n_images": size,
            "target_count": target_count,
            "target_prevalence": target_count / size,
            "mixed_target": 0 < target_count < size,
            "pool_size_pass": size >= MIN_POOL_SIZE,
            "category_count": category_count,
            "category_entropy": category_entropy,
            "max_category_share": max_category_share,
            "category_hhi": category_hhi,
            "source_count": source_count,
            "source_entropy": source_entropy,
            "max_source_share": max_source_share,
            "source_hhi": source_hhi,
            "timestamp_start": frame["exif_timestamp"].min(),
            "timestamp_end": frame["exif_timestamp"].max(),
            "duplicate_hash_cross_group_count_within_group": duplicate_cross_group,
            "source_file": relative(spec.manifest),
        })
        membership_rows.extend(
            {
                "group_id": str(group_id),
                "category": str(category),
            }
            for category in frame[spec.category_column].dropna().unique()
        )
    group_stats = pd.DataFrame(rows)
    category_membership = pd.DataFrame(membership_rows)
    if group_stats.empty:
        usable = group_stats
        prevalence_range = math.nan
    else:
        usable = group_stats[group_stats["pool_size_pass"]]
        prevalence_range = (
            float(
                usable["target_prevalence"].max()
                - usable["target_prevalence"].min()
            )
            if not usable.empty else math.nan
        )
    repeated_categories, min_groups_per_category, max_groups_per_category = (
        repeated_category_support(group_stats, category_membership)
        if not category_membership.empty else (0, 0, 0)
    )
    mixed_usable = int(
        (group_stats["pool_size_pass"] & group_stats["mixed_target"]).sum()
    ) if not group_stats.empty else 0
    common_support_pass = repeated_categories >= MIN_REPEATED_CATEGORIES
    independent_gate = bool(
        definition.provenance == INDEPENDENT_PROVENANCE
        and definition.target_blind
        and definition.documented_source_unit
        and len(usable) >= MIN_INDEPENDENT_GROUPS
        and mixed_usable >= MIN_MIXED_TARGET_GROUPS
        and pd.notna(prevalence_range)
        and prevalence_range >= MIN_PREVALENCE_RANGE
        and common_support_pass
    )
    summary = {
        "dataset": spec.dataset,
        "grouping_name": definition.grouping_name,
        "group_column": definition.column,
        "group_provenance": definition.provenance,
        "target_blind": definition.target_blind,
        "documented_source_unit": definition.documented_source_unit,
        "eligible_for_independent_pool": definition.eligible_for_independent_pool,
        "total_images": len(enriched),
        "metadata_covered_images": len(valid),
        "metadata_coverage_fraction": len(valid) / len(enriched),
        "group_count": len(group_stats),
        "group_size_min": int(group_stats["n_images"].min()) if not group_stats.empty else None,
        "group_size_median": float(group_stats["n_images"].median()) if not group_stats.empty else None,
        "group_size_max": int(group_stats["n_images"].max()) if not group_stats.empty else None,
        "groups_n_ge_40": int(group_stats["pool_size_pass"].sum()) if not group_stats.empty else 0,
        "mixed_target_groups": int(group_stats["mixed_target"].sum()) if not group_stats.empty else 0,
        "mixed_target_groups_n_ge_40": mixed_usable,
        "usable_prevalence_min": float(usable["target_prevalence"].min()) if not usable.empty else None,
        "usable_prevalence_max": float(usable["target_prevalence"].max()) if not usable.empty else None,
        "usable_prevalence_range": prevalence_range,
        "categories_with_at_least_2_usable_groups": repeated_categories,
        "min_usable_groups_per_category": min_groups_per_category,
        "max_usable_groups_per_category": max_groups_per_category,
        "common_support_pass": common_support_pass,
        "independent_pool_gate_pass": independent_gate,
        "notes": definition.notes,
    }
    return summary, group_stats


def selection_concentration(
    spec: DatasetSpec,
    selections: pd.DataFrame,
    enriched: pd.DataFrame,
    definition: GroupDefinition,
) -> pd.DataFrame:
    mapping = enriched[["sample_id", definition.column, "target_flag"]].copy()
    mapping = mapping.rename(columns={definition.column: "group_id"})
    joined = selections.merge(
        mapping, on="sample_id", how="left", validate="many_to_one"
    )
    if joined["group_id"].isna().any():
        # Missing metadata is a legitimate outcome and must remain explicit.
        joined["group_id"] = joined["group_id"].fillna("__MISSING_METADATA__")
    rows: list[dict[str, Any]] = []
    for (seed, strategy), frame in joined.groupby(
        ["acquisition_seed", "strategy"], sort=True
    ):
        represented, entropy, max_share, hhi = entropy_hhi(frame["group_id"])
        rows.append({
            "dataset": spec.dataset,
            "grouping_name": definition.grouping_name,
            "group_provenance": definition.provenance,
            "acquisition_seed": int(seed),
            "strategy": str(strategy),
            "baseline": spec.baseline,
            "selected_count": len(frame),
            "represented_groups": represented,
            "group_entropy": entropy,
            "max_group_share": max_share,
            "group_hhi": hhi,
            "target_yield_posthoc": int(frame["target_flag"].sum()),
            "represented_groups_delta_vs_random": None,
            "group_entropy_delta_vs_random": None,
            "max_group_share_delta_vs_random": None,
            "group_hhi_delta_vs_random": None,
            "target_yield_delta_vs_random": None,
            "source_file": (
                f"{relative(spec.selections)}; {relative(spec.manifest)}"
            ),
        })
    output = pd.DataFrame(rows)
    baselines = output[output["strategy"].eq(spec.baseline)].set_index(
        "acquisition_seed"
    )
    if baselines.index.nunique() != output["acquisition_seed"].nunique():
        raise RuntimeError(
            f"{spec.dataset}: Random baseline missing for {definition.grouping_name}"
        )
    delta_map = {
        "represented_groups": "represented_groups_delta_vs_random",
        "group_entropy": "group_entropy_delta_vs_random",
        "max_group_share": "max_group_share_delta_vs_random",
        "group_hhi": "group_hhi_delta_vs_random",
        "target_yield_posthoc": "target_yield_delta_vs_random",
    }
    for index, row in output.iterrows():
        baseline = baselines.loc[row["acquisition_seed"]]
        for metric, delta_column in delta_map.items():
            output.at[index, delta_column] = row[metric] - baseline[metric]
    return output


def summarize_selection_concentration(seed_rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "represented_groups_delta_vs_random",
        "group_entropy_delta_vs_random",
        "max_group_share_delta_vs_random",
        "group_hhi_delta_vs_random",
        "target_yield_delta_vs_random",
    ]
    rows = []
    for keys, frame in seed_rows.groupby(
        ["dataset", "grouping_name", "group_provenance", "strategy", "baseline"],
        sort=True,
    ):
        dataset, grouping, provenance, strategy, baseline = keys
        row = {
            "dataset": dataset,
            "grouping_name": grouping,
            "group_provenance": provenance,
            "strategy": strategy,
            "baseline": baseline,
            "n_seeds": int(frame["acquisition_seed"].nunique()),
        }
        for metric in metrics:
            values = pd.to_numeric(frame[metric], errors="coerce")
            row[f"mean_{metric}"] = float(values.mean())
            row[f"median_{metric}"] = float(values.median())
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_decision(dataset: str, feasibility: pd.DataFrame) -> str:
    if feasibility["independent_pool_gate_pass"].any():
        return "INDEPENDENT_POOL_PROTOCOL_ELIGIBLE"
    if dataset == "MPDD":
        return "CAPTURE_SESSION_CONFOUND_AUDIT_ONLY"
    if dataset == "GC10-DET":
        return "FILENAME_SEQUENCE_SENSITIVITY_ONLY"
    return "CATEGORY_ROBUSTNESS_ONLY"


def best_grouping(dataset: str) -> str:
    return {
        "GC10-DET": "filename_sequence_group",
        "MPDD": "exif_calendar_day",
        "VisA": "object_category",
    }[dataset]


def write_decision(
    output_path: Path,
    inventory: pd.DataFrame,
    feasibility: pd.DataFrame,
    concentration_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Metadata feasibility decision",
        "",
        "**CURRENT_DATASETS_INDEPENDENT_POOL_NO_GO**",
        "",
        "This audit used acquisition manifests, embedded EXIF, and frozen selection records only. It did not train, infer, extract embeddings, re-run selectors, screen FN events, or access final/locked data.",
        "",
        "## Frozen gate",
        "",
        f"- genuine, source-documented, target-blind group ID",
        f"- at least {MIN_INDEPENDENT_GROUPS} groups with N >= {MIN_POOL_SIZE}",
        f"- at least {MIN_MIXED_TARGET_GROUPS} mixed-target groups with N >= {MIN_POOL_SIZE}",
        f"- usable prevalence range >= {MIN_PREVALENCE_RANGE:.2f}",
        f"- at least {MIN_REPEATED_CATEGORIES} categories with repeated usable groups",
        "- no category/source field promoted to a production lot",
        "",
        "## Dataset decisions",
        "",
        "| dataset | best available grouping | provenance | groups | groups N>=40 | mixed N>=40 | prevalence range | decision |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for spec in SPECS:
        grouping = best_grouping(spec.dataset)
        row = feasibility[
            feasibility["dataset"].eq(spec.dataset)
            & feasibility["grouping_name"].eq(grouping)
        ].iloc[0]
        prevalence_range = row["usable_prevalence_range"]
        prevalence_text = (
            "missing" if pd.isna(prevalence_range) else f"{prevalence_range:.6f}"
        )
        lines.append(
            f"| {spec.dataset} | {grouping} | {row['group_provenance']} | "
            f"{int(row['group_count'])} | {int(row['groups_n_ge_40'])} | "
            f"{int(row['mixed_target_groups_n_ge_40'])} | {prevalence_text} | "
            f"{dataset_decision(spec.dataset, feasibility[feasibility['dataset'].eq(spec.dataset)])} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- GC10-DET: the filename prefix is target-blind but locally derived and undocumented as a production lot. EXIF is an ACDSee edit-time proxy, not verified capture metadata.",
        "- MPDD: camera-origin EXIF supports a capture-session confound audit. It does not provide enough documented, category-overlapping production pools for confirmatory prevalence analysis.",
        "- VisA: neither EXIF nor a source field distinct from object category is available. Annotation order and numeric filenames remain invalid group IDs.",
        "",
        "## Allowed next use",
        "",
        "MPDD capture-day coverage/concentration may be reported as a post-hoc mechanism audit using frozen selection records. GC10 filename groups and VisA categories may be used only for descriptive sensitivity/robustness summaries.",
        "",
        "## Not authorized",
        "",
        "This result does not authorize independent-pool selector validation, prevalence-effect estimation, new selector implementation, detector training, FN screening, or final-test access.",
        "",
        "## Inventory summary",
        "",
        inventory.groupby("dataset").agg(
            fields=("field", "size"),
            fields_with_full_coverage=("coverage_fraction", lambda value: int((value == 1.0).sum())),
            independent_group_fields=("usable_as_independent_group", "sum"),
        ).reset_index().to_markdown(index=False),
        "",
        "## Frozen-selection session concentration summary",
        "",
        concentration_summary.to_markdown(index=False),
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(output_dir: Path, decision_path: Path, sources: list[Path]) -> None:
    generated = sorted(
        [
            path for path in output_dir.iterdir()
            if path.is_file() and path.name != "generated_file_manifest.csv"
        ]
        + [decision_path, Path(__file__).resolve()]
    )
    rows = []
    for path in generated:
        rows.append({
            "role": "generated",
            "path": relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    for path in sources:
        rows.append({
            "role": "source",
            "path": relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    pd.DataFrame(rows).to_csv(
        output_dir / "generated_file_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and schemas only; do not read EXIF or write outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing audit output directory and decision file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    decision_path = args.decision.expanduser().resolve()
    for path in (output_dir, decision_path):
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"Audit outputs must stay inside the project root: {path}"
            ) from exc
    if path_is_prohibited(output_dir) or path_is_prohibited(decision_path):
        raise RuntimeError("Output path may not contain final/locked segments")

    validation = validate_inputs()
    if args.dry_run:
        print("[DRY RUN] input paths and schemas validated")
        print(json.dumps(validation, indent=2, ensure_ascii=False))
        print("[TRAINING] False")
        print("[INFERENCE] False")
        print("[FINAL TEST USED] False")
        return

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new path or --overwrite."
        )
    if decision_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Decision file exists: {decision_path}. Use --overwrite or a new path."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_path.parent.mkdir(parents=True, exist_ok=True)

    inventory_all: list[dict[str, Any]] = []
    exif_all: list[pd.DataFrame] = []
    feasibility_all: list[dict[str, Any]] = []
    group_stats_all: list[pd.DataFrame] = []
    concentration_all: list[pd.DataFrame] = []
    source_paths: list[Path] = []

    for spec in SPECS:
        manifest = pd.read_csv(spec.manifest)
        selections = pd.read_csv(spec.selections)
        manifest["target_flag"] = target_flag(spec, manifest)
        exif = extract_exif(spec, manifest)
        enriched = manifest.merge(
            exif.drop(columns=["dataset", "source_manifest", "source_row"]),
            on="sample_id", how="left", validate="one_to_one",
        )
        inventory_all.extend(inventory_rows(spec, manifest, exif))
        exif_all.append(exif)
        source_paths.extend([spec.manifest, spec.selections])

        definitions = [
            definition
            for definition in GROUP_DEFINITIONS
            if definition.dataset == spec.dataset
        ]
        for definition in definitions:
            feasibility, group_stats = group_feasibility(
                spec, enriched, definition
            )
            feasibility_all.append(feasibility)
            group_stats_all.append(group_stats)
            concentration_all.append(
                selection_concentration(
                    spec, selections, enriched, definition
                )
            )

    inventory = pd.DataFrame(inventory_all)
    exif_output = pd.concat(exif_all, ignore_index=True)
    feasibility = pd.DataFrame(feasibility_all)
    group_stats = pd.concat(group_stats_all, ignore_index=True)
    concentration = pd.concat(concentration_all, ignore_index=True)
    concentration_summary = summarize_selection_concentration(concentration)

    inventory.to_csv(
        output_dir / "metadata_field_inventory.csv",
        index=False, encoding="utf-8-sig",
    )
    exif_output.to_csv(
        output_dir / "exif_metadata_audit.csv",
        index=False, encoding="utf-8-sig",
    )
    feasibility.to_csv(
        output_dir / "capture_group_feasibility.csv",
        index=False, encoding="utf-8-sig",
    )
    group_stats.to_csv(
        output_dir / "metadata_target_confounding.csv",
        index=False, encoding="utf-8-sig",
    )
    concentration.to_csv(
        output_dir / "selection_session_concentration.csv",
        index=False, encoding="utf-8-sig",
    )
    concentration_summary.to_csv(
        output_dir / "selection_session_concentration_summary.csv",
        index=False, encoding="utf-8-sig",
    )

    config = {
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": "audit-only,analysis-only,decision-only",
        "initial_size": INITIAL_SIZE,
        "query_size": QUERY_SIZE,
        "frozen_gate": {
            "min_pool_size": MIN_POOL_SIZE,
            "min_independent_groups": MIN_INDEPENDENT_GROUPS,
            "min_mixed_target_groups": MIN_MIXED_TARGET_GROUPS,
            "min_prevalence_range": MIN_PREVALENCE_RANGE,
            "min_repeated_categories": MIN_REPEATED_CATEGORIES,
            "required_provenance": INDEPENDENT_PROVENANCE,
        },
        "training": False,
        "inference": False,
        "embedding_extraction": False,
        "selector_execution": False,
        "fn_screen": False,
        "final_test_used": False,
        "input_validation": validation,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_decision(
        decision_path, inventory, feasibility, concentration_summary
    )
    log_lines = [
        "mode=audit-only,analysis-only,decision-only",
        "training=False",
        "inference=False",
        "embedding_extraction=False",
        "selector_execution=False",
        "fn_screen=False",
        "final_test_used=False",
        f"inventory_rows={len(inventory)}",
        f"exif_rows={len(exif_output)}",
        f"feasibility_rows={len(feasibility)}",
        f"group_stats_rows={len(group_stats)}",
        f"selection_concentration_rows={len(concentration)}",
        "overall_decision=CURRENT_DATASETS_INDEPENDENT_POOL_NO_GO",
    ]
    (output_dir / "audit_execution_log.txt").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )
    write_manifest(output_dir, decision_path, sorted(set(source_paths)))

    print(f"[DONE] {decision_path}")
    print("[DECISION] CURRENT_DATASETS_INDEPENDENT_POOL_NO_GO")
    print("[MPDD] CAPTURE_SESSION_CONFOUND_AUDIT_ONLY")
    print("[TRAINING] False")
    print("[INFERENCE] False")
    print("[FINAL TEST USED] False")


if __name__ == "__main__":
    main()

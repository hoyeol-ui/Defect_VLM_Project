"""Build a forward-only GC10 pilot manifest set from explicit frozen sources.

The locked final manifest is neither an input nor a discovered file. The strict
schema's ``test`` role is an unused acquisition audit holdout, not an official or
final evaluation split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from defect_al.data.identity import (
    compute_file_sha256,
    make_canonical_id,
    normalize_relative_path,
    validate_sha256,
)

DATASET_NAME = "GC10-DET"
ACQUISITION_SEEDS = (0, 1, 2, 3, 4)
STRATEGIES = ("GTFreeRandom", "FrozenDINOVisualDiversity")
INITIAL_SIZE = 20
QUERY_SIZE = 20
MANIFEST_FIELDS = (
    "dataset_name",
    "relative_path",
    "image_sha256",
    "canonical_id",
    "legacy_sample_id",
    "source_role",
    "hard_group_id",
    "is_frozen_selected",
)


def legacy_sample_id(folder: str, filename: str) -> str:
    digest = hashlib.sha256(f"gc10|{folder}|{filename}".encode("utf-8")).hexdigest()
    return "gc10_" + digest[:20]


def _require_file(directory: Path, filename: str) -> Path:
    path = directory / filename
    if not path.is_file():
        raise RuntimeError(f"Required explicit GC10 source file is missing: {path}")
    return path


def _load_safe_source_rows(
    source_path: Path,
    *,
    dataset_root: Path,
    role: str,
) -> pd.DataFrame:
    frame = pd.read_csv(source_path)
    required = {
        "sample_id",
        "source_folder_label",
        "filename",
        "image_sha256",
        "hard_group_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{source_path} is missing required columns: {sorted(missing)}")
    if frame.empty or frame["sample_id"].duplicated().any():
        raise RuntimeError(f"Source manifest is empty or has duplicate sample_id values: {source_path}")

    rows: list[dict[str, Any]] = []
    hash_mismatches = []
    id_mismatches = []
    for row in frame.itertuples(index=False):
        folder = str(row.source_folder_label)
        filename = str(row.filename)
        expected_legacy_id = legacy_sample_id(folder, filename)
        if str(row.sample_id) != expected_legacy_id:
            id_mismatches.append(
                {"stored": str(row.sample_id), "expected": expected_legacy_id, "folder": folder, "filename": filename}
            )
            continue
        relative_path = normalize_relative_path(f"{folder}/{filename}")
        image_path = dataset_root.joinpath(*relative_path.split("/"))
        actual_sha = compute_file_sha256(image_path)
        stored_sha = validate_sha256(str(row.image_sha256), "source image_sha256")
        if actual_sha != stored_sha:
            hash_mismatches.append(
                {"sample_id": str(row.sample_id), "stored": stored_sha, "actual": actual_sha, "path": str(image_path)}
            )
            continue
        rows.append(
            {
                "dataset_name": DATASET_NAME,
                "relative_path": relative_path,
                "image_sha256": actual_sha,
                "canonical_id": make_canonical_id(DATASET_NAME, relative_path, actual_sha),
                "legacy_sample_id": str(row.sample_id),
                "source_role": role,
                "hard_group_id": str(row.hard_group_id),
            }
        )
    if id_mismatches:
        raise RuntimeError(f"Legacy GC10 sample ID mismatch; first={id_mismatches[0]}")
    if hash_mismatches:
        raise RuntimeError(f"GC10 source image SHA-256 mismatch; first={hash_mismatches[0]}")
    output = pd.DataFrame(rows)
    if len(output) != len(frame):
        raise RuntimeError(f"Incomplete source verification: expected={len(frame)}, verified={len(output)}")
    return output


def _initial_ids(blind: pd.DataFrame, acquisition_seed: int) -> list[str]:
    indices = np.arange(len(blind), dtype=int)
    selected = pd.DataFrame({"idx": indices}).sample(
        n=INITIAL_SIZE,
        random_state=acquisition_seed + 999,
        replace=False,
    )["idx"].astype(int)
    return blind.iloc[selected.tolist()]["sample_id"].astype(str).tolist()


def _sample_id_from_legacy_cache_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    if not normalized:
        raise RuntimeError("Legacy training manifest contains an empty line.")
    return Path(normalized).stem


def _selection_lineage(
    *,
    blind_path: Path,
    selection_records_path: Path,
    training_manifest_dir: Path,
    source_by_id: pd.DataFrame,
) -> tuple[set[str], list[dict[str, Any]], dict[str, str]]:
    blind = pd.read_csv(blind_path).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if blind.columns.tolist() != ["sample_id", "image_sha256", "phash64"]:
        raise RuntimeError(f"Unexpected blind-pool columns: {blind.columns.tolist()}")
    records = pd.read_csv(selection_records_path)
    source_lookup = source_by_id.set_index("legacy_sample_id")
    selected_union: set[str] = set()
    parity_rows: list[dict[str, Any]] = []
    training_manifest_hashes: dict[str, str] = {}

    for acquisition_seed in ACQUISITION_SEEDS:
        initial = _initial_ids(blind, acquisition_seed)
        for strategy in STRATEGIES:
            query = records[
                (records["acquisition_seed"] == acquisition_seed)
                & records["strategy"].eq(strategy)
            ].sort_values("rank_in_query", kind="mergesort")
            query_ids = query["sample_id"].astype(str).tolist()
            if len(query_ids) != QUERY_SIZE:
                raise RuntimeError(
                    f"Expected {QUERY_SIZE} frozen query IDs for seed={acquisition_seed}, strategy={strategy}; got {len(query_ids)}"
                )
            expected_ids = initial + query_ids
            if len(expected_ids) != INITIAL_SIZE + QUERY_SIZE or len(set(expected_ids)) != len(expected_ids):
                raise RuntimeError(f"Frozen selection cardinality/overlap failure for seed={acquisition_seed}, strategy={strategy}")
            missing_ids = set(expected_ids) - set(source_lookup.index)
            if missing_ids:
                raise RuntimeError(f"Frozen selection references unknown acquisition IDs: {sorted(missing_ids)[:5]}")

            key = f"acq{acquisition_seed}_{strategy}"
            legacy_manifest = training_manifest_dir / f"{key}.txt"
            if not legacy_manifest.is_file():
                raise RuntimeError(f"Missing legacy training manifest: {legacy_manifest}")
            legacy_lines = legacy_manifest.read_text(encoding="utf-8-sig").splitlines()
            legacy_ids = [_sample_id_from_legacy_cache_path(value) for value in legacy_lines]
            order_equal = legacy_ids == expected_ids
            id_set_equal = set(legacy_ids) == set(expected_ids)
            legacy_cache_sha_values = []
            cache_hash_mismatches = []
            for line, sample_id in zip(legacy_lines, legacy_ids):
                cache_path = Path(line.strip())
                if not cache_path.is_file():
                    raise RuntimeError(f"Legacy cache image is missing: {cache_path}")
                cache_sha = compute_file_sha256(cache_path)
                expected_sha = str(source_lookup.loc[sample_id, "image_sha256"])
                legacy_cache_sha_values.append(cache_sha)
                if cache_sha != expected_sha:
                    cache_hash_mismatches.append(
                        {"sample_id": sample_id, "cache_sha": cache_sha, "source_sha": expected_sha}
                    )
            forward_sha_values = [str(source_lookup.loc[sample_id, "image_sha256"]) for sample_id in expected_ids]
            sha_sequence_equal = legacy_cache_sha_values == forward_sha_values
            sha_set_equal = set(legacy_cache_sha_values) == set(forward_sha_values)
            passed = order_equal and id_set_equal and sha_sequence_equal and sha_set_equal and not cache_hash_mismatches
            if not passed:
                raise RuntimeError(
                    f"Legacy/forward selection parity failed for {key}: order={order_equal}, "
                    f"id_set={id_set_equal}, sha_sequence={sha_sequence_equal}, sha_set={sha_set_equal}, "
                    f"cache_hash_mismatches={len(cache_hash_mismatches)}"
                )
            selected_union.update(expected_ids)
            training_manifest_hashes[key] = compute_file_sha256(legacy_manifest)
            parity_rows.append(
                {
                    "set_key": key,
                    "rows": len(expected_ids),
                    "legacy_id_order_equal": order_equal,
                    "legacy_id_set_equal": id_set_equal,
                    "image_sha_sequence_equal": sha_sequence_equal,
                    "image_sha_set_equal": sha_set_equal,
                    "cache_hash_mismatches": len(cache_hash_mismatches),
                    "status": "PASS",
                }
            )
    return selected_union, parity_rows, training_manifest_hashes


def _write_manifest(path: Path, rows: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows.loc[:, MANIFEST_FIELDS].to_dict("records"):
            writer.writerow(row)
    return compute_file_sha256(path)


def _config_relative_path(config_path: Path, target: Path) -> str:
    return Path(os.path.relpath(target, config_path.parent)).as_posix()


def _report_path(path: Path, project_root: Path) -> str:
    """Prefer a portable repository-relative path in persisted evidence."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--detector-run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    protocol = args.protocol_dir.expanduser().resolve(strict=True)
    selection = args.selection_dir.expanduser().resolve(strict=True)
    detector_run = args.detector_run_dir.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    manifest_dir = args.manifest_dir.expanduser().resolve()
    config_output = args.config_output.expanduser().resolve()
    report_output = args.report_output.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve(strict=True)

    acquisition_path = _require_file(protocol, "gc10_acquisition_pool_gt_audit.csv")
    development_path = _require_file(protocol, "gc10_development_eval.csv")
    blind_path = _require_file(protocol, "gc10_acquisition_pool_blind.csv")
    selection_records_path = _require_file(selection, "gc10_selection_records_posthoc.csv")
    training_manifest_dir = detector_run / "training_manifests"
    if not training_manifest_dir.is_dir():
        raise RuntimeError(f"Missing legacy training manifest directory: {training_manifest_dir}")

    acquisition = _load_safe_source_rows(
        acquisition_path,
        dataset_root=dataset_root,
        role="acquisition",
    )
    development = _load_safe_source_rows(
        development_path,
        dataset_root=dataset_root,
        role="development",
    )
    source_by_id = acquisition.set_index("legacy_sample_id", drop=False)
    selected_union, parity_rows, training_manifest_hashes = _selection_lineage(
        blind_path=blind_path,
        selection_records_path=selection_records_path,
        training_manifest_dir=training_manifest_dir,
        source_by_id=acquisition,
    )

    selected_groups = set(source_by_id.loc[sorted(selected_union), "hard_group_id"].astype(str))
    train = acquisition[acquisition["hard_group_id"].isin(selected_groups)].copy()
    audit_holdout = acquisition[~acquisition["hard_group_id"].isin(selected_groups)].copy()
    if train.empty or audit_holdout.empty or development.empty:
        raise RuntimeError("GC10 pilot split construction produced an empty split.")
    train["source_role"] = train["legacy_sample_id"].map(
        lambda value: "frozen_selected_union" if value in selected_union else "selected_hard_group_closure"
    )
    train["is_frozen_selected"] = train["legacy_sample_id"].isin(selected_union).map({True: "true", False: "false"})
    development["source_role"] = "development"
    development["is_frozen_selected"] = "false"
    audit_holdout["source_role"] = "unused_acquisition_audit_holdout"
    audit_holdout["is_frozen_selected"] = "false"

    split_frames = {
        "train": train.sort_values("canonical_id", kind="mergesort").reset_index(drop=True),
        "validation": development.sort_values("canonical_id", kind="mergesort").reset_index(drop=True),
        "test": audit_holdout.sort_values("canonical_id", kind="mergesort").reset_index(drop=True),
    }
    canonical_overlaps = {}
    sha_overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        canonical_overlaps[f"{left}/{right}"] = len(
            set(split_frames[left]["canonical_id"]) & set(split_frames[right]["canonical_id"])
        )
        sha_overlaps[f"{left}/{right}"] = len(
            set(split_frames[left]["image_sha256"]) & set(split_frames[right]["image_sha256"])
        )
    if any(canonical_overlaps.values()) or any(sha_overlaps.values()):
        raise RuntimeError(
            f"Constructed GC10 pilot has split leakage: canonical={canonical_overlaps}, sha={sha_overlaps}"
        )

    # Do not leave partially valid manifests/configs behind when split validation fails.
    manifest_paths = {split: manifest_dir / f"{split}.csv" for split in split_frames}
    manifest_hashes = {
        split: _write_manifest(manifest_paths[split], frame)
        for split, frame in split_frames.items()
    }

    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_lines = [f"run_id: {args.run_id}", "", "manifests:"]
    for split in ("train", "validation", "test"):
        config_lines.extend(
            [
                f"  {split}:",
                f"    path: {_config_relative_path(config_output, manifest_paths[split])}",
                f"    manifest_sha256: \"{manifest_hashes[split]}\"",
            ]
        )
    config_output.write_text("\n".join(config_lines) + "\n", encoding="utf-8")

    report = {
        "status": "PASS",
        "run_id": args.run_id,
        "builder_script_sha256": compute_file_sha256(Path(__file__).resolve()),
        "dataset_name": DATASET_NAME,
        "source_files": {
            "acquisition": {"path": _report_path(acquisition_path, project_root), "sha256": compute_file_sha256(acquisition_path)},
            "development": {"path": _report_path(development_path, project_root), "sha256": compute_file_sha256(development_path)},
            "blind_pool": {"path": _report_path(blind_path, project_root), "sha256": compute_file_sha256(blind_path)},
            "selection_records": {"path": _report_path(selection_records_path, project_root), "sha256": compute_file_sha256(selection_records_path)},
            "legacy_training_manifests": training_manifest_hashes,
        },
        "actual_source_image_hashes_verified": len(acquisition) + len(development),
        "source_image_hash_mismatches": 0,
        "legacy_sample_id_mismatches": 0,
        "frozen_selection_sets_checked": len(parity_rows),
        "frozen_selected_union_rows": len(selected_union),
        "selected_hard_group_closure_rows": len(train),
        "unused_acquisition_audit_holdout_rows": len(audit_holdout),
        "selection_parity": parity_rows,
        "selection_parity_all_passed": all(row["status"] == "PASS" for row in parity_rows),
        "manifests": {
            split: {
                "path": _report_path(manifest_paths[split], project_root),
                "rows": len(split_frames[split]),
                "unique_canonical_ids": split_frames[split]["canonical_id"].nunique(),
                "unique_image_sha256s": split_frames[split]["image_sha256"].nunique(),
                "manifest_sha256": manifest_hashes[split],
            }
            for split in ("train", "validation", "test")
        },
        "canonical_id_overlaps": canonical_overlaps,
        "raw_image_sha256_overlaps": sha_overlaps,
        "strict_config": {"path": _report_path(config_output, project_root), "sha256": compute_file_sha256(config_output)},
        "test_role_semantics": "unused acquisition audit holdout; not official/final evaluation data",
        "locked_final_manifest_referenced": False,
        "locked_final_images_opened": False,
        "gpu_used": False,
        "training_performed": False,
        "inference_performed": False,
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "verified_source_images": report["actual_source_image_hashes_verified"],
        "selection_sets": report["frozen_selection_sets_checked"],
        "manifests": {split: item["rows"] for split, item in report["manifests"].items()},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

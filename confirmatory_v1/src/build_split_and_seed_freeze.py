"""Build prospective split/seed manifests without training or evaluation.

The script reads raw public images/XML only to construct the split and aggregate
class-support audit. Selector-facing acquisition manifests contain opaque sample
IDs only. It never loads a detector, DINO model, development metric, or final
metric.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


PACKAGE_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_SRC))

from confirmatory_v1.protocol_core import budget_grid, manifest_sha256, validate_split_partitions  # noqa: E402


PROTOCOL_ID = "confirmatory-random-dino-v1"
SPLIT_SEED = 20260822
SPLIT_RATIOS = {"acquisition": Decimal("0.70"), "development": Decimal("0.10"), "final": Decimal("0.20")}
SPLIT_ORDER = ("acquisition", "development", "final")

GC10_CLASS_NAMES = {
    1: "punching_hole",
    2: "welding_line",
    3: "crescent_gap",
    4: "water_spot",
    5: "oil_spot",
    6: "silk_spot",
    7: "inclusion",
    8: "rolled_pit",
    9: "crease",
    10: "waist_folding",
}
NEU_CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}
NEU_NAME_TO_ID = {name: class_id for class_id, name in NEU_CLASS_NAMES.items()}


@dataclass(frozen=True)
class SourceRow:
    dataset: str
    image_path: Path
    annotation_path: Path | None
    raw_sha256: str
    pixel_sha256: str
    annotation_sha256: str | None
    class_ids: tuple[int, ...]
    instance_counts: tuple[tuple[int, int], ...]
    annotation_signature: str | None


@dataclass(frozen=True)
class Sample:
    dataset: str
    sample_id: str
    image_path: Path
    annotation_path: Path
    raw_sha256: str
    annotation_sha256: str
    class_ids: tuple[int, ...]
    instance_counts: tuple[tuple[int, int], ...]
    collapsed_alias_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_pixel_sha256(path: Path) -> str:
    """Hash canonical RGB pixels plus dimensions, independent of path/JPEG encoding."""

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(b"CONFIRMATORY_RGB_U8_V1\0")
    digest.update(struct.pack(">II", int(array.shape[1]), int(array.shape[0])))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def gc10_class_id(raw_name: str, image_path: Path) -> int | None:
    raw = raw_name.strip().lower()
    prefix = raw.split("_", 1)[0]
    if prefix.isdigit() and 1 <= int(prefix) <= 10:
        return int(prefix)
    if image_path.parent.name.isdigit() and 1 <= int(image_path.parent.name) <= 10:
        return int(image_path.parent.name)
    return None


def parse_annotation(dataset: str, annotation_path: Path, image_path: Path) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...], str]:
    objects: list[tuple[int, tuple[int, int, int, int]]] = []
    root = ET.parse(annotation_path).getroot()
    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip()
        if dataset == "GC10-DET":
            class_id = gc10_class_id(raw_name, image_path)
        else:
            class_id = NEU_NAME_TO_ID.get(raw_name.lower())
        box = obj.find("bndbox")
        if class_id is None or box is None:
            continue
        coords = tuple(int(float(box.findtext(name) or "0")) for name in ("xmin", "ymin", "xmax", "ymax"))
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        objects.append((class_id, coords))
    objects.sort()
    counts = Counter(class_id for class_id, _ in objects)
    signature = hashlib.sha256(
        json.dumps(objects, separators=(",", ":"), sort_keys=False).encode("utf-8")
    ).hexdigest()
    return tuple(sorted(counts)), tuple(sorted(counts.items())), signature


def discover_source_rows(dataset_root: Path, dataset: str) -> list[SourceRow]:
    if dataset == "GC10-DET":
        image_paths = sorted(path for class_id in range(1, 11) for path in (dataset_root / str(class_id)).glob("*.jpg"))
        annotation_root = dataset_root / "lable"
    else:
        image_paths = sorted((dataset_root / "IMAGES").glob("*.jpg"))
        annotation_root = dataset_root / "ANNOTATIONS"
    rows: list[SourceRow] = []
    for image_path in image_paths:
        annotation_path = annotation_root / f"{image_path.stem}.xml"
        raw_sha = file_sha256(image_path)
        pixel_sha = canonical_pixel_sha256(image_path)
        if not annotation_path.is_file():
            rows.append(SourceRow(dataset, image_path, None, raw_sha, pixel_sha, None, (), (), None))
            continue
        class_ids, instance_counts, signature = parse_annotation(dataset, annotation_path, image_path)
        rows.append(
            SourceRow(
                dataset,
                image_path,
                annotation_path,
                raw_sha,
                pixel_sha,
                file_sha256(annotation_path),
                class_ids,
                instance_counts,
                signature,
            )
        )
    return rows


def curate_unique_samples(rows: list[SourceRow]) -> tuple[list[Sample], list[dict[str, str]]]:
    by_pixel: dict[str, list[SourceRow]] = defaultdict(list)
    for row in rows:
        by_pixel[row.pixel_sha256].append(row)
    samples: list[Sample] = []
    exclusions: list[dict[str, str]] = []
    for pixel_sha, group in sorted(by_pixel.items()):
        if any(row.annotation_path is None for row in group):
            for row in group:
                exclusions.append(
                    {
                        "dataset": row.dataset,
                        "relative_image_path": row.image_path.as_posix(),  # relativized by caller
                        "raw_sha256": row.raw_sha256,
                        "pixel_sha256": pixel_sha,
                        "reason": "MISSING_XML",
                    }
                )
            continue
        if any(not row.class_ids for row in group):
            for row in group:
                exclusions.append(
                    {
                        "dataset": row.dataset,
                        "relative_image_path": row.image_path.as_posix(),
                        "raw_sha256": row.raw_sha256,
                        "pixel_sha256": pixel_sha,
                        "reason": "EMPTY_OR_INVALID_GT",
                    }
                )
            continue
        signatures = {row.annotation_signature for row in group}
        if len(signatures) != 1:
            for row in group:
                exclusions.append(
                    {
                        "dataset": row.dataset,
                        "relative_image_path": row.image_path.as_posix(),
                        "raw_sha256": row.raw_sha256,
                        "pixel_sha256": pixel_sha,
                        "reason": "CONFLICTING_GT_FOR_IDENTICAL_PIXELS",
                    }
                )
            continue
        # Raw-SHA ordering is class-neutral; path is used only as an unreachable final tie.
        representative = min(group, key=lambda row: (row.raw_sha256, str(row.image_path)))
        assert representative.annotation_path is not None
        assert representative.annotation_sha256 is not None
        samples.append(
            Sample(
                representative.dataset,
                pixel_sha,
                representative.image_path,
                representative.annotation_path,
                representative.raw_sha256,
                representative.annotation_sha256,
                representative.class_ids,
                representative.instance_counts,
                len(group) - 1,
            )
        )
        for alias in group:
            if alias is representative:
                continue
            exclusions.append(
                {
                    "dataset": alias.dataset,
                    "relative_image_path": alias.image_path.as_posix(),
                    "raw_sha256": alias.raw_sha256,
                    "pixel_sha256": pixel_sha,
                    "reason": "COLLAPSED_IDENTICAL_PIXELS_EQUIVALENT_GT",
                }
            )
    return sorted(samples, key=lambda sample: sample.sample_id), exclusions


def largest_remainder_targets(total: int) -> dict[str, int]:
    raw = {split: Decimal(total) * ratio for split, ratio in SPLIT_RATIOS.items()}
    targets = {split: int(value.to_integral_value(rounding=ROUND_FLOOR)) for split, value in raw.items()}
    remaining = total - sum(targets.values())
    ranked = sorted(
        SPLIT_ORDER,
        key=lambda split: (-(raw[split] - targets[split]), SPLIT_ORDER.index(split)),
    )
    for split in ranked[:remaining]:
        targets[split] += 1
    return targets


def stable_key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def iterative_multilabel_split(samples: list[Sample]) -> tuple[dict[str, str], dict[str, int]]:
    targets = largest_remainder_targets(len(samples))
    global_support = Counter(class_id for sample in samples for class_id in sample.class_ids)
    desired_remaining = {
        split: {class_id: float(SPLIT_RATIOS[split] * count) for class_id, count in global_support.items()}
        for split in SPLIT_ORDER
    }
    capacity = dict(targets)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    unassigned = set(sample_by_id)
    remaining_by_label = {
        class_id: {sample.sample_id for sample in samples if class_id in sample.class_ids}
        for class_id in global_support
    }
    assignment: dict[str, str] = {}

    def assign(sample_id: str, focus_label: int | None) -> None:
        sample = sample_by_id[sample_id]
        candidates = [split for split in SPLIT_ORDER if capacity[split] > 0]
        if focus_label is None:
            max_capacity = max(capacity[split] / targets[split] for split in candidates)
            candidates = [
                split for split in candidates if abs(capacity[split] / targets[split] - max_capacity) <= 1e-15
            ]
        else:
            max_need = max(desired_remaining[split][focus_label] for split in candidates)
            candidates = [
                split for split in candidates if abs(desired_remaining[split][focus_label] - max_need) <= 1e-15
            ]
            if len(candidates) > 1:
                max_capacity = max(capacity[split] / targets[split] for split in candidates)
                candidates = [
                    split
                    for split in candidates
                    if abs(capacity[split] / targets[split] - max_capacity) <= 1e-15
                ]
        chosen = min(
            candidates,
            key=lambda split: stable_key(PROTOCOL_ID, SPLIT_SEED, sample_id, split),
        )
        assignment[sample_id] = chosen
        capacity[chosen] -= 1
        unassigned.remove(sample_id)
        for class_id in sample.class_ids:
            desired_remaining[chosen][class_id] -= 1.0
            remaining_by_label[class_id].discard(sample_id)

    while unassigned:
        active_labels = [
            (len(ids & unassigned), class_id)
            for class_id, ids in remaining_by_label.items()
            if ids & unassigned
        ]
        if not active_labels:
            for sample_id in sorted(
                unassigned,
                key=lambda value: stable_key(PROTOCOL_ID, SPLIT_SEED, "unlabeled", value),
            ):
                assign(sample_id, None)
            break
        _, focus_label = min(active_labels)
        focus_samples = sorted(
            remaining_by_label[focus_label] & unassigned,
            key=lambda value: stable_key(PROTOCOL_ID, SPLIT_SEED, focus_label, value),
        )
        for sample_id in focus_samples:
            if sample_id in unassigned:
                assign(sample_id, focus_label)

    actual = Counter(assignment.values())
    if any(actual[split] != targets[split] for split in SPLIT_ORDER):
        raise RuntimeError(f"split cardinality mismatch: actual={dict(actual)}, expected={targets}")
    return assignment, targets


def seed32(namespace: str) -> int:
    value = int.from_bytes(hashlib.sha256(namespace.encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF
    return value or 1


def build_seed_rows(dataset: str) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for index in range(10):
        acquisition_seed = seed32(f"{PROTOCOL_ID}|{dataset}|acquisition|{index}")
        rows.append(
            {
                "dataset": dataset,
                "acquisition_index": index,
                "acquisition_seed": acquisition_seed,
                "training_seed_a": seed32(f"{PROTOCOL_ID}|{dataset}|training-a|{index}"),
                "training_seed_b": seed32(f"{PROTOCOL_ID}|{dataset}|training-b|{index}"),
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    all_samples: list[Sample] = []
    all_exclusions: list[dict[str, str]] = []
    raw_counts: dict[str, int] = {}
    for dataset in ("GC10-DET", "NEU-DET"):
        dataset_root = args.data_root / dataset
        rows = discover_source_rows(dataset_root, dataset)
        samples, exclusions = curate_unique_samples(rows)
        raw_counts[dataset] = len(rows)
        # Populate paths after curation without allowing them into selector manifests.
        for exclusion in exclusions:
            exclusion["relative_image_path"] = Path(exclusion["relative_image_path"]).relative_to(args.data_root).as_posix()
        all_samples.extend(samples)
        all_exclusions.extend(exclusions)

    if len({sample.sample_id for sample in all_samples}) != len(all_samples):
        raise RuntimeError("cross-dataset duplicate canonical pixel SHA detected")

    assignments: dict[str, str] = {}
    target_counts: dict[str, dict[str, int]] = {}
    for dataset in ("GC10-DET", "NEU-DET"):
        subset = [sample for sample in all_samples if sample.dataset == dataset]
        assignment, targets = iterative_multilabel_split(subset)
        assignments.update(assignment)
        target_counts[dataset] = targets

    partitions: dict[str, list[str]] = defaultdict(list)
    for sample in all_samples:
        partitions[f"{sample.dataset}:{assignments[sample.sample_id]}"].append(sample.sample_id)
    validate_split_partitions(partitions)

    assignment_rows = [
        {"dataset": sample.dataset, "sample_id": sample.sample_id, "split": assignments[sample.sample_id]}
        for sample in sorted(all_samples, key=lambda item: (item.dataset, item.sample_id))
    ]
    write_csv(output / "split_assignment.csv", ["dataset", "sample_id", "split"], assignment_rows)

    for dataset in ("GC10-DET", "NEU-DET"):
        slug = "gc10" if dataset == "GC10-DET" else "neu"
        for split in SPLIT_ORDER:
            ids = sorted(
                sample.sample_id
                for sample in all_samples
                if sample.dataset == dataset and assignments[sample.sample_id] == split
            )
            filename = (
                f"{slug}_acquisition_universe_blind.csv"
                if split == "acquisition"
                else f"{slug}_{split}_locked_manifest.csv"
            )
            write_csv(output / filename, ["sample_id"], ({"sample_id": sample_id} for sample_id in ids))

    private_rows = []
    for sample in sorted(all_samples, key=lambda item: (item.dataset, item.sample_id)):
        private_rows.append(
            {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "split": assignments[sample.sample_id],
                "relative_image_path": sample.image_path.relative_to(args.data_root).as_posix(),
                "relative_annotation_path": sample.annotation_path.relative_to(args.data_root).as_posix(),
                "raw_sha256": sample.raw_sha256,
                "canonical_pixel_sha256": sample.sample_id,
                "annotation_sha256": sample.annotation_sha256,
                "collapsed_alias_count": sample.collapsed_alias_count,
            }
        )
    write_csv(
        output / "private_source_registry.csv",
        [
            "dataset",
            "sample_id",
            "split",
            "relative_image_path",
            "relative_annotation_path",
            "raw_sha256",
            "canonical_pixel_sha256",
            "annotation_sha256",
            "collapsed_alias_count",
        ],
        private_rows,
    )
    write_csv(
        output / "source_exclusions.csv",
        ["dataset", "relative_image_path", "raw_sha256", "pixel_sha256", "reason"],
        sorted(all_exclusions, key=lambda row: (row["dataset"], row["reason"], row["raw_sha256"])),
    )

    support_rows = []
    for dataset in ("GC10-DET", "NEU-DET"):
        class_names = GC10_CLASS_NAMES if dataset == "GC10-DET" else NEU_CLASS_NAMES
        subset = [sample for sample in all_samples if sample.dataset == dataset]
        for split in ("curated_total", *SPLIT_ORDER):
            selected = subset if split == "curated_total" else [s for s in subset if assignments[s.sample_id] == split]
            image_support = Counter(class_id for sample in selected for class_id in sample.class_ids)
            instance_support = Counter()
            for sample in selected:
                instance_support.update(dict(sample.instance_counts))
            for class_id, class_name in class_names.items():
                support_rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "class_id": class_id,
                        "class_name": class_name,
                        "image_support": image_support[class_id],
                        "instance_support": instance_support[class_id],
                    }
                )
    write_csv(
        output / "class_support.csv",
        ["dataset", "split", "class_id", "class_name", "image_support", "instance_support"],
        support_rows,
    )

    seed_rows = build_seed_rows("GC10-DET") + build_seed_rows("NEU-DET")
    write_csv(
        output / "seed_manifest.csv",
        ["dataset", "acquisition_index", "acquisition_seed", "training_seed_a", "training_seed_b"],
        seed_rows,
    )

    generated = sorted(path for path in output.glob("*.csv"))
    hashes = {path.name: file_sha256(path) for path in generated}
    dataset_summary = {}
    for dataset in ("GC10-DET", "NEU-DET"):
        acquisition_n = target_counts[dataset]["acquisition"]
        dataset_summary[dataset] = {
            "raw_image_rows": raw_counts[dataset],
            "curated_unique_samples": sum(sample.dataset == dataset for sample in all_samples),
            "split_counts": target_counts[dataset],
            "budget_grid": list(budget_grid(acquisition_n)),
            "initial_count": budget_grid(acquisition_n)[0],
            "query_count": budget_grid(acquisition_n)[1] - budget_grid(acquisition_n)[0],
            "acquisition_manifest_sha256": manifest_sha256(partitions[f"{dataset}:acquisition"]),
        }
    summary = {
        "protocol_id": PROTOCOL_ID,
        "split_seed": SPLIT_SEED,
        "split_ratios": {key: str(value) for key, value in SPLIT_RATIOS.items()},
        "split_rounding": "global largest remainder; tie order acquisition, development, final",
        "sample_id": "sha256(CONFIRMATORY_RGB_U8_V1\\0 || width_be32 || height_be32 || RGB_U8_pixels)",
        "datasets": dataset_summary,
        "exclusion_reason_counts": dict(Counter(row["reason"] for row in all_exclusions)),
        "files_sha256": hashes,
        "final_evaluation_executed": False,
        "training_executed": False,
        "embedding_executed": False,
    }
    summary_path = output / "data_freeze_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger_hashes = {**hashes, summary_path.name: file_sha256(summary_path)}
    ledger_path = output / "data_freeze_hashes.sha256"
    ledger_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(ledger_hashes.items())),
        encoding="ascii",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

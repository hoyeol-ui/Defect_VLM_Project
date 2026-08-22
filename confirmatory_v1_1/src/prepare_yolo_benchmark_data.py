"""Materialize outcome-free YOLO image/label adapters for the six frozen benchmark cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from PIL import Image


WORKTREE = Path(__file__).resolve().parents[2]
ROOT = WORKTREE / "confirmatory_v1_1"
REGISTRY = WORKTREE / "confirmatory_v1" / "data_freeze" / "private_source_registry.csv"
DATA_ROOT = Path("C:/Users/user/Desktop/vlm/Defect_VLM_Project/data")
NAMES = {
    "GC10-DET": ["punching_hole", "welding_line", "crescent_gap", "water_spot", "oil_spot", "silk_spot", "inclusion", "rolled_pit", "crease", "waist_folding"],
    "NEU-DET": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
}
NEU_MAP = {name: index for index, name in enumerate(NAMES["NEU-DET"])}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_id(dataset: str, raw: str, image_path: Path) -> int | None:
    if dataset == "NEU-DET":
        return NEU_MAP.get(raw.strip().lower())
    prefix = raw.strip().split("_", 1)[0]
    source = int(prefix) if prefix.isdigit() else int(image_path.parent.name)
    return source - 1 if 1 <= source <= 10 else None


def yolo_label(dataset: str, image_path: Path, xml_path: Path) -> str:
    root = ET.parse(xml_path).getroot()
    with Image.open(image_path) as image:
        width, height = image.size
    lines: list[str] = []
    for obj in root.findall("object"):
        cid = class_id(dataset, obj.findtext("name") or "", image_path)
        box = obj.find("bndbox")
        if cid is None or box is None:
            continue
        values = [float(box.findtext(key) or "0") for key in ("xmin", "ymin", "xmax", "ymax")]
        xmin = max(0.0, min(values[0], width - 1))
        ymin = max(0.0, min(values[1], height - 1))
        xmax = max(0.0, min(values[2], width - 1))
        ymax = max(0.0, min(values[3], height - 1))
        if xmax <= xmin or ymax <= ymin:
            continue
        xc = ((xmin + xmax) / 2.0) / width
        yc = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        lines.append(f"{cid} {xc:.10f} {yc:.10f} {bw:.10f} {bh:.10f}")
    if not lines:
        raise RuntimeError(f"no valid YOLO objects: {xml_path}")
    return "\n".join(lines) + "\n"


def link_or_copy(source: Path, target: Path) -> str:
    if target.exists():
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def manifest_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(NAMES), required=True)
    args = parser.parse_args()
    dataset = args.dataset
    output = ROOT / "yolo_data_cache" / dataset
    images_dir = output / "images"
    labels_dir = output / "labels"
    lists_dir = output / "lists"
    for path in (images_dir, labels_dir, lists_dir):
        path.mkdir(parents=True, exist_ok=True)

    with REGISTRY.open("r", newline="", encoding="utf-8") as handle:
        registry = [row for row in csv.DictReader(handle) if row["dataset"] == dataset and row["split"] in {"acquisition", "development"}]
    lookup = {row["sample_id"]: row for row in registry}
    base = ROOT / "manifest_seals" / dataset / "a00"
    round_paths = {0: base / "shared_R0.txt", 3: base / "Random_R3.txt", 5: base / "Random_R5.txt"}
    round_ids = {round_idx: manifest_ids(path) for round_idx, path in round_paths.items()}
    dev_ids = sorted(row["sample_id"] for row in registry if row["split"] == "development")
    needed = sorted(set(dev_ids) | set(round_ids[5]))
    if set(needed) - set(lookup):
        raise RuntimeError("manifest ID missing from source registry")
    methods: dict[str, int] = {"hardlink": 0, "copy": 0, "existing": 0}
    for sample_id in needed:
        row = lookup[sample_id]
        image_path = DATA_ROOT / row["relative_image_path"]
        xml_path = DATA_ROOT / row["relative_annotation_path"]
        if not image_path.is_file() or not xml_path.is_file():
            raise FileNotFoundError(sample_id)
        target_image = images_dir / f"{sample_id}.jpg"
        methods[link_or_copy(image_path, target_image)] += 1
        target_label = labels_dir / f"{sample_id}.txt"
        if not target_label.exists():
            target_label.write_text(yolo_label(dataset, image_path, xml_path), encoding="ascii", newline="\n")
    for round_idx, ids in round_ids.items():
        train_lines = [str((images_dir / f"{sample_id}.jpg").resolve()).replace("\\", "/") for sample_id in ids]
        (lists_dir / f"train_R{round_idx}.txt").write_text(
            "\n".join(train_lines) + "\n", encoding="utf-8", newline="\n"
        )
    development_lines = [
        str((images_dir / f"{sample_id}.jpg").resolve()).replace("\\", "/") for sample_id in dev_ids
    ]
    (lists_dir / "development.txt").write_text(
        "\n".join(development_lines) + "\n", encoding="utf-8", newline="\n"
    )
    for round_idx in round_paths:
        yaml_path = output / f"dataset_R{round_idx}.yaml"
        names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(NAMES[dataset]))
        yaml_path.write_text(f"path: {str(output.resolve()).replace(chr(92), '/')}\ntrain: lists/train_R{round_idx}.txt\nval: lists/development.txt\nnames:\n{names_yaml}\n", encoding="utf-8", newline="\n")
    summary = {
        "dataset": dataset,
        "status": "PASS",
        "train_counts": {str(key): len(value) for key, value in round_ids.items()},
        "development_count": len(dev_ids),
        "materialized_unique_images": len(needed),
        "materialization_methods": methods,
        "final_rows_accessed": 0,
        "yaml_sha256": {f"R{round_idx}": sha(output / f"dataset_R{round_idx}.yaml") for round_idx in round_paths},
        "train_list_sha256": {f"R{round_idx}": sha(lists_dir / f"train_R{round_idx}.txt") for round_idx in round_paths},
        "development_list_sha256": sha(lists_dir / "development.txt"),
    }
    (output / "data_adapter_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PHASE2B_OPERATIONAL_RESULT " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

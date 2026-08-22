"""Create one shared outcome-free YOLO adapter and 220 frozen state YAMLs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[3]
ROOT = WORKTREE / "confirmatory_v1_1"
FULL = ROOT / "full_run"
sys.path.insert(0, str(ROOT / "src"))

from prepare_yolo_benchmark_data import NAMES, yolo_label  # noqa: E402


DATA_ROOT = Path("C:/Users/user/Desktop/vlm/Defect_VLM_Project/data")
REGISTRY = WORKTREE / "confirmatory_v1" / "data_freeze" / "private_source_registry.csv"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, target: Path) -> str:
    if target.exists():
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def read_plan() -> list[dict[str, object]]:
    return [json.loads(line) for line in (FULL / "planned_cells.jsonl").read_text(encoding="utf-8").splitlines() if line]


def state_id(row: dict[str, object]) -> str:
    return f"a{int(row['acquisition_index']):02d}_{row['strategy_token']}_R{int(row['round'])}"


def main() -> None:
    plan = read_plan()
    if len(plan) != 440:
        raise RuntimeError("planned registry count mismatch")
    with REGISTRY.open("r", newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    summary: dict[str, object] = {"status": "PASS", "datasets": {}, "final_rows_accessed": 0}
    for dataset in ("GC10-DET", "NEU-DET"):
        output = FULL / "yolo_data_cache" / dataset
        images = output / "images"
        labels = output / "labels"
        lists = output / "lists"
        states = output / "states"
        for directory in (images, labels, lists, states):
            directory.mkdir(parents=True, exist_ok=True)
        rows = [row for row in source_rows if row["dataset"] == dataset and row["split"] in {"acquisition", "development"}]
        if any(row["split"] == "final" for row in rows):
            raise RuntimeError("final registry row crossed the adapter boundary")
        acquisition_rows = [row for row in rows if row["split"] == "acquisition"]
        development_rows = [row for row in rows if row["split"] == "development"]
        expected_acq = 1605 if dataset == "GC10-DET" else 1259
        expected_dev = 229 if dataset == "GC10-DET" else 180
        if len(acquisition_rows) != expected_acq or len(development_rows) != expected_dev:
            raise RuntimeError(f"source partition count mismatch for {dataset}")
        methods = {"hardlink": 0, "copy": 0, "existing": 0}
        for row in acquisition_rows + development_rows:
            sample_id = row["sample_id"]
            image_source = DATA_ROOT / row["relative_image_path"]
            xml_source = DATA_ROOT / row["relative_annotation_path"]
            if not image_source.is_file() or not xml_source.is_file():
                raise FileNotFoundError(sample_id)
            target_image = images / f"{sample_id}.jpg"
            methods[link_or_copy(image_source, target_image)] += 1
            target_label = labels / f"{sample_id}.txt"
            if not target_label.exists():
                target_label.write_text(yolo_label(dataset, image_source, xml_source), encoding="ascii", newline="\n")
        development_ids = sorted(row["sample_id"] for row in development_rows)
        development_list = lists / "development.txt"
        development_list.write_text("\n".join(str((images / f"{sample_id}.jpg").resolve()).replace("\\", "/") for sample_id in development_ids) + "\n", encoding="utf-8", newline="\n")
        unique_states: dict[str, dict[str, object]] = {}
        for row in (row for row in plan if row["dataset"] == dataset):
            unique_states.setdefault(state_id(row), row)
        if len(unique_states) != 110:
            raise RuntimeError(f"unique state count mismatch for {dataset}")
        state_hashes: dict[str, dict[str, object]] = {}
        names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(NAMES[dataset]))
        for key, row in sorted(unique_states.items()):
            manifest = Path(str(row["manifest_path"]))
            ids = [line.strip() for line in manifest.read_text(encoding="ascii").splitlines() if line.strip()]
            list_path = lists / f"{key}.txt"
            list_path.write_text("\n".join(str((images / f"{sample_id}.jpg").resolve()).replace("\\", "/") for sample_id in ids) + "\n", encoding="utf-8", newline="\n")
            yaml_path = states / f"{key}.yaml"
            yaml_path.write_text(
                f"path: {str(output.resolve()).replace(chr(92), '/')}\n"
                f"train: lists/{key}.txt\n"
                "val: lists/development.txt\n"
                f"names:\n{names_yaml}\n",
                encoding="utf-8",
                newline="\n",
            )
            if len(ids) != int(row["labeled_count"]):
                raise RuntimeError(f"state count mismatch: {key}")
            state_hashes[key] = {"list_sha256": sha(list_path), "yaml_sha256": sha(yaml_path), "count": len(ids)}
        dataset_audit = {
            "acquisition_count": len(acquisition_rows),
            "development_count": len(development_rows),
            "final_count_accessed": 0,
            "unique_states": len(unique_states),
            "materialization": methods,
            "development_list_sha256": sha(development_list),
            "state_hashes": state_hashes,
        }
        (output / "adapter_audit.json").write_text(json.dumps(dataset_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["datasets"][dataset] = {
            "acquisition": len(acquisition_rows),
            "development": len(development_rows),
            "unique_states": len(unique_states),
            "materialization": methods,
            "adapter_audit_sha256": sha(output / "adapter_audit.json"),
        }
    summary["total_unique_states"] = 220
    (FULL / "data_adapter_seal.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

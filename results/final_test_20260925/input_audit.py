"""Frozen-input verification and isolated dev/test evaluation views. No inference."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import struct
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PHASE1 = ROOT / "output/dataset_phase1_v2_20260919"
PHASE1B = ROOT / "output/dataset_phase1b_v2_20260919"
PACKAGE = ROOT / "output/final_main_runpod_execution_v2_20260919"
RAW = ROOT / "Defect_VLM_Project/data"
METADATA = ROOT / "output/final_main_results_retrieval_20260922/metadata_verified/final_main_v2_20260922"
RECOVERY = ROOT / "final_test_recovery/final_main_runpod_execution_v2_20260919/state_attempts/ANNOTATION_PATCH_v2_0_2/runs"
RUNS = METADATA / "state_attempts/ANNOTATION_PATCH_v2_0_2/runs"
INTEGRITY = METADATA / "state_attempts/ANNOTATION_PATCH_v2_0_2/gates/final_integrity.json"
EXPECTED = {"GC10-DET": (1604, 343, 344), "NEU-DET": (1258, 270, 270)}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def exclusive_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def check(cond, message):
    if not cond:
        raise RuntimeError(message)


def parsed_boxes(xml: Path, dataset: str, classes: list[str], size: tuple[int, int]):
    root = ET.parse(xml).getroot()
    w, h = size
    check((int(root.findtext("size/width", "0")), int(root.findtext("size/height", "0"))) == size,
          f"XML size mismatch: {xml}")
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().lower()
        if dataset == "GC10-DET":
            match = re.fullmatch(r"(10|[1-9])_(.+)", name)
            check(match is not None, f"GC10 class mapping: {xml}: {name}")
            name = classes[int(match.group(1)) - 1]
        check(name in classes, f"Unknown class: {xml}: {name}")
        x0, y0, x1, y1 = [float(obj.findtext("bndbox/" + k, "nan"))
                          for k in ("xmin", "ymin", "xmax", "ymax")]
        check(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, f"Invalid bbox: {xml}")
        boxes.append([name, x0, y0, x1, y1])
    boxes.sort()
    check(boxes, f"Empty XML annotation: {xml}")
    return boxes


def label_text(boxes, classes, w, h):
    lines = []
    for name, x0, y0, x1, y1 in boxes:
        lines.append(f"{classes.index(name)} {(x0+x1)/(2*w):.12f} {(y0+y1)/(2*h):.12f} "
                     f"{(x1-x0)/w:.12f} {(y1-y0)/h:.12f}")
    # The sealed annotation patch keeps the first exact row only.
    return "\n".join(dict.fromkeys(lines)) + "\n"


def stage_file(source: Path, target: Path, expected_sha: str):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        check(sha(target) == expected_sha, f"Existing output differs: {target}")
    else:
        shutil.copyfile(source, target)
        check(sha(target) == expected_sha, f"Copied output differs: {target}")


def main():
    config = read_json(METADATA / "config/protocol.json")
    check(config["design_id"] == "metal_al_final_v2_20260919", "Design mismatch")
    check(config["seeds"] == list(range(2026091901, 2026091906)), "Seed mismatch")
    check(config["learner_seed"] == 2026091900, "Learner seed mismatch")
    integrity = read_json(INTEGRITY)
    checks = integrity["evidence"]["checks"]
    check(integrity["status"] == "PASS" and len(checks) == 162, "Scientific integrity mismatch")
    actual_files = list(RECOVERY.rglob("best.pt"))
    check(len(actual_files) == 163, "Recovered best.pt count mismatch")
    run_to_file = {}
    for path in actual_files:
        rel = path.relative_to(RECOVERY).parts
        check(len(rel) == 4 and rel[1:] == ("train", "weights", "best.pt"), f"Unexpected checkpoint layout: {path}")
        check(rel[0] not in run_to_file, f"Duplicate run checkpoint: {rel[0]}")
        run_to_file[rel[0]] = path
    check("TECHNICAL_NEU-DET" in run_to_file and set(run_to_file) - {"TECHNICAL_NEU-DET"} == set(checks),
          "Scientific checkpoint identity mismatch")
    ckpt_sha = {}
    for run_id in sorted(checks):
        actual = sha(run_to_file[run_id])
        check(actual == checks[run_id]["best_sha"], f"Checkpoint SHA mismatch: {run_id}")
        ckpt_sha[run_id] = actual
    check(len(set(ckpt_sha.values())) == 162, "Duplicated scientific checkpoint SHA")

    all_records = []
    summary = {}
    for dataset, (train_n, dev_n, test_n) in EXPECTED.items():
        classes = config["classes"][dataset]
        manifest = PHASE1B / dataset / "SPLIT_MEMBERSHIP.csv"
        check(sha(manifest) == config["datasets"][dataset]["split_sha"], f"Split SHA mismatch: {dataset}")
        members = read_csv(manifest)
        counts = Counter(r["split"] for r in members)
        check(counts == {"train": train_n, "dev": dev_n, "test": test_n}, f"Split counts mismatch: {dataset}")
        for key in ("sample_id", "representative_path", "rgb_sha256"):
            check(len({r[key] for r in members}) == len(members), f"Cross-split duplicate {key}: {dataset}")
        parts = defaultdict(set)
        for r in members:
            parts[r["component_id"]].add(r["split"])
        check(all(len(p) == 1 for p in parts.values()), f"Component crosses split: {dataset}")
        canonical = {r["sample_id"]: r for r in read_csv(PHASE1B / dataset / "CANONICAL_INDEX.csv")}
        check(set(canonical) == {r["sample_id"] for r in members}, f"Canonical IDs mismatch: {dataset}")
        raw_index = {r["raw_relative_path"]: r for r in read_csv(PHASE1 / dataset / "RAW_IDENTITY.csv")}
        file_groups = defaultdict(set)
        for r in members:
            c = canonical[r["sample_id"]]
            raw = raw_index[r["representative_path"]]
            check(c["representative_path"] == r["representative_path"] and
                  c["rgb_sha256"] == r["rgb_sha256"] == r["sample_id"] and
                  raw["rgb_sha256"] == r["sample_id"] and
                  raw["annotation_signature"] == c["annotation_signature"],
                  f"Canonical/raw identity mismatch: {dataset} {r['sample_id']}")
            file_groups[raw["file_sha256"]].add(r["split"])
        check(all(len(p) == 1 for p in file_groups.values()), f"Cross-split file SHA duplicate: {dataset}")

        test_count = 0
        for r in sorted((r for r in members if r["split"] == "test"), key=lambda x: x["sample_id"]):
            c = canonical[r["sample_id"]]
            raw = raw_index[r["representative_path"]]
            image = (RAW / r["representative_path"]).resolve()
            xml = (RAW / raw["annotation_relative_path"]).resolve()
            check(image.is_relative_to(RAW.resolve()) and xml.is_relative_to(RAW.resolve()), "Source path escapes dataset root")
            check(image.is_file() and xml.is_file(), f"Missing test source: {r['sample_id']}")
            check(sha(image) == raw["file_sha256"] and sha(xml) == raw["annotation_sha256"],
                  f"Source image/XML SHA mismatch: {r['sample_id']}")
            with Image.open(image) as im:
                w, h = im.size
                rgb = im.convert("RGB").tobytes()
            rgb_sha = hashlib.sha256(b"PHASE1_RGB_UINT8_HWC\0" + struct.pack("<II", w, h) + rgb).hexdigest()
            check(rgb_sha == r["sample_id"] and (w, h) == (int(c["width"]), int(c["height"])),
                  f"Decoded RGB identity mismatch: {r['sample_id']}")
            boxes = parsed_boxes(xml, dataset, classes, (w, h))
            check(boxes == json.loads(c["boxes_json"]), f"XML/canonical boxes mismatch: {r['sample_id']}")
            text = label_text(boxes, classes, w, h)
            view = OUT / "datasets" / dataset
            image_out = view / "images/test" / (r["sample_id"] + image.suffix.lower())
            label_out = view / "labels/test" / (r["sample_id"] + ".txt")
            stage_file(image, image_out, raw["file_sha256"])
            label_out.parent.mkdir(parents=True, exist_ok=True)
            if label_out.exists():
                check(label_out.read_text(encoding="ascii") == text, f"Existing test label differs: {label_out}")
            else:
                with label_out.open("x", encoding="ascii", newline="\n") as f:
                    f.write(text)
            all_records.append({"dataset": dataset, "split": "test", "sample_id": r["sample_id"],
                                "source_image": str(image), "source_xml": str(xml),
                                "image_sha256": raw["file_sha256"], "xml_sha256": raw["annotation_sha256"],
                                "output_image": str(image_out), "output_label": str(label_out),
                                "label_sha256": sha(label_out), "bbox_rows": len(text.splitlines())})
            test_count += 1
        check(test_count == test_n, f"Test source count mismatch: {dataset}")
        check(len(list((OUT / "datasets" / dataset / "images/test").iterdir())) == test_n,
              f"Staged test image count mismatch: {dataset}")
        check(len(list((OUT / "datasets" / dataset / "labels/test").iterdir())) == test_n,
              f"Staged test annotation count mismatch: {dataset}")
        summary[dataset] = {"train_ids": train_n, "dev_ids": dev_n, "test_ids": test_n,
                            "actual_test_images": test_count, "actual_test_annotations": test_count,
                            "cross_split_id_rgb_file_component_duplicates": 0,
                            "class_names": classes, "split_sha256": sha(manifest)}
    exclusive_json(OUT / "preflight/test_dataset_inventory.json", all_records)
    exclusive_json(OUT / "preflight/preflight.json", {"status": "PASS", "design": config["design_id"],
                   "attempt": "ANNOTATION_PATCH_v2_0_2", "scientific_checkpoint_sha_verified": 162,
                   "technical_excluded": "TECHNICAL_NEU-DET", "datasets": summary})
    print(json.dumps({"status": "PASS", "datasets": summary, "checkpoint_sha_verified": 162}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

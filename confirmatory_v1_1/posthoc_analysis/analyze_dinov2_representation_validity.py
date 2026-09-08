"""Post-hoc DINOv2 representation audit for the sealed acquisition pools.

This script never extracts features, trains a model, changes acquisition, or reads
locked-test labels. It validates the canonical cache and acquisition-only YOLO
labels before writing descriptive representation-audit tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


HERE = Path(__file__).resolve()
RELEASE_ROOT = HERE.parents[2]
DEFAULT_SOURCE_ROOT = Path("C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1")
DEFAULT_RAW_DATA_ROOT = Path("C:/Users/user/Desktop/vlm/Defect_VLM_Project/data")
DEFAULT_OUTPUT = HERE.parent / "results" / "dinov2_representation_validity_20260907"

EXPECTED = {
    "GC10-DET": {"count": 1605, "classes": 10},
    "NEU-DET": {"count": 1259, "classes": 6},
}
NAMES = {
    "GC10-DET": [
        "punching_hole",
        "welding_line",
        "crescent_gap",
        "water_spot",
        "oil_spot",
        "silk_spot",
        "inclusion",
        "rolled_pit",
        "crease",
        "waist_folding",
    ],
    "NEU-DET": [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ],
}
EXPECTED_DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
EXPECTED_DINO_MODEL_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
EXPECTED_PROTOCOL_SHA256 = "6836f452d22c3de26dfa2680c45d0b5cb8f925b1a9274789c4ef81a1770f0b10"
MODEL_FILE = Path(
    "C:/Users/user/.cache/huggingface/hub/models--facebook--dinov2-small/"
    f"snapshots/{EXPECTED_DINO_REVISION}/model.safetensors"
)


class IntegrityError(RuntimeError):
    """Raised before outputs exist when a STOP condition is met."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def canonical_json(path: Path) -> str:
    return json.dumps(json.loads(path.read_text(encoding="utf-8")), sort_keys=True)


def validate_release_against_source(source_root: Path) -> dict[str, object]:
    pairs = [
        "confirmatory_v1/confirmatory_protocol_v1.yaml",
        "confirmatory_v1/data_freeze/seed_manifest.csv",
        "confirmatory_v1/data_freeze/gc10_development_locked_manifest.csv",
        "confirmatory_v1/data_freeze/gc10_final_locked_manifest.csv",
        "confirmatory_v1/data_freeze/neu_development_locked_manifest.csv",
        "confirmatory_v1/data_freeze/neu_final_locked_manifest.csv",
        "confirmatory_v1_1/confirmatory_protocol_v1_1.yaml",
        "confirmatory_v1_1/cache_seals/CANONICAL_CACHE_SEAL.json",
        "confirmatory_v1_1/full_run/full_run_config.json",
        "confirmatory_v1_1/full_run/runtime/FULL_TRAINING_INTEGRITY_AUDIT.json",
    ]
    hashes: dict[str, object] = {}
    for relative in pairs:
        release_path = RELEASE_ROOT / relative
        source_path = source_root / relative
        require(release_path.is_file() and source_path.is_file(), f"missing release/source file: {relative}")
        release_hash = sha256(release_path)
        source_hash = sha256(source_path)
        require(release_hash == source_hash, f"release/source scientific record mismatch: {relative}")
        hashes[relative] = release_hash

    # The packaged JSONL has CRLF while the run copy has LF. Compare its 440
    # scientific records line by line, and compare the registry seal semantically.
    relative = "confirmatory_v1_1/full_run/planned_cells.jsonl"
    release_lines = (RELEASE_ROOT / relative).read_text(encoding="utf-8").splitlines()
    source_lines = (source_root / relative).read_text(encoding="utf-8").splitlines()
    require(len(release_lines) == 440 and release_lines == source_lines, "planned 440 registry content mismatch")
    require(
        canonical_json(RELEASE_ROOT / "confirmatory_v1_1/full_run/planned_registry_seal.json")
        == canonical_json(source_root / "confirmatory_v1_1/full_run/planned_registry_seal.json"),
        "planned registry seal semantic mismatch",
    )
    hashes[relative] = {
        "scientific_records_exact": True,
        "line_count": 440,
        "byte_hash_difference_reason": "release CRLF; run copy LF",
    }
    return hashes


def parse_labels(label_dir: Path, ids: np.ndarray, class_count: int) -> list[frozenset[int]]:
    label_sets: list[frozenset[int]] = []
    expected_files = {f"{sample_id}.txt" for sample_id in ids.tolist()}
    available_files = {path.name for path in label_dir.glob("*.txt")}
    require(expected_files.issubset(available_files), "one or more acquisition label files are missing")
    for sample_id in ids.tolist():
        path = label_dir / f"{sample_id}.txt"
        labels: set[int] = set()
        for line in path.read_text(encoding="ascii").splitlines():
            tokens = line.split()
            require(len(tokens) == 5, f"invalid YOLO label row: {path.name}")
            label = int(tokens[0])
            require(0 <= label < class_count, f"out-of-range class in {path.name}")
            labels.add(label)
        require(bool(labels), f"empty acquisition label file: {path.name}")
        label_sets.append(frozenset(labels))
    return label_sets


def parse_source_xml_class_set(dataset: str, annotation_path: Path, image_path: Path) -> frozenset[int]:
    """Parse the class set with the same validity rules used by the data freeze."""
    name_to_id = {name: index for index, name in enumerate(NAMES["NEU-DET"])}
    labels: set[int] = set()
    root = ET.parse(annotation_path).getroot()
    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip().lower()
        if dataset == "GC10-DET":
            prefix = raw_name.split("_", 1)[0]
            if prefix.isdigit() and 1 <= int(prefix) <= 10:
                label = int(prefix) - 1
            elif image_path.parent.name.isdigit() and 1 <= int(image_path.parent.name) <= 10:
                label = int(image_path.parent.name) - 1
            else:
                label = None
        else:
            label = name_to_id.get(raw_name)
        box = obj.find("bndbox")
        if label is None or box is None:
            continue
        coords = tuple(int(float(box.findtext(name) or "0")) for name in ("xmin", "ymin", "xmax", "ymax"))
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        labels.add(label)
    require(bool(labels), f"empty or invalid source XML class set: {annotation_path.name}")
    return frozenset(labels)


def validate_source_annotations(
    raw_data_root: Path,
    dataset: str,
    ids: np.ndarray,
    private_registry: pd.DataFrame,
    adapter_label_sets: list[frozenset[int]],
) -> dict[str, object]:
    """Verify each analysis label against the sealed raw-XML hash and class set."""
    rows = private_registry[
        (private_registry["dataset"] == dataset) & (private_registry["split"] == "acquisition")
    ].copy()
    require(not rows.duplicated("sample_id").any(), f"duplicate acquisition source-registry ID: {dataset}")
    row_by_id = rows.set_index("sample_id").to_dict(orient="index")
    require(set(row_by_id) == set(ids.tolist()), f"source-registry/cache ID mismatch: {dataset}")
    for index, sample_id in enumerate(ids.tolist()):
        row = row_by_id[sample_id]
        annotation_path = raw_data_root / Path(row["relative_annotation_path"])
        image_path = raw_data_root / Path(row["relative_image_path"])
        require(annotation_path.is_file(), f"missing sealed source XML: {annotation_path}")
        require(image_path.is_file(), f"missing sealed source image: {image_path}")
        require(sha256(annotation_path) == row["annotation_sha256"], f"source XML hash mismatch: {sample_id}")
        source_labels = parse_source_xml_class_set(dataset, annotation_path, image_path)
        require(
            source_labels == adapter_label_sets[index],
            f"source XML/adapter class-set mismatch: {dataset}/{sample_id}",
        )
    return {
        "source_XML_files_hash_verified": int(len(ids)),
        "source_XML_class_sets_equal_adapter_label_sets": True,
        "raw_data_root": str(raw_data_root),
    }


def derive_neu_nominal_labels(private_registry: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    """Recover NEU's pre-existing image category from the sealed source filename."""
    rows = private_registry[
        (private_registry["dataset"] == "NEU-DET") & (private_registry["split"] == "acquisition")
    ].copy()
    require(not rows.duplicated("sample_id").any(), "duplicate NEU acquisition source-registry ID")
    path_by_id = rows.set_index("sample_id")["relative_image_path"].to_dict()
    require(set(path_by_id) == set(ids.tolist()), "NEU source-registry/cache ID mismatch")
    class_by_name = {name: index for index, name in enumerate(NAMES["NEU-DET"])}
    nominal: list[int] = []
    for sample_id in ids.tolist():
        stem = Path(path_by_id[sample_id]).stem
        category = re.sub(r"_\d+$", "", stem)
        require(category in class_by_name, f"unknown NEU nominal category in sealed filename: {stem}")
        nominal.append(class_by_name[category])
    return np.asarray(nominal, dtype=int)


def cache_content_hash(file_hashes: dict[str, str]) -> str:
    joined = "".join(file_hashes[name] for name in sorted(file_hashes))
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def load_and_validate_dataset(
    source_root: Path,
    raw_data_root: Path,
    dataset: str,
    split_assignment: pd.DataFrame,
    private_registry: pd.DataFrame,
    cache_seal: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, list[frozenset[int]], np.ndarray | None, dict[str, object]]:
    expected = EXPECTED[dataset]
    cache_root = source_root / "confirmatory_v1_1/cache_seals" / dataset
    cache_a = cache_root / "cache_A"
    cache_b = cache_root / "cache_B"
    require(cache_a.is_dir() and cache_b.is_dir(), f"missing cache A/B for {dataset}")

    metadata_a = json.loads((cache_a / "metadata.json").read_text(encoding="utf-8"))
    metadata_b = json.loads((cache_b / "metadata.json").read_text(encoding="utf-8"))
    for label, metadata in (("A", metadata_a), ("B", metadata_b)):
        require(metadata["status"] == "COMPLETE", f"cache {label} incomplete for {dataset}")
        require(metadata["dataset"] == dataset, f"cache {label} dataset mismatch")
        require(metadata["count"] == expected["count"], f"cache {label} count mismatch")
        require(metadata["dimension"] == 384, f"cache {label} dimension mismatch")
        require(metadata["dtype"] == "float32", f"cache {label} dtype mismatch")
        require(metadata["snapshot_revision"] == EXPECTED_DINO_REVISION, f"cache {label} checkpoint mismatch")
        require(metadata["precision"] == "FP32_AUTocast_OFF_TF32_OFF", f"cache {label} precision mismatch")
        observed_hashes = {
            name: sha256(cache_root / f"cache_{label}" / name)
            for name in ("embeddings.npy", "ordered_manifest.txt", "sample_ids.npy")
        }
        require(observed_hashes == metadata["file_sha256"], f"cache {label} file seal mismatch")
        require(cache_content_hash(observed_hashes) == metadata["cache_content_sha256"], f"cache {label} content seal mismatch")

    for name in ("embeddings.npy", "ordered_manifest.txt", "sample_ids.npy"):
        require(sha256(cache_a / name) == sha256(cache_b / name), f"cache A/B not bitwise exact: {dataset}/{name}")
    require(metadata_a["cache_content_sha256"] == cache_seal[dataset]["cache_content_sha256"], "canonical cache seal mismatch")
    require(cache_seal[dataset]["cache_A_B_exact"] is True, "canonical cache A/B flag failed")

    ids = np.load(cache_a / "sample_ids.npy", allow_pickle=False)
    embeddings = np.load(cache_a / "embeddings.npy", allow_pickle=False)
    require(ids.shape == (expected["count"],), f"sample ID shape mismatch for {dataset}")
    require(embeddings.shape == (expected["count"], 384), f"embedding shape mismatch for {dataset}")
    require(embeddings.dtype == np.float32, f"embedding dtype mismatch for {dataset}")
    require(len(set(ids.tolist())) == len(ids), f"duplicate cache IDs for {dataset}")
    require(ids.tolist() == sorted(ids.tolist()), f"cache IDs not sorted for {dataset}")
    require(np.isfinite(embeddings).all(), f"non-finite embeddings for {dataset}")
    norms = np.linalg.norm(embeddings, axis=1)
    require(np.allclose(norms, 1.0, atol=1e-4), f"embeddings not row-L2 normalized for {dataset}")
    ordered_ids = (cache_a / "ordered_manifest.txt").read_text(encoding="ascii").splitlines()
    require(ordered_ids == ids.tolist(), f"ordered manifest/sample_ids row order mismatch for {dataset}")

    blind_name = "gc10_acquisition_universe_blind.csv" if dataset == "GC10-DET" else "neu_acquisition_universe_blind.csv"
    blind = pd.read_csv(source_root / "confirmatory_v1/data_freeze" / blind_name, dtype=str)
    require(blind["sample_id"].tolist() == ids.tolist(), f"cache/blind sample ID mismatch for {dataset}")

    split_rows = split_assignment[split_assignment["dataset"] == dataset]
    acquisition_ids = set(split_rows.loc[split_rows["split"] == "acquisition", "sample_id"])
    development_ids = set(split_rows.loc[split_rows["split"] == "development", "sample_id"])
    final_ids = set(split_rows.loc[split_rows["split"] == "final", "sample_id"])
    cache_ids = set(ids.tolist())
    require(cache_ids == acquisition_ids, f"cache does not equal acquisition split for {dataset}")
    require(not (cache_ids & development_ids), f"development leakage into cache for {dataset}")
    require(not (cache_ids & final_ids), f"final leakage into cache for {dataset}")
    require(not (acquisition_ids & development_ids or acquisition_ids & final_ids or development_ids & final_ids), f"split overlap for {dataset}")

    label_dir = source_root / "confirmatory_v1_1/full_run/yolo_data_cache" / dataset / "labels"
    label_sets = parse_labels(label_dir, ids, int(expected["classes"]))
    source_annotation_record = validate_source_annotations(
        raw_data_root, dataset, ids, private_registry, label_sets
    )
    nominal_labels = derive_neu_nominal_labels(private_registry, ids) if dataset == "NEU-DET" else None
    multi_label_count = sum(len(labels) > 1 for labels in label_sets)

    record = {
        "cache_count": int(len(ids)),
        "cache_dimension": int(embeddings.shape[1]),
        "cache_dtype": str(embeddings.dtype),
        "cache_content_sha256": metadata_a["cache_content_sha256"],
        "cache_A_B_bitwise_exact": True,
        "sample_ids_equal_blind_acquisition_manifest": True,
        "sample_ids_equal_split_assignment_acquisition": True,
        "ordered_manifest_equal_sample_ids_row_order": True,
        "development_overlap": 0,
        "final_overlap": 0,
        "acquisition_label_files_used": int(len(label_sets)),
        "multi_label_image_count": int(multi_label_count),
        "model_revision": metadata_a["snapshot_revision"],
        "precision": metadata_a["precision"],
        **source_annotation_record,
    }
    if nominal_labels is not None:
        record["nominal_category_source"] = (
            "sealed private_source_registry.csv relative_image_path filename prefix; secondary diagnostic only"
        )
        record["nominal_category_counts"] = {
            NAMES[dataset][class_index]: int(np.sum(nominal_labels == class_index))
            for class_index in range(int(expected["classes"]))
        }
    return ids, embeddings, label_sets, nominal_labels, record


def distance_summary(dataset: str, relation: str, values: np.ndarray) -> dict[str, object]:
    require(values.size > 0, f"empty pair group: {dataset}/{relation}")
    return {
        "dataset": dataset,
        "pair_relation": relation,
        "pair_count": int(values.size),
        "mean_cosine_distance": float(np.mean(values)),
        "std_cosine_distance": float(np.std(values, ddof=0)),
        "min_cosine_distance": float(np.min(values)),
        "q25_cosine_distance": float(np.quantile(values, 0.25)),
        "median_cosine_distance": float(np.median(values)),
        "q75_cosine_distance": float(np.quantile(values, 0.75)),
        "max_cosine_distance": float(np.max(values)),
    }


def analyze_dataset(
    dataset: str,
    ids: np.ndarray,
    embeddings: np.ndarray,
    label_sets: list[frozenset[int]],
    nominal_labels: np.ndarray | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    np.fill_diagonal(similarity, -np.inf)
    # IDs are sorted; stable sorting therefore breaks exact ties by sample ID.
    neighbors = np.argsort(-similarity, axis=1, kind="stable")[:, :5]
    nn = neighbors[:, 0]

    sample_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []

    # Primary diagnostics use the actual annotation class set for both datasets.
    masks = np.asarray([sum(1 << label for label in labels) for labels in label_sets], dtype=np.int64)
    overlap_matrix = (masks[:, None] & masks[None, :]) != 0
    k1_agreement = overlap_matrix[np.arange(len(ids)), neighbors[:, 0]].astype(float)
    k5_agreement = overlap_matrix[np.arange(len(ids))[:, None], neighbors[:, :5]].mean(axis=1)
    jaccard_nn = np.asarray(
        [
            len(label_sets[index] & label_sets[int(nn[index])])
            / len(label_sets[index] | label_sets[int(nn[index])])
            for index in range(len(ids))
        ],
        dtype=float,
    )
    metric_rows.extend(
        [
            {"dataset": dataset, "metric": "nearest_neighbor_any_annotation_label_agreement", "value": float(k1_agreement.mean()), "n": len(ids)},
            {"dataset": dataset, "metric": "knn_neighbor_annotation_overlap_k1", "value": float(k1_agreement.mean()), "n": len(ids)},
            {"dataset": dataset, "metric": "knn_neighbor_annotation_overlap_k5", "value": float(k5_agreement.mean()), "n": len(ids)},
            {"dataset": dataset, "metric": "nearest_neighbor_annotation_label_jaccard_mean", "value": float(jaccard_nn.mean()), "n": len(ids)},
        ]
    )
    for index, sample_id in enumerate(ids.tolist()):
        sample_rows.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "annotation_labels_1based": ";".join(str(value + 1) for value in sorted(label_sets[index])),
                "nearest_neighbor_sample_id": ids[int(nn[index])],
                "nearest_neighbor_annotation_labels_1based": ";".join(
                    str(value + 1) for value in sorted(label_sets[int(nn[index])])
                ),
                "nearest_neighbor_cosine_distance": float(1.0 - similarity[index, int(nn[index])]),
                "nearest_neighbor_any_annotation_label_agreement": float(k1_agreement[index]),
                "nearest_neighbor_annotation_label_jaccard": float(jaccard_nn[index]),
                "k5_neighbor_annotation_overlap": float(k5_agreement[index]),
            }
        )
    for class_index, class_name in enumerate(NAMES[dataset]):
        members = np.asarray([class_index in labels for labels in label_sets], dtype=bool)
        class_agreement = np.asarray(
            [class_index in label_sets[int(nn[index])] for index in np.flatnonzero(members)], dtype=float
        )
        class_rows.append(
            {
                "dataset": dataset,
                "label_basis": "annotation_object_class",
                "class_index_0based": class_index,
                "class_number_1based": class_index + 1,
                "class_name": class_name,
                "image_support": int(members.sum()),
                "nearest_neighbor_preserves_class": float(class_agreement.mean()),
                "prior_representation_concern": dataset == "GC10-DET" and class_index + 1 == 8,
            }
        )
    upper = np.triu_indices(len(ids), k=1)
    distances = 1.0 - similarity[upper]
    overlap = overlap_matrix[upper]
    pair_rows.append(distance_summary(dataset, "annotation_label_overlap", distances[overlap]))
    pair_rows.append(distance_summary(dataset, "no_annotation_label_overlap", distances[~overlap]))

    # NEU's filenames encode one nominal category per image, while 83 acquisition
    # images have multi-class object annotations. Keep nominal-category diagnostics
    # secondary and report both label bases instead of collapsing object labels.
    if dataset == "NEU-DET":
        require(nominal_labels is not None and nominal_labels.shape == (len(ids),), "missing NEU nominal labels")
        pred_k1 = nominal_labels[neighbors[:, 0]]
        pred_k5 = np.asarray(
            [
                np.bincount(nominal_labels[row], minlength=EXPECTED[dataset]["classes"]).argmax()
                for row in neighbors[:, :5]
            ],
            dtype=int,
        )
        correct_k1 = (pred_k1 == nominal_labels).astype(float)
        correct_k5 = (pred_k5 == nominal_labels).astype(float)
        silhouette = float(silhouette_score(embeddings, nominal_labels, metric="cosine"))
        metric_rows.extend(
            [
                {"dataset": dataset, "metric": "nearest_neighbor_same_nominal_category", "value": float(correct_k1.mean()), "n": len(ids)},
                {"dataset": dataset, "metric": "leave_one_out_nominal_knn_accuracy_k1", "value": float(correct_k1.mean()), "n": len(ids)},
                {"dataset": dataset, "metric": "leave_one_out_nominal_knn_accuracy_k5", "value": float(correct_k5.mean()), "n": len(ids)},
                {"dataset": dataset, "metric": "nominal_category_silhouette_cosine", "value": silhouette, "n": len(ids)},
            ]
        )
        for index, row in enumerate(sample_rows):
            row.update(
                {
                    "nominal_category_0based": int(nominal_labels[index]),
                    "nominal_category_name": NAMES[dataset][int(nominal_labels[index])],
                    "nearest_neighbor_nominal_category_0based": int(nominal_labels[int(nn[index])]),
                    "nearest_neighbor_nominal_category_name": NAMES[dataset][int(nominal_labels[int(nn[index])])],
                    "nearest_neighbor_same_nominal_category": float(correct_k1[index]),
                    "nominal_knn_prediction_k5_0based": int(pred_k5[index]),
                    "nominal_knn_prediction_k5_name": NAMES[dataset][int(pred_k5[index])],
                    "nominal_knn_correct_k5": float(correct_k5[index]),
                }
            )
        for class_index, class_name in enumerate(NAMES[dataset]):
            members = nominal_labels == class_index
            class_rows.append(
                {
                    "dataset": dataset,
                    "label_basis": "nominal_image_category",
                    "class_index_0based": class_index,
                    "class_number_1based": class_index + 1,
                    "class_name": class_name,
                    "image_support": int(members.sum()),
                    "nearest_neighbor_preserves_class": float(correct_k1[members].mean()),
                    "prior_representation_concern": False,
                }
            )
        same_nominal = nominal_labels[upper[0]] == nominal_labels[upper[1]]
        pair_rows.append(distance_summary(dataset, "same_nominal_category", distances[same_nominal]))
        pair_rows.append(distance_summary(dataset, "different_nominal_category", distances[~same_nominal]))

    return metric_rows, pair_rows, class_rows, sample_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--raw-data-root", type=Path, default=DEFAULT_RAW_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    raw_data_root = args.raw_data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    require(source_root.is_dir(), f"source root missing: {source_root}")
    require(raw_data_root.is_dir(), f"raw data root missing: {raw_data_root}")
    require(not output_dir.exists(), f"output already exists: {output_dir}")

    release_source_hashes = validate_release_against_source(source_root)
    require(MODEL_FILE.is_file(), f"DINOv2 model file missing: {MODEL_FILE}")
    model_hash = sha256(MODEL_FILE)
    require(model_hash == EXPECTED_DINO_MODEL_SHA256, "DINOv2 model checkpoint hash mismatch")

    protocol_path = source_root / "confirmatory_v1_1/confirmatory_protocol_v1_1.yaml"
    require(sha256(protocol_path) == EXPECTED_PROTOCOL_SHA256, "440-run protocol hash mismatch")
    full_run_config = json.loads((source_root / "confirmatory_v1_1/full_run/full_run_config.json").read_text(encoding="utf-8"))
    require(full_run_config["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256, "full-run protocol pointer mismatch")
    require(full_run_config["planned_cells"] == 440, "full-run planned cell count mismatch")
    training_integrity = json.loads(
        (source_root / "confirmatory_v1_1/full_run/runtime/FULL_TRAINING_INTEGRITY_AUDIT.json").read_text(encoding="utf-8")
    )
    require(training_integrity["status"] == "PASS" and training_integrity["complete"] == 440, "440-run integrity failed")
    require(training_integrity["boundary_violations"] == 0, "440-run boundary violation found")
    require(training_integrity["unauthorized_config_changes"] == 0, "440-run config change found")

    cache_seal = json.loads(
        (source_root / "confirmatory_v1_1/cache_seals/CANONICAL_CACHE_SEAL.json").read_text(encoding="utf-8")
    )
    require(cache_seal["status"] == "PASS", "canonical cache seal failed")
    require(cache_seal["canonical_label"] == "A", "canonical cache label is not A")
    require(cache_seal["gt_used"] is False, "GT entered cache or acquisition")
    require(cache_seal["all_20_dino_trajectory_pairs_identical"] is True, "cache trajectories differ")

    freeze_dir = source_root / "confirmatory_v1/data_freeze"
    freeze_summary = json.loads((freeze_dir / "data_freeze_summary.json").read_text(encoding="utf-8"))
    for name in ("private_source_registry.csv", "split_assignment.csv"):
        require(sha256(freeze_dir / name) == freeze_summary["files_sha256"][name], f"data-freeze hash mismatch: {name}")
    split_assignment = pd.read_csv(freeze_dir / "split_assignment.csv", dtype=str)
    require(not split_assignment.duplicated(["dataset", "sample_id"]).any(), "duplicate split assignment")
    private_registry = pd.read_csv(freeze_dir / "private_source_registry.csv", dtype=str)
    require(not private_registry.duplicated(["dataset", "sample_id"]).any(), "duplicate private source-registry row")

    adapter_seal = json.loads(
        (source_root / "confirmatory_v1_1/full_run/data_adapter_seal.json").read_text(encoding="utf-8")
    )
    require(adapter_seal["status"] == "PASS", "data adapter seal failed")
    require(adapter_seal["final_rows_accessed"] == 0, "locked-test row entered training adapter")

    integrity: dict[str, object] = {
        "status": "PASS",
        "analysis_role": "posthoc_representation_audit_only",
        "feature_extraction_performed": False,
        "training_performed": False,
        "fine_tuning_performed": False,
        "acquisition_modified": False,
        "locked_test_labels_read": False,
        "gt_used_for": "posthoc_representation_audit_only",
        "source_root": str(source_root),
        "raw_data_root": str(raw_data_root),
        "dino_revision": EXPECTED_DINO_REVISION,
        "dino_model_sha256": model_hash,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "analysis_protocol_amendment": {
            "timing": "before any metric output was written or inspected",
            "trigger": "NEU-DET acquisition annotations contain multi-class images (83 of 1259)",
            "primary_label_basis": "full annotation object-class set; no label collapsed or excluded",
            "secondary_label_basis": (
                "pre-existing nominal image category parsed deterministically from the sealed source filename"
            ),
            "outcome_driven": False,
        },
        "release_source_records": release_source_hashes,
        "datasets": {},
    }

    all_metrics: list[dict[str, object]] = []
    all_pairs: list[dict[str, object]] = []
    all_classes: list[dict[str, object]] = []
    all_samples: list[dict[str, object]] = []
    for dataset in ("GC10-DET", "NEU-DET"):
        ids, embeddings, label_sets, nominal_labels, record = load_and_validate_dataset(
            source_root, raw_data_root, dataset, split_assignment, private_registry, cache_seal
        )
        metrics, pairs, classes, samples = analyze_dataset(
            dataset, ids, embeddings, label_sets, nominal_labels
        )
        all_metrics.extend(metrics)
        all_pairs.extend(pairs)
        all_classes.extend(classes)
        all_samples.extend(samples)
        integrity["datasets"][dataset] = record

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=output_dir.name + "_tmp_", dir=output_dir.parent))
    try:
        pd.DataFrame(all_metrics).to_csv(temp_dir / "dataset_metrics.csv", index=False, encoding="utf-8", lineterminator="\n")
        pd.DataFrame(all_pairs).to_csv(temp_dir / "pairwise_distance_summary.csv", index=False, encoding="utf-8", lineterminator="\n")
        pd.DataFrame(all_classes).to_csv(temp_dir / "classwise_nearest_neighbor_agreement.csv", index=False, encoding="utf-8", lineterminator="\n")
        pd.DataFrame(all_samples).to_csv(temp_dir / "per_sample_nearest_neighbors.csv", index=False, encoding="utf-8", lineterminator="\n")
        (temp_dir / "integrity_record.json").write_text(
            json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        output_hashes = {
            path.name: sha256(path)
            for path in sorted(temp_dir.iterdir())
            if path.is_file()
        }
        (temp_dir / "OUTPUT_SHA256.json").write_text(
            json.dumps(output_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(output_dir),
                "feature_extraction_performed": False,
                "training_performed": False,
                "datasets": {key: value["cache_count"] for key, value in integrity["datasets"].items()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

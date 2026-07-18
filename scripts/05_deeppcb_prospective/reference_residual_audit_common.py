#!/usr/bin/env python3
"""Shared, training-free utilities for the frozen DeepPCB residual audits.

This module deliberately reads only DeepPCB ``trainval.txt``.  It never opens
the official test list, test annotations, detector checkpoints, or final-test
artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "DeepPCB" / "PCBData"
ORIGINAL_RUN = ROOT / "runs" / "deeppcb_reference_residual_gate" / "prospective_main_20260718"
ELIGIBLE_GROUPS = ("group00041", "group13000", "group20085", "group44000", "group50600", "group77000")
RESERVED_DEVELOPMENT_GROUP = "group92000"
CLASS_NAMES = {1: "open", 2: "short", 3: "mousebite", 4: "spur", 5: "copper", 6: "pin_hole"}
SMALL_BOX_AREA = 1024
FROZEN_THRESHOLD = 32
FROZEN_MIN_AREA = 9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_original_freeze(run_dir: Path = ORIGINAL_RUN) -> dict:
    decision = read_json(run_dir / "gate_decision.json")
    config = read_json(run_dir / "config.json")
    selected = run_dir / "frozen_selected_images.csv"
    recorded_hash = (run_dir / "frozen_selection_sha256.txt").read_text(encoding="utf-8").strip()
    checks = {
        "decision_is_fail_stop": decision.get("decision") == "FAIL_STOP",
        "authorization_is_stop": decision.get("authorization") == "STOP",
        "selection_hash_matches": sha256(selected) == recorded_hash == decision.get("selection_sha256"),
        "threshold_is_32": config.get("threshold") == FROZEN_THRESHOLD,
        "min_component_area_is_9": config.get("min_component_area") == FROZEN_MIN_AREA,
        "small_box_area_is_1024": config.get("small_box_area") == SMALL_BOX_AREA,
        "official_test_unused": not decision.get("official_test_used", True),
        "final_test_unused": not decision.get("final_test_used", True),
        "training_unused": not decision.get("training_performed", True),
        "detector_inference_unused": not decision.get("detector_inference_performed", True),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen DeepPCB branch integrity failed: {checks}")
    return {"checks": checks, "decision": decision, "config": config, "selection_sha256": recorded_hash}


def load_trainval(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Load only trainval pairs and annotations; never read ``test.txt``."""
    split = data_root / "trainval.txt"
    rows: list[dict] = []
    for raw in split.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        image_rel, ann_rel = raw.split()
        base = data_root / image_rel
        tested = base.with_name(base.stem + "_test.jpg")
        template = base.with_name(base.stem + "_temp.jpg")
        annotation = data_root / ann_rel
        group = Path(image_rel).parts[0]
        rows.append(
            {
                "image_id": base.stem,
                "group": group,
                "canonical_path": image_rel.replace("\\", "/"),
                "tested_image": str(tested.resolve()),
                "template_image": str(template.resolve()),
                "annotation": str(annotation.resolve()),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 1000:
        raise RuntimeError(f"Unexpected DeepPCB trainval count: {len(frame)}")
    expected = set(ELIGIBLE_GROUPS) | {RESERVED_DEVELOPMENT_GROUP}
    if set(frame["group"]) != expected:
        raise RuntimeError(f"Unexpected trainval groups: {sorted(frame['group'].unique())}")
    for column in ("tested_image", "template_image", "annotation"):
        missing = frame.loc[~frame[column].map(lambda value: Path(value).is_file()), column]
        if len(missing):
            raise FileNotFoundError(missing.iloc[0])
    return frame


def read_boxes(annotation: str | Path) -> list[dict]:
    boxes: list[dict] = []
    for raw in Path(annotation).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        x1, y1, x2, y2, class_id = map(int, raw.split())
        area = max(0, x2 - x1) * max(0, y2 - y1)
        boxes.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "bbox_area": area,
                "is_small": area <= SMALL_BOX_AREA,
            }
        )
    return boxes


def load_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if image.shape != (640, 640):
        raise RuntimeError(f"Unexpected DeepPCB image shape {image.shape}: {path}")
    return image


@dataclass
class Components:
    mask: np.ndarray
    labels: np.ndarray
    stats: np.ndarray
    centroids: np.ndarray
    scores: np.ndarray


def filtered_components(
    binary_mask: np.ndarray,
    score_map: np.ndarray,
    min_area: int = FROZEN_MIN_AREA,
    score_quantile: float = 0.95,
) -> Components:
    binary = (binary_mask > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    if count > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    clean = keep[labels].astype(np.uint8)
    new_count, new_labels, new_stats, new_centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
    scores = []
    for label in range(1, new_count):
        values = score_map[new_labels == label]
        scores.append(float(np.quantile(values, score_quantile)) if values.size else 0.0)
    return Components(clean, new_labels, new_stats, new_centroids, np.asarray(scores, dtype=float))


def frozen_components(tested: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, Components]:
    difference = cv2.absdiff(tested, template)
    components = filtered_components(difference >= FROZEN_THRESHOLD, difference.astype(np.float32) / 255.0)
    return difference, components


def ssim_dissimilarity(first: np.ndarray, second: np.ndarray) -> tuple[float, np.ndarray]:
    """OpenCV implementation of the standard Gaussian-window SSIM map."""
    x = first.astype(np.float32)
    y = second.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = np.divide(numerator, denominator, out=np.ones_like(numerator), where=np.abs(denominator) > 1e-12)
    dissimilarity = np.clip((1.0 - ssim_map) / 2.0, 0.0, 1.0)
    margin = 32
    central_score = float(ssim_map[margin:-margin, margin:-margin].mean())
    return central_score, dissimilarity


def robust_unit_map(values: np.ndarray, margin: int = 32, percentile: float = 99.5) -> np.ndarray:
    central = values[margin:-margin, margin:-margin]
    low = float(np.median(central))
    high = float(np.percentile(central, percentile))
    scale = max(high - low, 1e-6)
    return np.clip((values.astype(np.float32) - low) / scale, 0.0, 1.0)


def conditional_phase_alignment(tested: np.ndarray, template: np.ndarray, max_shift: float = 8.0) -> tuple[np.ndarray, dict]:
    margin = 32
    before, _ = ssim_dissimilarity(template, tested)
    shift, response = cv2.phaseCorrelate(template.astype(np.float32), tested.astype(np.float32))
    dx, dy = float(shift[0]), float(shift[1])
    magnitude = math.hypot(dx, dy)
    matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned = cv2.warpAffine(tested, matrix, (tested.shape[1], tested.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    after, _ = ssim_dissimilarity(template, aligned)
    accepted = bool(magnitude <= max_shift and after > before + 1e-6)
    return (aligned if accepted else tested), {
        "phase_dx": dx,
        "phase_dy": dy,
        "phase_magnitude": magnitude,
        "phase_response": float(response),
        "central_ssim_before": before,
        "central_ssim_after": after,
        "alignment_accepted": accepted,
    }


def quantile_components(score_map: np.ndarray, quantile: float = 0.995, min_area: int = FROZEN_MIN_AREA) -> Components:
    margin = 32
    threshold = float(np.quantile(score_map[margin:-margin, margin:-margin], quantile))
    binary = (score_map >= threshold).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    return filtered_components(binary, score_map, min_area=min_area)


def component_records(components: Components) -> list[dict]:
    rows: list[dict] = []
    for label in range(1, len(components.stats)):
        x, y, width, height, area = components.stats[label].tolist()
        rows.append(
            {
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + width),
                "y2": int(y + height),
                "area": int(area),
                "score": float(components.scores[label - 1]),
            }
        )
    return rows


def box_iou(first: dict, second: dict) -> float:
    x1 = max(first["x1"], second["x1"])
    y1 = max(first["y1"], second["y1"])
    x2 = min(first["x2"], second["x2"])
    y2 = min(first["y2"], second["y2"])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first["x2"] - first["x1"]) * max(0, first["y2"] - first["y1"])
    second_area = max(0, second["x2"] - second["x1"]) * max(0, second["y2"] - second["y1"])
    union = first_area + second_area - intersection
    return float(intersection / union) if union else 0.0


def greedy_match(predictions: list[dict], boxes: list[dict], score_threshold: float, iou_threshold: float = 0.10) -> tuple[int, int, set[int]]:
    candidates = sorted((item for item in predictions if item["score"] >= score_threshold), key=lambda item: item["score"], reverse=True)
    unmatched = set(range(len(boxes)))
    hits: set[int] = set()
    false_positives = 0
    for prediction in candidates:
        best_index = None
        best_iou = iou_threshold
        for index in unmatched:
            overlap = box_iou(prediction, boxes[index])
            if overlap >= best_iou:
                best_index = index
                best_iou = overlap
        if best_index is None:
            false_positives += 1
        else:
            unmatched.remove(best_index)
            hits.add(best_index)
    return len(hits), false_positives, hits

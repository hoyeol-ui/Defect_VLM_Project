"""Pure selection and explanation logic for the DCAL-XAI experiment.

This module deliberately has no dataset, XML, Ultralytics, or final-test I/O.
The selector receives detector outputs and frozen image embeddings only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DifficultyEvidence:
    """Auditable image-level evidence derived from two detector views."""

    difficulty: float
    confidence_deficit: float
    localization_instability: float
    class_instability: float
    count_instability: float
    original_count: int
    flipped_count: int
    original_max_confidence: float
    flipped_max_confidence: float
    original_classes: tuple[int, ...]
    flipped_classes: tuple[int, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def unflip_xyxy(boxes: np.ndarray, image_width: int) -> np.ndarray:
    """Map boxes predicted on a horizontal flip back to original coordinates."""

    arr = np.asarray(boxes, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected Nx4 boxes, got {arr.shape}")
    restored = arr.copy()
    restored[:, 0] = image_width - arr[:, 2]
    restored[:, 2] = image_width - arr[:, 0]
    return restored


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the pairwise IoU matrix for two xyxy box arrays."""

    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _symmetric_localization_stability(
    original_boxes: np.ndarray,
    original_classes: np.ndarray,
    flipped_boxes: np.ndarray,
    flipped_classes: np.ndarray,
) -> float:
    if len(original_boxes) == 0 and len(flipped_boxes) == 0:
        return 0.0
    if len(original_boxes) == 0 or len(flipped_boxes) == 0:
        return 0.0
    ious = pairwise_iou(original_boxes, flipped_boxes)
    same_class = np.asarray(original_classes)[:, None] == np.asarray(flipped_classes)[None, :]
    matched = np.where(same_class, ious, 0.0)
    forward = matched.max(axis=1).mean()
    backward = matched.max(axis=0).mean()
    return float((forward + backward) / 2.0)


def _class_distribution(classes: np.ndarray, num_classes: int) -> np.ndarray:
    values = np.asarray(classes, dtype=int)
    counts = np.bincount(values, minlength=num_classes).astype(np.float64)
    if counts.sum() == 0:
        return counts
    return counts / counts.sum()


def compute_difficulty(
    original_boxes: np.ndarray,
    original_confidences: np.ndarray,
    original_classes: np.ndarray,
    flipped_boxes_restored: np.ndarray,
    flipped_confidences: np.ndarray,
    flipped_classes: np.ndarray,
    *,
    num_classes: int,
    weights: dict[str, float],
) -> DifficultyEvidence:
    """Compute detector-coupled difficulty without using labels.

    The score combines confidence deficit and original-vs-horizontal-flip
    instability. When both views have no detections, the all-defect GC10 pool
    treats the image as difficult rather than confidently normal.
    """

    expected = {"confidence", "localization", "class", "count"}
    if set(weights) != expected:
        raise ValueError(f"Difficulty weights must be {sorted(expected)}")
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Difficulty weights must sum to 1")
    arrays = [
        np.asarray(original_boxes, dtype=np.float64).reshape(-1, 4),
        np.asarray(original_confidences, dtype=np.float64).reshape(-1),
        np.asarray(original_classes, dtype=int).reshape(-1),
        np.asarray(flipped_boxes_restored, dtype=np.float64).reshape(-1, 4),
        np.asarray(flipped_confidences, dtype=np.float64).reshape(-1),
        np.asarray(flipped_classes, dtype=int).reshape(-1),
    ]
    ob, oc, ok, fb, fc, fk = arrays
    if not (len(ob) == len(oc) == len(ok) and len(fb) == len(fc) == len(fk)):
        raise ValueError("Box/confidence/class cardinalities do not match")
    if np.any((oc < 0) | (oc > 1)) or np.any((fc < 0) | (fc > 1)):
        raise ValueError("Confidences must be in [0, 1]")

    original_max = float(oc.max()) if len(oc) else 0.0
    flipped_max = float(fc.max()) if len(fc) else 0.0
    confidence_deficit = 1.0 - (original_max + flipped_max) / 2.0

    if len(ob) == 0 and len(fb) == 0:
        localization_instability = 1.0
        class_instability = 0.0
    else:
        stability = _symmetric_localization_stability(ob, ok, fb, fk)
        localization_instability = 1.0 - stability
        original_dist = _class_distribution(ok, num_classes)
        flipped_dist = _class_distribution(fk, num_classes)
        class_instability = float(np.abs(original_dist - flipped_dist).sum() / 2.0)

    count_instability = abs(len(ob) - len(fb)) / max(len(ob), len(fb), 1)
    components = {
        "confidence": confidence_deficit,
        "localization": localization_instability,
        "class": class_instability,
        "count": count_instability,
    }
    difficulty = float(sum(weights[name] * components[name] for name in expected))
    return DifficultyEvidence(
        difficulty=float(np.clip(difficulty, 0.0, 1.0)),
        confidence_deficit=float(np.clip(confidence_deficit, 0.0, 1.0)),
        localization_instability=float(np.clip(localization_instability, 0.0, 1.0)),
        class_instability=float(np.clip(class_instability, 0.0, 1.0)),
        count_instability=float(np.clip(count_instability, 0.0, 1.0)),
        original_count=len(ob),
        flipped_count=len(fb),
        original_max_confidence=original_max,
        flipped_max_confidence=flipped_max,
        original_classes=tuple(sorted(set(int(value) for value in ok))),
        flipped_classes=tuple(sorted(set(int(value) for value in fk))),
    )


def stable_top_k(sample_ids: Iterable[str], scores: np.ndarray, k: int) -> list[int]:
    ids = np.asarray(list(sample_ids), dtype=object)
    values = np.asarray(scores, dtype=np.float64)
    if len(ids) != len(values):
        raise ValueError("sample_ids and scores differ in length")
    if k < 1 or k > len(ids):
        raise ValueError(f"Invalid k={k} for n={len(ids)}")
    order = np.lexsort((ids.astype(str), -values))
    return order[:k].astype(int).tolist()


def hybrid_uncertainty_diversity_select(
    *,
    sample_ids: list[str],
    difficulties: np.ndarray,
    candidate_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    query_size: int,
    shortlist_multiplier: int,
    uncertainty_weight: float,
) -> list[int]:
    """Select difficult samples while protecting visual coverage.

    Candidates are first restricted to the top detector-difficulty shortlist.
    Greedy selection then combines difficulty rank and cosine distance from the
    initial labeled set plus samples already chosen in the query.
    """

    ids = list(sample_ids)
    difficulty = np.asarray(difficulties, dtype=np.float64)
    embeddings = np.asarray(candidate_embeddings, dtype=np.float64)
    references = np.asarray(reference_embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or len(embeddings) != len(ids):
        raise ValueError("Candidate embedding shape mismatch")
    if references.ndim != 2 or references.shape[1] != embeddings.shape[1]:
        raise ValueError("Reference embedding shape mismatch")
    if not 0.0 <= uncertainty_weight <= 1.0:
        raise ValueError("uncertainty_weight must be in [0, 1]")
    shortlist_size = min(len(ids), max(query_size, query_size * shortlist_multiplier))
    shortlist = stable_top_k(ids, difficulty, shortlist_size)
    short_embeddings = embeddings[shortlist]
    short_difficulty = difficulty[shortlist]
    short_ids = [ids[index] for index in shortlist]

    order = np.lexsort((np.asarray(short_ids, dtype=str), short_difficulty))
    ranks = np.empty(shortlist_size, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, shortlist_size, endpoint=True)
    max_similarity = short_embeddings @ references.T
    max_similarity = max_similarity.max(axis=1)

    chosen_local: list[int] = []
    available = np.ones(shortlist_size, dtype=bool)
    for _ in range(query_size):
        novelty = np.clip(1.0 - max_similarity, 0.0, 2.0) / 2.0
        objective = uncertainty_weight * ranks + (1.0 - uncertainty_weight) * novelty
        objective[~available] = -np.inf
        best_value = objective.max()
        tied = np.flatnonzero(np.isclose(objective, best_value, rtol=0.0, atol=1e-12))
        chosen = min(tied.tolist(), key=lambda index: short_ids[index])
        chosen_local.append(chosen)
        available[chosen] = False
        similarity = short_embeddings @ short_embeddings[chosen]
        max_similarity = np.maximum(max_similarity, similarity)
    return [shortlist[index] for index in chosen_local]


def grounded_explanation(sample_id: str, evidence: DifficultyEvidence, class_names: list[str]) -> dict[str, object]:
    """Convert numeric detector evidence to a non-hallucinatory review card."""

    reasons: list[str] = []
    if evidence.original_count == 0 and evidence.flipped_count == 0:
        reasons.append("두 입력 뷰 모두에서 검출이 없어 잠재적 미검출 여부를 확인해야 합니다")
    elif evidence.confidence_deficit >= 0.5:
        reasons.append("검출 최대 신뢰도가 낮습니다")
    if evidence.localization_instability >= 0.5:
        reasons.append("수평 반전 전후의 위치 예측이 안정적이지 않습니다")
    if evidence.class_instability >= 0.25:
        reasons.append("수평 반전 전후의 예측 클래스 구성이 달라집니다")
    if evidence.count_instability >= 0.5:
        reasons.append("수평 반전 전후의 검출 개수가 달라집니다")
    if not reasons:
        reasons.append("선택 후보 중 상대적으로 detector difficulty가 높습니다")

    class_indices = sorted(set(evidence.original_classes) | set(evidence.flipped_classes))
    predicted_names = [class_names[index] for index in class_indices if 0 <= index < len(class_names)]
    explanation = "; ".join(reasons) + "."
    return {
        "sample_id": sample_id,
        "explanation": explanation,
        "predicted_classes_only": predicted_names,
        "grounded_evidence": asdict(evidence),
        "claims_ground_truth": False,
        "vlm_used_for_selection": False,
        "review_warning": "이 문장은 detector 출력 근거만 요약하며 실제 결함 정답을 주장하지 않습니다.",
    }


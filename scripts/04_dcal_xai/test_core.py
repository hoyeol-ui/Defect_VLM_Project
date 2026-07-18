"""Synthetic tests for DCAL-XAI. No dataset access, inference, or training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from core import (  # noqa: E402
    compute_difficulty,
    grounded_explanation,
    hybrid_uncertainty_diversity_select,
    pairwise_iou,
    unflip_xyxy,
)


WEIGHTS = {"confidence": 0.40, "localization": 0.30, "class": 0.15, "count": 0.15}


def test_unflip() -> None:
    boxes = np.array([[10.0, 5.0, 30.0, 25.0]])
    restored = unflip_xyxy(boxes, 100)
    np.testing.assert_allclose(restored, [[70.0, 5.0, 90.0, 25.0]])


def test_iou() -> None:
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    np.testing.assert_allclose(pairwise_iou(boxes, boxes), [[1.0]])


def test_stable_prediction_is_easy() -> None:
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    evidence = compute_difficulty(
        box, np.array([0.95]), np.array([2]),
        box, np.array([0.95]), np.array([2]),
        num_classes=10, weights=WEIGHTS,
    )
    assert evidence.difficulty < 0.05
    assert evidence.localization_instability == 0.0


def test_no_detection_is_difficult_in_all_defect_pool() -> None:
    empty_boxes = np.empty((0, 4))
    evidence = compute_difficulty(
        empty_boxes, np.array([]), np.array([], dtype=int),
        empty_boxes, np.array([]), np.array([], dtype=int),
        num_classes=10, weights=WEIGHTS,
    )
    assert evidence.difficulty == 0.70
    assert evidence.confidence_deficit == 1.0
    assert evidence.localization_instability == 1.0


def test_class_and_count_instability_raise_score() -> None:
    original = np.array([[0.0, 0.0, 10.0, 10.0]])
    flipped = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    evidence = compute_difficulty(
        original, np.array([0.8]), np.array([1]),
        flipped, np.array([0.8, 0.7]), np.array([3, 3]),
        num_classes=10, weights=WEIGHTS,
    )
    assert evidence.class_instability == 1.0
    assert evidence.count_instability == 0.5
    assert evidence.difficulty > 0.4


def test_hybrid_selection_is_deterministic_and_diverse() -> None:
    ids = ["a", "b", "c", "d"]
    difficulty = np.array([0.9, 0.8, 0.7, 0.6])
    embeddings = np.array([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    references = np.array([[1.0, 0.0]])
    selected = hybrid_uncertainty_diversity_select(
        sample_ids=ids,
        difficulties=difficulty,
        candidate_embeddings=embeddings,
        reference_embeddings=references,
        query_size=2,
        shortlist_multiplier=2,
        uncertainty_weight=0.5,
    )
    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert selected == hybrid_uncertainty_diversity_select(
        sample_ids=ids,
        difficulties=difficulty,
        candidate_embeddings=embeddings,
        reference_embeddings=references,
        query_size=2,
        shortlist_multiplier=2,
        uncertainty_weight=0.5,
    )


def test_explanation_never_claims_ground_truth() -> None:
    empty = np.empty((0, 4))
    evidence = compute_difficulty(
        empty, np.array([]), np.array([], dtype=int),
        empty, np.array([]), np.array([], dtype=int),
        num_classes=10, weights=WEIGHTS,
    )
    card = grounded_explanation("sample", evidence, [str(i) for i in range(10)])
    assert card["claims_ground_truth"] is False
    assert card["vlm_used_for_selection"] is False
    assert "미검출" in str(card["explanation"])


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in sorted(tests, key=lambda fn: fn.__name__):
        test()
        print(f"[PASS] {test.__name__}")
    print(f"[DONE] {len(tests)} tests passed")


if __name__ == "__main__":
    main()


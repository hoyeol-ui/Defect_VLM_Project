"""Synthetic tests for V7 visual/instance acquisition helpers.

Run:
    .\.python311\python.exe .\tests\test_active_learning_v7_visual_instance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "02_active_learning"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_al_yolo_ablation_v7_visual_instance as v7  # noqa: E402


def synthetic_pool() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"dataset_type": "A", "image_name": "a0.jpg", "score_consistency_only": 0.1, "pseudo_box_count": 0, "class_hint": "x"},
            {"dataset_type": "A", "image_name": "a1.jpg", "score_consistency_only": 0.9, "pseudo_box_count": 1, "class_hint": "x"},
            {"dataset_type": "B", "image_name": "b0.jpg", "score_consistency_only": 0.2, "pseudo_box_count": 3, "class_hint": "y"},
            {"dataset_type": "B", "image_name": "b1.jpg", "score_consistency_only": 0.8, "pseudo_box_count": 2, "class_hint": "y"},
        ]
    )


def test_gtfree_view_blocks_class_hint() -> None:
    view = v7.make_gtfree_view(synthetic_pool())
    assert "class_hint" not in view.columns
    v7.assert_gtfree_view(view)
    try:
        v7.assert_gtfree_view(synthetic_pool())
    except AssertionError:
        pass
    else:
        raise AssertionError("GT-free view did not reject class_hint")


def test_alpha_one_matches_consistency_within_dataset_quota() -> None:
    pool = v7.make_gtfree_view(synthetic_pool())
    labeled = pool.iloc[0:0].copy()
    picked = v7.dataset_balanced_utility_select(
        pool,
        labeled,
        sample_size=2,
        seed=1,
        round_idx=1,
        embedding_lookup=None,
        alpha=1.0,
        beta=0.0,
        gamma=0.0,
        use_consistency=True,
        use_visual=False,
        use_pseudo_instance=False,
    )
    assert set(picked["image_name"]) == {"a1.jpg", "b1.jpg"}


def test_visual_distance_prefers_far_sample() -> None:
    pool = v7.make_gtfree_view(synthetic_pool())
    labeled = pd.DataFrame([{"dataset_type": "A", "image_name": "anchor.jpg", "score_consistency_only": 0.0, "pseudo_box_count": 0}])
    lookup = {
        ("A", "anchor.jpg"): np.array([1.0, 0.0], dtype=np.float32),
        ("A", "a0.jpg"): np.array([0.99, 0.01], dtype=np.float32),
        ("A", "a1.jpg"): np.array([0.0, 1.0], dtype=np.float32),
    }
    candidates = pool[pool["dataset_type"].eq("A")]
    scores = v7.score_candidates(
        candidates,
        labeled_df=labeled,
        selected_df=candidates.iloc[0:0],
        embedding_lookup=lookup,
        alpha=0.0,
        beta=1.0,
        gamma=0.0,
        use_consistency=False,
        use_visual=True,
        use_pseudo_instance=False,
    )
    assert scores.idxmax() == candidates[candidates["image_name"].eq("a1.jpg")].index[0]


def test_pseudo_instance_uses_pseudo_box_count_not_xml() -> None:
    pool = v7.make_gtfree_view(synthetic_pool())
    scores = v7.score_candidates(
        pool,
        labeled_df=pool.iloc[0:0],
        selected_df=pool.iloc[0:0],
        embedding_lookup=None,
        alpha=0.0,
        beta=0.0,
        gamma=1.0,
        use_consistency=False,
        use_visual=False,
        use_pseudo_instance=True,
    )
    assert pool.loc[scores.idxmax(), "image_name"] == "b0.jpg"


def main() -> None:
    test_gtfree_view_blocks_class_hint()
    test_alpha_one_matches_consistency_within_dataset_quota()
    test_visual_distance_prefers_far_sample()
    test_pseudo_instance_uses_pseudo_box_count_not_xml()
    print("V7 visual/instance helper tests passed.")


if __name__ == "__main__":
    main()

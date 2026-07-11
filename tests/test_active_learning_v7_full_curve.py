"""Synthetic tests for V7 full-curve helper functions.

Run:
    .\.python311\python.exe .\tests\test_active_learning_v7_full_curve.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "02_active_learning"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_al_yolo_ablation_v7_full_curve as fc  # noqa: E402


def synthetic_pool() -> pd.DataFrame:
    rows = []
    for ds in ["A", "B"]:
        for i in range(6):
            rows.append(
                {
                    "dataset_type": ds,
                    "image_name": f"{ds}_{i}.jpg",
                    "sample_id": f"{ds}_{i}",
                    "score_consistency_only": float(i),
                    "class_hint": "leak",
                    "actual_bbox_count": 999,
                    "resolved_image_path": f"C:/tmp/{ds}_{i}.jpg",
                    "image_sha256": f"sha_{ds}_{i}",
                }
            )
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)


def synthetic_embeddings(pool: pd.DataFrame) -> dict[str, np.ndarray]:
    lookup = {}
    for _, row in pool.iterrows():
        # A samples lie near x-axis, B samples near y-axis, later index farther.
        idx = int(str(row["image_name"]).split("_")[1].split(".")[0])
        if row["dataset_type"] == "A":
            vec = np.array([1.0, idx / 10.0], dtype=np.float32)
        else:
            vec = np.array([idx / 10.0, 1.0], dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        lookup[row["sample_id"]] = vec
    return lookup


def test_sample_id_depends_on_hash_and_path() -> None:
    a = fc.make_sample_id("A", "x.jpg", "C:/a/x.jpg", "sha1")
    b = fc.make_sample_id("A", "x.jpg", "C:/a/x.jpg", "sha2")
    c = fc.make_sample_id("A", "x.jpg", "C:/b/x.jpg", "sha1")
    assert a != b
    assert a != c


def test_dataset_balanced_consistency_uses_score_not_class_hint() -> None:
    pool = synthetic_pool()
    labeled = pool.iloc[0:0].copy()
    picked = fc.select_dataset_balanced_consistency(pool, labeled, sample_size=2)
    assert set(picked["dataset_type"]) == {"A", "B"}
    assert set(picked["image_name"]) == {"A_5.jpg", "B_5.jpg"}


def test_visual_k_center_prefers_far_samples() -> None:
    pool = synthetic_pool()
    lookup = synthetic_embeddings(pool)
    labeled = pool[pool["sample_id"].isin(["A_0", "B_0"])].copy()
    current = pool[~pool["sample_id"].isin(labeled["sample_id"])].copy()
    picked = fc.select_dataset_balanced_visual(current, labeled, sample_size=2, embedding_lookup=lookup)
    assert len(picked) == 2
    assert set(picked["dataset_type"]) == {"A", "B"}
    # The farthest candidates from the A_0/B_0 anchors are the largest indices.
    assert "A_5.jpg" in set(picked["image_name"])
    assert "B_5.jpg" in set(picked["image_name"])


def test_selection_plan_budgets_and_shared_initial() -> None:
    pool = synthetic_pool()
    lookup = synthetic_embeddings(pool)
    selected, cumulative = fc.build_selection_plan(
        pool,
        seeds=[42],
        strategies=fc.FULL_CURVE_STRATEGIES,
        initial_size=3,
        rounds=2,
        query_size=2,
        embedding_lookup=lookup,
    )
    budgets = sorted(cumulative.groupby(["strategy", "round"]).size().unique())
    assert budgets == [3, 5, 7]
    initials = {
        strategy: set(
            selected[(selected["strategy"].eq(strategy)) & (selected["round"].eq(0))]["sample_id"].astype(str)
        )
        for strategy in fc.FULL_CURVE_STRATEGIES
    }
    assert len({tuple(sorted(v)) for v in initials.values()}) == 1
    for _, sub in cumulative.groupby(["strategy", "round"]):
        assert not sub["sample_id"].duplicated().any()


def test_aulc_calculation() -> None:
    budgets = np.array([15, 20, 25, 30, 35])
    vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert abs(fc.normalized_aulc(budgets, vals) - 0.3) < 1e-9


def test_final_test_lock() -> None:
    old = os.environ.get("AL_EVAL_SPLIT")
    os.environ["AL_EVAL_SPLIT"] = "final_test_v7"
    try:
        try:
            fc.load_development_eval()
        except RuntimeError as exc:
            assert "Final test is locked" in str(exc)
        else:
            raise AssertionError("final_test_v7 was not locked")
    finally:
        if old is None:
            os.environ.pop("AL_EVAL_SPLIT", None)
        else:
            os.environ["AL_EVAL_SPLIT"] = old


def main() -> None:
    test_sample_id_depends_on_hash_and_path()
    test_dataset_balanced_consistency_uses_score_not_class_hint()
    test_visual_k_center_prefers_far_samples()
    test_selection_plan_budgets_and_shared_initial()
    test_aulc_calculation()
    test_final_test_lock()
    print("V7 full-curve helper tests passed.")


if __name__ == "__main__":
    main()

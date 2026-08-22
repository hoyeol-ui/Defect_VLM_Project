from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from confirmatory_v1.protocol_core import (  # noqa: E402
    EmbeddingIntegrityError,
    FinalEvaluationLockedError,
    GTBoundaryError,
    ManifestIntegrityError,
    acquisition_api_has_no_training_seed,
    budget_averaged_aulc,
    budget_grid,
    build_random_trajectory,
    farthest_first_query,
    initial_manifest,
    validate_embedding_cache,
    validate_sample_ids,
    validate_selector_records,
    validate_split_partitions,
    require_final_evaluation_authorization,
)


def sid(index: int) -> str:
    return f"{index:064x}"


class Phase2ACpuPreflight(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = [sid(index) for index in range(1, 101)]

    def test_01_same_acquisition_seed_identical_initial_manifest(self) -> None:
        first = initial_manifest(self.pool, size=10, dataset="GC10-DET", acquisition_seed=8301921)
        second = initial_manifest(self.pool, size=10, dataset="GC10-DET", acquisition_seed=8301921)
        self.assertEqual(first, second)

    def test_02_random_and_dino_share_one_initial_manifest(self) -> None:
        shared = initial_manifest(self.pool, size=10, dataset="NEU-DET", acquisition_seed=991337)
        random_initial = shared
        dino_initial = shared
        self.assertEqual(random_initial, dino_initial)

    def test_03_different_acquisition_seeds_change_initial_manifest(self) -> None:
        first = initial_manifest(self.pool, size=10, dataset="GC10-DET", acquisition_seed=1001)
        second = initial_manifest(self.pool, size=10, dataset="GC10-DET", acquisition_seed=1002)
        self.assertNotEqual(first, second)

    def test_04_training_seed_cannot_change_acquisition(self) -> None:
        self.assertTrue(acquisition_api_has_no_training_seed())
        first = build_random_trajectory(
            self.pool,
            initial_size=10,
            query_size=5,
            rounds=5,
            dataset="GC10-DET",
            acquisition_seed=33003,
        )
        second = build_random_trajectory(
            self.pool,
            initial_size=10,
            query_size=5,
            rounds=5,
            dataset="GC10-DET",
            acquisition_seed=33003,
        )
        self.assertEqual(first, second)

    def test_05_gt_or_class_columns_hard_fail(self) -> None:
        with self.assertRaises(GTBoundaryError):
            validate_selector_records([{"sample_id": sid(1), "true_class": "scratch"}])
        with self.assertRaises(GTBoundaryError):
            validate_selector_records([{"sample_id": sid(1), "bbox": [1, 2, 3, 4]}])

    def test_06_sample_id_contains_no_filename_or_class_text(self) -> None:
        validate_sample_ids([sid(123)])
        with self.assertRaises(ManifestIntegrityError):
            validate_sample_ids(["scratches_123.jpg::" + sid(123)])

    def test_07_manifest_overlap_and_duplicate_sha_hard_fail(self) -> None:
        with self.assertRaises(ManifestIntegrityError):
            validate_split_partitions({"acquisition": [sid(1)], "development": [sid(1)]})
        with self.assertRaises(ManifestIntegrityError):
            validate_split_partitions({"acquisition": [sid(1), sid(1)]})

    def test_08_dino_farthest_first_toy_embedding(self) -> None:
        ids = [sid(1), sid(2), sid(3), sid(4)]
        matrix = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
        cache = validate_embedding_cache(ids, ids, matrix, expected_dim=2)
        query = farthest_first_query(ids[1:], [ids[0]], cache, size=2)
        self.assertEqual(query, (ids[2], ids[1]))

    def test_09_exact_tie_break_is_sample_id_ascending(self) -> None:
        ids = [sid(1), sid(2), sid(3)]
        matrix = np.array([[1, 0], [0, 1], [0, -1]], dtype=np.float32)
        cache = validate_embedding_cache(ids, ids, matrix, expected_dim=2)
        query = farthest_first_query([ids[2], ids[1]], [ids[0]], cache, size=1)
        self.assertEqual(query, (ids[1],))

    def test_10_aulc_matches_hand_calculation(self) -> None:
        # Equal-width trapezoids: (0.1+0.3)/2 and (0.3+0.5)/2; average = 0.3.
        actual = budget_averaged_aulc([0.10, 0.20, 0.30], [0.10, 0.30, 0.50])
        self.assertAlmostEqual(actual, 0.30, places=12)

    def test_11_query_size_and_cumulative_budget_counts(self) -> None:
        self.assertEqual(budget_grid(1605), (161, 241, 321, 401, 481, 561))
        self.assertEqual(budget_grid(1259), (126, 189, 252, 315, 378, 441))

    def test_12_final_evaluation_lock_hard_fails(self) -> None:
        with self.assertRaises(FinalEvaluationLockedError):
            require_final_evaluation_authorization("final")
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "authorization.json"
            with self.assertRaises(FinalEvaluationLockedError):
                require_final_evaluation_authorization(
                    "final", authorization_flag=True, authorization_record=missing
                )

    def test_13_missing_embedding_hard_fails(self) -> None:
        expected = [sid(1), sid(2)]
        with self.assertRaises(EmbeddingIntegrityError):
            validate_embedding_cache(expected, [sid(1)], np.array([[1, 0]], dtype=np.float32), expected_dim=2)

    def test_14_nan_and_zero_vector_hard_fail(self) -> None:
        with self.assertRaises(EmbeddingIntegrityError):
            validate_embedding_cache([sid(1)], [sid(1)], np.array([[np.nan, 1]], dtype=np.float32), expected_dim=2)
        with self.assertRaises(EmbeddingIntegrityError):
            validate_embedding_cache([sid(1)], [sid(1)], np.array([[0, 0]], dtype=np.float32), expected_dim=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)


"""Fast protocol tests that do not read KSDD2 outcomes."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import _top_enrichment, decide_g1, decide_g2_g4
from common import assert_no_label_columns, load_config, stable_hash_key
from dataset_protocol import audit_mask


class ProtocolCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def test_target_blind_guard_rejects_outcome_column(self) -> None:
        with self.assertRaises(ValueError):
            assert_no_label_columns(
                [{"sample_id": "a", "image_path": "a.png", "has_defect": 1}],
                artifact="intentional leak",
            )
        assert_no_label_columns(
            [{"sample_id": "a", "image_path": "a.png", "patchcore_score": 1.2}],
            artifact="valid scores",
        )

    def test_split_hash_is_stable(self) -> None:
        first = stable_hash_key("salt", "10001")
        self.assertEqual(first, stable_hash_key("salt", "10001"))
        self.assertNotEqual(first, stable_hash_key("salt", "10002"))

    def test_mask_audit_uses_frozen_size_bins(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[1:6, 1:6] = 255  # 25 px: small
        mask[20:30, 20:30] = 255  # 100 px: medium
        mask[50:80, 50:80] = 255  # 900 px: large
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mask.png"
            self.assertTrue(cv2.imwrite(str(path), mask))
            result = audit_mask(path, self.config)
        self.assertEqual(result["has_defect"], 1)
        self.assertEqual(result["bbox_count"], 3)
        self.assertEqual(result["small_component_count"], 1)
        self.assertEqual(result["medium_component_count"], 1)
        self.assertEqual(result["large_component_count"], 1)

    def test_top_query_enrichment(self) -> None:
        rows = [
            {"sample_id": str(i), "score": float(20 - i), "has_defect": int(i < 5)}
            for i in range(20)
        ]
        self.assertEqual(_top_enrichment(rows, 5), 4.0)

    def test_g1_decision_pass_and_fail(self) -> None:
        metrics = {
            "image_auroc": {"estimate": 0.80, "ci95": [0.70, 0.90]},
            "bbox_count_spearman": {"estimate": 0.30, "ci95": [0.10, 0.50]},
            "top_query_enrichment": {"estimate": 2.0, "ci95": [1.5, 2.5]},
        }
        self.assertEqual(decide_g1(metrics, self.config)["decision"], "PASS")
        failed = copy.deepcopy(metrics)
        failed["image_auroc"]["estimate"] = 0.60
        self.assertEqual(decide_g1(failed, self.config)["decision"], "FAIL")

    def test_g2_g4_decision_and_stop(self) -> None:
        passing = []
        for seed in range(10):
            passing.append(
                {
                    "acquisition_seed": seed,
                    "positive_image_delta": 10,
                    "positive_image_enrichment": 1.5,
                    "small_defect_image_delta": 4,
                    "size_bin_coverage_delta": 0,
                    "js_divergence_vs_pool_delta": 0.01,
                    "patchcore_small_components": 30,
                    "patchcore_medium_components": 30,
                    "patchcore_large_components": 30,
                    "random_small_components": 20,
                    "random_medium_components": 20,
                    "random_large_components": 20,
                }
            )
        _, decision = decide_g2_g4(passing, self.config)
        self.assertEqual(decision["authorization"], "BOUNDED_SCREEN")
        self.assertTrue(decision["detector_training_authorized"])

        failing = copy.deepcopy(passing)
        for row in failing:
            row["positive_image_delta"] = 0
            row["positive_image_enrichment"] = 1.0
        _, decision = decide_g2_g4(failing, self.config)
        self.assertEqual(decision["authorization"], "STOP")
        self.assertFalse(decision["detector_training_authorized"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Static tests for the G5 protocol; no learner training or mask access."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from validate_prefreeze import load_config, validate_and_build_ledger


class G5ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_exactly_one_learner_configuration(self) -> None:
        learner = self.config["learner"]
        self.assertEqual(learner["name"], "compact_unet_binary_segmentation")
        self.assertFalse(learner["parameter_search"])
        self.assertFalse(learner["pretrained_encoder"])
        self.assertEqual(self.config["comparison"]["fixed_training_seed"], 260727)

    def test_run_count_is_paired_ten_by_two(self) -> None:
        seeds = self.config["comparison"]["acquisition_seeds"]
        strategies = self.config["comparison"]["strategies"]
        self.assertEqual(len(seeds), 10)
        self.assertEqual(strategies, ["patchcore", "random"])
        self.assertEqual(
            self.config["comparison"]["total_training_runs"],
            len(seeds) * len(strategies),
        )

    def test_no_model_selection_or_official_test(self) -> None:
        training = self.config["training"]
        evaluation = self.config["evaluation"]
        self.assertFalse(training["early_stopping"])
        self.assertFalse(training["validation_during_training"])
        self.assertEqual(training["checkpoint_selection"], "final_epoch_only")
        self.assertFalse(evaluation["official_test_evaluation"])
        self.assertFalse(self.config["dataset"]["official_test_access"])
        self.assertFalse(self.config["decision"]["any_outcome_authorizes_official_test"])

    def test_primary_metric_excludes_empty_normal_inflation(self) -> None:
        metric = self.config["evaluation"]["primary_metric"]
        self.assertEqual(metric["name"], "positive_image_macro_hard_dice")
        self.assertEqual(metric["images"], "development_positive_only")
        self.assertEqual(metric["empty_prediction_on_positive_target_value"], 0.0)

    def test_frozen_gate_has_effect_and_safety_requirements(self) -> None:
        gate = self.config["g5_gate"]
        self.assertGreaterEqual(
            gate["primary_learning_translation"]["macro_dice_delta_mean_min"],
            0.05,
        )
        self.assertGreater(
            gate["primary_learning_translation"]["positive_seed_fraction_min"],
            0.5,
        )
        self.assertLessEqual(
            gate["safety"]["candidate_normal_image_false_positive_rate_mean_max"],
            0.20,
        )

    def test_actual_upstream_inputs_validate_without_test_access(self) -> None:
        checks, ledger = validate_and_build_ledger(self.config)
        self.assertEqual(checks["upstream_authorization"], "BOUNDED_SCREEN")
        self.assertEqual(checks["paired_training_runs"], 20)
        self.assertFalse(checks["official_test_access"])
        self.assertFalse(ledger["official_test_content_accessed"])
        self.assertFalse(any("final_locked" in item["path"] for item in ledger["files"]))
        for seed in checks["seed_checks"].values():
            for arm in seed.values():
                self.assertEqual(arm["training_count"], 200)
                self.assertEqual(arm["initial_query_overlap"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

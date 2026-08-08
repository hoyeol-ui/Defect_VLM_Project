"""Unit tests for model, loss, data-independent metrics, and hash determinism."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from g5_common import load_config, set_deterministic, state_dict_sha256
from learner import build_model, segmentation_loss
from metrics import (
    HistogramAveragePrecision,
    component_detection,
    hard_dice,
    normal_false_positive,
)


class LearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_model_preserves_spatial_shape(self) -> None:
        model = build_model(self.config)
        inputs = torch.rand(1, 3, 128, 64)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (1, 1, 128, 64))

    def test_initial_state_hash_is_deterministic(self) -> None:
        set_deterministic(260727)
        first = state_dict_sha256(build_model(self.config).state_dict())
        set_deterministic(260727)
        second = state_dict_sha256(build_model(self.config).state_dict())
        self.assertEqual(first, second)

    def test_loss_is_finite_and_backpropagates(self) -> None:
        model = build_model(self.config)
        inputs = torch.rand(2, 3, 64, 32)
        targets = torch.zeros(2, 1, 64, 32)
        targets[:, :, 10:30, 5:20] = 1
        loss, parts = segmentation_loss(model(inputs), targets, self.config)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(parts["bce"], 0)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_hard_dice(self) -> None:
        target = np.zeros((8, 8), dtype=np.uint8)
        target[2:6, 2:6] = 1
        self.assertEqual(hard_dice(target, target), 1.0)
        self.assertEqual(hard_dice(np.zeros_like(target), target), 0.0)

    def test_component_safety_requires_minimum_area_and_overlap(self) -> None:
        target = np.zeros((16, 16), dtype=np.uint8)
        target[5:10, 5:10] = 1
        tiny = np.zeros_like(target)
        tiny[6:8, 6:8] = 1
        valid = np.zeros_like(target)
        valid[4:10, 4:10] = 1
        off_target = np.zeros_like(target)
        off_target[0:6, 10:16] = 1
        self.assertFalse(component_detection(tiny, target, 16))
        self.assertTrue(component_detection(valid, target, 16))
        self.assertFalse(component_detection(off_target, target, 16))
        self.assertTrue(normal_false_positive(off_target, 16))

    def test_histogram_ap_perfect_ranking(self) -> None:
        metric = HistogramAveragePrecision(bins=16)
        metric.update(
            np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32),
            np.array([1, 1, 0, 0], dtype=np.uint8),
        )
        self.assertAlmostEqual(metric.compute(), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


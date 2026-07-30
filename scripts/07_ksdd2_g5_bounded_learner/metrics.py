"""Memory-bounded segmentation metrics for the frozen G5 screen."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class HistogramAveragePrecision:
    bins: int = 4096

    def __post_init__(self) -> None:
        self.positive = np.zeros(self.bins, dtype=np.int64)
        self.negative = np.zeros(self.bins, dtype=np.int64)

    def update(self, probabilities: np.ndarray, targets: np.ndarray) -> None:
        indices = np.minimum(
            (np.clip(probabilities, 0.0, 1.0) * (self.bins - 1)).astype(np.int32),
            self.bins - 1,
        )
        flat_indices = indices.reshape(-1)
        flat_targets = targets.reshape(-1).astype(bool)
        self.positive += np.bincount(
            flat_indices[flat_targets],
            minlength=self.bins,
        )
        self.negative += np.bincount(
            flat_indices[~flat_targets],
            minlength=self.bins,
        )

    def compute(self) -> float:
        positives = int(self.positive.sum())
        if positives == 0:
            raise RuntimeError("Pixel AP is undefined without positive pixels")
        tp = np.cumsum(self.positive[::-1])
        fp = np.cumsum(self.negative[::-1])
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / positives
        recall_increment = np.diff(np.concatenate(([0.0], recall)))
        return float(np.sum(recall_increment * precision))


def hard_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    denominator = int(prediction.sum() + target.sum())
    if denominator == 0:
        return 1.0
    return float((2 * np.logical_and(prediction, target).sum()) / denominator)


def qualifying_components(
    prediction: np.ndarray,
    minimum_pixels: int,
) -> tuple[np.ndarray, list[int]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        prediction.astype(np.uint8),
        connectivity=8,
    )
    indices = [
        index
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_pixels
    ]
    return labels, indices


def component_detection(
    prediction: np.ndarray,
    target: np.ndarray,
    minimum_pixels: int,
) -> bool:
    labels, indices = qualifying_components(prediction, minimum_pixels)
    return any(np.logical_and(labels == index, target.astype(bool)).any() for index in indices)


def normal_false_positive(prediction: np.ndarray, minimum_pixels: int) -> bool:
    _, indices = qualifying_components(prediction, minimum_pixels)
    return bool(indices)


def image_auroc(targets: list[int], scores: list[float]) -> float:
    return float(roc_auc_score(np.asarray(targets), np.asarray(scores)))


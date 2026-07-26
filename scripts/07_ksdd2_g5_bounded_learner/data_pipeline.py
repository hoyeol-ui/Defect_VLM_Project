"""Deterministic KSDD2 image/mask loading for the frozen G5 learner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from g5_common import require_train_row, resolve_data_path


class CachedSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Resize once into uint8 memory; convert and augment per sample access."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        config: dict[str, Any],
        *,
        augment: bool,
    ) -> None:
        self.rows = rows
        self.augment = augment
        self.flip_probability = float(
            config["preprocessing"]["augmentation"]["horizontal_flip_probability"]
        )
        height, width = [int(value) for value in config["preprocessing"]["image_size"]]
        self.size = (width, height)
        self.images: list[torch.Tensor] = []
        self.masks: list[torch.Tensor] = []
        for row in rows:
            require_train_row(row)
            image_path = resolve_data_path(row["image_path"])
            mask_path = resolve_data_path(row["posthoc_mask_path"])
            with Image.open(image_path) as handle:
                image = handle.convert("RGB").resize(self.size, Image.Resampling.BILINEAR)
                image_array = np.asarray(image, dtype=np.uint8).copy()
            with Image.open(mask_path) as handle:
                mask = handle.convert("L").resize(self.size, Image.Resampling.NEAREST)
                mask_array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
            self.images.append(torch.from_numpy(image_array).permute(2, 0, 1).contiguous())
            self.masks.append(torch.from_numpy(mask_array[None, ...]).contiguous())

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image = self.images[index]
        mask = self.masks[index]
        if self.augment and torch.rand(()) < self.flip_probability:
            image = torch.flip(image, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        return (
            image.to(dtype=torch.float32).div_(255.0),
            mask.to(dtype=torch.float32),
            self.rows[index]["sample_id"],
        )


def assert_native_alignment(rows: list[dict[str, str]]) -> dict[str, Any]:
    checked = 0
    dimensions: set[tuple[int, int]] = set()
    for row in rows:
        require_train_row(row)
        image_path = resolve_data_path(row["image_path"])
        mask_path = resolve_data_path(row["posthoc_mask_path"])
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise RuntimeError(
                    f"Image/mask size mismatch for {row['sample_id']}: "
                    f"{image.size} != {mask.size}"
                )
            if image.height <= image.width:
                raise RuntimeError(f"Unexpected non-portrait KSDD2 sample: {image_path}")
            dimensions.add((image.height, image.width))
        checked += 1
    return {
        "checked": checked,
        "native_orientation": "portrait",
        "unique_height_width_pairs": len(dimensions),
        "minimum_height": min(value[0] for value in dimensions),
        "maximum_height": max(value[0] for value in dimensions),
        "minimum_width": min(value[1] for value in dimensions),
        "maximum_width": max(value[1] for value in dimensions),
    }


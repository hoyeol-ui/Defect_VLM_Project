"""Pre-freeze implementation QA without using any G5 learner outcome."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_pipeline import assert_native_alignment
from g5_common import (
    load_config,
    read_csv,
    resolve_data_path,
    resolve_source,
    run_root,
    seal_file,
    set_deterministic,
    write_json,
)
from learner import build_model, segmentation_loss


def synthetic_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = 128, 64
    images = torch.zeros((4, 3, height, width), dtype=torch.float32, device=device)
    masks = torch.zeros((4, 1, height, width), dtype=torch.float32, device=device)
    boxes = [(16, 8, 64, 32), (40, 24, 96, 56), (24, 16, 104, 44), (8, 20, 72, 60)]
    for index, (top, left, bottom, right) in enumerate(boxes):
        masks[index, :, top:bottom, left:right] = 1.0
        images[index, 0, top:bottom, left:right] = 1.0
        images[index, 1, top:bottom, left:right] = 0.7
        row_gradient = torch.linspace(0.0, 0.2, height, device=device)[:, None]
        images[index, 2] = row_gradient
    return images, masks


def synthetic_overfit(config: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config["comparison"]["training_seeds"][0])
    set_deterministic(seed)
    model = build_model(config).to(device)
    images, masks = synthetic_batch(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    initial_loss = None
    started = time.perf_counter()
    model.train()
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss, _ = segmentation_loss(logits, masks, config)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(config["training"]["gradient_clip_norm"]),
        )
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        prediction = torch.sigmoid(model(images)) >= 0.5
    intersection = (prediction & masks.bool()).sum(dim=(1, 2, 3)).float()
    denominator = prediction.sum(dim=(1, 2, 3)).float() + masks.sum(dim=(1, 2, 3))
    dice = ((2.0 * intersection) / denominator.clamp_min(1.0)).cpu().numpy()
    result = {
        "device": str(device),
        "steps": 500,
        "initial_loss": initial_loss,
        "final_loss": float(loss.detach()),
        "minimum_hard_dice": float(dice.min()),
        "mean_hard_dice": float(dice.mean()),
        "seconds": time.perf_counter() - started,
        "pass": bool(dice.min() >= 0.95 and float(loss.detach()) < float(initial_loss)),
    }
    if not result["pass"]:
        raise RuntimeError(f"Synthetic overfit QA failed: {result}")
    return result


def _positive_overlay(
    development: list[dict[str, str]],
    target: Path,
    count: int = 6,
) -> dict[str, Any]:
    tiles: list[Image.Image] = []
    sample_ids: list[str] = []
    for row in development:
        image_path = resolve_data_path(row["image_path"])
        mask_path = resolve_data_path(row["posthoc_mask_path"])
        with Image.open(mask_path) as handle:
            mask_native = np.asarray(handle.convert("L"), dtype=np.uint8)
        if not np.any(mask_native > 0):
            continue
        with Image.open(image_path) as handle:
            image = handle.convert("RGB").resize((128, 352), Image.Resampling.BILINEAR)
        mask = Image.fromarray((mask_native > 0).astype(np.uint8) * 255).resize(
            (128, 352),
            Image.Resampling.NEAREST,
        )
        array = np.asarray(image, dtype=np.uint8).copy()
        positive = np.asarray(mask, dtype=np.uint8) > 0
        array[positive] = (
            0.35 * array[positive] + 0.65 * np.array([255, 32, 32])
        ).astype(np.uint8)
        tiles.append(Image.fromarray(array))
        sample_ids.append(row["sample_id"])
        if len(tiles) == count:
            break
    if len(tiles) != count:
        raise RuntimeError("Could not find enough positive development masks for QA")
    canvas = Image.new("RGB", (128 * 3, 352 * 2), "white")
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % 3) * 128, (index // 3) * 352))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    return {"samples": sample_ids, "overlay": str(target)}


def main() -> int:
    config = load_config()
    source = resolve_source(config)
    acquisition = read_csv(source / "acquisition_manifest.csv")
    development = read_csv(source / "development_manifest.csv")
    alignment = assert_native_alignment(acquisition + development)
    expected_height, expected_width = config["preprocessing"]["image_size"]
    if expected_height <= expected_width:
        raise RuntimeError("Frozen resize must preserve KSDD2 portrait orientation")
    out = run_root(config) / "qa"
    overlay = _positive_overlay(development, out / "image_mask_alignment_overlay.png")
    result = {
        "status": "PASS",
        "official_test_accessed": False,
        "native_alignment": alignment,
        "frozen_resize_height_width": [expected_height, expected_width],
        "synthetic_overfit": synthetic_overfit(config),
        "visual_alignment": overlay,
    }
    target = out / "preflight.json"
    write_json(target, result)
    seal_file(target)
    seal_file(Path(overlay["overlay"]))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

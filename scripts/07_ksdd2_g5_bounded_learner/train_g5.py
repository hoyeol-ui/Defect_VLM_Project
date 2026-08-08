"""Execute the frozen 60-run KSDD2 G5 learner screen."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from data_pipeline import CachedSegmentationDataset
from g5_common import (
    ROOT,
    device_from_environment,
    load_config,
    read_csv,
    relative_to_root,
    resolve_data_path,
    run_root,
    seal_file,
    set_deterministic,
    sha256_file,
    state_dict_sha256,
    verify_frozen_execution,
    verify_seal,
    write_csv,
    write_json,
)
from learner import build_model, segmentation_loss
from metrics import (
    HistogramAveragePrecision,
    component_detection,
    hard_dice,
    image_auroc,
    normal_false_positive,
)


def _absolute(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def verify_dataset_files(
    run_plan: list[dict[str, str]],
    development_manifest: Path,
) -> dict[str, Any]:
    manifest_paths = {
        _absolute(row["training_manifest"]).resolve() for row in run_plan
    }
    manifest_paths.add(development_manifest.resolve())
    checked: dict[Path, str] = {}
    for manifest in sorted(manifest_paths, key=str):
        if not verify_seal(manifest):
            raise RuntimeError(f"Invalid exact-manifest seal: {manifest}")
        for row in read_csv(manifest):
            for path_key, hash_key in (
                ("image_path", "image_sha256"),
                ("posthoc_mask_path", "mask_sha256"),
            ):
                path = resolve_data_path(row[path_key])
                expected = row[hash_key]
                if path in checked:
                    if checked[path] != expected:
                        raise RuntimeError(f"Conflicting expected hash for {path}")
                    continue
                actual = sha256_file(path)
                if actual != expected:
                    raise RuntimeError(f"Dataset file hash mismatch: {path}")
                checked[path] = actual
    return {
        "manifests": len(manifest_paths),
        "unique_dataset_files": len(checked),
        "official_test_accessed": False,
    }


def _attempt(run_directory: Path) -> int:
    attempt_path = run_directory / "attempts.json"
    if attempt_path.exists():
        attempts = json.loads(attempt_path.read_text(encoding="utf-8"))
    else:
        attempts = {"attempts_started": 0, "history": []}
    if int(attempts["attempts_started"]) >= 2:
        raise RuntimeError(f"Frozen identical-retry limit exhausted: {run_directory}")
    attempts["attempts_started"] = int(attempts["attempts_started"]) + 1
    attempts["history"].append(
        {
            "attempt": attempts["attempts_started"],
            "started_unix": time.time(),
            "pid": os.getpid(),
        }
    )
    write_json(attempt_path, attempts)
    return int(attempts["attempts_started"])


def _make_loader(
    dataset: CachedSegmentationDataset,
    config: dict[str, Any],
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=True,
        drop_last=False,
        generator=generator,
    )


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    threshold = float(config["evaluation"]["probability_threshold"])
    minimum_pixels = int(
        config["evaluation"]["minimum_false_positive_component_pixels"]
    )
    histogram_ap = HistogramAveragePrecision(bins=4096)
    image_targets: list[int] = []
    image_scores: list[float] = []
    positive_dice: list[float] = []
    positive_detected: list[int] = []
    normal_false_positives: list[int] = []
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for images, masks, sample_ids in loader:
            probabilities = torch.sigmoid(
                model(images.to(device, non_blocking=True))
            ).cpu().numpy()[:, 0]
            targets = masks.numpy()[:, 0].astype(np.uint8)
            histogram_ap.update(probabilities, targets)
            for index, sample_id in enumerate(sample_ids):
                probability = probabilities[index]
                target = targets[index]
                prediction = probability >= threshold
                has_defect = int(target.any())
                score = float(probability.max())
                image_targets.append(has_defect)
                image_scores.append(score)
                row: dict[str, Any] = {
                    "sample_id": sample_id,
                    "has_defect": has_defect,
                    "max_pixel_probability": score,
                    "hard_dice": "",
                    "component_detected": "",
                    "normal_false_positive": "",
                }
                if has_defect:
                    dice = hard_dice(prediction, target)
                    detected = int(
                        component_detection(prediction, target, minimum_pixels)
                    )
                    positive_dice.append(dice)
                    positive_detected.append(detected)
                    row["hard_dice"] = dice
                    row["component_detected"] = detected
                else:
                    false_positive = int(
                        normal_false_positive(prediction, minimum_pixels)
                    )
                    normal_false_positives.append(false_positive)
                    row["normal_false_positive"] = false_positive
                rows.append(row)
    metrics = {
        "positive_image_macro_hard_dice": float(np.mean(positive_dice)),
        "all_pixel_average_precision_hist4096": histogram_ap.compute(),
        "image_auroc_using_max_pixel_probability": image_auroc(
            image_targets, image_scores
        ),
        "normal_image_false_positive_rate": float(
            np.mean(normal_false_positives)
        ),
        "positive_image_recall_at_overlapping_component": float(
            np.mean(positive_detected)
        ),
        "evaluation_images": float(len(rows)),
        "evaluation_positive_images": float(len(positive_dice)),
        "evaluation_normal_images": float(len(normal_false_positives)),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Non-finite evaluation metric: {metrics}")
    return metrics, rows


def train_one(
    plan: dict[str, str],
    config: dict[str, Any],
    freeze_path: Path,
    freeze: dict[str, Any],
    development_dataset: CachedSegmentationDataset,
    device: torch.device,
) -> dict[str, Any]:
    out = run_root(config)
    run_id = plan["run_id"]
    run_directory = out / "learner_runs" / run_id
    result_path = run_directory / "run_result.json"
    if verify_seal(result_path):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["freeze_manifest_sha256"] != sha256_file(freeze_path):
            raise RuntimeError(f"Completed run belongs to another freeze: {run_id}")
        print(f"[SKIP] {run_id}", flush=True)
        return result
    run_directory.mkdir(parents=True, exist_ok=True)
    attempt = _attempt(run_directory)

    acquisition_seed = int(plan["acquisition_seed"])
    training_seed = int(plan["training_seed"])
    strategy = plan["strategy"]
    set_deterministic(training_seed)
    model = build_model(config)
    initial_hash = state_dict_sha256(model.state_dict())
    if initial_hash != plan["initial_state_sha256"]:
        raise RuntimeError(f"Initial-state hash changed for {run_id}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    training_manifest = _absolute(plan["training_manifest"]).resolve()
    if sha256_file(training_manifest) != plan["training_manifest_sha256"]:
        raise RuntimeError(f"Training manifest changed for {run_id}")
    training_dataset = CachedSegmentationDataset(
        read_csv(training_manifest),
        config,
        augment=True,
    )
    training_loader = _make_loader(
        training_dataset,
        config,
        shuffle=True,
        seed=training_seed,
    )
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    epochs = int(config["training"]["epochs"])
    print(
        f"[START] {run_id} attempt={attempt} images={len(training_dataset)} "
        f"epochs={epochs}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        batches = 0
        for images, masks, _ in training_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss, parts = segmentation_loss(logits, masks, config)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {run_id}, epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["training"]["gradient_clip_norm"]),
            )
            optimizer.step()
            total_loss += float(loss.detach())
            total_bce += parts["bce"]
            total_dice += parts["soft_dice"]
            batches += 1
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / batches,
                "bce": total_bce / batches,
                "soft_dice_loss": total_dice / batches,
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[TRAIN] {run_id} epoch={epoch:03d}/{epochs} "
                f"loss={history[-1]['loss']:.6f}",
                flush=True,
            )

    evaluation_loader = _make_loader(
        development_dataset,
        config,
        shuffle=False,
        seed=training_seed,
    )
    metrics, per_image = evaluate(model, evaluation_loader, config, device)
    checkpoint_path = run_directory / "final_epoch80.pt"
    cpu_state = {
        name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
    }
    temporary_checkpoint = checkpoint_path.with_suffix(".tmp")
    torch.save(
        {
            "protocol_id": config["protocol"]["id"],
            "run_id": run_id,
            "epoch": epochs,
            "state_dict": cpu_state,
        },
        temporary_checkpoint,
    )
    temporary_checkpoint.replace(checkpoint_path)
    seal_file(checkpoint_path)
    write_csv(
        run_directory / "training_history.csv",
        history,
        ["epoch", "loss", "bce", "soft_dice_loss"],
    )
    seal_file(run_directory / "training_history.csv")
    write_csv(
        run_directory / "per_image_metrics.csv",
        per_image,
        [
            "sample_id",
            "has_defect",
            "max_pixel_probability",
            "hard_dice",
            "component_detected",
            "normal_false_positive",
        ],
    )
    seal_file(run_directory / "per_image_metrics.csv")
    result = {
        "protocol_id": config["protocol"]["id"],
        "run_id": run_id,
        "status": "COMPLETED",
        "attempt": attempt,
        "acquisition_seed": acquisition_seed,
        "strategy": strategy,
        "training_seed": training_seed,
        "training_images": len(training_dataset),
        "epochs": epochs,
        "initial_state_sha256": initial_hash,
        "training_manifest": relative_to_root(training_manifest),
        "training_manifest_sha256": sha256_file(training_manifest),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "checkpoint": relative_to_root(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_seconds": time.perf_counter() - started,
        "final_training_loss": history[-1]["loss"],
        "metrics": metrics,
        "official_test_accessed": False,
    }
    write_json(result_path, result)
    seal_file(result_path)
    print(
        f"[DONE] {run_id} dice={metrics['positive_image_macro_hard_dice']:.4f} "
        f"ap={metrics['all_pixel_average_precision_hist4096']:.4f} "
        f"fpr={metrics['normal_image_false_positive_rate']:.4f}",
        flush=True,
    )
    del optimizer, model, training_loader, training_dataset, evaluation_loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if not args.all and not args.run_id:
        parser.error("Specify --all or --run-id")
    config = load_config()
    freeze_path, freeze = verify_frozen_execution(config)
    out = run_root(config)
    run_plan_path = out / "frozen_inputs" / "run_plan.csv"
    development_manifest = (
        out / "frozen_inputs" / "development_evaluation_manifest.csv"
    )
    run_plan = read_csv(run_plan_path)
    if len(run_plan) != int(config["comparison"]["total_training_runs"]):
        raise RuntimeError("Frozen run plan count changed")
    if args.run_id:
        run_plan = [row for row in run_plan if row["run_id"] == args.run_id]
        if len(run_plan) != 1:
            raise RuntimeError(f"Unknown frozen run ID: {args.run_id}")
    verification_path = out / "dataset_input_verification.json"
    if not verify_seal(verification_path):
        verification = verify_dataset_files(
            read_csv(run_plan_path),
            development_manifest,
        )
        write_json(verification_path, verification)
        seal_file(verification_path)
    device = device_from_environment()
    development_dataset = CachedSegmentationDataset(
        read_csv(development_manifest),
        config,
        augment=False,
    )
    completed = 0
    for plan in run_plan:
        train_one(
            plan,
            config,
            freeze_path,
            freeze,
            development_dataset,
            device,
        )
        completed += 1
    print(
        json.dumps(
            {
                "requested_runs": len(run_plan),
                "processed_or_skipped": completed,
                "official_test_accessed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

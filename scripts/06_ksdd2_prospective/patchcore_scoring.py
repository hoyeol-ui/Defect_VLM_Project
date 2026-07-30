"""Target-blind PatchCore scoring for the frozen KSDD2 protocol."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import (
    assert_no_label_columns,
    project_path,
    read_csv,
    read_json,
    run_root,
    seal_file,
    state_dict_sha256,
    verify_seal,
    write_csv,
    write_json,
)


SCORE_FIELDS = [
    "sample_id",
    "image_path",
    "patchcore_score",
    "acquisition_seed",
    "score_rank",
]


class BlindImageDataset(Dataset):
    """Image-only dataset; no target or mask is accepted by construction."""

    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            raise ValueError("BlindImageDataset received no images")
        assert_no_label_columns(list(rows), artifact="PatchCore input manifest")
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from anomalib.data import ImageItem
        from anomalib.data.utils.image import read_image

        row = self.rows[index]
        path = project_path(row["image_path"])
        return ImageItem(
            image=read_image(path, as_tensor=True),
            image_path=str(path),
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def _loader(
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    from anomalib.data import ImageBatch

    return DataLoader(
        BlindImageDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=ImageBatch.collate,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def _flatten_predictions(predictions: list[Any]) -> list[tuple[str, float]]:
    flattened: list[tuple[str, float]] = []
    for batch in predictions:
        paths = batch.image_path
        scores = batch.pred_score
        if scores is None:
            raise RuntimeError("PatchCore prediction did not contain pred_score")
        if isinstance(paths, (str, Path)):
            paths = [paths]
        if isinstance(scores, torch.Tensor):
            score_values = scores.detach().cpu().reshape(-1).tolist()
        elif isinstance(scores, np.ndarray):
            score_values = scores.reshape(-1).tolist()
        elif isinstance(scores, (list, tuple)):
            score_values = [float(value) for value in scores]
        else:
            score_values = [float(scores)]
        if len(paths) != len(score_values):
            raise RuntimeError(
                f"Prediction path/score length mismatch: {len(paths)} != {len(score_values)}"
            )
        flattened.extend((str(path), float(score)) for path, score in zip(paths, score_values))
    return flattened


def score_rows(
    config: dict[str, Any],
    *,
    reference_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    seed: int,
    output_path: Path,
    pretrained_override: bool | None = None,
) -> dict[str, Any]:
    """Fit PatchCore on normal references and seal target-only image scores."""

    assert_no_label_columns(reference_rows, artifact="PatchCore reference rows")
    assert_no_label_columns(target_rows, artifact="PatchCore target rows")
    if {row["sample_id"] for row in reference_rows} & {row["sample_id"] for row in target_rows}:
        raise ValueError("Reference and target image sets overlap")

    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    _seed_everything(seed)
    params = config["patchcore"]
    pretrained = bool(params["pretrained"]) if pretrained_override is None else pretrained_override
    pre_processor = Patchcore.configure_pre_processor(
        image_size=tuple(int(value) for value in params["image_size"]),
        center_crop_size=None,
    )
    model = Patchcore(
        backbone=str(params["backbone"]),
        layers=tuple(params["layers"]),
        pre_trained=pretrained,
        coreset_sampling_ratio=float(params["coreset_sampling_ratio"]),
        num_neighbors=int(params["num_neighbors"]),
        pre_processor=pre_processor,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    if pretrained:
        freeze_path = run_root(config) / config["outputs"]["freeze_manifest"]
        if not freeze_path.exists() or not verify_seal(freeze_path):
            raise RuntimeError("A sealed freeze manifest is required for pretrained scoring")
        frozen = read_json(freeze_path)
        actual_backbone_hash = state_dict_sha256(model.model.feature_extractor.state_dict())
        expected_backbone_hash = frozen["model"]["state_dict_sha256"]
        if actual_backbone_hash != expected_backbone_hash:
            raise RuntimeError(
                "Loaded PatchCore backbone parameters do not match the frozen fingerprint"
            )
    accelerator = "gpu" if torch.cuda.is_available() and params["device"] == "auto" else params["device"]
    if accelerator == "auto":
        accelerator = "cpu"
    engine = Engine(
        accelerator=accelerator,
        devices=1,
        max_epochs=1,
        deterministic=True,
        barebones=True,
        default_root_dir=output_path.parent / "_engine",
    )
    train_loader = _loader(
        reference_rows,
        batch_size=int(params["train_batch_size"]),
        num_workers=int(params["num_workers"]),
        shuffle=False,
    )
    predict_loader = _loader(
        target_rows,
        batch_size=int(params["predict_batch_size"]),
        num_workers=int(params["num_workers"]),
        shuffle=False,
    )
    engine.fit(model=model, train_dataloaders=train_loader)
    predictions = engine.predict(
        model=model,
        dataloaders=predict_loader,
        return_predictions=True,
    )
    if predictions is None:
        raise RuntimeError("PatchCore returned no predictions")
    by_path = {
        str(Path(path).resolve()).lower(): score for path, score in _flatten_predictions(predictions)
    }
    decimals = int(params["score_round_decimals"])
    scored: list[dict[str, Any]] = []
    for row in target_rows:
        resolved = str(project_path(row["image_path"])).lower()
        if resolved not in by_path:
            raise RuntimeError(f"Missing PatchCore score for {row['sample_id']}")
        score = by_path[resolved]
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite PatchCore score for {row['sample_id']}: {score}")
        scored.append(
            {
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                "patchcore_score": round(score, decimals),
                "acquisition_seed": seed,
                "score_rank": 0,
            }
        )

    scored.sort(key=lambda row: (-float(row["patchcore_score"]), str(row["sample_id"])))
    for rank, row in enumerate(scored, start=1):
        row["score_rank"] = rank
    assert_no_label_columns(scored, artifact="sealed PatchCore score output")
    write_csv(output_path, scored, SCORE_FIELDS)
    seal_file(output_path)
    metadata = {
        "seed": seed,
        "reference_count": len(reference_rows),
        "target_count": len(target_rows),
        "pretrained": pretrained,
        "accelerator": accelerator,
        "score_file": output_path.name,
        "score_file_sha256_sealed": True,
        "target_or_mask_fields_consumed": False,
    }
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    seal_file(output_path.with_suffix(".metadata.json"))
    return metadata


def normal_reference_rows(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Join revealed initial outcomes only to identify normal PatchCore references."""

    out = run_root(config)
    initial = read_csv(out / "initial_sets" / f"initial_seed{seed:02d}.csv")
    initial_outcome_path = out / "initial_labels" / f"initial_labels_seed{seed:02d}.csv"
    if not verify_seal(initial_outcome_path):
        raise RuntimeError(f"Initial-label seal is missing or invalid for seed {seed}")
    outcomes = {
        row["sample_id"]: row
        for row in read_csv(initial_outcome_path)
    }
    references = [
        {
            "sample_id": row["sample_id"],
            "image_path": row["image_path"],
            "reference_role": "known_normal_reference",
        }
        for row in initial
        if int(outcomes[row["sample_id"]]["has_defect"]) == 0
    ]
    minimum = int(config["acquisition"]["minimum_normal_references"])
    if len(references) < minimum:
        raise RuntimeError(
            f"Only {len(references)} normal initial references; frozen minimum is {minimum}. STOP."
        )
    assert_no_label_columns(references, artifact="normal-reference image list")
    return references


def score_g1(config: dict[str, Any]) -> dict[str, Any]:
    out = run_root(config)
    seed = int(config["acquisition"]["g1_reference_seed"])
    references = normal_reference_rows(config, seed)
    targets = [
        {"sample_id": row["sample_id"], "image_path": row["image_path"]}
        for row in read_csv(out / config["outputs"]["development_manifest"])
    ]
    return score_rows(
        config,
        reference_rows=references,
        target_rows=targets,
        seed=seed,
        output_path=out / "g1" / f"g1_scores_seed{seed:02d}.csv",
    )


def score_selection_seed(config: dict[str, Any], seed: int) -> dict[str, Any]:
    out = run_root(config)
    references = normal_reference_rows(config, seed)
    initial_ids = {
        row["sample_id"]
        for row in read_csv(out / "initial_sets" / f"initial_seed{seed:02d}.csv")
    }
    targets = [
        {"sample_id": row["sample_id"], "image_path": row["image_path"]}
        for row in read_csv(out / config["outputs"]["acquisition_manifest"])
        if row["sample_id"] not in initial_ids
    ]
    return score_rows(
        config,
        reference_rows=references,
        target_rows=targets,
        seed=seed,
        output_path=out / "selection_scores" / f"selection_scores_seed{seed:02d}.csv",
    )

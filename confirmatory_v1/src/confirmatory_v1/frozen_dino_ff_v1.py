"""FrozenDINO-FF-v1 embedding adapter for the future Phase 2B cache build.

Only this adapter may see image paths. Its output boundary is an ordered array
of opaque sample IDs and sealed, row-wise L2-normalized 384-D embeddings. The
selector in protocol_core never receives paths, filenames, XML, class labels,
or detector metrics.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .protocol_core import EmbeddingIntegrityError, ValidatedEmbeddingCache, validate_embedding_cache, validate_sample_ids


MODEL_ID = "facebook/dinov2-small"
SNAPSHOT_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
ARTIFACT_SHA256 = {
    "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
}
PROCESSOR_CLASS = "BitImageProcessor"
REPRESENTATION = "outputs.pooler_output"
EMBEDDING_DIMENSION = 384
OUTPUT_DTYPE = "float32"
INFERENCE_PRECISION = "torch.float32; autocast disabled; TF32 disabled"
DEFAULT_BATCH_SIZE = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(snapshot_dir: Path) -> dict[str, str]:
    """Verify the recovered snapshot revision and all three sealed artifacts."""

    if snapshot_dir.name != SNAPSHOT_REVISION:
        raise EmbeddingIntegrityError(
            f"DINO snapshot revision mismatch: expected={SNAPSHOT_REVISION}, actual={snapshot_dir.name}"
        )
    observed: dict[str, str] = {}
    for filename, expected in ARTIFACT_SHA256.items():
        path = snapshot_dir / filename
        if not path.is_file():
            raise EmbeddingIntegrityError(f"missing DINO artifact: {path}")
        observed[filename] = _sha256(path)
        if observed[filename] != expected:
            raise EmbeddingIntegrityError(
                f"DINO artifact SHA-256 mismatch for {filename}: expected={expected}, actual={observed[filename]}"
            )
    return observed


def configure_torch_determinism(torch_module: object) -> None:
    """Apply the frozen FP32 determinism policy before model construction."""

    torch_module.use_deterministic_algorithms(True)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.backends.cuda.matmul.allow_tf32 = False


def extract_embeddings(
    items: Sequence[tuple[str, Path]],
    *,
    snapshot_dir: Path,
    device: str = "cuda:0",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ValidatedEmbeddingCache:
    """Extract the frozen pooler representation; intended for Phase 2B only."""

    if batch_size != DEFAULT_BATCH_SIZE:
        raise EmbeddingIntegrityError(
            f"batch size is frozen at {DEFAULT_BATCH_SIZE}; protocol re-freeze is required to change it"
        )
    verify_snapshot(snapshot_dir)
    ordered = sorted(((str(sample_id), Path(path)) for sample_id, path in items), key=lambda item: item[0])
    sample_ids = [sample_id for sample_id, _ in ordered]
    validate_sample_ids(sample_ids)

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    configure_torch_determinism(torch)
    processor = AutoImageProcessor.from_pretrained(snapshot_dir, local_files_only=True)
    if processor.__class__.__name__ != PROCESSOR_CLASS:
        raise EmbeddingIntegrityError(
            f"processor class mismatch: expected={PROCESSOR_CLASS}, actual={processor.__class__.__name__}"
        )
    model = AutoModel.from_pretrained(snapshot_dir, local_files_only=True)
    model.eval().to(device=device, dtype=torch.float32)
    rows: list[np.ndarray] = []
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        images = []
        try:
            for _, path in batch:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
        finally:
            for image in images:
                image.close()
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None or pixel_values.dtype != torch.float32:
            raise EmbeddingIntegrityError("processor did not return float32 pixel_values")
        with torch.inference_mode():
            outputs = model(pixel_values=pixel_values.to(device=device, dtype=torch.float32))
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None or tuple(pooled.shape) != (len(batch), EMBEDDING_DIMENSION):
            raise EmbeddingIntegrityError(
                f"pooler_output shape mismatch: expected={(len(batch), EMBEDDING_DIMENSION)}, "
                f"actual={None if pooled is None else tuple(pooled.shape)}"
            )
        rows.append(pooled.detach().to(device="cpu", dtype=torch.float32).numpy())
    matrix = np.concatenate(rows, axis=0) if rows else np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
    return validate_embedding_cache(sample_ids, sample_ids, matrix, expected_dim=EMBEDDING_DIMENSION)


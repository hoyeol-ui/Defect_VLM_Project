"""Build frozen DINOv2 embeddings for the blind MPDD acquisition pool only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AL_DIR = ROOT / "scripts" / "02_active_learning"
if str(AL_DIR) not in sys.path:
    sys.path.insert(0, str(AL_DIR))

from build_visual_embedding_cache_v7 import maybe_build_dinov2  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "runs" / "mpdd_annotation_triage_protocol" / "mpdd_protocol_20260715"
DEFAULT_OUT = ROOT / "outputs" / "mpdd_visual_embeddings" / "dinov2_small_protocol_20260715"
EXPECTED_ROWS = 1056
PROHIBITED = {"is_anomaly", "anomaly_type", "official_split", "mask_path", "image_path", "image_relative_path"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    protocol = args.protocol_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    config = json.loads((protocol / "mpdd_protocol_config.json").read_text(encoding="utf-8"))
    if bool(config.get("final_test_evaluated", True)):
        raise RuntimeError("Protocol does not certify final_test_evaluated=False")
    blind_path = protocol / "mpdd_acquisition_pool_blind.csv"
    loader_path = protocol / "mpdd_acquisition_loader_private.csv"
    blind = pd.read_csv(blind_path)
    leaked = PROHIBITED.intersection(blind.columns)
    if leaked:
        raise RuntimeError(f"Blind manifest leaks prohibited columns: {sorted(leaked)}")
    loader = pd.read_csv(loader_path)
    if list(loader.columns) != ["sample_id", "image_path"]:
        raise RuntimeError("Private loader map has unexpected columns")
    if len(blind) != EXPECTED_ROWS or blind["sample_id"].duplicated().any():
        raise RuntimeError(f"Expected {EXPECTED_ROWS} unique blind samples, got {len(blind)}")
    manifest = blind.sort_values("sample_id", kind="mergesort").merge(loader, on="sample_id", how="left", validate="one_to_one")
    if manifest["image_path"].isna().any():
        raise RuntimeError("Private loader join failed")
    for path in manifest["image_path"].map(Path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest.insert(0, "embedding_index", np.arange(len(manifest), dtype=int))

    out.mkdir(parents=True, exist_ok=True)
    blind_hash = sha256(blind_path)
    existing = out / "embedding_config.json"
    if existing.exists() and (out / "embeddings.npy").exists() and (out / "embedding_manifest.csv").exists():
        old = json.loads(existing.read_text(encoding="utf-8"))
        if old.get("status") == "success" and old.get("blind_manifest_sha256") == blind_hash:
            print(f"[CACHE HIT] {out}")
            return
        raise RuntimeError(f"Existing incompatible embedding output: {out}")

    os.environ["AL_ALLOW_MODEL_DOWNLOAD"] = "0"
    os.environ["AL_DINOV2_MODEL_ID"] = "facebook/dinov2-small"
    os.environ["AL_EMBEDDING_DEVICE"] = "cuda"
    os.environ["AL_EMBEDDING_AMP"] = "1"
    os.environ["AL_EMBEDDING_BATCH_SIZE"] = str(args.batch)
    started = time.perf_counter()
    embeddings, built_manifest, extra = maybe_build_dinov2(manifest)
    if embeddings.shape != (EXPECTED_ROWS, 384) or not np.isfinite(embeddings).all():
        raise RuntimeError(f"Unexpected DINO embedding output: {embeddings.shape}")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4):
        raise RuntimeError("DINO embeddings are not L2-normalized")
    if built_manifest["sample_id"].tolist() != manifest["sample_id"].tolist():
        raise RuntimeError("Embedding builder changed sample order")
    public_manifest = manifest[["embedding_index", "sample_id", "product_category", "image_sha256"]].copy()
    np.save(out / "embeddings.npy", embeddings.astype(np.float32))
    public_manifest.to_csv(out / "embedding_manifest.csv", index=False, encoding="utf-8-sig")
    result = {
        "status": "success",
        "backend": "dinov2",
        "model_id": "facebook/dinov2-small",
        "blind_manifest_sha256": blind_hash,
        "num_embeddings": len(public_manifest),
        "embedding_shape": list(embeddings.shape),
        "source_paths_exported": False,
        "uses_gt_labels": False,
        "uses_masks_or_bboxes": False,
        "final_test_used": False,
        "runtime_seconds": time.perf_counter() - started,
        **extra,
    }
    existing.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("=" * 100)
    print("[DONE] MPDD frozen DINOv2 embeddings")
    print("GT/mask/bbox used: False")
    print("Final test used: False")
    print(f"[OUTPUT] {out}")
    print("=" * 100)


if __name__ == "__main__":
    main()


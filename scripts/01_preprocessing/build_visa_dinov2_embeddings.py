"""Build frozen DINOv2 embeddings for the blind VisA acquisition pool only."""

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


DEFAULT_PROTOCOL = ROOT / "runs" / "visa_annotation_triage_protocol" / "visa_protocol_v2_20260715"
DEFAULT_OUT = ROOT / "outputs" / "visa_visual_embeddings" / "dinov2_small_protocol_v2_20260715"
PROHIBITED_COLUMNS = {"category", "label_raw", "is_anomaly", "mask_path", "mask_relative_path", "num_connected_components", "defect_pixel_area_ratio"}


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
    blind_path = protocol / "visa_acquisition_pool_blind.csv"
    config = json.loads((protocol / "visa_protocol_config.json").read_text(encoding="utf-8"))
    if bool(config.get("final_test_evaluated", True)):
        raise RuntimeError("Protocol does not certify final_test_evaluated=False.")
    blind = pd.read_csv(blind_path)
    leaked = PROHIBITED_COLUMNS.intersection(blind.columns)
    if leaked:
        raise RuntimeError(f"Blind acquisition manifest contains prohibited GT columns: {sorted(leaked)}")
    if len(blind) != 8650 or blind["sample_id"].duplicated().any():
        raise RuntimeError(f"Expected 8650 unique acquisition samples, got {len(blind)}")
    manifest = blind.sort_values("sample_id", kind="mergesort").reset_index(drop=True).copy()
    for path in manifest["image_path"].map(Path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest.insert(0, "embedding_index", np.arange(len(manifest), dtype=int))

    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = sha256(blind_path)
    existing_config = out / "embedding_config.json"
    if existing_config.exists() and (out / "embeddings.npy").exists() and (out / "embedding_manifest.csv").exists():
        old = json.loads(existing_config.read_text(encoding="utf-8"))
        if old.get("status") == "success" and old.get("blind_manifest_sha256") == manifest_hash:
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
    if embeddings.shape != (8650, 384) or not np.isfinite(embeddings).all():
        raise RuntimeError(f"Unexpected DINO embedding output: {embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise RuntimeError("DINO embeddings are not L2 normalized.")
    np.save(out / "embeddings.npy", embeddings.astype(np.float32))
    built_manifest.to_csv(out / "embedding_manifest.csv", index=False, encoding="utf-8-sig")
    result = {
        "status": "success",
        "backend": "dinov2",
        "model_id": "facebook/dinov2-small",
        "blind_manifest": str(blind_path),
        "blind_manifest_sha256": manifest_hash,
        "num_embeddings": len(built_manifest),
        "embedding_shape": list(embeddings.shape),
        "uses_gt_labels": False,
        "uses_masks_or_bboxes": False,
        "final_test_used": False,
        "runtime_seconds": time.perf_counter() - started,
        **extra,
    }
    existing_config.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("=" * 100)
    print("[DONE] VisA frozen DINOv2 embeddings")
    print("GT/mask/bbox used: False")
    print("Final test used: False")
    print(f"[OUTPUT] {out}")
    print("=" * 100)


if __name__ == "__main__":
    main()

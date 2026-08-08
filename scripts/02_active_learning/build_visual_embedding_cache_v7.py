"""Build a visual embedding cache for V7 selection.

This script never downloads model weights unless AL_ALLOW_MODEL_DOWNLOAD=1.

Backends:
    - handcrafted: small deterministic color/texture embedding for smoke tests.
      Not a paper-facing DINO/CLIP visual embedding.
    - dinov2: DINOv2 image embedding via Hugging Face transformers.  It uses
      cached weights by default and downloads only when AL_ALLOW_MODEL_DOWNLOAD=1.

Output:
    outputs/visual_embeddings_v7/<backend>_<timestamp>/
        embeddings.npy
        embedding_manifest.csv
        embedding_config.json
        embedding_build_runtime.csv
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402
from audit_detection_pipeline_v7 import (  # noqa: E402
    average_hash,
    build_image_index,
    compute_file_sha256,
    load_priority_scores,
    resolve_image_path_fast,
)


PROJECT_ROOT = v6.PROJECT_ROOT
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "visual_embeddings_v7"
DEFAULT_DINOV2_MODEL_ID = "facebook/dinov2-small"


def parse_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom <= 1e-12] = 1.0
    return x / denom


def resolve_embedding_device() -> str:
    override = os.environ.get("AL_EMBEDDING_DEVICE")
    if override is not None and override.strip():
        return override.strip()
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_embedding_manifest(df: pd.DataFrame, image_index) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        image_path = resolve_image_path_fast(row, image_index)
        if image_path is None:
            continue
        rows.append(
            {
                "embedding_index": len(rows),
                "source_row_index": idx,
                "dataset_type": row["dataset_type"],
                "image_name": row["image_name"],
                "image_path": str(image_path),
                "sha256": compute_file_sha256(image_path),
                "ahash": average_hash(image_path),
            }
        )
    return pd.DataFrame(rows)


def cache_manifest_matches(existing_manifest: pd.DataFrame, current_manifest: pd.DataFrame) -> bool:
    cols = ["dataset_type", "image_name", "sha256"]
    if any(c not in existing_manifest.columns for c in cols) or any(c not in current_manifest.columns for c in cols):
        return False
    left = existing_manifest[cols].astype(str).sort_values(cols).reset_index(drop=True)
    right = current_manifest[cols].astype(str).sort_values(cols).reset_index(drop=True)
    return left.equals(right)


def find_reusable_cache(backend: str, current_manifest: pd.DataFrame, model_id: str | None) -> Path | None:
    if not parse_bool_env("AL_REUSE_EXISTING_EMBEDDING_CACHE", True):
        return None
    if parse_bool_env("AL_FORCE_REBUILD_EMBEDDINGS", False):
        return None
    if not OUTPUT_ROOT.exists():
        return None

    candidates = sorted(
        [p for p in OUTPUT_ROOT.glob(f"{backend}_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for cache_dir in candidates:
        config_path = cache_dir / "embedding_config.json"
        manifest_path = cache_dir / "embedding_manifest.csv"
        embeddings_path = cache_dir / "embeddings.npy"
        if not config_path.exists() or not manifest_path.exists() or not embeddings_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("status") != "success" or str(config.get("backend", "")).lower() != backend:
                continue
            if backend == "dinov2" and model_id and config.get("model_id") != model_id:
                continue
            existing_manifest = pd.read_csv(manifest_path)
            if len(existing_manifest) != len(current_manifest):
                continue
            if not cache_manifest_matches(existing_manifest, current_manifest):
                continue
            embeddings = np.load(embeddings_path, mmap_mode="r")
            if int(embeddings.shape[0]) != len(existing_manifest):
                continue
            return cache_dir
        except Exception:
            continue
    return None


def handcrafted_embedding(image_path: Path) -> np.ndarray:
    """Deterministic image-only feature for pipeline smoke tests.

    It intentionally uses no class labels, XML, or class_hint.  It is not meant
    to replace DINO/CLIP in the final paper-facing method.
    """

    with Image.open(image_path) as img:
        img = img.convert("RGB").resize((128, 128))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        edges = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0

    feats = []
    for ch in range(3):
        channel = arr[:, :, ch]
        hist, _ = np.histogram(channel, bins=16, range=(0.0, 1.0), density=True)
        feats.extend(hist.tolist())
        feats.append(float(channel.mean()))
        feats.append(float(channel.std()))
    feats.extend(
        [
            float(gray.mean()),
            float(gray.std()),
            float(edges.mean()),
            float(edges.std()),
        ]
    )
    return np.asarray(feats, dtype=np.float32)


def build_handcrafted(manifest: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = []
    for _, row in manifest.iterrows():
        image_path = Path(str(row["image_path"]))
        embeddings.append(handcrafted_embedding(image_path))
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32), manifest
    return l2_normalize(np.vstack(embeddings).astype(np.float32)), manifest


def load_rgb_images(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def maybe_build_dinov2(manifest: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict]:
    allow_download = parse_bool_env("AL_ALLOW_MODEL_DOWNLOAD", False)
    model_id = os.environ.get("AL_DINOV2_MODEL_ID", DEFAULT_DINOV2_MODEL_ID).strip() or DEFAULT_DINOV2_MODEL_ID
    batch_size = int(os.environ.get("AL_EMBEDDING_BATCH_SIZE", "32"))
    device = resolve_embedding_device()
    amp_enabled = parse_bool_env("AL_EMBEDDING_AMP", True) and device.startswith("cuda")
    local_files_only = not allow_download

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:
        raise RuntimeError(
            "DINOv2 backend requires torch and transformers in .python311. "
            "Your environment should have these, but import failed."
        ) from exc

    try:
        processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        model = AutoModel.from_pretrained(model_id, local_files_only=local_files_only)
    except Exception as exc:
        if allow_download:
            raise RuntimeError(f"Failed to load/download DINOv2 model: {model_id}. Original error: {exc}") from exc
        raise RuntimeError(
            f"DINOv2 model is not available in the local Hugging Face cache: {model_id}. "
            "If you want to download it once, run with AL_ALLOW_MODEL_DOWNLOAD=1; "
            "after that the cache will be reused by hash/model manifest."
        ) from exc

    model.to(device)
    model.eval()
    embeddings = []
    paths = [Path(str(p)) for p in manifest["image_path"].tolist()]
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = load_rgb_images(batch_paths)
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with autocast_ctx:
                outputs = model(**inputs)
                pooled = getattr(outputs, "pooler_output", None)
                if pooled is None:
                    pooled = outputs.last_hidden_state[:, 0]
                feats = pooled.float().detach().cpu().numpy()
        embeddings.append(feats.astype(np.float32))

    emb = np.vstack(embeddings).astype(np.float32) if embeddings else np.zeros((0, 0), dtype=np.float32)
    extra = {
        "model_id": model_id,
        "model_source": "huggingface_transformers",
        "device": device,
        "amp": bool(amp_enabled),
        "batch_size": batch_size,
        "local_files_only": bool(local_files_only),
        "embedding_dim": int(emb.shape[1]) if emb.ndim == 2 and emb.shape[0] else 0,
    }
    return l2_normalize(emb), manifest, extra


def write_failure_report(save_dir: Path, config: dict, error: Exception) -> None:
    config = dict(config)
    config["status"] = "failed"
    config["error"] = str(error)
    (save_dir / "embedding_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"stage": "embedding_build", "status": "failed", "error": str(error)}]).to_csv(
        save_dir / "embedding_build_runtime.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend = os.environ.get("AL_EMBEDDING_BACKEND", "handcrafted").strip().lower()

    t0 = time.perf_counter()
    priority_csv, df = load_priority_scores()
    image_index = build_image_index()
    manifest = build_embedding_manifest(df, image_index)
    model_id = os.environ.get("AL_DINOV2_MODEL_ID", DEFAULT_DINOV2_MODEL_ID).strip() or DEFAULT_DINOV2_MODEL_ID
    reusable = find_reusable_cache(backend, manifest, model_id if backend == "dinov2" else None)
    if reusable is not None:
        print("=" * 100)
        print("[CACHE HIT] Reusing visual embedding cache")
        print(f"Backend   : {backend}")
        print(f"Output dir: {reusable}")
        print("=" * 100)
        return

    save_dir = OUTPUT_ROOT / f"{backend}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "priority_csv": str(priority_csv),
        "backend": backend,
        "uses_gt_labels": False,
        "uses_class_hint": False,
        "uses_xml_bbox": False,
        "paper_facing_warning": "handcrafted backend is for smoke tests unless explicitly justified.",
        "allow_model_download": parse_bool_env("AL_ALLOW_MODEL_DOWNLOAD", False),
        "reuse_existing_embedding_cache": parse_bool_env("AL_REUSE_EXISTING_EMBEDDING_CACHE", True),
        "force_rebuild_embeddings": parse_bool_env("AL_FORCE_REBUILD_EMBEDDINGS", False),
        "num_manifest_images": int(len(manifest)),
    }

    try:
        if backend == "handcrafted":
            embeddings, manifest = build_handcrafted(manifest)
            config["embedding_dim"] = int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] else 0
        elif backend == "dinov2":
            embeddings, manifest, extra = maybe_build_dinov2(manifest)
            config.update(extra)
        else:
            raise ValueError(f"Unknown AL_EMBEDDING_BACKEND: {backend}")

        np.save(save_dir / "embeddings.npy", embeddings)
        manifest.to_csv(save_dir / "embedding_manifest.csv", index=False, encoding="utf-8-sig")
        config["status"] = "success"
        config["num_embeddings"] = int(len(manifest))
        config["runtime_sec"] = time.perf_counter() - t0
        (save_dir / "embedding_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(
            [
                {
                    "stage": "embedding_build",
                    "backend": backend,
                    "num_embeddings": len(manifest),
                    "runtime_sec": config["runtime_sec"],
                    "status": "success",
                }
            ]
        ).to_csv(save_dir / "embedding_build_runtime.csv", index=False, encoding="utf-8-sig")
    except Exception as exc:
        write_failure_report(save_dir, config, exc)
        print(f"[ERROR] {exc}")
        print(f"[REPORT] {save_dir}")
        raise

    print("=" * 100)
    print("[DONE] Visual embedding cache built")
    print(f"Backend   : {backend}")
    print(f"Output dir: {save_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()

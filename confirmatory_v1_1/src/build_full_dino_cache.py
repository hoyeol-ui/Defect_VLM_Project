"""Build one sealed full-acquisition DINO cache without exposing GT or outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import psutil


HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[2]
V1_SRC = WORKTREE / "confirmatory_v1" / "src"
sys.path.insert(0, str(V1_SRC))

from confirmatory_v1.frozen_dino_ff_v1 import extract_embeddings  # noqa: E402


EXPECTED = {"GC10-DET": 1605, "NEU-DET": 1259}
DATA_ROOT = Path("C:/Users/user/Desktop/vlm/Defect_VLM_Project/data")
SNAPSHOT = Path("C:/Users/user/.cache/huggingface/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056")
REGISTRY = WORKTREE / "confirmatory_v1" / "data_freeze" / "private_source_registry.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ResourceSampler:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.peak_rss = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        process = psutil.Process(os.getpid())
        while not self.stop.wait(0.25):
            try:
                total = process.memory_info().rss + sum(child.memory_info().rss for child in process.children(recursive=True))
                self.peak_rss = max(self.peak_rss, total)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def __enter__(self) -> "ResourceSampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(EXPECTED), required=True)
    parser.add_argument("--cache-label", choices=["A", "B"], required=True)
    args = parser.parse_args()

    with REGISTRY.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == args.dataset and row["split"] == "acquisition"]
    rows.sort(key=lambda row: row["sample_id"])
    if len(rows) != EXPECTED[args.dataset] or len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError("acquisition registry count/identity mismatch")
    items = [(row["sample_id"], DATA_ROOT / row["relative_image_path"]) for row in rows]
    if any(not path.is_file() for _, path in items):
        raise FileNotFoundError("one or more acquisition images are missing")

    output = WORKTREE / "confirmatory_v1_1" / "cache_seals" / args.dataset / f"cache_{args.cache_label}"
    output.mkdir(parents=True, exist_ok=False)

    import torch

    torch.cuda.set_device("cuda:0")
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats("cuda:0")
    started = time.perf_counter()
    with ResourceSampler() as sampler:
        cache = extract_embeddings(items, snapshot_dir=SNAPSHOT, device="cuda:0", batch_size=32)
    runtime = time.perf_counter() - started

    ids = np.asarray(cache.sample_ids, dtype="<U64")
    embeddings = np.asarray(cache.normalized, dtype=np.float32)
    if ids.shape != (EXPECTED[args.dataset],) or embeddings.shape != (EXPECTED[args.dataset], 384):
        raise RuntimeError("cache shape mismatch")
    if embeddings.dtype != np.float32 or not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) <= 1e-12):
        raise RuntimeError("cache numeric integrity failure")
    np.save(output / "sample_ids.npy", ids, allow_pickle=False)
    np.save(output / "embeddings.npy", embeddings, allow_pickle=False)
    (output / "ordered_manifest.txt").write_text("\n".join(cache.sample_ids) + "\n", encoding="ascii", newline="\n")

    files = [output / "sample_ids.npy", output / "embeddings.npy", output / "ordered_manifest.txt"]
    hashes = {path.name: sha256(path) for path in files}
    content_sha = hashlib.sha256("".join(hashes[name] for name in sorted(hashes)).encode("ascii")).hexdigest()
    metadata = {
        "dataset": args.dataset,
        "cache_label": args.cache_label,
        "status": "COMPLETE",
        "count": len(ids),
        "dimension": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "runtime_seconds": runtime,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "peak_host_rss_bytes": int(sampler.peak_rss),
        "artifact_bytes": int(sum(path.stat().st_size for path in files)),
        "file_sha256": hashes,
        "cache_content_sha256": content_sha,
        "snapshot_revision": SNAPSHOT.name,
        "batch_size": 32,
        "precision": "FP32_AUTocast_OFF_TF32_OFF",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PHASE2B_OPERATIONAL_RESULT " + json.dumps({key: metadata[key] for key in ("dataset", "cache_label", "status", "count", "runtime_seconds", "peak_gpu_reserved_bytes", "peak_host_rss_bytes", "artifact_bytes", "cache_content_sha256")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

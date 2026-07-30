"""Shared, frozen helpers for the KSDD2 G5 bounded learner screen."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "config.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def run_root(config: dict[str, Any]) -> Path:
    return (ROOT / config["outputs"]["run_root"]).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash tensor values without depending on torch.save container metadata."""
    chunks: list[bytes] = []
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        chunks.extend(
            (
                name.encode("utf-8"),
                str(tensor.dtype).encode("ascii"),
                json.dumps(list(tensor.shape)).encode("ascii"),
                tensor.numpy().tobytes(order="C"),
            )
        )
    return sha256_bytes(chunks)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def seal_file(path: Path) -> Path:
    seal = path.with_suffix(path.suffix + ".sha256")
    seal.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="ascii")
    return seal


def verify_seal(path: Path) -> bool:
    seal = path.with_suffix(path.suffix + ".sha256")
    return (
        path.exists()
        and seal.exists()
        and seal.read_text(encoding="ascii").split()[0].lower() == sha256_file(path)
    )


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args],
        text=True,
        encoding="utf-8",
    ).strip()


def require_clean_git() -> str:
    status = git_output("status", "--porcelain")
    if status:
        raise RuntimeError(
            "G5 freeze requires a clean worktree; commit the reviewed implementation first."
        )
    return git_output("rev-parse", "HEAD")


def resolve_source(config: dict[str, Any], override: str | None = None) -> Path:
    path = Path(override or config["upstream"]["default_source_run_root"])
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def resolve_data_path(relative: str) -> Path:
    path = Path(relative)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if "test" in {part.lower() for part in resolved.parts}:
        raise RuntimeError(f"Official-test path is forbidden in G5: {resolved}")
    return resolved


def require_train_row(row: dict[str, str]) -> None:
    if row.get("official_split") != "train":
        raise RuntimeError(f"Non-train row entered G5: {row.get('sample_id')}")
    for key in ("image_path", "posthoc_mask_path"):
        value = row.get(key, "")
        if not value:
            raise RuntimeError(f"Missing {key} for {row.get('sample_id')}")
        resolve_data_path(value)


def set_deterministic(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_from_environment() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen G5 execution requires the reviewed CUDA environment.")
    return torch.device("cuda")


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def verify_frozen_execution(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    out = run_root(config)
    freeze_path = out / config["outputs"]["freeze_manifest"]
    if not verify_seal(freeze_path):
        raise RuntimeError("Missing or invalid G5 freeze manifest")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    current_commit = git_output("rev-parse", "HEAD")
    if current_commit != freeze["git"]["commit"]:
        raise RuntimeError("Current Git commit differs from frozen G5 commit")
    if git_output("status", "--porcelain"):
        raise RuntimeError("G5 execution refuses a dirty worktree")
    for item in freeze["frozen_files"]:
        path = Path(item["path"])
        path = path if path.is_absolute() else ROOT / path
        if not path.exists() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Frozen file changed or disappeared: {path}")
    return freeze_path, freeze


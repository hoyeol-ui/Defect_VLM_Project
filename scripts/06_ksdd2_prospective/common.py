"""Shared utilities for the KSDD2 prospective authorization branch."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "config.yaml"
PROTOCOL_PATH = HERE / "protocol.md"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def run_root(config: dict[str, Any]) -> Path:
    return project_path(config["outputs"]["run_root"])


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash_key(salt: str, sample_id: str) -> str:
    return sha256_text(f"{salt}:{sample_id}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def seal_file(path: Path) -> Path:
    seal = path.with_suffix(path.suffix + ".sha256")
    seal.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="ascii")
    return seal


def verify_seal(path: Path) -> bool:
    seal = path.with_suffix(path.suffix + ".sha256")
    if not seal.exists():
        return False
    expected = seal.read_text(encoding="ascii").split()[0].lower()
    return expected == sha256_file(path).lower()


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()

    status = run("status", "--porcelain")
    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def environment_snapshot() -> dict[str, Any]:
    import anomalib
    import cv2
    import numpy
    import pandas
    import scipy
    import sklearn
    import torch
    import torchvision

    return {
        "python": os.sys.version,
        "anomalib": anomalib.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "opencv": cv2.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def state_dict_sha256(state_dict: dict[str, Any]) -> str:
    """Content hash a tensor state_dict without serialization metadata."""

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(value) for value in tensor.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def patchcore_backbone_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    """Load the frozen pretrained extractor and fingerprint the actual parameters."""

    from anomalib.models import Patchcore

    params = config["patchcore"]
    model = Patchcore(
        backbone=str(params["backbone"]),
        layers=tuple(params["layers"]),
        pre_trained=bool(params["pretrained"]),
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    state = model.model.feature_extractor.state_dict()
    fingerprint = {
        "backbone": params["backbone"],
        "layers": list(params["layers"]),
        "pretrained": bool(params["pretrained"]),
        "pretrained_source": params["pretrained_source"],
        "algorithm": params["backbone_fingerprint_algorithm"],
        "state_dict_sha256": state_dict_sha256(state),
        "tensor_count": len(state),
    }
    expected = str(params["expected_backbone_state_sha256"])
    if fingerprint["state_dict_sha256"] != expected:
        raise RuntimeError(
            "Loaded pretrained backbone differs from config.yaml: "
            f"{fingerprint['state_dict_sha256']} != {expected}"
        )
    return fingerprint


def assert_no_label_columns(rows: list[dict[str, Any]], *, artifact: str) -> None:
    forbidden = ("label", "mask", "defect", "bbox", "class", "annotation", "ground_truth", "gt_")
    if not rows:
        raise ValueError(f"{artifact} is empty")
    columns = {str(key).lower() for key in rows[0]}
    leaked = sorted(column for column in columns if any(token in column for token in forbidden))
    if leaked:
        raise ValueError(f"{artifact} contains prohibited post-hoc columns: {leaked}")


def require_official_freeze(config: dict[str, Any]) -> dict[str, Any]:
    path = run_root(config) / config["outputs"]["freeze_manifest"]
    if not path.exists():
        raise RuntimeError("Official freeze is missing. Run `run.py freeze` from a clean commit first.")
    if not verify_seal(path):
        raise RuntimeError("Freeze manifest seal is missing or invalid.")
    freeze = read_json(path)
    if not freeze.get("official_freeze", False):
        raise RuntimeError("Development freeze cannot authorize prospective scoring.")
    verify_frozen_hashes(config, freeze)
    return freeze


def freeze_targets(config: dict[str, Any]) -> list[Path]:
    out = run_root(config)
    paths = [CONFIG_PATH]
    paths.extend(sorted(HERE.glob("*.md")))
    paths.extend(sorted(HERE.glob("requirements*.txt")))
    paths.extend(sorted(HERE.glob("*.py")))
    for key in (
        "blind_manifest",
        "acquisition_manifest",
        "development_manifest",
        "final_locked_manifest",
    ):
        paths.append(out / config["outputs"][key])
    paths.extend(sorted((out / "initial_sets").glob("initial_seed*.csv")))
    return sorted(set(paths), key=lambda path: path.as_posix())


def build_freeze_manifest(config: dict[str, Any], *, allow_dirty: bool) -> dict[str, Any]:
    snapshot = git_snapshot()
    if snapshot["dirty"] and not allow_dirty:
        raise RuntimeError(
            "Official freeze refused: Git worktree is dirty. Commit the intended evidence and "
            "protocol changes, then rerun without --allow-dirty."
        )
    targets = freeze_targets(config)
    missing = [str(path) for path in targets if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Freeze targets missing: {missing}")
    return {
        "protocol_id": config["protocol"]["id"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_freeze": not snapshot["dirty"],
        "development_only": snapshot["dirty"],
        "git": snapshot,
        "environment": environment_snapshot(),
        "model": patchcore_backbone_fingerprint(config),
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in targets
        ],
        "locks": {
            "development_scores_viewed_before_freeze": False,
            "query_masks_viewed_before_selection_seal": False,
            "official_test_image_content_read": False,
            "official_test_mask_content_read": False,
            "detector_training_performed": False,
            "official_test_used": False,
        },
    }


def verify_frozen_hashes(config: dict[str, Any], freeze: dict[str, Any] | None = None) -> None:
    out = run_root(config)
    freeze = freeze or read_json(out / config["outputs"]["freeze_manifest"])
    failures: list[str] = []
    for item in freeze["files"]:
        path = project_path(item["path"])
        if not path.exists():
            failures.append(f"missing:{item['path']}")
        elif sha256_file(path) != item["sha256"]:
            failures.append(f"hash:{item['path']}")
    if failures:
        raise RuntimeError(f"Frozen artifact verification failed: {failures}")


def list_outcome_artifacts(config: dict[str, Any]) -> list[Path]:
    out = run_root(config)
    patterns = (
        "g1_*.csv",
        "g1_*.json",
        "selection_scores_*.csv",
        "selected_*.csv",
        "posthoc_*.csv",
        "g2_g4_*.csv",
        "g2_g4_*.json",
        config["outputs"]["authorization_decision"],
    )
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(out.rglob(pattern))
    return sorted(set(paths))


def ensure_pre_freeze_clean(config: dict[str, Any]) -> None:
    outcomes = list_outcome_artifacts(config)
    if outcomes:
        relative = [path.relative_to(ROOT).as_posix() for path in outcomes]
        raise RuntimeError(f"Outcome artifacts already exist; refusing to re-freeze: {relative}")

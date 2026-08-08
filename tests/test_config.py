from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from defect_al.config import load_frozen_run_config
from defect_al.data.identity import compute_file_sha256


def _create_manifests(base: Path) -> dict[str, Path]:
    manifest_dir = base / "manifests"
    manifest_dir.mkdir(parents=True)
    paths = {}
    for split in ("train", "validation", "test"):
        path = manifest_dir / f"{split}.csv"
        path.write_text(
            f"dataset_name,relative_path,image_sha256\nGC10,images/{split}.png,{split[0] * 64}\n",
            encoding="utf-8",
        )
        paths[split] = path
    return paths


def _document(paths: dict[str, Path], config_dir: Path) -> dict:
    return {
        "run_id": "gc10_dev_confirm_20260715",
        "manifests": {
            split: {
                "path": path.relative_to(config_dir).as_posix(),
                "manifest_sha256": compute_file_sha256(path),
            }
            for split, path in paths.items()
        },
    }


def _write_json(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def _write_yaml(path: Path, document: dict) -> Path:
    lines = [f"run_id: {document['run_id']}", "", "manifests:"]
    for split in ("train", "validation", "test"):
        reference = document["manifests"][split]
        lines.extend(
            [
                f"  {split}:",
                f"    path: {reference['path']}",
                f"    manifest_sha256: \"{reference['manifest_sha256']}\"",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_valid_yaml_config_loads(tmp_path: Path) -> None:
    paths = _create_manifests(tmp_path)
    config = load_frozen_run_config(_write_yaml(tmp_path / "frozen.yaml", _document(paths, tmp_path)))
    assert config.run_id == "gc10_dev_confirm_20260715"
    assert config.manifests["train"].path == paths["train"].resolve()


def test_valid_json_config_loads(tmp_path: Path) -> None:
    paths = _create_manifests(tmp_path)
    config = load_frozen_run_config(_write_json(tmp_path / "frozen.json", _document(paths, tmp_path)))
    assert tuple(config.manifests) == ("train", "validation", "test")


def test_manifest_sha_mismatch_fails_fast(tmp_path: Path) -> None:
    paths = _create_manifests(tmp_path)
    document = _document(paths, tmp_path)
    document["manifests"]["train"]["manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA-256 mismatch.*train"):
        load_frozen_run_config(_write_json(tmp_path / "frozen.json", document))


def test_missing_run_id_is_rejected(tmp_path: Path) -> None:
    paths = _create_manifests(tmp_path)
    document = _document(paths, tmp_path)
    document.pop("run_id")
    with pytest.raises(RuntimeError, match="run_id"):
        load_frozen_run_config(_write_json(tmp_path / "frozen.json", document))


def test_latest_run_id_is_rejected(tmp_path: Path) -> None:
    paths = _create_manifests(tmp_path)
    document = _document(paths, tmp_path)
    document["run_id"] = "latest"
    with pytest.raises(RuntimeError, match="implicit-discovery value"):
        load_frozen_run_config(_write_json(tmp_path / "frozen.json", document))


def test_missing_required_split_is_rejected(tmp_path: Path) -> None:
    paths = _create_manifests(tmp_path)
    document = _document(paths, tmp_path)
    document["manifests"].pop("test")
    with pytest.raises(RuntimeError, match="missing required manifest split.*test"):
        load_frozen_run_config(_write_json(tmp_path / "frozen.json", document))


def test_unknown_split_is_rejected(tmp_path: Path) -> None:
    paths = _create_manifests(tmp_path)
    document = _document(paths, tmp_path)
    document["manifests"]["dev"] = document["manifests"]["validation"].copy()
    with pytest.raises(RuntimeError, match="unknown manifest split.*dev"):
        load_frozen_run_config(_write_json(tmp_path / "frozen.json", document))


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    paths = _create_manifests(tmp_path)
    document = _document(paths, tmp_path)
    document["notes"] = "not part of the frozen schema"
    with pytest.raises(RuntimeError, match="unknown key.*notes"):
        load_frozen_run_config(_write_json(tmp_path / "frozen.json", document))


def test_glob_manifest_path_is_rejected(tmp_path: Path) -> None:
    paths = _create_manifests(tmp_path)
    document = _document(paths, tmp_path)
    document["manifests"]["train"]["path"] = "manifests/train*.csv"
    with pytest.raises(RuntimeError, match="glob patterns are forbidden"):
        load_frozen_run_config(_write_json(tmp_path / "frozen.json", document))


def test_nonexistent_manifest_is_rejected(tmp_path: Path) -> None:
    paths = _create_manifests(tmp_path)
    document = _document(paths, tmp_path)
    document["manifests"]["train"]["path"] = "manifests/missing.csv"
    with pytest.raises(RuntimeError, match="does not exist.*missing.csv"):
        load_frozen_run_config(_write_json(tmp_path / "frozen.json", document))


@pytest.mark.parametrize("forbidden_key", ["latest_run_dir", "mtime", "auto_discover"])
def test_implicit_discovery_keys_are_rejected(tmp_path: Path, forbidden_key: str) -> None:
    paths = _create_manifests(tmp_path)
    document = _document(paths, tmp_path)
    document[forbidden_key] = "anything"
    with pytest.raises(RuntimeError, match="forbidden implicit-discovery key"):
        load_frozen_run_config(_write_json(tmp_path / "frozen.json", document))


def test_relative_paths_resolve_from_config_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "nested" / "config"
    config_dir.mkdir(parents=True)
    paths = _create_manifests(config_dir)
    config_path = _write_json(config_dir / "frozen.json", _document(paths, config_dir))
    old_cwd_manifest = tmp_path / "manifests" / "train.csv"
    assert not old_cwd_manifest.exists()
    config = load_frozen_run_config(config_path)
    assert config.manifests["train"].path == paths["train"].resolve()


def test_config_and_manifest_mapping_are_immutable(tmp_path: Path) -> None:
    paths = _create_manifests(tmp_path)
    config = load_frozen_run_config(_write_json(tmp_path / "frozen.json", _document(paths, tmp_path)))
    with pytest.raises(TypeError):
        config.manifests["train"] = config.manifests["test"]  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        config.run_id = "changed"  # type: ignore[misc]


def test_config_format_errors_are_clear_runtime_errors(tmp_path: Path) -> None:
    unsupported = tmp_path / "frozen.toml"
    unsupported.write_text("run_id = 'x'", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unsupported frozen-run config format"):
        load_frozen_run_config(unsupported)

    invalid_json = tmp_path / "broken.json"
    invalid_json.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid JSON frozen-run config"):
        load_frozen_run_config(invalid_json)


def test_non_object_config_is_rejected(tmp_path: Path) -> None:
    config_path = _write_json(tmp_path / "frozen.json", ["not", "an", "object"])
    with pytest.raises(RuntimeError, match="root must be an object/dict"):
        load_frozen_run_config(config_path)


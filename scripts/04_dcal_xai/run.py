"""Run the staged GC10 detector-coupled AL and grounded-XAI experiment.

Stages:
  audit   - protocol/manifests/embeddings only; no model execution
  acquire - warm-start training, pool scoring, frozen selection, post-hoc gate
  confirm - downstream development-only detector comparison after gate PASS

The final-test manifest is intentionally never read by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from core import (  # noqa: E402
    DifficultyEvidence,
    compute_difficulty,
    grounded_explanation,
    hybrid_uncertainty_diversity_select,
    stable_top_k,
    unflip_xyxy,
)


DEFAULT_CONFIG = HERE / "config.json"
EXPECTED_BLIND_COLUMNS = ["sample_id", "image_sha256", "phash64"]
RANDOM = "Random"
DIFFICULTY = "DetectorDifficulty"
HYBRID = "DetectorDifficultyDiversity"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_seed_override(raw: str | None, configured: list[int]) -> list[int]:
    if raw is None:
        return [int(value) for value in configured]
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("--seeds must contain unique comma-separated integers")
    unknown = sorted(set(values) - set(int(value) for value in configured))
    if unknown:
        raise ValueError(f"Seeds outside frozen config: {unknown}")
    return values


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    resolved = path.expanduser().resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    expected_strategies = [RANDOM, DIFFICULTY, HYBRID]
    if config.get("strategies") != expected_strategies:
        raise RuntimeError(f"Frozen strategies must be {expected_strategies}")
    if config.get("primary_strategy") != HYBRID:
        raise RuntimeError(f"Frozen primary strategy must be {HYBRID}")
    if bool(config.get("safety", {}).get("final_test_used", True)):
        raise RuntimeError("Config final-test safety flag failed")
    if bool(config.get("safety", {}).get("selector_reads_gt", True)):
        raise RuntimeError("Selector must not read GT")
    if bool(config.get("safety", {}).get("selector_reads_xml", True)):
        raise RuntimeError("Selector must not read XML")
    if bool(config.get("safety", {}).get("vlm_used_for_selection", True)):
        raise RuntimeError("VLM must not be used for selection")
    return config, resolved


def validate_protocol_and_embeddings(config: dict[str, Any]) -> dict[str, Any]:
    protocol = resolve_project_path(config["protocol_dir"])
    embedding_dir = resolve_project_path(config["embedding_dir"])
    model = resolve_project_path(config["model"])
    protocol_cfg_path = protocol / "gc10_protocol_config.json"
    embedding_cfg_path = embedding_dir / "embedding_config.json"
    protocol_cfg = json.loads(protocol_cfg_path.read_text(encoding="utf-8"))
    embedding_cfg = json.loads(embedding_cfg_path.read_text(encoding="utf-8"))
    if bool(protocol_cfg.get("final_test_evaluated", True)):
        raise RuntimeError("Protocol reports final-test evaluation")
    if bool(embedding_cfg.get("final_test_used", True)):
        raise RuntimeError("Embedding config reports final-test use")
    if protocol_cfg.get("split_sizes", {}).get("acquisition") != 1836:
        raise RuntimeError("Unexpected acquisition size")
    if protocol_cfg.get("split_sizes", {}).get("development") != 232:
        raise RuntimeError("Unexpected development size")
    if not model.exists():
        raise FileNotFoundError(model)

    blind = pd.read_csv(protocol / "gc10_acquisition_pool_blind.csv").sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if blind.columns.tolist() != EXPECTED_BLIND_COLUMNS or len(blind) != 1836 or blind["sample_id"].duplicated().any():
        raise RuntimeError("Blind acquisition manifest integrity failure")
    loader = pd.read_csv(protocol / "gc10_acquisition_loader_private.csv")
    if loader.columns.tolist() != ["sample_id", "image_path"] or loader["sample_id"].duplicated().any():
        raise RuntimeError("Private loader integrity failure")
    pool = blind.merge(loader, on="sample_id", how="left", validate="one_to_one")
    missing_images = [value for value in pool["image_path"] if not Path(str(value)).exists()]
    if missing_images:
        raise FileNotFoundError(f"Missing acquisition images: {len(missing_images)}")

    embedding_manifest = pd.read_csv(embedding_dir / "embedding_manifest.csv")
    embeddings = np.load(embedding_dir / "embeddings.npy", mmap_mode="r")
    if embedding_manifest["sample_id"].astype(str).tolist() != blind["sample_id"].astype(str).tolist():
        raise RuntimeError("Blind and embedding manifests are misaligned")
    if embeddings.shape != (1836, 384) or not np.isfinite(np.asarray(embeddings[:10])).all():
        raise RuntimeError(f"Unexpected embedding shape/content: {embeddings.shape}")
    return {
        "protocol": protocol,
        "embedding_dir": embedding_dir,
        "model": model,
        "protocol_config": protocol_cfg,
        "embedding_config": embedding_cfg,
        "pool": pool,
        "blind": blind,
        "embedding_manifest": embedding_manifest,
        "embeddings": embeddings,
        "model_sha256": sha256(model),
        "protocol_config_sha256": sha256(protocol_cfg_path),
        "embedding_config_sha256": sha256(embedding_cfg_path),
    }


def reconstruct_initial(blind: pd.DataFrame, acquisition_seed: int, size: int) -> list[str]:
    indices = np.arange(len(blind), dtype=int)
    chosen = pd.DataFrame({"idx": indices}).sample(
        n=size,
        random_state=acquisition_seed + 999,
        replace=False,
    )["idx"].astype(int).tolist()
    return blind.iloc[chosen]["sample_id"].astype(str).tolist()


def ensure_output_identity(out: Path, config_path: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    identity_path = out / "frozen_identity.json"
    current = {"config_path": str(config_path), "config_sha256": sha256(config_path)}
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if previous != current:
            raise RuntimeError(f"Refusing to mix configs in {out}")
    else:
        json_write(identity_path, current)


def run_audit(config: dict[str, Any], config_path: Path, out: Path, seeds: list[int]) -> Path:
    context = validate_protocol_and_embeddings(config)
    ensure_output_identity(out, config_path)
    initial_sets = {
        str(seed): reconstruct_initial(context["blind"], seed, int(config["initial_size"]))
        for seed in seeds
    }
    payload = {
        "status": "PASS",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256(config_path),
        "core_sha256": sha256(HERE / "core.py"),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "protocol_config_sha256": context["protocol_config_sha256"],
        "embedding_config_sha256": context["embedding_config_sha256"],
        "model": str(context["model"]),
        "model_sha256": context["model_sha256"],
        "acquisition_images": len(context["pool"]),
        "development_images": int(context["protocol_config"]["split_sizes"]["development"]),
        "embedding_shape": list(context["embeddings"].shape),
        "seeds": seeds,
        "initial_size": int(config["initial_size"]),
        "query_size": int(config["query_size"]),
        "initial_sets": initial_sets,
        "selector_reads_gt": False,
        "selector_reads_xml": False,
        "vlm_used_for_selection": False,
        "final_test_used": False,
        "note": "The final-test manifest was not opened.",
    }
    path = out / "audit.json"
    json_write(path, payload)
    return path


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def write_yolo_label(path: Path, boxes: pd.DataFrame, width: int, height: int) -> None:
    lines: list[str] = []
    for row in boxes.sort_values("object_index").itertuples(index=False):
        xc = (float(row.x_min) + float(row.x_max)) / (2.0 * width)
        yc = (float(row.y_min) + float(row.y_max)) / (2.0 * height)
        bw = (float(row.x_max) - float(row.x_min)) / width
        bh = (float(row.y_max) - float(row.y_min)) / height
        values = [xc, yc, bw, bh]
        if bw <= 0 or bh <= 0 or not all(0.0 <= value <= 1.0 for value in values):
            raise RuntimeError(f"Invalid box for {row.sample_id}: {values}")
        lines.append(f"{int(row.class_id) - 1} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
    if not lines:
        raise RuntimeError(f"No label objects for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def build_training_yaml(
    *,
    protocol: Path,
    out: Path,
    key: str,
    train_ids: list[str],
    class_names: list[str],
    include_development: bool,
) -> tuple[Path, dict[str, Any]]:
    """Materialize labels for committed labeled IDs only."""

    cache = out / "yolo_cache"
    acquisition = pd.read_csv(protocol / "gc10_acquisition_pool_gt_audit.csv")
    acquisition = acquisition[acquisition["sample_id"].isin(train_ids)].copy()
    if set(acquisition["sample_id"].astype(str)) != set(train_ids):
        raise RuntimeError(f"Selected acquisition GT mismatch for {key}")
    acquisition_boxes = pd.read_csv(protocol / "gc10_acquisition_bbox_gt_audit.csv")
    acquisition_boxes = acquisition_boxes[acquisition_boxes["sample_id"].isin(train_ids)].copy()
    grouped = {str(sample_id): frame for sample_id, frame in acquisition_boxes.groupby("sample_id")}
    modes: dict[str, int] = {}
    train_paths: list[str] = []
    for row in acquisition.itertuples(index=False):
        sample_id = str(row.sample_id)
        target = cache / "images" / key / f"{sample_id}.jpg"
        mode = link_or_copy(Path(str(row.image_path)), target)
        modes[mode] = modes.get(mode, 0) + 1
        write_yolo_label(cache / "labels" / key / f"{sample_id}.txt", grouped[sample_id], int(row.width), int(row.height))
        train_paths.append(target.resolve().as_posix())
    train_paths = sorted(train_paths)
    manifest_dir = out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    train_txt = manifest_dir / f"{key}_train.txt"
    train_txt.write_text("\n".join(train_paths) + "\n", encoding="utf-8")

    val_txt = train_txt
    if include_development:
        development = pd.read_csv(protocol / "gc10_development_eval.csv")
        development_boxes = pd.read_csv(protocol / "gc10_development_bbox_gt.csv")
        dev_grouped = {str(sample_id): frame for sample_id, frame in development_boxes.groupby("sample_id")}
        val_paths: list[str] = []
        for row in development.itertuples(index=False):
            sample_id = str(row.sample_id)
            target = cache / "images" / "development" / f"{sample_id}.jpg"
            mode = link_or_copy(Path(str(row.image_path)), target)
            modes[mode] = modes.get(mode, 0) + 1
            write_yolo_label(
                cache / "labels" / "development" / f"{sample_id}.txt",
                dev_grouped[sample_id],
                int(row.width),
                int(row.height),
            )
            val_paths.append(target.resolve().as_posix())
        val_txt = manifest_dir / "development.txt"
        val_txt.write_text("\n".join(sorted(val_paths)) + "\n", encoding="utf-8")

    yaml_path = manifest_dir / f"{key}.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(cache.resolve()),
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "names": {index: name for index, name in enumerate(class_names)},
    }, sort_keys=False), encoding="utf-8")
    return yaml_path, {"link_modes": modes, "train_images": len(train_ids), "development_included": include_development}


def train_checkpoint(
    *,
    model_path: Path,
    yaml_path: Path,
    out: Path,
    name: str,
    training_seed: int,
    training: dict[str, Any],
) -> Path:
    from ultralytics import YOLO

    train_dir = out / "train_runs" / name
    last = train_dir / "weights" / "last.pt"
    if last.exists():
        print(f"[TRAIN CACHE] {name}", flush=True)
        return last
    print(f"[TRAIN] {name}", flush=True)
    model = YOLO(str(model_path))
    model.train(
        data=str(yaml_path),
        epochs=int(training["epochs"]),
        imgsz=int(training["imgsz"]),
        batch=int(training["batch"]),
        workers=int(training["workers"]),
        cache=False,
        device=str(training["device"]),
        project=str(out / "train_runs"),
        name=name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=False,
        seed=int(training_seed),
        deterministic=True,
        plots=False,
        val=False,
        save_json=False,
    )
    if not last.exists():
        raise FileNotFoundError(f"Training did not create {last}")
    return last


def result_arrays(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 4)), np.array([], dtype=float), np.array([], dtype=int)
    return (
        boxes.xyxy.detach().cpu().numpy().astype(np.float64),
        boxes.conf.detach().cpu().numpy().astype(np.float64),
        boxes.cls.detach().cpu().numpy().astype(int),
    )


def score_pool(
    *,
    checkpoint: Path,
    candidates: pd.DataFrame,
    score_path: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    metadata_path = score_path.with_suffix(".meta.json")
    expected_metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "difficulty": config["difficulty"],
        "imgsz": int(config["training"]["imgsz"]),
        "final_test_used": False,
    }
    if score_path.exists():
        if not metadata_path.exists():
            raise RuntimeError(f"Score cache lacks metadata: {score_path}")
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached_metadata != expected_metadata:
            raise RuntimeError(f"Score cache/checkpoint mismatch: {score_path}")
        cached = pd.read_csv(score_path)
        if set(cached["sample_id"].astype(str)) != set(candidates["sample_id"].astype(str)):
            raise RuntimeError(f"Score cache does not match candidates: {score_path}")
        print(f"[SCORE CACHE] {score_path.name}", flush=True)
        return cached

    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    difficulty_cfg = config["difficulty"]
    training = config["training"]
    rows: list[dict[str, Any]] = []
    batch_size = max(1, int(training["batch"]) // 2)
    ordered = candidates.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    for start in range(0, len(ordered), batch_size):
        batch = ordered.iloc[start:start + batch_size]
        inputs: list[np.ndarray] = []
        widths: list[int] = []
        for row in batch.itertuples(index=False):
            image = cv2.imread(str(row.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read {row.image_path}")
            widths.append(int(image.shape[1]))
            inputs.extend([image, cv2.flip(image, 1)])
        results = model.predict(
            source=inputs,
            imgsz=int(training["imgsz"]),
            conf=float(difficulty_cfg["prediction_confidence"]),
            iou=float(difficulty_cfg["prediction_iou"]),
            max_det=int(difficulty_cfg["max_detections"]),
            device=str(training["device"]),
            verbose=False,
        )
        if len(results) != 2 * len(batch):
            raise RuntimeError("Unexpected prediction batch cardinality")
        for local_index, row in enumerate(batch.itertuples(index=False)):
            ob, oc, ok = result_arrays(results[2 * local_index])
            fb_raw, fc, fk = result_arrays(results[2 * local_index + 1])
            fb = unflip_xyxy(fb_raw, widths[local_index])
            evidence = compute_difficulty(
                ob, oc, ok, fb, fc, fk,
                num_classes=len(config["class_names"]),
                weights={name: float(value) for name, value in difficulty_cfg["weights"].items()},
            )
            payload = asdict(evidence)
            rows.append({
                "sample_id": str(row.sample_id),
                **{key: value for key, value in payload.items() if key not in {"original_classes", "flipped_classes"}},
                "original_classes": "|".join(str(value) for value in evidence.original_classes),
                "flipped_classes": "|".join(str(value) for value in evidence.flipped_classes),
            })
        print(f"[SCORE] {min(start + len(batch), len(ordered))}/{len(ordered)}", flush=True)
    frame = pd.DataFrame(rows).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(score_path, index=False, encoding="utf-8-sig")
    json_write(metadata_path, expected_metadata)
    return frame


def evidence_from_row(row: pd.Series) -> DifficultyEvidence:
    def classes(value: Any) -> tuple[int, ...]:
        if pd.isna(value) or str(value).strip() == "":
            return ()
        return tuple(int(part) for part in str(value).split("|") if part != "")

    return DifficultyEvidence(
        difficulty=float(row["difficulty"]),
        confidence_deficit=float(row["confidence_deficit"]),
        localization_instability=float(row["localization_instability"]),
        class_instability=float(row["class_instability"]),
        count_instability=float(row["count_instability"]),
        original_count=int(row["original_count"]),
        flipped_count=int(row["flipped_count"]),
        original_max_confidence=float(row["original_max_confidence"]),
        flipped_max_confidence=float(row["flipped_max_confidence"]),
        original_classes=classes(row["original_classes"]),
        flipped_classes=classes(row["flipped_classes"]),
    )


def choose_queries(
    *,
    scores: pd.DataFrame,
    acquisition_seed: int,
    initial_ids: list[str],
    embedding_lookup: dict[str, int],
    embeddings: np.ndarray,
    config: dict[str, Any],
) -> dict[str, list[str]]:
    ordered = scores.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    ids = ordered["sample_id"].astype(str).tolist()
    values = ordered["difficulty"].to_numpy(float)
    query_size = int(config["query_size"])

    rng = np.random.default_rng(20260718 + acquisition_seed)
    random_order = rng.permutation(len(ids))[:query_size].astype(int).tolist()
    difficulty_order = stable_top_k(ids, values, query_size)

    candidate_embeddings = np.asarray(embeddings[[embedding_lookup[sample_id] for sample_id in ids]], dtype=np.float64)
    reference_embeddings = np.asarray(embeddings[[embedding_lookup[sample_id] for sample_id in initial_ids]], dtype=np.float64)
    diversity_cfg = config["diversity"]
    hybrid_order = hybrid_uncertainty_diversity_select(
        sample_ids=ids,
        difficulties=values,
        candidate_embeddings=candidate_embeddings,
        reference_embeddings=reference_embeddings,
        query_size=query_size,
        shortlist_multiplier=int(diversity_cfg["shortlist_multiplier"]),
        uncertainty_weight=float(diversity_cfg["uncertainty_weight"]),
    )
    return {
        RANDOM: [ids[index] for index in random_order],
        DIFFICULTY: [ids[index] for index in difficulty_order],
        HYBRID: [ids[index] for index in hybrid_order],
    }


def parse_class_ids(value: Any) -> set[int]:
    return {int(part) for part in str(value).split("|") if str(part).strip()}


def write_grounded_review_html(cards: list[dict[str, Any]], pool: pd.DataFrame, out: Path) -> Path:
    """Create a local review interface containing detector evidence only."""

    image_lookup = pool.set_index("sample_id")["image_path"].astype(str).to_dict()
    sections: list[str] = []
    for card in cards:
        sample_id = str(card["sample_id"])
        source = Path(image_lookup[sample_id]).resolve().as_uri()
        evidence = card["grounded_evidence"]
        predicted = ", ".join(card["predicted_classes_only"]) or "검출 클래스 없음"
        sections.append(f"""
<article class="card">
  <img src="{html.escape(source)}" alt="{html.escape(sample_id)}">
  <div>
    <h2>seed {int(card['acquisition_seed'])} · {html.escape(str(card['strategy']))} · rank {int(card['rank'])}</h2>
    <p class="id">{html.escape(sample_id)}</p>
    <p>{html.escape(str(card['explanation']))}</p>
    <p><strong>Detector predicted classes:</strong> {html.escape(predicted)}</p>
    <dl>
      <dt>difficulty</dt><dd>{float(evidence['difficulty']):.4f}</dd>
      <dt>confidence deficit</dt><dd>{float(evidence['confidence_deficit']):.4f}</dd>
      <dt>localization instability</dt><dd>{float(evidence['localization_instability']):.4f}</dd>
      <dt>class instability</dt><dd>{float(evidence['class_instability']):.4f}</dd>
      <dt>count instability</dt><dd>{float(evidence['count_instability']):.4f}</dd>
    </dl>
    <p class="warning">{html.escape(str(card['review_warning']))}</p>
  </div>
</article>""")
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>DCAL-XAI grounded review</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f4f5f7; color: #172033; }}
header {{ max-width: 1100px; margin: auto; }}
.card {{ max-width: 1100px; margin: 18px auto; padding: 18px; background: white; border-radius: 12px;
         display: grid; grid-template-columns: 380px 1fr; gap: 22px; box-shadow: 0 2px 10px #0001; }}
.card img {{ width: 100%; max-height: 260px; object-fit: contain; background: #111; }}
.id {{ font-family: ui-monospace, monospace; color: #5b6475; }}
dl {{ display: grid; grid-template-columns: 220px 100px; margin: 8px 0; }}
dt, dd {{ padding: 3px 0; margin: 0; }}
.warning {{ color: #8a3b12; font-size: 0.9rem; }}
@media (max-width: 800px) {{ .card {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<header><h1>DCAL-XAI Grounded Selection Review</h1>
<p>GT/XML과 VLM 생성 내용을 포함하지 않습니다. 모든 문장은 detector evidence의 결정론적 요약입니다.</p></header>
{''.join(sections)}
</body></html>"""
    path = out / "grounded_review.html"
    path.write_text(document, encoding="utf-8")
    return path


def selection_posthoc_audit(
    *,
    records: pd.DataFrame,
    config: dict[str, Any],
    context: dict[str, Any],
    out: Path,
    seeds: list[int],
) -> tuple[bool, Path]:
    protocol = context["protocol"]
    gt = pd.read_csv(protocol / "gc10_acquisition_pool_gt_audit.csv").set_index("sample_id")
    embedding_lookup = {str(row.sample_id): int(row.embedding_index) for row in context["embedding_manifest"].itertuples(index=False)}
    embeddings = context["embeddings"]
    rare = set(int(value) for value in config["rare_class_ids"])
    metric_rows: list[dict[str, Any]] = []
    for acquisition_seed in seeds:
        initial_ids = reconstruct_initial(context["blind"], acquisition_seed, int(config["initial_size"]))
        initial_classes: set[int] = set()
        for sample_id in initial_ids:
            initial_classes.update(parse_class_ids(gt.loc[sample_id, "class_ids"]))
        initial_emb = np.asarray(embeddings[[embedding_lookup[value] for value in initial_ids]], dtype=np.float64)
        for strategy in config["strategies"]:
            subset = records[(records["acquisition_seed"] == acquisition_seed) & records["strategy"].eq(strategy)].sort_values("rank")
            query_ids = subset["sample_id"].astype(str).tolist()
            if len(query_ids) != int(config["query_size"]):
                raise RuntimeError(f"Selection cardinality failed for seed={acquisition_seed}, strategy={strategy}")
            classes_by_image = [parse_class_ids(gt.loc[sample_id, "class_ids"]) for sample_id in query_ids]
            query_classes = set().union(*classes_by_image)
            query_emb = np.asarray(embeddings[[embedding_lookup[value] for value in query_ids]], dtype=np.float64)
            pairwise = query_emb @ query_emb.T
            upper = pairwise[np.triu_indices(len(query_ids), k=1)]
            initial_similarity = query_emb @ initial_emb.T
            metric_rows.append({
                "acquisition_seed": acquisition_seed,
                "strategy": strategy,
                "query_unique_classes": len(query_classes),
                "combined_unique_classes": len(initial_classes | query_classes),
                "query_images_with_rare_class": sum(bool(values & rare) for values in classes_by_image),
                "query_unique_rare_classes": len(query_classes & rare),
                "query_instances": int(sum(int(gt.loc[value, "num_instances"]) for value in query_ids)),
                "query_difficulty_mean": float(subset["difficulty"].mean()),
                "query_pairwise_cosine_similarity_mean": float(upper.mean()),
                "query_min_distance_to_initial_mean": float((1.0 - initial_similarity.max(axis=1)).mean()),
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "selection_metrics_posthoc.csv", index=False, encoding="utf-8-sig")

    comparison_rows: list[dict[str, Any]] = []
    for comparator in [RANDOM, DIFFICULTY]:
        for metric in [
            "query_unique_classes", "combined_unique_classes", "query_images_with_rare_class",
            "query_unique_rare_classes", "query_instances", "query_difficulty_mean",
            "query_pairwise_cosine_similarity_mean", "query_min_distance_to_initial_mean",
        ]:
            pivot = metrics.pivot(index="acquisition_seed", columns="strategy", values=metric)
            difference = pivot[HYBRID] - pivot[comparator]
            low, high = bootstrap_ci(difference.to_numpy(float), seed=20260718)
            comparison_rows.append({
                "primary": HYBRID,
                "comparator": comparator,
                "metric": metric,
                "mean_difference": float(difference.mean()),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "wins": int((difference > 0).sum()),
                "losses": int((difference < 0).sum()),
                "ties": int((difference == 0).sum()),
            })
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(out / "selection_comparisons.csv", index=False, encoding="utf-8-sig")

    concentration_rows: list[dict[str, Any]] = []
    for strategy in config["strategies"]:
        counts = records[records["strategy"].eq(strategy)]["sample_id"].value_counts()
        concentration_rows.append({
            "strategy": strategy,
            "unique_images": int(len(counts)),
            "total_slots": int(counts.sum()),
            "top10_concentration": float(counts.head(10).sum() / counts.sum()),
            "max_image_frequency": int(counts.max()),
        })
    concentration = pd.DataFrame(concentration_rows)
    concentration.to_csv(out / "selection_concentration.csv", index=False, encoding="utf-8-sig")

    indexed = comparisons.set_index(["comparator", "metric"])["mean_difference"]
    primary_concentration = float(concentration.set_index("strategy").loc[HYBRID, "top10_concentration"])
    thresholds = config["selection_gate"]
    full_seed_set = seeds == [int(value) for value in config["acquisition_seeds"]]
    checks = [
        ("full_five_seed_acquisition", full_seed_set),
        ("combined_unique_classes_noninferiority_vs_random", float(indexed.loc[(RANDOM, "combined_unique_classes")]) >= float(thresholds["combined_unique_classes_gain_vs_random_min"])),
        ("rare_images_noninferiority_vs_random", float(indexed.loc[(RANDOM, "query_images_with_rare_class")]) >= float(thresholds["rare_images_gain_vs_random_min"])),
        ("query_instances_noninferiority_vs_random", float(indexed.loc[(RANDOM, "query_instances")]) >= float(thresholds["query_instances_gain_vs_random_min"])),
        ("difficulty_gain_vs_random", float(indexed.loc[(RANDOM, "query_difficulty_mean")]) >= float(thresholds["difficulty_gain_vs_random_min"])),
        ("diversity_pairwise_similarity_vs_difficulty", float(indexed.loc[(DIFFICULTY, "query_pairwise_cosine_similarity_mean")]) <= float(thresholds["pairwise_similarity_gain_vs_difficulty_max"])),
        ("top10_concentration", primary_concentration <= float(thresholds["top10_concentration_max"])),
    ]
    gate = pd.DataFrame([{"check": name, "passed": bool(value)} for name, value in checks])
    overall = bool(gate["passed"].all())
    gate.to_csv(out / "selection_gate.csv", index=False, encoding="utf-8-sig")
    report = [
        "# DCAL-XAI Selection-Only Gate", "",
        f"- Seeds: **{seeds}**",
        f"- Full frozen seed set: **{full_seed_set}**",
        f"- Gate: **{'PASS' if overall else 'FAIL'}**",
        "- Detector training for acquisition: **True**",
        "- Downstream confirmation training: **False**",
        "- XML/GT used by selector: **False**",
        "- XML/GT joined after committed selection: **True**",
        "- VLM used for selection: **False**",
        "- Final test used: **False**", "",
        "## Post-hoc selection metrics", "", metrics.to_markdown(index=False, floatfmt=".6f"), "",
        "## Primary comparisons", "", comparisons.to_markdown(index=False, floatfmt=".6f"), "",
        "## Cross-seed concentration", "", concentration.to_markdown(index=False, floatfmt=".6f"), "",
        "## Frozen gate", "", gate.assign(result=gate["passed"].map({True: "PASS", False: "FAIL"})).drop(columns="passed").to_markdown(index=False), "",
        "A PASS authorizes development-only confirmation. It does not authorize final-test access.",
    ]
    summary = out / "selection_summary.md"
    summary.write_text("\n".join(report) + "\n", encoding="utf-8")
    return overall, summary


def run_acquire(
    config: dict[str, Any],
    config_path: Path,
    out: Path,
    seeds: list[int],
    dry_run: bool,
) -> tuple[bool | None, Path]:
    context = validate_protocol_and_embeddings(config)
    ensure_output_identity(out, config_path)
    audit_path = run_audit(config, config_path, out, seeds)
    plan = {
        "stage": "acquire",
        "dry_run": dry_run,
        "config_sha256": sha256(config_path),
        "core_sha256": sha256(HERE / "core.py"),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "seeds": seeds,
        "warm_start_models": len(seeds),
        "pool_predictions_per_seed": 2 * (len(context["pool"]) - int(config["initial_size"])),
        "strategies": config["strategies"],
        "initial_size": int(config["initial_size"]),
        "query_size": int(config["query_size"]),
        "model": str(context["model"]),
        "development_evaluated": False,
        "final_test_used": False,
        "audit": str(audit_path),
    }
    plan_path = out / "acquisition_plan.json"
    json_write(plan_path, plan)
    if dry_run:
        return None, plan_path

    embedding_lookup = {str(row.sample_id): int(row.embedding_index) for row in context["embedding_manifest"].itertuples(index=False)}
    record_rows: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    started = time.perf_counter()
    for acquisition_seed in seeds:
        initial_ids = reconstruct_initial(context["blind"], acquisition_seed, int(config["initial_size"]))
        remaining = context["pool"][~context["pool"]["sample_id"].isin(initial_ids)].copy()
        warm_key = f"warm_acq{acquisition_seed}"
        warm_yaml, _ = build_training_yaml(
            protocol=context["protocol"], out=out, key=warm_key, train_ids=initial_ids,
            class_names=config["class_names"], include_development=False,
        )
        checkpoint = train_checkpoint(
            model_path=context["model"], yaml_path=warm_yaml, out=out, name=warm_key,
            training_seed=int(config["selection_training_seed"]), training=config["training"],
        )
        scores = score_pool(
            checkpoint=checkpoint,
            candidates=remaining,
            score_path=out / "scores" / f"acq{acquisition_seed}.csv",
            config=config,
        )
        selections = choose_queries(
            scores=scores,
            acquisition_seed=acquisition_seed,
            initial_ids=initial_ids,
            embedding_lookup=embedding_lookup,
            embeddings=context["embeddings"],
            config=config,
        )
        indexed_scores = scores.set_index("sample_id")
        for strategy, query_ids in selections.items():
            for rank, sample_id in enumerate(query_ids, start=1):
                score = indexed_scores.loc[sample_id]
                record_rows.append({
                    "acquisition_seed": acquisition_seed,
                    "strategy": strategy,
                    "rank": rank,
                    "sample_id": sample_id,
                    "difficulty": float(score["difficulty"]),
                    "confidence_deficit": float(score["confidence_deficit"]),
                    "localization_instability": float(score["localization_instability"]),
                    "class_instability": float(score["class_instability"]),
                    "count_instability": float(score["count_instability"]),
                    "warm_checkpoint_sha256": sha256(checkpoint),
                    "selector_used_gt": False,
                    "selector_used_vlm": False,
                    "final_test_used": False,
                })
                card = grounded_explanation(sample_id, evidence_from_row(score), config["class_names"])
                cards.append({"acquisition_seed": acquisition_seed, "strategy": strategy, "rank": rank, **card})
    records = pd.DataFrame(record_rows)
    expected = len(seeds) * len(config["strategies"]) * int(config["query_size"])
    if len(records) != expected:
        raise RuntimeError(f"Incomplete selections: {len(records)} != {expected}")
    records.to_csv(out / "selection_records.csv", index=False, encoding="utf-8-sig")
    with (out / "grounded_explanations.jsonl").open("w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
    write_grounded_review_html(cards, context["pool"], out)
    gate_pass, summary = selection_posthoc_audit(
        records=records, config=config, context=context, out=out, seeds=seeds,
    )
    json_write(out / "acquisition_config.json", {
        "status": "complete",
        "config_sha256": sha256(config_path),
        "core_sha256": sha256(HERE / "core.py"),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "seeds": seeds,
        "full_seed_set": seeds == [int(value) for value in config["acquisition_seeds"]],
        "selection_gate_pass": gate_pass,
        "runtime_seconds": time.perf_counter() - started,
        "development_evaluated": False,
        "final_test_used": False,
    })
    return gate_pass, summary


def recover_metrics(metrics: Any, class_names: list[str]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    results = getattr(metrics, "results_dict", {}) or {}
    aggregate = {
        "precision": float(results.get("metrics/precision(B)", getattr(metrics.box, "mp", np.nan))),
        "recall": float(results.get("metrics/recall(B)", getattr(metrics.box, "mr", np.nan))),
        "map50": float(results.get("metrics/mAP50(B)", getattr(metrics.box, "map50", np.nan))),
        "map5095": float(results.get("metrics/mAP50-95(B)", getattr(metrics.box, "map", np.nan))),
    }
    maps = np.asarray(getattr(metrics.box, "maps", []), dtype=float)
    if maps.shape != (len(class_names),):
        raise RuntimeError(f"Expected {len(class_names)} per-class AP values, got {maps.shape}")
    per_class = [
        {"class_index": index, "class_id": index + 1, "class_name": class_names[index], "ap5095": float(maps[index])}
        for index in range(len(class_names))
    ]
    return aggregate, per_class


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def run_or_recover_confirmation(
    *,
    item: dict[str, Any],
    training_seed: int,
    config: dict[str, Any],
    context: dict[str, Any],
    out: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from ultralytics import YOLO

    key = f"acq{item['acquisition_seed']}_{item['strategy']}"
    name = f"{key}_trainseed{training_seed}"
    result_dir = out / "confirmation_results" / name
    aggregate_path = result_dir / "aggregate.json"
    per_class_path = result_dir / "per_class.json"
    if aggregate_path.exists() and per_class_path.exists():
        return json.loads(aggregate_path.read_text(encoding="utf-8")), json.loads(per_class_path.read_text(encoding="utf-8"))
    yaml_path, _ = build_training_yaml(
        protocol=context["protocol"], out=out, key=key, train_ids=item["train_ids"],
        class_names=config["class_names"], include_development=True,
    )
    started = time.perf_counter()
    checkpoint = train_checkpoint(
        model_path=context["model"], yaml_path=yaml_path, out=out, name=name,
        training_seed=training_seed, training=config["training"],
    )
    print(f"[DEV VAL] {name}", flush=True)
    metrics = YOLO(str(checkpoint)).val(
        data=str(yaml_path), split="val", imgsz=int(config["training"]["imgsz"]),
        batch=int(config["training"]["batch"]), device=str(config["training"]["device"]),
        workers=int(config["training"]["workers"]), plots=False, save_json=False,
        verbose=False, project=str(out / "dev_val_runs"), name=name, exist_ok=True,
    )
    aggregate, per_class = recover_metrics(metrics, config["class_names"])
    common = {
        "acquisition_seed": int(item["acquisition_seed"]),
        "strategy": str(item["strategy"]),
        "training_seed": int(training_seed),
        "train_images": len(item["train_ids"]),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "runtime_seconds": time.perf_counter() - started,
        "evaluation_split": "development",
        "final_test_used": False,
    }
    aggregate_row = {**common, **aggregate}
    per_class_rows = [{**common, **row} for row in per_class]
    result_dir.mkdir(parents=True, exist_ok=True)
    json_write(aggregate_path, aggregate_row)
    json_write(per_class_path, per_class_rows)
    return aggregate_row, per_class_rows


def analyze_confirmation(
    aggregate: pd.DataFrame,
    per_class: pd.DataFrame,
    config: dict[str, Any],
    out: Path,
) -> tuple[bool, Path]:
    comparison_rows: list[dict[str, Any]] = []
    for comparator in [RANDOM, DIFFICULTY]:
        for metric in ["map5095", "map50", "precision", "recall"]:
            pivot = aggregate.pivot(index=["acquisition_seed", "training_seed"], columns="strategy", values=metric)
            paired = pivot[HYBRID] - pivot[comparator]
            by_acquisition = paired.groupby(level="acquisition_seed").mean()
            low, high = bootstrap_ci(by_acquisition.to_numpy(float), seed=20260718)
            comparison_rows.append({
                "primary": HYBRID,
                "comparator": comparator,
                "metric": metric,
                "primary_mean": float(aggregate[aggregate["strategy"].eq(HYBRID)][metric].mean()),
                "comparator_mean": float(aggregate[aggregate["strategy"].eq(comparator)][metric].mean()),
                "mean_difference": float(by_acquisition.mean()),
                "bootstrap_ci95_low_across_acquisition_seeds": low,
                "bootstrap_ci95_high_across_acquisition_seeds": high,
                "acquisition_seed_wins": int((by_acquisition > 0).sum()),
                "acquisition_seed_losses": int((by_acquisition < 0).sum()),
            })
    comparisons = pd.DataFrame(comparison_rows)

    rare_ids = set(int(value) for value in config["rare_class_ids"])
    macro = per_class.copy()
    macro["group"] = macro["class_id"].map(lambda value: "rare" if int(value) in rare_ids else "frequent")
    macro = macro.groupby(["acquisition_seed", "training_seed", "strategy", "group"])["ap5095"].mean().reset_index()
    macro_pivot = macro.pivot(index=["acquisition_seed", "training_seed", "group"], columns="strategy", values="ap5095").reset_index()
    macro_pivot["hybrid_minus_random"] = macro_pivot[HYBRID] - macro_pivot[RANDOM]
    macro_summary = macro_pivot.groupby("group").agg(
        random_mean=(RANDOM, "mean"),
        hybrid_mean=(HYBRID, "mean"),
        mean_difference=("hybrid_minus_random", "mean"),
    ).reset_index()

    class_means = per_class.groupby(["strategy", "class_id", "class_name"])["ap5095"].mean().reset_index()
    class_pivot = class_means.pivot(index=["class_id", "class_name"], columns="strategy", values="ap5095").reset_index()
    class_pivot["hybrid_minus_random"] = class_pivot[HYBRID] - class_pivot[RANDOM]

    indexed = comparisons.set_index(["comparator", "metric"])
    map_row = indexed.loc[(RANDOM, "map5095")]
    recall_row = indexed.loc[(RANDOM, "recall")]
    rare_gain = float(macro_summary.set_index("group").loc["rare", "mean_difference"])
    worst_class = float(class_pivot["hybrid_minus_random"].min())
    thresholds = config["detector_gate"]
    checks = [
        ("map5095_gain_vs_random", float(map_row["mean_difference"]) >= float(thresholds["map5095_gain_vs_random_min"])),
        ("acquisition_seed_wins", int(map_row["acquisition_seed_wins"]) >= int(thresholds["acquisition_seed_wins_min"])),
        ("bootstrap_ci_low_positive", float(map_row["bootstrap_ci95_low_across_acquisition_seeds"]) > float(thresholds["bootstrap_ci_low_min"])),
        ("rare_macro_noninferiority", rare_gain >= float(thresholds["rare_macro_gain_vs_random_min"])),
        ("recall_noninferiority", float(recall_row["mean_difference"]) >= float(thresholds["recall_gain_vs_random_min"])),
        ("worst_class_safety", worst_class >= float(thresholds["worst_class_gain_vs_random_min"])),
    ]
    gate = pd.DataFrame([{"check": name, "passed": bool(value)} for name, value in checks])
    overall = bool(gate["passed"].all())
    aggregate.to_csv(out / "confirmation_metrics.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(out / "confirmation_per_class.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(out / "confirmation_comparisons.csv", index=False, encoding="utf-8-sig")
    macro_summary.to_csv(out / "confirmation_macro_groups.csv", index=False, encoding="utf-8-sig")
    class_pivot.to_csv(out / "confirmation_class_means.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(out / "confirmation_gate.csv", index=False, encoding="utf-8-sig")
    report = [
        "# DCAL-XAI Development Detector Confirmation", "",
        f"- Models: **{len(aggregate)}**",
        "- Acquisition/training seeds: **5/3**",
        "- Labeled images per model: **40**",
        f"- Gate: **{'PASS' if overall else 'FAIL'}**",
        "- Evaluation: **development only**",
        "- Final test used: **False**", "",
        "## Aggregate comparisons", "", comparisons.to_markdown(index=False, floatfmt=".6f"), "",
        "## Rare/frequent macro AP", "", macro_summary.to_markdown(index=False, floatfmt=".6f"), "",
        "## Per-class AP", "", class_pivot.to_markdown(index=False, floatfmt=".6f"), "",
        "## Frozen gate", "", gate.assign(result=gate["passed"].map({True: "PASS", False: "FAIL"})).drop(columns="passed").to_markdown(index=False), "",
        "The final test remains locked regardless of this development result.",
    ]
    summary = out / "confirmation_summary.md"
    summary.write_text("\n".join(report) + "\n", encoding="utf-8")
    return overall, summary


def run_confirm(config: dict[str, Any], config_path: Path, out: Path, dry_run: bool) -> tuple[bool | None, Path]:
    context = validate_protocol_and_embeddings(config)
    ensure_output_identity(out, config_path)
    acquisition_config_path = out / "acquisition_config.json"
    records_path = out / "selection_records.csv"
    if not acquisition_config_path.exists() or not records_path.exists():
        raise RuntimeError("Run the full acquisition stage first")
    acquisition_config = json.loads(acquisition_config_path.read_text(encoding="utf-8"))
    if not bool(acquisition_config.get("full_seed_set", False)):
        raise RuntimeError("Smoke/subset acquisition cannot authorize confirmation")
    if not bool(acquisition_config.get("selection_gate_pass", False)):
        raise RuntimeError("Selection-only gate did not authorize confirmation")
    records = pd.read_csv(records_path)
    seeds = [int(value) for value in config["acquisition_seeds"]]
    plan = {
        "stage": "confirm",
        "dry_run": dry_run,
        "seeds": seeds,
        "strategies": config["strategies"],
        "training_seeds": config["confirmation_training_seeds"],
        "models": len(seeds) * len(config["strategies"]) * len(config["confirmation_training_seeds"]),
        "evaluation_split": "development",
        "final_test_used": False,
    }
    plan_path = out / "confirmation_plan.json"
    json_write(plan_path, plan)
    if dry_run:
        return None, plan_path

    items: list[dict[str, Any]] = []
    for acquisition_seed in seeds:
        initial_ids = reconstruct_initial(context["blind"], acquisition_seed, int(config["initial_size"]))
        for strategy in config["strategies"]:
            query = records[(records["acquisition_seed"] == acquisition_seed) & records["strategy"].eq(strategy)].sort_values("rank")
            query_ids = query["sample_id"].astype(str).tolist()
            train_ids = initial_ids + query_ids
            if len(train_ids) != int(config["initial_size"]) + int(config["query_size"]) or len(set(train_ids)) != len(train_ids):
                raise RuntimeError("Confirmation training set cardinality/overlap failure")
            items.append({"acquisition_seed": acquisition_seed, "strategy": strategy, "train_ids": train_ids})
    aggregate_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for item in items:
        for training_seed in config["confirmation_training_seeds"]:
            aggregate, per_class = run_or_recover_confirmation(
                item=item, training_seed=int(training_seed), config=config, context=context, out=out,
            )
            aggregate_rows.append(aggregate)
            per_class_rows.extend(per_class)
    aggregate = pd.DataFrame(aggregate_rows)
    per_class = pd.DataFrame(per_class_rows)
    expected_models = len(seeds) * len(config["strategies"]) * len(config["confirmation_training_seeds"])
    if len(aggregate) != expected_models or len(per_class) != expected_models * len(config["class_names"]):
        raise RuntimeError("Incomplete confirmation results")
    overall, summary = analyze_confirmation(aggregate, per_class, config, out)
    json_write(out / "confirmation_config.json", {
        "status": "complete",
        "models": expected_models,
        "gate_pass": overall,
        "evaluation_split": "development",
        "final_test_used": False,
    })
    return overall, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "acquire", "confirm"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", help="Frozen acquisition-seed subset, e.g. 0 or 0,1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    out = resolve_project_path(args.output_dir or config["output_dir"])
    seeds = parse_seed_override(args.seeds, config["acquisition_seeds"])

    if args.stage == "audit":
        path = run_audit(config, config_path, out, seeds)
        print(f"[AUDIT PASS] {path}")
        print("Training performed: False")
        print("Final test used: False")
        return
    if args.stage == "acquire":
        gate, path = run_acquire(config, config_path, out, seeds, args.dry_run)
        print(f"[{'DRY RUN' if gate is None else 'DONE'}] {path}")
        if gate is not None:
            print(f"[SELECTION GATE] {'PASS' if gate else 'FAIL'}")
        print(f"Training performed: {not args.dry_run}")
        print("Final test used: False")
        return
    if args.seeds is not None:
        raise ValueError("confirm always uses the full frozen acquisition-seed set")
    gate, path = run_confirm(config, config_path, out, args.dry_run)
    print(f"[{'DRY RUN' if gate is None else 'DONE'}] {path}")
    if gate is not None:
        print(f"[DETECTOR GATE] {'PASS' if gate else 'FAIL'}")
    print(f"Training performed: {not args.dry_run}")
    print("Final test used: False")


if __name__ == "__main__":
    main()

"""V9 detector-aware acquisition probe.

This is a diagnostic-only bridge between the V8 NEU-only results and a possible
V9 detector-aware runner.  It does not train YOLO and does not evaluate the
final test split.

Default behavior:
    - Reuse an existing V8 NEU-only seed42 round-0 checkpoint.
    - Score the remaining unlabeled NEU pool with YOLO predictions.
    - Build three candidate next-round selections:
        1. DetectorUncertainty
        2. DetectorUncertaintyDINO
        3. DetectorUncertaintyDINOBalanced
    - Compare the selected batch against V8 round-1 Random / Consistency /
      DINO Visual selections using XML only for post-hoc diagnosis.

The scoring view is GT-free.  XML/class_hint fields are used only after the
selection for diagnostics.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Safe defaults must be set before importing the V7 runner helpers.
os.environ.setdefault("AL_POOL_DATASET_FILTER", "NEU-DET")
os.environ.setdefault("AL_DEV_EVAL_DATASET_FILTER", "NEU-DET")
os.environ.setdefault("AL_EVAL_SPLIT", "development_eval_v7")
os.environ.setdefault("AL_EMBEDDING_BACKEND", "dinov2")
os.environ.setdefault("AL_ALLOW_MODEL_DOWNLOAD", "0")

import run_al_yolo_ablation_v6_deficit_diversity as v6  # noqa: E402
from analyze_instance_richness_v7 import parse_image_instances, strategy_stats  # noqa: E402
from audit_detection_pipeline_v7 import build_image_index, build_xml_index  # noqa: E402
from run_al_yolo_ablation_v7_full_curve import (  # noqa: E402
    load_dino_embeddings,
    load_priority_pool_with_identity,
)

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    raise ImportError("Install ultralytics before running V9 detector-aware probes.") from exc


PROJECT_ROOT = v6.PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "v9_detector_aware_selection_probe"
DEFAULT_SOURCE_RUN = (
    PROJECT_ROOT
    / "runs"
    / "active_learning_ablation_v8_neu_only"
    / "v8_neu_only_20260712_105644"
)


def parse_csv_env(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return default
    return [v.strip() for v in value.split(",") if v.strip()]


def normalize(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    if len(values) == 0:
        return values
    span = float(values.max() - values.min())
    if not np.isfinite(span) or span <= 1e-12:
        return pd.Series([0.0] * len(values), index=values.index, dtype=float)
    return (values - values.min()) / span


def table_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def resolve_source_run() -> Path:
    override = os.environ.get("AL_V9_SOURCE_RUN_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    return DEFAULT_SOURCE_RUN


def resolve_weights(source_run: Path, acquisition_seed: int) -> Path:
    override = os.environ.get("AL_V9_DETECTOR_WEIGHTS")
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p
    training_seed = 1000 + acquisition_seed
    return (
        source_run
        / "yolo_train_runs"
        / f"seed{acquisition_seed}___SHARED_ROUND0___R0_trainseed{training_seed}"
        / "weights"
        / "best.pt"
    )


def load_source_initial_and_baselines(source_run: Path, acquisition_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_path = source_run / "all_selected_samples_by_round.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing source selected samples CSV: {selected_path}")
    selected = pd.read_csv(selected_path)
    selected = selected[pd.to_numeric(selected["acquisition_seed"], errors="coerce").eq(acquisition_seed)].copy()
    if selected.empty:
        raise ValueError(f"No selected samples found for acquisition_seed={acquisition_seed} in {selected_path}")

    initial = selected[pd.to_numeric(selected["round"], errors="coerce").eq(0)].copy()
    initial = initial.drop_duplicates("sample_id", keep="first").reset_index(drop=True)
    if initial.empty:
        raise ValueError(f"No round-0 initial set found for acquisition_seed={acquisition_seed}")

    baselines = selected[pd.to_numeric(selected["round"], errors="coerce").eq(1)].copy()
    baselines = baselines[baselines["selection_type"].astype(str).ne("cumulative_labeled")].copy()
    return initial, baselines


def prediction_rows(model: YOLO, pool: pd.DataFrame, *, device: str, imgsz: int, conf: float, iou: float) -> pd.DataFrame:
    paths = pool["resolved_image_path"].astype(str).tolist()
    if not paths:
        return pd.DataFrame()

    results = model.predict(
        source=paths,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
        stream=False,
    )
    names = getattr(model, "names", {}) or {}
    rows: list[dict[str, Any]] = []
    for (_, sample), result in zip(pool.iterrows(), results):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "detector_pseudo_box_count": 0,
                    "detector_no_box": 1.0,
                    "detector_max_conf": 0.0,
                    "detector_mean_conf": 0.0,
                    "detector_min_conf": 0.0,
                    "detector_pred_class": "__no_box__",
                    "detector_uncertainty": 1.0,
                    "pseudo_instance_score": 0.0,
                }
            )
            continue

        confs = boxes.conf.detach().cpu().numpy().astype(float)
        clss = boxes.cls.detach().cpu().numpy().astype(int)
        top_idx = int(np.argmax(confs))
        top_cls = int(clss[top_idx])
        pred_class = str(names.get(top_cls, top_cls))
        max_conf = float(confs[top_idx])
        mean_conf = float(np.mean(confs))
        # For a defect-only NEU pool, "no box" and low top confidence both mean
        # the current detector is struggling to localize a defect.
        conf_uncertainty = 1.0 - max_conf
        instance_score = min(float(len(confs)), 5.0) / 5.0
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "detector_pseudo_box_count": int(len(confs)),
                "detector_no_box": 0.0,
                "detector_max_conf": max_conf,
                "detector_mean_conf": mean_conf,
                "detector_min_conf": float(np.min(confs)),
                "detector_pred_class": pred_class,
                "detector_uncertainty": conf_uncertainty,
                "pseudo_instance_score": instance_score,
            }
        )
    return pd.DataFrame(rows)


def min_cosine_distance(sample_id: str, reference_ids: list[str], embedding_lookup: dict[str, np.ndarray]) -> float:
    emb = embedding_lookup.get(sample_id)
    if emb is None:
        return 0.0
    refs = [embedding_lookup[sid] for sid in reference_ids if sid in embedding_lookup]
    if not refs:
        return 1.0
    ref_mat = np.vstack(refs)
    sims = ref_mat @ emb
    return float(1.0 - np.max(sims))


def class_deficit_score(
    pred_class: str,
    labeled_counts: Counter,
    selected_pred_counts: Counter,
    class_universe: list[str],
) -> float:
    if pred_class == "__no_box__" or pred_class not in class_universe:
        return 0.0
    current_total = sum(labeled_counts.values()) + sum(selected_pred_counts.values()) + 1
    target = current_total / max(1, len(class_universe))
    current_class_count = labeled_counts.get(pred_class, 0) + selected_pred_counts.get(pred_class, 0)
    return max(0.0, float(target - current_class_count))


def select_detector_uncertainty(scored_pool: pd.DataFrame, query_size: int) -> pd.DataFrame:
    return (
        scored_pool.sort_values(
            ["detector_uncertainty", "pseudo_instance_score", "sample_id"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .head(query_size)
        .copy()
    )


def select_detector_uncertainty_dino(
    scored_pool: pd.DataFrame,
    initial_df: pd.DataFrame,
    embedding_lookup: dict[str, np.ndarray],
    *,
    query_size: int,
    candidate_fraction: float,
) -> pd.DataFrame:
    n_candidates = max(query_size, int(math.ceil(len(scored_pool) * candidate_fraction)))
    candidates = (
        scored_pool.sort_values(["detector_uncertainty", "sample_id"], ascending=[False, True], kind="mergesort")
        .head(n_candidates)
        .copy()
    )
    selected_parts: list[pd.DataFrame] = []
    selected_ids: list[str] = []
    reference_ids = initial_df["sample_id"].astype(str).tolist()
    remaining = candidates.copy()
    for _ in range(min(query_size, len(remaining))):
        refs = reference_ids + selected_ids
        remaining = remaining.copy()
        remaining["_visual_distance"] = [
            min_cosine_distance(str(sid), refs, embedding_lookup) for sid in remaining["sample_id"].astype(str)
        ]
        pick_idx = remaining.sort_values(
            ["_visual_distance", "detector_uncertainty", "sample_id"],
            ascending=[False, False, True],
            kind="mergesort",
        ).index[0]
        picked = remaining.loc[[pick_idx]].copy()
        selected_parts.append(picked)
        selected_ids.append(str(picked.iloc[0]["sample_id"]))
        remaining = remaining.drop(index=pick_idx)
    return pd.concat(selected_parts, ignore_index=True) if selected_parts else scored_pool.iloc[0:0].copy()


def select_detector_uncertainty_dino_balanced(
    scored_pool: pd.DataFrame,
    initial_df: pd.DataFrame,
    embedding_lookup: dict[str, np.ndarray],
    *,
    query_size: int,
    candidate_fraction: float,
) -> pd.DataFrame:
    n_candidates = max(query_size, int(math.ceil(len(scored_pool) * candidate_fraction)))
    candidates = (
        scored_pool.sort_values(["detector_uncertainty", "sample_id"], ascending=[False, True], kind="mergesort")
        .head(n_candidates)
        .copy()
    )
    class_universe = sorted(str(v) for v in getattr(v6, "NEU_CLASSES", []) if str(v))
    if not class_universe:
        class_universe = sorted(
            c
            for c in scored_pool["detector_pred_class"].dropna().astype(str).unique().tolist()
            if c != "__no_box__"
        )
    labeled_counts = Counter(initial_df["class_hint"].dropna().astype(str))
    selected_pred_counts: Counter = Counter()
    selected_parts: list[pd.DataFrame] = []
    selected_ids: list[str] = []
    reference_ids = initial_df["sample_id"].astype(str).tolist()
    remaining = candidates.copy()

    for _ in range(min(query_size, len(remaining))):
        refs = reference_ids + selected_ids
        remaining = remaining.copy()
        remaining["_visual_distance_raw"] = [
            min_cosine_distance(str(sid), refs, embedding_lookup) for sid in remaining["sample_id"].astype(str)
        ]
        remaining["_balance_deficit_raw"] = [
            class_deficit_score(str(cls), labeled_counts, selected_pred_counts, class_universe)
            for cls in remaining["detector_pred_class"].astype(str)
        ]
        remaining["_detector_norm"] = normalize(remaining["detector_uncertainty"])
        remaining["_visual_norm"] = normalize(remaining["_visual_distance_raw"])
        remaining["_balance_norm"] = normalize(remaining["_balance_deficit_raw"])
        remaining["_instance_norm"] = normalize(remaining["pseudo_instance_score"])
        # Fixed diagnostic weights.  Do not tune these after seeing performance.
        remaining["_v9_balanced_score"] = (
            0.35 * remaining["_detector_norm"]
            + 0.35 * remaining["_visual_norm"]
            + 0.20 * remaining["_balance_norm"]
            + 0.10 * remaining["_instance_norm"]
        )
        pick_idx = remaining.sort_values(
            ["_v9_balanced_score", "detector_uncertainty", "_visual_distance_raw", "sample_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).index[0]
        picked = remaining.loc[[pick_idx]].copy()
        selected_parts.append(picked)
        selected_ids.append(str(picked.iloc[0]["sample_id"]))
        pred_cls = str(picked.iloc[0]["detector_pred_class"])
        if pred_cls != "__no_box__":
            selected_pred_counts[pred_cls] += 1
        remaining = remaining.drop(index=pick_idx)
    return pd.concat(selected_parts, ignore_index=True) if selected_parts else scored_pool.iloc[0:0].copy()


def add_source_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    out = df.copy()
    out["source_strategy"] = strategy
    return out


def posthoc_stats(selected_df: pd.DataFrame, image_index, xml_index) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in selected_df.iterrows():
        rows.extend(parse_image_instances(row, image_index, xml_index))
    instance_df = pd.DataFrame(rows)
    stats_df = strategy_stats(selected_df, instance_df) if len(selected_df) else pd.DataFrame()
    return instance_df, stats_df


def write_summary(
    save_dir: Path,
    *,
    config: dict[str, Any],
    batch_stats: pd.DataFrame,
    cumulative_stats: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    selected_view_cols = [
        "source_strategy",
        "image_name",
        "class_hint",
        "detector_pred_class",
        "detector_pseudo_box_count",
        "detector_uncertainty",
        "detector_max_conf",
        "pseudo_instance_score",
    ]
    lines = [
        "# V9 Detector-aware Selection Probe",
        "",
        "Diagnostic only. No YOLO training. No final-test evaluation.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Selected next-round batches",
        "",
        table_md(selected[[c for c in selected_view_cols if c in selected.columns]]),
        "",
        "## Post-hoc batch XML statistics",
        "",
        table_md(batch_stats),
        "",
        "## Post-hoc cumulative-after-round1 XML statistics",
        "",
        table_md(cumulative_stats),
        "",
        "## Reading guide",
        "",
        "- `DetectorUncertainty` checks whether the current YOLO model finds genuinely hard samples.",
        "- `DetectorUncertaintyDINO` checks whether DINO diversity changes the hard-sample batch composition.",
        "- `DetectorUncertaintyDINOBalanced` checks whether a fixed balance term avoids selecting visually diverse but detector-weak batches.",
        "- XML statistics are post-hoc only and must not be used by acquisition logic.",
    ]
    (save_dir / "v9_detector_aware_selection_probe_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RUNS_ROOT / f"v9_detector_probe_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    acquisition_seed = int(os.environ.get("AL_ACQUISITION_SEED", "42"))
    query_size = int(os.environ.get("AL_QUERY_SIZE", "5"))
    candidate_fraction = float(os.environ.get("AL_V9_CANDIDATE_FRACTION", "0.30"))
    imgsz = int(os.environ.get("AL_IMGSZ", "640"))
    device = os.environ.get("AL_YOLO_DEVICE", "0")
    conf = float(os.environ.get("AL_V9_PREDICT_CONF", "0.05"))
    iou = float(os.environ.get("AL_V9_PREDICT_IOU", "0.70"))
    source_run = resolve_source_run()
    weights = resolve_weights(source_run, acquisition_seed)
    if not weights.exists():
        raise FileNotFoundError(f"Detector weights not found: {weights}")

    print("[V9 PROBE] loading pool and embeddings...")
    _, full_pool, _ = load_priority_pool_with_identity()
    embedding_dir, _, _, embedding_lookup, dino_config = load_dino_embeddings(full_pool)
    initial_df, baseline_round1 = load_source_initial_and_baselines(source_run, acquisition_seed)
    current_pool = full_pool[~full_pool["sample_id"].isin(initial_df["sample_id"])].copy().reset_index(drop=True)
    if current_pool.empty:
        raise ValueError("Current unlabeled pool is empty after removing initial set.")

    print(f"[V9 PROBE] pool={len(full_pool)} initial={len(initial_df)} unlabeled={len(current_pool)}")
    print(f"[V9 PROBE] weights={weights}")
    model = YOLO(str(weights))
    detector_scores = prediction_rows(model, current_pool, device=device, imgsz=imgsz, conf=conf, iou=iou)
    scored_pool = current_pool.merge(detector_scores, on="sample_id", how="left")
    scored_pool.to_csv(save_dir / "detector_uncertainty_scores.csv", index=False, encoding="utf-8-sig")

    print("[V9 PROBE] building diagnostic selections...")
    selections = [
        add_source_strategy(select_detector_uncertainty(scored_pool, query_size), "DetectorUncertainty"),
        add_source_strategy(
            select_detector_uncertainty_dino(
                scored_pool,
                initial_df,
                embedding_lookup,
                query_size=query_size,
                candidate_fraction=candidate_fraction,
            ),
            "DetectorUncertaintyDINO",
        ),
        add_source_strategy(
            select_detector_uncertainty_dino_balanced(
                scored_pool,
                initial_df,
                embedding_lookup,
                query_size=query_size,
                candidate_fraction=candidate_fraction,
            ),
            "DetectorUncertaintyDINOBalanced",
        ),
    ]
    baseline_keep = baseline_round1.merge(detector_scores, on="sample_id", how="left")
    baseline_keep["source_strategy"] = baseline_keep["strategy"].astype(str)
    selected_batches = pd.concat([baseline_keep] + selections, ignore_index=True, sort=False)
    selected_batches.to_csv(save_dir / "v9_probe_selected_samples.csv", index=False, encoding="utf-8-sig")

    image_index = build_image_index()
    xml_index = build_xml_index()
    batch_instances, batch_stats = posthoc_stats(selected_batches, image_index, xml_index)
    batch_instances.to_csv(save_dir / "v9_probe_batch_instances.csv", index=False, encoding="utf-8-sig")
    batch_stats.to_csv(save_dir / "v9_probe_batch_actual_instance_stats.csv", index=False, encoding="utf-8-sig")

    cumulative_parts = []
    for strategy, batch in selected_batches.groupby("source_strategy", dropna=False):
        init = initial_df.copy()
        init["source_strategy"] = strategy
        batch_copy = batch.copy()
        batch_copy["source_strategy"] = strategy
        cumulative_parts.append(pd.concat([init, batch_copy], ignore_index=True, sort=False))
    cumulative_selected = pd.concat(cumulative_parts, ignore_index=True, sort=False)
    cumulative_instances, cumulative_stats = posthoc_stats(cumulative_selected, image_index, xml_index)
    cumulative_instances.to_csv(save_dir / "v9_probe_cumulative_instances.csv", index=False, encoding="utf-8-sig")
    cumulative_stats.to_csv(save_dir / "v9_probe_cumulative_actual_instance_stats.csv", index=False, encoding="utf-8-sig")

    config = {
        "status": "diagnostic_only",
        "final_test_used": False,
        "yolo_training_run": False,
        "source_run": str(source_run),
        "acquisition_seed": acquisition_seed,
        "query_size": query_size,
        "candidate_fraction": candidate_fraction,
        "weights": str(weights),
        "imgsz": imgsz,
        "device": device,
        "predict_conf": conf,
        "predict_iou": iou,
        "pool_size": int(len(full_pool)),
        "initial_size": int(len(initial_df)),
        "unlabeled_size": int(len(current_pool)),
        "embedding_dir": str(embedding_dir),
        "embedding_config": dino_config,
        "fixed_balanced_weights": {
            "detector_uncertainty": 0.35,
            "dino_visual_distance": 0.35,
            "predicted_class_deficit": 0.20,
            "pseudo_instance_count": 0.10,
        },
    }
    (save_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(
        save_dir,
        config=config,
        batch_stats=batch_stats,
        cumulative_stats=cumulative_stats,
        selected=selected_batches,
    )

    print("=" * 100)
    print("[DONE] V9 detector-aware selection probe finished")
    print(f"Output dir: {save_dir}")
    print("No YOLO training. No final-test evaluation.")
    print("=" * 100)


if __name__ == "__main__":
    main()

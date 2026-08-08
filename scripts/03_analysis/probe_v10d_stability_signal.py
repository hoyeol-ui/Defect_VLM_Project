"""V10d perturbation-stability signal validity probe.

This script is selection-only / audit-only:
  - no YOLO training
  - no final-test evaluation
  - no GT usage for acquisition
  - GT XML is used only post-hoc to test whether perturbation instability
    predicts real detector error better than pseudo box count did.

Default source is the completed V10c 24/6 seed47 run.  The script reuses its
Round0 checkpoint and detector-score CSV, then runs lightweight photometric
perturbations on a candidate subset.

Outputs:
  runs/v10d_stability_signal_probe/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_RUN = PROJECT_ROOT / "runs" / "active_learning_v10c_recall_guard_onecycle" / "v10c_recall_guard_onecycle_20260713_201155"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "runs" / "v10d_stability_signal_probe"
NEU_ANNOTATIONS = PROJECT_ROOT / "data" / "NEU-DET" / "ANNOTATIONS"

NEU6 = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def normalize_class_name(name: Any) -> str:
    return str(name).strip().replace("rolled-in-scale", "rolled-in_scale")


def safe_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def image_stem_from_row(row: pd.Series) -> str:
    image_name = str(row.get("image_name", "")).strip()
    if image_name:
        return Path(image_name).stem
    image_path = str(row.get("resolved_image_path", row.get("image_path", ""))).strip()
    return Path(image_path.split("::")[0]).stem


def xml_path_for_row(row: pd.Series) -> Path:
    return NEU_ANNOTATIONS / f"{image_stem_from_row(row)}.xml"


def parse_gt_xml(xml_path: Path) -> dict[str, Any]:
    if not xml_path.exists():
        return {"gt_box_count": np.nan, "gt_classes": "", "gt_majority_class": "", "_gt_boxes": []}
    root = ET.parse(xml_path).getroot()
    boxes: list[tuple[float, float, float, float, str]] = []
    classes: list[str] = []
    for obj in root.findall("object"):
        cls = normalize_class_name(obj.findtext("name", ""))
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        vals = []
        for key in ["xmin", "ymin", "xmax", "ymax"]:
            vals.append(safe_float(bnd.findtext(key)))
        if not all(np.isfinite(v) for v in vals):
            continue
        boxes.append((vals[0], vals[1], vals[2], vals[3], cls))
        classes.append(cls)
    counts = Counter(classes)
    return {
        "gt_box_count": len(boxes),
        "gt_classes": "|".join(sorted(counts.keys())),
        "gt_majority_class": counts.most_common(1)[0][0] if counts else "",
        "gt_class_counts_json": json.dumps(dict(sorted(counts.items())), ensure_ascii=False),
        "_gt_boxes": boxes,
    }


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def greedy_match_iou(ref_boxes: list[np.ndarray], alt_boxes: list[np.ndarray], threshold: float = 0.1) -> list[tuple[int, int, float]]:
    pairs: list[tuple[float, int, int]] = []
    for i, rb in enumerate(ref_boxes):
        for j, ab in enumerate(alt_boxes):
            pairs.append((iou_xyxy(rb, ab), i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])
    used_ref: set[int] = set()
    used_alt: set[int] = set()
    out: list[tuple[int, int, float]] = []
    for value, i, j in pairs:
        if value < threshold:
            break
        if i in used_ref or j in used_alt:
            continue
        used_ref.add(i)
        used_alt.add(j)
        out.append((i, j, value))
    return out


def perturbations(image: Image.Image) -> dict[str, Image.Image]:
    rgb = image.convert("RGB")
    return {
        "orig": rgb,
        "bright_down": ImageEnhance.Brightness(rgb).enhance(0.92),
        "bright_up": ImageEnhance.Brightness(rgb).enhance(1.08),
        "contrast_down": ImageEnhance.Contrast(rgb).enhance(0.92),
        "contrast_up": ImageEnhance.Contrast(rgb).enhance(1.08),
        "mild_blur": rgb.filter(ImageFilter.GaussianBlur(radius=0.6)),
    }


def yolo_predict_image(model: Any, image: Image.Image, *, imgsz: int, conf: float, device: str) -> dict[str, Any]:
    result = model.predict(image, imgsz=imgsz, conf=conf, iou=0.70, device=device, verbose=False)[0]
    names = getattr(result, "names", {})
    boxes = []
    classes = []
    scores = []
    if getattr(result, "boxes", None) is not None and len(result.boxes):
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for b, c, s in zip(xyxy, cls, confs):
            boxes.append(np.asarray(b, dtype=float))
            label = names.get(int(c), str(int(c))) if isinstance(names, dict) else str(int(c))
            classes.append(normalize_class_name(label))
            scores.append(float(s))
    return {"boxes": boxes, "classes": classes, "scores": scores}


def pseudo_quality_from_orig(orig_pred: dict[str, Any], gt_boxes: list[tuple[float, float, float, float, str]], threshold: float = 0.5) -> dict[str, Any]:
    pred_boxes = orig_pred["boxes"]
    gt_xyxy = [np.asarray(b[:4], dtype=float) for b in gt_boxes]
    matches = greedy_match_iou(gt_xyxy, pred_boxes, threshold=threshold)
    matched_gt = {m[0] for m in matches}
    matched_pred = {m[1] for m in matches}
    tp = len(matches)
    pred_count = len(pred_boxes)
    gt_count = len(gt_xyxy)
    fp = pred_count - len(matched_pred)
    fn = gt_count - len(matched_gt)
    return {
        "gt_box_count": gt_count,
        "orig_pred_box_count": pred_count,
        "pseudo_tp": tp,
        "pseudo_fp": fp,
        "pseudo_fn": fn,
        "pseudo_precision": tp / pred_count if pred_count else np.nan,
        "pseudo_recall": tp / gt_count if gt_count else np.nan,
        "pseudo_fn_ratio": fn / gt_count if gt_count else np.nan,
    }


def stability_for_predictions(preds: dict[str, dict[str, Any]]) -> dict[str, float]:
    names = list(preds.keys())
    orig = preds["orig"]
    orig_boxes = orig["boxes"]
    orig_classes = orig["classes"]
    counts = np.asarray([len(preds[n]["boxes"]) for n in names], dtype=float)
    count_mean = float(np.mean(counts))
    count_std = float(np.std(counts))
    u_count = count_std / (count_mean + 1e-6)

    loc_instabilities = []
    cls_changes = []
    conf_stds = []
    match_counts = []
    for name in names:
        if name == "orig":
            continue
        alt = preds[name]
        matches = greedy_match_iou(orig_boxes, alt["boxes"], threshold=0.1)
        match_counts.append(len(matches))
        if matches:
            loc_instabilities.append(float(np.mean([1.0 - m[2] for m in matches])))
            same_cls = 0
            conf_pairs = []
            for oi, ai, _ in matches:
                if orig_classes[oi] == alt["classes"][ai]:
                    same_cls += 1
                if oi < len(orig["scores"]) and ai < len(alt["scores"]):
                    conf_pairs.append([orig["scores"][oi], alt["scores"][ai]])
            cls_changes.append(1.0 - same_cls / len(matches))
            if conf_pairs:
                conf_stds.append(float(np.mean([np.std(pair) for pair in conf_pairs])))
        else:
            loc_instabilities.append(1.0 if len(orig_boxes) or len(alt["boxes"]) else 0.0)
            cls_changes.append(1.0 if len(orig_boxes) or len(alt["boxes"]) else 0.0)
            conf_stds.append(0.0)
    return {
        "count_mean": count_mean,
        "count_std": count_std,
        "u_count": u_count,
        "u_loc": float(np.mean(loc_instabilities)) if loc_instabilities else np.nan,
        "u_cls": float(np.mean(cls_changes)) if cls_changes else np.nan,
        "u_conf": float(np.mean(conf_stds)) if conf_stds else np.nan,
        "mean_match_count_to_orig": float(np.mean(match_counts)) if match_counts else np.nan,
    }


def minmax_norm(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    lo = vals.min()
    hi = vals.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series([0.0] * len(vals), index=series.index)
    return (vals - lo) / (hi - lo)


def build_candidate_set(scores: pd.DataFrame, *, seed: int, top_k: int, random_k: int) -> pd.DataFrame:
    scores = scores.copy()
    for col in ["detector_uncertainty", "detector_pseudo_box_count", "detector_max_conf"]:
        scores[col] = pd.to_numeric(scores.get(col), errors="coerce").fillna(0.0)
    scores["_candidate_rank_score"] = (
        0.50 * minmax_norm(scores["detector_uncertainty"])
        + 0.25 * (1.0 - minmax_norm(scores["detector_max_conf"]))
        + 0.25 * minmax_norm(scores["detector_pseudo_box_count"])
    )
    top = scores.sort_values(["_candidate_rank_score", "sample_id"], ascending=[False, True], kind="mergesort").head(top_k)
    rest = scores[~scores["sample_id"].astype(str).isin(set(top["sample_id"].astype(str)))].copy()
    if random_k > 0 and len(rest):
        rnd = rest.sample(n=min(random_k, len(rest)), random_state=seed)
        out = pd.concat([top.assign(candidate_source="ranked_top"), rnd.assign(candidate_source="random_reference")], ignore_index=True, sort=False)
    else:
        out = top.assign(candidate_source="ranked_top")
    return out.drop_duplicates("sample_id", keep="first").reset_index(drop=True)


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    df = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(df) < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan
    return float(df["x"].rank().corr(df["y"].rank()))


def class_coverage(df: pd.DataFrame, class_col: str) -> int:
    classes = set()
    for value in df.get(class_col, pd.Series(dtype=str)).fillna("").astype(str):
        for part in value.split("|"):
            norm = normalize_class_name(part)
            if norm in NEU6:
                classes.add(norm)
    return len(classes)


def plot_scatter(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str) -> None:
    if x not in df.columns or y not in df.columns:
        return
    plot_df = df[[x, y, "gt_majority_class"]].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5.2), constrained_layout=True)
    for cls, sub in plot_df.groupby("gt_majority_class", dropna=False):
        ax.scatter(sub[x], sub[y], label=str(cls), alpha=0.75, s=30)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_top_group_distribution(top: pd.DataFrame, out_path: Path) -> None:
    if top.empty:
        return
    counts = Counter()
    for value in top["gt_classes"].fillna("").astype(str):
        for part in value.split("|"):
            norm = normalize_class_name(part)
            if norm:
                counts[norm] += 1
    labels = sorted(counts)
    values = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(labels, values, color="#1f77b4")
    ax.set_title("Top V10d instability group GT class coverage")
    ax.set_ylabel("image-level class count")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default=str(DEFAULT_SOURCE_RUN))
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--top-k", type=int, default=120)
    parser.add_argument("--random-k", type=int, default=60)
    parser.add_argument("--select-k", type=int, default=30)
    parser.add_argument("--device", default=os.environ.get("AL_YOLO_DEVICE", "0"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is required for V10d stability probing") from exc

    source_run = Path(args.source_run)
    detector_scores_path = source_run / f"seed{args.seed}_round1_detector_scores_v10c.csv"
    weights = source_run / "yolo_train_runs" / f"seed{args.seed}___SHARED_ROUND0___R0_trainseed{1000 + args.seed}" / "weights" / "best.pt"
    if not detector_scores_path.exists():
        raise FileNotFoundError(detector_scores_path)
    if not weights.exists():
        raise FileNotFoundError(weights)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"v10d_stability_signal_seed{args.seed}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(detector_scores_path)
    candidates = build_candidate_set(scores, seed=args.seed, top_k=args.top_k, random_k=args.random_k)
    model = YOLO(str(weights))

    rows = []
    print("=" * 100)
    print("[V10d stability signal probe]")
    print(f"Source run : {source_run}")
    print(f"Candidates : {len(candidates)}")
    print(f"Output dir : {out_dir}")
    print("No training. No final-test evaluation.")
    print("=" * 100)

    for idx, row in candidates.iterrows():
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"[{idx + 1}/{len(candidates)}] probing...")
        image_path = Path(str(row.get("resolved_image_path", row.get("image_path", ""))).split("::")[0])
        image = Image.open(image_path).convert("RGB")
        preds = {
            name: yolo_predict_image(model, img, imgsz=args.imgsz, conf=args.conf, device=args.device)
            for name, img in perturbations(image).items()
        }
        stab = stability_for_predictions(preds)
        gt = parse_gt_xml(xml_path_for_row(row))
        pseudo = pseudo_quality_from_orig(preds["orig"], gt["_gt_boxes"], threshold=args.iou_thresh)
        out = row.to_dict()
        out.update({k: v for k, v in gt.items() if k != "_gt_boxes"})
        out.update(pseudo)
        out.update(stab)
        out["gt_majority_class"] = normalize_class_name(out.get("gt_majority_class", ""))
        out["pred_class_in_gt_classes"] = normalize_class_name(out.get("detector_pred_class", "")) in set(str(out.get("gt_classes", "")).split("|"))
        rows.append(out)

    df = pd.DataFrame(rows)
    for col in ["u_count", "u_loc", "u_cls", "u_conf"]:
        df[f"{col}_norm"] = minmax_norm(df[col])
    df["u_stability_equal_weight"] = df[["u_count_norm", "u_loc_norm", "u_cls_norm", "u_conf_norm"]].mean(axis=1)
    df["u_stability_no_weight_claim"] = df["u_stability_equal_weight"]
    df = df.sort_values(["u_stability_no_weight_claim", "sample_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    df["v10d_instability_rank"] = np.arange(1, len(df) + 1)
    top = df.head(args.select_k).copy()

    corr_rows = []
    for signal in ["u_count", "u_loc", "u_cls", "u_conf", "u_stability_no_weight_claim", "detector_pseudo_box_count", "detector_uncertainty"]:
        for target in ["pseudo_fn_ratio", "pseudo_recall", "pseudo_precision", "pseudo_fn", "pred_class_in_gt_classes"]:
            if target == "pred_class_in_gt_classes":
                target_values = df[target].astype(float)
            else:
                target_values = df[target]
            corr_rows.append(
                {
                    "signal": signal,
                    "target": target,
                    "spearman": spearman_corr(df[signal], target_values),
                    "n": int(pd.DataFrame({"x": pd.to_numeric(df[signal], errors="coerce"), "y": pd.to_numeric(target_values, errors="coerce")}).dropna().shape[0]),
                }
            )
    corr = pd.DataFrame(corr_rows)

    top_summary = pd.DataFrame(
        [
            {
                "group": "all_candidates",
                "n": len(df),
                "pseudo_fn_ratio_mean": float(pd.to_numeric(df["pseudo_fn_ratio"], errors="coerce").mean()),
                "pseudo_recall_mean": float(pd.to_numeric(df["pseudo_recall"], errors="coerce").mean()),
                "pseudo_precision_mean": float(pd.to_numeric(df["pseudo_precision"], errors="coerce").mean()),
                "gt_class_coverage": class_coverage(df, "gt_classes"),
                "pred_class_coverage": class_coverage(df, "detector_pred_class"),
                "mean_pairwise_claim": "not_computed_in_this_probe",
            },
            {
                "group": f"top_{args.select_k}_instability",
                "n": len(top),
                "pseudo_fn_ratio_mean": float(pd.to_numeric(top["pseudo_fn_ratio"], errors="coerce").mean()),
                "pseudo_recall_mean": float(pd.to_numeric(top["pseudo_recall"], errors="coerce").mean()),
                "pseudo_precision_mean": float(pd.to_numeric(top["pseudo_precision"], errors="coerce").mean()),
                "gt_class_coverage": class_coverage(top, "gt_classes"),
                "pred_class_coverage": class_coverage(top, "detector_pred_class"),
                "mean_pairwise_claim": "not_computed_in_this_probe",
            },
        ]
    )

    all_fn = float(top_summary.loc[top_summary["group"].eq("all_candidates"), "pseudo_fn_ratio_mean"].iloc[0])
    top_fn = float(top_summary.loc[top_summary["group"].str.startswith("top_"), "pseudo_fn_ratio_mean"].iloc[0])
    top_recall = float(top_summary.loc[top_summary["group"].str.startswith("top_"), "pseudo_recall_mean"].iloc[0])
    top_cov = int(top_summary.loc[top_summary["group"].str.startswith("top_"), "gt_class_coverage"].iloc[0])
    top_gt_classes = set()
    for value in top["gt_classes"].fillna("").astype(str):
        top_gt_classes.update(normalize_class_name(x) for x in value.split("|") if x)
    gate_rows = [
        {"criterion": "top_instability_fn_ratio_gt_all_candidates", "pass": bool(top_fn > all_fn), "value": top_fn, "reference": all_fn},
        {"criterion": "top_instability_pseudo_recall_not_extreme_low", "pass": bool(top_recall >= 0.50), "value": top_recall, "reference": 0.50},
        {"criterion": "top_instability_gt_class_coverage_at_least_4", "pass": bool(top_cov >= 4), "value": top_cov, "reference": 4},
        {"criterion": "inclusion_present_in_top_group", "pass": "inclusion" in top_gt_classes, "value": "inclusion" in top_gt_classes, "reference": True},
        {"criterion": "patches_present_in_top_group", "pass": "patches" in top_gt_classes, "value": "patches" in top_gt_classes, "reference": True},
    ]
    gate = pd.DataFrame(gate_rows)

    df.to_csv(out_dir / "v10d_stability_candidate_scores.csv", index=False, encoding="utf-8-sig")
    top.to_csv(out_dir / "v10d_top_instability_selection.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(out_dir / "v10d_signal_error_correlations.csv", index=False, encoding="utf-8-sig")
    top_summary.to_csv(out_dir / "v10d_top_group_summary.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(out_dir / "v10d_selection_only_gate.csv", index=False, encoding="utf-8-sig")

    plot_scatter(df, "u_stability_no_weight_claim", "pseudo_fn_ratio", out_dir / "fig01_stability_vs_pseudo_fn_ratio.png", "V10d stability vs pseudo FN ratio")
    plot_scatter(df, "u_stability_no_weight_claim", "pseudo_recall", out_dir / "fig02_stability_vs_pseudo_recall.png", "V10d stability vs pseudo recall")
    plot_scatter(df, "detector_pseudo_box_count", "pseudo_fn_ratio", out_dir / "fig03_box_count_vs_pseudo_fn_ratio.png", "Pseudo box count vs pseudo FN ratio")
    plot_top_group_distribution(top, out_dir / "fig04_top_instability_gt_class_distribution.png")

    config = {
        "source_run": str(source_run),
        "seed": args.seed,
        "weights": str(weights),
        "detector_scores": str(detector_scores_path),
        "top_k": args.top_k,
        "random_k": args.random_k,
        "select_k": args.select_k,
        "perturbations": list(perturbations(Image.new("RGB", (8, 8))).keys()),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou_thresh": args.iou_thresh,
        "final_test_used": False,
        "gt_used_for_posthoc_audit_only": True,
        "no_training": True,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# V10d stability signal probe",
        "",
        "Selection-only validity audit. No training and no final-test evaluation.",
        "",
        "## Gate",
        "",
        md_table(gate),
        "",
        "## Top group summary",
        "",
        md_table(top_summary),
        "",
        "## Signal-error correlations",
        "",
        md_table(corr.sort_values("spearman", ascending=False, na_position="last").head(20)),
        "",
        "## Interpretation rule",
        "",
        "Proceed to training only if instability predicts pseudo FN / low pseudo recall better than pseudo box count, while preserving class coverage and avoiding noisy recall collapse.",
    ]
    (out_dir / "v10d_stability_signal_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("=" * 100)
    print("[DONE] V10d stability signal probe finished")
    print(f"Output dir: {out_dir}")
    print(f"Candidates scored: {len(df)}")
    print(f"Top selection size: {len(top)}")
    print(f"Gate passed: {int(gate['pass'].sum())}/{len(gate)}")
    print("No training. Final test used=False. GT used only post-hoc=True.")
    print("=" * 100)


if __name__ == "__main__":
    main()


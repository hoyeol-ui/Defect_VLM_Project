"""Recover per-class metrics from existing V7 round-4 checkpoints.

This is evaluation-only:
- uses existing best.pt checkpoints
- uses each run's existing development_eval_v7 data.yaml
- never calls train()
- never opens final_test_v7
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL_RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "active_learning_ablation_v7_full_curve"
    / "v7_full_curve_20260711_211848"
)
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "v7_round4_per_class_recovery"
RANDOM = "GTFreeRandom"
VISUAL = "GTFreeDatasetBalancedVisualDiversity"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_development_only(yaml_path: Path) -> None:
    text = yaml_path.read_text(encoding="utf-8")
    lowered = text.lower().replace("\\", "/")
    if "final_test" in lowered or "final-test" in lowered:
        raise RuntimeError(f"Refusing to evaluate a final-test yaml: {yaml_path}")
    if "images/val" not in lowered:
        raise RuntimeError(f"Expected development val split in yaml, got: {yaml_path}")


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def main() -> None:
    out_dir = OUTPUT_ROOT / f"round4_per_class_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)

    full_cfg = json.loads((FULL_RUN_DIR / "config.json").read_text(encoding="utf-8"))
    if bool(full_cfg.get("final_test_used", False)):
        raise RuntimeError("Full-curve config says final_test_used=True; refusing to run.")

    all_results = pd.read_csv(FULL_RUN_DIR / "all_round_results.csv")
    targets = all_results[
        all_results["round"].eq(4)
        & all_results["strategy"].isin([RANDOM, VISUAL])
    ].sort_values(["acquisition_seed", "strategy"], kind="mergesort")

    per_class_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []

    for _, row in targets.iterrows():
        seed = int(row["acquisition_seed"])
        strategy = str(row["strategy"])
        yaml_path = Path(str(row["yaml_path"]))
        ckpt = Path(str(row["train_run_dir"])) / "weights" / "best.pt"
        ensure_development_only(yaml_path)
        if not ckpt.exists():
            registry_rows.append(
                {
                    "acquisition_seed": seed,
                    "strategy": strategy,
                    "round": 4,
                    "checkpoint_path": str(ckpt),
                    "checkpoint_exists": False,
                    "status": "missing_checkpoint",
                }
            )
            continue

        ckpt_sha = file_sha256(ckpt)
        val_name = f"seed{seed}_{strategy}_round4_dev_eval"
        print(f"[VAL ONLY] seed={seed} strategy={strategy} checkpoint={ckpt.name}")
        model = YOLO(str(ckpt))
        metrics = model.val(
            data=str(yaml_path),
            split="val",
            imgsz=640,
            batch=8,
            device=0,
            workers=0,
            plots=False,
            save_json=False,
            verbose=False,
            project=str(out_dir / "ultralytics_val_runs"),
            name=val_name,
            exist_ok=True,
        )

        registry_rows.append(
            {
                "acquisition_seed": seed,
                "strategy": strategy,
                "round": 4,
                "yaml_path": str(yaml_path),
                "checkpoint_path": str(ckpt),
                "checkpoint_exists": True,
                "checkpoint_sha256": ckpt_sha,
                "development_eval_only": True,
                "final_test_used": False,
                "status": "success",
                "aggregate_map50_recovered": to_float(metrics.results_dict.get("metrics/mAP50(B)")),
                "aggregate_map5095_recovered": to_float(metrics.results_dict.get("metrics/mAP50-95(B)")),
                "aggregate_map50_original": to_float(row.get("map50")),
                "aggregate_map5095_original": to_float(row.get("map5095")),
                "save_dir": str(metrics.save_dir),
            }
        )

        for item in metrics.summary():
            per_class_rows.append(
                {
                    "acquisition_seed": seed,
                    "strategy": strategy,
                    "round": 4,
                    "training_seed": int(row["training_seed"]),
                    "class_name": item.get("Class"),
                    "val_images": to_float(item.get("Images")),
                    "val_instances": to_float(item.get("Instances")),
                    "precision": to_float(item.get("Box-P")),
                    "recall": to_float(item.get("Box-R")),
                    "f1": to_float(item.get("Box-F1")),
                    "ap50": to_float(item.get("mAP50")),
                    "ap5095": to_float(item.get("mAP50-95")),
                    "checkpoint_sha256": ckpt_sha,
                }
            )

    registry = pd.DataFrame(registry_rows)
    per_class = pd.DataFrame(per_class_rows)
    registry.to_csv(out_dir / "checkpoint_evaluation_registry.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(out_dir / "recovered_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    diffs: list[dict[str, Any]] = []
    if not per_class.empty:
        v = per_class[per_class["strategy"].eq(VISUAL)]
        r = per_class[per_class["strategy"].eq(RANDOM)]
        merged = v.merge(
            r,
            on=["acquisition_seed", "round", "class_name"],
            suffixes=("_visual", "_random"),
        )
        for _, m in merged.iterrows():
            diffs.append(
                {
                    "acquisition_seed": int(m["acquisition_seed"]),
                    "round": 4,
                    "class_name": m["class_name"],
                    "val_instances": m["val_instances_visual"],
                    "ap50_visual": m["ap50_visual"],
                    "ap50_random": m["ap50_random"],
                    "ap50_visual_minus_random": m["ap50_visual"] - m["ap50_random"],
                    "ap5095_visual": m["ap5095_visual"],
                    "ap5095_random": m["ap5095_random"],
                    "ap5095_visual_minus_random": m["ap5095_visual"] - m["ap5095_random"],
                    "recall_visual": m["recall_visual"],
                    "recall_random": m["recall_random"],
                    "recall_visual_minus_random": m["recall_visual"] - m["recall_random"],
                    "precision_visual": m["precision_visual"],
                    "precision_random": m["precision_random"],
                    "precision_visual_minus_random": m["precision_visual"] - m["precision_random"],
                }
            )
    diff_df = pd.DataFrame(diffs)
    diff_df.to_csv(out_dir / "per_class_visual_minus_random.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    if not diff_df.empty:
        for seed, sub in diff_df.groupby("acquisition_seed"):
            weighted_gap = (sub["ap5095_visual_minus_random"] * sub["val_instances"]).sum() / sub["val_instances"].sum()
            worst = sub.sort_values("ap5095_visual_minus_random", kind="mergesort").head(3)
            best = sub.sort_values("ap5095_visual_minus_random", ascending=False, kind="mergesort").head(3)
            summary_rows.append(
                {
                    "acquisition_seed": seed,
                    "weighted_ap5095_gap_visual_minus_random": weighted_gap,
                    "mean_ap5095_gap_visual_minus_random": sub["ap5095_visual_minus_random"].mean(),
                    "worst_classes_for_visual": "; ".join(
                        f"{x.class_name}({x.ap5095_visual_minus_random:+.4f})" for x in worst.itertuples()
                    ),
                    "best_classes_for_visual": "; ".join(
                        f"{x.class_name}({x.ap5095_visual_minus_random:+.4f})" for x in best.itertuples()
                    ),
                }
            )
    seed_summary = pd.DataFrame(summary_rows)
    seed_summary.to_csv(out_dir / "seed_per_class_gap_summary.csv", index=False, encoding="utf-8-sig")

    report = [
        "# V7 round-4 per-class recovery\n\n",
        f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "This run used existing round-4 Random/Visual `best.pt` checkpoints and each run's existing development `data.yaml`. No training and no final-test evaluation were performed.\n\n",
        "## Seed-level class gap summary\n\n",
        seed_summary.to_markdown(index=False) if not seed_summary.empty else "_No seed summary._",
        "\n\n## Registry\n\n",
        registry[["acquisition_seed", "strategy", "status", "checkpoint_exists", "aggregate_map5095_recovered", "aggregate_map5095_original"]].to_markdown(index=False)
        if not registry.empty
        else "_No registry._",
        "\n",
    ]
    (out_dir / "per_class_recovery_summary.md").write_text("".join(report), encoding="utf-8")

    print("=" * 100)
    print("[DONE] V7 round-4 per-class recovery")
    print(f"Output dir: {out_dir}")
    print("No YOLO training. No final-test evaluation.")
    print("=" * 100)


if __name__ == "__main__":
    main()


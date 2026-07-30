"""Post-hoc, training-free diagnosis of the frozen V2.2 four-model smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715"
DEFAULT_SMOKE = ROOT / "runs" / "dcal_xai" / "v2_backbone_smoke"
POLICIES = ["Random140", "ClusterK40_140"]
BACKBONES = ["YOLOv8n", "YOLOv8s"]
RARE_IDS = {8, 9, 10}


def training_dynamics(smoke: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((smoke / "train_runs").glob("*/results.csv")):
        frame = pd.read_csv(path)
        name = path.parent.name
        policy = "ClusterK40_140" if "ClusterK40_140" in name else "Random140"
        backbone = "YOLOv8s" if "YOLOv8s" in name else "YOLOv8n"
        tail = frame.tail(min(10, len(frame)))
        x = np.arange(len(tail), dtype=float)
        row = {"policy": policy, "backbone": backbone, "epochs": len(frame)}
        for column, short in [
            ("train/box_loss", "box"), ("train/cls_loss", "cls"),
            ("train/dfl_loss", "dfl"),
        ]:
            values = tail[column].to_numpy(float)
            row[f"final_{short}_loss"] = float(frame.iloc[-1][column])
            row[f"last10_{short}_slope"] = float(np.polyfit(x, values, 1)[0])
        rows.append(row)
    return pd.DataFrame(rows)


def class_support(smoke: Path, per_class: pd.DataFrame) -> pd.DataFrame:
    selections = pd.read_csv(smoke / "frozen_initial_sets.csv")
    selections = selections[selections["acquisition_seed"].eq(20000)]
    gt = pd.read_csv(PROTOCOL / "gc10_acquisition_pool_gt_audit.csv")
    boxes = pd.read_csv(PROTOCOL / "gc10_acquisition_bbox_gt_audit.csv")
    rows = []
    for policy in POLICIES:
        ids = set(selections[selections["policy"].eq(policy)]["sample_id"].astype(str))
        chosen_gt = gt[gt["sample_id"].astype(str).isin(ids)]
        chosen_boxes = boxes[boxes["sample_id"].astype(str).isin(ids)]
        for class_id in range(1, 11):
            image_count = int(chosen_gt["class_ids"].fillna("").astype(str).map(
                lambda value: class_id in {int(item) for item in value.split("|") if item}
            ).sum())
            instance_count = int(chosen_boxes["class_id"].eq(class_id).sum())
            for backbone in BACKBONES:
                ap = per_class[
                    per_class["policy"].eq(policy)
                    & per_class["backbone"].eq(backbone)
                    & per_class["class_id"].eq(class_id)
                ]
                if len(ap) != 1:
                    raise RuntimeError("Smoke per-class cardinality failure")
                rows.append({
                    "policy": policy, "backbone": backbone, "class_id": class_id,
                    "class_name": str(ap.iloc[0]["class_name"]),
                    "rare": class_id in RARE_IDS, "selected_images": image_count,
                    "selected_instances": instance_count, "ap5095": float(ap.iloc[0]["ap5095"]),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE)
    args = parser.parse_args()
    smoke = args.smoke_dir.expanduser().resolve()
    validation = json.loads((smoke / "smoke_validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or validation.get("final_test_used") is not False:
        raise RuntimeError("Technical smoke validation did not pass safely")
    aggregate = pd.read_csv(smoke / "smoke_metrics.csv")
    per_class = pd.read_csv(smoke / "smoke_per_class.csv")
    support = class_support(smoke, per_class)
    dynamics = training_dynamics(smoke)

    aggregate_pivot = aggregate.pivot(index="backbone", columns="policy")
    contrast_rows = []
    for backbone in BACKBONES:
        subset = support[support["backbone"].eq(backbone)]
        class_pivot = subset.pivot(index=["class_id", "class_name", "rare"], columns="policy", values="ap5095")
        differences = class_pivot["ClusterK40_140"] - class_pivot["Random140"]
        contrast_rows.append({
            "backbone": backbone,
            "map5095_k40_minus_random": float(
                aggregate_pivot.loc[backbone, ("map5095", "ClusterK40_140")]
                - aggregate_pivot.loc[backbone, ("map5095", "Random140")]
            ),
            "recall_k40_minus_random": float(
                aggregate_pivot.loc[backbone, ("recall", "ClusterK40_140")]
                - aggregate_pivot.loc[backbone, ("recall", "Random140")]
            ),
            "rare_macro_ap_k40_minus_random": float(differences[differences.index.get_level_values("rare")].mean()),
            "worst_class_ap_difference": float(differences.min()),
            "classes_improved": int((differences > 0).sum()),
            "classes_tied": int((differences == 0).sum()),
            "classes_degraded": int((differences < 0).sum()),
        })
    contrasts = pd.DataFrame(contrast_rows)

    support.to_csv(smoke / "smoke_class_support_posthoc.csv", index=False, encoding="utf-8-sig")
    dynamics.to_csv(smoke / "smoke_training_dynamics.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(smoke / "smoke_performance_contrasts.csv", index=False, encoding="utf-8-sig")
    report = [
        "# V2.2 Smoke Post-hoc Diagnosis", "",
        "- Technical smoke: **PASS**", "- Inferential performance claim: **not permitted (one acquisition/training seed)**",
        "- Recommended next stage: **YOLOv8n-only frozen screen before any v8s main expansion**",
        "- Final test used: **False**", "", "## Performance warning", "",
        contrasts.to_markdown(index=False, floatfmt=".6f"), "", "## Training dynamics", "",
        dynamics.to_markdown(index=False, floatfmt=".6f"), "",
        "K40 coverage is not detector utility. The full 60-model factorial should not be launched until the cheaper YOLOv8n screen is prospectively frozen.",
    ]
    path = smoke / "smoke_posthoc_diagnosis.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[DONE] {path}")
    print("Training performed: False")
    print("Final test used: False")


if __name__ == "__main__":
    main()

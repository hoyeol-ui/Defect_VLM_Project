"""Compare frozen paired-compliance gates across VLMs without changing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


GATE_METRICS = {
    "json_parse_rate": 0.90,
    "schema_compliance_rate": 0.90,
    "positive_sensitivity": 0.70,
    "background_specificity": 0.70,
    "balanced_accuracy": 0.70,
    "positive_bbox_coverage": 0.70,
    "positive_median_bbox_iou": 0.10,
}


def parse_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Model spec must be NAME=PATH_TO_AUDIT_DIR")
    name, path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("Model name cannot be empty")
    return name.strip(), Path(path.strip())


def load_model_row(name: str, audit_dir: Path) -> dict[str, object]:
    metrics_path = audit_dir / "paired_compliance_metrics.csv"
    gate_path = audit_dir / "paired_compliance_gate.csv"
    if not metrics_path.is_file() or not gate_path.is_file():
        raise FileNotFoundError(f"Missing audit outputs in {audit_dir}")
    metrics = pd.read_csv(metrics_path).set_index("metric")["value"].to_dict()
    gate = pd.read_csv(gate_path)
    row: dict[str, object] = {"model": name, "audit_dir": str(audit_dir.resolve())}
    for metric in GATE_METRICS:
        row[metric] = float(metrics.get(metric, float("nan")))
    row["gate_pass"] = bool((gate["result"] == "PASS").all())
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=parse_spec, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.model) < 2:
        raise RuntimeError("At least two model audits are required")

    rows = pd.DataFrame(load_model_row(name, path) for name, path in args.model)
    if rows["model"].duplicated().any():
        raise RuntimeError("Duplicate model name")
    pass_count = int(rows["gate_pass"].sum())
    if pass_count:
        passed_names = ", ".join(rows.loc[rows["gate_pass"], "model"])
        decision = (
            f"At least one frozen gate passed ({passed_names}). The Qwen2-VL-2B failure is "
            "model/capacity-specific within this comparison; only passing models may proceed to "
            "a separately planned GT-free blind-tiling pilot."
        )
    else:
        decision = (
            "No tested model passed the frozen paired gate. Within this model family and hardware-"
            "feasible scale, VLM structured consistency is not supported as an acquisition signal. "
            "Stop prompt/model iteration and retain the result as a boundary/failure finding."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "paired_model_gate_comparison.csv", index=False, encoding="utf-8-sig")
    config = {
        "models": [{"name": name, "audit_dir": str(path.resolve())} for name, path in args.model],
        "frozen_thresholds": GATE_METRICS,
        "detector_training_performed": False,
        "final_test_evaluated": False,
    }
    (args.output_dir / "comparison_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    display_columns = ["model", *GATE_METRICS.keys(), "gate_pass"]
    summary = [
        "# Frozen Paired VLM Gate Comparison",
        "",
        "- Detector training performed: **False**",
        "- Final test evaluated: **False**",
        "- Gate thresholds changed after seeing results: **False**",
        "",
        "## Model comparison", "", rows[display_columns].to_markdown(index=False), "",
        "## Decision", "", decision, "",
    ]
    summary_path = args.output_dir / "paired_model_gate_comparison_summary.md"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print(f"[DONE] models={len(rows)} passed={pass_count}")
    print(f"[SUMMARY] {summary_path}")


if __name__ == "__main__":
    main()

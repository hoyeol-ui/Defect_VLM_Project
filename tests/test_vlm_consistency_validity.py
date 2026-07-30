from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_legacy_extract_row_normalizes_two_point_groundedness():
    module = load_module(
        "legacy_validity",
        "scripts/03_analysis/audit_legacy_vlm_consistency_validity.py",
    )
    row = module.extract_row(
        {
            "image_name": "sample.jpg",
            "dataset_type": "NEU-DET",
            "consistency_score": 0.75,
            "groundedness": {
                "total_score": 1.0,
                "primary_location_correct": True,
                "primary_scale_correct": False,
            },
        }
    )
    assert row["groundedness_norm"] == 0.5
    assert row["inconsistency_score"] == 0.25
    assert row["severe_grounding_failure"] == 0
    assert row["any_grounding_error"] == 1


def test_legacy_metrics_detect_monotonic_validity_signal():
    module = load_module(
        "legacy_validity_metrics",
        "scripts/03_analysis/audit_legacy_vlm_consistency_validity.py",
    )
    consistency = np.linspace(0.1, 0.9, 40)
    grounded = np.concatenate([np.zeros(20), np.ones(20)])
    df = pd.DataFrame(
        {
            "dataset_type": ["A"] * 20 + ["B"] * 20,
            "consistency_score": consistency,
            "inconsistency_score": 1.0 - consistency,
            "groundedness_norm": grounded,
            "severe_grounding_failure": 1.0 - grounded,
        }
    )
    metrics = module.compute_metrics(df, module.BootstrapConfig(reps=100, seed=7))
    overall = metrics[metrics["scope"] == "overall"].set_index("metric")
    assert overall.loc["spearman_consistency_vs_groundedness", "value"] > 0.8
    assert overall.loc["inconsistency_auc_severe_failure", "value"] > 0.9


def test_structured_response_scoring_uses_field_and_box_agreement():
    module = load_module(
        "structured_validity",
        "scripts/01_score_generation/score_structured_vlm_validity.py",
    )
    base = {
        "defect_present": True,
        "defect_type": "scratch",
        "bbox_norm": [0.1, 0.1, 0.4, 0.4],
        "location_zone": "top-left",
        "scale": "small",
        "appearance": "thin dark line",
        "visual_evidence": "visible line",
        "abstain": False,
        "confidence": 0.8,
    }
    records = []
    for index in range(5):
        value = dict(base)
        if index == 4:
            value["location_zone"] = "center-left"
        records.append(
            {
                "image_id": "x",
                "image_path": "x.jpg",
                "dataset": "demo",
                "split_role": "development_pilot",
                "raw_response": json.dumps(value),
            }
        )
    result = module.score_image(records)
    assert result["valid_response_rate"] == 1.0
    assert result["presence_agreement"] == 1.0
    assert result["class_agreement"] == 1.0
    assert result["location_agreement"] == 0.8
    assert result["bbox_iou_agreement"] == 1.0
    assert result["consensus_defect_type"] == "scratch"
    assert result["consensus_bbox_x1"] == 0.1
    assert 0.9 < result["structured_consistency"] < 1.0


def test_structured_parser_accepts_fenced_json():
    module = load_module(
        "structured_validity_parser",
        "scripts/01_score_generation/score_structured_vlm_validity.py",
    )
    parsed = module.parse_json_object('```json\n{"abstain": true}\n```')
    assert parsed == {"abstain": True}


def test_null_structured_fields_do_not_create_false_consistency():
    module = load_module(
        "structured_validity_nulls",
        "scripts/01_score_generation/score_structured_vlm_validity.py",
    )
    raw = json.dumps(
        {
            "defect_present": None,
            "defect_type": None,
            "bbox_norm": None,
            "location_zone": None,
            "scale": None,
            "appearance": None,
            "visual_evidence": None,
            "abstain": None,
            "confidence": 0.0,
        }
    )
    result = module.score_image(
        [{"image_id": "x", "raw_response": raw} for _ in range(5)]
    )
    assert result["valid_response_rate"] == 1.0
    assert result["schema_complete_response_rate"] == 0.0
    assert result["informative_response_rate"] == 0.0
    assert np.isnan(result["structured_consistency"])


def test_gc10_posthoc_geometry_helpers():
    module = load_module(
        "structured_gc10_audit",
        "scripts/03_analysis/audit_structured_vlm_gc10_groundedness.py",
    )
    assert module.box_iou([0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.4, 0.4]) == 1.0
    zone, scale = module.zone_and_scale([0.0, 0.0, 0.1, 0.1])
    assert zone == "top-left"
    assert scale == "small"

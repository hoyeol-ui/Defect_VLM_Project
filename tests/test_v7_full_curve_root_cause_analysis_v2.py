import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "02_active_learning" / "analyze_v7_full_curve_root_causes_v2.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_v7_full_curve_root_causes_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_canonical_identity_uses_relative_path_and_sha(tmp_path):
    mod = load_module()
    img = tmp_path / "a.jpg"
    img.write_bytes(b"same-image")
    idx = {("X", "a.jpg"): [img]}
    df = pd.DataFrame([{"dataset_type": "X", "image_name": "a.jpg"}])

    out = mod.annotate_identity(df, idx)

    assert out.loc[0, "canonical_image_sha256"]
    assert "|" in out.loc[0, "canonical_sample_id"]


def test_failure_seeds_are_metric_driven(tmp_path):
    mod = load_module()
    df = pd.DataFrame(
        [
            {"acquisition_seed": 1, "strategy": mod.VISUAL, "normalized_aulc_map5095": 0.2, "final_map5095": 0.2, "normalized_aulc_map50": 0.3, "final_map50": 0.3},
            {"acquisition_seed": 1, "strategy": mod.RANDOM, "normalized_aulc_map5095": 0.1, "final_map5095": 0.1, "normalized_aulc_map50": 0.2, "final_map50": 0.2},
            {"acquisition_seed": 2, "strategy": mod.VISUAL, "normalized_aulc_map5095": 0.1, "final_map5095": 0.05, "normalized_aulc_map50": 0.1, "final_map50": 0.1},
            {"acquisition_seed": 2, "strategy": mod.RANDOM, "normalized_aulc_map5095": 0.2, "final_map5095": 0.2, "normalized_aulc_map50": 0.2, "final_map50": 0.2},
        ]
    )

    fail, _, _ = mod.failure_seeds(df, tmp_path)

    row1 = fail[fail["acquisition_seed"].eq(1)].iloc[0]
    row2 = fail[fail["acquisition_seed"].eq(2)].iloc[0]
    assert not bool(row1["visual_fails_normalized_aulc_map5095"])
    assert bool(row2["visual_fails_normalized_aulc_map5095"])


def test_parse_yolo_label_separates_actual_bbox_from_pseudo(tmp_path):
    mod = load_module()
    label = tmp_path / "x.txt"
    label.write_text("0 0.5 0.5 0.1 0.2\n1 0.2 0.2 0.01 0.01\n", encoding="utf-8")

    parsed = mod.parse_yolo_label(label)

    assert parsed["actual_bbox_count"] == 2
    assert parsed["actual_unique_class_count"] == 2
    assert parsed["actual_small_bbox_count"] == 1


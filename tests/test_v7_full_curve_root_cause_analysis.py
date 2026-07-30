import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "02_active_learning" / "analyze_v7_full_curve_root_causes.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_v7_full_curve_root_causes", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_cumulative_gate_set_falls_back_to_dataset_image_key_without_sample_id():
    mod = load_module()
    df = pd.DataFrame(
        [
            {"strategy": "GTFreeRandom", "round": 0, "dataset_type": "NEU-DET", "image_name": "a.jpg"},
            {"strategy": "GTFreeRandom", "round": 1, "dataset_type": "NEU-DET", "image_name": "b.jpg"},
            {"strategy": "GTFreeRandom", "round": 1, "dataset_type": "NEU-DET", "image_name": "b.jpg"},
            {"strategy": "Other", "round": 0, "dataset_type": "NEU-DET", "image_name": "c.jpg"},
        ]
    )

    out = mod.cumulative_gate_set(df, "GTFreeRandom", 1)

    assert len(out) == 2
    assert set(out["sample_key"]) == {"NEU-DET::a.jpg", "NEU-DET::b.jpg"}


def test_corr_pair_returns_nan_for_too_few_points():
    mod = load_module()
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 4.0]})

    out = mod.corr_pair(df, "x", "y")

    assert out["n"] == 2
    assert pd.isna(out["pearson"])
    assert pd.isna(out["spearman"])


def test_bootstrap_ci_single_value_is_that_value():
    mod = load_module()

    low, high = mod.bootstrap_ci([0.123], n_boot=10, seed=1)

    assert low == 0.123
    assert high == 0.123


"""Training-free unit tests for V2.3 detector signal validity."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import run_v2_detector_signal_validity as audit


def test_class_aware_matching() -> None:
    gt = pd.DataFrame([
        {"class_id": 1, "x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100},
        {"class_id": 9, "x_min": 200, "y_min": 200, "x_max": 300, "y_max": 300},
    ])
    predictions = [
        {"class_index": 0, "confidence": 0.9, "xyxy": [0, 0, 100, 100]},
        {"class_index": 7, "confidence": 0.8, "xyxy": [200, 200, 300, 300]},
    ]
    result = audit.match_errors(predictions, gt)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["rare_fn"] == 1


def synthetic_valid_signal() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensemble_rows = []
    merged_rows = []
    for acquisition_seed in audit.ACQUISITION_SEEDS:
        for image_index in range(20):
            is_error = int(image_index >= 16)
            score = float(image_index / 19)
            ensemble_rows.append({
                "acquisition_seed": acquisition_seed,
                "sample_id": f"{acquisition_seed}_{image_index}",
                "ensemble_confidence_uncertainty": score,
                "ensemble_disagreement_uncertainty": score,
                audit.PRIMARY_SIGNAL: score,
                "mean_error_count": float(is_error),
                "mean_fn_count": float(is_error),
                "mean_rare_fn_count": float(is_error),
                "majority_error": is_error,
            })
            for training_seed in audit.TRAINING_SEEDS:
                merged_rows.append({
                    "acquisition_seed": acquisition_seed,
                    "training_seed": training_seed,
                    "sample_id": f"{acquisition_seed}_{image_index}",
                    "confidence_deficit": score,
                })
    return pd.DataFrame(ensemble_rows), pd.DataFrame(merged_rows)


def test_valid_signal_passes_frozen_gate() -> None:
    ensemble, merged = synthetic_valid_signal()
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp)
        _, summary, stability = audit.evaluate_signals(ensemble, merged, out)
        passed, path = audit.decide(summary, stability, out)
        assert passed
        assert path.exists()
        decision = json.loads((out / "decision.json").read_text(encoding="utf-8"))
        assert decision["authorizes_selection_only_query_audit"] is True
        assert np.isclose(decision["mean_training_seed_rank_stability"], 1.0)


def main() -> None:
    tests = [test_class_aware_matching, test_valid_signal_passes_frozen_gate]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"[DONE] {len(tests)} tests passed")


if __name__ == "__main__":
    main()

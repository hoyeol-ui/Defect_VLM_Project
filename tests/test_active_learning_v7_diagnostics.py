"""Small synthetic checks for V7 methodology-audit helpers.

Run:
    .\.python311\python.exe .\tests\test_active_learning_v7_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "02_active_learning"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_detection_pipeline_v7 as audit  # noqa: E402
import run_training_variance_v7 as variance  # noqa: E402


def test_hamming_hex() -> None:
    assert audit.hamming_hex("0", "0") == 0
    assert audit.hamming_hex("0", "f") == 4
    assert audit.hamming_hex(None, "f") is None


def test_class_hint_mismatch_report() -> None:
    conversion_df = pd.DataFrame(
        [
            {
                "split": "priority_pool",
                "dataset_type": "GC10-DET",
                "image_name": "a.jpg",
                "class_hint": "crease",
                "mapped_class": "crease",
                "conversion_status": "converted",
            },
            {
                "split": "priority_pool",
                "dataset_type": "GC10-DET",
                "image_name": "b.jpg",
                "class_hint": "crease",
                "mapped_class": "rolled_pit",
                "conversion_status": "converted",
            },
        ]
    )
    mismatch = audit.make_class_hint_mismatch_report(conversion_df)
    assert len(mismatch) == 1
    assert mismatch.iloc[0]["image_name"] == "b.jpg"
    assert mismatch.iloc[0]["mapped_classes_in_xml"] == "rolled_pit"


def test_signal_to_noise_handles_dry_run_nans() -> None:
    summary_df = pd.DataFrame(
        [
            {"strategy": "GTFreeRandom", "metric": "map50", "mean": np.nan, "std": np.nan},
            {"strategy": "GTFreeConsistency", "metric": "map50", "mean": np.nan, "std": np.nan},
        ]
    )
    out = variance.make_signal_to_noise(summary_df)
    assert len(out) == 1
    assert out.iloc[0]["interpretation"] == "not_available_in_dry_run_or_single_seed"


def main() -> None:
    test_hamming_hex()
    test_class_hint_mismatch_report()
    test_signal_to_noise_handles_dry_run_nans()
    print("V7 diagnostic helper tests passed.")


if __name__ == "__main__":
    main()

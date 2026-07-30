"""Training-free tests for the V2.2 factorial analysis and cache guards."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import run_v2_backbone as study


def synthetic_main_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows = []
    per_class_rows = []
    map_values = {
        ("Random140", "YOLOv8n"): 0.200,
        ("ClusterK40_140", "YOLOv8n"): 0.210,
        ("Random140", "YOLOv8s"): 0.220,
        ("ClusterK40_140", "YOLOv8s"): 0.235,
    }
    for acquisition_seed in study.ACQUISITION_SEEDS:
        for policy in study.POLICIES:
            for backbone in study.BACKBONES:
                value = map_values[(policy, backbone)]
                for training_seed in study.TRAINING_SEEDS:
                    common = {
                        "acquisition_seed": acquisition_seed,
                        "policy": policy,
                        "backbone": backbone,
                        "training_seed": training_seed,
                        "runtime_seconds": 1.0,
                    }
                    aggregate_rows.append({
                        **common,
                        "map5095": value,
                        "map50": value + 0.1,
                        "precision": value + 0.2,
                        "recall": value + 0.15,
                    })
                    for class_id, class_name in enumerate(study.CLASS_NAMES, start=1):
                        per_class_rows.append({
                            **common,
                            "class_id": class_id,
                            "class_name": class_name,
                            "ap5095": value,
                        })
    return pd.DataFrame(aggregate_rows), pd.DataFrame(per_class_rows)


def test_factorial_interaction_and_gate() -> None:
    aggregate, per_class = synthetic_main_results()
    with tempfile.TemporaryDirectory() as temp:
        passed, _ = study.analyze_main(aggregate, per_class, Path(temp))
        assert passed
        decision = json.loads((Path(temp) / "decision.json").read_text(encoding="utf-8"))
        assert decision["selected_backbone"] == "YOLOv8s"
        contrasts = pd.read_csv(Path(temp) / "factorial_contrasts.csv")
        row = contrasts[
            contrasts["contrast"].eq("interaction|(K40-Random)x(v8s-v8n)")
            & contrasts["metric"].eq("map5095")
        ]
        assert len(row) == 1
        assert np.isclose(float(row.iloc[0]["mean_difference"]), 0.005)


def test_partial_cache_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temp:
        aggregate = Path(temp) / "aggregate.json"
        per_class = Path(temp) / "per_class.json"
        aggregate.write_text("{}", encoding="utf-8")
        try:
            study.recover_cached_result(aggregate, per_class, "expected")
        except RuntimeError as error:
            assert "Partial result cache" in str(error)
        else:
            raise AssertionError("Partial cache was accepted")


def test_smoke_validation_serializes_numpy_booleans() -> None:
    aggregate, per_class = synthetic_main_results()
    aggregate = aggregate[
        aggregate["acquisition_seed"].eq(study.ACQUISITION_SEEDS[0])
        & aggregate["training_seed"].eq(study.TRAINING_SEEDS[0])
    ].copy()
    per_class = per_class[
        per_class["acquisition_seed"].eq(study.ACQUISITION_SEEDS[0])
        & per_class["training_seed"].eq(study.TRAINING_SEEDS[0])
    ].copy()
    aggregate["evaluation_split"] = "development"
    aggregate["final_test_used"] = False
    per_class["final_test_used"] = False
    aggregate["run_signature"] = [f"signature-{index}" for index in range(4)]
    with tempfile.TemporaryDirectory() as temp:
        path = study.validate_smoke(aggregate, per_class, Path(temp))
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == "PASS"


def test_v8n_screen_gate_and_serialization() -> None:
    aggregate, per_class = synthetic_main_results()
    aggregate = aggregate[aggregate["backbone"].eq("YOLOv8n")].copy()
    per_class = per_class[per_class["backbone"].eq("YOLOv8n")].copy()
    with tempfile.TemporaryDirectory() as temp:
        passed, path = study.analyze_v8n_screen(aggregate, per_class, Path(temp))
        assert passed
        assert path.exists()
        payload = json.loads((Path(temp) / "v8n_screen_decision.json").read_text(encoding="utf-8"))
        assert payload["authorizes_v8s_expansion"] is True
        assert np.isclose(payload["map_difference"], 0.01)


def main() -> None:
    tests = [
        test_factorial_interaction_and_gate,
        test_partial_cache_is_rejected,
        test_smoke_validation_serializes_numpy_booleans,
        test_v8n_screen_gate_and_serialization,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"[DONE] {len(tests)} tests passed")


if __name__ == "__main__":
    main()

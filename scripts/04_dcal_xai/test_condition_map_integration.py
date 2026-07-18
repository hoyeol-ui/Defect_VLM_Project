"""Integrity tests for the frozen condition-map integration outputs."""

from __future__ import annotations

import json
import hashlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "docs/condition_map_master_20260718.csv"
BRANCHES = ROOT / "docs/branch_decision_table_20260718.csv"
REGISTRY = ROOT / "runs/integrated_condition_map_20260718/source_metric_registry.csv"


class ConditionMapIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.master = pd.read_csv(MASTER)
        cls.branches = pd.read_csv(BRANCHES)
        cls.registry = pd.read_csv(REGISTRY)

    def test_equal_budget_yields_are_not_directly_mismatched(self) -> None:
        rows = self.master.dropna(subset=["target_yield_baseline", "target_yield_strategy", "target_enrichment"])
        self.assertTrue(rows["acquisition_budget"].notna().all())
        expected = rows["target_yield_strategy"] / rows["target_yield_baseline"]
        self.assertTrue(np.allclose(rows["target_enrichment"], expected, atol=1e-9, rtol=1e-8))

    def test_metric_semantics_are_preserved(self) -> None:
        mpdd_visa = self.master[self.master["dataset"].isin(["MPDD", "VisA"])]
        self.assertTrue(mpdd_visa["downstream_metric_name"].isna().all())
        self.assertTrue(mpdd_visa["task_type"].isin(["industrial_anomaly_localization", "industrial_anomaly_discovery"]).all())
        rare_rows = self.master[self.master["target_definition"].str.contains("rare", case=False, na=False)]
        anomaly_rows = self.master[self.master["target_definition"].str.contains("anomaly", case=False, na=False)]
        self.assertGreater(len(rare_rows), 0)
        self.assertGreater(len(anomaly_rows), 0)

    def test_confirmatory_fail_not_upgraded(self) -> None:
        failed = self.master[self.master["gate_result"].eq("FAIL")]
        self.assertFalse(failed["evidence_level"].eq("confirmatory_pass").any())
        closed = self.branches[self.branches["observed_result"].str.contains("FAIL", na=False)]
        self.assertFalse(closed["decision"].isin(["PASS", "CONTINUE"]).any())

    def test_no_final_test_results(self) -> None:
        self.assertFalse(self.master["source_file"].str.lower().str.contains("final|locked").any())
        self.assertFalse(self.registry["source_file"].str.lower().str.contains("final|locked").any())
        scanned = (ROOT / "runs/integrated_condition_map_20260718/scanned_files.txt").read_text(encoding="utf-8").lower()
        self.assertNotIn("final", scanned)
        self.assertNotIn("locked", scanned)
        self.assertTrue(self.branches["final_test_consumed"].astype(str).str.lower().eq("false").all())

    def test_dataset_task_types(self) -> None:
        expected = {
            "NEU-DET": {"object_detection"},
            "MPDD": {"industrial_anomaly_localization"},
            "VisA": {"industrial_anomaly_discovery"},
        }
        for dataset, task_types in expected.items():
            self.assertEqual(set(self.master.loc[self.master["dataset"].eq(dataset), "task_type"]), task_types)
        gc10_tasks = set(self.master.loc[self.master["dataset"].eq("GC10-DET"), "task_type"])
        self.assertTrue({"object_detection", "vlm_signal_validity"}.issubset(gc10_tasks))

    def test_missing_values_not_zero_filled(self) -> None:
        anomaly = self.master[self.master["dataset"].isin(["MPDD", "VisA"])]
        self.assertTrue(anomaly["downstream_baseline"].isna().all())
        self.assertTrue(anomaly["downstream_strategy"].isna().all())
        oracle = self.master[self.master["strategy"].eq("Qwen2-VL structured validity response")].iloc[0]
        self.assertTrue(pd.isna(oracle["target_yield_baseline"]))

    def test_every_metric_row_has_source(self) -> None:
        self.assertTrue(self.master["source_file"].notna().all())
        self.assertTrue(self.master["source_file"].astype(str).str.len().gt(0).all())
        self.assertTrue(self.registry["source_file"].notna().all())
        self.assertTrue(self.registry["source_locator"].notna().all())

    def test_gate_thresholds_unchanged(self) -> None:
        v23 = json.loads((ROOT / "runs/dcal_xai/v2_detector_signal_validity/decision.json").read_text(encoding="utf-8"))
        self.assertEqual(v23["status"], "FAIL")
        row = self.master[self.master["gate_name"].eq("V2.3 detector signal validity")].iloc[0]
        self.assertIn("total-error enrichment>=1.50", row["gate_threshold"])
        self.assertAlmostEqual(float(row["target_enrichment"]), float(v23["primary_metrics"]["error_enrichment_mean"]), places=12)
        k40 = json.loads((ROOT / "runs/dcal_xai/v2_budget_extended/decision.json").read_text(encoding="utf-8"))
        registry_thresholds = self.registry[self.registry["metric_id"].str.startswith("gc10.k40.threshold.")]
        self.assertEqual(len(registry_thresholds), len(k40["thresholds"]))
        screen = json.loads((ROOT / "runs/dcal_xai/v2_backbone_main/v8n_screen_decision.json").read_text(encoding="utf-8"))
        self.assertFalse(screen["authorizes_v8s_expansion"])

    def test_rare_fn_not_promoted_to_primary_confirmation(self) -> None:
        fn_rows = self.master[self.master["strategy"].str.contains("FN_diagnostic", na=False)]
        self.assertEqual(len(fn_rows), 1)
        self.assertEqual(fn_rows.iloc[0]["evidence_level"], "exploratory_positive")
        self.assertEqual(fn_rows.iloc[0]["gate_result"], "NOT_PRIMARY")
        branch = self.branches[self.branches["branch"].eq("FN/rare-miss exploratory signal")].iloc[0]
        self.assertEqual(branch["decision"], "HYPOTHESIS_ONLY")
        self.assertIn("post-hoc", branch["stopping_reason"])


def refresh_generated_manifest() -> None:
    out = ROOT / "runs/integrated_condition_map_20260718"
    files = [
        ROOT / "docs/condition_map_master_20260718.csv",
        ROOT / "docs/branch_decision_table_20260718.csv",
        ROOT / "docs/three_dataset_condition_map_decision_20260718.md",
        ROOT / "docs/fn_failure_prediction_extension_feasibility_20260718.md",
        ROOT / "docs/figures/discovery_coverage_pareto.png",
        ROOT / "docs/figures/selection_learning_translation.png",
        ROOT / "docs/figures/validity_translation_chain.png",
        ROOT / "docs/figures/branch_decision_waterfall.png",
        ROOT / "scripts/04_dcal_xai/build_condition_map_integration.py",
        ROOT / "scripts/04_dcal_xai/test_condition_map_integration.py",
        out / "scanned_files.txt",
        out / "source_metric_registry.csv",
        out / "metric_conflicts.csv",
        out / "missing_evidence.csv",
        out / "executed_commands.txt",
        out / "condition_map_validation.log",
    ]
    lines = []
    for path in files:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        lines.append(f"{relative}\tbytes={path.stat().st_size}\tsha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
    (out / "generated_file_manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ConditionMapIntegrationTests)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = stream.getvalue()
    print(report, end="")
    log = ROOT / "runs/integrated_condition_map_20260718/condition_map_validation.log"
    base = log.read_text(encoding="utf-8").split("\nUNIT TESTS\n", 1)[0].rstrip()
    log.write_text(base + "\n\nUNIT TESTS\n" + report, encoding="utf-8")
    refresh_generated_manifest()
    sys.exit(0 if result.wasSuccessful() else 1)

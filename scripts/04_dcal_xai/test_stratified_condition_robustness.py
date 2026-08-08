"""Unit tests for the frozen stratified condition-robustness audit."""

from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/stratified_condition_robustness_20260718"
BUILD_PATH = ROOT / "scripts/04_dcal_xai/build_stratified_condition_robustness.py"

spec = importlib.util.spec_from_file_location("stratified_build", BUILD_PATH)
assert spec and spec.loader
build = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = build
spec.loader.exec_module(build)


class StratifiedConditionRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = pd.read_csv(OUT / "pool_design_registry.csv")
        cls.outcomes = pd.read_csv(OUT / "seed_level_outcomes.csv")
        cls.summary = pd.read_csv(OUT / "stratified_effect_summary.csv")
        cls.registry = pd.read_csv(OUT / "metric_source_registry.csv")
        cls.missing = pd.read_csv(OUT / "missing_or_nonidentifiable.csv")

    def test_01_selected_outcomes_are_not_matching_covariates(self) -> None:
        self.assertTrue(build.MATCHING_COVARIATE_WHITELIST.isdisjoint(build.FORBIDDEN_MATCHING_COVARIATES))
        self.assertFalse(any(name.startswith("selected_") for name in build.MATCHING_COVARIATE_WHITELIST))

    def test_02_pool_hashes_are_not_counted_once_per_strategy(self) -> None:
        expected_strategies = {"GC10-DET": 2, "MPDD": 3, "VisA": 2}
        for dataset, frame in self.design.groupby("dataset"):
            self.assertEqual(frame["acquisition_seed"].nunique(), 200)
            self.assertEqual(frame["pool_hash"].nunique(), 200)
            self.assertTrue((frame.groupby("pool_hash")["strategy"].nunique() == expected_strategies[dataset]).all())

    def test_03_type_c_cannot_produce_prevalence_effect(self) -> None:
        self.assertFalse(build.prevalence_effect_allowed("TYPE_C_repeated_selection_same_pool", True))
        self.assertFalse(build.prevalence_effect_allowed("TYPE_A_paired_variable_pool", False))

    def test_04_different_budgets_are_never_paired(self) -> None:
        budget_counts = self.design.groupby(["dataset", "acquisition_seed"])["budget"].nunique()
        self.assertTrue((budget_counts == 1).all())

    def test_05_dataset_targets_are_not_raw_pooled(self) -> None:
        self.assertFalse(self.summary["dataset"].str.lower().isin({"all", "pooled", "cross_dataset"}).any())
        self.assertEqual(set(self.summary["dataset"]), {"GC10-DET", "MPDD", "VisA"})

    def test_06_inferential_unit_is_seed_or_pool_not_image(self) -> None:
        self.assertEqual(build.INFERENTIAL_UNIT, "acquisition_seed_or_pool_realization")
        self.assertTrue(self.summary["inferential_unit"].eq(build.INFERENTIAL_UNIT).all())
        self.assertTrue((self.summary["n_units"] == 200).all())

    def test_07_no_final_or_locked_source_path(self) -> None:
        for frame, column in ((self.design, "source_file"), (self.outcomes, "source_file"), (self.summary, "source_file"), (self.registry, "source_file")):
            paths = frame[column].astype(str).str.lower()
            self.assertFalse(paths.str.contains(r"(?:^|[/\\])(?:final|locked)(?:[/\\]|$)", regex=True).any())
        log = (OUT / "audit_execution_log.txt").read_text(encoding="utf-8")
        self.assertIn("final_test_used=False", log)

    def test_08_failed_matching_balance_cannot_be_adjusted_evidence(self) -> None:
        self.assertFalse(build.adjusted_evidence_allowed("TYPE_B_unmatched_variable_pool", False))
        self.assertFalse(build.adjusted_evidence_allowed("TYPE_B_unmatched_variable_pool", None))
        self.assertTrue(build.adjusted_evidence_allowed("TYPE_B_unmatched_variable_pool", True))
        self.assertFalse((OUT / "matching_balance.csv").exists())

    def test_09_missing_prevalence_effects_remain_missing_not_zero(self) -> None:
        rows = self.summary[self.summary["stratum_name"].eq("prevalence_low_medium_high")]
        for column in ("unadjusted_gain", "adjusted_gain", "ci_lower", "ci_upper", "coverage_delta"):
            self.assertTrue(rows[column].isna().all(), column)
        self.assertTrue(rows["stratum_value"].eq("not_identifiable").all())

    def test_10_every_numeric_output_metric_has_source_locator(self) -> None:
        numeric_columns = {
            "pool_design_registry.csv": ["pool_size", "budget", "target_count", "target_prevalence", "category_count", "category_entropy", "max_category_share", "source_count", "source_entropy", "max_source_share"],
            "seed_level_outcomes.csv": ["target_yield", "target_enrichment", "category_coverage", "category_entropy", "max_category_share", "source_coverage", "source_entropy", "max_source_share", "concentration_hhi", "source_concentration_hhi", "rare_image_count", "rare_class_count", "instance_count", "paired_discovery_delta", "paired_coverage_delta", "paired_concentration_delta", "target_yield_per_category", "target_yield_per_source"],
            "stratified_effect_summary.csv": ["n_units", "unadjusted_gain", "adjusted_gain", "ci_lower", "ci_upper", "positive_fraction", "p_value", "coverage_delta", "concentration_delta", "retained_effect_fraction", "standardized_paired_effect_size", "median_gain", "iqr_gain", "min_gain", "max_gain", "positive_gain_with_safety_loss_fraction"],
        }
        frames = {
            "pool_design_registry.csv": self.design,
            "seed_level_outcomes.csv": self.outcomes,
            "stratified_effect_summary.csv": self.summary,
        }
        keys = set(zip(self.registry["output_file"], self.registry["output_row_id"].astype(int), self.registry["metric"]))
        for filename, columns in numeric_columns.items():
            frame = frames[filename]
            for row_id, row in frame.iterrows():
                for metric in columns:
                    if pd.notna(row[metric]):
                        self.assertIn((filename, row_id, metric), keys)
        self.assertTrue(self.registry["source_file"].notna().all())
        self.assertTrue(self.registry["verified"].astype(bool).all())
        for locator in self.registry["source_file"].drop_duplicates():
            for relative in str(locator).split("; "):
                self.assertTrue((ROOT / relative).exists(), relative)

    def test_11_rare_fn_is_not_promoted_to_primary_endpoint(self) -> None:
        self.assertFalse(self.summary["stratum_name"].str.contains("fn", case=False, na=False).any())
        self.assertFalse(self.outcomes.columns.to_series().str.fullmatch(r".*fn.*", case=False).any())
        decision = (ROOT / "docs/stratified_condition_robustness_decision_20260718.md").read_text(encoding="utf-8")
        self.assertIn("DESIGN_PROTOCOL_ONLY retained", decision)

    def test_12_strengthened_decision_criteria_are_frozen_in_code(self) -> None:
        expected = (
            "positive_adjusted_gain_in_at_least_2_datasets",
            "ci_lower_above_zero_in_at_least_2_datasets",
            "retained_effect_fraction_at_least_0p50",
            "positive_discovery_with_safety_loss_in_at_least_2_datasets",
            "not_dominated_by_one_category_source_or_few_seeds",
            "no_outcome_leakage_in_pairing_or_matching",
        )
        self.assertEqual(build.STRENGTHENED_CRITERIA, expected)
        decision = (ROOT / "docs/stratified_condition_robustness_decision_20260718.md").read_text(encoding="utf-8")
        self.assertIn("B. CONDITION_MAP_DESCRIPTIVE_ONLY", decision)

    def test_13_required_outputs_exist_and_are_nonempty(self) -> None:
        required = [
            OUT / "pool_design_registry.csv",
            OUT / "seed_level_outcomes.csv",
            OUT / "stratified_effect_summary.csv",
            OUT / "metric_source_registry.csv",
            OUT / "missing_or_nonidentifiable.csv",
            OUT / "audit_execution_log.txt",
            ROOT / "docs/stratified_audit_design_feasibility_20260718.md",
            ROOT / "docs/stratified_condition_robustness_decision_20260718.md",
            ROOT / "docs/figures/stratified_discovery_gain.png",
            ROOT / "docs/figures/discovery_safety_quadrants.png",
            ROOT / "docs/figures/category_source_concentration.png",
            ROOT / "docs/figures/adjusted_vs_unadjusted_effect.png",
        ]
        for path in required:
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 0, path)

    def test_14_pairing_is_complete_and_strategy_safe(self) -> None:
        for dataset, frame in self.design.groupby("dataset"):
            by_seed = frame.groupby("acquisition_seed")
            self.assertTrue((by_seed["pool_hash"].nunique() == 1).all(), dataset)
            self.assertTrue((by_seed["initial_set_id"].nunique() == 1).all(), dataset)
            self.assertTrue((by_seed["paired_random_id"].nunique() == 1).all(), dataset)

    def test_15_prevalence_rows_are_explicitly_nonidentifiable(self) -> None:
        rows = self.summary[self.summary["stratum_name"].eq("prevalence_low_medium_high")]
        self.assertTrue(rows["identifiability"].str.startswith("not_identifiable").all())
        missing = self.missing[self.missing["requested_analysis"].eq("prevalence low/medium/high effect")]
        self.assertEqual(set(missing["dataset"]), {"GC10-DET", "MPDD", "VisA"})


def update_manifest() -> None:
    manifest = OUT / "generated_file_manifest.txt"
    tracked = [
        ROOT / "scripts/04_dcal_xai/test_stratified_condition_robustness.py",
        OUT / "test_results.txt",
    ]
    existing = []
    if manifest.exists():
        existing = [line for line in manifest.read_text(encoding="utf-8").splitlines() if not any(str(path.relative_to(ROOT)).replace("\\", "/") in line for path in tracked)]
    for path in tracked:
        relative = path.relative_to(ROOT).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        existing.append(f"{relative}\tbytes={path.stat().st_size}\tsha256={digest}")
    manifest.write_text("\n".join(existing) + "\n", encoding="utf-8")


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StratifiedConditionRobustnessTests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    output = stream.getvalue()
    (OUT / "test_results.txt").write_text(output, encoding="utf-8")
    update_manifest()
    print(output, end="")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

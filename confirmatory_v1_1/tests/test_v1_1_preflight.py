from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from confirmatory_v1_1.final_lock import (  # noqa: E402
    FinalLockError,
    development_access,
    fixture_authorization,
    fixture_registry,
    primary_locked_final_aulc,
    require_actual_final_access,
    validate_final_authorization_fixture,
)


P = "p" * 64
F = "f" * 40
A = "a" * 64


class Phase2BAdditionalPreflight(unittest.TestCase):
    def test_15_development_access_passes(self) -> None:
        development_access("development")

    def test_16_real_final_access_without_authorization_hard_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FinalLockError):
                require_actual_final_access(
                    authorization_flag=False,
                    authorization_path=Path(directory) / "FINAL_EVALUATION_AUTHORIZATION.json",
                    expected_protocol_sha256=P,
                    expected_freeze_commit_sha=F,
                    cell_registry=fixture_registry(P),
                    expected_attempt_registry_sha256=A,
                )

    def test_17_incomplete_439_registry_rejected(self) -> None:
        registry = fixture_registry(P, 439)
        auth = fixture_authorization(P, F, registry, A)
        with self.assertRaises(FinalLockError):
            validate_final_authorization_fixture(auth, expected_protocol_sha256=P, expected_freeze_commit_sha=F, cell_registry=registry, expected_attempt_registry_sha256=A)

    def test_18_complete_registry_wrong_protocol_sha_rejected(self) -> None:
        registry = fixture_registry(P)
        auth = fixture_authorization("q" * 64, F, registry, A)
        with self.assertRaises(FinalLockError):
            validate_final_authorization_fixture(auth, expected_protocol_sha256=P, expected_freeze_commit_sha=F, cell_registry=registry, expected_attempt_registry_sha256=A)

    def test_19_complete_fixture_authorization_logic_passes_but_is_not_actual_file(self) -> None:
        registry = fixture_registry(P)
        auth = fixture_authorization(P, F, registry, A)
        validate_final_authorization_fixture(auth, expected_protocol_sha256=P, expected_freeze_commit_sha=F, cell_registry=registry, expected_attempt_registry_sha256=A)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FinalLockError):
                require_actual_final_access(authorization_flag=True, authorization_path=Path(directory) / "missing.json", expected_protocol_sha256=P, expected_freeze_commit_sha=F, cell_registry=registry, expected_attempt_registry_sha256=A)

    def test_20_primary_aulc_reads_final_only(self) -> None:
        rows = [{"round": i, "metric_source": "locked_final", "budget_fraction": 0.1 + i * 0.05, "final_map5095": 0.2 + i * 0.01} for i in range(6)]
        result = primary_locked_final_aulc(rows)
        self.assertEqual(result.status, "COMPLETE")
        self.assertIsNotNone(result.value)

    def test_21_development_metric_in_primary_hard_fails(self) -> None:
        rows = [{"round": i, "metric_source": "development", "budget_fraction": 0.1 + i * 0.05, "development_map5095": 0.2} for i in range(6)]
        with self.assertRaises(FinalLockError):
            primary_locked_final_aulc(rows)

    def test_22_missing_one_final_point_is_incomplete_without_interpolation(self) -> None:
        rows = [{"round": i, "metric_source": "locked_final", "budget_fraction": 0.1 + i * 0.05, "final_map5095": 0.2} for i in range(5)]
        result = primary_locked_final_aulc(rows)
        self.assertEqual(result.status, "INCOMPLETE_NOT_ESTIMATED")
        self.assertIsNone(result.value)


if __name__ == "__main__":
    unittest.main(verbosity=2)

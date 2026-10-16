"""Synthetic CPU tests only: no original images, caches, selections or GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import preflight as p


class PreflightTests(unittest.TestCase):
    def disk_fixture(self, root):
        ids = [f"synthetic{i:04}" for i in range(1604)]
        # Deliberately equal normalized embeddings: expected lexical order is known
        # independently, without asking the implementation to generate its own oracle.
        features = np.zeros((1604, 384), dtype=np.float32)
        features[:, 0] = 1
        np.save(root / "features.npy", features)
        values = {"cache_ids": ids, "train_ids": ids, "r4": ids[:481],
                  "r5": ids[:561], "new_order": ids[481:561]}
        inputs = {}
        for name in p.INPUTS:
            path = root / (name + (".npy" if name == "features" else ".json"))
            if name != "features":
                path.write_text(json.dumps(values.get(name, {"synthetic": True})))
            inputs[name] = {"path": path.name, "sha256": p.sha256(path),
                            "hash_evidence": "SYNTHETIC TEST FIXTURE ONLY", "format": "json"}
        cfg = {"scope": {"dataset": "GC10-DET", "acquisition_seed": 2026091901, "transition": "R4_to_R5"},
               "original_protocol_review": {"confirmed": True, "notes": "Synthetic fixture, no historical experiment used."},
               "inputs": inputs}
        p.write_json(root / "config.json", cfg)
        return cfg

    def fixture(self):
        rng = np.random.default_rng(77)
        x = rng.normal(size=(23, 6)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        x[4] = x[3]  # duplicate vectors exercise nearest-center ties
        ids = [f"i{i:02}" for i in range(len(x))]
        return p.distance_matrix(x), ids, ids[:4], ids[4:8]

    def test_screen_matches_independent_brute_force(self):
        d, ids, r4, new = self.fixture()
        got = []
        report = p.screen_initial_swaps(d, ids, r4, new, seconds=30,
                                       on_pair=lambda *r: got.append(r), chunk_size=3)
        selected = r4 + new
        original = p.coverage(d, [ids.index(x) for x in selected])["radius"]
        expected = {}
        for out in new:
            for inc in sorted(set(ids) - set(selected)):
                centers = [ids.index(x) for x in selected if x != out] + [ids.index(inc)]
                radius = p.coverage(d, centers)["radius"]
                if radius <= original:
                    expected[(out, inc)] = radius
        self.assertEqual({(a, b): c for a, b, c, _ in got}, expected)
        self.assertEqual(report["checked_pairs"], 4 * 15)
        self.assertEqual(report["feasible_pairs"], len(expected))
        self.assertEqual(report["status"], "COMPLETE")
        self.assertTrue(all(out not in r4 for out, *_ in got))

    def test_zero_pairs_is_local_not_universal_impossibility(self):
        d = np.array([[0, 1, 1], [1, 0, 1], [1, .2, 0.]])
        result = p.screen_initial_swaps(d, ["a", "b", "c"], ["a"], ["b"], seconds=30)
        self.assertEqual(result["feasible_pairs"], 0)
        self.assertIn("simultaneous multi-swaps are untested", result["interpretation"])
        self.assertIsNone(result["research_success"])

    def test_tie_tolerance_is_not_radius_allowance(self):
        d = np.array([[0, 1, 1], [1, 0, .2 + 5e-13], [1, .2, 0.]])
        result = p.screen_initial_swaps(d, ["a", "b", "c"], ["a"], ["b"], seconds=30)
        self.assertEqual(result["feasible_pairs"], 0)
        self.assertEqual(result["small_positive_deltas_rejected"], 1)

    def test_timeout_is_incomplete(self):
        d, ids, r4, new = self.fixture()
        with patch.object(p.time, "monotonic", side_effect=[0, 2, 2]):
            report = p.screen_initial_swaps(d, ids, r4, new, seconds=1)
        self.assertEqual(report["status"], "INCOMPLETE_TIME_LIMIT")
        self.assertEqual(report["checked_pairs"], 0)
        self.assertIn("does not establish infeasibility", report["interpretation"])

    def test_lexical_near_tie_and_immediate_update(self):
        ids = ["z", "a", "l", "b"]
        d = np.array([[0, .1, 1, .2], [.1, 0, 1 - 5e-13, .9],
                      [1, 1 - 5e-13, 0, .5], [.2, .9, .5, 0]])
        chosen, trace = p.replay_ff(d, ids, ["l"], 2)
        self.assertEqual(chosen, ["a", "b"])
        self.assertEqual(trace[0]["tie_count"], 2)

    def test_row_permutation_keeps_id_mapping(self):
        d, ids, r4, _ = self.fixture()
        expected, _ = p.replay_ff(d, ids, r4, 5)
        perm = np.random.default_rng(3).permutation(len(ids))
        actual, _ = p.replay_ff(d[np.ix_(perm, perm)], [ids[i] for i in perm], r4, 5)
        self.assertEqual(actual, expected)

    def test_does_not_renormalize_clip_or_zero_diagonal(self):
        x = np.array([[1.0000001, 0], [0, 1]], dtype=np.float32)
        d = p.distance_matrix(x)
        self.assertLess(d[0, 0], 0)
        np.testing.assert_array_equal(d, 1 - x.astype(np.float64) @ x.astype(np.float64).T)

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.json"
            path.write_text('["a", "a"]')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                p.read_ids(path, {"format": "json"})

    def test_absent_original_order_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.json"
            path.write_text('{"cumulative_ids": ["a"]}')
            with self.assertRaises(KeyError):
                p.read_ids(path, {"format": "json", "keys": ["unverified_new_order"]})

    def test_input_error_writes_block_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{}')
            code = p.main(["verify", "--config", str(root / "config.json"), "--out", str(root / "out")])
            self.assertEqual(code, 2)
            self.assertEqual(p.read_json(root / "out/blocked.json")["status"], "BLOCKED_INPUT_OR_PROTOCOL")

    def test_screen_always_reverifies_and_blocks_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(p, "load_inputs", return_value=(
                    {"scope": {}, "original_protocol_review": {}}, None, None, {}, 0)), \
                 patch.object(p, "verify", return_value=(None, {"status": "BLOCKED_FF_MISMATCH"})), \
                 patch.object(p, "screen_initial_swaps") as screen:
                code = p.main(["screen", "--config", "unused", "--out", str(Path(tmp) / "out"), "--seconds", "10"])
                self.assertEqual(code, 2)
                screen.assert_not_called()

    def test_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "original.txt"
            sentinel.write_text("untouched")
            with self.assertRaises(SystemExit):
                p.main(["verify", "--config", "unused", "--out", tmp])
            self.assertEqual(sentinel.read_text(), "untouched")

    def test_full_schema_verify_cli_and_hash_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.disk_fixture(root)
            code = p.main(["verify", "--config", str(root / "config.json"), "--out", str(root / "out")])
            self.assertEqual(code, 0)
            result = p.read_json(root / "out/ff_reproduction.json")
            self.assertTrue(result["last_80_order_match"])
            self.assertTrue(result["cumulative_r5_set_match"])
            self.assertFalse((root / "out/swap_feasibility.json").exists())
            (root / "cache_ids.json").write_text('["tampered"]')
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch: cache_ids"):
                p.load_inputs(root / "config.json")

    def test_same_set_wrong_order_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.disk_fixture(root)
            path = root / "new_order.json"
            p.write_json(path, list(reversed(p.read_json(path))))
            cfg["inputs"]["new_order"]["sha256"] = p.sha256(path)
            p.write_json(root / "config.json", cfg)
            _, features, lists, _, _ = p.load_inputs(root / "config.json")
            _, result = p.verify(features, lists)
            self.assertTrue(result["last_80_set_match"])
            self.assertFalse(result["last_80_order_match"])
            self.assertEqual(result["status"], "BLOCKED_FF_MISMATCH")

    def test_non_normalized_cache_is_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.disk_fixture(root)
            path = root / "features.npy"
            x = np.load(path)
            x[0] *= 2
            np.save(path, x)
            cfg["inputs"]["features"]["sha256"] = p.sha256(path)
            p.write_json(root / "config.json", cfg)
            with self.assertRaisesRegex(ValueError, "not normalized"):
                p.load_inputs(root / "config.json")


if __name__ == "__main__":
    unittest.main()

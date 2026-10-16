"""Synthetic tests only. No actual image, cache, selection or training is used."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import preflight as p
import patch_signal as s
import patch_posthoc as h
import test_preflight


def record(sid="r4a"):
    q = np.zeros((16, 2), dtype=np.float32)
    q[:, 0] = 1
    q[5] = [0, 1]
    return {"id": sid, **s.contribution(q, np.ones(16, bool), np.eye(2, dtype=np.float32), np.array([1., 0.]))}


class PatchDiagnosticsTests(unittest.TestCase):
    def test_contribution_equals_facility_location_delta(self):
        q = np.zeros((16, 3), dtype=np.float32)
        q[:, 0] = 1
        q[3] = [0, 1, 0]
        q[7] = [0, 0, 1]
        proto = np.eye(3, dtype=np.float32)
        ref = np.array([.8, .7, .1])
        rec = s.contribution(q, np.ones(16, bool), proto, ref)
        expected = np.maximum(ref, (q @ proto.T).max(axis=0)).mean() - ref.mean()
        self.assertAlmostEqual(rec["gain"], expected)
        self.assertEqual(rec["top_regions"], [7, 3, 0])
        h.validate_record({"id": "x", **rec})

    def test_invalid_regions_cannot_win_and_saturation_has_no_highlights(self):
        q = np.eye(16, dtype=np.float32)
        valid = np.ones(16, bool)
        valid[2] = False
        rec = s.contribution(q, valid, q, np.ones(16))
        self.assertEqual(rec["top_regions"], [])
        self.assertEqual(rec["gain"], 0)
        self.assertNotIn(2, rec["valid_regions"])

    def test_r4_leave_one_out_not_self_reference(self):
        q = np.zeros((2, 16, 2), dtype=np.float32)
        q[0, :, 0] = 1
        q[1, :, 1] = 1
        cache = (["a", "b"], q, np.ones((2, 16), bool), np.eye(2, dtype=np.float32), {})
        result = s.score_r4({"r4": ["a", "b"], "sample_ids": ["a"]}, cache)
        self.assertEqual(result["records"][0]["gain"], .5)
        self.assertFalse(result["labels_read"])

    def test_area_pooling_excludes_full_padding_and_is_normalized(self):
        tokens = np.zeros((37, 37, 384), dtype=np.float32)
        tokens[:, :, 0] = 1
        # Exact patch-boundary crop: outside rows contain unrelated huge vectors.
        tokens[:10, :, 0] = 0
        tokens[:10, :, 1] = 100
        q, valid = s.pool_saved_tokens(tokens, [0, 10/37, 1, 1])
        np.testing.assert_array_equal(q[:, 0], np.ones(16))
        np.testing.assert_array_equal(q[:, 1:], np.zeros((16, 383)))
        self.assertTrue(valid.all())

    def test_pooling_partial_cell_intersections_and_zero_norm(self):
        tokens = np.zeros((37, 37, 384), dtype=np.float32)
        tokens[:, :, 0] = 2
        q, valid = s.pool_saved_tokens(tokens, [.01, .23, .99, .77])
        self.assertTrue(valid.all())
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1)
        q, valid = s.pool_saved_tokens(np.zeros_like(tokens), [0, 0, 1, 1])
        self.assertFalse(valid.any())
        self.assertFalse(q.any())

    def test_box_union_no_double_count(self):
        self.assertAlmostEqual(h.union_area([[0, 0, .75, 1], [.25, 0, 1, 1]]), 1)
        self.assertAlmostEqual(h.box_fraction([0, 0, .5, .5], [[0, 0, 1, 1], [0, 0, 1, 1]]), 1)
        self.assertEqual(h.box_fraction([0, 0, .25, .25], [[.5, .5, 1, 1]]), 0)

    def test_overlap_random_baseline_and_determinism(self):
        plan = {"sample_ids": ["r4a"], "r4": ["r4a", "r4b"]}
        scores = {"kind": "R4_LEAVE_ONE_OUT_DIAGNOSTIC", "labels_read": False, "records": [record()]}
        ann = {"coordinate_system": "original_image_normalized_xyxy", "boxes": {"r4a": [[.25, .25, .5, .5]]}}
        a = h.evaluate_r4(plan, scores, ann)
        self.assertEqual(a, h.evaluate_r4(plan, scores, ann))
        row = a["records"][0]
        self.assertEqual(row["observed_box_fraction"], 1)
        self.assertEqual(row["random_expected_box_fraction"], 1/16)
        self.assertEqual(row["difference_from_random"], 15/16)
        self.assertIsNone(a["research_success"])

    def test_gt_outside_frozen_sample_is_rejected(self):
        plan = {"sample_ids": ["r4a"], "r4": ["r4a", "r4b"]}
        scores = {"kind": "R4_LEAVE_ONE_OUT_DIAGNOSTIC", "labels_read": False, "records": [record()]}
        for extra in ("r4b", "unlabeled_id", "test_id"):
            ann = {"coordinate_system": "original_image_normalized_xyxy", "boxes": {"r4a": [], extra: []}}
            with self.assertRaisesRegex(ValueError, "exactly the frozen R4"):
                h.evaluate_r4(plan, scores, ann)

    def test_wrong_coordinate_system_and_malformed_box_rejected(self):
        plan = {"sample_ids": ["r4a"], "r4": ["r4a"]}
        scores = {"kind": "R4_LEAVE_ONE_OUT_DIAGNOSTIC", "labels_read": False, "records": [record()]}
        ann = {"coordinate_system": "yolo_xywh", "boxes": {"r4a": []}}
        with self.assertRaisesRegex(ValueError, "coordinates"):
            h.evaluate_r4(plan, scores, ann)
        for box in ([.5, .5, .1, .1], [0, 0, 20, 20], [0, 0, float("nan"), 1]):
            with self.assertRaises(ValueError):
                h.check_box(box)

    def test_saturated_and_no_box_counts_are_not_hidden(self):
        rec = record()
        rec.update(gain=0., region_weights=[0.] * 16, top_regions=[])
        plan = {"sample_ids": ["r4a"], "r4": ["r4a"]}
        scores = {"kind": "R4_LEAVE_ONE_OUT_DIAGNOSTIC", "labels_read": False, "records": [rec]}
        ann = {"coordinate_system": "original_image_normalized_xyxy", "boxes": {"r4a": []}}
        result = h.evaluate_r4(plan, scores, ann)
        self.assertEqual(result["saturated_n"], 1)
        self.assertEqual(result["no_gt_boxes_n"], 1)
        self.assertIsNone(result["mean_difference_evaluated_only"])

    def test_signal_cli_does_not_accept_annotations(self):
        with self.assertRaises(SystemExit):
            s.main(["score", "--plan", "unused", "--plan-sha", "unused", "--cache", "unused", "--cache-sha", "unused", "--meta", "unused", "--meta-sha", "unused", "--out", "unused", "--annotations", "anything"])
        with self.assertRaises(SystemExit):
            h.main(["render-pairs", "--plan", "x", "--plan-sha", "x", "--scores", "x", "--scores-sha", "x", "--annotations", "x", "--out", "unused"])

    def test_hash_tamper_and_changed_highlights_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x"
            path.write_text("before")
            sha = p.sha256(path)
            path.write_text("after")
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                s.hashed(path, sha)
        rec = record()
        rec["top_regions"] = [4]
        with self.assertRaisesRegex(ValueError, "changed"):
            h.validate_record(rec)

    def test_freeze_binds_screen_inputs_and_deterministic_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_preflight.PreflightTests().disk_fixture(root)
            _, _, lists, provenance, _ = p.load_inputs(root / "config.json")
            screen = root / "screen"
            screen.mkdir()
            p.write_json(screen / "provenance.json", {"inputs": provenance, "checker_sha256": p.sha256(p.__file__)})
            p.write_json(screen / "ff_reproduction.json", {"status": "PASS", "last_80_set_match": True, "last_80_order_match": True, "cumulative_r5_set_match": True})
            p.write_json(screen / "swap_feasibility.json", {"status": "COMPLETE", "checked_pairs": 83440, "expected_pairs": 83440, "feasible_pairs": 1})
            out = root / "plan"
            out.mkdir()
            plan = s.freeze(root / "config.json", screen, out)
            loaded = s.load_plan(out / "plan.json", p.sha256(out / "plan.json"))
            self.assertEqual(plan, loaded)
            self.assertEqual(len(plan["sample_ids"]), 24)
            self.assertLessEqual(set(plan["sample_ids"]), set(lists["r4"]))
            self.assertEqual(s.rank_ids(list(reversed(lists["r4"])), "R4-patch-audit-v1")[:24], plan["sample_ids"])
            evidence = p.read_json(screen / "provenance.json")
            evidence["inputs"]["config"]["sha256"] = "0" * 64
            p.write_json(screen / "provenance.json", evidence)
            with self.assertRaisesRegex(ValueError, "inputs differ"):
                s.freeze(root / "config.json", screen, out)

    def test_pair_diagnosis_no_change_and_unmatched_counts(self):
        ids = [f"s{i:04}" for i in range(164)]
        plan = {"r4": ids[:2], "r5": ids[:82]}
        q = np.zeros((164, 16, 2), dtype=np.float32)
        q[:, :, 0] = 1
        cache = (ids, q, np.ones((164, 16), bool), np.eye(2, dtype=np.float32), {})
        arms = {name: {"new_ids": ids[2:82], "accepted_swaps": 0, "seed": s.CONTROL_SEEDS.get(name)} for name in ("patch", "random_1", "random_2", "random_3")}
        arms["patch"] = {"new_ids": ids[3:83], "accepted_swaps": 1}
        report, key = s.pair_review(plan, cache, {"arms": arms}, ids, np.zeros((164, 164)))
        self.assertEqual(len(report["records"]), 1)
        self.assertFalse(report["matched_operation_counts"])
        self.assertFalse(report["labels_read"])
        self.assertEqual(set(key[report["records"][0]["pair_id"]].values()), {"incoming", "outgoing"})
        arms["patch"]["new_ids"][0] = ids[0]
        with self.assertRaisesRegex(ValueError, "Invalid final"):
            s.pair_review(plan, cache, {"arms": arms}, ids, np.zeros((164, 164)))

    def test_render_synthetic_r4_and_pairs(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "toy.png"
            Image.new("RGB", (160, 80), "gray").save(path)
            images = {"r4a": {"path": "toy.png", "sha256": p.sha256(path)}}
            scores = {"kind": "R4_LEAVE_ONE_OUT_DIAGNOSTIC", "records": [record()]}
            h.render(scores, images, root, root, {"r4a": [[.25, .25, .5, .5]]})
            content = (root / "review.html").read_text()
            self.assertIn("data:image/png;base64", content)
            self.assertIn("#00ff88", content)
            self.assertIn("#ff9800", content)
            pair_scores = {"kind": "FROZEN_FINAL_PAIR_REVIEW", "records": [{"pair_id": "arm_1_01", "sides": [record(), record()]}]}
            with self.assertRaisesRegex(ValueError, "cannot receive GT"):
                h.render(pair_scores, images, root, root, {"r4a": []})

    def test_cache_contract_and_r4_cli_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_preflight.PreflightTests().disk_fixture(root)
            _, _, lists, provenance, _ = p.load_inputs(root / "config.json")
            plan = {"version": 1, "policy": s.POLICY, "r4": lists["r4"], "r5": lists["r5"],
                    "train_ids": lists["train_ids"], "sample_ids": s.rank_ids(lists["r4"], "R4-patch-audit-v1")[:24],
                    "top_k": 3, "random_repeats": 200, "inputs": provenance, "signal_code_sha256": p.sha256(s.__file__), "posthoc_code_sha256": p.sha256(h.__file__)}
            plan_path = root / "plan.json"
            p.write_json(plan_path, plan)
            q = np.zeros((1604, 16, 384), np.float32)
            q[:, :, 0] = 1
            proto = np.zeros((256, 384), np.float32)
            proto[:, 0] = 1
            cache_path = root / "cache.npz"
            np.savez_compressed(cache_path, ids=np.array(lists["train_ids"]), q=q,
                                valid=np.ones((1604, 16), bool), prototypes=proto)
            meta = {"policy": s.POLICY, "prototype_seed": s.PROTOTYPE_SEED, "prototype_reservoir_limit": 10000, "used_gt": False, "cls_reproduction_passed": True,
                    "canonical_cls_sha256": provenance["features"]["sha256"],
                    "train_ids_digest": s.digest_ids(lists["train_ids"]), "prototype_fit_ids_digest": s.digest_ids(lists["train_ids"]),
                    "extractor_sha256": "1"*64, "weights_sha256": "2"*64, "preprocessing_sha256": "3"*64,
                    "cls_check_evidence": "SYNTHETIC fixture; no feature extraction performed", "prototype_fit_evidence": "SYNTHETIC prototypes; no fitting performed"}
            meta_path = root / "meta.json"
            p.write_json(meta_path, meta)
            args = ["score", "--plan", str(plan_path), "--plan-sha", p.sha256(plan_path),
                    "--cache", str(cache_path), "--cache-sha", p.sha256(cache_path),
                    "--meta", str(meta_path), "--meta-sha", p.sha256(meta_path), "--out", str(root / "score")]
            self.assertEqual(s.main(args), 0)
            scores_path = root / "score/scores.json"
            scores = p.read_json(scores_path)
            self.assertEqual(len(scores["records"]), 24)
            self.assertTrue(all(not r["top_regions"] for r in scores["records"]))
            ann_path = root / "ann.json"
            p.write_json(ann_path, {"coordinate_system": "original_image_normalized_xyxy",
                                   "boxes": {sid: [[.25, .25, .5, .5]] for sid in plan["sample_ids"]}})
            self.assertEqual(h.main(["evaluate-r4", "--plan", str(plan_path), "--plan-sha", p.sha256(plan_path),
                                     "--scores", str(scores_path), "--scores-sha", p.sha256(scores_path),
                                     "--annotations", str(ann_path), "--annotations-sha", p.sha256(ann_path),
                                     "--out", str(root / "posthoc")]), 0)
            self.assertEqual(p.read_json(root / "posthoc/box_overlap.json")["saturated_n"], 24)
            meta["used_gt"] = True
            p.write_json(meta_path, meta)
            with self.assertRaisesRegex(ValueError, "label-free"):
                s.load_cache(cache_path, p.sha256(cache_path), meta_path, p.sha256(meta_path), plan)
            meta["used_gt"] = False
            meta["prototype_fit_ids_digest"] = "0" * 64
            p.write_json(meta_path, meta)
            with self.assertRaisesRegex(ValueError, "train pool"):
                s.load_cache(cache_path, p.sha256(cache_path), meta_path, p.sha256(meta_path), plan)

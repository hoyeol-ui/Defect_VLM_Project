"""Label-free preparation and diagnostics for already-extracted patch caches.
No extraction, training, prototype fitting or acquisition algorithm is included.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import preflight as p

POLICY = "F518_lastnorm_unpadded_4x4_area_weighted_L2_v1"
TOP_K, SAMPLE_N, REPEATS = 3, 24, 200
GAIN_FLOOR = 1e-9
PROTOTYPE_SEED = 2026101600
CONTROL_SEEDS = {"random_1": 2026101601, "random_2": 2026101602, "random_3": 2026101603}


def digest_ids(ids):
    return hashlib.sha256(json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def hashed(path, expected):
    p.require(len(expected) == 64 and p.sha256(path) == expected, f"Hash mismatch: {path}")
    return Path(path)


def rank_ids(ids, salt):
    return sorted(ids, key=lambda x: (hashlib.sha256((salt + "\0" + x).encode()).hexdigest(), x))


def load_plan(path, expected):
    plan = p.read_json(hashed(path, expected))
    p.require(plan["version"] == 1 and plan["policy"] == POLICY, "Unsupported frozen plan")
    p.require(len(plan["r4"]) == len(set(plan["r4"])) == 481, "Invalid R4")
    p.require(len(plan["train_ids"]) == len(set(plan["train_ids"])) == 1604, "Invalid train")
    p.require(set(plan["r4"]) < set(plan["r5"]) <= set(plan["train_ids"]) and len(set(plan["r5"])) == len(plan["r5"]) == 561, "Invalid R5")
    p.require(plan["sample_ids"] == rank_ids(plan["r4"], "R4-patch-audit-v1")[:SAMPLE_N], "Sample was changed")
    p.require(plan["top_k"] == TOP_K and plan["random_repeats"] == REPEATS, "Protocol was changed")
    return plan


def freeze(config, screen_dir, out):
    cfg, _, lists, provenance, _ = p.load_inputs(config)
    screen_dir = Path(screen_dir)
    evidence = p.read_json(screen_dir / "provenance.json")
    ff = p.read_json(screen_dir / "ff_reproduction.json")
    screen = p.read_json(screen_dir / "swap_feasibility.json")
    p.require(evidence["inputs"] == provenance, "Screen inputs differ from current verified inputs")
    p.require(evidence["checker_sha256"] == p.sha256(p.__file__), "Different preflight implementation")
    p.require(ff["status"] == "PASS" and all(ff[k] is True for k in ("last_80_set_match", "last_80_order_match", "cumulative_r5_set_match")), "FF gate failed")
    p.require(screen["status"] == "COMPLETE" and screen["checked_pairs"] == screen["expected_pairs"] == 83440 and screen["feasible_pairs"] > 0, "Require completed positive initial-swap screen")
    plan = {"version": 1, "scope": cfg["scope"], "policy": POLICY,
            "train_ids": lists["train_ids"], "r4": lists["r4"], "r5": lists["r5"],
            "sample_ids": rank_ids(lists["r4"], "R4-patch-audit-v1")[:SAMPLE_N],
            "top_k": TOP_K, "random_repeats": REPEATS,
            "inputs": provenance,
            "screen_sha256": p.sha256(screen_dir / "swap_feasibility.json"),
            "ff_report_sha256": p.sha256(screen_dir / "ff_reproduction.json"),
            "signal_code_sha256": p.sha256(__file__),
            "posthoc_code_sha256": p.sha256(Path(__file__).with_name("patch_posthoc.py")), "labels_read": False}
    p.write_json(out / "plan.json", plan)
    (out / "plan.sha256").write_text(p.sha256(out / "plan.json") + "\n")
    return plan


def load_cache(path, expected, meta_path, meta_sha, plan):
    meta = p.read_json(hashed(meta_path, meta_sha))
    p.require(meta["policy"] == POLICY and meta["used_gt"] is False, "Cache must use the frozen label-free policy")
    p.require(meta["prototype_seed"] == PROTOTYPE_SEED and meta["prototype_reservoir_limit"] == 10000, "Prototype settings differ from the frozen proposal")
    p.require(meta["cls_reproduction_passed"] is True, "Reproduce canonical CLS before using new patch features")
    p.require(meta["canonical_cls_sha256"] == plan["inputs"]["features"]["sha256"], "Wrong canonical CLS cache")
    for key in ("train_ids_digest", "prototype_fit_ids_digest"):
        p.require(meta[key] == digest_ids(plan["train_ids"]), "Only this train pool may supply features/prototypes")
    for key in ("extractor_sha256", "weights_sha256", "preprocessing_sha256"):
        p.require(len(meta[key]) == 64 and all(c in "0123456789abcdef" for c in meta[key]), f"Missing provenance: {key}")
    p.require(len(meta["prototype_fit_evidence"].strip()) >= 20, "Document prototype fitting source/reservoir/stopping rules before GT review")
    p.require(len(meta["cls_check_evidence"].strip()) >= 20, "Document extraction/CLS/coordinate verification")
    with np.load(hashed(path, expected), allow_pickle=False) as data:
        p.require(set(data.files) == {"ids", "q", "valid", "prototypes"}, "Unexpected cache fields; no annotations permitted")
        ids, q, valid, prototypes = (data[k] for k in ("ids", "q", "valid", "prototypes"))
    p.require(ids.ndim == 1 and ids.dtype.kind == "U", "ids must be a Unicode array")
    ids = ids.tolist()
    p.require(len(ids) == len(set(ids)) == 1604 and set(ids) == set(plan["train_ids"]), "Cache must cover train only")
    p.require(q.shape == (1604, 16, 384) and q.dtype == np.float32, "Expected float32 q[N,16,384]")
    p.require(valid.shape == (1604, 16) and valid.dtype == np.bool_ and valid.any(axis=1).all(), "Invalid region mask")
    p.require(prototypes.shape == (256, 384) and prototypes.dtype == np.float32, "Expected float32 prototypes[256,384]")
    p.require(np.isfinite(q).all() and np.isfinite(prototypes).all(), "Nonfinite features")
    p.require(np.max(np.abs(np.linalg.norm(q[valid].astype(float), axis=1) - 1)) <= 1e-6, "q is not normalized")
    p.require(np.max(np.abs(np.linalg.norm(prototypes.astype(float), axis=1) - 1)) <= 1e-6, "prototypes not normalized")
    p.require(np.all(q[~valid] == 0), "Invalid regions must be zeroed")
    return ids, q, valid, prototypes, meta


def pool_saved_tokens(tokens, valid_xyxy):
    """CPU adapter for already-saved lastnorm patch tokens; no model invocation.
    valid_xyxy is the exact normalized valid-image rectangle in the 518 canvas.
    Weight token/cell overlap area; never use GT boxes to construct descriptors.
    """
    tokens = np.asarray(tokens)
    p.require(tokens.shape == (37, 37, 384) and np.isfinite(tokens).all(), "Expected finite lastnorm [37,37,384]")
    x1, y1, x2, y2 = valid_xyxy
    p.require(0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1, "Invalid F518 valid-image rectangle")
    q = np.zeros((16, 384), dtype=np.float32)
    valid = np.zeros(16, dtype=bool)
    lo, hi = np.arange(37) / 37, (np.arange(37) + 1) / 37
    for row in range(4):
        bottom, top = y1 + (y2-y1)*row/4, y1 + (y2-y1)*(row+1)/4
        wy = np.maximum(0, np.minimum(hi, top) - np.maximum(lo, bottom))
        for col in range(4):
            left, right = x1 + (x2-x1)*col/4, x1 + (x2-x1)*(col+1)/4
            wx = np.maximum(0, np.minimum(hi, right) - np.maximum(lo, left))
            area = wy[:, None] * wx[None, :]
            vector = np.sum(tokens.astype(float) * area[:, :, None], axis=(0, 1)) / area.sum()
            norm = np.linalg.norm(vector)
            if norm > 0:
                q[row*4+col] = vector / norm
                valid[row*4+col] = True
    return q, valid


def affinities(q, valid, prototypes):
    # At most 16x256 per image; never build all raw patch pairwise distances.
    out = np.empty((len(q), len(prototypes)), dtype=np.float64)
    for i in range(len(q)):
        sim = np.maximum(0, q[i].astype(float) @ prototypes.astype(float).T)
        sim[~valid[i]] = 0
        out[i] = sim.max(axis=0)
    return out


def contribution(q, valid, prototypes, reference_affinity):
    """Exact additive marginal contribution to the fixed facility-location objective.
    Attribute each prototype gain to its best region (smallest cell index on ties).
    """
    p.require(bool(np.any(valid)), "No valid regions")
    sim = np.maximum(0, q.astype(float) @ prototypes.astype(float).T)
    sim[~valid] = -np.inf
    winners = sim.argmax(axis=0)
    own = sim[winners, np.arange(len(prototypes))]
    gains = np.maximum(0, own - reference_affinity) / len(prototypes)
    weights = np.bincount(winners, weights=gains, minlength=16)
    gain = float(weights.sum())
    top = sorted((i for i in range(16) if valid[i] and weights[i] > 0), key=lambda i: (-weights[i], i))[:TOP_K]
    if gain <= GAIN_FLOOR:
        top = []  # do not invent salient cells from ties in a saturated objective
    return {"gain": gain, "status": "SATURATED_OR_NUMERICALLY_TINY" if not top else "POSITIVE",
            "top_regions": top, "region_weights": weights.tolist(), "valid_regions": np.flatnonzero(valid).tolist()}


def score_r4(plan, cache):
    ids, q, valid, prototypes, _ = cache
    lookup = {v: i for i, v in enumerate(ids)}
    refs = [lookup[v] for v in plan["r4"]]
    a = affinities(q[refs], valid[refs], prototypes)
    records = []
    for sample_id in plan["sample_ids"]:
        pos = plan["r4"].index(sample_id)
        idx = lookup[sample_id]
        reference = np.delete(a, pos, axis=0).max(axis=0)
        records.append({"id": sample_id, **contribution(q[idx], valid[idx], prototypes, reference)})
    return {"kind": "R4_LEAVE_ONE_OUT_DIAGNOSTIC", "records": records, "labels_read": False,
            "interpretation": "Same objective, different reference: this is NOT the R5 acquisition score or proof of detection utility."}


def match_changes(d, ids, original, final):
    """Deterministic greedy CLS matching, not optimal matching or an acquisition trace."""
    lookup = {v: i for i, v in enumerate(ids)}
    outgoing, incoming = set(original) - set(final), set(final) - set(original)
    pairs = []
    for inc in sorted(incoming):
        out = min(outgoing, key=lambda x: (float(d[lookup[inc], lookup[x]]), x))
        pairs.append((out, inc))
        outgoing.remove(out)
    return pairs


def pair_review(plan, cache, selections, cls_ids, d):
    ids, q, valid, prototypes, _ = cache
    lookup = {v: i for i, v in enumerate(ids)}
    cls_lookup = {v: i for i, v in enumerate(cls_ids)}
    a = affinities(q, valid, prototypes)
    original = set(plan["r5"]) - set(plan["r4"])
    baseline_radius = p.coverage(d, [cls_lookup[v] for v in plan["r5"]])["radius"]
    p.require(set(selections["arms"]) == {"patch", "random_1", "random_2", "random_3"}, "Need Patch and three random-control arms")
    records, key, arm_reports = [], {}, []
    arm_names = rank_ids(list(selections["arms"]), "blind-arm-v1")
    for arm_index, name in enumerate(arm_names, 1):
        arm = selections["arms"][name]
        if name in CONTROL_SEEDS:
            p.require(arm["seed"] == CONTROL_SEEDS[name], "Unexpected diagnostic control seed")
        final = arm["new_ids"]
        p.require(len(final) == len(set(final)) == 80 and set(final) <= set(ids) - set(plan["r4"]), "Invalid final batch")
        p.require(isinstance(arm["accepted_swaps"], int) and 0 <= arm["accepted_swaps"] <= 80, "Invalid swap count")
        p.require(len(set(final) - original) <= arm["accepted_swaps"], "Net replacements exceed accepted operations")
        centers = plan["r4"] + final
        cov = p.coverage(d, [cls_lookup[v] for v in centers])
        p.require(cov["radius"] <= baseline_radius, "Final selection violates epsilon=0 radius")
        pairs = match_changes(d, cls_ids, original, final)
        pair_ids = {json.dumps(pair): pair for pair in pairs}
        chosen = [pair_ids[x] for x in rank_ids(list(pair_ids), "pair-review-v1")[:20]]
        blind = f"arm_{arm_index}"
        key[blind] = name
        for j, (outgoing, incoming) in enumerate(chosen, 1):
            # Shared reference T0 minus matched outgoing: objective-linked pair diagnosis,
            # not necessarily the reference used at a sequential swap's actual step.
            reference_ids = [v for v in plan["r5"] if v != outgoing]
            reference = a[[lookup[v] for v in reference_ids]].max(axis=0)
            sides = [outgoing, incoming]
            if int(hashlib.sha256((blind + str(j)).encode()).hexdigest(), 16) % 2:
                sides.reverse()
            rec = {"pair_id": f"{blind}_{j:02}", "arm": blind, "sides": []}
            for sid in sides:
                idx = lookup[sid]
                rec["sides"].append({"id": sid, **contribution(q[idx], valid[idx], prototypes, reference)})
            key[rec["pair_id"]] = {"A": "incoming" if sides[0] == incoming else "outgoing", "B": "incoming" if sides[1] == incoming else "outgoing"}
            records.append(rec)
        arm_reports.append({"arm": blind, "net_replacements": len(pairs), "accepted_swaps": arm["accepted_swaps"],
                            "coverage": cov, "sampled_pairs": len(chosen)})
    counts = [x["accepted_swaps"] for x in arm_reports]
    return {"kind": "FROZEN_FINAL_PAIR_REVIEW", "records": records, "arms": arm_reports,
            "matched_operation_counts": len(set(counts)) == 1, "labels_read": False,
            "interpretation": "T0-minus-matched-outgoing marginal contributions. Not sequential selection causality. Eight is not a success threshold."}, key


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--config", required=True, type=Path)
    f.add_argument("--screen-dir", required=True, type=Path)
    for cmd in ("score", "pairs"):
        s = sub.add_parser(cmd)
        for name in ("plan", "cache", "meta"):
            s.add_argument("--" + name, required=True, type=Path)
            s.add_argument("--" + name + "-sha", required=True)
        if cmd == "pairs":
            s.add_argument("--config", required=True, type=Path)
            s.add_argument("--selections", required=True, type=Path)
            s.add_argument("--selections-sha", required=True)
    for parser in (f, *[sub.choices[x] for x in ("score", "pairs")]):
        parser.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error("Output exists; choose a new directory")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        if args.command == "freeze":
            freeze(args.config, args.screen_dir, args.out)
        else:
            plan = load_plan(args.plan, args.plan_sha)
            p.require(plan["signal_code_sha256"] == p.sha256(__file__), "Code changed after freezing; review and freeze a new protocol version")
            cache = load_cache(args.cache, args.cache_sha, args.meta, args.meta_sha, plan)
            if args.command == "score":
                result = score_r4(plan, cache)
            else:
                selections = p.read_json(hashed(args.selections, args.selections_sha))
                for name in ("plan", "cache", "meta"):
                    p.require(selections[name + "_sha256"] == getattr(args, name + "_sha"), "Selections used different frozen artifacts")
                p.require(selections["used_gt"] is False, "Selection must not use GT")
                _, features, lists, provenance, _ = p.load_inputs(args.config)
                p.require(provenance == plan["inputs"], "Different CLS input evidence")
                result, key = pair_review(plan, cache, selections, lists["cache_ids"], p.distance_matrix(features))
                p.write_json(args.out / "unblind_key.json", key)
                result["selections_sha256"] = args.selections_sha
            result.update(plan_sha256=args.plan_sha, cache_sha256=args.cache_sha, meta_sha256=args.meta_sha,
                          signal_code_sha256=p.sha256(__file__))
            p.write_json(args.out / "scores.json", result)
            (args.out / "scores.sha256").write_text(p.sha256(args.out / "scores.json") + "\n")
        print("Prepared frozen diagnostic artifacts; no extraction or training ran.")
        return 0
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        p.write_json(args.out / "blocked.json", {"status": "BLOCKED", "error": str(exc)})
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""CPU-only FF replay and initial one-for-one feasibility audit. No model imports."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

# Stable default for the NEW checker; the original environment must be reviewed.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import numpy as np

TIE_ATOL = 1e-12  # FF tie handling only. NEVER a radius allowance.
INPUTS = ("features", "cache_ids", "train_ids", "r4", "r5", "new_order",
          "selector_source", "feature_manifest")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_ids(path, spec):
    """Explicit format/key: never infer ordering from a set or a sorted manifest."""
    fmt = spec["format"]
    if fmt == "json":
        values = read_json(path)
        for key in spec.get("keys", []):
            values = values[key]
    elif fmt == "npy":
        values = np.load(path, allow_pickle=False).tolist()
    elif fmt == "txt":
        values = Path(path).read_text(encoding="utf-8-sig").splitlines()
    else:
        raise ValueError(f"Unsupported ID format: {fmt}; export an explicitly ordered JSON array")
    require(isinstance(values, list) and values, "IDs must be a non-empty list")
    require(all(isinstance(x, str) and x and x == x.strip() for x in values),
            "IDs must be non-empty, unmodified strings (no implicit conversions)")
    require(len(values) == len(set(values)), "Duplicate IDs")
    return values


def distance_matrix(features):
    """No renormalization, clipping or forced zero diagonal; use saved float32 values."""
    x = features.astype(np.float64)
    return 1.0 - x @ x.T


def replay_ff(distances, ids, r4, count):
    lookup = {v: i for i, v in enumerate(ids)}
    references = [lookup[v] for v in r4]
    remaining = sorted(set(ids) - set(r4))
    nearest = distances[:, references].min(axis=1)
    chosen, trace = [], []
    for step in range(count):
        rows = np.array([lookup[v] for v in remaining], dtype=int)
        scores = nearest[rows]
        best = float(scores.max())
        tied = np.flatnonzero(np.isclose(scores, best, rtol=0, atol=TIE_ATOL))
        pick = int(tied[0])  # remaining is lexicographically sorted opaque IDs
        sample_id = remaining.pop(pick)
        chosen.append(sample_id)
        trace.append({"step": step + 1, "id": sample_id,
                      "max_score": best, "chosen_score": float(scores[pick]),
                      "tie_count": int(len(tied))})
        nearest = np.minimum(nearest, distances[:, lookup[sample_id]])
    return chosen, trace


def coverage(distances, centers):
    nearest = distances[:, centers].min(axis=1)
    return {"radius": float(nearest.max()), "mean": float(nearest.mean()),
            "q95": float(np.quantile(nearest, .95))}


def load_inputs(config_path):
    cfg = read_json(config_path)
    require(cfg.get("scope") == {"dataset": "GC10-DET", "acquisition_seed": 2026091901,
                                 "transition": "R4_to_R5"}, "Scope must be GC10 / 2026091901 / R4->R5")
    review = cfg.get("original_protocol_review", {})
    require(review.get("confirmed") is True and len(review.get("notes", "").strip()) >= 20,
            "First inspect the ORIGINAL selector/cache/order/tie arithmetic; record review notes")
    paths, provenance = {}, {}
    for name in INPUTS:
        spec = cfg["inputs"][name]
        path = Path(spec["path"])
        if not path.is_absolute():
            path = Path(config_path).resolve().parent / path
        path = path.resolve()
        require(path.is_file(), f"Missing {name}: {path}")
        expected = spec.get("sha256", "")
        require(len(expected) == 64 and all(c in "0123456789abcdef" for c in expected),
                f"Supply the verified historical SHA256 for {name}")
        actual = sha256(path)
        require(actual == expected, f"SHA256 mismatch: {name}")
        require(bool(spec.get("hash_evidence", "").strip()), f"Record hash source for {name}")
        paths[name] = path
        provenance[name] = {"path": str(path), "sha256": actual,
                            "hash_evidence": spec["hash_evidence"]}
    lists = {name: read_ids(paths[name], cfg["inputs"][name])
             for name in ("cache_ids", "train_ids", "r4", "r5", "new_order")}
    ids, train, r4, r5, order = (lists[k] for k in ("cache_ids", "train_ids", "r4", "r5", "new_order"))
    require(len(ids) == len(train) == 1604 and set(ids) == set(train), "Train/cache IDs must be the same 1604 IDs")
    require(len(r4) == 481 and len(r5) == 561 and len(order) == 80, "Expected R4=481, R5=561, new=80")
    require(set(r4) < set(r5) <= set(train), "Invalid nesting or non-train selection")
    require(set(order) == set(r5) - set(r4), "Original acquisition order must contain exactly R5 minus R4")
    features = np.load(paths["features"], allow_pickle=False)
    require(isinstance(features, np.ndarray) and features.shape == (1604, 384), "Expected .npy shape (1604,384)")
    require(features.dtype == np.dtype("float32"), "Canonical cache must be float32")
    require(np.isfinite(features).all(), "Non-finite feature values")
    norm_error = float(np.max(np.abs(np.linalg.norm(features.astype(np.float64), axis=1) - 1)))
    require(norm_error <= 1e-6, "Cache is not normalized; do NOT silently normalize it")
    provenance["config"] = {"sha256": sha256(config_path)}
    return cfg, features, lists, provenance, norm_error


def verify(features, lists):
    d = distance_matrix(features)
    predicted, trace = replay_ff(d, lists["cache_ids"], lists["r4"], 80)
    expected = lists["new_order"]
    set_match = set(predicted) == set(expected)
    order_match = predicted == expected
    report = {"status": "PASS" if set_match and order_match else "BLOCKED_FF_MISMATCH",
              "last_80_set_match": set_match, "last_80_order_match": order_match,
              "cumulative_r5_set_match": set(lists["r4"]) | set(predicted) == set(lists["r5"]),
              "missing_ids": sorted(set(expected) - set(predicted)),
              "unexpected_ids": sorted(set(predicted) - set(expected)),
              "first_order_mismatch": next((i + 1 for i, (a, b) in enumerate(zip(predicted, expected)) if a != b), None),
              "trace": trace,
              "claim": "This checks one R4->R5 transition, not feature re-extraction or all historical rounds."}
    return d, report


def screen_initial_swaps(d, ids, r4, new_order, *, seconds, chunk_size=32, on_pair=None):
    """All initial one-for-one swaps, never sequentially mutate the original set."""
    require(math.isfinite(seconds) and seconds > 0, "seconds must be finite and positive")
    require(chunk_size > 0, "chunk_size must be positive")
    start = time.monotonic()
    lookup = {v: i for i, v in enumerate(ids)}
    selected = [lookup[v] for v in r4 + new_order]
    incoming = [lookup[v] for v in sorted(set(ids) - set(r4) - set(new_order))]
    ds = d[:, selected]
    owner = ds.argmin(axis=1)
    first = ds.min(axis=1)
    second = np.partition(ds, 1, axis=1)[:, 1]
    radius = float(first.max())
    checked = feasible = near_rejected = 0
    in_ids, out_ids = set(), set()
    complete = True
    for out_pos, out_id in enumerate(new_order, start=len(r4)):
        without = np.where(owner == out_pos, second, first)
        for offset in range(0, len(incoming), chunk_size):
            if time.monotonic() - start >= seconds:
                complete = False
                break
            chunk = incoming[offset:offset + chunk_size]
            radii = np.minimum(without[:, None], d[:, chunk]).max(axis=0)
            checked += len(chunk)
            for idx, new_radius in zip(chunk, radii):
                delta = float(new_radius - radius)
                if new_radius <= radius:  # exact stored float64 comparison; epsilon = 0
                    feasible += 1
                    in_ids.add(ids[idx])
                    out_ids.add(out_id)
                    if on_pair:
                        on_pair(out_id, ids[idx], float(new_radius), delta)
                elif delta <= TIE_ATOL:
                    near_rejected += 1  # numerical boundary disclosed, never admitted
        if not complete:
            break
    return {"status": "COMPLETE" if complete else "INCOMPLETE_TIME_LIMIT",
            "scope": "INITIAL_STATE_ONE_FOR_ONE_ONLY", "epsilon": 0,
            "expected_pairs": len(new_order) * len(incoming), "checked_pairs": checked,
            "feasible_pairs": feasible, "unique_incoming": len(in_ids),
            "unique_outgoing": len(out_ids), "small_positive_deltas_rejected": near_rejected,
            "baseline_coverage": coverage(d, selected), "elapsed_seconds": time.monotonic() - start,
            "interpretation": ("Initial one-for-one swaps exist; patch usefulness is untested." if feasible else
                               "No initial one-for-one swap exists under this arithmetic and epsilon=0; simultaneous multi-swaps are untested." if complete else
                               "Incomplete search; zero observed pairs does not establish infeasibility."),
            "research_success": None, "patch_or_yolo_executed": False}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "screen"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New directory only; never overwritten")
    parser.add_argument("--seconds", type=float, help="Required for screen: explicit CPU screening time cap")
    args = parser.parse_args(argv)
    if args.command == "screen" and (args.seconds is None or not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("screen requires a positive finite --seconds limit")
    if args.out.exists():
        parser.error("--out already exists; choose a new output directory")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        cfg, features, lists, provenance, norm_error = load_inputs(args.config)
        config_text = io.StringIO()
        with contextlib.redirect_stdout(config_text):
            np.show_config()
        write_json(args.out / "provenance.json", {
            "inputs": provenance, "scope": cfg["scope"], "protocol_review": cfg["original_protocol_review"],
            "checker_sha256": sha256(__file__), "python": sys.version, "numpy": np.__version__,
            "platform": platform.platform(), "numpy_build": config_text.getvalue(),
            "thread_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
            "max_norm_error": norm_error, "distance": "1 - float64(saved_float32) @ transpose",
            "ff_tie_atol": TIE_ATOL, "radius_allowance": 0})
        d, result = verify(features, lists)
        write_json(args.out / "ff_reproduction.json", result)
        if result["status"] != "PASS":
            print("BLOCKED: original last 80 IDs/order not reproduced. No swap screen ran.")
            return 2
        if args.command == "verify":
            print("PASS: original last 80 IDs and acquisition order reproduced. Screen not run.")
            return 0
        with (args.out / "feasible_initial_pairs.csv").open("x", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["outgoing_id", "incoming_id", "radius", "delta_from_original_radius"])
            result = screen_initial_swaps(d, lists["cache_ids"], lists["r4"], lists["new_order"],
                                          seconds=args.seconds, on_pair=lambda *row: writer.writerow(row))
        write_json(args.out / "swap_feasibility.json", result)
        print(f"{result['status']}: checked={result['checked_pairs']}, feasible={result['feasible_pairs']}")
        return 0 if result["status"] == "COMPLETE" else 3
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        write_json(args.out / "blocked.json", {"status": "BLOCKED_INPUT_OR_PROTOCOL", "error": str(exc)})
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

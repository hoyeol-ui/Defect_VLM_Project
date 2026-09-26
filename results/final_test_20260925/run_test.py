"""Run exactly the 162 frozen scientific test evaluations, then analyze once."""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate, prepare_dev_view, write_json_x
from input_audit import METADATA, OUT, RECOVERY, check, read_json, sha

PYTHON = Path(r"C:\Users\user\Desktop\vlm\Defect_VLM_Project\.python311\python.exe")
ATTEMPT = "ANNOTATION_PATCH_v2_0_2"


def native_test_label_gate():
    import numpy as np
    from ultralytics.data.utils import verify_image_label

    inventory = read_json(OUT / "preflight/test_dataset_inventory.json")
    config = read_json(METADATA / "config/protocol.json")
    check(len(inventory) == 614, "Test inventory count mismatch")
    counts = Counter()
    class_counts = Counter()
    for row in inventory:
        ds = row["dataset"]
        image = Path(row["output_image"])
        label = Path(row["output_label"])
        check(image.is_file() and label.is_file() and sha(image) == row["image_sha256"] and
              sha(label) == row["label_sha256"], f"Staged test bytes changed: {row['sample_id']}")
        native = np.loadtxt(label, dtype=np.float32, ndmin=2)
        check(len(native) == len(np.unique(native, axis=0)), f"Native float32 duplicate label: {row['sample_id']}")
        result = verify_image_label((str(image), str(label), "test_label_gate: ", False,
                                     len(config["classes"][ds]), 0, 0, False))
        check(result[0] is not None and result[5:9] == (0, 1, 0, 0) and not result[9],
              f"Native test label warning/error: {row['sample_id']}: {result[-1]}")
        check(np.array_equal(native, result[1]), f"Native test label changed: {row['sample_id']}")
        counts[ds] += 1
        for cls in set(native[:, 0].astype(int)):
            class_counts[(ds, int(cls))] += 1
    check(counts == {"GC10-DET": 344, "NEU-DET": 270}, "Native test image count mismatch")
    for ds, names in config["classes"].items():
        check(all(class_counts[(ds, i)] > 0 for i in range(len(names))), f"Test GT class absent: {ds}")
    write_json_x(OUT / "preflight/native_test_labels.json", {
        "status": "PASS", "counts": dict(counts),
        "class_image_counts": {f"{ds}/{config['classes'][ds][i]}": n for (ds, i), n in class_counts.items()},
        "duplicate_float32_rows": 0, "native_parser_warnings": 0})


def expected_order():
    checks = read_json(METADATA / f"state_attempts/{ATTEMPT}/gates/final_integrity.json")["evidence"]["checks"]
    cells = {r["run_id"]: r for r in read_json(METADATA / "config/cells.json")}
    check(set(cells) == set(checks) and len(cells) == 162, "Scientific run matrix mismatch")
    method_order = {"SHARED": 0, "Random": 1, "DINO_F518": 2, "Entropy": 3, "Full_data": 4}
    order = sorted(cells, key=lambda run: (cells[run]["dataset"],
                    cells[run]["seed"] if cells[run]["seed"] is not None else 9999999999,
                    method_order[cells[run]["method"]],
                    cells[run]["round"] if cells[run]["round"] is not None else 99))
    return order, checks


def check_completeness(order, checks):
    results = OUT / "raw/runs"
    files = sorted(results.glob("*/result.json"))
    observed = [p.parent.name for p in files]
    missing = sorted(set(order) - set(observed))
    unexpected = sorted(set(observed) - set(order))
    duplicate = sorted(k for k, n in Counter(observed).items() if n > 1)
    bad = []
    config = read_json(METADATA / "config/protocol.json")
    for path in files:
        r = read_json(path)
        run = path.parent.name
        if (r.get("status") != "SUCCESS" or r.get("run_id") != run or
            r.get("checkpoint_sha256") != checks.get(run, {}).get("best_sha") or
            r.get("evaluation_dataset") != r.get("dataset", "") + "/test" or
            set(r.get("per_class_AP50_95", {})) != set(config["classes"].get(r.get("dataset"), [])) or
            any(not math.isfinite(v) or not 0 <= v <= 1 for v in [r.get("mAP50_95", float("nan")),
                r.get("mAP50", float("nan")), *r.get("per_class_AP50_95", {}).values()])):
            bad.append(run)
    report = {"status": "PASS" if len(files) == 162 and not (missing or unexpected or duplicate or bad)
              else "FAIL", "expected": 162, "successful": len(files), "missing": missing,
              "unexpected": unexpected, "duplicated_run_ids": duplicate, "invalid_results": bad,
              "technical_included": "TECHNICAL_NEU-DET" in observed}
    if report["technical_included"]:
        report["status"] = "FAIL"
    write_json_x(OUT / "preflight/test_completeness.json", report)
    return report


def main():
    check(read_json(OUT / "preflight/preflight.json")["status"] == "PASS", "Input preflight failed")
    check(read_json(OUT / "preflight/dev_dry_run.json")["status"] == "PASS", "Dev dry-run failed")
    check(not (OUT / "raw").exists(), "Raw results directory already exists; refuse overwrite")
    check(not (OUT / "preflight/native_test_labels.json").exists(), "Native label gate already exists")
    check(not (OUT / "preflight/test_completeness.json").exists(), "Completeness gate already exists")
    check((OUT / "analysis.py").is_file(), "Analysis method must be frozen before test")
    for ds in ("GC10-DET", "NEU-DET"):
        prepare_dev_view(ds)
    native_test_label_gate()
    order, checks = expected_order()
    lock = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "design": "metal_al_final_v2_20260919", "attempt": ATTEMPT,
            "checkpoint_count": 162, "checkpoint_run_sha256": {k: checks[k]["best_sha"] for k in order},
            "evaluation_script_sha256": sha(OUT / "evaluate.py"),
            "analysis_script_sha256": sha(OUT / "analysis.py"),
            "runner_script_sha256": sha(OUT / "run_test.py"),
            "input_audit_script_sha256": sha(OUT / "input_audit.py"),
            "dev_dry_run_sha256": sha(OUT / "preflight/dev_dry_run.json"),
            "test_inventory_sha256": sha(OUT / "preflight/test_dataset_inventory.json"),
            "native_label_gate_sha256": sha(OUT / "preflight/native_test_labels.json"),
            "evaluation_order": order, "bootstrap_draws": 100000,
            "bootstrap_seeds": {"GC10-DET": 2026091912, "NEU-DET": 2026091913},
            "primary_comparisons": ["DINO_F518 - Random", "DINO_F518 - Entropy"],
            "auxiliary_comparison": "Entropy - Random", "holmed_primary_family_size": 4}
    write_json_x(OUT / "preflight/protocol_lock.json", lock)
    (OUT / "raw/runs").mkdir(parents=True)
    for i, run_id in enumerate(order, 1):
        for name, key in (("evaluate.py", "evaluation_script_sha256"),
                          ("analysis.py", "analysis_script_sha256"),
                          ("run_test.py", "runner_script_sha256")):
            check(sha(OUT / name) == lock[key], f"Code changed after test opened: {name}")
        result = evaluate(run_id, "test")
        check(result["checkpoint_sha256"] == checks[run_id]["best_sha"], "Checkpoint drift")
        print(f"TEST_EVALUATION_COMPLETE {i}/162 {run_id}", flush=True)
    complete = check_completeness(order, checks)
    check(complete["status"] == "PASS", "Test result completeness gate failed; analysis blocked")
    cmd = [str(PYTHON), "-B", str(OUT / "analysis.py")]
    finished = subprocess.run(cmd, cwd=OUT, check=False)
    check(finished.returncode == 0, "Post-evaluation analysis failed; raw results preserved")
    print("FINAL TEST COMPLETE", flush=True)


if __name__ == "__main__":
    main()

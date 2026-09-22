"""Read-only structural audit of the clean-split v2 development export."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
SOURCE = HERE / "evidence/final_main_runpod_execution_v2_20260919"
STATE = SOURCE / "state_attempts/ANNOTATION_PATCH_v2_0_2"
OUT = Path(__file__).resolve().parents[1] / "00_preflight/preflight_machine.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    cfg = read(SOURCE / "config/protocol.json")
    cells = read(SOURCE / "config/cells.json")
    latest = {}
    ledger = [json.loads(line) for line in (STATE / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    for event in ledger:
        latest[event["run_id"]] = event
    require(len(cells) == 162 and len({c["run_id"] for c in cells}) == 162, "cell matrix is not 162 unique runs")
    require({c["run_id"] for c in cells} == {rid for rid in latest if not rid.startswith("TECHNICAL_")}, "ledger/cell matrix mismatch")
    require(all(latest[c["run_id"]]["status"] == "COMPLETE" for c in cells), "non-complete scientific run")
    require(read(STATE / "gates/final_integrity.json")["status"] == "PASS", "final integrity failed")
    require(not (STATE / "STOP.json").exists(), "current attempt has STOP")
    require("TEST NOT RELEASED" in (STATE / "logs/controller-20260919T102406Z-9632.log").read_text(), "controller log lacks test boundary")
    require(len(cfg["seeds"]) == 5, "seed count")
    detail = {}
    for dataset, p in cfg["datasets"].items():
        counts = p["counts"]
        with (SOURCE / "inputs" / dataset / "SPLIT_MEMBERSHIP.csv").open(encoding="utf-8", newline="") as f:
            membership = list(csv.DictReader(f))
        split_counts = collections.Counter(row["split"] for row in membership)
        require([split_counts[x] for x in ("train", "dev", "test")] == [p["train"], p["dev"], p["test"]], f"split count {dataset}")
        train = {row["sample_id"] for row in membership if row["split"] == "train"}
        require(len(train) == p["train"], f"train ID uniqueness {dataset}")
        image_ids = {r["id"] for r in read(SOURCE / "inputs" / dataset / "train_images.json")}
        require(image_ids == train, f"train image IDs {dataset}")
        require(read(STATE / "features" / dataset / "manifest.json")["shape"] == [p["train"], 384], f"feature cache shape {dataset}")
        require(read(STATE / "gates/features_valid.json")["evidence"][dataset]["coverage"] == p["train"], f"feature coverage {dataset}")
        expected = {(seed, "SHARED", 0) for seed in cfg["seeds"]}
        expected |= {(seed, method, rnd) for seed in cfg["seeds"] for method in ("Random", "DINO_F518", "Entropy") for rnd in range(1, 6)}
        actual = {(c["seed"], c["method"], c["round"]) for c in cells if c["dataset"] == dataset and c["method"] != "Full_data"}
        require(actual == expected, f"matrix {dataset}")
        full = [c for c in cells if c["dataset"] == dataset and c["method"] == "Full_data"]
        require(len(full) == 1 and full[0]["labeled_count"] == p["train"], f"full reference {dataset}")
        share = {}
        for c in (x for x in cells if x["dataset"] == dataset):
            run = STATE / "runs" / c["run_id"]
            recorded = read(run / "cell.json")
            require({k: recorded[k] for k in ("run_id", "dataset", "method", "seed", "round", "labeled_count")} == {k: c[k] for k in ("run_id", "dataset", "method", "seed", "round", "labeled_count")}, f"cell mismatch {c['run_id']}")
            require(read(run / "resolved_args.json")["seed"] == cfg["learner_seed"], f"learner seed {c['run_id']}")
            metrics = read(run / "sealed_dev_metrics.json")
            require(metrics["partition"] == "development", f"partition {c['run_id']}")
            require(all(math.isfinite(v) and 0 <= v <= 1 for v in metrics["overall"].values()), f"overall metric {c['run_id']}")
            require(all(math.isfinite(v) and 0 <= v <= 1 for v in metrics["per_class_AP50_95"].values()), f"class metric {c['run_id']}")
            require(len(metrics["per_class_AP50_95"]) == (10 if dataset == "GC10-DET" else 6), f"class count {c['run_id']}")
            require(abs(metrics["overall"]["fitness"] - metrics["overall"]["metrics/mAP50-95(B)"]) < 1e-12, f"fitness metric {c['run_id']}")
            epochs=read(run / "epoch_metrics.json")
            require(len(epochs) == 100, f"epoch count {c['run_id']}")
            with (run / "train/results.csv").open(encoding="utf-8",newline="") as f:
                result_rows=list(csv.DictReader(f))
            require(len(result_rows)==100, f"train/results.csv count {c['run_id']}")
            for e,row in zip(epochs,result_rows):
                require(e["epoch"]==int(row["epoch"]) and abs(e["fitness"]-float(row["metrics/mAP50-95(B)"]))<1e-7, f"training metric trace {c['run_id']}")
            best=read(run / "best_selection.json")
            require(best["all_epochs"]==100 and best["metric"]=="development_mAP50-95",f"best policy {c['run_id']}")
            require(best["epoch"]==max(epochs,key=lambda e:(e["fitness"],e["epoch"]))["epoch"],f"best epoch {c['run_id']}")
            require(abs(best["fitness"]-epochs[best["epoch"]-1]["fitness"])<1e-7,f"best fitness {c['run_id']}")
            artifact = read(run / "artifacts.json")
            require(artifact["files"]["sealed_dev_metrics.json"] == sha(run / "sealed_dev_metrics.json"), f"sealed metric SHA {c['run_id']}")
            require(artifact["files"]["best_selection.json"] == sha(run / "best_selection.json"), f"best selection SHA {c['run_id']}")
            require(artifact["files"]["train/weights/best.pt"] == latest[c["run_id"]]["best_sha"], f"best checkpoint SHA {c['run_id']}")
            require(best["best_sha"]==artifact["files"]["train/weights/best.pt"],f"best checkpoint manifest {c['run_id']}")
            sel = read(run / "selection_manifest.json")
            ids = sel["cumulative_ids"]
            expected_n = p["train"] if c["method"] == "Full_data" else counts[c["round"]]
            require(len(ids) == len(set(ids)) == c["labeled_count"] == expected_n, f"selection size {c['run_id']}")
            require(set(ids) <= train, f"selection outside train {c['run_id']}")
            require(sel["split_sha"] == p["split_sha"], f"split SHA {c['run_id']}")
            if c["method"] == "SHARED":
                share[c["seed"]] = set(ids)
            elif c["method"] != "Full_data":
                require(share[c["seed"]] <= set(ids), f"R0 divergence {c['run_id']}")
        # Each method's cumulative membership must be nested across rounds.
        for seed in cfg["seeds"]:
            for method in ("Random", "DINO_F518", "Entropy"):
                prior = share[seed]
                for rnd in range(1, 6):
                    rid = next(c["run_id"] for c in cells if c["dataset"] == dataset and c["seed"] == seed and c["method"] == method and c["round"] == rnd)
                    ids = set(read(STATE / "runs" / rid / "selection_manifest.json")["cumulative_ids"])
                    require(prior < ids, f"non-nested acquisition {rid}")
                    prior = ids
        detail[dataset] = {"corpus_n": p["n"], "acquisition_pool_n": p["train"], "split_counts": dict(split_counts), "counts": counts, "percentages": [100 * x / p["train"] for x in counts], "seeds": cfg["seeds"], "full_reference_runs": [x["run_id"] for x in full]}
    report = {"status": "PASS", "attempt": "ANNOTATION_PATCH_v2_0_2", "scientific_runs": len(cells), "methods": dict(collections.Counter(c["method"] for c in cells)), "learner_seed": cfg["learner_seed"], "datasets": detail, "metric_source": "state_attempts/ANNOTATION_PATCH_v2_0_2/runs/<run_id>/sealed_dev_metrics.json", "final_integrity": "PASS", "test_released": False}
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

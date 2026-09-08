"""Verify the public CPU diagnostic package; private source checks are opt-in.

Default: verify sealed public bytes and output consistency without source access.
--seal: create a new output seal once, after all report/figure edits are complete.
--check-sources: rehash the explicit analysis-input list and stat protected paths.
No model framework, inference, training, selection, or evaluation is invoked.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEAL = ROOT / "OUTPUT_SHA256.json"
IGNORED = {"__pycache__", ".pytest_cache"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def package_files():
    return sorted(p for p in ROOT.rglob("*") if p.is_file()
                  and p != SEAL and not IGNORED.intersection(p.relative_to(ROOT).parts))


def check_tables():
    expected = {"chance_baseline.csv": 2, "class_compactness.csv": 16,
                "selection_vs_geometry.csv": 96, "spearman_by_round.csv": 40,
                "permutation_null_samples.csv": 20000}
    tables = {}
    for name, count in expected.items():
        with (ROOT / "results" / name).open(encoding="utf-8", newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
        require(len(tables[name]) == count, f"Wrong row count: {name}")
    chance = {r["dataset"]: r for r in tables["chance_baseline.csv"]}
    require(set(chance) == {"GC10-DET", "NEU-DET"}, "Missing dataset")
    nulls = tables["permutation_null_samples.csv"]
    require(Counter(r["dataset"] for r in nulls) == {d: 10000 for d in chance},
            "Wrong permutation counts")
    for dataset, row in chance.items():
        observed, mean = float(row["observed_rate"]), float(row["permutation_mean"])
        require(math.isclose(observed - mean, float(row["observed_minus_chance"]), abs_tol=1e-10),
                "Chance difference mismatch")
        values = [float(r["shared_label_rate"]) for r in nulls if r["dataset"] == dataset]
        require(all(0 <= value <= 1 for value in values), "Invalid null rate")
        require(math.isclose(sum(values) / len(values), mean, abs_tol=1e-10), "Null mean mismatch")
    metrics = read_json(ROOT / "analysis_config.json")["geometry_order"]
    compact = {(r["dataset"], r["class_name"]): r for r in tables["class_compactness.csv"]}
    require(len(compact) == 16, "Duplicate class")
    require(Counter(key[0] for key in compact) == {"GC10-DET": 10, "NEU-DET": 6}, "Class counts")
    seen = set()
    for row in tables["selection_vs_geometry.csv"]:
        dataset, name, round_id = row["dataset"], row["class_name"], int(row["round"])
        key = (dataset, name, round_id)
        require(key not in seen and 0 <= round_id <= 5, "Duplicate or invalid class/round")
        seen.add(key)
        require(all(row[m] == compact[(dataset, name)][m] for m in metrics), "Geometry join mismatch")
        require(all(0 <= float(row[m]) <= 2 for m in metrics), "Invalid cosine distance")
        random, dino = float(row["historical_random_mean"]), float(row["dino_selected_mean"])
        require(math.isclose(dino - random, float(row["dino_minus_random"]), abs_tol=1e-9), "Count difference")
        require(math.isclose(dino / random, float(row["selection_ratio_dino_over_random"]), abs_tol=1e-9), "Count ratio")
        expected_count = int(row["labeled_count"]) * int(row["available_class_images"]) / int(chance[dataset]["pool_images"])
        require(math.isclose(expected_count, float(row["random_expected_selected_count"]), abs_tol=1e-8), "Random expectation")
        require(int(row["acquisition_trajectories"]) == 10, "Wrong selection unit")
        if round_id == 0:
            require(random == dino, "Shared R0 mismatch")
    correlations = tables["spearman_by_round.csv"]
    expected_keys = {(d, r, m) for d in chance for r in range(1, 6) for m in metrics}
    require({(r["dataset"], int(r["round"]), r["geometry_metric"]) for r in correlations} == expected_keys,
            "Missing or duplicate metric/round correlation")
    require(all(-1 <= float(r["spearman_rho"]) <= 1 and r["p_value_computed"] == "False"
                for r in correlations), "Invalid descriptive correlation")
    integrity = read_json(ROOT / "results/analysis_integrity.json")
    require(integrity["status"] == "PASS", "Analysis integrity not PASS")
    require(digest(ROOT / "analysis_config.json") == integrity["config_sha256"], "Config changed")
    require(digest(ROOT / "analyze.py") == integrity["script_sha256"], "Analysis source changed")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--check-sources", action="store_true")
    args = parser.parse_args()
    if args.seal:
        require(not SEAL.exists(), "Existing seal must not be overwritten")
    else:
        sealed = read_json(SEAL)
        actual = {p.relative_to(ROOT).as_posix(): digest(p) for p in package_files()}
        require(actual == sealed["files"], "Public package bytes or file inventory changed")
    counts = check_tables()
    source_count = protected_count = 0
    if args.check_sources:
        # Only files already read by the permitted analysis are re-opened.
        for row in read_json(ROOT / "results/input_files_and_hashes.json"):
            require(digest(row["path"]) == row["sha256"], f"Source changed: {row['path']}")
            source_count += 1
        # Protected checkpoint/test paths are stat-only: content is never opened.
        for row in read_json(ROOT / "results/protected_metadata_before.json"):
            stat = Path(row["path"]).stat()
            require(stat.st_size == row["size"] and stat.st_mtime_ns == row["mtime_ns"],
                    f"Protected metadata changed: {row['path']}")
            protected_count += 1
    if args.seal:
        sealed = {"algorithm": "SHA-256", "scope": "All public package files except this seal and Python caches",
                  "files": {p.relative_to(ROOT).as_posix(): digest(p) for p in package_files()}}
        with SEAL.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(sealed, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps({"status": "PASS", "sealed_files": len(sealed["files"]), "csv_rows": counts,
                      "source_hashes_rechecked": source_count, "protected_paths_stat_rechecked": protected_count,
                      "seal_sha256": digest(SEAL)}))


if __name__ == "__main__":
    main()

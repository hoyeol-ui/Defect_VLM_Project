"""CPU-only verification of the public controlled-result package.

This script does not access datasets, checkpoints, or private run directories.
It independently checks published table hashes, cardinalities, paired means,
exact sign-flip p-values, and two-test Holm adjustment.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "thesis_final" / "tables"

FILES = {
    "locked_final_primary.csv": "8891f3905e0510372d02bb06a2b9abd853d38aece3980caa9075ad4181cecf1a",
    "acquisition_paired_effects.csv": "205411b2dc7d4483a745dafe2f7c6b73e3cf15483e2c515e5db7b0c3023b665e",
    "budget_level.csv": "1fba8cfa0dff6b32f3c01866ed6f6ca6cec014f26490400a746b09f2bcb95c12",
    "curve_level_aulc.csv": "54128964654d9954f9f550062ffa0e950d82c8e693c4bf8b7d88f7a35810b8ab",
    "secondary_acquisition_effects.csv": "b1b1f0deb4c4d778545ba01f976a9994d90296c3ca35f5955be7a3c000d532b4",
    "secondary_subgroup.csv": "75f963c1d5a0e29be7a18adf8172570cfab99463d67cdc09bba0274e03a6ea11",
    "classwise_final.csv": "0b3849f6e7daf866da9f5e9f144b68706bf3d29db4959605a5ba4ad92d95c93c",
    "selection_composition_raw.csv": "32fedb86d2fa1aabdb4c7445e5740dec2ee188f14f750c7ab3c60767da353c4f",
    "selection_composition_summary.csv": "da75425fd95695c245e9a1aacd2b21a8d7bd94dc773e6dc9dd188cb2b368ecc3",
    "training_seed_sensitivity.csv": "ae40720abdd47d00902d19f8eafe10b13231ecd22d8d507cd49e803fdd1ba7d3",
    "locked_final_conceptual_480.csv": "ff0e6861d44efa5d9266c4a216b601f28fd7f1acd2541a85f385f652a39f3a86",
}

EXPECTED = {
    "GC10-DET": {
        "random": 0.24464283988417482,
        "dino": 0.25045436582161146,
        "delta": 0.005811525937436679,
        "p": 0.001953125,
        "holm": 0.00390625,
        "wins": 10,
    },
    "NEU-DET": {
        "random": 0.35194952207373686,
        "dino": 0.3577310868587151,
        "delta": 0.005781564784978243,
        "p": 0.00390625,
        "holm": 0.00390625,
        "wins": 9,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def exact_sign_flip(values: list[float]) -> float:
    observed = abs(sum(values) / len(values))
    extreme = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(sum(sign * value for sign, value in zip(signs, values)) / len(values))
        extreme += statistic >= observed - 1e-15
    return extreme / total


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, name: str) -> None:
    require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), f"{name}: {actual} != {expected}")


def main() -> None:
    for name, expected_hash in FILES.items():
        path = TABLES / name
        require(path.is_file(), f"missing table: {path}")
        require(sha256(path) == expected_hash, f"hash mismatch: {name}")

    require(len(rows("locked_final_conceptual_480.csv")) == 480, "conceptual rows != 480")
    require(len(rows("curve_level_aulc.csv")) == 80, "learner curves != 80")
    require(len(rows("acquisition_paired_effects.csv")) == 20, "paired effects != 20")
    require(len(rows("selection_composition_raw.csv")) == 240, "selection states != 240")
    require(len(rows("budget_level.csv")) == 12, "budget summary rows != 12")

    paired = rows("acquisition_paired_effects.csv")
    summary = {row["dataset"]: row for row in rows("locked_final_primary.csv")}
    raw_p: dict[str, float] = {}
    for dataset, expected in EXPECTED.items():
        subset = [row for row in paired if row["dataset"] == dataset]
        require(len(subset) == 10, f"{dataset}: acquisition count != 10")
        random_values = [float(row["random_two_training_seed_mean"]) for row in subset]
        dino_values = [float(row["dino_two_training_seed_mean"]) for row in subset]
        deltas = [float(row["dino_minus_random"]) for row in subset]
        close(sum(random_values) / 10, expected["random"], f"{dataset} Random mean")
        close(sum(dino_values) / 10, expected["dino"], f"{dataset} DINO mean")
        close(sum(deltas) / 10, expected["delta"], f"{dataset} paired delta")
        require(sum(value > 0 for value in deltas) == expected["wins"], f"{dataset}: wins")
        raw_p[dataset] = exact_sign_flip(deltas)
        close(raw_p[dataset], expected["p"], f"{dataset} exact p")
        close(float(summary[dataset]["mean_dino_minus_random"]), expected["delta"], f"{dataset} summary delta")

    ordered = sorted(raw_p, key=raw_p.get)
    running = 0.0
    holm: dict[str, float] = {}
    for rank, dataset in enumerate(ordered):
        adjusted = min(1.0, raw_p[dataset] * (len(ordered) - rank))
        running = max(running, adjusted)
        holm[dataset] = running
    for dataset, expected in EXPECTED.items():
        close(holm[dataset], expected["holm"], f"{dataset} Holm")
        close(float(summary[dataset]["holm_adjusted_p"]), expected["holm"], f"{dataset} published Holm")

    integrity = json.loads((TABLES / "analysis_integrity.json").read_text(encoding="utf-8"))
    require(integrity["status"] == "PASS", "analysis integrity status is not PASS")
    require(integrity["raw_scientific_evaluations"] == 440, "raw evaluations != 440")
    require(integrity["missing_or_interpolated_primary_points"] == 0, "missing/interpolated primary points")
    require(integrity["cross_dataset_pooling"] is False, "cross-dataset pooling unexpectedly enabled")

    print("Verified 11 table hashes, structural counts, paired means, exact sign-flip, and Holm values.")
    print("PUBLIC RESULT VERIFICATION: PASS")


if __name__ == "__main__":
    main()

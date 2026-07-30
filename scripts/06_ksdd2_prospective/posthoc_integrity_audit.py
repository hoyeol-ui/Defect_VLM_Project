"""Post-hoc, non-preregistered shortcut audit using official-train data only."""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from common import load_config, project_path, read_csv, run_root, seal_file, write_json


def directional_auc(y: list[int], x: list[float]) -> dict[str, float]:
    raw = float(roc_auc_score(y, x))
    return {"raw": raw, "direction_free": max(raw, 1.0 - raw)}


def perceptual_hash(image: np.ndarray) -> int:
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_frequency = cv2.dct(resized)[:8, :8].reshape(-1)[1:]
    median = float(np.median(low_frequency))
    value = 0
    for bit in low_frequency > median:
        value = (value << 1) | int(bit)
    return value


def main() -> None:
    config = load_config()
    out = run_root(config)
    blind = read_csv(out / config["outputs"]["blind_manifest"])
    train = [row for row in blind if row["official_split"] == "train"]
    if any(row["official_split"] != "train" for row in train):
        raise RuntimeError("Official-test row entered the post-hoc audit")

    labels = {
        row["sample_id"]: row
        for row in read_csv(out / "g2_g4" / "posthoc_acquisition_labels.csv")
    }
    acquisition = read_csv(out / config["outputs"]["acquisition_manifest"])
    if set(labels) != {row["sample_id"] for row in acquisition}:
        raise RuntimeError("Post-hoc labels do not exactly match acquisition IDs")

    hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train:
        hashes[row["image_sha256"]].append(row)
    duplicate_groups = [rows for rows in hashes.values() if len(rows) > 1]
    cross_role_duplicate_groups = [
        rows for rows in duplicate_groups if len({row["role"] for row in rows}) > 1
    ]

    feature_rows = []
    acquisition_hashes: dict[str, int] = {}
    for row in acquisition:
        image = cv2.imread(str(project_path(row["image_path"])), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read {row['image_path']}")
        feature_rows.append(
            {
                "sample_id": row["sample_id"],
                "y": int(labels[row["sample_id"]]["has_defect"]),
                "numeric_id": float(int(row["sample_id"])),
                "height": float(image.shape[0]),
                "width": float(image.shape[1]),
                "mean": float(image.mean()),
                "std": float(image.std()),
                "laplacian_variance": float(cv2.Laplacian(image, cv2.CV_64F).var()),
            }
        )
        acquisition_hashes[row["sample_id"]] = perceptual_hash(image)
    development = read_csv(out / config["outputs"]["development_manifest"])
    development_hashes = {}
    for row in development:
        image = cv2.imread(str(project_path(row["image_path"])), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read {row['image_path']}")
        development_hashes[row["sample_id"]] = perceptual_hash(image)
    phash_distance_counts = {"distance_0": 0, "distance_le_2": 0, "distance_le_4": 0}
    minimum_phash_distance = 63
    for development_hash in development_hashes.values():
        for acquisition_hash in acquisition_hashes.values():
            distance = (development_hash ^ acquisition_hash).bit_count()
            minimum_phash_distance = min(minimum_phash_distance, distance)
            phash_distance_counts["distance_0"] += int(distance == 0)
            phash_distance_counts["distance_le_2"] += int(distance <= 2)
            phash_distance_counts["distance_le_4"] += int(distance <= 4)
    y = [row["y"] for row in feature_rows]
    feature_names = ["numeric_id", "height", "width", "mean", "std", "laplacian_variance"]
    univariate = {
        name: directional_auc(y, [float(row[name]) for row in feature_rows])
        for name in feature_names
    }
    x = np.array([[row[name] for name in feature_names] for row in feature_rows], dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=260726)
    simple_cv = cross_val_score(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=260726),
        ),
        x,
        np.array(y),
        cv=cv,
        scoring="roc_auc",
    )

    candidates: dict[int, set[str]] = {}
    random_sets: dict[int, set[str]] = {}
    score_aurocs = []
    for seed in [int(value) for value in config["acquisition"]["seeds"]]:
        candidates[seed] = {
            row["sample_id"]
            for row in read_csv(out / "selections" / f"selected_patchcore_seed{seed:02d}.csv")
        }
        random_sets[seed] = {
            row["sample_id"]
            for row in read_csv(out / "selections" / f"selected_random_seed{seed:02d}.csv")
        }
        scores = read_csv(
            out / "selection_scores" / f"selection_scores_seed{seed:02d}.csv"
        )
        score_aurocs.append(
            float(
                roc_auc_score(
                    [int(labels[row["sample_id"]]["has_defect"]) for row in scores],
                    [float(row["patchcore_score"]) for row in scores],
                )
            )
        )

    def pairwise_jaccard(sets: dict[int, set[str]]) -> list[float]:
        return [
            len(sets[a] & sets[b]) / len(sets[a] | sets[b])
            for a, b in itertools.combinations(sorted(sets), 2)
        ]

    candidate_jaccard = pairwise_jaccard(candidates)
    random_jaccard = pairwise_jaccard(random_sets)
    candidate_frequency = Counter(
        sample_id for selected in candidates.values() for sample_id in selected
    )
    candidate_union = set(candidate_frequency)
    all_seed_intersection = set.intersection(*candidates.values())
    candidate_union_positive = sum(int(labels[sample_id]["has_defect"]) for sample_id in candidate_union)

    highest_simple_auc = max(item["direction_free"] for item in univariate.values())
    red_flags = {
        "exact_duplicate_image_hashes": bool(duplicate_groups),
        "cross_development_acquisition_exact_duplicates": bool(cross_role_duplicate_groups),
        "single_global_feature_directional_auc_ge_0_80": highest_simple_auc >= 0.80,
        "six_feature_cv_auc_ge_0_80": float(simple_cv.mean()) >= 0.80,
    }
    audit = {
        "status": "POSTHOC_NOT_PREREGISTERED",
        "scope": "official_train acquisition/development only; official test not accessed",
        "acquisition_images": len(acquisition),
        "acquisition_positive_images": sum(y),
        "acquisition_prevalence": float(np.mean(y)),
        "exact_hash_audit": {
            "duplicate_groups": len(duplicate_groups),
            "duplicate_images": sum(len(rows) for rows in duplicate_groups),
            "cross_role_duplicate_groups": len(cross_role_duplicate_groups),
        },
        "perceptual_hash_cross_role_audit": {
            "algorithm": "63-bit grayscale DCT pHash; post-hoc diagnostic threshold",
            "minimum_distance": minimum_phash_distance,
            **phash_distance_counts,
        },
        "simple_shortcut_audit": {
            "univariate_auroc": univariate,
            "six_feature_logistic_cv_auroc_folds": simple_cv.tolist(),
            "six_feature_logistic_cv_auroc_mean": float(simple_cv.mean()),
        },
        "patchcore_posthoc_diagnostics": {
            "selection_pool_auroc_by_seed": score_aurocs,
            "selection_pool_auroc_mean": float(np.mean(score_aurocs)),
            "selection_pool_auroc_min": float(np.min(score_aurocs)),
            "candidate_pairwise_jaccard_mean": float(np.mean(candidate_jaccard)),
            "candidate_pairwise_jaccard_min": float(np.min(candidate_jaccard)),
            "random_pairwise_jaccard_mean": float(np.mean(random_jaccard)),
            "candidate_union_images": len(candidate_union),
            "candidate_union_positive_images": candidate_union_positive,
            "candidate_union_positive_fraction": candidate_union_positive / len(candidate_union),
            "candidate_all_seed_intersection_images": len(all_seed_intersection),
            "candidate_selected_all_10_seeds": sum(
                count == 10 for count in candidate_frequency.values()
            ),
        },
        "red_flags": red_flags,
        "interpretation": (
            "No obvious exact-duplicate or simple global-statistic shortcut detected."
            if not any(red_flags.values())
            else "At least one post-hoc shortcut red flag is present; inspect before learner escalation."
        ),
        "limitations": [
            "Post-hoc audit was not part of the preregistered G1-G4 decision.",
            "pHash is a coarse diagnostic and cannot exclude semantic or acquisition-session shortcuts.",
            "Ten acquisition seeds perturb initial sets on one shared KSDD2 train pool.",
        ],
    }
    target = out / "posthoc_diagnostics" / "shortcut_integrity_audit.json"
    write_json(target, audit)
    seal_file(target)

    md = f"""# KSDD2 post-hoc shortcut integrity audit

Status: **POSTHOC_NOT_PREREGISTERED**

- Acquisition prevalence: {audit['acquisition_positive_images']}/{audit['acquisition_images']} ({audit['acquisition_prevalence']:.1%})
- Exact duplicate groups: {audit['exact_hash_audit']['duplicate_groups']}
- Cross development/acquisition exact-duplicate groups: {audit['exact_hash_audit']['cross_role_duplicate_groups']}
- Cross-role pHash minimum distance: {minimum_phash_distance}
- Cross-role pHash pairs at distance ≤4: {phash_distance_counts['distance_le_4']}
- Highest single global-feature directional AUROC: {highest_simple_auc:.3f}
- Six-feature logistic 5-fold CV AUROC: {simple_cv.mean():.3f}
- PatchCore pool AUROC: mean {np.mean(score_aurocs):.3f}, minimum {np.min(score_aurocs):.3f}
- Candidate pairwise Jaccard: mean {np.mean(candidate_jaccard):.3f}, minimum {np.min(candidate_jaccard):.3f}
- Candidate union: {candidate_union_positive}/{len(candidate_union)} positive ({candidate_union_positive / len(candidate_union):.1%})
- Selected in all ten candidate sets: {len(all_seed_intersection)}

Interpretation: {audit['interpretation']}

This audit does not change the frozen G1-G4 decision. It cannot rule out
semantic or acquisition-session shortcuts, and it does not support
production-pool generalization.
"""
    md_path = out / "posthoc_diagnostics" / "shortcut_integrity_audit.md"
    md_path.write_text(md, encoding="utf-8")
    seal_file(md_path)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""CPU post-hoc diagnostics on sealed pretrained DINOv2 acquisition features.

Never import training libraries or instantiate a selector. Prior graph and
selection counts are reused; manifests are checked, never regenerated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
import time
from datetime import datetime, timezone

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {
    "GC10-DET": ["punching_hole", "welding_line", "crescent_gap", "water_spot", "oil_spot", "silk_spot", "inclusion", "rolled_pit", "crease", "waist_folding"],
    "NEU-DET": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
}
EXPECTED = {"GC10-DET": (1605, 161, 80), "NEU-DET": (1259, 126, 63)}
METRICS = ["within_class_pair_median", "within_class_knn5_mean_median", "centroid_distance_median", "centroid_distance_p90"]
INPUTS: dict[str, dict] = {}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path, role):
    path = Path(path).resolve()
    # Only explicit source files are recorded; no source-tree content scan.
    forbidden = {"images", "labels", "final_evaluation", "validation_outputs"}
    require(not forbidden.intersection(path.parts), f"Forbidden input: {path}")
    require(path.name not in {"private_source_registry.csv", "split_assignment.csv"}, "Private split/source registry not needed")
    require(path.suffix.lower() not in {".pt", ".jpg", ".jpeg", ".safetensors"}, "Model/image access prohibited")
    require(path.is_file(), f"Missing input: {path}")
    stat = path.stat()
    INPUTS[str(path)] = {"path": str(path), "role": role, "sha256": sha(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return path


def read_json(path, role):
    return json.loads(record(path, role).read_text(encoding="utf-8-sig"))


def read_table(path, role):
    return pd.read_csv(record(path, role))


def write_json(path, obj):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, mode="x", index=False, lineterminator="\n", float_format="%.12g")


def rank_correlation(x, y):
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def cosine_matrix(features):
    z = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    require(np.all(norms > 0), "Zero-norm embedding")
    z = z / norms  # Temporary CPU copy only; canonical cache remains unchanged.
    distance = 1.0 - np.clip(z @ z.T, -1.0, 1.0)
    return z, distance


def geometry(z, distances, member_indices):
    n = len(member_indices)
    require(n > 5, "Insufficient class support for within-class 5-NN")
    block = distances[np.ix_(member_indices, member_indices)].copy()
    pair_median = float(np.median(block[np.triu_indices(n, 1)]))
    np.fill_diagonal(block, np.inf)
    per_image_knn_mean = np.partition(block, 4, axis=1)[:, :5].mean(axis=1)
    centroid = z[member_indices].mean(axis=0)
    require(np.linalg.norm(centroid) > 0, "Undefined centroid direction")
    centroid /= np.linalg.norm(centroid)
    cd = 1.0 - np.clip(z[member_indices] @ centroid, -1.0, 1.0)
    return {
        "within_class_pair_median": pair_median,
        "within_class_knn5_mean_median": float(np.median(per_image_knn_mean)),
        "centroid_distance_median": float(np.median(cd)),
        "centroid_distance_p90": float(np.quantile(cd, .90, method="linear")),
    }


def label_masks(label_strings):
    sets = [frozenset(int(t) - 1 for t in str(s).split(";")) for s in label_strings]
    return sets, np.array([sum(1 << c for c in s) for s in sets], dtype=np.int64)


def chance_distribution(masks, neighbors, repeats, seed):
    require(np.all(np.arange(len(masks)) != neighbors), "Self edge in saved graph")
    rng = np.random.Generator(np.random.PCG64(seed))
    values = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        permuted = masks[rng.permutation(len(masks))]
        values[i] = np.mean((permuted & permuted[neighbors]) != 0)
    return values


def protected_stat_snapshot(root):
    guard = read_json(root / "entropy_extension_v1/historical_guard.json", "historical path/hash guard; no evaluation inputs opened")
    meta = read_json(root / "entropy_extension_v1/official_release_v1/protected_metadata.json", "protected metadata; content reads not required")
    paths = {str((root / name).resolve()) for name in guard}
    paths.update(row["path"] for row in meta)
    observed = []
    for name in sorted(paths):
        p = Path(name)
        st = p.stat()
        observed.append({"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    require(all(Path(r["path"]).stat().st_size == r["size"] and Path(r["path"]).stat().st_mtime_ns == r["mtime_ns"] for r in meta), "Prior protected metadata changed")
    return observed, len(guard), len(meta)


def load_cache(root, prior, dataset, prior_integrity, canonical):
    cache = root / "confirmatory_v1_1/cache_seals" / dataset / "cache_A"
    md = read_json(cache / "metadata.json", "canonical acquisition feature metadata")
    require(md["status"] == "COMPLETE" and md["count"] == EXPECTED[dataset][0] and md["dimension"] == 384, "Invalid cache shape metadata")
    require(md["snapshot_revision"] == prior_integrity["dino_revision"], "DINO revision changed")
    hashes = {}
    for name in ["embeddings.npy", "sample_ids.npy", "ordered_manifest.txt"]:
        p = record(cache / name, "sealed acquisition-only feature cache")
        hashes[name] = INPUTS[str(p)]["sha256"]
    require(hashes == md["file_sha256"], "Cache file hash mismatch")
    content_hash = hashlib.sha256("".join(hashes[n] for n in sorted(hashes)).encode("ascii")).hexdigest()
    require(content_hash == md["cache_content_sha256"] == canonical[dataset]["cache_content_sha256"] == prior_integrity["datasets"][dataset]["cache_content_sha256"], "Cache content seal mismatch")
    ids = np.load(cache / "sample_ids.npy", allow_pickle=False).tolist()
    features = np.load(cache / "embeddings.npy", allow_pickle=False)
    require(features.dtype == np.float32 and features.shape == (len(ids), 384), "Invalid feature array")
    require(np.isfinite(features).all() and np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-4), "Invalid feature norms")
    require(ids == sorted(set(ids)), "IDs not sorted and unique")
    require((cache / "ordered_manifest.txt").read_text(encoding="ascii").splitlines() == ids, "Cache row order mismatch")
    prefix = "gc10" if dataset == "GC10-DET" else "neu"
    blind = read_table(root / f"confirmatory_v1/data_freeze/{prefix}_acquisition_universe_blind.csv", "acquisition membership only")
    require(blind.sample_id.tolist() == ids, "Cache outside allowed acquisition pool")
    samples = prior["per_sample_nearest_neighbors.csv"]
    samples = samples[samples.dataset == dataset].set_index("sample_id").loc[ids]
    sets, masks = label_masks(samples.annotation_labels_1based)
    require(all(0 <= c < len(NAMES[dataset]) for s in sets for c in s), "Invalid class index")
    index = {s: i for i, s in enumerate(ids)}
    neighbors = np.array([index[s] for s in samples.nearest_neighbor_sample_id])
    # Validate saved graph/labels without performing a new nearest-neighbor search.
    saved_matches = samples.nearest_neighbor_any_annotation_label_agreement.to_numpy()
    require(np.array_equal(saved_matches, (masks & masks[neighbors]) != 0), "Saved graph label mismatch")
    _, saved_neighbor_masks = label_masks(samples.nearest_neighbor_annotation_labels_1based)
    require(np.array_equal(saved_neighbor_masks, masks[neighbors]), "Neighbor label-set mismatch")
    metrics = prior["dataset_metrics.csv"]
    observed = float(metrics.loc[(metrics.dataset == dataset) & (metrics.metric == "nearest_neighbor_any_annotation_label_agreement"), "value"].iloc[0])
    require(np.isclose(saved_matches.mean(), observed), "Published observed kNN rate mismatch")
    return ids, features, sets, masks, neighbors, observed


def validate_selections(root, ledger, raw, dataset, ids, sets):
    index = {s: i for i, s in enumerate(ids)}
    membership = np.array([[c in s for c in range(len(NAMES[dataset]))] for s in sets], dtype=np.int64)
    checked = 0
    seed_rows = read_table(root / "confirmatory_v1/data_freeze/seed_manifest.csv", "historical acquisition and learner seeds")
    for a in range(10):
        expected_seed = int(seed_rows.loc[(seed_rows.dataset == dataset) & (seed_rows.acquisition_index == a), "acquisition_seed"].iloc[0])
        states = {}
        for strategy, token in [("Random-PCG64-v1", "Random"), ("FrozenDINO-FF-v1", "DINO")]:
            previous = set()
            for r in range(6):
                logical = "SHARED" if r == 0 else strategy
                lr = ledger[(ledger.dataset == dataset) & (ledger.acquisition_index == a) & (ledger.strategy == logical) & (ledger["round"] == r) & (ledger.kind == "cumulative")]
                require(len(lr) == 1, "Trajectory ledger key mismatch")
                lr = lr.iloc[0]
                path = root / "confirmatory_v1_1" / Path(str(lr.path).replace("\\", "/"))
                record(path, "historical cumulative selection manifest")
                require(sha(path) == lr.sha256, "Selection manifest seal mismatch")
                selected = path.read_text(encoding="ascii").splitlines()
                require(selected == sorted(set(selected)) and set(selected).issubset(index), "Duplicate or out-of-pool selection")
                require(len(selected) == EXPECTED[dataset][1] + r * EXPECTED[dataset][2] == int(lr["count"]), "Budget mismatch")
                require(previous.issubset(selected), "Cumulative selection lost images")
                if r:
                    qr = ledger[(ledger.dataset == dataset) & (ledger.acquisition_index == a) & (ledger.strategy == strategy) & (ledger["round"] == r) & (ledger.kind == "query")]
                    require(len(qr) == 1, "Missing query manifest")
                    qp = root / "confirmatory_v1_1" / Path(str(qr.iloc[0].path).replace("\\", "/"))
                    record(qp, "historical query manifest; read-only")
                    require(sha(qp) == qr.iloc[0].sha256, "Query manifest hash mismatch")
                    require(qp.read_text(encoding="ascii").splitlines() == sorted(set(selected) - previous), "Reselection or query mismatch")
                rr = raw[(raw.dataset == dataset) & (raw.acquisition_index == a) & (raw.strategy == strategy) & (raw["round"] == r)]
                require(len(rr) == 1, "Composition key mismatch")
                rr = rr.iloc[0]
                require(rr.manifest_sha256 == lr.sha256 and int(rr.acquisition_seed) == expected_seed == int(lr.acquisition_seed), "Archived selection source mismatch")
                counts = membership[[index[s] for s in selected]].sum(axis=0)
                for c, name in enumerate(NAMES[dataset]):
                    require(counts[c] == int(rr[f"class_image_count_{c}_{name}"]), f"Archived class count mismatch: {dataset}/{a}/{r}/{name}")
                states[(token, r)] = set(selected)
                previous = set(selected)
                checked += 1
        require(states[("DINO", 0)] == states[("Random", 0)], "R0 not shared")
    return checked


def create_figures(out, chance, compact, relation, correlations):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": "pretrained_dino_geometry_20260909"})
    def save(fig, name):
        fig.savefig(out / (name + ".png"), dpi=240, facecolor="white")
        fig.savefig(out / (name + ".svg"), metadata={"Date": None})
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    fig.suptitle("POST-HOC DIAGNOSTIC | Pretrained DINOv2 1-NN label overlap")
    for ax, row in zip(axes, chance):
        lo, mu, hi, obs = [100 * row[x] for x in ["permutation_p025", "permutation_mean", "permutation_p975", "observed_rate"]]
        ax.errorbar([0], [mu], yerr=[[mu - lo], [hi - mu]], fmt="o", capsize=8, color="#667085", markersize=8)
        ax.scatter([1], [obs], c="#1570EF", s=70)
        ax.text(0, hi + 5, f"{mu:.2f}%\n[{lo:.2f}, {hi:.2f}]", ha="center")
        ax.text(1, obs - 9, f"{obs:.2f}%", ha="center", color="#175CD3")
        ax.set(xticks=[0, 1], xticklabels=["Permuted label sets", "Observed (reused)"], ylim=(0, 108), xlim=(-.5, 1.5), ylabel="Shared-label rate (%)", title=row["dataset"])
        ax.grid(axis="y", alpha=.2)
    fig.supxlabel("10,000 whole-label-set permutations; 95% null interval, not an observed-rate CI", fontsize=9)
    save(fig, "figure_1_chance_baseline")
    for j, dataset in enumerate(["NEU-DET", "GC10-DET"], 2):
        sub = relation[(relation.dataset == dataset) & (relation["round"] == 5)]
        rho = correlations[(correlations.dataset == dataset) & (correlations["round"] == 5) & (correlations.geometry_metric == METRICS[0])].iloc[0].spearman_rho
        fig, ax = plt.subplots(figsize=(10, 6.5), layout="constrained")
        ax.axhline(0, color="#98A2B3", linestyle="--", linewidth=1)
        for k, row in enumerate(sub.itertuples()):
            ax.scatter(row.within_class_pair_median, row.dino_minus_random, s=55, color="#B42318" if row.dino_minus_random < 0 else "#175CD3")
            offset = (7, -14 if k % 2 else 8)
            ax.annotate(row.class_name, (row.within_class_pair_median, row.dino_minus_random), xytext=offset, textcoords="offset points", fontsize=10)
        ax.margins(x=.25, y=.18)
        ax.set(xlabel="Within-class median cosine distance (smaller = more compact)", ylabel="R5 selected class images: Pretrained DINO - Random\n(mean over 10 acquisition indices)", title=f"POST-HOC DIAGNOSTIC | {dataset}\nClass-level Spearman rho = {rho:+.3f}; n = {len(sub)} classes")
        ax.grid(alpha=.2)
        fig.supxlabel("Annotation class membership; overlapping classes are not independent. Association is not causation.", fontsize=9)
        save(fig, f"figure_{j}_{'neu' if dataset == 'NEU-DET' else 'gc10'}_selection_geometry")
    sub = relation[(relation.dataset == "NEU-DET") & (relation["round"] == 5)].sort_values(METRICS[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, layout="constrained")
    fig.suptitle("POST-HOC DIAGNOSTIC | NEU class geometry and R5 selection")
    pos = np.arange(len(sub))
    axes[0].barh(pos, sub[METRICS[0]], color="#1570EF")
    axes[0].set(yticks=pos, yticklabels=sub.class_name, xlabel="Median within-class cosine distance\n(smaller = more compact)")
    axes[0].invert_yaxis()
    axes[1].barh(pos, sub.dino_minus_random, color=["#B42318" if v < 0 else "#1570EF" for v in sub.dino_minus_random])
    axes[1].axvline(0, color="#98A2B3")
    axes[1].set_xlabel("Pretrained DINO - Random selected images\n(mean over 10 acquisition indices)")
    for i, row in enumerate(sub.itertuples()):
        axes[0].text(row.within_class_pair_median + .004, i, f"{row.within_class_pair_median:.3f}", va="center", fontsize=9)
        axes[1].text(row.dino_minus_random + (1 if row.dino_minus_random >= 0 else -1), i, f"{row.dino_minus_random:+.1f}", ha="left" if row.dino_minus_random >= 0 else "right", va="center")
    axes[0].margins(x=.2)
    axes[1].margins(x=.3)
    for ax in axes: ax.grid(axis="x", alpha=.15)
    fig.supxlabel("Class order follows the prespecified primary diagnostic; all four metrics are retained in the tables.", fontsize=9)
    save(fig, "figure_4_neu_ranked_geometry")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("analysis_config.json"))
    args = parser.parse_args()
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    config = read_json(args.config, "analysis definitions fixed before calculations")
    record(__file__, "exact analysis source")
    require(config["geometry_order"] == METRICS and config["permutations"] >= 10000, "Analysis definition changed")
    out = args.config.parent / config["output_subdirectory"]
    require(not out.exists(), f"Refuse to overwrite outputs: {out}")
    root, previous = Path(config["source_root"]), Path(config["prior_analysis_root"])
    protected, n_guard, n_meta = protected_stat_snapshot(root)
    prior_seal = read_json(previous / "OUTPUT_SHA256.json", "prior analysis output seal")
    prior = {}
    for name in ["per_sample_nearest_neighbors.csv", "dataset_metrics.csv", "classwise_nearest_neighbor_agreement.csv", "pairwise_distance_summary.csv"]:
        prior[name] = read_table(previous / name, "reuse prior acquisition-only representation analysis")
        require(sha(previous / name) == prior_seal[name], f"Prior output hash mismatch: {name}")
    prior_integrity = read_json(previous / "integrity_record.json", "prior cache/label source verification")
    require(sha(previous / "integrity_record.json") == prior_seal["integrity_record.json"], "Prior integrity hash mismatch")
    require(prior_integrity["status"] == "PASS" and prior_integrity["locked_test_labels_read"] is False, "Prior analysis boundary failure")
    canonical = read_json(root / "confirmatory_v1_1/cache_seals/CANONICAL_CACHE_SEAL.json", "canonical acquisition cache seal")
    ledger = read_table(root / "confirmatory_v1_1/manifest_seals/trajectory_ledger.csv", "historical sealed selection ledger")
    raw = read_table(root / "confirmatory_v1_1/final_analysis/results/selection_composition_raw_240.csv", "archived selection counts only; no checkpoint or test metrics")
    require(len(raw) == 240, "Selection count table incomplete")
    old_summary = read_table(root / "confirmatory_v1_1/final_analysis/results/selection_composition_summary.csv", "reuse/verify existing selection summaries")
    postrun = read_json(root / "reports/entropy_postrun_a00_20260908/after_analysis_integrity.json", "previous official audit, provenance only")
    state_path = record(root / "entropy_extension_v1/runtime/official/registry.sqlite", "read-only official job status for handoff")
    require(sha(state_path) == postrun["official_registry_sha256"], "Official registry changed since prior audit")
    with sqlite3.connect(state_path.as_uri() + "?mode=ro", uri=True) as db:
        job_states = [dict(zip(["kind", "state", "count", "attempts"], row)) for row in db.execute("SELECT CASE WHEN node_id LIKE '%_Q_%' THEN 'acquisition' ELSE 'training' END,state,count(*),sum(attempt) FROM jobs GROUP BY 1,2")]
    official_hashes = {}
    for name in ["freeze.json", "official_release_v1/freeze.json", "authorizations/official_tranche_a00.json"]:
        p = record(root / "entropy_extension_v1" / name, "unchanged Entropy freeze/authorization metadata")
        official_hashes[name] = sha(p)
    compact, chance, relation, permutations, input_checks = [], [], [], [], []
    for dataset in config["datasets"]:
        ids, features, sets, masks, nn, observed = load_cache(root, prior, dataset, prior_integrity, canonical)
        checked = validate_selections(root, ledger, raw, dataset, ids, sets)
        class_prior = prior["classwise_nearest_neighbor_agreement.csv"]
        class_prior = class_prior[(class_prior.dataset == dataset) & (class_prior.label_basis == "annotation_object_class")]
        z, distances = cosine_matrix(features)
        n = len(ids)
        values = chance_distribution(masks, nn, config["permutations"], config["seed_by_dataset"][dataset])
        # Independent expectation check uses all off-diagonal label-set pairs.
        overlap = (masks[:, None] & masks[None, :]) != 0
        expected_null = float((overlap.sum() - np.trace(overlap)) / (n * (n - 1)))
        mc_se = float(values.std(ddof=1) / np.sqrt(len(values)))
        require(abs(values.mean() - expected_null) <= max(6 * mc_se, 1e-9), "Permutation mean incompatible with exact finite-pool null")
        chance.append({"dataset": dataset, "pool_images": n, "multilabel_images": sum(len(s) > 1 for s in sets), "observed_rate": observed, "permutation_mean": float(values.mean()), "permutation_p025": float(np.quantile(values, .025)), "permutation_p975": float(np.quantile(values, .975)), "observed_minus_chance": float(observed - values.mean()), "exact_finite_pool_chance": expected_null, "permutations": len(values), "seed": config["seed_by_dataset"][dataset], "interval_type": "95% label-permutation null interval, not confidence interval"})
        permutations.extend({"dataset": dataset, "permutation_index": i, "shared_label_rate": float(v)} for i, v in enumerate(values))
        for c, name in enumerate(NAMES[dataset]):
            members = np.array([i for i, s in enumerate(sets) if c in s])
            prior_row = class_prior[class_prior.class_name == name].iloc[0]
            require(len(members) == int(prior_row.image_support), "Prior class support mismatch")
            row = {"dataset": dataset, "class_index": c, "class_name": name, "available_class_images": len(members), "prevalence": len(members) / n, "multilabel_member_images": sum(len(sets[i]) > 1 for i in members), "prior_class_1nn_preservation": float(prior_row.nearest_neighbor_preserves_class), **geometry(z, distances, members)}
            compact.append(row)
            for r in range(6):
                sub = raw[(raw.dataset == dataset) & (raw["round"] == r)]
                col = f"class_image_count_{c}_{name}"
                rm = sub[sub.strategy == "Random-PCG64-v1"].sort_values("acquisition_index")[col].to_numpy(dtype=float)
                dm = sub[sub.strategy == "FrozenDINO-FF-v1"].sort_values("acquisition_index")[col].to_numpy(dtype=float)
                require(len(rm) == len(dm) == 10, "Missing trajectory")
                delta = float((dm - rm).mean())
                if dataset == "NEU-DET" and r:
                    prior_summary = old_summary[(old_summary.dataset == dataset) & (old_summary["round"] == r) & (old_summary.metric == col)]
                    require(len(prior_summary) == 1 and np.isclose(prior_summary.iloc[0].mean_dino_minus_random, delta), "NEU archived summary mismatch")
                labeled = EXPECTED[dataset][1] + r * EXPECTED[dataset][2]
                relation.append({**row, "round": r, "labeled_count": labeled, "acquisition_trajectories": 10, "random_expected_selected_count": labeled * len(members) / n, "historical_random_mean": float(rm.mean()), "dino_selected_mean": float(dm.mean()), "dino_minus_random": delta, "selection_ratio_dino_over_random": float(dm.mean() / rm.mean()), "dino_minus_random_expected": float(dm.mean() - labeled * len(members) / n)})
        input_checks.append({"dataset": dataset, "feature_rows": n, "cache_sha256": canonical[dataset]["cache_content_sha256"], "conceptual_states_checked": checked, "selection_counts_exact": True, "shared_R0": True, "duplicate_reselection_outside_acquisition": 0, "permutation_exact_null_check": True})
        print(f"{dataset}: cache/manifests PASS; 10,000 permutations and all four metrics complete", flush=True)
    relation_df = pd.DataFrame(relation)
    correlations = []
    for dataset in config["datasets"]:
        for r in range(1, 6):
            sub = relation_df[(relation_df.dataset == dataset) & (relation_df["round"] == r)]
            for metric in METRICS:
                correlations.append({"dataset": dataset, "round": r, "class_n": len(sub), "geometry_metric": metric, "spearman_rho": rank_correlation(sub[metric], sub.dino_minus_random), "p_value_computed": False})
    for r in protected:
        stat = Path(r["path"]).stat()
        require(stat.st_size == r["size"] and stat.st_mtime_ns == r["mtime_ns"], f"Protected artifact changed: {r['path']}")
    for row in INPUTS.values(): require(sha(row["path"]) == row["sha256"], "Input changed during analysis")
    require("torch" not in sys.modules and "ultralytics" not in sys.modules, "Training framework unexpectedly imported")
    out.mkdir()
    write_csv(out / "chance_baseline.csv", chance)
    write_csv(out / "permutation_null_samples.csv", permutations)
    write_csv(out / "class_compactness.csv", compact)
    write_csv(out / "selection_vs_geometry.csv", relation)
    write_csv(out / "spearman_by_round.csv", correlations)
    write_json(out / "input_files_and_hashes.json", sorted(INPUTS.values(), key=lambda r: r["path"]))
    write_json(out / "protected_metadata_before.json", protected)
    create_figures(out, chance, compact, relation_df, pd.DataFrame(correlations))
    write_json(out / "analysis_integrity.json", {
        "status": "PASS", "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "runtime_seconds": time.perf_counter() - start,
        "config_sha256": sha(args.config), "script_sha256": sha(__file__), "datasets": input_checks,
        "historical_guard_paths_stat_unchanged": n_guard, "prior_protected_metadata_paths_match": n_meta, "protected_unique_paths_stat_unchanged": len(protected), "all_analysis_input_hashes_unchanged": True,
        "protection_scope": "Hash checks on every analysis input plus before/after size/mtime on protected historical files. No fresh 440-checkpoint hash audit or test-content access.",
        "official_registry_unchanged": True, "official_registry_sha256": sha(state_path), "official_job_states": job_states, "entropy_frozen_hashes": official_hashes,
        "feature_extraction_calls": 0, "training_calls": 0, "selection_calls": 0, "locked_test_content_reads": 0, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "access_evidence": "Explicit input allowlist and code path; no OS-wide syscall audit claimed", "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "full_reference_run_directory_exists": (root / "runs/full_data_reference_20260908").exists(),
    })
    print(json.dumps({"status": "PASS", "output": str(out), "runtime_seconds": time.perf_counter() - start}), flush=True)


if __name__ == "__main__":
    main()

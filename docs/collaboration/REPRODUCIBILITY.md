# Reproducibility Guide

## Two reproducibility targets

### A. Public result verification (recommended first)

This verifies released tables without dataset bytes, GPU training, or private checkpoints:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-thesis.txt
.\.venv\Scripts\python.exe scripts\verify_public_results.py
```

Expected output ends with `PUBLIC RESULT VERIFICATION: PASS`. The script checks canonical table SHA-256 values, row/cardinality structure, primary Random/DINO means, paired differences, exact sign-flip p-values, and Holm values. It does not claim to regenerate detector predictions.

Canonical public inputs are in `results/thesis_final/tables/`. The authoritative original analysis package is `confirmatory_v1_1/final_analysis/results/`.

### B. Full computational reproduction

This recreates data curation, selection, 440 training jobs, locked-final evaluation, and analysis. It requires the owner-distributed GC10-DET/NEU-DET snapshots, upstream checkpoints, a compatible CUDA environment, substantial storage, and about 68 observed GPU-hours on the recorded RTX 3070 for training. A different GPU/software stack is a computational replication, not guaranteed byte identity.

Do not begin with the orchestrator. Reconstruct and verify each freeze layer first.

## Recorded controlled environment

| Item | Value |
|---|---|
| OS | Windows 10 build `10.0.26200` |
| Python | 3.11.9 |
| GPU | NVIDIA GeForce RTX 3070 |
| driver | 560.94 |
| PyTorch | `2.12.1+cu126` |
| CUDA build | 12.6 |
| cuDNN | 91002 |
| NumPy | 2.4.6 |
| Pillow | 12.3.0 |
| Transformers | 5.13.0 |
| Ultralytics | 8.4.90 |

`requirements-thesis.txt` records this evidence. It is not a newly invented universal lockfile; PyTorch/CUDA installation remains platform-specific.

## Required external inputs

- GC10-DET and NEU-DET obtained from owner/authorized sources, matching the audited snapshots in `DATASETS.md`.
- `facebook/dinov2-small` at revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, with recorded config/preprocessor/model hashes.
- `yolov8n.pt` with SHA-256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.
- Enough disk for prepared YOLO data, 440 training outputs, and checkpoints. None are bundled here.

## Freeze verification order

1. Read `confirmatory_v1/confirmatory_protocol_v1.yaml` and `confirmatory_v1_1/confirmatory_protocol_v1_1.yaml`.
2. Configure local source roots outside the repository. The original scripts contain site-local Windows paths where exact execution provenance required them; adjust path configuration in a new local reproduction copy, not in the frozen public source.
3. Run `confirmatory_v1/src/build_split_and_seed_freeze.py` and verify exact curated/split counts and hashes.
4. Run the CPU preflight/tests; require all leakage, identity, Random, DINO, AULC, and registry checks to pass.
5. Build the two full-universe DINO caches. Verify row count, uniqueness, dimension 384, finite values, nonzero norm, artifact hashes, and manifest seals.
6. Materialize acquisition trajectories and compare `confirmatory_v1_1/manifest_seals/` byte-for-byte.
7. Build the 440 registry and compare `planned_registry_seal.json`.
8. Run only under a new explicitly named reproduction output root. Never overwrite the published study outputs.
9. Require the training integrity audit before final evaluation.
10. Keep the final split locked until a result-independent authorization record exists.
11. Run final evaluation once, then analysis and plotting from the generated raw metrics.

## Controlled execution entry points

The frozen historical invocation was site-specific. Portable users should inspect `--help` and the source before running:

```powershell
python confirmatory_v1_1\full_run\src\build_full_registry.py
python confirmatory_v1_1\full_run\src\run_full_440_orchestrator.py
python confirmatory_v1_1\final_analysis\src\authorize_final_evaluation.py
python confirmatory_v1_1\final_analysis\src\run_final_evaluation.py
python confirmatory_v1_1\final_analysis\src\analyze_final_results.py
python confirmatory_v1_1\final_analysis\src\plot_final_results.py
```

Some final-analysis scripts preserve the original absolute data root for provenance. A reproduction must explicitly configure its own local root and record the change as an environmental path adaptation, while leaving protocol, manifests, seeds, model hashes, and numerical rules unchanged.

## Expected structure and invariants

- 440 planned/completed detector jobs
- 40 shared Round0, 200 Random R1–R5, 200 DINO R1–R5
- 440 raw locked-final evaluations
- 480 conceptual strategy-budget rows after Round0 duplication
- 80 complete learner curves
- 40 acquisition/strategy curves after learner averaging
- 20 paired acquisition effects
- no missing/interpolated primary point
- no GC10/NEU pooling

## Seeds and determinism

Exact acquisition and learner seeds are in the protocol/seed manifest. NumPy PCG64 version and seed sequence are part of Random's definition. DINO uses FP32, no autocast or TF32, deterministic algorithms, and sample-ID tie-breaking. YOLO is deterministic as configured, but hardware/library differences can still produce floating-point variation. Report both hash equality and numerical tolerance; do not claim bit identity if hashes differ.

## Expected outputs

- prepared/frozen indices under `confirmatory_v1/data_freeze/`
- acquisition seals under `confirmatory_v1_1/manifest_seals/`
- registry/audit under `confirmatory_v1_1/full_run/`
- final evaluation audit under `confirmatory_v1_1/final_analysis/final_evaluation/`
- tables, figures, and analysis integrity JSON under `confirmatory_v1_1/final_analysis/`

## What is intentionally unreproducible from this branch alone

Dataset pixels/labels, base/model weights, 440 `best.pt` files, private per-checkpoint caches, and raw Ultralytics run directories are excluded for licensing and size. The branch supports independent numerical audit of released results and full reproduction after authorized external inputs are obtained; it is not a self-contained dataset/weight mirror.

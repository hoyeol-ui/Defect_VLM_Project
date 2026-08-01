# Deterministic Test Harness Migration Audit

Audit date: 2026-08-01
Audited branch/HEAD: `codex/last_paper` / `eb6ab0067f7a79918857212b34bf4b663d4bce12`

## Scope and preservation boundary

This migration adds CPU-only safeguards for future active-learning runs. It does
not recalculate, rename, or rewrite historical manifests, selections, metrics,
seals, or evidence. In particular, no file under `runs/` or `docs/results/` was
used as a write target. Evidence Freeze v3, the DeepPCB STOP decision, KSDD2
G1-G5 decisions, and all official/final-test locks remain unchanged.

The requested working branch in the original harness brief was
`codex/evidence-freeze-v3`. At audit time both that local branch and
`codex/last_paper` pointed to the same HEAD, while the active working tree
contained unrelated user-authored document and script changes. The active branch
was therefore not switched and the dirty worktree was not reset.

## Pre-migration repository audit

- No `AGENTS.md`, `pyproject.toml`, `setup.cfg`, or `pytest.ini` existed.
- Existing tests lived in `tests/`, while additional executable checks lived
  beside historical scripts.
- `requirements-analysis.txt` already included PyYAML but did not include pytest.
- The unanchored `data/` ignore rule also hid `src/defect_al/data/`; it was
  narrowed to `/data/` (and the other dataset-root rules were similarly anchored)
  so production data modules are tracked while local datasets remain ignored.
- The audited historical V7 identity in
  `scripts/02_active_learning/canonical_sampling_v7.py` is
  `resolved_image_path::image_sha256`.
- Historical protocol/audit/runner code contains implicit discovery via glob and
  modification time. Representative locations are
  `prepare_evaluation_protocol_v7.py`,
  `audit_v7_dataset_split_integrity.py`, and
  `run_al_yolo_ablation_v7_full_curve.py`.
- The V7 full-curve runner retains an explicit final-test lock, but its default
  development manifest and resume directory discovery are not suitable for a
  newly frozen run contract.

The repository-wide text audit (excluding raw data, outputs, frozen runs, and
`docs/results`) found the following occurrence counts at migration time. These
counts include historical documentation and do not imply that every occurrence
is an executable defect.

| Pattern | Count |
|---|---:|
| `latest_` | 123 |
| `latest_run` | 12 |
| `st_mtime` | 50 |
| `glob(` | 109 |
| `resolved_image_path` | 49 |
| `canonical_sample_id` | 61 |
| `sample_id` | 810 |
| `sha256` | 561 |
| `final_test` | 322 |
| `official_test` | 85 |

## Final forward-only structure

```text
src/defect_al/
├─ __init__.py
├─ config.py
└─ data/
   ├─ __init__.py
   ├─ identity.py
   └─ manifests.py
tests/
├─ conftest.py
├─ test_identity.py
├─ test_config.py
└─ test_leakage.py
configs/examples/
└─ frozen_run.example.yaml
docs/
└─ test_harness_migration_audit.md
pyproject.toml
```

`src/defect_al/data/identity.py` defines the forward-only identity:

```text
cid:v1:SHA256(canonical JSON(dataset_name, relative_path, image_sha256))
```

The JSON payload never contains an absolute path. Dataset-relative paths are NFC
normalized, converted to strict POSIX form, and rejected if they are absolute,
drive-qualified, UNC, traversing, ambiguous, or glob-based. File-backed identity
also proves that the resolved image remains inside the resolved dataset root.

`src/defect_al/config.py` accepts only explicit YAML/JSON paths and sealed
train/validation/test manifest hashes. It has no latest-run helper, mtime ordering,
glob discovery, or implicit fallback. Unknown schema keys and discovery-like keys
or values fail immediately.

`src/defect_al/data/manifests.py` parses CSV/JSON identities, validates stored
`cid:v1` values against recomputation when possible, rejects duplicate canonical
IDs, and checks both canonical-ID and raw-image-SHA intersections across all three
split pairs. The raw SHA check catches renamed copies whose relative paths differ.

## Compatibility and migration policy

1. Historical IDs remain legacy IDs and must not be batch-converted.
2. Historical runners remain readable/reproducible and were not silently changed
   to a new identity or config contract.
3. A future runner should first load one explicit `FrozenRunConfig`, parse all
   referenced manifests, and call `validate_split_leakage` before selection,
   training, or evaluation.
4. New manifests should persist `canonical_id`/`canonical_sample_id` using
   `cid:v1` plus `image_sha256`. Dataset name and relative path should also be
   retained so a stored ID can be recomputed and checked.
5. The example config contains placeholder seals and is intentionally not an
   implicit pointer to any historical run.

## Verification

Synthetic checks require no dataset, model, GPU, training, or inference:

```powershell
python -m pip install -r requirements-analysis.txt
python -m pytest
```

An explicitly requested real frozen-manifest check can be added without changing
the default suite:

```powershell
$env:DEFECT_AL_FROZEN_CONFIG = "C:\path\to\frozen_run.yaml"
python -m pytest -m integration
```

If the environment variable is absent, the integration check is skipped. If it
is present but the config, manifests, seals, identities, or split boundaries are
invalid, the check fails rather than falling back to another run.

Final local verification completed with `77 passed, 2 skipped`. The skips were
the opt-in real frozen-config check and a Windows account-dependent symlink
fixture; the production escape check remains implemented and the ordinary
outside-root path test passed. The package also built and imported successfully
through an editable `pyproject.toml` installation.

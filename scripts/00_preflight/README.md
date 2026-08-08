# Forward-only research preflight

These CPU-only commands validate explicit manifests before any selector or
learner is allowed to run. They never discover a latest directory and they do
not import a detector/VLM runtime.

Use the repository analysis environment (or install
`requirements-analysis.txt`) for the GC10 builder/audit, and install the
harness package once:

```powershell
python -m pip install -r requirements-analysis.txt
python -m pip install -e .
```

Build the forward-only GC10 manifests from the explicitly named acquisition,
development, selection, and historical training-manifest sources:

```powershell
python scripts/00_preflight/build_gc10_forward_manifest_v1.py `
  --project-root . `
  --protocol-dir runs/gc10_taxonomy_protocol/gc10_protocol_20260715 `
  --selection-dir runs/gc10_taxonomy_selection_audit/gc10_random_vs_dino_200seed_20260715 `
  --detector-run-dir runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715 `
  --dataset-root data/GC10-DET `
  --manifest-dir manifests/harness/gc10_dev_confirm_harness_v1 `
  --config-output configs/frozen/gc10_dev_confirm_harness_v1.yaml `
  --report-output reports/preflight/gc10_dev_confirm_harness_v1/manifest_build_report.json `
  --run-id gc10_dev_confirm_harness_v1
```

```powershell
python scripts/00_preflight/validate_frozen_run.py `
  --project-root . `
  --config configs/frozen/gc10_dev_confirm_harness_v1.yaml `
  --output-dir reports/preflight/gc10_dev_confirm_harness_v1
```

The command exits with a `RuntimeError` before creating a PASS report when a
seal, identity, duplicate, or split boundary is invalid. A PASS is a data-input
authorization only; it is not permission to access a locked final split or to
train/evaluate a model.

Independently audit the historical cache, labels, checkpoints, training
arguments, stored metrics, inferential unit, and selection sensitivities:

```powershell
python scripts/00_preflight/audit_gc10_research_lineage.py `
  --project-root . `
  --protocol-dir runs/gc10_taxonomy_protocol/gc10_protocol_20260715 `
  --detector-run-dir runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715 `
  --forward-build-report reports/preflight/gc10_dev_confirm_harness_v1/manifest_build_report.json `
  --output-dir reports/preflight/gc10_dev_confirm_harness_v1
```

All three commands are CPU-only and perform no selection, training, detector
inference, or final-split evaluation. The strict schema's `test.csv` is the
unused acquisition audit holdout, not the locked official/final evaluation set.

# Development analysis source and public reproduction

`original/` contains the seven original Python files, including the frozen numerical-function module as present in the 86-file local package (actual files enumerated in the provenance manifest). Byte equality or path-only sanitization is recorded in `results/development_20260913/audit/analysis_package_provenance.json`. The original private pipeline depends on the immutable local source bundle and is archived for provenance, not presented as a standalone public command.

Run the public CSV verifier from the repository root:

```sh
python scripts/verify_development_20260913.py
python scripts/verify_development_20260913.py --bootstrap
```

The first command uses Python standard library only. The optional second command uses NumPy to reproduce all six 100,000-draw paired percentile bootstrap intervals. It performs no model calls and reads no private dataset/checkpoint. Figure regeneration uses the explicitly adjusted wrapper `plot_figures.py` (NumPy and Matplotlib required); it reads the published tables and geometry only. No original numerical definition is changed.

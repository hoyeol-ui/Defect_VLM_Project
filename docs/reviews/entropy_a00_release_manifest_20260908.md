# Entropy a00 audit release manifest

This branch is an additive research-audit release based on `origin/al-selector`.

Included:

- Entropy scorer, orchestration, detached-runner code, and unit/contract tests.
- Frozen protocol, training configuration, official freeze, authorization, historical guard, and compact a00 audit records.
- a00 interim development curve, nAULC summary, integrity audit, runtime estimate, and plots.
- Existing DINOv2 representation-validity code and aggregate class-wise/nearest-neighbor/distance outputs.
- Historical final-analysis tables and figures already present in `al-selector`.
- The read-only mentor/repository cross-audit at `docs/reviews/mentor_repository_audit_20260908.md`.

Excluded deliberately:

- Raw images, labels, caches, model weights, private source registries, private logs, and runtime SQLite databases.
- Integration and official training output directories containing checkpoints.
- The mentor's private PDF/DOCX files and local absolute-path-only source artifacts.
- Locked-test outputs and any new evaluation of the locked test set.

The JSON and Markdown records retain the original SHA-256 values and local provenance paths. A fresh clone therefore provides the audit evidence and code, while reproduction of training or cache generation still requires the separately protected local data/checkpoint environment.

# Locked-final GT Duplicate Annotation Audit

- Dataset: `NEU-DET`
- Split: prospectively locked final
- Sample ID: `52b858174c15ad6aa441eff6e1712d6e25fbbe961f0c0583ebebadbc0a1c6451`
- Source image: `NEU-DET/IMAGES/crazing_120.jpg`
- Source annotation: `NEU-DET/ANNOTATIONS/crazing_120.xml`
- Source annotation SHA-256: `4235dc1d18cb5d0f49554e90f9c9231446b98a581c226d3efd6877f491d2328f`
- Raw converted label lines across 359 final images: `830`
- Unique converted label lines: `829`
- Duplicate label lines: `1`
- Files containing a duplicate: `1`

The source XML contains the `crazing` bounding box `(xmin=1, ymin=81, xmax=117, ymax=146)` twice. The cache generation preserved the source annotation verbatim. Ultralytics detected and removed the identical duplicate during every checkpoint evaluation.

No source file, locked manifest, converted label, checkpoint, or metric was changed after discovery. The same deterministic removal applies to all 440 model evaluations, so it is not a strategy-specific difference. This remains a disclosed locked-final dataset-quality limitation.

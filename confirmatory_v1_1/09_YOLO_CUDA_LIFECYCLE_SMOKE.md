# YOLO CUDA Lifecycle Smoke

## Tiny non-scientific fixture

**PASS.** A one-epoch GC10 acquisition/development fixture verified CUDA load,
forward/backward, validation, cold start, `resume=False`, `best.pt`, `last.pt`,
and 10-class metric extraction. Its outcome was not used or exposed.

## Exact 100-epoch benchmark lifecycle

All six fixed cells completed on their first attempt with the frozen checkpoint,
seed A, Random acquisition index 0, R0/R3/R5 manifests, batch 8, imgsz 640,
100 epochs and the exact v1 optimizer/augmentation/cold-start settings.

- PASS: 6/6
- OOM or CUDA incompatibility: 0
- retries: 0
- missing best/last: 0
- per-class extraction mismatch: 0
- final split access: 0
- human-readable performance values exposed to readiness: 0

Because hashes match and no outcome-guided decision was made, all six exact
100-epoch cells are classified `REUSABLE_SCIENTIFIC_CELL`. They count toward the
planned 440, leaving 434 new training jobs after explicit full-run authorization.


# DINO Cache and Reproducibility Audit

## Result

**PASS — BITWISE_REPRODUCIBLE on both datasets**

| Dataset | Acquisition images | Cache A | Cache B | Runtime A / B | Peak reserved VRAM | Classification |
|---|---:|---|---|---:|---:|---|
| GC10-DET | 1,605 | `cc1c9826…a29b8` | same | 45.43 / 43.43 s | 374 MiB | BITWISE_REPRODUCIBLE |
| NEU-DET | 1,259 | `1f6c92f6…9a326` | same | 8.91 / 8.90 s | 374 MiB | BITWISE_REPRODUCIBLE |

Both runs used the sealed DINOv2-small snapshot, exact `pooler_output` 384-D,
FP32, autocast off, TF32 off, batch 32, row-wise L2 normalization and the full
acquisition universe. Counts, ordered opaque IDs, dtype, dimension, finiteness,
nonzero norm, missing/extra/duplicate checks and file SHA seals passed.

All 20 acquisition-seed DINO trajectories from cache A and B were identical.
Cache A is canonical. Total A+B GPU runtime was 106.66 seconds and total cache
artifact size was 10.64 MB. No GT, development or final metric entered the cache
or selector.

One infrastructure-only first attempt failed before model/data processing because
the CUDA peak-memory counter needed explicit device initialization. It is retained
in `attempt_registry/dino_cache_attempts.csv`; scientific settings did not change.


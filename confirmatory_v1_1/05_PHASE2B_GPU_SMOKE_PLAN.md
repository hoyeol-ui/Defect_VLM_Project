# Phase 2B GPU Smoke Plan

## Authorized work

1. Run the original 14 CPU gates plus eight v1.1 final-source gates.
2. Build full FP32 DINO cache A and independent rerun B over the complete
   acquisition universe: GC10 1,605 and NEU 1,259 images.
3. Validate count, opaque ID order, dimension 384, float32, finite/nonzero rows,
   file hashes, host/GPU memory, runtime and bytes. Compare A/B exactly; if bytes
   differ compare numerically and require all 20 DINO trajectories to match.
4. Materialize 20 shared initial, 100 Random query, 100 DINO query and 220 unique
   cumulative manifest states without GT.
5. Run a non-scientific tiny CUDA lifecycle fixture.
6. If it passes, run at most six exact 100-epoch feasibility cells: each dataset,
   acquisition index 0, Random, training seed A, R0/R3/R5.

Scientific outcome values stay in local outcome-private artifacts. Readiness is
based only on runtime, peak VRAM/RAM, storage, lifecycle integrity and retries.
The six benchmark cells may later be reused only if all frozen hashes match and
no outcome-guided change occurred.

## Stop conditions

Hash/manifest mismatch, final-lock bypass, DINO coverage or trajectory
instability, CUDA incompatibility/OOM, missing best/last/per-class artifact, or
any required scientific setting change stops the phase. Full 440 and actual
final evaluation are not authorized here.


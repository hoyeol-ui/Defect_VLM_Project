# Protocol v1 -> v1.1 Changelog

- Phase: **PHASE 2B — PROTOCOL v1.1 REFREEZE + GPU FEASIBILITY SMOKE**
- Status: `FROZEN_CANDIDATE_GPU_SMOKE_ONLY`
- Parent protocol: `confirmatory_v1/confirmatory_protocol_v1.yaml`
- Parent SHA-256: `fa6ad82f5f6b282b1fde5529af41cb54eb5471d3f2950f00d32f491b8ed0c663`

## The only scientific change

`best.pt` is selected using the development split. The primary confirmatory
evaluation is the locked final split, performed once only after all 440 planned
training cells close, integrity checks pass, and an explicit final-evaluation
authorization is supplied. Primary AULC reads the six locked-final mAP50-95
values. Development AULC is supplementary/diagnostic only.

No split membership, budget, seed, Random/DINO selector, DINO snapshot or
precision, GT-free boundary, sample identity, YOLO lifecycle/hyperparameter,
subgroup, analysis, or claim field changed. Any such change requires
`PROTOCOL_REFREEZE_REQUIRED`.


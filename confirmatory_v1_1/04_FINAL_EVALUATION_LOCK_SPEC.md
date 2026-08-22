# Final Evaluation Lock Specification

The lock validates all of the following before exposing a real final registry:

- explicit CLI/API authorization flag;
- external record status `AUTHORIZED`;
- exact protocol YAML SHA and pre-GPU freeze commit SHA;
- exactly 440 unique scientific cells, each complete;
- each cell's protocol, code, model, manifest and environment seals;
- append-only attempt-registry SHA and all retries closed;
- training checkpoint and manifest-integrity audit PASS;
- final-lock audit PASS.

An in-memory fixture may test this logic. It is not an authorization artifact.
No `FINAL_EVALUATION_AUTHORIZATION.json` is created by Phase 2B.


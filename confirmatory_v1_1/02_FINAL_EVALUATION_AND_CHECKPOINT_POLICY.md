# Final Evaluation and Checkpoint Policy

1. Every training cell is cold-started from the sealed YOLOv8n checkpoint.
2. Development validation selects `best.pt`; `last.pt` remains an audit artifact.
3. Development metrics may be inspected only after the outcome-blinded Phase 2B
   feasibility decision. They never enter primary AULC.
4. Actual final access hard-fails until all 440 cell records are complete and
   match protocol, code, manifest, checkpoint, environment and attempt-registry
   seals, retries are closed, and an explicit external authorization record is
   present.
5. After authorization, each sealed `best.pt` is evaluated once on the locked
   final split. No retraining, retuning, seed replacement or post-final rescue is
   allowed.
6. Primary AULC requires exactly six locked-final mAP50-95 values. A missing point
   yields `INCOMPLETE`; interpolation and development substitution are forbidden.

Phase 2B authorization permits acquisition-cache construction and feasibility
training only. It does not authorize actual final access or the remaining full
440 run.


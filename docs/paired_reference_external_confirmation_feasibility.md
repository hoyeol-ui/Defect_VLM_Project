# Paired-reference external confirmation feasibility

- Development result: **NO_CANDIDATE**
- Frozen candidate: **None**
- Detector screen currently allowed: **NO**
- External confirmation currently authorized: **NO**

## Required independent data

1. Defective query and defect-free reference must be paired by the same product layout/revision.
2. The reference must not contain the query defect and must be usable without GT-driven reference selection.
3. Bboxes or pixel masks must exist, with enough bbox-area <=1024 px^2 instances for group-resampled inference.
4. Production lot/time/source groups must support independent group-level resampling.
5. No DeepPCB image, derivative, or board group may overlap.
6. Candidate code, operating threshold, 20% query budget, endpoints, and STOP rules must be frozen before labels are opened.
7. Selection must be hashed before annotations are joined.

## Prohibited substitution

DeepPCB group92000 is development data and the six eligible groups are post-hoc audit data; neither can serve as independent confirmation. The official DeepPCB test remains locked unless the supervisor explicitly retires it from final-test use under a separate one-shot protocol.

## Current decision

No development candidate passed the frozen adequacy criteria. Do not start an external confirmation or detector screen from this branch.

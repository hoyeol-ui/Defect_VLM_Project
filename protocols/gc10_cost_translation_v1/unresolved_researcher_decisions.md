# Unresolved Researcher Decisions

Protocol version: `gc10-cost-translation-v1-candidate`

Package status: not authorized for human annotation until every item below is signed.

Stage 0 may estimate instrumentation error, repeatability, log completeness, clock reconciliation, and idle behavior only. It may not compare DINO and Random. Phase A arm results may not be opened before the thresholds below are frozen.

| ID | Required decision | Selected value | Decision owner | Evidence/reference | Decision date UTC | Signature/status |
|---|---|---|---|---|---|---|
| D01 | Annotator count: 2 or 4 | UNRESOLVED | Researcher + supervisor | Workload and claim scope |  | OPEN |
| D02 | Primary time budget \(T\) source and rule | UNRESOLVED | Researcher + supervisor/industry stakeholder | External operating session preferred |  | OPEN |
| D03 | Idle threshold | UNRESOLVED | Researcher + supervisor | Stage 0 instrumentation-only evidence |  | OPEN |
| D04 | Phase A burden materiality margin | UNRESOLVED | Researcher + supervisor/industry stakeholder | Measurement resolution + operational meaning |  | OPEN |
| D05 | Phase A quality non-inferiority margin | UNRESOLVED | Researcher + supervisor | Missed-defect risk tolerance |  | OPEN |
| D06 | Phase B minimum mAP50–95 effect | UNRESOLVED | Researcher + supervisor | Existing +0.010 gate or external requirement |  | OPEN |
| D07 | Rare AP/recall safety margins and class 8/9/10 guards | UNRESOLVED | Researcher + supervisor | Safety policy; known result cannot justify weakening |  | OPEN |
| D08 | Annotator disagreement and adjudication rule | UNRESOLVED | Researcher + supervisor | Label provenance and review cost |  | OPEN |
| D09 | Human-participation approval, consent, compensation, retention | UNRESOLVED | Institution + researcher | Formal institutional determination |  | OPEN |

## Freeze rule

All nine rows must have a value, named decision owner, dated approval, and corresponding config update. The finalized package must then be committed and tagged before Phase A starts. A researcher may decide not to proceed; that is a valid decision and must remain in the ledger.

## Prohibited sequence

```text
Phase A arm result
  -> choose T, idle cutoff, burden margin, quality margin, or safety margin
```

This sequence is prohibited because it turns the study into outcome-dependent threshold selection. If any Phase A arm result is inspected before D01–D09 are frozen, the study must be relabeled exploratory.

# RAW Confirmatory Results Master v1.1

This is the Phase 2B operational result canon. It is separate from the historical
`RAW_EXPERIMENT_RESULTS_MASTER_20260822.md`. Scientific outcome values generated
incidentally by development validation remain in `outcomes_private/` and were not
read for feasibility or readiness.

## Protocol and execution seals

- active freeze commit: `5d6708c01edcb064395326648ce4e333effb98be`
- protocol YAML SHA-256: `6836f452d22c3de26dfa2680c45d0b5cb8f925b1a9274789c4ef81a1770f0b10`
- parent v1 YAML SHA-256: `fa6ad82f5f6b282b1fde5529af41cb54eb5471d3f2950f00d32f491b8ed0c663`
- attempt registry SHA-256: `09a8562790b55c7357c39ae251bb9ae9be40a79c3d5df4243ed52b5f46e5dbb4`
- trajectory ledger SHA-256: `bbdd09a645cc84849cbe88662abf61ec1e7e2937f382c071840dfc432a6824ea`

## Raw operational rows

The exact seven lifecycle rows are preserved in
`benchmark_registry/OPERATIONAL_BENCHMARK_MASTER.csv`. Six 100-epoch scientific
cells are PASS/reusable and one tiny lifecycle fixture is PASS/non-scientific.
All exact runtimes, peak memory, bytes, retry count and lifecycle flags are kept
there without performance metrics.

## Cache and manifest raw status

| Artifact | Raw status |
|---|---|
| GC10 cache A/B | COMPLETE, exact same content SHA `cc1c9826…a29b8` |
| NEU cache A/B | COMPLETE, exact same content SHA `1f6c92f6…9a326` |
| DINO trajectory A/B pairs | 20/20 identical |
| shared initial / Random query / DINO query | 20 / 100 / 100 |
| unique cumulative states | 220 |

## Confirmatory outcome registry state

| Field | Current raw value |
|---|---|
| planned detector training cells | 440 |
| completed reusable detector cells | 6 |
| remaining detector cells | 434 |
| development outcome values | `SEALED_OUTCOME_PRIVATE_NOT_READ` |
| primary locked-final mAP50-95 rows | `NULL_LOCKED_NOT_RUN` |
| primary AULC rows | `NULL_LOCKED_NOT_RUN` |
| actual final evaluations | 0 |
| failed scientific training attempts | 0 |
| infrastructure-only DINO pre-model attempt | 1 retained |

No null, negative, failure or future outcome row will be deleted or replaced.
Primary rows stay null until authorized final evaluation supplies all six points;
development values can never fill them.


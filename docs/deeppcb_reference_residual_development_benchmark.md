# DeepPCB reference-residual development benchmark

## Scope and locks

- Parent prospective decision: **FAIL_STOP (unchanged)**
- Phase A mechanism result: **A2_MECHANISM_AMBIGUOUS**
- Data: **group92000 trainval rows only (111 images)**
- Status: development-only, not confirmatory evidence
- Implementation amendment: component confidence uses the cited implementation's fixed 0.95 within-component quantile; the earlier max-score run is preserved as invalid due to saturation
- Pre-execution policy SHA-256: `237e23a69c4e1991d75752dfb50962bfd8959e6316cf81a2aa690fd1265c24cf`
- Detector training/inference: **0 / 0**
- Official/final test use: **0 / 0**

## Method comparison at development operating points

| Method | Threshold | Recall | Small recall | FP/image | Hit classes | Adequate |
|---|---:|---:|---:|---:|---:|---|
| M1_ABSDIFF | inf | 0.0000 | 0.0000 | 0.0000 | 0 | False |
| M2_SSIM | 0.50000 | 0.1803 | 0.1786 | 0.9730 | 5 | False |
| M3_FUSED | inf | 0.0000 | 0.0000 | 0.0000 | 0 | False |

## Development decision

- Result: **NO_CANDIDATE**
- Candidate: **None**

This candidate, if present, is only a frozen object for an independent prospective selection-only study. It does not rescue the original FAIL_STOP branch, validate spatial mechanism, authorize detector training, or establish external utility.

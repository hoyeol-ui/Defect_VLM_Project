# Full 440 Feasibility Estimate

## Detector runtime

Dataset-specific planned runtime is calculated as:

```text
20*T0 + 40*(T1 + T2 + T3 + T4 + T5)
```

This preserves 10 acquisition seeds × two training seeds, shares R0 within each
acquisition/training pair, and retains two post-R0 strategies.

| Component | Point estimate | Conservative adjacent-anchor range |
|---|---:|---:|
| GC10 220 training cells | 38.78 GPU-h | 34.70–42.85 GPU-h |
| NEU 220 training cells | 31.69 GPU-h | 28.47–34.92 GPU-h |
| total 440 training cells | 70.47 GPU-h | 63.17–77.77 GPU-h |
| separate best.pt development evaluation | 1.16 GPU-h | measured-rate extrapolation |
| authorized full training + development lifecycle | **71.62 GPU-h** | **64.32–78.93 GPU-h** |

On the current single RTX 3070, ideal continuous sequential wall time is about
2.98 days. A practical uninterrupted reservation with 15–30% operational margin
is about 3.4–3.9 days. Six already completed cells are reusable, so remaining
compute is slightly below the full point estimate; the table intentionally reports
the complete 440-study footprint.

Two/four equivalent independent GPUs would have theoretical lower bounds of
35.8/17.9 hours before scheduling and I/O overhead. This machine has one GPU;
parallel figures are planning scenarios, not measurements. A larger GPU was not
benchmarked, so no unsupported speedup is claimed.

## Storage and final inference

- detector artifact point estimate: 5.50 GB
- extrapolated console logs: about 0.23 GB
- DINO A+B caches: 0.011 GB
- recommended local reserve including registries/final metrics/margin: **10 GB**
- current free disk: 105.48 GB

Actual final has not been accessed. Scaling the measured development inference
time only by locked-final image count gives a later, authorization-dependent
estimate of 1.17 GPU-h for GC10 plus 1.14 GPU-h for NEU, about **2.31 GPU-h**.
This is inference only and is not included in the current 440 training count.

## Lifecycle counts

- planned training cells: 440
- completed/reusable now: 6
- new training cells after authorization: 434
- development evaluations completed now: 6; planned total: 440
- actual final evaluations completed now: 0; planned after lock release: 440


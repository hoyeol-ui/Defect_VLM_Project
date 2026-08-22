# Runtime, VRAM and Storage Benchmark

All times below are operational only. Scientific performance values remain
outcome-private.

| Dataset | Point | Images | Runtime status | Train time | Separate best.pt dev eval | Peak GPU used | Artifact bytes |
|---|---|---:|---|---:|---:|---:|---:|
| GC10 | T0 | 161 | measured | 383.29 s | 9.70 s | 2,139 MiB | 12,482,101 |
| GC10 | T1 | 241 | scaled estimate, range T0–T3 | 476.76 s | — | — | — |
| GC10 | T2 | 321 | scaled estimate, range T0–T3 | 570.23 s | — | — | — |
| GC10 | T3 | 401 | measured | 663.70 s | 9.55 s | 2,140 MiB | 12,482,103 |
| GC10 | T4 | 481 | scaled estimate, range T3–T5 | 750.37 s | — | — | — |
| GC10 | T5 | 561 | measured | 837.05 s | 9.51 s | 2,144 MiB | 12,482,121 |
| NEU | T0 | 126 | measured | 321.25 s | 9.51 s | 2,452 MiB | 12,531,531 |
| NEU | T1 | 189 | scaled estimate, range T0–T3 | 392.96 s | — | — | — |
| NEU | T2 | 252 | scaled estimate, range T0–T3 | 464.67 s | — | — | — |
| NEU | T3 | 315 | measured | 536.39 s | 9.22 s | 2,325 MiB | 12,531,515 |
| NEU | T4 | 378 | scaled estimate, range T3–T5 | 611.35 s | — | — | — |
| NEU | T5 | 441 | measured | 686.32 s | 9.26 s | 2,334 MiB | 12,531,536 |

T1/T2/T4 are labeled-count linear interpolations between adjacent measured
anchors; they are not represented as exact observations. A conservative bound
uses the adjacent measured endpoints themselves.

Observed maximums were 2,452 MiB GPU memory by `nvidia-smi`, 1.29 GiB PyTorch
reserved memory, and 14.63 GB summed process-tree RSS. The RSS measure is a
conservative process-tree sum and may double-count shared pages, but is retained
as measured. Current C: free space after the smoke is 105.48 GB.


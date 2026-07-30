# GC10 D2R representation branch closure

Decision date: 2026-07-15

## Decision

Close the `frozen DINOv2-small + label-aware representation repair` branch for GC10. Do not preregister D2R-v2 and do not run the conditional detector confirmation.

## Evidence

The frozen D2R selection-only gate failed despite strong discovery performance versus Random. Three repair checks failed: top-10 concentration ratio 0.800625, rare within-class distance difference -0.079473, and representation target-hit rate 0.299.

The first post-hoc replay reconstructed all 1,000 representation micro-rounds and 32,000 centroid-neighborhood candidates. Replacing max-novelty with centroid-nearest increased overall target hit from 0.299 to 0.494, but class 8 remained at 0.116809 with only 0.049947 target purity in the top 32.

The final label-aware retrieval diagnostic evaluated the complete remaining pool at each frozen state:

| method | overall top-1 hit | class-8 top-1 hit | class-8 P@5 | class-8 target in top-32 | class-8 unique ratio |
|---|---:|---:|---:|---:|---:|
| centroid similarity | 0.494 | 0.116809 | 0.060969 | 0.413105 | 0.116809 |
| nearest target exemplar | 0.526 | 0.153846 | 0.070655 | 0.427350 | 0.074074 |
| contrastive margin | 0.487 | 0.096866 | 0.110541 | 0.612536 | 0.176638 |
| top-3 target similarity | 0.495 | 0.116809 | 0.060969 | 0.413105 | 0.116809 |
| top-5 target similarity | 0.494 | 0.116809 | 0.060969 | 0.413105 | 0.116809 |

No method passed any complete class-8 recovery gate. Nearest-exemplar retrieval selected only 26 unique images over 351 class-8 target rounds and selected class 6 on 204 class-8 rounds. Contrastive margin increased class-8 top-32 availability but reduced top-1 hit below the centroid baseline.

## Interpretation

Frozen DINO provides useful global discovery signals for several GC10 defect classes, but its local geometry is not aligned reliably enough with the full defect taxonomy. The representation loop repeatedly targets class 8 because misses do not increase its labeled count, creating a self-reinforcing acquisition failure. This cannot be repaired by centroid, exemplar, contrastive-margin, or top-k positive retrieval on the available selected-label feedback.

## Consequences

- Conditional YOLOv8n D2R training remains prohibited.
- Do not weaken the frozen gate or edit the selection config.
- Do not claim that DINO is generally useless: the GC10 discovery-only gains remain valid development evidence.
- Do not claim a D2R-v2 result from the off-policy diagnostics; hypothetical choices were not propagated into later states.
- The avoided computation is 15 additional development detector trainings plus any follow-on final-test consumption.

## Research implication

This closure supports the validity-gated workflow contribution: a selector can show strong taxonomy/rare-class discovery metrics while its representation mechanism is structurally invalid for detector learning. Pre-training gates identify that failure before detector cost and final-test exposure.

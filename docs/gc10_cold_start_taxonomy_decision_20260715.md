# GC10-DET Cold-Start Taxonomy Decision

Date: 2026-07-15

## Final decision for this branch

The GC10 selection-only gate **passed**, but the pre-registered development
detector confirmation gate **failed**. The final test remains locked and must
not be evaluated. No further GC10 selector tuning or detector escalation is
authorized from this branch.

## Selection-only result

Across 200 paired acquisitions with shared initial 20 and query 20:

- Initial-plus-query unique classes: Random 9.070, DINO 9.895 (+0.825; paired
  bootstrap CI [0.715, 0.940]).
- Rare-class images in query: 2.170 versus 4.890 (+2.720; CI [2.455, 2.980]).
- Unique rare classes: 1.450 versus 2.660 (+1.210).
- Instances: 31.640 versus 33.735 (+2.095).
- DINO query redundancy was much lower and distance from initial much higher.

All 11 pre-registered selection checks passed. This supports DINO farthest-first
as a taxonomy/prototype discovery tool in an all-defect cold-start pool.

## Development detector result

Thirty `yolov8n` models were trained: five acquisition seeds, two strategies,
and three training seeds, using 40 labeled images per model. Fixed 75-epoch
`last.pt` checkpoints were evaluated on the 232-image development split.

Positive aggregate signals:

- mAP50-95: 0.124235 Random versus 0.141614 DINO, difference +0.017378.
- Acquisition-seed bootstrap CI: [+0.000580, +0.036332].
- DINO won 4/5 acquisition seeds.
- Recall: 0.316877 versus 0.352259, difference +0.035381.
- Frequent-class macro AP50-95: +0.033345.

Binding failures:

- Rare-class macro AP50-95: 0.036507 versus 0.016631, difference -0.019877.
- Class 9 AP50-95: -0.031907.
- Class 10 AP50-95: -0.028495.
- The pre-registered rare macro and class-9 checks failed; therefore the overall
  development confirmation is FAIL despite the positive aggregate mAP.

## Why discovery did not translate

DINO did not fail to find rare labels. It failed to supply representative
within-class coverage.

- Across the full 200-seed audit, 4,000 DINO query slots used only 143 unique
  images. The most frequent image appeared in 194/200 seeds, and the top ten
  images occupied 40% of all slots.
- For detector acquisition seeds 0--4, mean DINO query overlap between seed
  pairs was 10.0/20 versus 0.4/20 for Random.
- DINO put 1.8 class-9 images in each query on average, but only three unique
  class-9 query images appeared across all five acquisition seeds.
- DINO class-9 boxes were substantially larger/more extreme than Random class-9
  boxes, while class-9 AP decreased.
- DINO selected fewer class-10 images than Random (0.6 versus 1.6 per query) and
  only one unique class-10 query image across the five DINO sets.

Farthest-first found stable global visual prototypes. Repeating those prototypes
improved broad shape/surface coverage, recall, and several frequent classes, but
did not create the within-class sample support needed for rare detectors.

## Research claim supported by current evidence

The defensible claim is narrower than “DINO beats Random for active learning”:

> In an all-defect cold-start pool, frozen DINO farthest-first can accelerate
> defect-taxonomy/prototype discovery and can improve aggregate development
> recall/mAP, but prototype discovery is not equivalent to representative
> rare-class learning. A second, representation-oriented annotation phase is
> required before detector benefit can be claimed for rare classes.

## Recommended direction

Treat cold-start annotation as a two-stage workflow rather than another
one-shot selector contest:

1. **Discovery phase:** frozen DINO farthest-first surfaces a small set of
   distinct visual prototypes and reveals the initial taxonomy.
2. **Representation phase:** after those annotations are available, retrieve or
   cluster neighbors around discovered prototypes and choose representative
   medoids/random neighbors rather than additional global extremes.

This second phase is a new hypothesis. It must be pre-registered and tested on
an independent all-defect dataset or a new untouched protocol, not tuned and
confirmed on the same GC10 development result. More GC10 seeds or final-test use
would not resolve the prototype-dependence problem.


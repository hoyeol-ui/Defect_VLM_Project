# MPDD Selection-Only Decision

Date: 2026-07-15

## Decision

The MPDD detector-training gate is **FAIL**. No detector training is authorized
from this result, and the final split remains unused.

The failure is narrow but binding: hierarchical DINO query coverage averaged
5.475 product categories versus 5.800 for Random, a difference of -0.325. The
pre-registered non-inferiority margin was -0.25. The other eight gate checks
passed.

## What was positive

Across 200 paired cold-start acquisitions (initial 20, query 20):

- Random found 4.060 anomaly images per query.
- Category-Balanced Random found 4.805 (+0.745 versus Random).
- Frozen Category-Balanced DINO found 10.305 (+6.245 versus Random; paired 95%
  bootstrap CI [5.885, 6.605]; 197 wins, 0 losses, 3 ties).
- DINO also beat Category-Balanced Random by +5.500 anomalies per query (CI
  [5.195, 5.805]; 200/200 strict wins), so the effect was not explained by the
  category quota alone.
- The leave-one-product-category-out gain versus Random remained positive for
  all six categories.

This is strong evidence that frozen visual distance changes which MPDD images
are surfaced during cold-start triage. It is not yet clean evidence that the
distance is defect-specific.

## Source-origin confound

MPDD's official train partition contains only normal images; every anomaly is
in the official test partition. Although official origin and source paths were
hidden from selectors, image appearance can still encode capture-session or
partition differences.

Hierarchical DINO selected 13.390 official-test images per query versus 6.580
for Random. A descriptive mean-level decomposition attributes:

- 4.202 of the +6.245 anomaly gain (67.3%) to selecting more official-test
  images;
- 2.043 (32.7%) to a higher anomaly rate within selected official-test images.

The within-test enrichment is encouraging, but the two effects cannot be
causally separated in MPDD because official train contains no anomalies.

## Research implication

The current evidence supports a narrower claim worth continuing to test:

> Frozen DINO distance can be useful for cold-start annotation triage when the
> pool has heterogeneous visual regimes, but apparent anomaly yield must be
> separated from batch/domain/source-origin discovery, and coverage must be
> protected by the operational unit that actually matters.

It does not support detector training on MPDD under the current protocol, nor a
claim that hierarchical DINO improves detector mAP.

## Recommended next condition

Use GC10-DET as a separate **all-defect taxonomy-discovery** condition, not as an
anomaly-detection benchmark. Hide folder names and source paths because folders
1--10 are defect labels. Compare only frozen Random and frozen DINO
farthest-first on shared initial/query sets, and audit rare-class discovery,
unique class coverage, bbox/instance richness, visual redundancy, and overlap.

Before selection, reconcile the observed 2,312 JPG files with 2,294 XML files
and lock a duplicate-safe acquisition/development/final split. The experiment
must remain selection-only until a new gate is written and passed.


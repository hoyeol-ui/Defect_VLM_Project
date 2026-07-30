# Final decision

**B. CONDITION_MAP_DESCRIPTIVE_ONLY**

Same-seed discovery gains are supported by the frozen records, but the 200
seeds do not constitute independent prevalence conditions. Each pool is a
high-overlap leave-20 perturbation of one fixed acquisition pool. The audit
supports paired selector effects and discovery-safety diagnostics, not a
general pool-sparsity law. Category/source dominance also prevents decision A.

# Design identifiability

| dataset | design_type | seeds | unique_pool_hashes | paired | prevalence_range | category_entropy_range | source_variation | matching_identifiable | reason |
|---|---|---:|---:|---|---|---|---|---|---|
| GC10-DET | TYPE A: paired_variable_pool (leave-20 finite-pool perturbation) | 200 | 200 | yes | 0.10572687 to 0.10903084 | 2.072682 to 2.078980 | entropy 5.792409 to 5.815139 | paired selector effect: yes; prevalence matching: no | 200 unique candidate hashes arise only because seed-specific initial20 is removed from one fixed acquisition pool; same-seed strategies are paired, but condition range is near-fixed. Mean pairwise pool Jaccard=0.978446. |
| MPDD | TYPE A: paired_variable_pool (leave-20 finite-pool perturbation) | 200 | 200 | yes | 0.19787645 to 0.20656371 | 1.733891 to 1.743225 | entropy 0.630792 to 0.639591 | paired selector effect: yes; prevalence matching: no | 200 unique candidate hashes arise only because seed-specific initial20 is removed from one fixed acquisition pool; same-seed strategies are paired, but condition range is near-fixed. Mean pairwise pool Jaccard=0.962827. |
| VisA | TYPE A: paired_variable_pool (leave-20 finite-pool perturbation) | 200 | 200 | yes | 0.11042874 to 0.11123986 | 2.447660 to 2.448413 | not separately available | paired selector effect: yes; prevalence matching: no | 200 unique candidate hashes arise only because seed-specific initial20 is removed from one fixed acquisition pool; same-seed strategies are paired, but condition range is near-fixed. Mean pairwise pool Jaccard=0.995386. |

- Pairing: exact same reconstructed candidate pool within each seed.
- Matching: not performed; no TYPE B dataset exists.
- Common support: 100% for same-seed strategy comparisons, but practical
  prevalence-condition support is absent.
- Not estimable: prevalence tertile effects, prevalence-gain correlation, and
  cross-dataset causal condition effects.
- The reported confidence intervals quantify acquisition-seed variation, not
  sampling from independent production pools.

# Dataset results

## VisA

- Paired anomaly discovery: gain 14.480000, 95% CI [14.195000, 14.770000], positive seeds 100.0%, coverage delta -4.110000, HHI delta +0.286025.
- Quadrants: {'Q2_gain_positive_safety_loss': 200}.
- Leave-one-category-out gain falls from 14.480 to 3.995;
  the implied largest-category contribution is 72.4%.
- Prevalence-stratum effect: **not identifiable**.
- Separate source robustness is unavailable because source is not distinct from product category.

## MPDD

- Paired anomaly discovery: gain 6.245000, 95% CI [5.885000, 6.605000], positive seeds 98.5%, coverage delta -0.325000, HHI delta -0.008000.
- Quadrants: {'Q1_gain_positive_safety_nonnegative': 113, 'Q2_gain_positive_safety_loss': 84, 'Q3_gain_nonpositive_safety_nonnegative': 2, 'Q4_gain_nonpositive_safety_loss': 1}.
- Leave-one-product-category gain remains positive (minimum 3.555),
  while official-test-origin composition explains 67.3%
  of the observed anomaly gain.
- Source origin and anomaly status remain separate outcomes; neither was used for matching.
- Prevalence-stratum effect: **not identifiable**.

## GC10-DET

- Paired rare discovery: gain 2.720000, 95% CI [2.455000, 2.980000], positive seeds 88.0%, coverage delta +0.825000, HHI delta -0.019188.
- Quadrants: {'Q1_gain_positive_safety_nonnegative': 172, 'Q3_gain_nonpositive_safety_nonnegative': 22, 'Q2_gain_positive_safety_loss': 4, 'Q4_gain_nonpositive_safety_loss': 2}.
- Existing q20 detector translation: mAP50-95 +0.017378 but rare macro AP -0.019877.
- Existing K40 result: selection coverage PASS but mAP50-95 -0.001678,
  rare AP -0.018290, and recall -0.021871.
- Seed45 fixed-set +0.016236 did not generalize in the independent acquisition
  confirmation (+0.007019, CI crosses zero, p=0.322266).
- No regression was fitted across the few non-exchangeable aggregate branches.

# Claims retained from the integrated condition map

1. Same-pool discovery gains exist relative to Random in each dataset.
2. Discovery gain and category/source safety can diverge.
3. GC10 selection/coverage gain does not automatically translate into rare AP,
   recall, or overall detector utility.
4. Training-seed stability does not establish acquisition-set generalization.
5. A validity gate prevented additional training and final-test consumption.

# Claims weakened after this audit

- Pool prevalence or sparsity as a moderator: **not identifiable**.
- VisA category-agnostic robustness: weakened by pipe_fryum dominance.
- MPDD representation-only mechanism: weakened by the 67.3% source-composition contribution.
- Cross-dataset condition law: descriptive association only; dataset identity
  cannot be exchanged with prevalence.

# Pool sparsity decision

**Not identifiable.** Between-dataset sparsity differences do not identify a
prevalence effect, and within each dataset the candidate pools are nearly fixed.

# Discovery-coverage trade-off decision

**Dataset-specific descriptive support.** The joint outcome repeats, but its
size and mechanism are dataset-dependent and cannot be promoted to a universal law.

# Selection-learning translation decision

**Repeatedly separated in GC10.** Positive selection-stage results did not
reliably translate to rare AP, recall, or mAP. The available aggregate branches
are not independent units for a learner-alignment coefficient.

# Effect on thesis viability

The central question survives only in a narrower, conditional form: an
industrial validity-gated workflow that separates candidate-signal validity,
discovery-safety behavior, and selection-to-learning translation. It does not
support a general sparsity law or a broadly superior selector.

# FN extension

**DESIGN_PROTOCOL_ONLY retained.** This audit adds neither FN event-count
evidence nor an untouched validation resource. It does not authorize an FN
screen, feature implementation, training, inference, or final-test access.

# Supervisor briefing

The original 200-seed records support same-candidate-pool discovery gains over
Random. However, the seeds differ only by removing 20 items from one fixed pool,
so they cannot identify whether target prevalence moderates those gains. VisA
and MPDD also show category/source concentration, while GC10 repeatedly shows
that selection-stage gains need not translate into rare-class detector utility.
The defensible thesis contribution is therefore a validity-gated audit workflow
and failure-condition map, with pool-sparsity claims explicitly withheld.

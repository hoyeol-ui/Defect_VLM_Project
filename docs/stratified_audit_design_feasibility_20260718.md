# Stratified audit design feasibility

## Design classification

| dataset | design_type | seeds | unique_pool_hashes | paired | prevalence_range | category_entropy_range | source_variation | matching_identifiable | reason |
|---|---|---:|---:|---|---|---|---|---|---|
| GC10-DET | TYPE A: paired_variable_pool (leave-20 finite-pool perturbation) | 200 | 200 | yes | 0.10572687 to 0.10903084 | 2.072682 to 2.078980 | entropy 5.792409 to 5.815139 | paired selector effect: yes; prevalence matching: no | 200 unique candidate hashes arise only because seed-specific initial20 is removed from one fixed acquisition pool; same-seed strategies are paired, but condition range is near-fixed. Mean pairwise pool Jaccard=0.978446. |
| MPDD | TYPE A: paired_variable_pool (leave-20 finite-pool perturbation) | 200 | 200 | yes | 0.19787645 to 0.20656371 | 1.733891 to 1.743225 | entropy 0.630792 to 0.639591 | paired selector effect: yes; prevalence matching: no | 200 unique candidate hashes arise only because seed-specific initial20 is removed from one fixed acquisition pool; same-seed strategies are paired, but condition range is near-fixed. Mean pairwise pool Jaccard=0.962827. |
| VisA | TYPE A: paired_variable_pool (leave-20 finite-pool perturbation) | 200 | 200 | yes | 0.11042874 to 0.11123986 | 2.447660 to 2.448413 | not separately available | paired selector effect: yes; prevalence matching: no | 200 unique candidate hashes arise only because seed-specific initial20 is removed from one fixed acquisition pool; same-seed strategies are paired, but condition range is near-fixed. Mean pairwise pool Jaccard=0.995386. |

For all three datasets, the frozen protocol draws the initial 20 with
`random_state=seed + 999` and removes those samples from one fixed acquisition
pool. Random and selector strategies share that reconstructed candidate pool
within a seed. This is formally TYPE A for the **same-seed selector effect**,
but the 200 candidate pools are highly overlapping leave-20 perturbations, not
200 independently sampled industrial pools.

## Prevalence effect gate

The frozen minimum is a target-count range of at least
20, a relative prevalence range of at least
0.10, and at least 20
independent pool realizations per tertile. No dataset passes. Low/medium/high
prevalence effects, prevalence-gain correlations, and prevalence
matching/regression are not estimated. Bootstrap intervals below describe
acquisition-seed variability, not a population of independent production pools.

## Allowed analyses

- Same-seed, same-candidate-pool selector-minus-Random target-yield difference.
- Paired category/source coverage and HHI differences.
- Selected-set concentration and discovery-safety quadrants.
- Seed-level variability plus category/source dominance diagnostics.

## Prohibited interpretations

- Claiming that lower target prevalence increases selector gain.
- Treating between-dataset differences as a prevalence moderation effect.
- Treating leave-20 perturbations as independently sampled production pools.
- Using selected yield or coverage as a matching covariate.

There is no TYPE B dataset, so `matching_balance.csv` is intentionally not created.

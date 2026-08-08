# Frozen DeepPCB small-defect mechanism audit

## Immutable parent decision

- Original decision: **FAIL_STOP (unchanged)**
- Frozen selection SHA-256: `15b45b559b887b9abd53322b2919931330a0718bb031dc4dd1c98944a9450e2c`
- Detector training/inference: **0 / 0**
- Official/final test use: **0 / 0**
- Small definition: bbox area <= 1024 px^2

## Mechanism decision

- Decision: **A2_MECHANISM_AMBIGUOUS**
- Small-share uplift: 1.1828 (group bootstrap 95% CI 1.0814, 1.3711)
- Groups with size-selectivity > 1: 5/6
- Groups with small-bbox inside/outside ratio > 1: 6/6
- Leave-one-group-out directions retained: 6/6
- Max group share of positive small-box excess: 0.4579
- Max class share of positive small-box excess: 0.3576
- Nuisance-confound dominated: False

## Frozen criteria

- PASS — `small_share_ci_low_gt_1`
- PASS — `size_selectivity_gt_1_in_ge_5_groups`
- PASS — `small_inside_outside_gt_1_in_ge_5_groups`
- PASS — `leave_one_group_out_small_share_gt_1_all`
- FAIL — `single_group_positive_excess_share_lt_0_40`
- PASS — `single_class_positive_excess_share_lt_0_50`
- PASS — `not_nuisance_confound_dominated`

## Group effects

| Group | Total enrich. | Small enrich. | Size selectivity | Small excess |
|---|---:|---:|---:|---:|
| group00041 | 1.0554 | 1.1875 | 1.1252 | 15.00 |
| group13000 | 1.1998 | 1.6959 | 1.4135 | 65.66 |
| group20085 | 1.0906 | 1.2175 | 1.1164 | 52.53 |
| group44000 | 1.0450 | 1.3333 | 1.2759 | 3.00 |
| group50600 | 1.0284 | 1.2857 | 1.2502 | 6.00 |
| group77000 | 1.2286 | 1.1224 | 0.9136 | 1.20 |

## Size-bin localization of the effect

| Bbox area bin | Pool boxes | Selected boxes | Expected random | Enrichment |
|---|---:|---:|---:|---:|
| <=256 | 0 | 0 | 0.00 | NA |
| 257-576 | 35 | 0 | 7.09 | 0.0000 |
| 577-1024 | 2222 | 599 | 448.52 | 1.3355 |
| 1025-4096 | 3939 | 788 | 792.29 | 0.9946 |
| >4096 | 67 | 15 | 13.45 | 1.1152 |

## Spatial grounding

| Group | Size | Boxes | In/out ratio | Hit@0 | Hit@3 | Hit@5 |
|---|---|---:|---:|---:|---:|---:|
| group00041 | non_small | 1135 | 10.8271 | 1.0000 | 1.0000 | 1.0000 |
| group00041 | small | 400 | 5.7829 | 1.0000 | 1.0000 | 1.0000 |
| group13000 | non_small | 878 | 11.1126 | 1.0000 | 1.0000 | 1.0000 |
| group13000 | small | 467 | 4.1863 | 1.0000 | 1.0000 | 1.0000 |
| group20085 | non_small | 1007 | 5.0030 | 1.0000 | 1.0000 | 1.0000 |
| group20085 | small | 1191 | 3.8066 | 1.0000 | 1.0000 | 1.0000 |
| group44000 | non_small | 333 | 16.1781 | 1.0000 | 1.0000 | 1.0000 |
| group44000 | small | 45 | 15.0404 | 1.0000 | 1.0000 | 1.0000 |
| group50600 | non_small | 177 | 5.2693 | 1.0000 | 1.0000 | 1.0000 |
| group50600 | small | 105 | 5.0627 | 1.0000 | 1.0000 | 1.0000 |
| group77000 | non_small | 476 | 11.0633 | 1.0000 | 1.0000 | 1.0000 |
| group77000 | small | 49 | 10.6651 | 1.0000 | 1.0000 | 1.0000 |

## Interpretation boundary

This is a post-hoc read-only mechanism audit of a prospectively frozen FAIL_STOP branch. It cannot convert the parent selection gate into a PASS and is not evidence of detector learning utility, external replication, annotation-time reduction, or industrial generalization.

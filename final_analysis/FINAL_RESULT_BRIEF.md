# Final Result Brief

## Question

Under a fixed industrial-defect active-learning protocol, does a GT-blind frozen DINOv2 farthest-first selector improve downstream YOLOv8n learning relative to uniform Random selection, and is any overall gain compatible with subgroup safety?

## Design and integrity

- GC10-DET and NEU-DET, analyzed separately.
- 10 acquisition seeds per dataset; two learner seeds nested inside each selected set.
- Six cumulative budgets (about 10–35% of the acquisition universe).
- 440 cold-start, 100-epoch detector jobs; 440 locked-final evaluations.
- Primary: normalized six-budget locked-final mAP50-95 AULC.
- Acquisition-level paired bootstrap (100,000 replicates), exact sign-flip, Holm correction across the two dataset-level primary tests.
- Integrity: `PASS`; 440/440 complete, zero missing evaluations, zero development substitutions, zero post-final retraining/retuning.

## Primary result

| Dataset | Random | Frozen DINO | DINO − Random | 95% CI | seed direction | decision |
|---|---:|---:|---:|---:|---:|---|
| GC10-DET | 0.244643 | 0.250454 | +0.005812 | [+0.003970, +0.007576] | 10 positive / 0 negative | positive |
| NEU-DET | 0.351950 | 0.357731 | +0.005782 | [+0.003100, +0.008119] | 9 positive / 1 negative | positive |

Both Holm-adjusted p-values are `0.003906`. The effect is small—about `+0.58 percentage points` in budget-averaged mAP50-95—and strongest around early/middle budgets rather than growing monotonically to the final budget.

## Safety-relevant secondary result

GC10 rare-class macro AULC improved by `+0.673 pp` (95% CI `+0.302` to `+1.068`, 9/10). NEU worst-class AULC deteriorated by `−1.488 pp` (95% CI `−1.978` to `−1.002`, 0/10), and final worst-class AP deteriorated by `−3.118 pp`. The NEU selection composition shifted away from crazing and rolled-in-scale and toward scratches and pitted-surface images, matching the direction of the class-level trade-off.

## Scientific verdict

**Bounded positive primary result with a material subgroup safety failure.** Frozen DINO produced a modest, acquisition-reproducible overall learning-curve gain in both controlled datasets. It did not improve every class and cannot be called universally superior or safe. The stronger thesis contribution is the empirical demonstration that candidate validity, discovery, selected-set composition, reproducibility, and downstream learning are distinct evidence layers and can disagree.

## Allowed conclusion

Within these two datasets and the frozen 10–35% image-budget YOLOv8n protocol, FrozenDINO-FF-v1 yielded a small positive paired downstream AULC difference relative to Random, while NEU class-level results exposed a safety-relevant trade-off.

## Not established

Universal DINO superiority; superiority to state-of-the-art active learning; all-class benefit; production or monetary annotation savings; causal industrial generalization; statistical independence of 440 jobs; historically untouched public test performance.

## Where to inspect

- Primary table: `results/thesis_final/tables/locked_final_primary.csv`
- Paired seeds: `results/thesis_final/tables/acquisition_paired_effects.csv`
- Full conceptual curves: `results/thesis_final/tables/locked_final_conceptual_480.csv`
- Final report: `confirmatory_v1_1/final_analysis/CONFIRMATORY_FINAL_ANALYSIS_REPORT.md`
- Protocol: `docs/collaboration/CONTROLLED_440_PROTOCOL.md`

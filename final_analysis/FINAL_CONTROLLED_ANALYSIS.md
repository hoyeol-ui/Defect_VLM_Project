# Final Controlled Analysis

## Executive summary

The locked Random–Frozen DINO study completed with integrity `PASS`. Frozen DINO produced a small positive primary mAP50-95 AULC difference in GC10-DET (`+0.581 pp`, 10/10 acquisition seeds) and NEU-DET (`+0.578 pp`, 9/10). Both acquisition-bootstrap intervals exclude zero and both Holm-adjusted exact sign-flip p-values are `0.003906`. This is a bounded cross-dataset positive result under the frozen protocol.

The result is not uniformly beneficial. NEU worst-class AULC fell `1.488 pp` and final worst-class AP fell `3.118 pp`; the deterioration repeated in all ten acquisition seeds. Consequently, the academically defensible verdict is **bounded positive overall learning utility with a material class-level safety trade-off**.

## Protocol and estimand

The study compared uniform Random-PCG64-v1 with FrozenDINO-FF-v1 on fixed prospective GC10-DET and NEU-DET splits. For each dataset, ten acquisition seeds created paired trajectories. Two training seeds were run on each selected set. Each learner was trained from the same sealed YOLOv8n initialization for 100 epochs at six cumulative image budgets; the development split selected `best.pt`, and the locked final split supplied the outcome.

The primary estimand is the dataset-specific mean of ten paired acquisition effects. For each acquisition and strategy, the two learner-seed AULCs are averaged first. AULC is trapezoidal area over the six observed budget fractions divided by the evaluated budget span. GC10 and NEU are not pooled.

## Integrity

Training and final evaluation both completed 440/440. There are 480 conceptual strategy-budget rows after shared Round0 is attached to both trajectories, 80 complete learner curves, 40 acquisition-level strategy curves, and 20 paired primary contrasts. No primary point was missing or interpolated. No development metric substituted for a final metric, and no post-final training or tuning occurred. See `FINAL_INTEGRITY_AUDIT.md`.

## Primary and statistical result

| Dataset | Random mean AULC | DINO mean AULC | mean paired delta | median delta | 95% CI | wins/losses | exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GC10-DET | 0.244643 | 0.250454 | +0.005812 | +0.006889 | [+0.003970, +0.007576] | 10/0 | 0.001953 | 0.003906 |
| NEU-DET | 0.351950 | 0.357731 | +0.005782 | +0.005687 | [+0.003100, +0.008119] | 9/1 | 0.003906 | 0.003906 |

The uncertainty interval is a 100,000-replicate acquisition-cluster percentile bootstrap. The two-sided test enumerates all `2^10` sign assignments within each dataset, followed by Holm correction across the two primary dataset tests. These rules were frozen before the locked-final outcomes were opened.

## Budget-level curves

Shared Round0 produces a zero contrast by design. The largest differences occurred at R2: GC10 `+1.142 pp` and NEU `+1.091 pp`. GC10 was also clearly positive at R4; NEU was clearly positive at R1. Other individual-round intervals generally included zero, including both final budgets. The AULC result therefore describes a modest advantage accumulated across the trajectory, not an endpoint-only victory or a monotonically widening gap.

## Secondary outcomes

GC10 recall AULC increased `+0.723 pp` (95% CI `+0.258` to `+1.251`); NEU recall AULC was `+0.412 pp` with an interval crossing zero. Final recall intervals crossed zero in both datasets. Final mAP50-95 differences were `+0.590 pp` in GC10 and `+0.274 pp` in NEU, also with intervals crossing zero. Secondary results are descriptive and do not rescue or redefine the primary claim.

## Class and subgroup results

The frozen GC10 rare-class macro AULC increased `+0.673 pp` (95% CI `+0.302` to `+1.068`, 9/10), although the final rare-class effect was uncertain. Individual final-budget classes remained mixed: welding line, crescent gap, water spot, and crease improved; silk spot, inclusion, and rolled pit decreased.

NEU's macro metric equals its overall metric because its six classes are macro-averaged in the detector output, but its worst-class endpoint reveals the hidden trade-off. Worst-class AULC changed `−1.488 pp` (95% CI `−1.978` to `−1.002`, 0/10), and final worst-class AP changed `−3.118 pp` (95% CI `−3.822` to `−2.372`, 0/10). At the final budget, scratches (`+5.449 pp`) and pitted surface (`+2.236 pp`) improved, while crazing (`−3.118 pp`) and rolled-in scale (`−3.336 pp`) deteriorated.

## Selection composition and mechanism-level interpretation

At GC10 R5, Frozen DINO selected on average 14.7 more images containing the frozen rare targets, raised normalized image-incidence entropy by 0.0207, and added 46.6 instances relative to Random. This is consistent with the positive rare-macro and overall results.

At NEU R5, Frozen DINO reduced normalized entropy by 0.0610 and selected 54.9 fewer total instances. Relative image counts shifted by `−41.5` crazing, `−35.1` rolled-in scale, `+31.6` pitted surface, and `+57.6` scratches. The class AP changes follow the same broad direction. This is an association from a randomized acquisition comparison, not a proof that those count shifts alone caused the AP changes.

## Robustness

GC10's paired acquisition effects ranged from `+0.155` to `+1.001 pp`; all ten were positive. NEU ranged from `−0.400` to `+1.248 pp`; nine were positive. The conclusion does not depend on a single favorable acquisition seed. Learner-seed variation remains nonzero, which is why the two learner repeats are averaged rather than treated as independent acquisitions.

## Relationship to historical evidence

- Historical NEU fixed-set improvement (`+0.016236`, 5/5) measured training stability, whereas the repeated-acquisition estimate (`+0.007019`, interval including zero) did not establish acquisition reproducibility. The controlled study resolves a new, prospectively frozen estimand without pooling those earlier runs.
- Historical GC10 q20 showed overall improvement with rare-macro deterioration. The controlled result again shows that overall and class/subgroup conclusions must be inspected separately, although the frozen rare macro here improved.
- MPDD and VisA measured discovery/composition, KSDD2 used a task-specific gate, and DeepPCB supplied a prospective STOP decision. Those experiments motivate the evidence-layer framework but do not enter the controlled statistical sample.
- Early VLM validity failures explain why semantic-looking selector scores cannot stand in for downstream evidence.

## Scientific verdict and thesis implications

- GC10: **bounded positive**, with mixed individual classes.
- NEU: **bounded positive primary, mixed/safety-negative subgroup evidence**.
- Cross-dataset: **bounded positive under the exact protocol**, not a universal selector theorem.

The thesis central argument is supported: selection-stage validity, discovery, composition, reproducibility, and downstream learning are distinct. A representation-diversity selector can show a reproducible overall gain while redistributing class exposure and harming the worst class. Strong active-learning evaluation therefore needs full learning curves, acquisition replication, nested learner repeats, and subgroup reporting—not a single selection score or final endpoint.

## Claim boundaries and limitations

Allowed: a small positive paired downstream AULC difference in both evaluated datasets under the frozen Random-versus-Frozen-DINO protocol; a replicated NEU worst-class deterioration; an empirical need to separate evidence stages.

Forbidden: universal DINO superiority; all-class safety; superiority to SOTA active learning; production cost savings; annotation-time or monetary savings; causal generalization to all industrial data; treating 440 jobs as `n=440`; calling the prospective split a historically untouched public test set.

The evidence is limited to two datasets, one detector/configuration, one image-level frozen DINO representation, about 10–35% image budgets, ten acquisition seeds, and two nested learner seeds. Dataset provenance and license uncertainty also limit redistribution, not the internal numerical audit.

## Manuscript updates

1. Lead with the evidence-layer question, not DINO novelty.
2. Report dataset-specific primary estimates and acquisition-seed raw effects.
3. Put the NEU worst-class failure next to the overall positive result.
4. Explain the 440 lifecycle separately from statistical `n=10`.
5. Keep historical evidence in a separate section and use only relational language such as “consistent with,” “contrasts with,” or “does not address.”
6. Include the primary and classwise figures from `results/thesis_final/figures/` and cite the canonical CSVs.

# Collaborator Brief

## The research in one sentence

This thesis tests whether attractive **selection-stage signals** in industrial-defect active learning actually translate into reproducible **downstream detector learning**, while keeping discovery, selected-set composition, acquisition variability, learner variability, and subgroup safety separate.

## Where it started and why it changed

The project began with a constructive hypothesis: combine VLM semantics, groundedness, DINO visual diversity, and detector uncertainty to choose more useful images than Random. Several candidates produced promising-looking selections. Yet signal-validity checks were weak, Random was hard to beat downstream, and gains in target discovery could coexist with category collapse or rare-class damage. A favorable selected set could also remain favorable across training seeds but fail to reproduce across newly selected acquisition sets.

That evidence changed the question. The thesis is no longer “Did we invent a novel DINO selector?” It asks: **At which evidence stage does an apparent selection benefit survive, and under what conditions does it reach downstream learning?** Negative and mixed findings are therefore part of the answer, not failed attempts to hide.

## The five evidence stages

1. **Candidate validity:** whether a VLM score, embedding distance, or groundedness proxy corresponds to the intended property.
2. **Discovery:** whether the selector finds more anomalies, rare classes, or target images at a fixed review budget.
3. **Composition:** whether the selected set remains balanced across categories, sources, and defect classes.
4. **Reproducibility:** whether direction persists across acquisition seeds; separately, whether training on one selected set is stable across learner seeds.
5. **Downstream learning:** whether detector performance improves across a complete label-budget curve on locked outcomes.

These are not interchangeable. MPDD and VisA, for example, found more anomalies but failed composition/safety checks. Historical NEU fixed-set training was stable, but new acquisition sets were inconclusive. GC10 showed cases where overall and rare-class downstream effects pointed in different directions.

## Why Random versus Frozen DINO

Random is the honest baseline: cheap, GT-blind, and often unexpectedly competitive. FrozenDINO-FF-v1 was selected as a clean, prospectively specified representation-diversity candidate—not as a claimed new algorithm. It extracts fixed `facebook/dinov2-small` image embeddings and greedily chooses the point farthest in cosine distance from the currently labeled set, with deterministic sample-ID tie-breaking. It does not use ground-truth class, box, rarity, development outcome, final outcome, or learner seed.

## What “440 experiments” means

For each of two datasets there are 10 acquisition seeds and two learner seeds. Each trajectory has six budgets, but Random and DINO share Round0. The physical detector-job count is:

`2 datasets × 10 acquisitions × 2 learners × [1 shared R0 + 5 Random + 5 DINO] = 440`.

This is not statistical `n=440`. Multiple jobs share an acquisition trajectory and selected set. The primary unit is the acquisition seed, so the inferential sample is `n=10` paired DINO-minus-Random effects per dataset. Learner A/B repeats are averaged within acquisition and strategy.

## Outcome protection

The development split selected each job's `best.pt`. The locked-final split was unavailable to selection, training, or protocol choices and was evaluated only after 440/440 training integrity passed and explicit final authorization was recorded. “Locked final” describes this study-specific prospective lock; it is not a claim that the public data had never been used historically.

## Primary endpoint

The primary endpoint is locked-final YOLOv8n `mAP50-95`, summarized as normalized trapezoidal area under the six-budget learning curve. Each learner obtains an AULC; learner A/B are averaged; then DINO minus Random is paired within acquisition. GC10 and NEU are reported separately. The frozen uncertainty/test rules are a 100,000-replicate paired acquisition bootstrap, exact `2^10` sign-flip test, and Holm adjustment over the two dataset-level primary tests.

## Final result

GC10: Random AULC `0.244643`, DINO `0.250454`, paired delta `+0.005812` (95% CI `+0.003970` to `+0.007576`), 10/10 positive acquisitions.

NEU: Random AULC `0.351950`, DINO `0.357731`, paired delta `+0.005782` (95% CI `+0.003100` to `+0.008119`), 9/10 positive acquisitions.

Both Holm-adjusted p-values are `0.003906`. The gain is small and concentrated more in early/middle budgets than at the final endpoint. It is a bounded positive primary result.

## The result's essential complication

Overall performance did not imply subgroup safety. GC10's frozen rare-class macro AULC improved `+0.673 pp`, but individual GC10 classes remained mixed. NEU worst-class AULC deteriorated `−1.488 pp` and final worst-class AP deteriorated `−3.118 pp`, both with 0/10 positive acquisitions. The selected-set composition moved away from crazing and rolled-in scale and strongly toward scratches and pitted surface, alongside corresponding class-performance changes.

## What it means

- A simple frozen representation-diversity selector can produce a small, repeated overall downstream gain under two specific controlled protocols.
- Selection composition helps explain why overall and class-level results can diverge.
- Full curves, acquisition replication, nested learner repeats, and subgroup endpoints materially change the scientific interpretation.
- The thesis's evidence-layer argument is supported even though DINO itself is not a novel universal selector.

## What it does not mean

It does not show universal DINO superiority, all-class benefit, production deployment success, monetary annotation savings, superiority over SOTA active-learning algorithms, or generalization to all industrial data. The 440 jobs are not 440 independent samples. Historical evidence is not pooled with controlled evidence.

## Main limitations

Two datasets; one YOLOv8n lifecycle; one global DINOv2 representation; image-count rather than annotation-time budgets; roughly 10–35% of the acquisition universe; ten acquisition and two nested learner seeds; public-benchmark rather than live-production data; incomplete/uncertain redistribution terms for some source datasets.

## First code and results to inspect

1. `confirmatory_v1/confirmatory_protocol_v1.yaml`
2. `confirmatory_v1/src/confirmatory_v1/protocol_core.py`
3. `confirmatory_v1/src/confirmatory_v1/frozen_dino_ff_v1.py`
4. `confirmatory_v1_1/full_run/src/run_full_440_orchestrator.py`
5. `confirmatory_v1_1/final_analysis/src/analyze_final_results.py`
6. `results/thesis_final/tables/acquisition_paired_effects.csv`
7. `final_analysis/FINAL_CONTROLLED_ANALYSIS.md`

Use `CODE_INDEX.md` before treating any other Python file as authoritative.

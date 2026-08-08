# GC10 frozen-result lineage and sensitivity audit

- Overall forensic verdict: **PASS_WITH_REPRODUCIBILITY_CAVEATS**
- Frozen research gate remains: **FAIL**
- Training/inference performed by this audit: **False / False**
- Locked final manifest used by this audit script: **False**

## What was independently checked

- Source/cache image bytes checked: 2068; mismatches: 0
- Generated YOLO labels checked: 2068; mismatches: 0
- Checkpoints checked: 30; hash mismatches: 0
- Recorded training args checked: 30; mismatches: 0
- Metric rows independently recomputed: aggregate 30, per-class 300

## Independently reproduced conclusion

- Random mean mAP50-95: 0.124235376
- DINO mean mAP50-95: 0.141613783
- Paired acquisition-unit delta: +0.017378407
- Correct five-acquisition bootstrap CI: [0.000580286, 0.036332212]
- Rare macro acquisition deltas: {'0': -0.035031807806458155, '1': 0.0026153391594049055, '2': -0.009687337219629019, '3': 0.0006528148833431693, '4': -0.05793273674379265}
- Interpretation: overall development mAP improved in this protocol, but the pre-registered rare-class safety requirement failed; the branch-level verdict therefore remains FAIL/STOP.

## Could a different implementation or request have changed the outcome?

- Without canonical sorting, the same five initial seeds overlap the actual 20-image initial sets by only 0.00/20 on average. Different row order would therefore train on materially different images.
- Leave-one-acquisition-out overall mAP deltas range from +0.008674 to +0.024409; the effect is not identical across acquisition realizations.
- Leave-one-acquisition-out rare-macro deltas remain between -0.025500 and -0.010363; none approaches the required +0.020 safety gain.
- Near-duplicate review candidates (pHash Hamming <=2): 0 pairs across 0 development images. These are not confirmed SHA duplicates and require separate visual/domain review.
- A different detector, optimizer policy, data split, augmentation, or official test would answer a different question. Their direction cannot be inferred from these artifacts without a newly authorized experiment.

## Remaining reproducibility risks

- Historical result-cache reuse checks only whether aggregate/per-class JSON exists; it does not reseal the config, checkpoint, or manifests before returning cached metrics.
- Historical image-cache reuse accepts an existing target without verifying its content at cache-build time. The present audit found zero current mismatches, but the runner itself lacked this guard.
- Training YAML/TXT files contain machine-specific absolute paths and are not portable as frozen manifests.
- The historical run config records the base model SHA but not Git commit, runner SHA, source-manifest seals, Python/PyTorch/CUDA/Ultralytics versions, or a complete environment lock.
- All 30 best.pt files differ in SHA-256 from last.pt. The protocol explicitly evaluated last.pt, so this is not a protocol error, but a best-checkpoint policy would be a different experiment and could produce different metrics.
- Initial sets are reconstructed from row positions rather than persisted in the selection record; canonical sorting makes the actual run consistent, but a changed row order would change the data request.
- The five acquisition seeds are the correct independent units. Treating 15 training runs as independent would understate selection-level uncertainty.
- Perceptual hash was audit-only, not a hard split constraint. Hamming-near candidates are not exact leaks but remain a domain-review item.
- The current audit environment uses pandas 3.0.3, while requirements-analysis.txt specifies pandas<3. The independent summaries still matched exactly, but the environment should be reconciled before a frozen rerun.

## Bottom line

The stored GC10 result is internally consistent with the current source manifests, cache, labels, checkpoints, training arguments, and analysis code. No evidence of an arithmetic, split-SHA, cache-content, or checkpoint-association error was found. Exact rerun reproducibility is still weaker than the result audit because software versions and every input seal were not recorded by the historical runner. The negative rare-class gate is the correct frozen conclusion for the experiment that was actually run.

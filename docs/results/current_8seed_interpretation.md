# Current 8-seed Interpretation

This snapshot summarizes the latest lab-meeting evidence without tracking raw YOLO
training folders or generated score directories.

## Main Takeaway

The current experiments do not support a strong claim that the proposed acquisition
score robustly outperforms random sampling. However, they do support a more precise
and defensible claim:

> Expert-designed VLM prompt-family consistency is a meaningful GT-free acquisition
> signal, while pseudo groundedness is a weak auxiliary signal whose direction and
> calibration still require further validation.

## Evidence

- `ConsistencyOnly` is clearly stronger than `GroundednessOnlySoft`, which supports
  the core hypothesis around semantic explanation consistency.
- `CombinedSoftPenalty` and `CombinedSuppressNoPseudo` do not clearly improve over
  `ConsistencyOnly`, so pseudo groundedness should not be framed as the main method.
- `Random` remains a strong detector-level reference baseline, especially in final
  mAP, so the current method should not be presented as closed or superior.
- `LowPrioritySoft` has strong AULC, which means score direction remains an open
  calibration issue rather than an exception to ignore.

## Next Research Moves

1. Put `ConsistencyOnly` back at the center of the paper narrative.
2. Treat pseudo groundedness as an auxiliary extension and failure-analysis source.
3. Add diversity-aware or class-balanced selection to address why Random is strong.
4. Report per-class AP and paired seed-level comparisons, not only aggregate mAP.
5. Validate prompt-family versions and prompt sensitivity as part of the core method.

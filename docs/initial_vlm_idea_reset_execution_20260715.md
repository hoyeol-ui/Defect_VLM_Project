# Initial VLM Idea Reset: Execution Note

Date: 2026-07-15  
Detector training performed: no  
Final test evaluated: no

## Completed work

1. Frozen a validity-first protocol for explanation consistency and
   groundedness.
2. Re-audited the existing 99-image Qwen2-VL pilot without new inference.
3. Excluded four legacy pilot images found in locked-final manifests before
   computing any statistic.
4. Replaced the old location/scale/appearance SBERT comparison with a frozen
   five-paraphrase structured prompt family.
5. Added field-level consistency scoring for defect presence, class, bbox,
   location, scale, abstention, and confidence.
6. Added a post-hoc GC10 bbox/location/scale groundedness audit.
7. Generated a safe 20-image GC10 development pilot plan containing 100 VLM
   calls.

## Legacy result

The legacy consistency signal failed every diagnostic check on 95 safe images.

| Metric | Result |
|---|---:|
| Spearman consistency vs groundedness | -0.181199 |
| Bootstrap 95% CI | [-0.389654, 0.036410] |
| Inconsistency AUROC for severe grounding failure | 0.373433 |
| High-minus-low consistency quartile groundedness | -0.187500 |
| GC10 within-dataset Spearman | -0.311545 |
| NEU within-dataset Spearman | -0.084942 |

The highest-consistency quartile had mean groundedness 0.2708 and a severe
failure rate of 0.5833. The lowest-consistency quartile had mean groundedness
0.4583 and a severe failure rate of 0.3333. There were 14 high-consistency
severe failures and six low-consistency fully grounded counterexamples.

Therefore the old SBERT score must not be reused as an acquisition or error
signal. This conclusion is limited to the old, semantically mismatched prompt
family; it does not decide the new structured-prompt hypothesis.

## Final-lock reconciliation

The current GC10 development manifest has 232 rows. A conservative union of
the older V7 final lock and the current GC10 final lock identified 15 filenames
inside that development manifest. They are automatically excluded, leaving 217
eligible rows. The frozen 20-image pilot is sampled from those 217 rows with
seed `20260715`.

## Frozen 20-image pilot

- Dataset: GC10-DET development only
- Eligible pool after all known final locks: 217
- Pilot images: 20
- Equivalent prompt variants per image: 5
- Planned responses: 100
- Model: `Qwen/Qwen2-VL-2B-Instruct`
- Decoding: deterministic
- GT visible during generation: no
- Prompt-family SHA256:
  `b1d37a9bf02987f3d75d63b58d1a024cd7e7c6cf7b9f9c0444e0684c53e12b44`

The first one-image smoke attempt began the model download but was stopped
before inference because the model was not cached. About 0.87 GB was cached;
no structured VLM response was produced and no training occurred.

## Commands

Run from the project root in PowerShell.

### 1. Generate the frozen 20-image responses

```powershell
.\.venv\Scripts\python.exe scripts\01_score_generation\generate_structured_vlm_validity_responses.py `
  --manifest runs\gc10_taxonomy_protocol\gc10_protocol_20260715\gc10_development_eval.csv `
  --output-dir runs\vlm_consistency_groundedness_validity\structured_prompt_pilot20_gc10_20260715 `
  --default-dataset GC10-DET `
  --default-split-role development_eval `
  --max-images 20 `
  --sample-seed 20260715
```

The partial Hugging Face download should resume. The response JSONL is
append-only and the command skips completed image/prompt pairs when rerun.

### 2. Compute structured consistency

```powershell
.\.venv\Scripts\python.exe scripts\01_score_generation\score_structured_vlm_validity.py `
  --responses runs\vlm_consistency_groundedness_validity\structured_prompt_pilot20_gc10_20260715\structured_vlm_responses.jsonl `
  --output runs\vlm_consistency_groundedness_validity\structured_prompt_pilot20_gc10_20260715\structured_vlm_signal_scores.csv
```

### 3. Join XML only after generation and audit groundedness

```powershell
.\.venv\Scripts\python.exe scripts\03_analysis\audit_structured_vlm_gc10_groundedness.py `
  --signals runs\vlm_consistency_groundedness_validity\structured_prompt_pilot20_gc10_20260715\structured_vlm_signal_scores.csv `
  --output-dir runs\vlm_consistency_groundedness_validity\structured_prompt_pilot20_gc10_20260715\groundedness_audit
```

## Pilot decision

This 20-image run is a pipeline and output-quality pilot, not an AL result.

- If valid JSON response rate is below 90%, repair only the output parser or
  response-format instruction and repeat the same 20 images.
- If bbox response coverage is below 80%, do not claim localization
  consistency.
- If structured consistency still has no positive relationship with actual
  groundedness, stop the VLM-consistency acquisition branch.
- If the signal is directionally valid, freeze the exact prompt/model/hash and
  pre-register a larger development-only error-prediction study before any new
  detector training.


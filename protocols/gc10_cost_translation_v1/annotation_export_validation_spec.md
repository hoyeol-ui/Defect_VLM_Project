# Annotation Export Validation Specification

Protocol version: `gc10-cost-translation-v1-candidate`

## Inputs

- frozen blinded task manifest
- raw append-only event JSONL
- final human annotation export
- optional correction/adjudication event JSONL
- frozen idle threshold and tool version

## Hard validation checks

| Check | PASS condition |
|---|---|
| JSON schema | Every event satisfies `annotation_event_schema.json` |
| Event ID | Unique within the complete study export |
| Checksum | Every event checksum matches canonical event content |
| Protocol/tool version | Single frozen values per study phase |
| Monotonic clock | Nondecreasing inside each session |
| UTC audit clock | Parseable ISO-8601 UTC; not used for duration |
| State machine | No illegal transition, orphan edit, or unclosed image |
| Task linkage | Every submitted image maps to exactly one blinded task and sample |
| Order linkage | `order_index` matches the annotator-specific order manifest |
| Pause handling | No active time accrued while paused |
| Idle handling | Intervals above the frozen cutoff are excluded and flagged |
| Active/wall reconciliation | Difference is within the Stage 0 frozen tolerance |
| Final export | Every `IMAGE_SUBMIT` has a parseable annotation snapshot |
| Blinding | No strategy, seed, rank, GT, rare/small/multilabel field in annotator inputs |
| Duplicate exposure | Each Phase A sample occurs once per annotator |
| Corrections | All post-submit changes have correction reason and separate active time |

Any hard-check failure blocks Phase A analysis until the cause is resolved without looking at strategy outcomes.

## Derived per-image export

The validator must produce one row per `annotator_id × blinded_task_id` containing:

```text
protocol_version
tool_version
annotator_id
session_id
blinded_task_id
sample_id
order_index
image_start_utc
image_submit_utc
wall_ms
active_ms
explicit_pause_ms
idle_excluded_ms
system_overhead_ms
box_create_count
box_move_count
box_resize_count
box_delete_count
class_assign_count
class_change_count
submitted_box_count
annotation_export_path
annotation_export_sha256
raw_event_path
raw_event_sha256
validation_pass
validation_error_codes
```

## Stage 0 completeness decision

The exact completeness and reconciliation tolerances remain researcher decisions. The candidate audit recommends that at least 95% of Stage 0 tasks have complete, recoverable logs and that all clock/state/linkage errors be resolved before Phase A. This recommendation is not frozen until it is signed in `unresolved_researcher_decisions.md`.

## Prohibited repair

- Do not infer missing active time from strategy identity or mean arm time.
- Do not delete slow tasks because they weaken an arm.
- Do not change the idle threshold after viewing Phase A arm contrasts.
- Do not silently replace a corrupted task with another image.
- If a task must be repeated for technical reasons, preserve both attempts and a blinded reason code.

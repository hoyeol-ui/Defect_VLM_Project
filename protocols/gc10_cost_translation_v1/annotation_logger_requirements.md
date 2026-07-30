# Annotation Logger Requirements

Protocol version: `gc10-cost-translation-v1-candidate`

Status: specification only; no UI is implemented or authorized by this package.

## 1. Measurement objective

The logger must recover, for every blinded image and annotator, wall-clock duration, active annotation duration, explicit pause duration, idle-excluded duration, box-edit operations, class-edit operations, submission, and any post-submission correction. The primary burden endpoint must not be reconstructed from a screen recording or a session-level stopwatch when image-level events are available.

## 2. Required state machine

```text
SESSION_START
  -> IMAGE_START
      -> active edit events
      -> IMAGE_PAUSE -> IMAGE_RESUME (zero or more)
      -> IMAGE_SUBMIT
      -> optional correction session recorded separately
  -> next IMAGE_START
  -> SESSION_END
```

Invalid sequences must be rejected by the export validator. An image cannot be submitted without an `IMAGE_START`; edit events cannot occur outside an open image; a paused image cannot accrue active time.

## 3. Required events

- `SESSION_START`
- `IMAGE_START`
- `IMAGE_PAUSE`
- `IMAGE_RESUME`
- `BOX_CREATE`
- `BOX_MOVE`
- `BOX_RESIZE`
- `BOX_DELETE`
- `CLASS_ASSIGN`
- `CLASS_CHANGE`
- `IMAGE_SUBMIT`
- `SESSION_END`

All events must satisfy `annotation_event_schema.json`. Event records are append-only. Previously written events must never be silently rewritten.

## 4. Clock requirements

- `event_utc` is retained for audit and cross-file ordering.
- `monotonic_timestamp_ms` is the duration source and must not move backward within a session.
- System-clock changes must not alter active-time calculations.
- UI load time before the image is usable must be recorded separately or excluded by placing `IMAGE_START` after the image-ready signal.
- The logger must flush events locally at every `IMAGE_SUBMIT` and `SESSION_END`.

## 5. Active-time calculation

For an image \(i\) and annotator \(a\):

1. The measurement window starts at `IMAGE_START` and ends at `IMAGE_SUBMIT`.
2. Wall time is the monotonic end minus monotonic start.
3. Explicit `IMAGE_PAUSE` to `IMAGE_RESUME` intervals contribute zero active time.
4. For consecutive valid events while not paused, an interval contributes to active time only when its duration is less than or equal to the frozen idle threshold.
5. An interval above the idle threshold contributes zero and the receiving event must have `idle_flag=true`.
6. `active_ms_since_previous` stores the included portion, not the raw wall interval.
7. The exported image active time is the sum of `active_ms_since_previous` within the image window.
8. The reconciliation identity must hold:

   `wall_ms = active_ms + explicit_pause_ms + idle_excluded_ms + system_overhead_ms`

   within the Stage 0 tolerance that is frozen before Phase A.

The idle threshold is unresolved. Stage 0 may compare candidate thresholds only for log repeatability and obvious idle contamination. It may not compare DINO and Random or use strategy outcomes.

## 6. Correction time

Any change after `IMAGE_SUBMIT` must create a separate correction record with the original blinded task ID, correction actor, reason, start/end monotonic timestamps, events, and active time. Adjudication and correction time are cost components; they must not be absorbed into the initial annotation time or omitted.

## 7. Blinding

The UI input may contain only the fields in `stage0_annotator_blinded_manifest.csv` or `phase_a_annotator_blinded_manifest.csv`, plus an annotator-specific counterbalanced order created after researcher decisions. It must not expose:

- strategy
- acquisition seed
- query rank
- GT class or box
- rare/small/multilabel audit flags
- selection frequency

## 8. Integrity and privacy

- `annotator_id` must be a study pseudonym, not a name or email.
- `checksum` is SHA-256 of canonical UTF-8 JSON with lexicographically sorted keys and the `checksum` field omitted.
- Export must preserve the original event JSONL and a derived per-image table.
- Tool version, protocol version, OS/browser version, display resolution, input device, and annotation-guide version must be recorded.
- Screen recording is disabled by default and requires separate institutional and participant approval.

## 9. Stage 0 acceptance scope

Stage 0 may estimate only:

- event completeness
- checksum validity
- state-machine validity
- clock monotonicity
- active/wall reconciliation
- export repeatability
- image/annotation linkage
- idle-threshold measurement behavior

It may not estimate strategy effect, choose a favorable burden threshold, or contribute images to Phase A/B.

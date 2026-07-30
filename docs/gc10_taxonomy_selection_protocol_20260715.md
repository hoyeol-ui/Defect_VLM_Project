# GC10-DET All-Defect Cold-Start Taxonomy Selection Protocol

Date: 2026-07-15

## Question

Can frozen DINOv2 farthest-first selection improve early defect-taxonomy and
rare-class discovery over uniform Random in an all-defect industrial pool?

This is a selection-only experiment. It does not claim detector improvement and
does not authorize YOLO training unless every pre-registered gate passes.

## Source audit and split

- Source files: 2,312 JPG and 2,294 XML.
- Canonical usable labeled images: 2,292.
- Exclusions: 12 noncanonical copies sharing a basename with another folder, 6
  images without XML, and 2 XMLs without any object.
- One undefined object name (`d`) is ignored while the valid class-2 object in
  the same image is retained.
- `10_yaozhed` is normalized to class ID 10 / `10_yaozhe`.
- Valid boxes: 3,563.
- Exact duplicate images among canonical images: 0.
- Filename-derived production bursts and exact SHA duplicates are hard split
  groups. pHash is audit-only.
- Deterministic split: acquisition 1,836, development 232, final locked 224.
- Final manifest SHA-256:
  `748c21ba9a6f486ca776172646afca14ba5982fcd3c9d764258998fd8c72c37a`.

The final split must not be read by embedding, selection, gate, or development
code.

## Leakage controls

Folders 1--10 encode defect labels. Filenames and production groups may also
encode source regime. Therefore the selector manifest contains only opaque
sample ID, image SHA-256, and pHash. It contains no path, filename, folder,
production group, class, XML, or bbox field.

A private loader map may join opaque ID to source path only for frozen pixel
embedding extraction. Source paths are not exported with embeddings. XML and
bbox data are joined only after both selectors have fixed their query IDs.

## Frozen experiment

- Acquisition seeds: 0--199.
- Shared Random initial set: 20 images.
- Query budget: 20 images.
- Representation: frozen `facebook/dinov2-small`, L2-normalized.
- `GTFreeRandom`: uniform sampling without replacement.
- `FrozenDINOVisualDiversity`: greedy cosine farthest-first from the shared
  initial set and prior query selections; ties use opaque sample-ID order.
- No category constraint or class proxy is allowed.

The rare set is frozen before selection as the three least prevalent
acquisition classes by image presence: class 8 (41 images), class 9 (43), and
class 10 (114). The definition and membership cannot change after results.

## Post-hoc outputs

- unique classes in query and initial-plus-query;
- new classes added beyond initial;
- query images containing a rare class;
- unique/new rare classes;
- annotations to first rare-class image;
- bbox instances, multi-label images, and summed bbox-area ratio;
- query overlap;
- within-query cosine redundancy and distance to initial;
- per-class selected-image and instance yields.

## Pre-registered YOLO authorization gate

All checks must pass:

1. DINO minus Random mean initial-plus-query unique-class coverage is at least
   +0.25 class.
2. Its paired 95% bootstrap CI lower bound is above zero.
3. Its paired loss rate is at most 0.15.
4. DINO minus Random mean rare-class-image yield is at least +0.75 image per
   20-image query.
5. Its paired 95% bootstrap CI lower bound is above zero.
6. Its strict paired win rate is at least 0.60.
7. DINO minus Random mean unique-rare-class coverage is at least +0.25 class.
8. Its paired 95% bootstrap CI lower bound is above zero.
9. Mean bbox-instance yield is non-inferior to Random with a margin of -1.0
   instance per query.
10. DINO mean within-query cosine similarity is lower than Random.
11. DINO mean minimum distance to the initial set is higher than Random.

Checks 10--11 are implementation/signal sanity checks, not detector evidence.
If any check fails, YOLO training is prohibited. If all pass, a separate,
pre-registered development-only detector confirmation is required; the final
test remains locked.

## Expected artifacts

- source/split audit and exclusion tables;
- blind acquisition and private loader manifests;
- frozen embedding array and path-free manifest;
- 200-seed selection records, overlap, per-class yields, paired metrics, gate
  table, and Markdown decision summary.


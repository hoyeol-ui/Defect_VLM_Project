# GC10 Bounding-Box Annotation Guide Outline

Protocol version: `gc10-cost-translation-v1-candidate`

Status: outline only; class examples and edge-case rulings must be finalized before Stage 0.

## 1. Purpose

- Draw one tight bounding box around every visible target defect.
- Assign one of the ten frozen GC10 classes.
- Prioritize completeness and correct class over speed.
- Do not infer strategy, seed, rarity, or expected number of defects.

## 2. Class dictionary

For each class 1–10, the final guide must include:

- Korean and source class name;
- short visual definition;
- positive examples;
- hard negative examples;
- boundary cases;
- distinction from visually similar classes;
- rule for multiple adjacent instances.

Rare status must not be shown to annotators.

## 3. Bounding-box geometry

- Enclose the visible defect tightly.
- Do not include large amounts of unaffected background.
- Do not crop the defect.
- Define how to handle truncated defects at the image edge.
- Define whether disconnected regions are one defect or multiple instances.
- Define how to handle overlapping defects.
- Define minimum visible size and zoom use without exposing the research small-defect threshold.

## 4. Completeness scan

Before submission:

1. scan the full image in a consistent direction;
2. inspect high-texture and edge regions;
3. verify every box has a class;
4. check for duplicate boxes;
5. confirm small defects were not omitted;
6. submit only when complete.

This checklist is the same for every image and annotator.

## 5. Uncertainty

- Do not consult hidden GT.
- Use a frozen uncertainty flag or reason code if the UI supports it.
- Do not ask the researcher which strategy selected the image.
- Guide questions from Stage 0 must be resolved in the guide before Phase A.
- Phase A guide changes are prohibited unless a technical/ethical stop is declared and the study is restarted under a new version.

## 6. Time and pause controls

- Start only when ready to inspect the image.
- Use explicit pause for interruption, break, or question.
- Do not leave an image open while doing unrelated work.
- Resume before editing.
- The logger records time and edit events; it is not an employee productivity score.

## 7. Training and qualification

The final protocol must define:

- training image set disjoint from Stage 0 and Phase A;
- minimum guide comprehension check;
- practice-box feedback procedure;
- whether a qualification threshold is required;
- rule for retraining without using Phase A images.

## 8. Stage 0

Stage 0 contains rare, small, large, single-box, multi-box, high-box-count, and multilabel stress cases. It tests logger and guide clarity only. No DINO-versus-Random result is computed.

## 9. Adjudication

Annotators must be told whether and how disagreements are reviewed, but the primary annotation must remain unchanged and separately archived. Review time and corrections are logged.

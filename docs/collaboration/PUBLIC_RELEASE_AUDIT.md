# Public Release Audit

Audit date: 2026-08-28  
Target: `origin/al-selector`  
Verdict: **PASS with documented non-blocking legacy limitations**

## Security

| Check | Result |
|---|---|
| `.env`, private-key, credential, or secret-named tracked files | none found |
| common API/Hugging Face/OpenAI/AWS token patterns in current tree | none found |
| private-key headers | none found |
| remote URL | HTTPS GitHub URL without embedded credential |
| newly added private authorization text | not included |
| personally identifying research-subject data | none; datasets depict industrial objects |

No secret value was printed into this audit. A filename-only token/private-key pattern scan across all 42 reachable commits returned zero file hits.

## Data and large artifacts

- No raw JPEG/XML/mask dataset file is tracked.
- No model/checkpoint weight extension is tracked.
- The largest existing tracked files are documentation PDFs/DOCX, a legacy Notion ZIP, the frozen source registry, split indices, and figures; no file exceeds common GitHub's 100 MB hard limit.
- The new locked-final public package is approximately table/figure scale, not a checkpoint cache. Private 440 raw-metric directories, materialized data caches, and weights were explicitly excluded.
- `notion_import_package_20260808.zip` predates this branch and was inspected: it contains manuscript files and derived figures, not datasets or weights.

## Copyright and licensing

- Third-party datasets and weights are referenced, not redistributed.
- Derived result figures and tables are project outputs.
- Dataset uncertainties are explicit in `DATASETS.md`; no mirror license is treated as original authority.
- Ultralytics's current AGPL/commercial dual-license context is disclosed; this branch does not redistribute Ultralytics source or weights.
- Historical PDFs/DOCX and the Notion package were already tracked on the base public project. They are retained for provenance and classified as historical/deprecated; their presence is not newly expanded by this release.

## Local paths and portability

The inherited research tree contains site-local Windows/macOS paths in 60+ legacy/config/output files. They do not contain credentials, but they reduce portability and expose the generic local account label `user`. Two exact final-analysis scripts preserve the original Windows data root as execution provenance. Rather than rewrite frozen code after seeing outcomes, `REPRODUCIBILITY.md` tells replicators to configure a local reproduction copy and record the path-only adaptation. This is accepted as a transparency/portability limitation, not a secret blocker.

## Canonicality and outcome integrity

- New collaborator-facing docs explicitly classify code as CANONICAL, HISTORICAL-RELEVANT, SUPPORTING, DEPRECATED, or SCRATCH.
- Final tables are byte-copied from the sealed analysis package and hash-verified.
- Public verification recomputes primary means, acquisition direction, exact sign-flip, and Holm correction.
- Null, negative, mixed, and subgroup-harm evidence remains visible.
- No heavy training, retraining, retuning, seed replacement, or outcome-dependent protocol edit was performed for this branch.

## Remaining non-blocking human checks

1. Verify the canonical NEU-DET publication/citation and original redistribution terms before thesis deposit.
2. If the thesis institution requires a repository-wide license, choose one with supervisor/institution guidance; it cannot override upstream data/model terms.
3. Review the old manuscript snapshots for whether they should remain public long-term. They are not required to understand the final result but were not destructively removed.

These checks do not block publishing this branch because the uncertain third-party bytes are excluded and uncertainty is disclosed.

# Public development release audit — 2026-09-13

## Scope and provenance

Base: `origin/research/entropy-a00-audit-20260908`, fetched and resolved to `9d83770376e05fe0e325573b1771941285bdd9e4`. New branch: `research/development-3way-fullref-20260913`. A separate clean worktree protects the user's modified working directory. No existing branch is rewritten, and no main/master merge or force push is used.

The local analysis package contains 86 files; all 85 payload hashes and sizes passed against its FILE_MANIFEST before copying. Original package SHA: `09f5cf4e599151e4e6afe239650b87153eca2ad304557231973755c95301db67`. Its archive is not published. All useful payloads are retained as result files, public-safe audit derivatives or archived analysis code. Source bytes and any transformations are recorded in `analysis_package_provenance.json`.

## Published material

- 23 CSV/Markdown table pairs, covering physical runs, conceptual runs, seed means, nAULC, paired effects, budgets, Full-data and selection diagnostics.
- Five representative figures, each PNG/SVG/PDF. Their original bytes are retained; they were visually inspected during the completed local analysis. Public checks verify signatures and source-table mapping.
- Existing completion records: Entropy 200 training/100 acquisition and Full-data 4; original scientific source commit `608cbb5af2692a34949a42e285c9f56087fa22f2` is distinct from this documentation release.
- Original analysis scripts, an explicitly adapted plotting wrapper, and a public CSV verifier. Archived execution sources are evidence, not new execution authorization.

## Public safety and identity

Path sanitization preserves 5,751 input SHA-256 values and byte sizes. User home paths become logical roots such as `<AGENT_UPLOAD_ROOT>`, `<HISTORICAL_LOCAL_ROOT>` or `<LOCAL_WORKSPACE>`. Remote file identities use `<RUNPOD_NETWORK_VOLUME>`. Logical identities are not asserted to exist in a public clone. Private original-path JSON is excluded.

The release contains no checkpoint bytes, raw dataset images, raw annotation files, private archives, environments/caches, credentials, SSH keys or private connection endpoints. Acquisition-pool class membership is derived image-ID/class-index metadata explicitly requested for release; it contains no images or bounding-box annotation content. Dataset attribution and redistribution boundaries remain in the repository's DATASETS.md.

Scans cover every newly added file and the full content of modified presentation documents. They check user absolute paths, credential/private-key patterns, connection patterns, prohibited file types, per-file size, terminology and relative links. They do not claim that all unrelated historical files or all git history were newly sanitized. Existing sealed historical results and protocol/freeze files remain unchanged.

The previous role-based review filename is renamed with git history retained, and affected links/headings use neutral research-record wording. The immutable base branch continues to preserve its historical snapshot.

Generated SVG path whitespace and original source end-of-file whitespace are retained as byte-preserved evidence. They are not executable or numerical defects; stripping them would change the original artifact hashes. New presentation/source wrappers use LF line endings.

## Evidence boundaries

**Development validation:** acquisition seed n=10, learner A/B averaged within seed/round; 644 physical cells, 720 conceptual AL rows, 360 seed-round rows, 60 nAULC rows. Full-data is a **full-label reference** with two actual learners per dataset. Endpoint identity grade A; overall interpretation grade B because GPU/OS differences and incomplete historical environment fields remain.

**Historical locked-final:** original Random/DINO results are preserved and independently verified by the existing verifier. They are not combined with new development metrics as a final inferential comparison.

**Exploratory selection diagnostics:** class composition, DINO geometry and score/count associations are descriptive, not causal. New locked-final extension: **NOT EXECUTED / PENDING SEPARATE FREEZE**. No new training, acquisition, inference, checkpoint evaluation, locked-test or RunPod access occurred in this release task.

## Verification

Public-release scan and both numerical verifiers must pass before push. Reproduction commands from the repository root:

```sh
python scripts/verify_public_results.py
python scripts/verify_development_20260913.py
python scripts/verify_development_20260913.py --bootstrap
```

The new default verifier uses the standard library; optional bootstrap uses NumPy. `PUBLIC_FILE_MANIFEST.json` seals published bytes, including original-source derivatives and changed presentation documents, and maps all five figures to their input tables. Byte-preserving gitattributes prevent checkout newline conversion from invalidating sealed evidence.

Status: **PASS**. Original package 85/85 hash checks; private-path, secret-pattern, terminology and relative-link scans PASS; existing public verifier PASS; new public arithmetic and 100,000-draw bootstrap × 6 verification PASS. Training/acquisition/inference/locked-test/RunPod accesses: 0. Remote branch publication is checked separately after push.

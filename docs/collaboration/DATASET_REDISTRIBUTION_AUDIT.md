# Dataset Redistribution Audit

Audit date: 2026-08-28  
Branch target: `al-selector`  
Verdict: **PASS — no raw dataset bytes are newly included**

## Scope

The worktree and tracked file list were searched for dataset-like image, annotation, mask, archive, and weight extensions. Figures were distinguished from source data by path and role. The existing ZIP was inspected by member name.

## Findings

| Category | Finding | Decision |
|---|---|---|
| JPEG/JPG/BMP/TIFF source images | no tracked files | PASS |
| XML source annotations | no tracked files | PASS |
| raw masks/annotation JSON newly added | none | PASS |
| model weights (`.pt`, `.pth`, `.ckpt`, `.bin`, `.safetensors`, `.onnx`) | no tracked files | PASS |
| PNG files | 65 tracked before this release plus six final result figures; all under `docs/`, `package/assets/`, or `results/.../figures` and classified as derived figures | PASS |
| dataset archives | none | PASS |
| existing `notion_import_package_20260808.zip` | documentation/manuscript/derived-figure package only; member listing contains no raw dataset or weight extension | PASS, legacy package retained |
| split/manifests | hashes, sample IDs, split roles, and relative source filenames; no pixels or box coordinates | allowed as reproducibility metadata |
| `private_source_registry.csv` | despite its historical name, contains relative image/annotation locators and SHA-256 hashes, not file contents; already tracked at the base freeze | retained, documented risk below |

## Dataset-specific decision

| Dataset | Verified terms | Raw included? | Release decision |
|---|---|---:|---|
| GC10-DET | explicit original redistribution license not verified | No | metadata/results only; redistribution unknown |
| NEU-DET | explicit original redistribution license not verified | No | metadata/results only; redistribution unknown |
| MPDD | CC BY-NC-SA 4.0 | No | use owner download; noncommercial/share-alike terms apply |
| VisA | CC BY 4.0 | No | use owner download and attribution |
| KSDD2 | CC BY-NC-SA 4.0 | No | use ViCoS download; commercial use requires contact |
| DeepPCB | README says research use only; code's MIT license is not assumed to license data | No | research-only owner download |

## Metadata risk

The controlled freeze intentionally publishes sample IDs, split membership, manifest membership, hashes, and upstream-relative filenames. These make the experiment auditable but can reveal original dataset naming conventions. No personal subject data or proprietary local absolute source root appears in the registry. Since the benchmark images are non-personal industrial objects and the metadata was already part of the frozen branch, this is accepted as a low public-release risk. Future releases should prefer opaque IDs alone when a dataset's terms prohibit filename disclosure.

## Reproduction policy

Collaborators obtain raw data from owners, store it outside the repository, and reconstruct/verify the frozen identities locally. A snapshot that does not reproduce curated counts and seals is reported as a new replication. Do not commit dataset files or derived crops/masks without a new license audit.

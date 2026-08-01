# Frozen-run preflight: gc10_dev_confirm_harness_v1

- Verdict: **PASS**
- Harness version: `1.1`
- Config SHA-256: `7e5d2f320d6176f65c3833fcf1508accb97ae3d4854b0f0181e9389a98452449`
- Git commit: `3b74b0449e2a9db7d48d7b95a3fd477ff9a7348c`
- Git dirty: `True`

## Manifest seals and identities

| split | rows | canonical IDs | image SHA-256s | manifest SHA-256 |
|---|---:|---:|---:|---|
| train | 969 | 969 | 969 | `928b2507a22862073c99ca8e58d6d92cc17e1c86f974222bd20de8844ef513dc` |
| validation | 232 | 232 | 232 | `73fae346943a30ba96f7f5c8886bb16e95cea07cc61302a44d9978c95c5c5778` |
| test | 867 | 867 | 867 | `7d54e082def62b0a4c68d555b058bbc8f0e3faa99b17d35841c4af03c1adcfdd` |

## Pairwise leakage checks

| left | right | identity | overlap | example |
|---|---|---|---:|---|
| train | validation | canonical ID | 0 |  |
| train | validation | image SHA-256 | 0 |  |
| train | test | canonical ID | 0 |  |
| train | test | image SHA-256 | 0 |  |
| validation | test | canonical ID | 0 |  |
| validation | test | image SHA-256 | 0 |  |

Preflight performs no selection, training, inference, or model evaluation.
A PASS only allows a separately authorized downstream protocol to continue.

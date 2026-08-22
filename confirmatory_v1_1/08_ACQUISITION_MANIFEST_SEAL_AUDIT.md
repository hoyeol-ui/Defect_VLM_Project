# Acquisition Manifest Seal Audit

- Result: **PASS**
- shared initial manifests: 20
- Random query events: 100
- DINO query events: 100
- unique cumulative selected states: 220
- DINO cache A/B full-trajectory comparisons: 20/20 identical
- GT-bearing fields used during selection: 0
- trajectory ledger SHA-256: `bbdd09a645cc84849cbe88662abf61ec1e7e2937f382c071840dfc432a6824ea`

Every query has the frozen per-round count, every cumulative state has the frozen
budget count, IDs are unique and contained in the acquisition universe, and the
shared R0 state is identical across Random/DINO. Training seeds were not accepted
by the acquisition builder. The canonical manifests are sealed under
`manifest_seals/` and are ready for both training seeds.


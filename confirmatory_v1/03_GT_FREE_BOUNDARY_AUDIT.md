# GT-Free Boundary Audit

- Prospective implementation 판정: **PASS**
- Historical V7 판정: **CONDITIONAL** 유지
- 범위: acquisition/selector boundary만 평가하며 post-seal GT audit와 detector training GT 사용은 별도 stage다.

## 1. Boundary architecture

```text
private source registry (path; selector DENY)
  -> embedding adapter (raw RGB only; no XML)
  -> sealed [sample_id, 384-D embedding] cache
  -> selector core (opaque ID + membership + seed/round)
  -> query/cumulative manifest SHA seal
  -> post-seal GT join for training and secondary audit
```

Selector-facing acquisition manifest의 column은 `sample_id` 하나다. Source path와 annotation path가 있는 private registry는 embedding/build/training adapter만 사용하며 selector 함수에는 전달할 수 없다.

## 2. Allow / deny

| ALLOW | DENY |
|---|---|
| embedding adapter 내부 raw RGB | true class, GT bbox |
| sealed DINO embedding | rare/frequent, anomaly, bbox size |
| opaque canonical sample ID | filename, parent directory |
| labeled/unlabeled/already-selected membership | development/final metric |
| acquisition seed와 round | future-round state, training seed |

Selector record에 allow-list 외 column이 하나라도 존재하면 hard fail한다. `class`, `label`, `bbox`, `path`, `filename`, `dev`, `test`, `metric` 계열 field가 포함된 CPU fixture는 실제로 실패했다.

## 3. Identity audit

- GC10 raw JPEG rows: 2,312
- missing XML: 6 rows 제외
- empty/invalid GT: 2 rows 제외
- 동일 RGB 픽셀 + 동등 GT alias: 12 rows collapse
- NEU 동일 이미지 + 서로 다른 bbox: 2 rows 모두 제외
- curated unique: GC10 2,292, NEU 1,798
- curated cross-split/cross-dataset duplicate canonical ID: 0

NEU class-bearing filename과 GC10 numeric parent directory는 selector identity/order에서 제거됐다. Representative 선택이 필요한 동등 duplicate는 class-neutral raw SHA ascending으로 정한다.

## 4. Static and unit audit

| Check | Status |
|---|---|
| Selector core가 XML/annotation을 import하지 않음 | PASS |
| Acquisition API signature에 `training_seed` 없음 | PASS |
| GT/label-bearing selector field hard fail | PASS |
| non-SHA semantic sample ID hard fail | PASS |
| split overlap/duplicate canonical SHA hard fail | PASS |
| missing/extra/NaN/zero embedding hard fail | PASS |
| final split unapproved access hard fail | PASS |

## 5. 왜 historical과 판정이 다른가

Historical V7은 `resolved path::raw image SHA`를 canonical ID와 tie-break에 사용해 NEU class filename이 간접 order에 들어갈 수 있었다. FrozenDINO-FF-v1은 canonical RGB digest만 사용하고 selector input schema를 allow-list로 제한한다. 이 변경은 historical result chasing이 아니라 새 결과 전에 수행한 prospective leakage cleaning이다.

## 6. 잔여 조건

PASS는 code/input architecture 판정이다. Phase 2B cache build 때 ordered manifest count, artifact/code hash, cache hash가 맞지 않으면 이 판정은 즉시 무효가 되고 STOP한다.


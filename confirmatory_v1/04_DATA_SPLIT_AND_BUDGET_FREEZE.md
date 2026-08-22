# Data Split and Budget Freeze

- 최종 split: **prospective common 70/10/20 v1**
- 용어: **prospectively locked confirmatory evaluation split**
- 금지 용어: completely untouched test
- split seed: `20260822`

## 1. Historical audit와 선택

Historical V7은 99-image mixed priority pool, development 180, 별도 final manifest 계보였고, GC10 q20 confirmation은 development 232/final 224 및 다른 acquisition construction을 사용했다. 두 dataset의 acquisition universe, split ratio와 budget이 같지 않아 controlled comparison의 공통 기반으로 유지하기 어렵다.

따라서 raw public data를 outcome-independent identity cleanup한 뒤 동일 70/10/20 규칙을 한 번 적용했다. Public GT가 과거 분석에 쓰였으므로 final을 untouched라고 부르지 않는다. 새 final membership은 지금부터 prospectively lock한다.

## 2. Split unit과 algorithm

- unit: canonical RGB pixel identity group
- ratios: acquisition 70%, development 10%, final 20%
- global size rounding: largest remainder; tie order acquisition → development → final
- allocation: deterministic rare-label-first iterative multilabel stratification
- selector는 split 후 acquisition membership만 받음
- 모든 class가 development/final에서 image support `>0`이어야 함

## 3. Final cardinality

| Dataset | raw JPEG | curated unique | acquisition | development | final |
|---|---:|---:|---:|---:|---:|
| GC10-DET | 2,312 | 2,292 | 1,605 | 229 | 458 |
| NEU-DET | 1,800 | 1,798 | 1,259 | 180 | 359 |

GC10 rare endpoint support는 다음과 같다.

| class | curated | acquisition | development | final |
|---|---:|---:|---:|---:|
| 8 rolled_pit | 46 | 32 | 5 | 9 |
| 9 crease | 53 | 37 | 5 | 11 |
| 10 waist_folding | 140 | 98 | 14 | 28 |

NEU의 각 class development image support는 30–43, final은 60–75다. 최종 zero-support cell은 0개다. 전체 표는 [`class_support.csv`](./data_freeze/class_support.csv)에 있다.

## 4. Budget rule

Initial과 query count는 fixed original acquisition universe에 percentage를 곱한 뒤 decimal round-half-up으로 한 번 계산한다. Query는 remaining pool의 5%가 아니다.

| Dataset | acquisition N | initial 10% | query 5% | six cumulative budgets |
|---|---:|---:|---:|---|
| GC10-DET | 1,605 | 161 | 80 | 161, 241, 321, 401, 481, 561 |
| NEU-DET | 1,259 | 126 | 63 | 126, 189, 252, 315, 378, 441 |

두 final labeled fractions는 각각 0.34953과 0.35028로 같은 약 35% span을 평가한다. Round0를 포함해 six points를 모두 유지한다.

## 5. Hashes와 lock

- split assignment SHA-256: `e4e0ad0f56efa28ce59b3710b852737b02fc774cd88d7121e98f477bd618133d`
- GC10 blind acquisition file SHA-256: `da1aeaf9987437c922c49b5d1ab127f4d57000587bd6fb7d4d089c3f3c21db38`
- NEU blind acquisition file SHA-256: `2398ec523cb4cd878b6f23a515a0ee863ea81cacdd028703cc4414c37019e069`
- data-freeze ledger SHA-256: `5975afafc5c871bee2aa2256303cfd63f402e4be6f40d70d45b7231114a2a991`
- builder repeat run byte-identical: PASS

Final manifest는 생성·hash lock만 했고 evaluation에 사용하지 않았다.


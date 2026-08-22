# Seed and Manifest Freeze

- acquisition seeds: dataset당 10개
- training seeds: acquisition당 2개
- seed source: result-independent SHA-256 namespace derivation
- seed manifest SHA-256: `31372e88b78478fc3400ed363030681e48c0468b8bc599af7d86fbc33defd93d`

## 1. Seed generation

각 seed는 다음 namespace string의 SHA-256 첫 4 bytes를 positive signed-32-bit 범위로 변환했다.

```text
confirmatory-random-dino-v1 | dataset | purpose | acquisition_index
```

Historical favorable/unfavorable seed를 선별 재사용하지 않았고 새 outcome을 보지 않았다.

## 2. Acquisition seeds

| Dataset | frozen list |
|---|---|
| GC10-DET | 118406135, 1544253486, 1207468566, 1047824994, 489338187, 1247201608, 1159501161, 1248489736, 1498171501, 1117821898 |
| NEU-DET | 556094945, 1401936841, 308869329, 1765023412, 1863187687, 1566547639, 1039657086, 1644540782, 1157893727, 1944238630 |

Training A/B mapping 전체는 [`seed_manifest.csv`](./data_freeze/seed_manifest.csv)에 봉인했다. 같은 acquisition index의 A와 B를 Random/DINO에 paired하게 그대로 사용한다.

## 3. Shared initial과 Random

- canonical pool order: `sample_id` ascending
- engine: NumPy 2.4.6 `Generator(PCG64)` + `SeedSequence`
- initial entropy: protocol namespace, dataset code, acquisition seed, round 0, purpose `INIT`
- query entropy: protocol namespace, dataset code, acquisition seed, round, purpose `QUER`
- sampling: remaining pool에서 uniform without replacement
- initial manifest: 같은 dataset/acquisition에서 두 strategy가 하나를 공유
- training seed: initial/query RNG에 전달 불가

## 4. Trajectory materialization

```text
dataset + acquisition seed
  -> shared initial manifest
  -> Random R1...R5
  -> Frozen DINO R1...R5
each cumulative manifest
  -> training seed A
  -> training seed B
```

Trajectory는 learner training 전에 한 번만 만들고 cumulative `sample_id`를 ascending sort하여 LF-delimited bytes로 SHA-256 seal한다. 두 training seeds는 같은 manifest SHA를 참조한다. Training seed별 acquisition 재실행, embedding rebuild, ordering 변경은 hard fail이다.

## 5. Counts와 integrity

- shared initial manifests: 20
- post-Round0 query manifests: 200
- unique cumulative states: 220
- physical strategy×budget manifests를 만들 경우 duplicated Round0 contents의 SHA는 반드시 동일
- duplicate ID, split overlap, query reuse, budget count mismatch는 STOP

CPU preflight에서 same seed determinism, different seed difference, shared initial, training-seed isolation과 cumulative counts가 모두 PASS했다.


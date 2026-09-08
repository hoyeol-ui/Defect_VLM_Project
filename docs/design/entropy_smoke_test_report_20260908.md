# Entropy R0 smoke test report 20260908

**Smoke: PASS WITH WARNINGS. Official 200-cell readiness: BLOCKED (B1/B2; see pre-flight audit).**

각 데이터셋의 acquisition index 0, learner A, validation-selected best.pt로 전체 U0를 두 개의 독립 프로세스에서 추론했다. 공식 학습·GT/AP 분석·locked-test evaluation·threshold sweep은 0회다. 첫 canonical ID 1장씩 추가 계측하여 실제 predictor kwargs/FP32를 확인했고 full smoke 점수와 일치했다.

| Metric | GC10-DET | NEU-DET |
|---|---:|---:|
| Unlabeled pool | 1444 | 1133 |
| Zero detection images | 199 | 63 |
| Zero detection rate | 0.13781163434903046 | 0.05560458958517211 |
| Unique entropy scores | 1246 | 1071 |
| Exact tie excess (N - unique) | 198 | 62 |
| Images in tied groups | 199 | 63 |
| Exact equal-score pairs | 19701 | 1953 |
| Selected query size | 80 | 63 |
| NaN/inf | 0 | 0 |
| Spearman entropy vs detection count | 0.9465703557308419 | 0.9737869484306498 |
| n_detections min | 0 | 0 |
| n_detections median | 1.0 | 2 |
| n_detections mean | 1.6710526315789473 | 2.731685789938217 |
| n_detections max | 11 | 12 |
| entropy_sum min | 0.0 | 0.0 |
| entropy_sum median | 0.6753040127235079 | 1.2995613920322593 |
| entropy_sum mean | 0.9478369066261423 | 1.5678917695182977 |
| entropy_sum max | 7.015130879906412 | 7.541349812573278 |

## Critical observations

GC10 무검출률은 13.7812%, NEU는 5.5605%다. 동점은 모두 no-detection=0 그룹에서 발생했다. 양쪽 모두 양수 score는 서로 달라 전체 순위가 상수로 붕괴한 상태는 아니다. 다만 Spearman ρ는 각각 0.94657와 0.97379로 매우 높다. 이 smoke에서는 sum entropy 순위가 detection count에 강하게 연관된다. 이를 충분한 uncertainty 독립 신호가 입증되었다고 해석하면 안 된다. 상관만으로 완전한 count-only selector라고 단정하지도 않는다. 사후 cutoff를 만들어 method를 고르거나 conf/aggregation을 튜닝하지 않았다.

같은 환경·checkpoint·manifest·batch=1·640 square·FP32에서 두 full runs의 score table, 전체 ranking, selected IDs hash가 모두 동일하다. 다른 GPU/버전/batch의 bitwise 재현성은 검증하지 않았다. timestamp-bearing CSV는 실행 시각 때문에 파일 해시가 달라지는 것이 정상이다.

## Pool integrity

| Check | GC10-DET | NEU-DET |
|---|---|---|
| query_size_correct | True | True |
| labeled_selected_count | 0 | 0 |
| validation_selected_count | 0 | 0 |
| test_selected_count | 0 | 0 |
| split_hash_unchanged | True | True |
| image_hashes_unchanged | True | True |
| pixel_identity_count_verified | 1444 | 1133 |
| next_pools_disjoint | True | True |
| next_pools_union_is_train | True | True |
| duplicate_ids | 0 | 0 |

GC10 L1=241 / U1=1364, NEU L1=189 / U1=1070. Query는 U0의 부분집합; L1∩U1=∅, L1∪U1=fixed train pool. Validation/test membership은 integrity wrapper에서만 확인했다. selector는 train image ID/path와 labeled/train membership만 받았다.

## Exact deterministic hashes

### GC10-DET

| Artifact | SHA-256 (repeat 1 = repeat 2) |
|---|---|
| score_table | `b9f8b1aa7cbe0b1c84e4799f6b71d97b2475d41982a622523214735fd9a7b433` |
| ranking | `fb496a242041678214c47e93263558012a93ae220a906e2f8fd510f3ae591485` |
| selected_ids | `993330564d42423c28d88f0f6a7f972c5fc072bb64c3c5b3b8dd8f4633172ff8` |
| scores.json file | `9fd9a36647ec19c61223977e39bb7a4d1ad551880e12f5cea7a48b4f320e725d` |

Repeat elapsed seconds: 58.020, 52.244.

Canonical IDs only; no GT inspection:

top_ids:
```text
a06a176f92440721963e2e8923ad4b8e48bb721f26e5872c92976b0776f83451
6829122a30194158019076628e0d845a53b8c290d8f024471e383d401cc36b51
6358f24343bbebd7904cb76c066e50a22bd7918c6940838933e4da7e125ea747
```

middle_ids:
```text
e38776db2b6cc6855d67c090618268170ac24f835fa158c4bf8404155b00727c
8ec6a23c53a0e5b03d43a7a3bd775acd0fa9ea354d78daca91b6e69f37f532e8
94a2f37e6cd8481909c15db3ad018e96d9f16e5ce2f1a060ccbca7a0434ca738
```

bottom_ids:
```text
f6d0a68c64e7b4a7d4aa6abd733c5fb9145f69a030a64d2692939021b1f52867
fb2c1e2c10b3e6d8510237eeeb1f33487c0bc87d5982aef73aa3ddd377f9cd2a
fe701e5e914503167d6e462c924486a3310fa20536b70afcd4b0d2ed85402d9f
```

### NEU-DET

| Artifact | SHA-256 (repeat 1 = repeat 2) |
|---|---|
| score_table | `98c653d671542dba1492a7d0242c895c6b7bd2c460e6159cced1e0da77bc6e67` |
| ranking | `a8970fdb6a0a1cd6a611634114cfd1bb0db2f7fd33888f359e9a9649b37d7187` |
| selected_ids | `326f7137cdc96f331f5b499d3e2eb007d965ab6dd9f36ba627c3c78b03b88338` |
| scores.json file | `c5e7634e71b391fb869054e6ccb35981cbb213a08a92c961ae672ba24af35d27` |

Repeat elapsed seconds: 22.221, 21.395.

Canonical IDs only; no GT inspection:

top_ids:
```text
4c3e9dbfee10b7fbadbc40ed068173b6cb60e0cda7f121c7cc6445f19f841ee3
f2005ee4952282bc52b5f9312e0cdd310f1e23277c8262dc04fb248f61cfa436
411924653bc99955861ffa237bbe691f82cfaa554c9dada06b80971b149481cf
```

middle_ids:
```text
531bd06f4bc9e373c4dab171e996c33ed31442cfd33938941a1903acce5780bc
51ae914f18e1af0d3ab33257b4813b7f5ef287ee8c72560fe7c371fd7c59ea6d
e996328d1f2efa68b09f67afd6ffe94078e318c7b365eba58c786cf43b5f0ad3
```

bottom_ids:
```text
f51d2c411f9e9b7884e93efcc55a87e6f7e03cd4b4e40991119f133f3180b3cf
f90a18111a4a52c247482ab306f38a9ecaeb397a7aee1e5efc38fa50e0535084
f99b45d8bcc6f2701116427ef8264f5ef8e81350b55d809db30e62919dded283
```

## Reproduction commands

```powershell
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" scripts/entropy_preflight/run_smoke.py --dataset GC10-DET --repeat 1
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" scripts/entropy_preflight/run_smoke.py --dataset GC10-DET --repeat 2
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" scripts/entropy_preflight/run_smoke.py --dataset NEU-DET --repeat 1
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" scripts/entropy_preflight/run_smoke.py --dataset NEU-DET --repeat 2
```

기존 repeat 출력 폴더가 있으면 runner가 덮어쓰기를 거부한다. 이미 완료된 결과를 재사용하며 위 명령을 현재 폴더에서 그대로 재실행하지 않는다.

## Artifact layout

`reports/entropy_preflight_20260908/{dataset}/repeat_{1,2}/` 아래에 scores.json, scores_with_provenance.csv, summary.json, query_R1.txt, labeled_R1.txt, unlabeled_R1.txt를 저장했다. 데이터셋별 effective_predict_args.json은 전체 resolved args와 실제 precision을 보존한다. 선택 이미지를 class distribution/defect hit rate로 해석하지 않았고 성능 결과 파일을 열지 않았다.

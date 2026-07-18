# Metadata Feasibility Audit Protocol

## 목적

GC10-DET, MPDD, VisA의 기존 acquisition 데이터에 production lot, capture
session 또는 source를 나타내는 genuine metadata가 존재하는지 확인한다.
목적은 independent-pool 실험을 통과시키는 것이 아니라, 해당 실험이
통계적으로 식별 가능한지 실행 전에 판정하는 것이다.

## 허용 입력

- 각 데이터셋의 `*_acquisition_pool_gt_audit.csv`
- 기존 frozen selection record
- acquisition manifest가 가리키는 이미지의 embedded EXIF

GT/anomaly/rare annotation은 group을 정의하는 데 사용하지 않고, metadata로
group을 정의한 뒤 prevalence와 교락을 post-hoc으로 감사하는 데만 사용한다.

## 금지 입력 및 작업

- `final` 또는 `locked` 경로와 split
- detector/anomaly model 학습 및 inference
- embedding 추출
- selector 또는 score 재실행
- local-feature/FN 구현
- category, anomaly label, annotation row, 숫자 파일명을 production lot으로 승격
- 결과를 보고 session threshold 또는 gate 변경

## 사전 고정 independent-pool gate

모든 조건을 충족해야 한다.

1. Source가 제공하거나 명시적으로 문서화한 target-blind genuine group ID
2. `N >= 40`인 group 최소 20개 (`initial 20 + query 20`)
3. `N >= 40`이면서 target/non-target이 혼합된 group 최소 20개
4. Usable group 간 target prevalence range 0.10 이상
5. 최소 2개 category가 각각 2개 이상의 usable group에 반복 출현
6. 동일 이미지/duplicate가 서로 다른 group에 걸치지 않음

EXIF calendar day/hour, filename prefix, object category는 이 gate의 genuine
group ID로 인정하지 않는다. 이들은 capture-session confound 또는 descriptive
sensitivity audit에만 사용한다.

## 사전 고정 group 정의

| dataset | grouping | provenance | 허용 해석 |
|---|---|---|---|
| GC10-DET | filename prefix before final underscore token | filename sequence proxy | descriptive sequence sensitivity |
| GC10-DET | EXIF calendar day/hour | edited-time proxy when ACDSee/no camera identity | weak time sensitivity |
| MPDD | EXIF calendar day/hour | camera datetime proxy | capture-session confound audit |
| MPDD | official train/test origin | protocol/label-confounded field | confound diagnostic only |
| VisA | object category / source annotation CSV | category/domain only | category robustness only |

## 출력

기본 실행 경로:

`runs/metadata_feasibility_audit_20260718/metadata_feasibility_main/`

- `metadata_field_inventory.csv`
- `exif_metadata_audit.csv`
- `capture_group_feasibility.csv`
- `metadata_target_confounding.csv`
- `selection_session_concentration.csv`
- `selection_session_concentration_summary.csv`
- `config.json`
- `audit_execution_log.txt`
- `generated_file_manifest.csv`

결정문:

`docs/metadata_feasibility_decision_20260718.md`

## 실행 순서

먼저 경로와 schema만 검증한다.

```powershell
.\.venv\Scripts\python.exe scripts\04_dcal_xai\audit_dataset_metadata_feasibility.py --dry-run
```

Dry-run이 통과한 뒤 한 번만 본 감사를 실행한다.

```powershell
.\.venv\Scripts\python.exe scripts\04_dcal_xai\audit_dataset_metadata_feasibility.py
```

기존 출력이 있을 경우 자동 덮어쓰지 않는다. 재실행이 필요한 경우 먼저
기존 결과를 보존하고 새로운 `--output-dir`와 `--decision` 경로를 사용한다.

## 가능한 최종 판정

- `INDEPENDENT_POOL_PROTOCOL_ELIGIBLE`
- `CAPTURE_SESSION_CONFOUND_AUDIT_ONLY`
- `FILENAME_SEQUENCE_SENSITIVITY_ONLY`
- `CATEGORY_ROBUSTNESS_ONLY`

이 감사 결과만으로 새 selector, detector 학습, FN screen 또는 final-test
접근은 승인되지 않는다.

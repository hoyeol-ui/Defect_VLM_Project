# GC10 실제 파이프라인 preflight 적용 및 결과 재검증 리뷰

- 작성일: 2026-08-01
- 대상 연구 실행: `gc10_dev_confirm_5acq_3train_20260715`
- 감사 결론: **PASS_WITH_REPRODUCIBILITY_CAVEATS**
- 기존 frozen research gate: **FAIL / STOP 유지**
- 이번 작업에서 새 학습·추론·GPU 실행: **없음**

## 1. 결론부터 쉽게 설명하면

이번 검증에서 현재 보관된 GC10 결과가 파일 혼동, 이미지 바이트 변경,
라벨 변환 오류, checkpoint 연결 오류, 단순 통계 재계산 오류 때문에 만들어졌다는
증거는 발견되지 않았다. 과거 결과 CSV와 독립 재계산 결과도 허용 오차
`1e-12` 안에서 일치했다.

그러나 이것을 “어떤 환경에서 다시 실행해도 완전히 같은 결과가 나온다” 또는
“다른 데이터 요청·checkpoint·모델을 써도 결론이 같다”는 뜻으로 해석하면 안 된다.
과거 runner에는 Git commit, runner SHA, 모든 입력 manifest seal, Python/PyTorch/CUDA/
Ultralytics 버전, 완전한 환경 lock이 남아 있지 않다. 따라서 이번 감사가 확인한 것은
**현재 남아 있는 산출물의 내부 일관성**이고, 역사적 실행의 bit-for-bit 재실행 가능성은
그보다 약하다.

연구 결과 자체는 다음처럼 읽는 것이 정확하다.

- FrozenDINO 선택은 Random보다 development mAP50-95가 평균 `+0.0173784` 높았다.
- 5개 acquisition seed 중 4개에서 양수였고, 올바른 acquisition 단위 bootstrap 95% CI도
  `[+0.0005803, +0.0363322]`로 0보다 높았다.
- 반면 희소 결함군 rare macro AP50-95는 평균 `-0.0198767`로 나빠졌고,
  사전 정의 조건 `+0.020 이상`을 충족하지 못했다.
- class 9 개선 조건도 실패했다.
- 그러므로 “전체 평균에서는 유망한 신호가 있었지만 희소 결함 안전 조건을 통과하지
  못했으므로 다음 단계로 진행하지 않는다”는 기존 `FAIL / STOP` 판정이 맞다.

즉, **FAIL은 실행 오류를 뜻하지 않는다. 사전에 정한 안전·유효성 조건 중 일부가 실제로
실패했다는 뜻**이다.

## 2. 이번에 실제로 적용한 범위

### 2.1 실제 GC10 입력을 strict preflight에 연결

다음 명시적 입력만 이용해 forward-only manifest를 만들었다.

- acquisition pool GT audit
- development evaluation manifest
- blind acquisition pool
- frozen selection records
- 과거 10개 detector training manifest
- 로컬 GC10 원본 이미지 바이트

`latest` 검색이나 최근 수정 시각 탐색을 사용하지 않았으며, 파일 경로와 SHA-256이
고정된 strict config를 만들었다.

생성된 preflight 경계는 다음과 같다.

| 경계 | 행 수 | 의미 |
|---|---:|---|
| train | 969 | frozen 선택 234장과 같은 hard group의 모든 acquisition 이미지 |
| validation | 232 | 기존 development 평가 이미지 |
| test | 867 | 선택 hard group 밖의 미사용 acquisition audit holdout |

여기서 `test.csv`는 schema가 요구하는 세 번째 격리 경계일 뿐, 잠긴 공식 final test가
아니다. 공식/final 평가 결과도 계산하지 않았다.

또한 969행 `train.csv`는 detector 한 번의 40장 학습 manifest가 아니다. 10개의 과거
학습셋은 `20 initial + 20 query = 40장`씩이며, forward builder는 이 10개 각각에 대해
과거 ID 순서, ID 집합, 이미지 SHA 순서, 이미지 SHA 집합을 따로 비교했다.

### 2.2 preflight가 검사하는 항목

- manifest 파일 자체의 SHA-256 seal
- `cid:v1` canonical ID의 독립 재계산
- manifest 내부 canonical ID 중복
- train/validation/test 사이 canonical ID 중복
- 이름만 바뀐 동일 이미지까지 잡는 raw image SHA-256 중복
- `latest`, glob, 암묵적 run 탐색 표현
- 절대경로가 섞인 sample relative path
- 필수 split 누락

PASS가 나더라도 selection이나 training을 자동 승인하지 않는다. 생성된 report의
`selection_authorized`와 `training_authorized`는 둘 다 `false`이며, 별도의 연구 protocol이
명시적으로 승인해야 한다.

## 3. 검증 결과

### 3.1 데이터와 split 무결성

| 검사 | 확인 수 | 불일치/누출 | 판정 |
|---|---:|---:|---|
| 원본 acquisition + development 이미지 SHA-256 | 2,068 | 0 | PASS |
| 현재 YOLO cache 이미지와 기록 SHA 비교 | 2,068 | 0 | PASS |
| bbox CSV에서 YOLO label을 독립 재생성해 byte 비교 | 2,068 | 0 | PASS |
| development cache manifest ID 및 순서 | 232 | 0 | PASS |
| forward/legacy 40장 선택셋 | 10 | 0 | PASS |
| train-validation canonical ID overlap | - | 0 | PASS |
| train-test canonical ID overlap | - | 0 | PASS |
| validation-test canonical ID overlap | - | 0 | PASS |
| 세 경계 사이 raw image SHA-256 overlap | - | 0 | PASS |
| acquisition-development pHash Hamming distance <= 2 | - | 0 pair | PASS |

이 결과는 현재 데이터 파일, cache 이미지, 변환 라벨 사이에 확인 가능한 내용 불일치가
없다는 뜻이다. pHash 검사는 근접 중복 후보를 찾기 위한 보조 검사이며, 이번에는
Hamming distance 2 이하 후보도 없었다.

### 3.2 학습 산출물 계보

| 검사 | 확인 수 | 불일치 | 판정 |
|---|---:|---:|---|
| base model SHA-256 | 1 | 0 | PASS |
| 기록된 `last.pt` checkpoint SHA-256 | 30 | 0 | PASS |
| `args.yaml` 핵심 학습 인자 | 30 | 0 | PASS |
| training YAML의 train/val/name 연결 | 10 | 0 | PASS |
| 결과 행의 `final_test_used` | 30 | 모두 false | PASS |

확인한 핵심 인자는 epochs 75, image size 640, batch 16, workers 0,
training seed 42/43/44, deterministic true, pretrained true, validation during
training false, optimizer `auto`이다.

30개 run 모두 `best.pt`와 `last.pt`의 SHA-256이 달랐다. 과거 protocol은 명시적으로
`last.pt`를 development에서 평가했으므로 이것은 protocol 위반이 아니다. 다만
`best.pt`를 선택했다면 숫자가 달라졌을 가능성이 높고, 어느 방향으로 얼마나 달라지는지는
새 추론 없이는 알 수 없다.

### 3.3 지표 독립 재계산

aggregate 30행과 per-class 300행을 원자료에서 다시 집계했다. 보관된 metric summary,
class-group macro summary, gate 표와 모두 일치했다.

| 지표 | Random 평균 | FrozenDINO 평균 | paired 차이 | 5-acquisition 95% CI |
|---|---:|---:|---:|---:|
| mAP50-95 | 0.1242354 | 0.1416138 | +0.0173784 | [+0.0005803, +0.0363322] |
| mAP50 | 0.3061127 | 0.3233134 | +0.0172007 | [-0.0123112, +0.0522350] |
| precision | 0.4865934 | 0.4419228 | -0.0446706 | [-0.1106054, +0.0301245] |
| recall | 0.3168772 | 0.3522586 | +0.0353814 | [+0.0172797, +0.0550292] |

핵심 mAP50-95의 acquisition seed별 차이는 다음과 같다.

| acquisition seed | FrozenDINO - Random |
|---:|---:|
| 0 | -0.0107441 |
| 1 | +0.0521949 |
| 2 | +0.0081299 |
| 3 | +0.0203648 |
| 4 | +0.0169465 |

training seed별 평균 차이도 모두 양수였다.

| training seed | FrozenDINO - Random 평균 |
|---:|---:|
| 42 | +0.0212178 |
| 43 | +0.0139686 |
| 44 | +0.0169488 |

따라서 mAP 개선이 특정 training seed 하나에서만 생긴 현상으로 보이지는 않는다.
하지만 acquisition seed 1의 효과가 매우 크고 seed 0은 음수이므로 선택되는 데이터의
구성에는 분명히 민감하다.

### 3.4 gate를 왜 통과하지 못했는가

| gate | 결과 |
|---|---|
| 평균 mAP50-95 gain >= +0.010 | PASS |
| acquisition seed win >= 3/5 | PASS |
| acquisition 단위 bootstrap CI lower > 0 | PASS |
| rare macro AP50-95 gain >= +0.020 | **FAIL** |
| class 8 AP50-95 gain > 0 | PASS |
| class 9 AP50-95 gain > 0 | **FAIL** |
| recall non-inferiority >= -0.020 | PASS |
| frequent macro non-inferiority >= -0.020 | PASS |

frequent macro는 `+0.0333449`였지만 rare macro는 `-0.0198767`이었다. rare macro의
acquisition seed별 차이는 `-0.0350318`, `+0.0026153`, `-0.0096873`, `+0.0006528`,
`-0.0579327`이었다. 하나의 acquisition seed를 차례로 제외해도 rare macro 평균은
`-0.0254998`부터 `-0.0103627` 사이였다. 어느 경우도 요구치 `+0.020`에 근접하지 않는다.

따라서 현재 5개 seed 중 하나가 우연히 결과를 망쳤다고 설명해서 gate를 뒤집기는 어렵다.
**전체 평균 개선 신호와 희소군 성능 저하는 동시에 참**이다.

## 4. “그때 다른 변경이 있었다면 결과가 달라졌을까?”에 대한 답

### 4.1 확실히 입력을 바꾸는 변경

초기 20장은 정렬된 blind manifest의 row 위치에서 seed로 추출됐다. 원본 blind
manifest는 이미 정렬된 상태가 아니었다. canonical sort를 빼고 같은 seed와 같은 row
위치를 쓰는 반사실 비교에서, 실제 초기 20장과의 교집합은 5개 seed 모두 `0/20`이었다.

그러므로 당시 canonical sort가 빠졌거나 입력 manifest의 행 순서가 바뀌었다면 학습
요청 데이터 자체가 완전히 달라졌을 것이다. 이 경우 detector 결과도 달라질 가능성이
매우 높지만, 새 학습을 하지 않았으므로 방향은 단정할 수 없다. 실제 과거 selection
audit와 detector confirmation은 둘 다 canonical sort를 사용했기 때문에 현재 보관된
실험 내부에는 서로 다른 정렬을 사용한 오류가 없다.

### 4.2 acquisition seed 구성이 달랐다면

한 acquisition seed씩 제외한 mAP50-95 평균 차이 범위는 `+0.0086743`부터
`+0.0244090`이다. seed 1을 제외하면 평균이 사전 기준 `+0.010` 아래로 내려간다.
즉, “평균 성능 개선의 크기”는 어떤 acquisition realization을 뽑았는지에 민감하다.

다만 rare macro는 같은 leave-one-out 검사에서 계속 음수였다. 현재 표본 안에서는
희소군 안전 실패가 전체 mAP threshold보다 더 안정적인 결론이다.

### 4.3 통계 단위를 다르게 잡았다면

독립 실험 단위는 training seed 15개가 아니라 acquisition seed 5개다. 같은 선택셋을
여러 training seed로 학습한 값은 selection 전략 관점에서 완전히 독립인 15개 표본이
아니다.

- 올바른 5-acquisition bootstrap CI: `[+0.0005803, +0.0363322]`
- 15개 training run을 독립으로 잘못 취급한 진단용 CI: `[+0.0061881, +0.0289405]`

잘못된 방식은 CI를 더 좁게 만들어 확실성을 과장한다. 저장된 연구 결과는 올바른
5-acquisition 단위를 사용했다.

### 4.4 `best.pt`를 썼다면

모든 30개 run에서 `best.pt`와 `last.pt`가 서로 다른 파일이었다. 따라서 checkpoint
정책을 바꾸면 metric이 달라질 수 있다. 하지만 과거 protocol의 평가 대상은 `last.pt`로
명시돼 있으므로 현재 결과를 오류라고 부를 수 없다. `best.pt` 결과는 새로 승인된 별도
민감도 실험으로만 확인할 수 있다.

### 4.5 optimizer·라이브러리·GPU 환경이 달랐다면

`optimizer="auto"`는 Ultralytics 버전에 따라 실제 선택 또는 동작이 달라질 수 있다.
과거 기록에는 정확한 Ultralytics, PyTorch, CUDA, driver 버전이 없다. deterministic
flag가 true여도 모든 GPU 연산과 라이브러리 조합에서 bitwise 동일성을 보장하지는 않는다.

이번 독립 감사 환경은 Python 3.11.9, NumPy 2.4.6, pandas 3.0.3, PyYAML 6.0.3이었다.
현재 `requirements-analysis.txt`의 pandas 범위는 `<3`인데 로컬 `.venv`는 3.0.3이다.
그 환경에서도 보관된 통계가 정확히 재현됐으므로 이번 집계 결론이 pandas 3 때문에
달라졌다는 증거는 없다. 다만 최종 재현 환경을 정리할 때는 요구 버전과 실제 환경을
일치시키고 lock 파일을 남겨야 한다.

### 4.6 데이터 수집·split·라벨 요청이 달랐다면

다음 변경은 모두 결과를 바꿀 수 있지만, 기존 연구와 같은 질문을 유지하지 않을 수도 있다.

- acquisition/development의 생산 batch 또는 hard group 구성 변경
- 초기 20장·query 20장의 budget 변경
- 희소 클래스 oversampling 또는 class-aware query 추가
- bbox annotation 정책·경계 기준 변경
- 다른 detector/backbone, image size, augmentation, epoch 또는 early stopping
- final test를 development tuning에 노출

특히 희소군을 더 많이 요청하는 class-aware 전략은 rare 결과를 개선할 가능성이 있지만,
현재 FrozenDINO와 동일한 selector라고 볼 수 없고 결과 방향도 새 실험 전에는 확정할 수
없다. final test를 사용해 전략을 고르는 것은 성능을 좋아 보이게 만들 수 있지만 연구
타당성을 훼손하므로 허용되는 대안이 아니다.

## 5. 발견한 실제 코드 위험과 현재 결과에 미친 영향

### 5.1 결과 cache가 존재 여부만 확인

과거 `train_or_recover()`는 `aggregate.json`과 `per_class.json`이 있으면 config,
checkpoint, manifest seal을 다시 비교하지 않고 바로 반환한다. 잘못된 cache가 섞일 수
있는 설계 위험이다.

이번 감사에서는 30개 checkpoint SHA, 30개 결과 행, 300개 per-class 행을 독립적으로
연결·재계산했고 불일치가 0이었다. 따라서 **위험한 구조는 맞지만 현재 보관 결과가 그
위험으로 오염됐다는 증거는 없었다.**

### 5.2 이미지 cache가 기존 target을 신뢰

과거 `link_or_copy()`는 target이 이미 있으면 source와 내용 비교 없이 `existing`으로
처리한다. source가 바뀌었거나 cache가 오염됐다면 잘못된 이미지가 남을 수 있다.

이번에 2,068개 cache 이미지 SHA를 source manifest의 SHA와 모두 비교했고 불일치가
0이었다. 현재 cache는 이 위험 때문에 잘못됐다고 볼 근거가 없다.

### 5.3 machine-specific 절대경로

training TXT/YAML과 과거 config에는 로컬 절대경로가 포함돼 있다. 같은 컴퓨터에서는
작동했지만 다른 Windows 사용자나 Linux CI에서는 그대로 재사용할 수 없다. 새 strict
config와 새 report는 repository-relative 경로를 사용하도록 만들었다.

### 5.4 initial set을 별도 파일로 freeze하지 않음

초기 20장을 selection record 자체에 저장하지 않고 sorted blind manifest의 row 위치로
복원한다. 지금은 selection audit와 detector runner의 정렬 및 10개 legacy manifest가
모두 일치해 실제 오류는 없다. 하지만 원본 행 순서나 정렬 코드가 바뀌면 다른 데이터가
선택되므로 향후에는 초기 ID 목록과 SHA 목록을 직접 freeze해야 한다.

## 6. 잠긴 final 경계에 관한 감사 기록

이번 작업 초기에 저장소 전체를 넓게 조사하던 inventory 명령이 잠긴 final CSV의
metadata/header/행 수와 첫 레코드를 읽은 접근 사고가 있었다. 이미지는 열지 않았고,
final inference·metric 계산·모델 또는 전략 선택에는 사용하지 않았다. 이후 즉시 해당
파일의 모든 추가 사용을 중단했다.

따라서 이번 전체 세션에 대해 “final manifest를 한 번도 읽지 않았다”고 주장할 수는
없다. 다만 새 forward builder, strict preflight, forensic audit script에는 그 파일의
경로가 입력 또는 검색 대상으로 들어 있지 않으며, 이 스크립트 실행에서 final 이미지와
final metric 사용량은 0이다. 이 접근 사고는 향후 repository-wide discovery 자체를
금지 목록으로 차단해야 하는 이유이기도 하다.

## 7. 오류 여부를 네 범주로 정리

### A. 현재 증거로 오류가 없다고 확인한 것

- source/cache 이미지 바이트 연결
- bbox CSV에서 YOLO label로의 현재 변환 결과
- 10개 frozen 학습셋의 ID 및 이미지 SHA 순서
- 30개 result와 `last.pt` checkpoint 연결
- 기록된 핵심 training args
- 30행 aggregate, 300행 per-class 재집계
- acquisition 단위 bootstrap 및 gate 판정
- exact canonical ID / raw SHA split 누출

### B. 코드 설계 위험은 있지만 현재 산출물 오염 증거가 없는 것

- 존재 여부만 보는 result cache reuse
- 기존 image cache를 재검증하지 않는 `link_or_copy()`
- machine-specific 절대경로
- initial set의 row-position 복원
- 과거 실행 환경과 source manifest seal 기록 부족

### C. 바뀌었다면 결과가 달라질 가능성이 큰 것

- canonical sort 또는 blind manifest 행 순서
- acquisition seed 구성
- `last.pt` 대신 `best.pt`
- optimizer/Ultralytics/PyTorch/CUDA 조합
- budget, class-aware request, 라벨 정책, detector 구조

### D. 새로 승인한 실험 없이는 알 수 없는 것

- `best.pt`가 rare macro를 개선하는지
- class-aware/rare-aware selector가 gate를 통과하는지
- 다른 backbone이나 augmentation에서 전략 우위가 유지되는지
- 잠긴 final에서 development 개선이 일반화되는지

## 8. 최종 판단과 권고

현재 보관 결과를 계산 오류로 폐기하거나 frozen gate를 PASS로 뒤집을 근거는 없다.
가장 방어적인 해석은 다음 한 문장이다.

> 고정된 GC10 development protocol에서 FrozenDINO는 Random 대비 전체 mAP50-95를
> 개선했지만, 희소 결함군의 사전 정의 안전 gate를 실패했으므로 추가 detector/final 단계
> 확장을 중단했다.

다음 연구를 한다면 기존 결과를 덮어쓰거나 재해석하지 말고 새 run ID로 분리해야 한다.
우선순위는 다음과 같다.

1. 모든 선택셋의 canonical ID와 image SHA를 직접 freeze한다.
2. Git commit, runner SHA, source manifest SHA, model SHA, Python/PyTorch/CUDA/
   Ultralytics 버전과 environment lock을 함께 기록한다.
3. cache 재사용 전에 config + input manifests + checkpoint + output schema seal을 확인한다.
4. `last.pt` 대 `best.pt`는 사전 정의된 checkpoint 민감도 실험으로 분리한다.
5. rare-aware 변경은 기존 FrozenDINO의 사후 수정이 아니라 새 hypothesis/protocol로 등록한다.
6. 최종 평가 전까지 locked final manifest의 discovery와 read를 기술적으로 차단한다.

## 9. 재현 가능한 근거 파일

- Strict config: `configs/frozen/gc10_dev_confirm_harness_v1.yaml`
- Forward manifest build report:
  `reports/preflight/gc10_dev_confirm_harness_v1/manifest_build_report.json`
- Generic preflight report:
  `reports/preflight/gc10_dev_confirm_harness_v1/preflight_report.json`
- 독립 계보·민감도 감사 JSON:
  `reports/preflight/gc10_dev_confirm_harness_v1/gc10_research_lineage_audit.json`
- 사람이 읽는 자동 감사 요약:
  `reports/preflight/gc10_dev_confirm_harness_v1/gc10_research_lineage_audit.md`
- 실행 방법: `scripts/00_preflight/README.md`

역사적 `runs/` 결과와 `docs/results/`는 이번 작업에서 수정하지 않았다.

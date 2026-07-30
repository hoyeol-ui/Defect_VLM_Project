# KSDD2 PatchCore prospective 실험 실행서

이 실험의 목적은 PatchCore를 “좋은 selector”로 홍보하는 것이 아니다.
사전에 고정한 G1–G4가 실제로 다음 실험을 허용하거나 중단하는지를 보여주는
prospective authorization 사례를 만드는 것이다.

## 1. 고정된 질문

| Gate | 질문 | FAIL 시 |
|---|---|---|
| G1 | PatchCore score가 결함 이미지·연결성분 수를 실제로 rank하는가? | 즉시 `STOP` |
| G2 | query 100장에서 Random보다 결함 이미지를 더 찾는가? | 즉시 `STOP` |
| G3 | 작은 결함과 크기 구성이 과도하게 붕괴하지 않는가? | `DISCOVERY_ONLY_STOP` |
| G4 | 10개 initial/acquisition seed에서도 paired 효과가 반복되는가? | `STOP` |
| 전체 PASS | 딱 한 번의 별도 동결 G5 learner screen을 해도 되는가? | `BOUNDED_SCREEN` |

KSDD2는 단일 의미 결함 클래스이고 생산 lot 정보가 없으므로 G3는
category/source safety가 아니다. 작은 결함, 결함 크기 3-bin, 위치 3×3 grid,
pool 대비 Jensen–Shannon divergence만 평가한다.

## 2. 코드와 산출물

- `config.yaml`: split, PatchCore, 통계, gate threshold의 단일 원천
- `protocol.md`: OSF/Git freeze에 사용할 사전등록 본문
- `dataset_protocol.py`: target-blind split, initial set, mask post-hoc audit
- `patchcore_scoring.py`: image-only PatchCore fit/score 및 SHA-256 seal
- `analysis.py`: G1–G4 통계, 자동 STOP/authorization, 발표 요약 생성
- `run.py`: 순차 실행 CLI
- `test_core.py`: gate·누출 방지·mask bin 단위 테스트
- `smoke_patchcore.py`: anomalib GPU/CPU adapter 실제 smoke test

실험 결과는 `runs/ksdd2_patchcore_prospective_v1/`에 생성된다. 공식 test는
`final_locked.csv`에 경로와 잠금 상태만 남으며 G1–G4 코드가 이미지나 mask
내용을 읽지 않는다.

## 3. 최초 1회 준비

PowerShell에서 프로젝트 루트로 이동한다.

```powershell
cd "C:\Users\user\Desktop\vlm\Defect_VLM_Project"

.\.python311\python.exe -m pip install `
  -r scripts\06_ksdd2_prospective\requirements.txt

.\.python311\python.exe scripts\06_ksdd2_prospective\run.py download --extract
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py prepare
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py verify-pre-freeze
```

정상 출력에는 다음이 모두 포함되어야 한다.

- archive SHA-256 일치
- train 2,331 / development 466 / acquisition 1,865 / final lock 1,004
- initial set 10개, 각 100장
- `final_locked.csv`의 image/mask access가 모두 `false`
- outcome artifact 없음

## 4. 실행 전 검증

```powershell
.\.python311\python.exe scripts\06_ksdd2_prospective\test_core.py
.\.python311\python.exe scripts\06_ksdd2_prospective\smoke_patchcore.py
```

두 명령이 모두 PASS여야 한다. smoke test는 pretrained weight를 받지 않는
작은 합성 이미지 검사이므로 실제 KSDD2 outcome을 사용하지 않는다.

## 5. 공식 freeze

`verify-pre-freeze` 이후 code/config/protocol을 검토하고 의도한 변경을
commit한다. 다른 작업 중인 파일까지 임의로 commit하지 않는다. worktree가
깨끗한 상태에서만 다음을 실행한다.

```powershell
git status --short
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py freeze
```

`official_freeze: true`인지 확인한다. `--allow-dirty`는 동작 점검용
development freeze일 뿐 prospective 결과를 허용하지 않는다. 공식 실행에는
사용하지 않는다.

OSF를 사용할 경우 분석 전에 아래 자료를 하나의 registration에 첨부한다.

- `protocol.md`
- `config.yaml`
- 생성된 `freeze_manifest.json`과 `.sha256`
- `blind_dataset_manifest.csv`
- `development_manifest.csv`
- `acquisition_manifest.csv`
- `final_locked.csv`
- `initial_sets/initial_seed00.csv`부터 `initial_seed09.csv`
- 동결 Git commit SHA

## 6. G1 실행

```powershell
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py reveal-initial --seed 0
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py score-g1
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py analyze-g1

Get-Content runs\ksdd2_patchcore_prospective_v1\g1\g1_decision.json
```

`FAIL`이면 여기서 끝낸다. CLI도 다른 seed scoring을 차단한다. `PASS`일
때에만 G2–G4로 진행한다.

G1 발표용 자동 요약:

`runs/ksdd2_patchcore_prospective_v1/g1/g1_presentation_summary.md`

## 7. G2–G4 실행

중간 실패 후 재개하기 쉽도록 seed별 실행을 권한다.

```powershell
0..9 | ForEach-Object {
  .\.python311\python.exe scripts\06_ksdd2_prospective\run.py score-selection --seed $_
  .\.python311\python.exe scripts\06_ksdd2_prospective\run.py select --seed $_
}

.\.python311\python.exe scripts\06_ksdd2_prospective\run.py analyze-g2-g4
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py verify-run
```

`analyze-g2-g4`는 열 개 PatchCore/Random selection CSV의 seal을 모두
검증한 뒤에만 acquisition mask를 연다.

최종 발표용 자동 요약:

`runs/ksdd2_patchcore_prospective_v1/g2_g4/presentation_summary.md`

## 8. 결과별 발표 문장

### `BOUNDED_SCREEN`

“새 후보를 탐색한 것이 아니라 사전 동결한 단일 PatchCore 후보를
selection-only gate로 검증했다. G1–G4가 모두 통과했기 때문에, 이제 딱
하나의 별도 동결 learner 설정에 한해 G5를 수행할 근거가 생겼다.”

### `DISCOVERY_ONLY_STOP`

“결함 발견 수는 개선됐지만 작은 결함/크기 구성 안전성 기준을 통과하지
못했다. 따라서 detector 실험으로 확장하지 않았다. Discovery와 Safety를
분리해야 한다는 workflow 가설을 prospective하게 확인했다.”

### `STOP`

“사전 기준에 따라 detector 학습 전에 중단했다. 이것은 후보 실패를
성공으로 포장한 결과가 아니라, downstream 실험 권한을 결과 확인 후
자의적으로 부여하지 않았다는 process-conformance 증거다.”

## 9. 허용되지 않는 해석

- PatchCore의 보편적 selector 우위
- detector mAP 향상
- 독립 production pool 일반화
- annotation time, GPU hour, 비용 절감
- workflow의 predictive accuracy 또는 false-stop rate
- official test 성능

`BOUNDED_SCREEN`이 나오더라도 다음 단계는 별도 G5 protocol을 먼저
작성·freeze한 뒤 실행해야 한다.

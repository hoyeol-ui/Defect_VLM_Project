# 집에서 실행할 FF 재현·교체 가능성 점검

**현재 상태: 코드 준비 및 합성 입력 검증. 원본 GC10 실행 결과는 아직 없다.**
범위는 GC10-DET / 획득 seed 2026091901 / 기존 FF R4→R5다. CPU와 NumPy만 사용한다. 이미지·GT·모델을 읽지 않으며 Patch 추출, 새 표본 선택, YOLO 학습을 수행하는 기능은 없다.

## 먼저 알아둘 사항

- 원래 R4 481장을 고정하고 마지막 80장을 재계산한다. **80장 집합과 실제 획득 순서가 모두 일치**해야 진단이 열린다. R5 누적 목록을 정렬하거나 R5−R4를 임의로 나열해 원래 획득 순서라고 넣으면 안 된다.
- 현재 구현은 공개된 최종 규약을 옮긴 **독립 재계산기**다. 실제 최종 실행 코드의 복사본이 아니다. 원본 소스와 캐시 생성 기록에서 아래 계산 규칙을 확인한 뒤 설정의 `original_protocol_review`를 채운다. 다른 버전의 FF 코드로 대신 확인하지 않는다.
- 규칙: 저장된 float32 CLS를 float64로 변환, `1−dot`, 재정규화·거리 clipping·대각선 0 강제 없음, 매 선택 즉시 최근접 거리 갱신, 최대값과 absolute tolerance 1e-12 내 동점은 opaque ID 문자열 오름차순. 반경 판정에는 tolerance를 더하지 않는다.
- 새로운 계산기는 전체 행렬 곱을 사용한다. 원본의 연산 순서/BLAS가 다르면 근접 동점에서 차이가 생길 수 있다. **불일치하면 중단하고 원인을 조사한다. 통과시키려고 tolerance나 ID를 바꾸지 않는다.** NumPy·BLAS·thread 환경은 결과에 기록된다.
- 해시 일치만으로 파일의 역사적 정당성을 증명하지는 않는다. 원래 cache seal, 실행 manifest, 선택 로그와 대조한 출처를 `hash_evidence`에 남긴다. 현재 파일의 해시만 새로 계산해 정식 캐시로 간주하지 않는다.
- 최초 1대1 허용쌍 0은 이 초기 상태와 계산 규칙에 대한 결과다. 다중 교체·다른 제약까지 불가능하다는 결론은 아니다.
- **8/80은 후속 탐색을 판단할 실무 참고값**이다. 이 도구는 성공/실패 판정에 사용하지 않는다. 8장의 배경 변화보다 5장의 분명한 결함 형태 차이가 더 흥미로울 수 있다. 이 도구 자체는 교체 집합을 만들지 않으므로 순교체 수도 아직 계산하지 않는다.

## 1. 기존 Windows 실험 폴더와 별도 위치에 받기

PowerShell에서 새 폴더 이름을 골라 실행한다. 이미 있는 작업 디렉터리를 덮어쓰거나 reset하지 않는다.

```powershell
git clone --single-branch --branch research/ff-coverage-preflight-20261016 https://github.com/hoyeol-ui/Defect_VLM_Project.git Defect_FF_Preflight
cd Defect_FF_Preflight
python -m unittest discover -s research_tools/ff_coverage_preflight -p "test_*.py" -v
```

NumPy가 설치된 Python 환경을 사용한다. 테스트는 합성 데이터만 사용하며 원본 캐시를 탐색하지 않는다. NumPy가 없으면 별도 venv에 설치하고 버전을 기록한다. 과거 실험 환경의 패키지를 업그레이드하지 않는다.

## 2. 원본 연결 — 가장 중요한 단계

`config.example.json`을 같은 폴더의 `config.local.json`으로 복사한다. 모든 `path`, `sha256`, `hash_evidence`와 ID 파일의 `format`/`keys`를 실제 파일에 맞춰 채운다. JSON 경로는 `/` 또는 `\\`를 사용한다. 상대 경로는 config 파일 위치 기준이다. 예제는 경로를 자동 추측하지 않도록 비워져 있다.

| 입력 | 필요한 실제 내용 |
|---|---|
| features | 정식 GC10 F518 train CLS `(1604,384)` float32 `.npy`; labmeeting의 다른 cache와 혼동 금지 |
| cache_ids | feature 행 순서와 정확히 대응하는 opaque ID 목록. 절대 재정렬 금지 |
| train_ids | 최종 clean split의 train 1604 ID. dev/test 제외를 증명할 manifest에서 확인 |
| r4 / r5 | seed2026091901 DINO_F518의 누적481/561 ID, 원본 `selection_manifest.json`의 `cumulative_ids` 형식은 공개 감사 코드에서 확인됨 |
| new_order | 원본 R5의 **실제 획득 순서**80 ID. 해당 필드명은 집에서 확인 필요. 순서가 저장되지 않았다면 원본 실행 소스로 복원·대조하는 별도 확인부터 진행 |
| selector_source | 실제 사용한 최종 선택 소스 파일. 읽기/해시만 수행하며 임의 import나 실행하지 않음 |
| feature_manifest | 원래 cache seal/manifest. 캐시 SHA·전처리·모델·행 ID 대응을 사람이 확인한 기록 |

ID는 JSON 배열(필요시 `keys: ["cumulative_ids"]`), 문자열형 `.npy` 또는 한 줄 한 ID인 `.txt`를 지원한다. CSV·객체형 pickle은 자동 해석하지 않는다. 원본이 다른 형식이면 **별도 파생 파일**로 내보내고 원본 파일 hash와 순서 보존 근거를 기록한다. 원본을 수정하지 않는다. `.npy` features가 아닌 경우도 원본 비트·행 대응을 보존한 변환이 먼저 필요하다.

현재 맥에서 원본 경로와 schema 전체를 검증할 수 없었으므로 자동 연결은 제공하지 않는다. 참고할 원본 상대 경로는 기존 공개 감사 코드의 `state_attempts/ANNOTATION_PATCH_v2_0_2/features/GC10-DET/manifest.json`, `runs/GC10_v2_a2026091901_DINO_F518_R4/selection_manifest.json`, R5의 같은 파일이다. 이것은 기존 export 내부 상대 경로이며 Windows의 절대 위치는 집에서 찾아야 한다.

## 3. FF만 재현하기

```powershell
python research_tools/ff_coverage_preflight/preflight.py verify --config research_tools/ff_coverage_preflight/config.local.json --out research_tools/ff_coverage_preflight/outputs/verify_01
```

`ff_reproduction.json`에서 마지막80장 집합, 획득 순서, 누적 R5 집합 일치를 확인한다. step별 점수·동점 수·최초 순서 불일치·누락/추가 ID도 저장된다. `blocked.json`이면 입력부터 해결한다. `PASS`는 **이 cache에서 이 전이의 재현**만 의미하며 전처리 재추출이나 모든 라운드의 재현을 뜻하지 않는다. 먼저 결과를 검토하고 다음 명령의 실행 여부를 결정한다.

## 4. 통과한 경우 최초 1대1 교체만 검사

```powershell
python research_tools/ff_coverage_preflight/preflight.py screen --config research_tools/ff_coverage_preflight/config.local.json --out research_tools/ff_coverage_preflight/outputs/screen_01 --seconds 1800
```

1800초는 제안한 **screen 루프 시간 상한**이며 예상 소요시간이 아니다. 파일 검사/FF 재현/행렬 생성 시간은 별도다. 새 실행도 입력 hash와 FF를 다시 검사한다. FF 불일치 시 screen으로 넘어갈 수 없다. 최초 상태의80×1043=83,440쌍을 검사하고, R4는 제거하지 않는다. 원래 전체561장 기준 최대 반경을 유지하는 쌍만 CSV에 기록한다. 원본 집합은 바꾸지 않는다.

- `swap_feasibility.json`: 검사 수/전체 수, 허용쌍 수, 고유 유입·유출 수, 원래 반경·평균·95분위, 시간, 완료 여부.
- `feasible_initial_pairs.csv`: 각 허용쌍의 유입·유출 ID와 반경 변화. 미완료일 때는 **부분 결과**다.
- `provenance.json`: 입력·도구 hash와 환경. 로컬 절대 경로가 포함되므로 결과 공유 전 확인한다.
- 종료코드0=정상 완료(허용쌍0도 포함), 2=입력/재현 차단, 3=시간 상한으로 미완료. 양의 반경 차이가1e-12 이하여도 허용하지 않고 별도로 센다.

허용쌍이 있으면 Patch 비용·표현·정성 점검 설계를 다시 확인한다. 자동으로 Patch나 YOLO를 실행하는 후속 단계는 없다. 0이거나 미완료라면 각각 다른 의미로 보고한다. 논문 성과/노블티/AP 개선은 이 결과로 판정하지 않는다.

## 집의 Codex에 전달할 요청

> 이 폴더의 README와 DECISION_KO.md를 읽고 원본 GC10 정식 CLS cache, 행 ID, train manifest, seed2026091901 FF R4/R5 목록과 실제 R5 획득 순서, 최종 실행 선택 소스를 읽기 전용으로 찾아 연결해 주세요. 과거 다른 F518 cache로 대체하지 마세요. 실제 원본 규칙과 새 checker의 차이를 먼저 보고하고 config.local.json을 채워 주세요. 새 특징 추출·학습·원본 수정은 하지 마세요. 우선 verify만 실행해 마지막80장의 집합과 순서 재현 결과를 보고해 주세요. screen은 그 결과를 검토한 뒤 결정하겠습니다.

## 이번 맥북 검증 기록

2026-10-16, Python 3.12.14, NumPy 2.3.5에서 합성 입력 테스트15개 통과. 빠른 교체 판정과 독립적인 직접 전수 재계산의 일치, 동점/즉시 갱신, 행-ID 대응, 허용 반경 경계, 시간 상한, 실제 설정 schema 로딩, 해시 변조, 비정규화 cache, 순서 불일치 차단, 출력 덮어쓰기 방지를 검사했다. `git diff --check`도 통과했다. Windows와 원본 GC10 자료에 대한 검증은 귀가 후 남아 있다.

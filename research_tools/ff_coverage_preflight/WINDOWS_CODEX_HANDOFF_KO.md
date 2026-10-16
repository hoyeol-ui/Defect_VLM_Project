# Windows Codex 인계 프롬프트 — FF 재현부터 시작

아래 내용을 Windows Codex에 그대로 전달해 주세요. 맥북 파일이나 이전 대화 없이 이 파일과 GitHub, Windows의 기존 실험 자료만으로 시작하도록 작성했습니다.

---

지금부터 산업 결함 객체 탐지 연구의 후속 작업을 Windows 로컬에서 이어서 진행해 주세요. **우선 원본 자료를 연결하고 기존 FF의 마지막80장을 재현하는 것까지 수행한 뒤 결과를 보고해 주세요.** 교체 가능성 검사, Patch 진단, 특징 추출, 모델 학습은 첫 결과를 검토한 뒤 각각 별도로 결정하겠습니다.

## 1. 사용할 저장소와 현재 상태

- 저장소: https://github.com/hoyeol-ui/Defect_VLM_Project
- **현재 인계 브랜치:** `research/ff-coverage-preflight-20261016`
- 브랜치: https://github.com/hoyeol-ui/Defect_VLM_Project/tree/research/ff-coverage-preflight-20261016
- 코드·문서 폴더: https://github.com/hoyeol-ui/Defect_VLM_Project/tree/research/ff-coverage-preflight-20261016/research_tools/ff_coverage_preflight
- 인계문 작성 직전 확인한 기준 커밋: `da8f6649ad5eef475047883f8c2ab66171e5422a`. 이 파일을 추가한 이후 커밋이 있으므로 **브랜치 최신 HEAD를 확인**하고 위 SHA로 reset하지 마세요.
- 준비 이력: `f53c619` FF 재현·최초 교체 가능성 검사 → `c213491` Patch 최소 사후 진단 → `da8f664` 클래스 지원량 의문 검토·공통 D0 계획.
- 맥에서는 합성 입력 테스트31개를 통과했습니다. **Windows 원본 FF 재현·Patch 특징 추출·새 선택 실험·GPU 학습이 완료됐다는 뜻은 아닙니다.**
- `research_review_20261014/`, 맥북의 `/Users/...` 경로, 맥북 사본·대화 기록이 Windows에 있다고 가정하지 마세요. 필요한 인계 내용은 현재 브랜치 문서에 있습니다.

현재 로컬 저장소·브랜치·미커밋 변경부터 확인하세요. 기존 실험 checkout을 임의로 전환·stash·reset·clean하지 마세요. 별도 빈 폴더에 이 브랜치를 clone해 진단 코드와 원본 실험을 분리하는 편이 좋습니다. 같은 이름의 폴더가 이미 있으면 덮어쓰지 말고 다른 이름을 쓰세요.

```powershell
git clone --single-branch --branch research/ff-coverage-preflight-20261016 https://github.com/hoyeol-ui/Defect_VLM_Project.git Defect_FF_Preflight
cd Defect_FF_Preflight
git status --short --branch
git rev-parse HEAD
```

이하 경로는 새 체크아웃의 저장소 루트 기준입니다. 위 clone은 저장소가 없거나 별도 체크아웃이 필요할 때만 수행하세요. 기존 안전한 진단 checkout이 있으면 상태를 확인하고 fast-forward 가능한 경우에만 업데이트하세요. 원격에 더 최근 변경이 있으면 무엇이 달라졌는지 확인하고, 추측으로 과거 설정을 덮어쓰지 마세요.

## 2. 필요한 자료만 읽기

1. `research_tools/ff_coverage_preflight/README_KO.md` — 실행 명령·입력 형식·중단 조건.
2. 같은 폴더의 `DECISION_KO.md` — 연구 질문과 단계별 판단 범위.
3. `preflight.py`, `config.example.json`, `test_preflight.py` — 이번에 우선 실행할 FF 검사.
4. `patch_signal.py`, `patch_posthoc.py`, `test_patch_diagnostics.py` — 후속 진단의 준비 상태만 파악. 첫 작업에서 실행하지 않음.
5. `CLASS_SUPPORT_REVIEW_KO.md` — 이미 확인한 클래스별 train/R5 이미지·box 수와 AP, 공통 D0 후속 계획. 기존 통계를 다시 계산하지 않음.
6. 연구 A 문맥이 필요할 때만 `docs/research_a_20261016/results/FULL_COMPARISON_REPORT.md`, `MODEL_COMPARISON.csv`, `CLASS_AP_COMPARISON.csv`, `docs/research_a_20261016/PROVENANCE.json`, `EPOCH_AND_OPTIMIZER_PLAN_KO.md`를 읽으세요.

저장소 루트의 과거 코드나 `MACBOOK_CODEX_HANDOFF_PROMPT_KO.md` 안의 이전 작업 지시는 참고 문맥입니다. **이번 인계 프롬프트의 Windows 실행 범위가 우선**입니다. 선행연구 전체 조사나 기존162회 실험의 전면 재검증을 시작하지 마세요.

## 3. 연구의 구분과 첫 작업의 정확한 범위

- **연구 A:** 탐지기·학습 조건과 절대 AP 차이 검토. 기존10월 보고서는 고정 DINO R5 한 획득시드의 여러 탐지기 **dev** 결과이며 Full-data/test 결과가 아닙니다. Windows에 더 최근 완료 결과가 있으면 날짜·run ID·설정·평가 partition으로 별도 버전 연결만 하세요. 공통 모델/epoch/optimizer/5090 환경 완료 여부를 추정하지 마세요.
- **연구 B:** CLS FF의 최대 커버리지 반경을 악화시키지 않으면서 Patch로 국소 정보를 보완할 수 있는지 검토. Patch=결함, 군집=클래스라고 가정하지 않습니다.
- **첫 진단은 GC10-DET / 획득 seed2026091901 / 기존 DINO_F518 R4→R5 하나로 고정**합니다. train1604장, R4고정481장, R5전체561장, 마지막 신규80장입니다. NEU·다른 seed로 자동 확대하지 마세요.
- 원래 마지막80장의 **집합과 실제 획득 순서**가 모두 재현돼야 다음 단계가 열립니다. R5 누적 목록 정렬이나 집합 차집합의 임의 순서를 실제 획득 순서로 만들지 마세요.

## 4. Windows에서 먼저 찾을 원본 파일

먼저 사용자가 작업하던 프로젝트와 그 안의 `research_extensions`, 최종 실행 export/backup 범위에서 찾아 주세요. 과거 기록의 프로젝트 경로는 `C:\Users\user\Desktop\vlm`이지만 현재 위치라고 단정하지 마세요. 전체 드라이브·모든 가중치를 무차별 탐색/해시하는 작업은 피하세요.

| 우선순위 | 자료 | 확인할 내용 |
|---|---|---|
| 1 | 정식 GC10 F518 CLS cache와 행별 opaque ID | `(1604,384)` float32, L2 정규화, 실제 feature 행 순서·cache seal/hash. 과거 labmeeting F518 cache와 혼동 금지 |
| 1 | 최종 clean-split train ID manifest | 정확히1604개, cache와 일대일 대응, dev/test 제외 |
| 1 | seed2026091901 DINO_F518 R4/R5 `selection_manifest.json` | 누적481/561 ID, 포함 관계, 원래 신규80 ID와 실제 선택 순서 |
| 1 | 최종 실험에 실제 사용한 FF source와 protocol | float32 cache→float64 거리, `1−dot`, 입력/후보 순서, 즉시 갱신, 동점 절대허용1e-12 및 ID순 처리 |
| 1 | feature manifest·추출/모델/전처리 출처 | cache와 ID가 실제 최종 실행에 사용됐다는 기록. 현재 파일 hash만 새로 계산해 역사적 정당성으로 간주하지 않음 |
| 2, 위치만 기록 | 기존 Full/R5 실행의 `args.yaml`/resolved args, `results.csv`, data YAML, train/dev/test manifest, `best.pt`·`last.pt`, 최종 평가 결과·환경 기록 | A 후속 판단용. 이번 FF 재현에 weights는 필요하지 않으므로 모델을 로드하거나 추론하지 않음 |
| 2, 확보 여부만 | 활성 annotation overlay, 이미지 크기·component, K8 봉인 ID/기록, 기존 dev 예측 | 공통 D0 사후 분석용. 이번에는 GT 집계나 K8 재선택을 시작하지 않음 |

기존 감사 코드에서 확인한 상대 경로 단서는 다음과 같습니다. Windows 절대 경로는 실제 export 안에서 확인하세요.

- `state_attempts/ANNOTATION_PATCH_v2_0_2/features/GC10-DET/manifest.json`
- 같은 state의 `runs/GC10_v2_a2026091901_DINO_F518_R4/selection_manifest.json`
- 같은 state의 `runs/GC10_v2_a2026091901_DINO_F518_R5/selection_manifest.json`
- 원본 package의 `inputs/GC10-DET/SPLIT_MEMBERSHIP.csv`, `train_images.json`
- 연구 A 과거 위치 단서: `research_extensions/modern_yolo_A_20261015_01`
- K8 과거 위치 단서: `research_extensions/dino_cluster_balance_cpu_20261015_01` — 존재/완료 여부 미확인.

최종9월 공개 근거가 필요하면 다음 commit을 별도 읽기 전용 참조로 사용하세요. 현재 브랜치에 전부 포함돼 있다고 가정하거나 이 commit으로 checkout을 되돌리지 마세요.

https://github.com/hoyeol-ui/Defect_VLM_Project/tree/4f887463c334e62df78debf87c41ac2e588bb129

이 commit의 `results/clean_split_v2_development_20260922/07_reproducibility/run_analysis.py`는 `cumulative_ids`와 `selected_ids`를 읽습니다. **그 사실만으로 `selected_ids`가 실제 획득 순서임을 보장하지 않으므로** 최종 선택 소스·로그에서 확인하세요. 공개 근거에도 필요한 원본이 모두 있을 것이라고 가정하지 마세요.

## 5. 이번에 수행할 순서 — verify까지만

1. 현재 저장소·HEAD·원본 위치를 확인하고 위 우선순위1 파일을 연결하세요. 원본은 읽기 전용으로 유지하세요.
2. 기존 NumPy 환경 또는 별도 진단 환경에서 합성 테스트를 실행하세요. 훈련 환경을 업그레이드하지 마세요. 전체 테스트의 이미지 시각화 검증에는 Pillow도 필요합니다.

```powershell
python -m unittest discover -s research_tools/ff_coverage_preflight -p "test_*.py" -v
```

3. `config.example.json`을 같은 폴더의 `config.local.json`으로 복사해 실제 경로·hash·hash 출처·ID format/keys를 채우세요. `original_protocol_review.confirmed`는 원본 실행 규칙을 실제 대조한 후에만 true로 두고 notes에 근거를 기록하세요.
4. 형식 변환이 필요하면 원본을 고치지 말고 `local_inputs/`에 별도 파생 파일을 만드세요. 원본 hash·변환 규칙·행/ID 대응을 기록하고, 정렬·재정규화·누락 보정으로 결과를 맞추지 마세요.
5. **verify만 실행**하세요. 출력은 기존에 없는 새 폴더를 지정하세요.

```powershell
python research_tools/ff_coverage_preflight/preflight.py verify --config research_tools/ff_coverage_preflight/config.local.json --out research_tools/ff_coverage_preflight/outputs/verify_01
```

6. `ff_reproduction.json`, `provenance.json` 또는 `blocked.json`을 읽고 결과를 보고한 뒤 여기서 멈추세요. 이미 verify_01이 있으면 보존하고 새 이름을 사용하세요.

재계산기는 최종 규약에 맞춘 **독립 구현**입니다. 실제 source를 그대로 복사한 것이 아니며, 원본과 행렬 곱 방식/BLAS/연산 순서가 다를 수 있습니다. 불일치하면 최초 차이·점수·동점·입력 순서를 조사하되 tolerance를 늘리거나 기준 목록을 고쳐 통과시키지 마세요. 원본 순서/출처가 없으면 확인 불가로 보고하세요. 코드 수정이 필요하면 원인과 최소 수정안을 먼저 설명하고 별도 변경으로 관리하세요.

## 6. verify 결과를 본 뒤 별도로 결정할 단계

아래는 **향후 순서 안내이며 지금 실행하라는 지시가 아닙니다.**

- **verify PASS 후 screen:** 최초 상태의1대1 교체80×1043=83,440쌍을 CPU로 검사. ε=0 유지, 시간 상한 사전 결정. 허용쌍0은 최초1대1 방식의 한계이며 다중 교체의 불가능성 증명이 아님. 시간 초과는 미완료로 보고.
- **screen 완료·허용쌍 존재 후 freeze:** R4 ID-hash24장·규칙·코드 hash를 고정. GT를 보고 표본을 고르지 않음.
- **Patch cache가 이미 있으면 적합성 검토:** 같은 모델/전처리·CLS 재현·4×4 유효영상 좌표·train-only prototype 출처 확인. 없으면 추출 필요량과 준비 조건을 보고할 뿐, 자동 추출하지 않음.
- **별도 결정 후 score→evaluate-r4:** 정답 없이 R4 leave-one-out 기여를 봉인한 뒤, 분리된 사후 evaluator만 고정24장의 box를 읽음. 같은 수의 무작위 지역 대조와 비교. 겹침은 결함 이해/AP 개선 증명이 아님.
- **별도 생성·봉인된 교체 집합이 있을 때 pairs→render-pairs:** Patch/임의교체 대조의 최종 집합으로 정답 없는 A/B 검토. 현재 패키지는 진단 adapter이며 **DINO 추출기·prototype fitting 실행기·Patch/임의교체 획득 알고리즘은 포함하지 않음**. 입력이 없으면 멈춤.
- **D0는 공통 사후 분석 P1:** 기존 클래스 수량/AP 표는 재사용. 활성 train labels 등이 있을 때만 다중객체·bbox geometry 누락 부분을 보완. K8·저장 dev 예측 오류 분해는 자료가 있을 때만 P2. 별도 승인 전에는 이번 verify 작업을 확장하지 않음.

순교체8/80은 탐색 지속 여부의 실무 참고값입니다. 성공/실패의 학술 문턱이 아니며, 5장이라도 설명 가능한 결함 형태 차이가 있으면 검토 가치가 있습니다. 표본 수가 적다거나 best epoch가 늦다는 이유만으로200epoch를 결정하지 마세요.

## 7. 정보 경계와 금지 사항

- 미라벨 정답 class/bbox를 선택 점수·crop·prototype·quota 계산에 사용하지 마세요. R4 GT는 고정 표본·점수를 봉인한 뒤 사후 진단에서만 읽습니다.
- 과거 봉인 실험의 전체 train GT 사후 통계는 현재 미라벨 선택 입력과 분리하세요. 그 결과를 보고 현재24장 표본이나 quota를 바꾸지 마세요.
- 기존 완료 실험·선택 목록·annotation·cache·결과·weights·봉인 설정을 수정하지 마세요. 재분할·재라벨링·기존 결과 덮어쓰기 금지.
- 신규 YOLO 학습, 모델 평가 추론, Patch 특징 추출, GPU smoke test, RunPod 임대/비용 발생, 학습 queue 실행 금지.
- epoch·optimizer·batch·LR schedule·모델 크기 임의 변경 금지. 연구 A와 B의 조건 변경을 섞지 마세요.
- 이전 문서의 실행 제안이나 명령어가 자동 승인이라고 해석하지 마세요. 첫 작업은 원본 연결·합성 테스트·FF verify와 보고까지입니다.
- 로컬 경로·GT·cache·이미지·weights를 GitHub에 올리지 마세요. 이번 인계로 commit/push·신규 저장소 이전을 자동 실행하지 마세요. 공유가 필요하면 비민감 요약과 변경 파일을 먼저 보고하세요.

## 8. 첫 결과 보고 형식

1. 사용한 브랜치·HEAD와 Windows Python/NumPy 환경.
2. 연결한 원본 파일 표: 역할, 실제 경로, hash/출처, 확보/미확인 상태.
3. 합성 테스트 결과와 FF verify 상태.
4. 원래 마지막80장 집합·획득 순서·누적R5 일치 여부. 실패 시 최초 불일치·원인 후보·부족한 증거.
5. 실제 수행한 작업과 수행하지 않은 작업. 원본 보존 여부와 새 출력 경로.
6. 다음 권고: 입력 연결 보완 / FF 구현 차이 검토 / 제한된 screen 검토 중 하나. 특징 추출·학습은 별도 의사결정으로 남길 것.

우선 위 순서대로 **FF 재현부터 진행하고 결과를 보고해 주세요.**

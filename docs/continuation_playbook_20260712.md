# Continuation playbook — 2026-07-12

이 문서는 “맥북에서 문서작업을 이어가다가, 필요하면 Windows GPU 머신으로 다시 실험을 넘기는” 실전 플레이북이다.

## A. 맥북에서 문서작업만 할 때

### A1. 최신 코드 받기

```bash
git clone https://github.com/hoyeol-ui/Defect_VLM_Project.git
cd Defect_VLM_Project
git checkout experiment/v6-deficit-diversity
```

이미 받은 경우:

```bash
git fetch origin
git checkout experiment/v6-deficit-diversity
git pull --ff-only origin experiment/v6-deficit-diversity
```

### A2. 문서 패키지 확인

```bash
python3 scripts/docs/check_handoff_package.py
```

### A3. 핵심 문서 열기

```bash
bash scripts/docs/macbook_open_docs.sh
```

### A4. Notion에 업데이트

추천 소스:

```text
docs/results/v10b_seed42_documentation_20260712_215841/V10b_Seed42_Development_Gate_Updated_Report.md
```

Notion에는 Markdown을 먼저 붙이고, 깨지는 표/그림은 아래 폴더에서 개별 업로드한다.

```text
docs/results/v10b_seed42_documentation_20260712_215841/figures/
docs/results/v10b_seed42_documentation_20260712_215841/tables/
```

## B. 맥북에서 문서 수정 후 GitHub 반영

문서만 수정했다면:

```bash
git status
git add docs scripts/docs
git commit -m "docs: update research handoff notes"
git push origin experiment/v6-deficit-diversity
```

DOCX를 수정했다면 커밋 전에 꼭 파일이 열리는지 확인한다.

```bash
open docs/results/v10b_seed42_documentation_20260712_215841/V10b_Seed42_Development_Gate_Updated_Report.docx
```

## C. Windows GPU 머신에서 다시 실험할 때

맥북에서 문서를 수정해 push한 뒤 Windows로 돌아오면:

```powershell
cd "C:\Users\user\Desktop\vlm\Defect_VLM_Project"
git pull --ff-only origin experiment/v6-deficit-diversity
```

가상환경 또는 bundled Python 확인:

```powershell
.\.python311\python.exe --version
.\.python311\python.exe -c "import torch; print(torch.cuda.is_available())"
```

주의:

- `.venv`에는 torch가 없을 수 있다.
- 이전 실험은 주로 `.\.python311\python.exe` 기준으로 실행했다.
- `final test`는 계속 사용 금지다.

## D. V10b 계열 주요 실행 명령

아래 명령은 기록용이다. 맥북 문서작업에는 필요 없다. Windows GPU 머신에서만 실행하는 것을 권장한다.

### D1. V10 large-pool smoke

```powershell
cd "C:\Users\user\Desktop\vlm\Defect_VLM_Project"

$env:AL_YOLO_DEVICE = "0"
$env:AL_BATCH_SIZE = "8"
$env:AL_WORKERS = "4"
$env:AL_YOLO_CACHE = "false"
$env:AL_YOLO_PLOTS = "false"
$env:AL_INITIAL_SEED_SIZE = "60"
$env:AL_QUERY_SIZE = "30"

.\.python311\python.exe `
  scripts\02_active_learning\run_v10_neu_large_pool_smoke.py
```

### D2. V10b selection probe

```powershell
cd "C:\Users\user\Desktop\vlm\Defect_VLM_Project"

$env:AL_V10B_W_UNCERTAINTY = "0.25"
$env:AL_V10B_W_DINO = "0.35"
$env:AL_V10B_W_BALANCE = "0.15"
$env:AL_V10B_W_INSTANCE = "0.25"

.\.python311\python.exe `
  scripts\02_active_learning\probe_v10b_selection_from_existing_v10.py
```

### D3. V10b single training from existing selection

```powershell
cd "C:\Users\user\Desktop\vlm\Defect_VLM_Project"

.\.python311\python.exe `
  scripts\02_active_learning\train_v10b_from_existing_selection.py
```

### D4. V10 per-class audit recovery

```powershell
cd "C:\Users\user\Desktop\vlm\Defect_VLM_Project"

.\.python311\python.exe `
  scripts\02_active_learning\recover_v10_per_class_audit.py
```

### D5. V10b seed43~46 one-cycle multiseed

```powershell
cd "C:\Users\user\Desktop\vlm\Defect_VLM_Project"

$env:AL_ACQUISITION_SEEDS = "43,44,45,46"
$env:AL_YOLO_DEVICE = "0"
$env:AL_BATCH_SIZE = "8"
$env:AL_WORKERS = "4"
$env:AL_YOLO_CACHE = "false"
$env:AL_YOLO_PLOTS = "false"

$env:AL_V10B_W_UNCERTAINTY = "0.25"
$env:AL_V10B_W_DINO = "0.35"
$env:AL_V10B_W_BALANCE = "0.15"
$env:AL_V10B_W_INSTANCE = "0.25"

$env:AL_V9_CANDIDATE_FRACTION = "1.00"
$env:AL_V9_PREDICT_CONF = "0.05"
$env:AL_V9_PREDICT_IOU = "0.70"

$env:AL_V9B_MAX_NO_BOX = "0"
$env:AL_V9B_MIN_PSEUDO_BOXES = "2"
$env:AL_V9B_MAX_PER_PRED_CLASS = "2"

$env:AL_INITIAL_SEED_SIZE = "60"
$env:AL_QUERY_SIZE = "30"

.\.python311\python.exe `
  scripts\02_active_learning\run_v10b_multiseed_onecycle.py
```

## E. 다음 연구 판단 기준

V10b 이후 실험을 진행할 때는 다음 기준을 먼저 본다.

1. Random 대비 paired mean mAP50-95 diff가 seed variation보다 충분히 큰가?
2. win/loss/tie가 우연으로 보기 어려운가?
3. precision gain이 recall loss를 정당화하는가?
4. per-class에서 특정 class 손실이 반복되는가?
5. final test를 보지 않고도 방법 선택이 가능한가?

현재 V10b의 답:

```text
1. 충분히 크지 않다.
2. 3/1/0은 좋아 보이나 평균 차이가 너무 작다.
3. 아직 애매하다. precision은 오르고 recall은 떨어졌다.
4. seed42 per-class 기준 inclusion, rolled-in-scale 손실이 있었다.
5. 가능하다. final test는 아직 locked다.
```

따라서 다음 실험은 “V10b full-scale 확증”보다는 “recall penalty를 줄이는 V10c 설계”가 더 자연스럽다.

## F. 문서작업용 체크리스트

- [ ] `docs/research_context_handoff_20260712.md`에서 최신 결론 확인
- [ ] `docs/preview.html`로 전체 흐름 확인
- [ ] `V10b_Seed42_Development_Gate_Updated_Report.md`를 Notion 초안으로 사용
- [ ] multiseed 결과는 “Random 대비 확정 승리”로 쓰지 않기
- [ ] final test는 사용하지 않았다고 명시
- [ ] raw runs/checkpoints는 GitHub에 없다고 명시
- [ ] 다음 실험은 V10c recall-aware 또는 longer curve로 제안


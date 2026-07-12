# MacBook handoff guide — 2026-07-12

이 문서는 Windows/VS Code에서 진행한 Defect VLM Active Learning 연구를 맥북에서 문서작업 중심으로 이어받기 위한 핸즈온 가이드다.

핵심 목표는 간단하다.

- GitHub에서 최신 브랜치를 받는다.
- 오늘까지 정리한 HTML, Markdown, DOCX, 그림, 표를 바로 연다.
- raw run 폴더나 GPU 학습환경 없이도 Notion/논문/발표 문서작업을 이어간다.
- 필요하면 다시 Windows GPU 머신으로 돌아가 실험을 이어갈 수 있게 맥락을 보존한다.

## 1. 처음 받는 경우

맥북 터미널에서 다음을 실행한다.

```bash
cd ~/Desktop
git clone https://github.com/hoyeol-ui/Defect_VLM_Project.git
cd Defect_VLM_Project
git checkout experiment/v6-deficit-diversity
```

이미 클론해 둔 폴더가 있다면 다음만 실행한다.

```bash
cd /path/to/Defect_VLM_Project
git fetch origin
git checkout experiment/v6-deficit-diversity
git pull --ff-only origin experiment/v6-deficit-diversity
```

이 가이드 작성 직전, V10b 문서 산출물을 고정했던 기준 커밋은 다음이다.

```text
bf014af docs: update V10b detector-aware validation artifacts
```

이 핸드오프 가이드 자체가 추가된 뒤에는 더 최신 커밋이 생기므로, 맥북에서는 항상 `git pull` 후 아래 명령으로 최신 위치를 확인하면 된다.

```bash
git log -1 --oneline
```

## 2. 문서 패키지 점검

외부 패키지 설치 없이 기본 `python3`만 있으면 된다.

```bash
python3 scripts/docs/check_handoff_package.py
```

HTML/DOCX/Markdown을 바로 열고 싶으면:

```bash
bash scripts/docs/macbook_open_docs.sh
```

열기 없이 점검만 하고 싶으면:

```bash
bash scripts/docs/macbook_open_docs.sh --check-only
```

## 3. 가장 먼저 열 파일

문서작업은 아래 순서가 가장 편하다.

1. `docs/preview.html`
   - 전체 아키텍처 변천사와 최신 워크플로우를 브라우저에서 훑어보는 용도.

2. `docs/vlm_gt_free_al_workflow.html`
   - GT-free active learning 워크플로우를 시각적으로 정리한 문서.

3. `docs/results/v10b_seed42_documentation_20260712_215841/V10b_Seed42_Development_Gate_Updated_Report.docx`
   - Word/Pages에서 열 수 있는 seed42 기준 업데이트 연구보고서.

4. `docs/results/v10b_seed42_documentation_20260712_215841/V10b_Seed42_Development_Gate_Updated_Report.md`
   - Notion에 붙여넣기 쉬운 Markdown 버전.

5. `docs/research_context_handoff_20260712.md`
   - 지금까지의 대화, 실험 흐름, 결론, 다음 의사결정을 복원하기 위한 문맥 문서.

6. `docs/continuation_playbook_20260712.md`
   - 맥북에서 문서 수정, GitHub 반영, Windows GPU 머신으로 실험을 넘기는 실전 플레이북.

## 4. Notion 업데이트 추천 순서

Notion에는 다음 구조로 옮기면 좋다.

```text
Defect VLM Active Learning 연구 업데이트
├─ 1. 한 줄 결론
├─ 2. 왜 Random을 이기기 어려웠나
├─ 3. V7/V8/V9/V10/V10b 실험 흐름
├─ 4. V10b seed42 development gate 결과
├─ 5. V10b seed43~46 독립 seed 결과
├─ 6. 현재 해석: 후보 방향이지만 최종 승리 아님
├─ 7. 다음 실험 제안
└─ 8. Appendix: 표/그림/실행 경로
```

복사 소스는 Markdown 보고서가 제일 편하다.

```text
docs/results/v10b_seed42_documentation_20260712_215841/V10b_Seed42_Development_Gate_Updated_Report.md
```

그림은 다음 폴더에 있다.

```text
docs/results/v10b_seed42_documentation_20260712_215841/figures/
```

표는 다음 폴더에 있다.

```text
docs/results/v10b_seed42_documentation_20260712_215841/tables/
```

## 5. 맥북에 없는 것

GitHub에는 문서작업에 필요한 curated artifact만 올려 두었다. 다음은 의도적으로 제외되어 있다.

- `runs/`
- `datasets/`
- `data/`
- `.python311/`
- `.venv/`
- `.pt`, `.pth` 같은 모델 checkpoint

이유는 간단하다. 용량이 크고, raw training output은 GitHub에 올리기 적합하지 않다. 대신 보고서에 필요한 CSV, PNG, Markdown, DOCX는 `docs/results/` 아래에 복사해 추적 가능하게 만들었다.

raw run이 꼭 필요하다면 Windows 머신의 다음 계열 폴더를 별도로 압축/외장 저장장치/클라우드로 옮겨야 한다.

```text
runs/active_learning_ablation_v10_neu_large_pool/
runs/v10b_single_training/
runs/v10b_independent_result_audit/
runs/active_learning_v10b_multiseed_onecycle/
datasets/active_learning_ablation_v10_neu_large_pool/
datasets/v10b_single_training/
```

## 6. 맥북에서 실험을 다시 돌릴 수 있나?

문서작업은 바로 가능하다. 하지만 YOLO 학습 실험은 맥북에서 권장하지 않는다.

현재 파이프라인은 CUDA GPU가 있는 Windows/VS Code 환경을 기준으로 설계되었다. 맥북의 Apple Silicon/MPS 환경에서도 일부 PyTorch 실행은 가능할 수 있지만, 이번 실험의 재현 기준과 시간 추정이 달라진다. 따라서 맥북에서는 문서, 분석, 보고서 편집을 하고, 실제 학습은 Windows GPU 머신에서 이어가는 편이 안전하다.

## 7. 맥북에서 문서 수정 후 GitHub에 올리기

문서를 수정한 뒤:

```bash
git status
git add docs scripts/docs
git commit -m "docs: update MacBook handoff notes"
git push origin experiment/v6-deficit-diversity
```

다시 Windows로 돌아와 받을 때:

```powershell
cd "C:\Users\user\Desktop\vlm\Defect_VLM_Project"
git pull --ff-only origin experiment/v6-deficit-diversity
```

## 8. 충돌이 생겼을 때

맥북과 Windows에서 동시에 같은 문서를 수정하면 Git conflict가 날 수 있다. 그럴 때는 우선 다음만 확인한다.

```bash
git status
```

충돌 파일이 `docs/*.md`라면 보통 수동으로 쉽게 해결 가능하다. 하지만 `docx`는 binary라 충돌 해결이 어렵다. DOCX를 고칠 때는 한 기기에서만 수정하고 바로 push/pull 하는 습관이 좋다.

## 9. 오늘 기준 가장 중요한 연구 해석

V10b는 seed42 development gate에서는 Random보다 좋아 보였지만, seed43~46 one-cycle 독립 검증에서는 평균 차이가 매우 작았다.

```text
Random mean mAP50-95: 0.299207
V10b mean mAP50-95:   0.300157
paired mean diff:     +0.000949
wins/losses/ties:     3/1/0
precision mean diff:  +0.023781
recall mean diff:     -0.026170
final test used:      False
weights frozen:       True
```

즉 지금의 결론은 “V10b가 Random을 확실히 이겼다”가 아니라, “V10b는 V9b보다 나은 방향이고 Random과 거의 동률까지 왔지만, recall penalty를 해결해야 일반화 우위를 주장할 수 있다”에 가깝다.

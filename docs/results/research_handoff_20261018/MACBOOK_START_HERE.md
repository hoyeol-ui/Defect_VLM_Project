# MacBook에서 이어받기

기준일: 2026-10-18(KST). 내일은 수동으로 이어간다. 예약 작업·자동 학습·추가 진단은 없다.

## 새 폴더에 clone

```bash
git clone --branch codex/gc10-research-handoff-20261018 https://github.com/hoyeol-ui/Defect_VLM_Project.git Defect_VLM_Project_handoff
cd Defect_VLM_Project_handoff
open docs/results/research_handoff_20261018/README_KO.md
```

기존 Mac 저장소가 수정 중이라면 새 폴더 clone이 현재 파일을 건드리지 않는다. 기본 main 대신 위 브랜치가 이번 문서를 포함한다.

## 무거운 환경 없이 읽기

Markdown은 GitHub나 편집기에서 읽는다. RESULTS.json과 tables의 CSV는 정밀 수치를 포함한다. Python 3.9 이상이 있으면 패키지 무결성만 확인할 수 있다:

```bash
cd docs/results/research_handoff_20261018
python3 tools/verify_bundle.py
```

이 명령은 파일 해시만 확인한다. torch·Ultralytics·CUDA·모델 가중치를 로드하지 않고 추론이나 학습을 하지 않는다. 추가 패키지 설치도 필요 없다.

## 다음 대화에 붙여넣을 문장

> 이 저장소의 docs/results/research_handoff_20261018/README_KO.md, EXPERIMENT_RESULTS_KO.md, DECISIONS_AND_LIMITS_KO.md를 먼저 읽어주세요. GPU 실험과 원인 진단은 현재 중단했습니다. 기존 결과를 근거로 연구 서술과 주장 범위를 정리하는 것부터 이어가세요. 새 학습·추론·특징 추출·표본 선택·test 평가는 승인 없이 실행하지 마세요. 전체 mAP 최고는 CM15(31.268124%)이고 inclusion 중심 기준은 CM20(전체31.063447%, inclusion AP50-95 8.933868%, AP75 5.487196%)입니다. CM15는 water_spot 회복 가설을 충족하지 못했습니다. 일반 감사나 파라미터 탐색 루프를 반복하지 마세요.

## Windows에 남겨둔 파일

Windows 원본 루트는 `C:/Users/user/Desktop/vlm/`다. CHECKPOINT_LOCATIONS.csv의 상대 경로를 붙이면 각 best.pt를 찾을 수 있다. CM20/CM15 모델을 실제로 Mac으로 옮기려면 별도 파일 전송이 필요하다. 이번 Git 업로드에는 모델을 포함하지 않았으며, 새로 다운로드하거나 재생성하지 않는다.

recipes 코드는 실행 당시 Windows CUDA·로컬 Python·데이터 경로를 전제로 한 참고 사본이다. dataset/assets가 생략돼 있으므로 Mac에서 controller.py를 바로 실행하지 않는다. 재실행에는 별도 환경 설계·파일 확보·사용자 승인이 필요하다.

이번 커밋은 새 인계 폴더만 포함한다. Windows 저장소에서 이미 수정 중이던 논문 파일과 기타 미추적 파일은 그대로 보존돼 있으며, 이번 신규 커밋에 포함하지 않았다.

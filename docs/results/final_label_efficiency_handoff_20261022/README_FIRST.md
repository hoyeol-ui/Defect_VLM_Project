# 맥북 인계 — 2026년 10월 22일

기준: **2026-10-22T00:08:57+09:00**. 이 패키지는 해당 시점의 스냅샷이다. Windows의 정규 GPU 캠페인은 별도로 계속 실행 중이다. 맥북에서 학습·획득·개발 평가·최종 테스트를 시작하지 않는다.

신규 학습 6/101회 완료, 기존 GC10 Full 1개 재사용, 현재 NEU-DET_Random_pair3_R0 64/100 epoch. 잔여 학습+획득 추정 **44.2 GPU시간**, 모델 완료 추정 2026-10-23T20:21:55+09:00. 최종 테스트·중단 시간은 제외한 변동 가능한 추정이다.

## 바로 열 파일

1. [맥북 Codex 요청](MACBOOK_CODEX_PROMPT.md): 내용을 새 맥북 Codex 채팅에 붙여넣는다.
2. [최신 학술지 Word](journal_manuscript_jkics/manuscript/JKICS-FrozenDINOv2-LabelEfficiency-20261021.docx), [본문 대조본](journal_manuscript_jkics/manuscript/JKICS-FrozenDINOv2-LabelEfficiency-20261021.md).
3. [최종 프로토콜](docs/20261021_final_experiment_protocol.md), [연구 감사](docs/20261021_research_audit.md), [Entropy 복구](docs/20261021_entropy_recovery_report.md).
4. [실행 현황 스냅샷](research_extensions/final_label_efficiency_20261021_01/campaign/STATUS_KO.md), [실행 원장](research_extensions/final_label_efficiency_20261021_01/campaign/execution_ledger.csv), [획득 진단 요약](ACQUISITION_SUMMARY.json).
5. [학위논문 관리 경계](thesis_manuscript/README.md), [투고 지침 및 게재논문 검토](journal_manuscript_jkics/submission_check/20261021_JKICS_review.md).

## 내용과 버전 해석

- `docs/`: 10월21일 감사·실험 설계·원고·교수님 보고 그림. 과거 시각이 남은 문서는 이력이다. 최신 현황은 SNAPSHOT과 원장을 우선한다.
- `journal_manuscript_jkics/`: 투고용 8쪽 익명 Word·Markdown·영문 그림·공식 양식·작성 코드·QA. 결과·저자·결론 일부는 평가 대기 상태다.
- `research_extensions/`: frozen 실행 소스, 설정, train/dev 경로·해시, 선택 manifest, 완료 셀의 로그·개발 평가, 코드 변경 전후·복구 회귀 근거. 소스를 실행하지 않고 검토한다.
- `historical_working_copy/`: 기존 저장소의 미커밋 문서·스크립트 수정과 작은 학위논문 작성 자료를 별도 보존했다. 현 최종 프로토콜과 혼합하지 않는다.
- GitHub 브랜치의 다른 기존 경로는 10월18일까지의 연구 이력을 포함한다. 옛 문서의 중단 지시·더 작은 모델·이전 성능은 현재 프로토콜을 대체하지 않는다.
- 과거 `FAILURE.json`과 원장의 traceback은 복구 이력으로 보존되어 있다. 해당 GC10 셀은 checkpoint를 재사용하여 SUCCEEDED로 복구됐다. 현 실패 여부는 스냅샷·원장의 state로 판단한다.

## Windows에 남는 자료

원본 이미지·라벨, 모델 best/last 가중치, 전체 raw tensor NPZ, Python/CUDA/렌더러 설치 환경은 포함하지 않는다. [가중치 인덱스](CHECKPOINT_INDEX.json), [대용량 제외 목록](EXCLUDED_LARGE_ARTIFACTS.json)과 소스 freeze에 경로·해시가 있다. 이미지별 낮은 confidence 역전 좌표와 원래 인덱스의 JSONL 진단은 포함한다. 원본은 Windows에서 그대로 보존된다. 이 ZIP은 학습 이주·재시작용이 아니다.

논문 PDF 원문을 통째로 재배포하지 않고 지침 검토 문서의 공식 링크를 제공한다. 이 ZIP은 최종 테스트 결과나 모든 학습 완료를 의미하지 않는다. GitHub 및 ZIP은 자동 갱신되지 않는다.

## 무결성 확인

이 디렉터리에서 `python3 verify_manifest.py`를 실행하면 첨부 파일 해시와 개수를 확인한다. 연구 코드나 ML 패키지를 불러오지 않는다. 수정 전 검증하고, 원고 편집 후에는 원래 manifest와 달라지는 것이 정상이다.

Windows는 전원을 유지하고 절전·최대 절전·재부팅·종료를 피한다. 화면 끄기·잠금은 가능하다. 이 스냅샷으로 실제 프로세스가 지금도 살아 있다고 판단하지 않는다.

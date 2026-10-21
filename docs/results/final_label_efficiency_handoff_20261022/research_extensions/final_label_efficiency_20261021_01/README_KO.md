# 2026.10.21 최종 실험 준비물

준비 당시 기술 점검 기준은 `evidence/readiness.json`이다. 후속 사용자 지시에 따라 **2026-10-21 20:57 KST에 신규101회 정규 학습의 순차 실행을 착수했다.** 현재 상태는 `campaign/status.json`과 `campaign/STATUS_KO.md`, 실행 원장은 `campaign/execution_ledger.csv`가 기준이다. GPU 임대나 test 추론은 실행하지 않았다.

- `../../docs/20261021_research_audit.md`: 실제 기록·분할·재사용·한계
- `../../docs/20261021_final_experiment_protocol.md`: 고정 설정·반복·학습 수·통계·실행 순서
- `../../docs/20261021_thesis_paper_plan.md`: 교수님 제출·논문 작업 계획
- `execution_matrix.csv`: 102개 물리 셀, 1개 재사용·101개 새 학습
- `config/`: 전체 학습 kwargs와 최종 프로토콜
- `inputs/`: 기존 train/dev 이미지·정정 라벨의 절대 경로와 SHA. test 이미지·라벨 없음
- `selections/`: 검증하여 계승한 Random/DINO 72개 manifest. Entropy는 새 모델에서 획득
- `cell.py`: 명시적으로 지정한 한 셀을 실행. round0은 공유 셀, Full GC10 pair1은 재학습 거부
- `acquire.py`: 직전 새 checkpoint로 Entropy 획득, 후보 GT를 읽지 않음
- `summarize.py`: 실제 test 결과 114개 곡선·Full 행 검증 및 집계
- `results_input_template.csv`: 평가 뒤 채울 빈 틀. AP를 0으로 가정하지 않음
- `checks.py`: 좌표·class score·Entropy·통계 입력 경계 검증
- `smoke/`: 이번 기술 점검과 실패 원시 기록. 학술 성능에 사용하지 않음
- `evidence/execution_freeze.json`: 코드·설정·입력·기존 선택 목록 및 설치 수치 소스 SHA
- `campaign_controller.py`: NEU Full→공유10% 우선→Entropy 검증→나머지 행렬의 순차 실행. 실패 시 중단
- `acquisition_diagnostics/`: 예측 없음·정확 동점·cutoff ID·ID 순위 분포 및 반복 추론 검증
- `execution_revisions/`: 준비 당시 코드와 실행 기록 보강·진단 통계 수정 전후 봉인
- `../../docs/20261021_paper_draft.md`, `../../docs/figures/20261021_research_flow.svg`: 병행 원고·교수님 보고 그림

`cell.py`와 `acquire.py`는 실행 봉인과 로컬 준비 판정을 검사하고 고유 출력 및 GPU lock을 사용한다. OOM·NaN 처리 실패 시 다른 배치·설정으로 자동 재시도하지 않는다. 실패를 조사한 후 새 기술 시도나 변경 프로토콜을 별도 기록해야 한다. 최종test 평가는 모든 설정 고정 후 별도 평가 목록으로 수행한다.

실행 중에는 컨트롤러나 개별 셀을 중복 시작하지 않는다. `prepare.py`, `audit.py`, `seal_ready.py`, `execution_preflight.py`는 준비 스냅샷 생성용이므로 진행 중인 캠페인에 다시 적용하지 않는다. Full은 각 learner seed당 한 번이며 GC10 pair1은 검증한 기존 학습을 재사용한다. 완료 시 별도test 평가 준비 목록만 생성하고 test를 실행하지 않는다.

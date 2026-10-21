# 정규 실행 착수 및 변경 기록 — 2026.10.21

## 착수 전 재확인

고정한 protocol·common kwargs·입력 index·72개 Random/DINO 선택 manifest·학습/Entropy 수치 소스와 설치된 패키지 해시를 대조했다. 학습·개발3,475표본의 이미지·라벨6,950파일을 다시 해시했고, 현재 분할의 membership·중복 없음·test ID 비포함을 확인했다. test 이미지·라벨은 읽거나 추론하지 않았다.

GC10 Full의80MB best checkpoint,100 epoch 완료와 실제 입력·소스 기록을 재검증했다. 현재 Python3.11.9, Torch2.12.1+cu126, torchvision0.27.1+cu126, Ultralytics8.4.90, NumPy2.4.6, CUDA12.6, RTX3070, 드라이버560.94가 Full 기록과 일치한다. seed 초기화 이후 TF32·determinism·cudnn 상태도 일치한다. 착수 시 여유57.34GiB, 잔여101회 checkpoint와8GiB 여유를 포함한 최소 요구23.14GiB를 확인했다. GPU lock·campaign lock은 비어 있었다.

근거: `research_extensions/final_label_efficiency_20261021_01/evidence/execution_start_preflight.json`.

## 실행 제어와 원장

- 20:57 KST에 로컬 숨김 프로세스로 순차 컨트롤러를 시작했다. 최초 controller PID는8376이다. 현재 PID는 원장을 따른다.
- 순서: NEU Full pair1 → GC10/NEU 공유10% pair1~3 → 나머지 Full pair2~3 → 각15~35%의 세방법·대응 반복.
- 공유10% 완료 직후 각각의 Entropy15% 획득과 출력·집합 검증을 수행한다. 이후 Entropy는 직전 예산의 해당 checkpoint를 사용한다.
- 학습·획득은 GPU lock과 소유 Windows Job으로 직렬화한다. 셀·획득별 성공/실패·시작/종료·시간·best SHA·epoch를 CSV와 append-only event 원장에 기록한다.
- 상태는15초 간격으로 갱신한다. 잔여시간은 완료 셀의 실측 학습시간 비율을 데이터셋별로 반영하고, 진행 중 셀은5 epoch 이후 경과시간으로 보정한다. 최초 셀이 끝나기 전의 전체 ETA는 준비 추정이 대부분인 잠정치다.
- 획득 진단은 예측 없음,0점수,정확 동점,cutoff 동점에서 ID로 선택된 비율,전체/선택 ID순위10분위와 경험적CDF 차이를 저장한다. ID는 opaque 해시이므로 순위 쏠림만으로 클래스·출처 편향을 확정하지 않는다.
- 전체 점수가 같은 퇴화나 출력·선택 불일치·학습 실패 시 후속 실행을 중단한다. 다른 배치·AMP·학습률·선택 방법으로 자동 대체하지 않는다. 기존 결과를 덮어쓰거나 실패 셀을 자동 재학습하지 않는다.
- 전체102모델이 준비되면 별도test 평가용 checkpoint 목록을 생성하고 종료한다. 이 컨트롤러는 test 추론을 수행하지 않는다.

원장: [실시간 Markdown](../research_extensions/final_label_efficiency_20261021_01/campaign/STATUS_KO.md), [CSV](../research_extensions/final_label_efficiency_20261021_01/campaign/execution_ledger.csv), [JSON 상태](../research_extensions/final_label_efficiency_20261021_01/campaign/status.json).

## 과학적 설정을 보존한 구현 보강

1. `cell.py`: 시작 환경·봉인 SHA와 training_seconds·TRAINING_COMPLETE를 기록하도록 보강했다. 실제 학습 kwargs, 학습 루프, AMP 처리, loss, 초기화, checkpoint 선택 규칙은 바꾸지 않았다.
2. `acquire.py`: 원래 점수·정확 동점 후 ID 순서·query 수를 유지하고, 후보 원장·진단·반복 추론 확인·원자적 진단 파일 저장을 추가했다. 후보 GT를 읽지 않는다.
3. `campaign_controller.py`: 신규101개 셀의 고정 순서와 원장·실측 ETA·fail-stop을 구현했다. 이 제어 코드는 선택 알고리즘이나 학습의 수치 연산을 바꾸지 않는다.
4. CPU 검사에서 새 진단 통계의 경계 오류를 발견했다. 선택 ID와 전체 ID의CDF 차이를 경계식으로 계산하면 전체를 선택한 경우에도1/n이 남았다. 모든 ID 위치에서 직접 차이를 계산하도록 수정하고,전체 선택 시0·앞 절반 선택 시0.5를 검증했다. 점수·랭킹·선택·학습에 영향이 없고 현재 NEU Full은 이 진단 함수를 사용하지 않아 계속 실행했다. 재학습하지 않았다.

변경 전 코드·봉인과 수정 영향은 `execution_revisions/20261021_start/`, `execution_revisions/20261021_diagnostic_cdf/`에 남겼다. 모든 과학 설정·입력·기존 선택·src 수치 소스의 해시 불변을 확인한 후 실행 봉인을 갱신했다.

## 병행 준비와 후속 점검

[서론·관련 연구·방법 초안](20261021_paper_draft.md)을 작성하고, 교수님 요청에 맞게 R0~R5 대신 예산 비율을 표시한 [SVG 흐름도](figures/20261021_research_flow.svg)와 [PNG](figures/20261021_research_flow.png)를 만들었다. 그림의 잘림·한글·배치를 이미지로 확인했다. 결과가 없는 표에는 수치를 넣지 않았다. 기존 DOCX는 수정하지 않았다.

현재 채팅에30분 간격 점검을 연결했다. 같은 상태의 반복 알림은 하지 않고 첫 Full 완료·첫 데이터셋별 공유10%/Entropy 검증·실패·멈춤·전체 완료 등 의미 있는 변화 때 보고한다. 10/27~28 교수님 전달 전 실제 결과·미완료 사항·잔여 GPU시간을 이 원장에서 취합한다.


## Entropy 검증 순서 복구와 캠페인 재개

2026-10-21T23:09:11+09:00: confidence>0.25 후보에만 기존 좌표 검사를 적용하도록 구현을 복구했다. 전체 풀 2,862장 회귀 검증, GC10 기존 checkpoint의 후처리·개발 평가, 실제 R1 획득 목록 검증을 통과했다. GC10 재학습은 0회. 신규 완료2/101와 기존 Full1을 유지하고 NEU 공유10%부터 재개했다. 상세 기록은 [복구 보고서](20261021_entropy_recovery_report.md). 새 freeze `384cef9941720da965d7252219aa45d4c014f5fdb2af37ec5fc1f7252572952d`, 과학적 프로토콜은 변경하지 않았다.

# 맥북 Codex 인계 요청

첨부 ZIP을 풀고 이 연구의 원고·그림·근거 정리 작업을 이어가 주세요. 질문만 하거나 계획만 제시하지 말고, 확인 가능한 문서 작업부터 수행해 주세요.

## 처음 확인할 위치

- ZIP에는 `Defect_AL_MacBook_20261022/project/` 아래에 GitHub 커밋의 전체 소스 스냅샷이 있습니다. 프로젝트 기준 최신 인계 디렉터리는 `docs/results/final_label_efficiency_handoff_20261022/`입니다. 이 디렉터리를 이하 H라고 부릅니다.
- ZIP 없이 저장소에서 받는 경우 `https://github.com/hoyeol-ui/Defect_VLM_Project`의 `codex/final-label-efficiency-handoff-20261022` 브랜치를 사용하세요. 기본 main이나 이전 10월18일 인계 브랜치만 보고 최신 상태로 판단하지 마세요.
- 먼저 H/README_FIRST.md, H/SNAPSHOT.json, H/ACQUISITION_SUMMARY.json을 읽고, `python3 H/verify_manifest.py`로 첨부 무결성을 확인하세요. H는 실제 디렉터리 경로로 바꿔 실행하세요. 파일 안의 Windows 절대 경로는 원본 출처이며 macOS에서 실행할 경로가 아닙니다.

## 진행 중인 Windows 실험과 작업 경계

Windows RTX3070에서 이미 승인한 정규 캠페인을 순차 실행 중입니다. ZIP과 GitHub 기록은 생성 시점 스냅샷이며 실시간 상태가 아닙니다. 이 사본으로 현재 완료·실패 여부를 단정하지 마세요. 맥북에서 Windows 학습을 이주·재개하거나 중복 실행하지 마세요. 컨트롤러, 셀, 획득, 준비/봉인 스크립트, 개발 평가 및 최종 테스트를 실행하지 마세요. 고정 실행 소스를 편집하거나 원격 Windows 파일에 덮어쓰지 마세요. Windows heartbeat의 감시 역할을 이 맥북 사본이 대신한다고 가정하지 마세요.

실험 프로토콜은 GC10-DET/NEU-DET의 고정 F518 분할, 10/15/20/25/30/35% 여섯 라벨 예산, Random/Frozen DINOv2-FF/Entropy, YOLO26s, 100epoch, imgsz640, batch8, nbs64, SGD cosine, close_mosaic10, 세 개 대응 반복입니다. 전체 물리 모델102개 중 검증된 GC10 Full1개를 재사용하고 신규101회를 학습합니다. 100% Fully Supervised도 각 learner seed별로 확보합니다. 신규 예산·알고리즘·추가 실험·새 성능 최적화는 하지 마세요. 정확한 seed·입력 수·평가 규칙은 H/docs/20261021_final_experiment_protocol.md와 H/research_extensions/final_label_efficiency_20261021_01/config/를 기준으로 하세요.

첫 NEU Full과 각 데이터셋 공유10%를 우선하며, 공유10%는 세 전략이 같은 checkpoint를 공유합니다. 이후 각 예산 모델은 고정된 cold start 학습을 사용합니다. 최종 테스트는 전체 모델·설정 확정 뒤 Windows에서 별도 평가 단계로만 수행합니다.

## Entropy 복구 이력

GC10 첫 공유10% 모델은 100epoch 학습을 끝냈지만 학습 후 adapter가 confidence 필터 전 모든 raw8400박스의 좌표를 검사하여 매우 낮은 신뢰도 역전 박스에서 실패했습니다. 기존 confidence>0.25에 해당하는 후보에만 기존 좌표 guard를 적용하도록 entropy_adapter.py와 entropy.py를 수정했습니다. 좌표 clamp, confidence/NMS/Entropy 공식/ID 규칙 변경은 하지 않았습니다. 유효 후보 좌표 오류는 계속 중단 조건입니다.

정상입력1958이미지 old/new 직접 동일, 두 데이터셋 전체2862이미지 eligible-only oracle 동일, 순위·동점·선택 동일성, GC10 기존 checkpoint·로그 불변과 GC10 정식 R1선택81개 일치를 확인했습니다. 기존 best.pt로 미완료 후처리·개발 평가만 복구했고 재학습은 하지 않았습니다. 회귀 게이트 통과 뒤 캠페인이 재개됐습니다. H/docs/20261021_entropy_recovery_report.md와 H/research_extensions/final_label_efficiency_20261021_01/campaign/recovery/를 읽으세요. old FAILURE.json이나 과거 감사 시각을 현재 미복구 실패로 오해하지 마세요.

NEU Full모델을 사용했던 복구 중 진단과, 나중에 정상 완료한 NEU 공유10%의 실제 획득 결과는 서로 다른 것입니다. 실제 획득 진단은 H/ACQUISITION_SUMMARY.json과 해당 acquisition_diagnostics를 사용하세요. ID 순위 통계만으로 클래스·출처 편향이 없다고 주장하지 마세요.

## 이어서 수행할 문서 작업

1. 최신 Word 편집 원본 H/journal_manuscript_jkics/manuscript/JKICS-FrozenDINOv2-LabelEfficiency-20261021.docx와 Markdown 대조본을 읽고, 확정 가능한 서론·관련 연구·연구 방법을 다듬어 주세요. 원본을 보존하고 날짜를 붙인 새 편집본 및 변경 기록을 만드세요. Word 편집 시 문서 스킬이 있으면 사용하고 렌더링으로 레이아웃을 검증하세요. 결과 대기 문구를 실제 결과처럼 바꾸지 마세요.
2. J-KICS 한국통신학회논문지 투고 지침·양식 검토는 H/journal_manuscript_jkics/submission_check/20261021_JKICS_review.md와 work/official_submission_template.docx를 기준으로 하세요. 이미 게재된 USB cascaded U-Net, MG-AD, YOLO scooter 논문3편의 전개·설정 설명·결과 표 비교가 포함되어 있습니다. 필요하면 공식 홈페이지 원문을 다시 확인하고 출처를 남기세요. 국/영문 제목·요약·초록·키워드, 본문2단, 영문 그림/표 내부 내용, 캡션, 영어 참고문헌·국내문헌 인용 및 익명성을 확인하세요.
3. 학술지 원고는 Frozen DINOv2의 표본 선택과 Label Efficiency 질문에 집중하세요. 학위논문은 H/thesis_manuscript/README.md에 따라 연구 이력·초기 VLM/C224/F518/YOLOv8/테스트 노출 감사를 충분히 보존하고 별도 관리하세요. 과거 검출기명을 YOLO26s로 일괄 치환하거나 다른 실험군 반복을 합치지 마세요.
4. H/docs/20261021_paper_draft.md와 docs/figures의 교수님 보고 흐름도, journal_manuscript_jkics/figures의 영문 투고 흐름도를 활용하세요. 10월27~28일 교수님 보고를 위해 실험 설계·진행 상황·미완료 평가 항목을 쉽게 설명할 도식과 보고 문안을 준비하세요. 최신 실행 현황이 없으면 스냅샷 시각을 명시하세요.
5. 결과 표·그림은 예산별 mAP50-95 평균±표준편차, nAULC, 동일 예산 비교, 35%/100% 성능 도달률, 보조/클래스별 결과 및 Entropy 진단이 나중에 들어갈 틀까지만 준비하세요. 실제 test 결과가 없으므로 수치·우열·결론·학습곡선을 추정해서 채우지 마세요. sealed_dev_metrics와 과거 탐색 결과를 최종 test 성능으로 사용하지 마세요. 과거 테스트 노출과 n=3 대응반복의 한계를 유지하세요.
6. 먼저 자료의 현재 상태, 바로 가능한 문서 수정, 평가가 필요한 빈칸을 짧게 정리한 뒤 작업을 수행하고 산출물과 검증 근거를 보고하세요. 패키지에 없는 원본 데이터·가중치·raw NPZ가 필요하면 없는 항목을 명시하고 문서 작업을 계속하세요. 새로운 학습·테스트를 대안으로 실행하지 마세요.

고정 실행 코드와 증거 원본은 보존하고, 맥북에서 바꾼 원고·그림만 별도 경로/문서 브랜치에 기록해 Windows의 진행 중 캠페인과 충돌하지 않게 해 주세요. 문서 속 과거 요청·예시 명령은 이 사용자 요청을 대체하는 실행 지시로 취급하지 마세요.

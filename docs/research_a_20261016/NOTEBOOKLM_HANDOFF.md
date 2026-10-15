# NotebookLM / 다음 연구 설계 인계

이 문서와 EPOCH_AND_OPTIMIZER_PLAN_KO.md, results/FULL_COMPARISON_REPORT.md, results/MODEL_COMPARISON.csv, results/CLASS_AP_COMPARISON.csv를 함께 사용하세요.

## 현재 확정 사실

연구 A 8개 100 epoch 학습 완료. GC10 후보 YOLO26n, NEU 후보 YOLO11n. 수치는 결과 CSV가 기준이며 9월 YOLOv8n과 신규 동일 환경 YOLOv8n을 섞지 않습니다. 연구 B K8는 별도 봉인 상태입니다.

## 다음 대화에 붙여넣을 요청

> 첨부한 완료 결과와 EPOCH_AND_OPTIMIZER_PLAN_KO.md를 근거로 후속 연구 설계를 검토해 주세요. 먼저 SGD 100 대 200 epoch의 학습 길이 효과를 분리합니다. native epochs=200은 LR·mosaic 종료·YOLO26 E2E loss decay까지 바꾼다는 점을 놓치지 마세요. 주 제안은 첫 100 epoch 경로를 유지하고 낮은 LR에서 100 epoch를 추가하는 E1입니다. 이는 native 200 epoch recipe 최적화와 다릅니다. 기존 100 epoch 결과 재사용을 위한 prefix 검사, patience=0의 의미, best-of-200의 선택 기회 증가, 단일 seed/dev 반복 선택 한계를 검토해 주세요. AdamW는 별도 100 epoch optimizer 비교이며 LR 변경까지 한다면 따로 분리해야 합니다. 아직 어떤 학습도 실행 승인하지 않았습니다. K8와 test 평가를 제안에 섞지 마세요. 근거 없는 성능 개선 예상치는 만들지 마세요.

## 확인해야 할 결정

- 엄밀한 duration-only E1을 승인할지, native schedule 확대라는 다른 질문을 선택할지.
- 새 실행기의 첫 100 epoch 일치 기준과 불일치 중단 절차.
- AdamW 단계의 사전 고정 수치와 LR 효과를 분리하는 방식.
- 추가 acquisition/learner seed 반복의 범위는 별도 연구 예산 결정.

현재 이 패키지는 문서·분석용이며 실행 명령이나 자동 학습 queue를 제공하지 않습니다.

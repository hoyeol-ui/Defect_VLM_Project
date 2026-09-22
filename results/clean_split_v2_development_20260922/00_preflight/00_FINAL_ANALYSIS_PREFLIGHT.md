# 최종 v2 개발셋 분석 사전 점검

점검 대상: `FINAL_RESULTS_ONLY_20260922.tar.gz`, `FINAL_BEST_WEIGHTS_20260922.tar.gz`의 `state_attempts/ANNOTATION_PATCH_v2_0_2`. 원본은 수정하지 않았다. 실행 결과: **PASS**, `preflight.py` 재실행으로 확인. 이번 분석은 **development 평가**이며 봉인된 test를 열거나 평가하지 않는다.

## 실행과 출처

- 압축파일 SHA-256: 결과 `b978fd30179ce0136cfac81b59399169de9b64daa889a1dddce8bddd0f06bd18`, 가중치 `79b7d9c75973b5bee4e2d37609cef9795dbe1dd535289daa3897b3c67d2ab021`. RunPod `.sha256`과 일치했다.
- 최종 시도 `ANNOTATION_PATCH_v2_0_2`의 `ledger.jsonl`: scientific 162/162 COMPLETE. `gates/final_integrity.json`: PASS, scientific 162, technical 2. Controller log: `162/162 COMPLETE; integrity PASS. TEST NOT RELEASED.` 현재 시도에 STOP.json이 없다.
- 루트 `state/STOP.json` 및 앞선 `RUNTIME_PATCH_v2_0_1/STOP.json`은 과거 실패 이력이다. 최종 시도 `ATTEMPT_LINEAGE.json`에 부모 시도와 보호된 파일 SHA가 기록되어 있다. 이 STOP을 현재 시도의 실패로 합산하지 않는다.
- 로컬 재검증 `shutdown_audit.json`: scientific best.pt 162개 해시 일치, 비가중치 2,156개 해시 일치·누락 없음. 163번째 best.pt는 `TECHNICAL_NEU-DET` smoke. `last.pt`는 회수하지 않았지만 현재 development 결과 재계산에는 필요하지 않다.
- 앞선 `AL_ANALYSIS_EXPORT_20260921.tar.gz`와 최종 archive가 공통으로 가진 sealed dev metric **97개를 바이트 SHA로 교차 대조**했고 차이는 0개였다. 분석 대상은 완성본인 최종 archive이다.
- `FINAL_MAIN_RESULTS_v2_20260922`라는 **별도 완성 압축파일**은 로컬에 없다. `audit/FINAL_MAIN_RESULTS_v2_20260922/export_stream.py`는 그 이름의 메타데이터/체크포인트 export를 만드는 스크립트다. 현재 받은 두 압축파일에 필요한 sealed metrics, 설정, 선택, feature, Entropy 기록, best checkpoints가 있으므로 분석의 결손으로 보지 않는다.

## A–G. 모집단과 실행 구조

| 항목 | GC10-DET | NEU-DET |
|---|---:|---:|
| 전체 canonical 이미지 | 2,291 | 1,798 |
| acquisition pool = train | 1,604 | 1,258 |
| development | 343 | 270 |
| withheld test | 344 | 270 |
| R0 누적 이미지 | 160 | 126 |
| R1 | 241 | 189 |
| R2 | 321 | 252 |
| R3 | 401 | 315 |
| R4 | 481 | 377 |
| R5 | 561 | 440 |
| R0–R5 실제 비율 (%) | 9.9751, 15.0249, 20.0125, 25.0000, 29.9875, 34.9751 | 10.0159, 15.0238, 20.0318, 25.0397, 29.9682, 34.9762 |

acquisition seed는 데이터셋마다 `2026091901`–`2026091905`의 5개다. learner seed는 모든 모델에 `2026091900` 하나다. 따라서 반복 단위는 acquisition seed이고 learner seed나 라운드를 독립 반복으로 세지 않는다. 데이터셋마다 shared R0 5개, Random R1–R5 25개, DINO_F518 R1–R5 25개, Entropy R1–R5 25개, Full-data 1개로 81개다. 전체 162개는 shared R0 10, 각 선택 방법 50씩, Full-data 2로 구성된다. 각 Full-data는 train 전체로 학습한 **1회 참고 모델**이다.

## H–I. 결과 지표의 정확한 출처

최종 development 그림과 표에는 각 run의 `state_attempts/ANNOTATION_PATCH_v2_0_2/runs/<run_id>/sealed_dev_metrics.json`의 `overall["metrics/mAP50-95(B)"]`를 쓴다. `overall["metrics/mAP50(B)"]`는 보조 지표, `per_class_AP50_95`는 클래스별 지표다. 값 0–1을 그림과 본문에서 100배 하여 %로 표시한다. 각 파일은 `artifacts.json`의 SHA와 대조했다. `partition`이 전부 `development`이고 전체/클래스 값은 유한한 0–1 범위다.

`best_selection.json`의 `fitness` 및 `epoch_metrics.json`의 fitness는 **학습 중 dev loader로 best.pt를 고르기 위한 기록**이다. 봉인된 `sealed_dev_metrics.json`은 선택된 checkpoint를 별도 고정 evaluator로 재평가한 값이다. 두 값은 실제로 다를 수 있다. 예: `GC10_v2_a2026091901_DINO_F518_R5`의 학습 시 best fitness는 0.26138이고 봉인 평가 mAP50–95는 0.2402811329이다. 논문 그림에는 **봉인 평가값**을 사용한다. 값이 다른 것을 전사 오류로 취급하지 않는다.

162개 run 모두에서 `train/results.csv` 100행과 `epoch_metrics.json` 100개 epoch의 학습 중 mAP50–95가 일치하며, `best_selection.json`은 마지막 최고 fitness epoch와 일치한다. best checkpoint SHA는 `artifacts.json`·ledger·best 선택 기록 사이에서 일치한다.

`config/protocol.json`, `config/cells.json`, `inputs/*/SPLIT_MEMBERSHIP.csv`, `train_images.json`, 각 run의 `cell.json`·`selection_manifest.json`·`resolved_args.json`을 대조했다. 실제 split 수와 train ID가 일치하고, 모든 누적 선택이 train ID 안에 있으며, 각 seed의 R0가 세 방법에 공유되고 R1–R5가 중첩된다. feature cache는 train 전체 1,604/1,258 × 384이고 `features_valid` gate가 PASS다. Entropy 점수 파일 50개가 archive inventory에 있다. 압축파일에는 test 이미지/라벨 파일이 없고 split metadata에는 test 소속 ID만 있다.

선택 구성의 라벨 출처는 `annotation_overlays/ANNOTATION_PATCH_v2_0_2/labels/<dataset>/train/`이다. overlay는 train 2,862장 각각에 파일이 있고, `TRAIN_DEV_NORMALIZATION.json`의 `after_label_sha`와 전부 일치한다. NEU train 2장과 dev 1장에서 동일 박스 중복이 각 1개 제거된 최종 라벨이다. base `data/`의 train 라벨 286개는 tar hardlink이므로 regular file만 골라 추출하면 초기 R0 라벨이 빠진다. 성능 지표는 봉인된 evaluator 결과를 그대로 읽으며 이 라벨 복사본으로 재평가하지 않았다.

## 분석계획의 우선순위

`reference/FINAL_MAIN_EXPERIMENT_PROTOCOL_v2_0_KO.md` 9절이 이 clean split v2의 분석계획이다. 별도의 v2 SAP 파일은 찾지 못했다. 이전 `confirmatory_v1(_1)` 계획과 9월 15일의 `analysis_plan_frozen.json`은 다른 split·seed에 관한 문서이므로 섞지 않는다. v2 계획의 **1차 지표는 추후 test mAP50–95 nAULC**이다. 현 development 계산은 결과 탐색과 논문 그림 준비용이며, test 1차 검정이 완료된 것처럼 표현하지 않는다.

원자료 보존: 모든 신규 파일은 `analysis_20260922/final_thesis_analysis/` 아래에만 작성한다. 점검 코드와 기계 판독 결과는 `07_reproducibility/preflight.py`, `00_preflight/preflight_machine.json`에 있다.

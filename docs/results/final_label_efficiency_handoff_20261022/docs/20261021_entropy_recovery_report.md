# GC10 공유 10% Entropy 점검 복구 결과

보고 기준: 2026-10-21T23:09:11+09:00. 복구 검증을 통과하고 승인된 캠페인을 재개했다. 기존 GC10 100 epoch 학습은 반복하지 않았으며 복구·검증 중 정규 학습과 최종 테스트를 실행하지 않았다. 이후 새 학습은 재개 게이트 통과 뒤 NEU 공유 10%부터 시작했다.

## 오류 원인과 변경 범위

YOLO26s의 one2one raw 출력에는 매우 낮은 confidence의 배경 후보에서 역전된 XYXY 좌표가 존재했다. 기존 adapter와 Entropy 함수가 confidence 필터보다 먼저 전체 raw 좌표를 검사하여 중단했다. 실제 사용 후보를 정의하는 조건은 기존과 같은 float32 confidence > 0.25이다.

두 함수의 기존 nonnegative(>=0) 크기 검사 적용 범위만 confidence 통과 후보로 제한했다. 이때 유효 후보의 좌표 역전은 adapter와 score 양쪽에서 여전히 오류다. 0 크기의 기존 허용 조건도 새로 바꾸지 않았고, 실제 전체 풀에서 유효 후보의 0 크기 박스는 0개였다.

좌표 clamp, threshold 변경, NMS IoU/클래스 처리/정렬/최대 검출 수 변경, 클래스 점수 정규화·Entropy 공식·이미지 집계·예측 없음·정확 동점/ID 선택 규칙 변경은 없다. 원래 anchor 인덱스와 미사용 raw 좌표를 유지했다. 추가 변경은 raw 진단 기록과 완료 셀을 건너뛰는 명시적 캠페인 재개 경로다. 학습 코드 cell.py와 학습 인자·분할·예산·검출기·반복 수는 그대로다.

## 수정 전후 동일성 검증

- 합성 정상 입력 18개: nc=6/10, 빈 후보, 경계 confidence, 동점, 중복 박스 NMS, max300, 기존 0 폭 조건을 포함해 기존/수정 출력이 정확히 일치했다.
- 경계·오류 검증 8개: 역전 박스가 confidence=0.25일 때 제외되고, 0.25 바로 위 float32 값이면 두 경로 모두 오류가 발생함을 확인했다.
- 보존된 실제 실패 raw 2개도 확인했다. 낮은 confidence 역전 박스는 최종 anchor 목록에 포함되지 않았다.
- 전체 풀에서 기존 코드가 직접 처리 가능한 정상 이미지 1,958장(GC10 709, NEU 1,249)은 기존/수정 점수 함수 출력이 정확히 일치했다. 각 데이터셋 정상 영상 3장은 기존/수정 adapter의 실제 반복 forward 출력까지 비교했다.
- 전체 2,862장은 기존 함수를 confidence 통과 후보에만 적용한 독립 참조 경로와 비교했다. 원래 인덱스를 되돌린 NMS anchor, 박스 좌표, 클래스·confidence·전체 클래스 점수, 박스별/이미지별 Entropy, 예측 수가 모두 정확히 일치했다. 허용오차를 주어 다르게 나온 값을 통과시키지 않았다.
- 두 풀의 전체 순위, 정확 동점 진단 및 cutoff 선택 ID도 참조 경로와 일치했다. 실제 GC10 R1 획득을 별도 실행한 뒤 1,444개 전체 순위와 81개 선택 ID가 앞선 검증 결과와 정확히 일치함을 다시 확인했다.

## 실제 모델의 전체 학습 풀 검사

| 항목 | GC10 공유 10% 모델 | NEU Full 모델 진단 |
|---|---:|---:|
| 검사 이미지 | 1604 | 1258 |
| raw 역전이 있는 이미지 | 895 (55.7980%) | 9 (0.7154%) |
| raw 역전 박스 | 6159 | 9 |
| 전체 raw 박스 중 역전 비율 | 0.0457% | 0.0001% |
| confidence 통과 후보의 역전 | 0 | 0 |
| R0 제외 후 미선택 후보 이미지 | 1444 | 1132 |
| 예측 없음 / 정확 동점 후보 비율 | 22.9224% | 2.8269% |
| cutoff ID로 잘라 선택한 비율 | 0% | 0% |

NEU 검사는 현재 완료된 실제 YOLO26s Full checkpoint를 사용한 입력 형식·좌표·회귀 진단이다. NEU 공유 10% 모델의 공식 획득을 대신하지 않으며, 해당 모델 학습 완료 뒤 원래 캠페인의 획득 검증을 따로 수행한다. 두 모델의 전체 학습 풀을 검사했으며 개발/테스트 이미지를 획득 점수에 입력하지 않았다. 획득 점수 계산에 후보 정답 라벨도 읽지 않았다.

전체 raw XYXY/클래스 점수와 원래 8,400개 anchor 인덱스를 campaign/recovery/.../<dataset>/raw/*.npz에 보존했다. per_image.jsonl에는 점수·박스·역전 진단이 남아 있다. 이후 공식 획득도 역전 raw를 압축 보존하고 이미지별 geometry 진단을 기록한다.

## 실제 GC10 획득 진단과 ID 순위

- 미선택 1,444장 중 예측 없음 331장(22.9224%). 정확 동점도 이 0점 집합 331장이고, 다른 점수는 서로 달랐다.
- 획득 수 81장. cutoff score 0.055320168200548893, cutoff 집합 1장이므로 ID에 의한 잘라 선택은 0%다.
- 선택 ID의 10분위 개수: [10, 9, 9, 10, 9, 8, 9, 8, 3, 6]. 평균 ID percentile 0.444833, 기술적 KS D 0.106682.
- 이번 경계에서 동점 ID 규칙이 표본을 결정하지 않았다는 뜻이다. 이 순위 요약만으로 클래스·원천 자료에 대한 의미적 편향이 없다고 단정하지 않는다. 동점 또는 예측 없음 규칙은 바꾸지 않았다.

## checkpoint 재사용과 후처리

기존 best epoch 94 / best SHA `54507df99a510f6e56dea40639bb12fae0e8bbbcb92e135614be88309da20bbd` 및 last SHA `76fccb4d13ae091562b15aa56e0b66539426f88db5526b7d5e4103243e4ccb3b`가 모두 일치한다. 두 가중치를 별도 사본으로 백업했고 기존 학습 로그·TRAINING_COMPLETE·progress·best_selection·FAILURE 기록은 원본 그대로 남겨 두었다. 완료 학습 재실행 횟수는 0이다.

미완료 adapter 반복 추론과 원래 FP32 개발 평가(batch8, rect=False, conf0.001, IoU0.7, max_det300, quantize32)만 재수행했다. 후처리 소요는 13.974초다. 개발 평가는 343장에 대한 것이며 최종 테스트 결과가 아니다. RESULT.json은 PASS로 새로 생성했고 실행 원장의 해당 셀을 SUCCEEDED로 갱신했다. 원래 오류 traceback과 파일은 삭제하지 않았다.

## 소스와 실행 버전

과학적 프로토콜 SHA: `85f373041df3777ce35b1d91f8b704b5a714f35bccfc956aa91f2bc19c1f479b` (변경 없음). common_args.json, 원래 입력/선택 manifest 및 설치 numerical source의 해시도 유지했다.

이전 실행 freeze: `3f0da9a5528d4c34022f55d9e9eef725e875b38dfcf76bcee46e1ee9be5cbd7c`. 새 실행 freeze: `384cef9941720da965d7252219aa45d4c014f5fdb2af37ec5fc1f7252572952d`. 학습 시점의 기존 ENVIRONMENT_AT_START는 유지하고, 후처리 실행 버전은 별도 기록했다.

| 변경 파일 | 수정 전 SHA256 | 수정 후 SHA256 |
|---|---|---|
| src/entropy.py | `32b6acef32a4a206a04610b7df25cbb4db21795b2246a960a18356b22c7579a6` | `f3017ef08e072451f7df1c7ed52bf15e16c201433651d4388dbb54569852c432` |
| src/entropy_adapter.py | `e8d0d77d99d13315d63053fd191509bf22fa4c0aadd92f88c21b6f483fe2361a` | `d496b5f33cd19a3174ed7c4b3bb5004a034288cbeb337df549b01bab7ad7a243` |
| acquire.py | `102c2e9c828a1b539f422a4ef678830ba3deb935dba43086703e5b234e2f40d6` | `3832df5b21a0541f536d1e61da5ec62e3ba16e30f56a401921ec735061f2f8aa` |
| campaign_controller.py | `28ed374bbeb6ad1238fb7cbc503f11e6bbcf02cb4418ce9ac06e6a2d9678d29b` | `56fac8c48a90e76372b5c2276aaf32db2aaaf4e127083ede0ca85e88c1fe09a4` |

execution_revisions/20261021_entropy_eligible_guard_01/before, after, before_manifest.json, source_changes.json과 새 동결 manifest에 원본·수정본·unified diff를 보존했다. 새 검증/복구 스크립트도 동결 목록에 포함했다. 초기 준비 실행은 파일 읽기 인코딩/구 freeze 검사 단계에서 중단됐고 GPU 검증이나 학습 전이었다. 그 실패 기록은 별도 setup_attempt01 폴더에 보존한 뒤 UTF-8로 봉인하고 위 전체 검증을 수행했다.

## 캠페인 재개와 잔여 시간

- 신규 완료 2/101, 기존 Full 재사용 1, 현재 1개 실행, 대기 98개, 현재 실패 0개.
- 2026-10-21T23:09:11+09:00 기준 현재 셀 `NEU-DET_Random_pair1_R0`, epoch 22/100. controller PID 20228; 실제 lock 소유 프로세스와 부모·자식 관계를 확인했다.
- GC10 R1 획득 검증 1/30 완료. 기존 순서를 유지하며 NEU 공유10% → 나머지 공유10% → 예정된 Full 및 이후 예산 셀을 실행한다. 기존 완료/재사용 모델은 재학습하지 않는다.
- 잔여 학습 43.03 + 획득 0.56 = **43.59 GPU시간**. 현 추정 모델 완료 2026-10-23T18:44:33+09:00. 중단 대기와 최종 테스트 시간은 제외하며 원장 실측으로 계속 갱신한다.
- 앞선 실패 보고서의 “실패 셀 재활용이 ETA에 반영되지 않았을 수 있다”는 설명은 정정한다. 실제 ETA 함수는 FAILED를 제외하므로 당시 약41.76시간도 미착수99개 기준이었다. 이번 증가의 주 원인은 GC10의 실제 837.4초를 이용한 보정계수1.0871 반영이다. 모델을 추가하거나 완료 학습을 반복해서 증가한 것이 아니다.
- 이후 모델에서 유효 후보 역전·수치 오류·획득 퇴화가 발견되면 기존 fail-stop 정책대로 멈춘다. 이번 통과는 현재 확인한 두 실제 모델과 풀의 결과이며 앞으로의 모든 모델을 미리 보증하는 것은 아니다.

## 검증 원자료 위치

- campaign/recovery/20261021_entropy_eligible_guard_01/VALIDATION_PASS.json 및 cpu_regression.json
- 같은 폴더의 GC10-DET/summary.json, NEU-DET/summary.json, per_image.jsonl, raw/ 및 순위·선택 진단
- 같은 폴더의 POSTPROCESS_PASS.json, RECOVERY_PASS.json
- acquisition_diagnostics/GC10-DET/2026091901/R1_diagnostics.json, R1_geometry.jsonl, R1_ranking.json
- selections/GC10-DET/2026091901/Entropy/R1.json
- campaign/execution_ledger.csv, events.jsonl, status.json, STATUS_KO.md

최종 테스트는 실행하지 않았다. 학술지 원고의 성능 결과표에도 개발 점수나 예상 수치를 입력하지 않았다.

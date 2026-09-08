# 2026-09-08 이후 논문 실행 순서

## 오늘 잠긴 결정

- 연구 정의와 제목 1·2순위를 `thesis_core_definition_20260907.md`로 고정했다.
- DINOv2 표현 사후 감사는 기존 cache로 완료했으며 재추출하지 않는다.
- 100% reference는 Plan A, 4 runs를 추천안으로 고정했다.
- Distinct AL baseline은 Minmax least-confidence 한 가지, 새 training 200 cells로 고정했다.
- 내일 가장 먼저 할 일은 **제목 1·2순위와 연구 목적 한 문장을 지도교수에게 보내 최종 선택을 받는 것**이다.

# Step 1

**작업:** 제목 및 연구 목적 확정

**목적:** 9월 9일 제출 전에 환경·Task·목적·방법이 제목과 설명에서 같은 연구를 가리키게 한다.

**필요 시간/계산:** 30~60분, GPU 0. `thesis_core_definition_20260907.md`의 1순위·2순위와 Q1 답변만 사용한다.

**산출물:** 지도교수가 선택한 국문·영문 제목, 승인된 연구 목적 한 문장, 학과의 접수 후 제목 변경 절차 확인 메모.

**완료 조건:** 제목 하나와 목적 문장 하나가 서면으로 확정되고, 이후 문서가 이 표현을 기준으로 사용한다.

**다음 단계 진입 조건:** 지도교수 답변을 받았거나, 9월 9일 제출 시각 때문에 1순위안을 제출해야 하는 행정 시점이 도래함.

# Step 2

**작업:** DINOv2 feature 정량 감사 결과 고정 및 논문용 표 선택

**목적:** 일반 이미지 사전학습 표현이 두 산업 결함 데이터에서 어떤 라벨 구조를 보존하고 어디서 약한지 직접 답한다.

**필요 시간/계산:** 분석은 2026-09-07 완료. 내일은 30~60분의 CPU 검토만 수행하고 feature 재추출·학습은 0회다.

**산출물:** `dinov2_representation_validity_audit_20260907.md`, raw CSV 4종, integrity record와 출력 SHA-256, 논문에 넣을 dataset-level 표 1개와 GC10 취약 class 표 1개.

**완료 조건:** 무결성 PASS, 출력 hash 재검증 PASS, “평균 구조는 있으나 모든 class를 보장하지 않는다”는 해석 문장이 고정됨.

**다음 단계 진입 조건:** audit metric을 selector tuning에 사용하지 않으며 기존 440-run을 수정하지 않는다는 경계 확인.

# Step 3

**작업:** 100% full-data reference Plan A 실행

**목적:** 약 35% AL 결과가 동일 lifecycle의 100% train 성능과 얼마나 차이 나는지 기술적 기준을 만든다.

**필요 시간/계산:** 4 YOLO runs, 최대 400 epochs, 572,800 image-epochs. 기존 R5 wall time의 단순 비례 추정은 약 2.3 GPU-hours이며 validation·test·audit overhead를 별도로 둔다.

**산출물:** 별도 sealed full-data manifest, 4개 `best.pt`와 training logs, integrity audit, seed별/평균 mAP50-95·mAP50·precision·recall, R5 대비 gap 표.

**완료 조건:** 4/4 training PASS 후 잠긴 test 4회 평가 완료, 기존 split·checkpoint·hyperparameter와 동일함이 hash로 확인됨.

**다음 단계 진입 조건:** 지도교수의 4-run 범위 확인 및 별도 manifest 봉인. 기존 잠긴 test 절차를 바꿔야 하면 진입하지 않고 STOP-6 보고.

# Step 4

**작업:** Distinct AL baseline 규칙 승인 및 봉인

**목적:** 결과를 보기 전에 Minmax least-confidence의 점수, empty rule, policy checkpoint, inference 설정, tie-break를 재현 가능하게 고정한다.

**필요 시간/계산:** 1~2시간 문서·selection-only 설계, GPU 학습 0. 실제 detector pool inference도 승인 전에는 실행하지 않는다.

**산출물:** `distinct_al_baseline_execution_plan_20260907.md`를 바탕으로 한 별도 protocol/manifest와 dry-run checklist.

**완료 조건:** `U(x)=1-max box confidence`, no detection=1, 직전 round learner A `best.pt`, 고정 inference 인자, 200-cell 범위를 지도교수가 확인함.

**다음 단계 진입 조건:** 별도 runner의 selection-only dry-run에서 sample ID completeness, GT 비개입, validation/test leakage 0, deterministic tie-break PASS.

# Step 5

**작업:** Distinct baseline 실행

**목적:** Random과 표현 기반 다양성 외에 detector-dependent uncertainty와 비교하라는 교수 요구를 충족한다.

**필요 시간/계산:** 공유 R0 40 cells 재사용, R1~R5 새 training 200 cells, 최대 20,000 cell-epochs·7,160,000 image-epochs. 기존 round wall time 기준 약 32 GPU-hours에 acquisition inference 114,550장과 잠긴 test image-evaluations 81,700건이 추가된다.

**산출물:** 10 acquisition seeds × 2 learner seeds의 R0~R5 trajectory, training integrity audit, 별도 authorization 후 primary/secondary 결과표.

**완료 조건:** 200/200 training PASS, boundary violation·config change·GT acquisition 사용 0, 그 후에만 잠긴 test 평가 완료.

**다음 단계 진입 조건:** 모든 결과와 failure를 숨김없이 봉인하고 outcome을 이유로 규칙을 바꾸지 않음.

# Step 6

**작업:** 최종 결과 통합

**목적:** Random, DINOv2 표현 기반 farthest-first, Minmax least-confidence, 100% reference를 서로 다른 질문에 답하는 결과로 정렬한다.

**필요 시간/계산:** 2~4시간 CPU 분석. 새 학습 0.

**산출물:** R0~R5 nAULC 비교표, R5와 100% 절대 gap 표, class-level 위험 표, representation audit 표, 제한사항 목록.

**완료 조건:** aggregate와 class-level 결과가 분리되고, 100% 단일점을 nAULC처럼 취급하지 않으며 annotation 비용 절감·SOTA 주장이 없음.

**다음 단계 진입 조건:** 모든 숫자가 raw CSV와 한 번씩 대조되고 figure/table caption에 dataset·이미지 수·train pool 비율·metric 시점이 적힘.

# Step 7

**작업:** 논문 본문 수정

**목적:** 연구 변천사가 아니라 산업 문제 → 제한된 라벨 → 객체 탐지 Task → 연구 목적 → 방법 → 통제 질문 순으로 논문을 재구성한다.

**필요 시간/계산:** 1~2일 문서 작업, GPU 0.

**산출물:** 확정 제목, 국·영문 초록, 서론, 방법, 실험, 결과, 제한사항, 관련연구가 일관된 심사용 원고.

**완료 조건:** 핵심 정의 정본과 모순이 없고, 새 알고리즘·모든 class 개선·실제 비용 절감·산업 전체 일반화·SOTA를 주장하지 않음.

**다음 단계 진입 조건:** 지도교수 검토용 변경 요약과 근거표가 준비됨.

# Step 8

**작업:** 발표 및 디펜스 대응자료 작성

**목적:** 교수님이 다시 물을 가능성이 높은 목적, 제목, DINOv2 역할, 자동 selection, 데이터 비율, 100% reference, distinct baseline을 짧고 일관되게 답한다.

**필요 시간/계산:** 반나절~1일 문서·슬라이드 작업, GPU 0.

**산출물:** `professor_next_meeting_quick_answers_20260907.md` 기반 1장 즉답표, 방법 flowchart, 데이터 비율 표, 핵심 결과·한계 슬라이드.

**완료 조건:** 각 질문에 20~40초 답변과 숫자 근거가 있고, 표현·acquisition·learner·comparator의 역할을 섞지 않음.

**다음 단계 진입 조건:** 최종 rehearsal에서 제목 한 문장, 목적 한 문장, 결과 한 문장, 한계 한 문장을 자료 없이 설명 가능함.

## 이번 논문에서 실행하지 않는 항목

Domain Generalization 전환, open-set 전환, 새 image+text detector, 과거 VLM 실험 전수 복원, 새 Qwen inference/fusion selector, DINOv2 fine-tuning, 현 방법과 같은 feature·거리의 별도 Core-set arm은 P2/future work로만 남긴다.

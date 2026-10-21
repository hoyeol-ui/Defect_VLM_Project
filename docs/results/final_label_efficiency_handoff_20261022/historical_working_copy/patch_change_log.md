# Patch Change Log

## 0. 적용 범위

이 문서는 identity alignment audit에서 수정 필요 또는 조건부 통과로 확정된 표현 가운데 이번 7개 산출물 범위에 해당하는 항목의 실제 반영 내역을 기록한다. 원본 v2와 기존 skeleton은 변경하지 않았으며, 새 claim·수치·anchor·선행연구를 추가하지 않았다.

## 1. 실제 수정 내역

| change_id | 대상 파일·위치 | 원본 표현 | 수정 후 표현 | 근거 감사 항목 id | 위험 등급 |
|---|---|---|---|---|---|
| PCL-01 | chapter4 v3 MD, 4.1 목적 | 이 절은 타당성 검증…을 판단하는 절이 아니다 | 이 절은 선택 단계 측정 결과만 제시하며…판단하지 않는다 | FIX-01 | 높음 |
| PCL-02 | chapter4 v3 MD, 4.1 문단 1 | 후보 신호의 타당성 | 후보 신호의 선택 단계 측정 결과 | FIX-02 | 높음 |
| PCL-03 | chapter4 v3 CSV, 4.1 purpose | 타당성 검증 | 선택 단계 측정 결과만 제시 | FIX-03 | 높음 |
| PCL-04 | chapter4 v3 MD, 4.5 목적·표·문단·허용 표현 | supporting·safety check / Safety checks / 보조·안전성 관찰값 | 보조 지표·정상 이미지 FPR 점검 / 정상 이미지 FPR 점검 / 보조 지표와 정상 이미지 FPR 관찰값 | FIX-04 | 높음 |
| PCL-05 | chapter4 v3 CSV, 4.5 관련 필드 | supporting·safety check / safety checks | 보조 지표·정상 이미지 FPR 점검 | FIX-05 | 높음 |
| PCL-06 | chapter5 v3 MD, 5.1 해석 문장 1 | 평균 최소효과 기준 | 결과 확인 전에 고정한 평균 최소효과 기준 | FIX-08 | 중간 |
| PCL-07 | chapter5 v3 CSV, 5.1 allowed interpretation | 평균 최소효과 기준 | 결과 확인 전에 고정한 평균 최소효과 기준 | FIX-09 | 중간 |
| PCL-08 | chapter5 v3 MD, 5.3 제목·목적·요약 | 학습 안정성과 선택 재현성 / acquisition 반복 | 고정 선택 집합의 training-seed 안정성(fixed-set training stability) / 동일 fixed pool에서의 acquisition reproducibility | FIX-06 | 높음 |
| PCL-09 | chapter5 v3 CSV, 5.3 title·purpose·summary | 학습 안정성과 선택 재현성 | 고정 선택 집합의 training-seed 안정성과 acquisition reproducibility | FIX-07 | 높음 |
| PCL-10 | chapter5 v3 MD, 5.6 교차참조 | 보조·안전성 수치 | 보조 지표와 정상 이미지 FPR 수치 | FIX-10 | 중간 |
| PCL-11 | chapter5 v3 CSV, 5.6 교차참조 | 보조·안전성 수치 | 보조 지표와 정상 이미지 FPR 수치 | FIX-11 | 중간 |
| PCL-12 | chapter1 skeleton v2, RQ4 직후 | EP01의 RQ3·RQ4 통합 역할이 암묵적 | 감사에서 확정한 비결과형 통합 답변 예고문 1개 삽입 | DOC-02 | 중간 |
| PCL-13 | chapter2 skeleton v2, 2.1 앞 | Anchor 흐름이 절별 표에 분산됨 | ALR·ODA·VFM·IDD 16개 anchor의 최초 등장·재사용 고정표 삽입 | ANCH-01–ANCH-16 | 낮음 |

## 2. 변경 없이 확인한 항목

- 4.5의 허용 관찰·한 문장 요약에는 KSDD2를 success 또는 no-effect로 판정하는 문장이 없었다. 금지 표현 목록은 경계 장치이므로 유지했다.
- 5.1은 이미 세 분기의 이질성과 서로 다른 미충족 지점만 설명하고 EP01 최종 문장을 본문에 반복하지 않았다. FIX-08의 시간적 지위 표현만 반영했다.
- 5.5에 비용·annotation-time·예산 한계를 집중하고 5.7은 5.5를 교차참조하는 v2 구조가 이미 적용되어 있어 추가 수정하지 않았다.
- Chapter 2의 2.5 자기 규정 3문장은 원문 그대로 유지했다.

## 3. 파일별 파생 관계

| 생성 파일 | 기준 원본 | 변경 범위 |
|---|---|---|
| chapter4_writing_packets_v3.md | chapter4_writing_packets_v2.md | FIX-01·02·04 |
| chapter4_writing_packets_v3.csv | chapter4_writing_packets_v2.csv | FIX-03·05 |
| chapter5_writing_packets_v3.md | chapter5_writing_packets_v2.md | FIX-06·08·10 |
| chapter5_writing_packets_v3.csv | chapter5_writing_packets_v2.csv | FIX-07·09·11 |
| chapter1_introduction_skeleton_v2.md | chapter1_introduction_skeleton.md | DOC-02 |
| chapter2_related_work_skeleton_v2.md | chapter2_related_work_skeleton.md | ANCH-01–16 |

## v3에서 실제로 반영된 표현 봉합 항목 목록

- 4.1의 타당성·검증 중심 표현을 선택 단계 측정 결과 표현으로 축소했다.
- KSDD2의 safety check를 보조 지표와 정상 이미지 FPR 점검으로 구체화했다.
- 5.3을 고정 선택 집합의 training-seed 안정성과 동일 fixed pool에서의 acquisition reproducibility로 통일했다.
- 5.1의 평균 최소효과 기준에 결과 확인 전 고정이라는 시간적 지위를 복원했다.
- 1.3 RQ4 직후에 EP01의 RQ3·RQ4 통합 역할을 예고하는 비결과형 문장을 삽입했다.
- 제2장에 16개 기존 anchor의 제1장 최초 등장과 제2장 재사용 위치를 명시했다.

## 반영되지 않은 감사 항목이 있다면 그 이유

- FIX-12, FIX-17, FIX-21, FIX-22는 감사에서 권고 수정으로 분류되어 이번의 수정 필요·조건부 통과 적용 범위에 포함하지 않았다.
- FIX-13–FIX-20과 FIX-23은 outline map, final adoption 문서, contribution positioning CSV에 관한 항목이며 사용자가 지정한 7개 생성 파일에 해당 문서의 v3 산출물이 포함되지 않아 원본을 수정하지 않았다. 초안에서는 identity alignment audit의 우선순위 규칙에 따라 구버전 표현을 사용하지 않는다.
- Cross-chapter audit의 과거 표현은 변경 이력으로 보존하는 자료이므로 수정하지 않았다.

## 초안 작성 진입 여부 판정 (진입 가능 / 재봉합 필요)

**진입 가능.** 이번 v3 패킷과 skeleton v2를 초안 source로 사용할 경우 4.1 타당성, KSDD2 safety 용어, 5.3 재현성 단위, 1.3 EP01 예고의 필수 봉합이 반영되어 있다. 기존 v2·구버전 map의 표현을 다시 가져오지 않는 것을 전제로 한다.

## 초안 진입 시 첫 절과 그 이유

**4.2 결함 발견량과 선택 집합 구성부터 시작한다.** EP02의 결과 절로서 관찰값과 learner outcome의 NOT_AVAILABLE 경계가 이미 고정되어 있고, 4.1의 평가 전제나 4.3의 post-hoc 상태보다 해석 선취 위험이 낮다.


# 논문 그림 목록과 한국어 caption 초안

> 아래는 최초 디자인의 그림 목록이다. 논문 삽입용 개정안과 새 캡션은 `redesign_v2/REDESIGN_COMPARISON.md`를 우선 확인한다. 최초 그림은 수치·디자인 비교를 위해 보존한다.

모든 그림은 같은 이름의 **PNG 300 dpi / vector PDF / SVG** 세 형식으로 저장했다. 내부 축·범례는 영문이다. 전체 그림의 값은 **development 평가**다. 본문 그림 1–4는 `03_figures_main`, 두 paired 차이 그림은 `04_figures_appendix`에 있다. 제작 코드: `07_reproducibility/make_figures.py`.

| 번호 | 파일 stem | 본문/부록 | 목적 |
|---|---|---|---|
| 그림 1 | `FIGURE_1_GC10_AL_CURVE` | 본문 | GC10 예산–mAP 곡선 |
| 그림 2 | `FIGURE_2_NEU_AL_CURVE` | 본문 | NEU 예산–mAP 곡선 |
| 그림 3 | `FIGURE_3_NAULC_COMPARISON` | 본문 | 두 데이터셋의 nAULC 절대값 |
| 그림 4a | `FIGURE_4_GC10_R5_CLASS_AP` | 본문 | GC10 R5 10개 클래스 AP |
| 그림 4b | `FIGURE_4_NEU_R5_CLASS_AP` | 본문 | NEU R5 6개 클래스 AP |
| 부록 그림 A1 | `APPENDIX_PAIRED_DELTA_GC10` | 부록 | GC10 round별 및 nAULC paired 차이 |
| 부록 그림 A2 | `APPENDIX_PAIRED_DELTA_NEU` | 부록 | NEU paired 차이 |

**그림 1.** GC10-DET의 라벨 이미지 예산에 따른 development mAP@0.50:0.95. 각 선은 5 acquisition seed의 평균, 음영은 seed 간 표본 표준편차(`ddof=1`)다. x좌표는 acquisition pool 1,604장에 대한 실제 누적 비율이며, R0–R5 이미지 수는 160, 241, 321, 401, 481, 561장이다. 세 방법은 seed별 R0 모델을 공유한다. 검은 점선은 train 전체 1,604장으로 학습한 Full-data **1회** 참고값이다. 음영은 유의성이나 신뢰구간을 뜻하지 않는다.

**그림 2.** NEU-DET의 라벨 이미지 예산에 따른 development mAP@0.50:0.95. 각 선은 5 acquisition seed의 평균, 음영은 seed 간 표본 표준편차다. acquisition pool 1,258장의 실제 누적 수는 126, 189, 252, 315, 377, 440장이다. 세 방법은 seed별 R0를 공유한다. 점선은 train 전체 1,258장의 Full-data 1회 참고값이다. 그림 1과 동일한 0–50% y축을 사용했다.

**그림 3.** GC10-DET와 NEU-DET의 development nAULC 비교. 각 작은 빈 마커는 acquisition seed 1개의 nAULC, 큰 마커와 세로 막대는 5 seed 평균±표본 표준편차다. 연한 회색 선은 같은 seed의 세 방법을 연결한다. nAULC는 실제 누적 이미지 수를 x축으로 한 R0–R5 AP 곡선의 사다리꼴 면적을 `n5−n0`으로 나눈 값이며 %로 표시했다. 시작을 0으로 둔 동일한 0–40% y축을 사용했다. Full-data는 곡선이 없어 이 그림에 포함되지 않는다.

**그림 4a–b.** 이미지 예산 약 35%(R5)에서의 클래스별 development AP@0.50:0.95. 마커는 각 방법의 5 seed 평균, 가로 막대는 seed 간 표본 표준편차다. 모든 GC10 10개 및 NEU 6개 클래스를 표시하고 x축은 0–100%로 고정했다. 클래스별 차이는 전체 평균효과가 각 클래스에 동일하게 적용되지 않음을 보여주는 기술적 결과다.

**부록 그림 A1–A2.** seed를 맞춘 development 방법 간 차이. 왼쪽은 budget별 평균 AP 차이와 5 seed의 표본 SD 음영, 오른쪽은 seed별 nAULC 차이의 개별 마커와 평균(마름모)±표본 SD(세로 막대)다. 0선이 차이 없음의 기준이다. 표시된 구간은 p값이나 신뢰구간이 아니다. 사전 정의 exact sign-flip/Holm 결과는 `06_STATISTICAL_TESTS.md`에 별도 제시한다.

**본문 배치 제안.** 그림 1·2를 나란한 크기로 결과 절 앞부분에 배치하고, 그림 3으로 곡선 요약을 보여준 다음 그림 4a–b로 클래스 불균일성을 설명한다. 부록 A1–A2는 라운드별 부호 변화와 paired 원자료를 보여줄 때 인용한다. 그림 4는 본문 공간이 부족하면 클래스 전체 숫자표와 함께 부록으로 이동할 수 있지만 연구질문 RQ2의 근거이므로 결과 절에서 핵심 차이는 반드시 언급한다.
